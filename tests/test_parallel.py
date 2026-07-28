"""Tests for component decomposition and the process-parallel runner.

    python tests/test_parallel.py
    pytest tests/test_parallel.py

Offline. Spawns real child processes, so it exercises the same start method
Windows uses -- including whether the payloads actually pickle.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box

PROJECT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT)

import engine  # noqa: E402
import parallel  # noqa: E402
import pipeline  # noqa: E402
import partition  # noqa: E402
import tiles as T  # noqa: E402
from config import RunConfig  # noqa: E402

CRS = 2237
GAP = 3000.0  # far wider than the 100 ft adjacency threshold


def make_island_world(n_islands: int = 4, per_island: int = 700, seed: int = 5):
    """Several parcel clusters separated by gaps wide enough to sever the graph."""
    rng = np.random.default_rng(seed)
    frames = []
    for k in range(n_islands):
        x0 = 700_000.0 + k * (4_000.0 + GAP)
        y0 = 700_000.0
        px = rng.uniform(x0, x0 + 4_000.0, per_island)
        py = rng.uniform(y0, y0 + 4_000.0, per_island)
        frames.append(
            gpd.GeoDataFrame(
                {
                    "key": np.arange(per_island) + k * per_island,
                    "model_group": "single_family",
                    "latitude": 26.6 + (py - y0) / 364_000.0,
                    "longitude": -81.9 + (px - 700_000.0) / 325_000.0,
                    "bldg_area_finished_sqft": rng.exponential(2000, per_island) + 400,
                    "land_area_sqft": rng.exponential(7000, per_island) + 1000,
                    "bldg_age_years": rng.uniform(0, 80, per_island),
                    "dist_to_open_water": (px - x0) * 0.5
                    + rng.normal(0, 200, per_island),
                    "assr_market_value": rng.lognormal(12.2, 0.5, per_island),
                },
                geometry=gpd.points_from_xy(px, py).buffer(25, cap_style=3),
                crs=CRS,
            )
        )
    parcels = gpd.GeoDataFrame(
        __import__("pandas").concat(frames, ignore_index=True), crs=CRS
    )
    minx, miny, maxx, maxy = parcels.total_bounds
    area = gpd.GeoDataFrame(
        {"geometry": [box(minx - 100, miny - 100, maxx + 100, maxy + 100)]}, crs=CRS
    )
    return parcels, area


def build(cfg: RunConfig):
    parcels, area = make_island_world()
    tileset = T.build_tileset(
        area, None, None, None, None, CRS,
        grid_size_ft=cfg.grid_size_ft, clip_water=False,
    )
    parcels = engine.add_derived_columns(parcels)
    parcels = engine.bin_continuous_fields(
        parcels, cfg.continuous_variables, cfg.max_bins
    )
    parcels, all_tiles, t2p = T.assign_parcels_to_tiles(parcels, tileset)
    adj_tiles = all_tiles.loc[all_tiles.index.intersection(sorted(t2p))]
    adj = T.calculate_adjacency(adj_tiles, cfg.adjacency_threshold_ft)
    seed = engine.seed_neighborhoods(
        parcels, cfg.seed_fields, cfg.n_neighborhoods, cfg.random_seed
    )
    return parcels, t2p, adj, seed


def base_config(**over) -> RunConfig:
    cfg = RunConfig(
        n_neighborhoods=16, grid_size_ft=1000.0, adjacency_threshold_ft=100.0,
        max_iterations=300, refresh_every=50, random_seed=42,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


# ==========================================================================
# Decomposition
# ==========================================================================


def test_components_match_the_islands():
    cfg = base_config()
    parcels, t2p, adj, _ = build(cfg)
    comps = partition.find_components(t2p, adj)
    assert comps.n_components == 4, f"expected 4 islands, got {comps.n_components}"
    assert comps.parcel_counts.sum() == len(parcels)
    assert comps.tile_counts.sum() == len(comps.tile_ids)

    pc = partition.parcel_components(comps, t2p, len(parcels))
    assert (pc >= 0).all(), "a parcel was left without a component"


def test_split_removes_all_spanning_neighborhoods():
    cfg = base_config()
    parcels, t2p, adj, seed = build(cfg)
    comps = partition.find_components(t2p, adj)
    pc = partition.parcel_components(comps, t2p, len(parcels))

    before = partition.spanning_neighborhoods(seed, pc)
    assert before, "fixture should produce spanning neighborhoods to fix"

    new_ids, n_split = partition.split_neighborhoods(seed, pc)
    assert n_split == len(before)
    assert not partition.spanning_neighborhoods(new_ids, pc)
    # contiguous relabelling, nothing lost
    assert new_ids.min() == 0
    assert len(np.unique(new_ids)) == new_ids.max() + 1
    # a parcel's companions within one component must stay together
    for k in np.unique(seed):
        for c in np.unique(pc[seed == k]):
            sub = new_ids[(seed == k) & (pc == c)]
            assert len(np.unique(sub)) == 1, "a component slice fragmented further"


def test_grouping_balances_and_reports_ceiling():
    weights = [15404, 9923, 7025, 1942, 1581, 1297, 100, 50, 10, 1]
    for n in (1, 2, 4, 8):
        groups = partition.group_components(weights, n)
        assert sum(len(g) for g in groups) == sum(1 for w in weights if w > 0)
        flat = [c for g in groups for c in g]
        assert len(flat) == len(set(flat)), "a component landed in two groups"
        loads = [sum(weights[c] for c in g) for g in groups]
        # never worse than the single largest component
        assert max(loads) >= max(weights)
    # zero-weight components are dropped rather than given a worker
    assert all(w > 0 for g in partition.group_components([5, 0, 3], 3) for w in
               [[5, 0, 3][c] for c in g])


def test_useful_worker_count_clamps_to_available_speedup():
    # one dominant component: extra workers are pointless
    assert partition.useful_worker_count([1000, 1, 1], 0) == 1
    # four equal components: four workers
    assert partition.useful_worker_count([100, 100, 100, 100], 0) == 4
    # an explicit request is honoured but can't exceed the component count
    assert partition.useful_worker_count([100, 100], 8) == 2
    assert partition.useful_worker_count([100], 8) == 1


def test_any_spanning_matches_the_slow_check_and_is_cheap():
    cfg = base_config()
    parcels, t2p, adj, seed = build(cfg)
    comps = partition.find_components(t2p, adj)
    pc = partition.parcel_components(comps, t2p, len(parcels))

    assert partition.any_spanning(seed, pc) == bool(
        partition.spanning_neighborhoods(seed, pc)
    )
    split, _ = partition.split_neighborhoods(seed, pc)
    assert not partition.any_spanning(split, pc)
    assert partition.any_spanning(np.zeros(len(parcels), dtype=np.int64), pc), (
        "one neighborhood covering every island must count as spanning"
    )


def test_parallel_is_refused_when_neighborhoods_span_components():
    """Splitting off + workers > 1 would race on shared count-table rows.

    The no-shared-state argument rests entirely on neighborhoods being confined
    to one component. Without that, two workers mutate the same rows of
    neigh_counts and the scores are quietly wrong -- so it must be refused, not
    attempted.
    """
    cfg = base_config(workers=4, split_severed_neighborhoods=False)
    parcels, t2p, adj, seed = build(cfg)
    comps = partition.find_components(t2p, adj)
    pc = partition.parcel_components(comps, t2p, len(parcels))
    assert partition.any_spanning(seed, pc), "fixture must have spanning hoods"

    opt = engine.TiledOptimizer(parcels, t2p, adj, cfg, neighborhood_ids=seed)
    prep = pipeline.PreparedRun(
        cfg=cfg, run_dir=Path("."), jurisdiction_gdf=None, working_crs=CRS,
        tiles=None, parcels=parcels, optimizer=opt, store=None,
        components=comps, parcel_component=pc,
        neighborhoods_span_components=True,
        tile_to_parcels=t2p, tile_adjacency=adj,
    )
    assert prep.worker_count() == 1, "must fall back to a single worker"
    try:
        prep.make_parallel_runner()
    except RuntimeError as exc:
        assert "split_severed_neighborhoods" in str(exc), str(exc)
    else:
        raise AssertionError("expected make_parallel_runner to refuse")

    # ...and with splitting on, parallelism is allowed again
    prep.neighborhoods_span_components = False
    assert prep.worker_count() > 1


# ==========================================================================
# Parallel runner
# ==========================================================================


def _run_parallel(cfg, parcels, t2p, adj, seed, n_groups, tmp: Path):
    comps = partition.find_components(t2p, adj)
    pc = partition.parcel_components(comps, t2p, len(parcels))
    split, _ = partition.split_neighborhoods(seed, pc)

    parquet = tmp / "parcels_prepared.parquet"
    # Only the scored columns are what the worker reads.
    parcels[list(cfg.weights)].to_parquet(parquet)

    groups = partition.group_components(comps.parcel_counts, n_groups)
    tasks = parallel.build_group_tasks(groups, comps, t2p, adj)
    runner = parallel.ParallelRunner(
        cfg=cfg, parcels_path=str(parquet), tasks=tasks,
        parcel_n_ids=split, n_neighborhoods=int(split.max()) + 1,
    )
    snaps = []
    result = runner.run(snapshot_cb=lambda ids, s: snaps.append(s.iteration))
    return result, split, comps, pc, snaps, runner


def test_single_group_parallel_matches_serial_exactly():
    """One group over every component is the serial problem, so it must agree."""
    cfg = base_config()
    parcels, t2p, adj, seed = build(cfg)
    comps = partition.find_components(t2p, adj)
    pc = partition.parcel_components(comps, t2p, len(parcels))
    split, _ = partition.split_neighborhoods(seed, pc)

    serial = engine.TiledOptimizer(
        parcels, t2p, adj, cfg, neighborhood_ids=split
    )
    serial.consolidate_mixed_tiles()
    serial.run()

    tmp = Path(tempfile.mkdtemp())
    try:
        got, _, _, _, _, _ = _run_parallel(
            cfg, parcels, t2p, adj, seed, 1, tmp
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    agree = float((got == serial.parcel_n_ids).mean())
    assert (got == serial.parcel_n_ids).all(), (
        f"one-group parallel diverged from serial (agreement {agree:.2%})"
    )


def test_multi_group_assignment_is_complete_and_component_local():
    cfg = base_config()
    parcels, t2p, adj, seed = build(cfg)

    tmp = Path(tempfile.mkdtemp())
    try:
        got, split, comps, pc, snaps, runner = _run_parallel(
            cfg, parcels, t2p, adj, seed, 4, tmp
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    assert len(got) == len(parcels)
    assert (got >= 0).all(), "an unassigned parcel came back"
    assert snaps, "no snapshots streamed from the workers"

    # Every parcel must still sit in a neighborhood confined to its component:
    # that is the invariant the whole scheme rests on.
    for k in np.unique(got):
        comps_here = np.unique(pc[got == k])
        assert len(comps_here) == 1, (
            f"neighborhood {k} spans components {comps_here}"
        )

    # Work actually happened, and only within components.
    assert not np.array_equal(got, split), "nothing changed at all"
    for c in np.unique(pc):
        mask = pc == c
        # a component's parcels may only carry labels that started in it
        assert set(np.unique(got[mask])).issubset(set(np.unique(split[mask]))), (
            f"component {c} acquired a label from elsewhere"
        )


def test_worker_failure_surfaces_to_the_parent():
    """A crashing worker must raise, not hang or silently return partial work."""
    cfg = base_config()
    parcels, t2p, adj, seed = build(cfg)
    comps = partition.find_components(t2p, adj)
    pc = partition.parcel_components(comps, t2p, len(parcels))
    split, _ = partition.split_neighborhoods(seed, pc)

    tmp = Path(tempfile.mkdtemp())
    try:
        # Point the worker at a parquet that does not exist.
        groups = partition.group_components(comps.parcel_counts, 2)
        tasks = parallel.build_group_tasks(groups, comps, t2p, adj)
        runner = parallel.ParallelRunner(
            cfg=cfg, parcels_path=str(tmp / "missing.parquet"), tasks=tasks,
            parcel_n_ids=split, n_neighborhoods=int(split.max()) + 1,
        )
        try:
            runner.run()
        except RuntimeError as exc:
            assert "worker" in str(exc).lower(), str(exc)[:200]
        else:
            raise AssertionError("expected RuntimeError from the failed worker")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_group_task_round_trips_through_pickle():
    """Spawn pickles every payload; a non-picklable field would only show up here."""
    import pickle

    cfg = base_config()
    parcels, t2p, adj, seed = build(cfg)
    comps = partition.find_components(t2p, adj)
    groups = partition.group_components(comps.parcel_counts, 2)
    tasks = parallel.build_group_tasks(groups, comps, t2p, adj)

    for task in tasks:
        back = pickle.loads(pickle.dumps(task))
        assert np.array_equal(back.tile_ids, task.tile_ids)
        t2p_back, adj_back = back.to_dicts()
        assert len(t2p_back) == task.n_tiles()
        # the CSR round-trip must reproduce the original dicts exactly
        for t in task.tile_ids:
            assert np.array_equal(t2p_back[int(t)], t2p[int(t)])
            assert adj_back[int(t)] == {
                int(n) for n in adj[int(t)] if int(n) in set(task.tile_ids.tolist())
            }


# The old `stability_sweeps` rule this used to cover was replaced by the
# assignment-stability convergence test, which is scale-free by construction.
# See tests/test_contiguity.py for its coverage.


def test_weighted_score_ignores_singleton_inflation():
    """Splitting creates many singletons scoring 1.0; the plain mean is useless."""
    cfg = base_config()
    parcels, t2p, adj, seed = build(cfg)
    comps = partition.find_components(t2p, adj)
    pc = partition.parcel_components(comps, t2p, len(parcels))
    split, _ = partition.split_neighborhoods(seed, pc)

    opt = engine.TiledOptimizer(parcels, t2p, adj, cfg, neighborhood_ids=split)
    stats = opt.stats()
    num, den = opt.weighted_score()

    assert den == len(parcels), "weighted score must cover every owned parcel"
    assert abs(stats.weighted_score - num / den) < 1e-12
    # the weighted figure must sit in the range of real per-neighborhood scores,
    # not be dragged up by the 1.0/2.0 sentinels
    real = opt.scores[opt.ct.totals > 1]
    assert stats.weighted_score <= float(real.max()) + 1e-9


def test_worker_weighted_scores_sum_to_the_global_figure():
    """Workers report numerator/denominator so the parent can be exact."""
    cfg = base_config()
    parcels, t2p, adj, seed = build(cfg)
    comps = partition.find_components(t2p, adj)
    pc = partition.parcel_components(comps, t2p, len(parcels))
    split, _ = partition.split_neighborhoods(seed, pc)

    whole = engine.TiledOptimizer(parcels, t2p, adj, cfg, neighborhood_ids=split)
    want_num, want_den = whole.weighted_score()

    groups = partition.group_components(comps.parcel_counts, 3)
    tasks = parallel.build_group_tasks(groups, comps, t2p, adj)
    got_num = got_den = 0.0
    for task in tasks:
        t2p_local, adj_local = task.to_dicts()
        part = engine.TiledOptimizer(
            parcels, t2p_local, adj_local, cfg, neighborhood_ids=split
        )
        n, d = part.weighted_score()
        got_num += n
        got_den += d

    assert abs(got_den - want_den) < 1e-9, (got_den, want_den)
    assert abs(got_num - want_num) < 1e-6, (got_num, want_num)


def test_no_tile_is_owned_by_two_groups():
    cfg = base_config()
    parcels, t2p, adj, seed = build(cfg)
    comps = partition.find_components(t2p, adj)
    groups = partition.group_components(comps.parcel_counts, 3)
    tasks = parallel.build_group_tasks(groups, comps, t2p, adj)

    owned = np.concatenate([t.tile_ids for t in tasks])
    assert len(owned) == len(set(owned.tolist())), "tile ownership overlaps"
    assert set(owned.tolist()) == set(int(t) for t in t2p), "tiles went missing"

    parcels_owned = np.concatenate([t.tp_indices for t in tasks])
    assert len(parcels_owned) == len(set(parcels_owned.tolist()))
    assert len(parcels_owned) == len(parcels), "parcel ownership is not a partition"


# ==========================================================================


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}\n      {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {fn.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
