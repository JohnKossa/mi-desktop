"""Tests for the contiguity gate and the assignment-stability stopping rule.

    python tests/test_contiguity.py
    pytest tests/test_contiguity.py

Offline. The fixture is a plain grid so connectivity is easy to reason about.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine  # noqa: E402
import tiles as T  # noqa: E402
from checkpoints import Checkpoint, CheckpointStore  # noqa: E402
from config import RunConfig  # noqa: E402

CRS = 2237
X0, Y0, W, H = 700_000.0, 700_000.0, 10_000.0, 10_000.0


def make_parcels(n: int = 5000, seed: int = 7) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(seed)
    px = rng.uniform(X0 + 20, X0 + W - 20, n)
    py = rng.uniform(Y0 + 20, Y0 + H - 20, n)
    return gpd.GeoDataFrame(
        {
            "key": np.arange(n),
            "model_group": "single_family",
            "latitude": 26.6 + (py - Y0) / 364_000.0,
            "longitude": -81.9 + (px - X0) / 325_000.0,
            "bldg_area_finished_sqft": rng.exponential(2000, n) + 400,
            "land_area_sqft": rng.exponential(7000, n) + 1000,
            "bldg_age_years": rng.uniform(0, 80, n),
            "dist_to_open_water": (px - X0) * 0.5 + rng.normal(0, 200, n),
            "assr_market_value": rng.lognormal(12.2, 0.5, n),
        },
        geometry=gpd.points_from_xy(px, py).buffer(25, cap_style=3),
        crs=CRS,
    )


def base_config(**over) -> RunConfig:
    cfg = RunConfig(
        n_neighborhoods=20, grid_size_ft=1000.0, adjacency_threshold_ft=100.0,
        max_iterations=600, refresh_every=100, random_seed=42,
        assignment_stability_iters=0,  # off unless a test asks for it
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def build(cfg: RunConfig):
    parcels = make_parcels()
    area = gpd.GeoDataFrame({"geometry": [box(X0, Y0, X0 + W, Y0 + H)]}, crs=CRS)
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


def component_counts(opt: engine.TiledOptimizer) -> dict:
    return {
        n: opt._components_of(tiles)
        for n, tiles in opt.n_to_tiles.items() if tiles
    }


def excess_components(opt: engine.TiledOptimizer) -> int:
    """Fragmentation, independent of how many neighborhoods survive.

    Raw component totals are the wrong measure: absorption removes whole
    neighborhoods, so the total can fall while the survivors fragment badly.
    """
    return sum(max(0, c - 1) for c in component_counts(opt).values())


def disconnected_count(opt: engine.TiledOptimizer) -> int:
    return sum(1 for c in component_counts(opt).values() if c > 1)


def fine_config(**over) -> RunConfig:
    """Enough tiles per neighborhood that fragmentation is actually reachable."""
    return base_config(grid_size_ft=250.0, n_neighborhoods=60, **over)


# ==========================================================================
# The gate
# ==========================================================================


_RUN_CACHE: dict = {}


def _run_and_measure(enforce: bool):
    """Gated and ungated runs, computed once each and shared between tests."""
    if enforce in _RUN_CACHE:
        return _RUN_CACHE[enforce]
    cfg = fine_config(enforce_contiguity=enforce)
    parcels, t2p, adj, seed = build(cfg)
    opt = engine.TiledOptimizer(parcels, t2p, adj, cfg, neighborhood_ids=seed)
    opt.consolidate_mixed_tiles()
    before = (excess_components(opt), disconnected_count(opt))
    opt.run()
    after = (excess_components(opt), disconnected_count(opt))
    _RUN_CACHE[enforce] = (opt, before, after)
    return _RUN_CACHE[enforce]


def test_gate_never_worsens_connectivity():
    """No surviving neighborhood may gain components, and fragmentation can't rise."""
    opt, before, after = _run_and_measure(enforce=True)
    assert after[0] <= before[0], (
        f"excess components rose {before[0]} -> {after[0]}"
    )
    assert after[1] <= before[1], (
        f"disconnected neighborhoods rose {before[1]} -> {after[1]}"
    )
    assert opt.accepted > 0, "gate blocked everything; nothing was learned"
    assert opt.blocked_batches == 0 or opt.accepted > opt.blocked_batches


def test_without_the_gate_connectivity_does_degrade():
    """Control: prove the gated result isn't vacuous.

    Without the gate this fixture fragments badly (~6 -> ~76 excess components),
    so the gated run holding at or below its starting value is meaningful.
    """
    _, before, after = _run_and_measure(enforce=False)
    assert after[0] > before[0] * 2, (
        "ungated run did not fragment, so the gated test proves nothing "
        f"(excess components {before[0]} -> {after[0]})"
    )


def test_gate_changes_the_outcome():
    """Belt and braces: the two runs must actually differ."""
    gated, _, gated_after = _run_and_measure(enforce=True)
    ungated, _, ungated_after = _run_and_measure(enforce=False)
    assert gated_after[0] < ungated_after[0]
    assert not np.array_equal(gated.parcel_n_ids, ungated.parcel_n_ids)


def test_ranked_gating_equals_prefiltering():
    """Gating after ranking must pick the same move as gating every candidate.

    This is the whole reason the gate is cheap: filtering first then taking the
    best is the same as taking the best that passes, but only the latter can stop
    checking once it finds one.
    """
    cfg = base_config(enforce_contiguity=True)
    parcels, t2p, adj, seed = build(cfg)
    opt = engine.TiledOptimizer(parcels, t2p, adj, cfg, neighborhood_ids=seed)
    opt.consolidate_mixed_tiles()

    rng = np.random.default_rng(1)
    pool = sorted(opt.boundary)
    checked = 0
    for _ in range(40):
        picks = rng.choice(len(pool), size=min(64, len(pool)), replace=False)
        edges = [pool[int(k)] for k in picks]

        ranked = opt._best_move(edges)

        # Reference implementation: gate every candidate up front, then take the
        # best survivor.
        cands = []
        for ti, tj in edges:
            if (ti, tj) not in opt.boundary:
                continue
            if ti == tj:
                n_primary = int(opt.tile_n_ids[ti])
                recips = {int(v) for v in np.unique(
                    opt.parcel_n_ids[opt.tile_parcels[ti]])}
                for j in opt._neighbors(ti):
                    recips.add(int(opt.tile_n_ids[int(j)]))
                for n_recip in recips:
                    r = opt._eval_donation(ti, n_primary, n_recip)
                    if r:
                        cands.append(r)
                continue
            n_i, n_j = int(opt.tile_n_ids[ti]), int(opt.tile_n_ids[tj])
            for donor, nd, nr in ((ti, n_i, n_j), (tj, n_j, n_i)):
                r = opt._eval_donation(donor, nd, nr)
                if r:
                    cands.append(r)
            r = opt._eval_swap(ti, tj)
            if r:
                cands.append(r)

        legal = [c for c in cands if not opt._move_breaks_contiguity(c[1])]
        legal.sort(key=lambda c: -c[0])
        expected = legal[0] if legal else None

        if expected is None:
            assert ranked is None
        else:
            assert ranked is not None, "ranked gating returned nothing"
            assert abs(ranked[0] - expected[0]) < 1e-12, (ranked[0], expected[0])
            assert ranked[1] == expected[1]
        checked += 1
    assert checked == 40


def test_gate_detects_a_hand_built_articulation_point():
    """A 3-in-a-row neighborhood must not be allowed to lose its middle tile."""
    cfg = base_config(enforce_contiguity=True)
    parcels, t2p, adj, seed = build(cfg)
    opt = engine.TiledOptimizer(parcels, t2p, adj, cfg, neighborhood_ids=seed)

    # Find a tile whose neighbourhood-mates form a line through it.
    found = False
    for pos in range(opt.n_tiles):
        neigh = [int(n) for n in opt._neighbors(pos)]
        if len(neigh) < 2:
            continue
        a, b = neigh[0], neigh[1]
        # a and b must not touch each other, so `pos` is their only link
        if b in {int(x) for x in opt._neighbors(a)}:
            continue
        opt.n_to_tiles = {999: {pos, a, b}}
        assert opt._removal_disconnects(999, pos), "middle tile not detected"
        # and removing a leaf is fine
        assert not opt._removal_disconnects(999, a)
        # an isolated tile can't disconnect anything
        opt.n_to_tiles = {999: {pos}}
        assert not opt._removal_disconnects(999, pos)
        found = True
        break
    assert found, "fixture produced no suitable articulation point"


def test_addition_island_check_is_swap_specific():
    cfg = base_config(enforce_contiguity=True)
    parcels, t2p, adj, seed = build(cfg)
    opt = engine.TiledOptimizer(parcels, t2p, adj, cfg, neighborhood_ids=seed)

    pos = 0
    neigh = [int(n) for n in opt._neighbors(pos)]
    assert neigh, "fixture tile has no neighbours"
    other = neigh[0]

    # `other` is the only member, and it is leaving -> `pos` would be marooned
    opt.n_to_tiles = {7: {other}}
    assert opt._addition_creates_island(7, other, pos)
    # a second, staying member anchors it
    extra = [int(n) for n in opt._neighbors(pos) if int(n) != other]
    if extra:
        opt.n_to_tiles = {7: {other, extra[0]}}
        assert not opt._addition_creates_island(7, other, pos)
    # empty recipient is always an island
    opt.n_to_tiles = {7: set()}
    assert opt._addition_creates_island(7, other, pos)


# ==========================================================================
# The stopping rule
# ==========================================================================


def _prime(opt, iteration, window=500):
    """Give the optimizer a full window of event history at `iteration`."""
    opt.iteration = iteration
    opt._event_marks = [(iteration - window, 0)]


def test_convergence_needs_history_before_it_can_fire():
    cfg = base_config(assignment_stability_iters=500)
    parcels, t2p, adj, seed = build(cfg)
    opt = engine.TiledOptimizer(parcels, t2p, adj, cfg, neighborhood_ids=seed)

    # Fresh state: no event history at all, so the ratio is unknown, not zero.
    assert opt.progress_ratio(500) is None
    opt.iteration = 100
    assert not opt._assignment_converged()
    assert opt._stable_streak == 0


def test_convergence_fires_on_churn_not_on_low_volume():
    """The signal is re-touching the same parcels, not moving few of them."""
    cfg = base_config(
        assignment_stability_iters=100, assignment_stability_streak=3,
        assignment_progress_ratio=0.20,
    )
    parcels, t2p, adj, seed = build(cfg)
    opt = engine.TiledOptimizer(parcels, t2p, adj, cfg, neighborhood_ids=seed)
    owned = opt._owned_parcels

    # Churn: 20 parcels relabelled over and over -> 2,000 events, 20 distinct.
    _prime(opt, 10_000, window=100)
    opt.last_change_iter[:] = 0
    opt.last_change_iter[owned[:20]] = 10_000
    opt.touch_events = 2_000
    ratio = opt.progress_ratio(100)
    assert ratio is not None and ratio < 0.05, ratio
    fired = [opt._assignment_converged() for _ in range(3)]
    assert fired == [False, False, True], fired

    # Low volume but all novel: few parcels moved, every one of them fresh.
    # This must NOT be read as convergence -- it is what a big dataset looks
    # like, and mistaking it for convergence is what stopped Lee County early.
    opt2 = engine.TiledOptimizer(parcels, t2p, adj, cfg, neighborhood_ids=seed)
    owned2 = opt2._owned_parcels
    _prime(opt2, 10_000, window=100)
    opt2.last_change_iter[:] = 0
    opt2.last_change_iter[owned2[:30]] = 10_000
    opt2.touch_events = 30          # 30 events, 30 distinct -> ratio 1.0
    assert opt2.progress_ratio(100) == 1.0
    assert not opt2._assignment_converged()


def test_progress_ratio_does_not_depend_on_dataset_size():
    """The regression that shipped: a metric whose ceiling shrank with the data.

    Normalising by total parcels made the achievable value inversely
    proportional to dataset size -- fine on a 5k-parcel fixture, permanently
    below threshold on a 276k-parcel county. The replacement divides two
    quantities that scale together, so identical *behaviour* must score the same
    at any size.
    """
    cfg = base_config(assignment_stability_iters=100)
    parcels, t2p, adj, seed = build(cfg)

    ratios, fractions = [], []
    for keep in (1.0, 0.25):
        subset = dict(list(sorted(t2p.items()))[: max(1, int(len(t2p) * keep))])
        opt = engine.TiledOptimizer(parcels, subset, adj, cfg, neighborhood_ids=seed)
        owned = opt._owned_parcels
        _prime(opt, 10_000, window=100)
        opt.last_change_iter[:] = 0
        # Same *behaviour* in both: 200 relabel events, all reaching fresh parcels.
        n = min(200, len(owned))
        opt.last_change_iter[owned[:n]] = 10_000
        opt.touch_events = n
        ratios.append(opt.progress_ratio(100))
        fractions.append(opt.recent_change_fraction())

    assert abs(ratios[0] - ratios[1]) < 1e-9, (
        f"novelty ratio moved with dataset size: {ratios}"
    )
    # And demonstrate why the old statistic could not be used this way.
    assert fractions[0] != fractions[1], (
        "fraction-of-parcels was expected to be size dependent; fixture too small "
        "to show it"
    )


def test_fraction_of_parcels_has_a_size_dependent_ceiling():
    """Arithmetic guard on the bug, independent of any fixture.

    A move relabels a fixed handful of parcels no matter how big the dataset is,
    so the achievable fraction falls as parcels rise. Any threshold expressed in
    those units is therefore only valid at one scale.
    """
    window, parcels_per_move = 500, 10.0
    ceilings = {
        n: window * parcels_per_move / n for n in (5_000, 50_000, 276_617)
    }
    assert ceilings[5_000] > 0.5, ceilings
    assert ceilings[276_617] < 0.02, ceilings
    # the old default threshold sat *inside* the county-scale ceiling
    assert ceilings[276_617] < 0.02 and 0.01 < ceilings[276_617], (
        "1% threshold vs a ~1.8% ceiling -- no usable dynamic range"
    )


def test_missing_last_change_cannot_cause_false_convergence():
    """The failure mode that matters: a checkpoint without the array."""
    cp = Checkpoint(
        iteration=50_000, temperature=1e-6, stability_counter=0,
        accepted=1, rejected=1, mean_score=0.0, n_neighborhoods=10,
        parcel_n_ids=np.zeros(1000, dtype=np.int64),
        last_change_iter=None,
    )
    restored = cp.restore_last_change(1000)
    assert (restored == 50_000).all(), (
        "must read as 'everything just changed', not as zeros"
    )
    # zeros here would mean age == 50,000 for every parcel -> 0% recent -> exit
    age = cp.iteration - restored
    assert (age == 0).all()


def test_checkpoint_round_trips_the_change_history():
    cfg = base_config(assignment_stability_iters=100)
    parcels, t2p, adj, seed = build(cfg)
    opt = engine.TiledOptimizer(parcels, t2p, adj, cfg, neighborhood_ids=seed)
    opt.consolidate_mixed_tiles()

    tmp = Path(tempfile.mkdtemp())
    try:
        store = CheckpointStore(tmp, keep=0)
        cfg2 = base_config(max_iterations=300, assignment_stability_iters=100)
        opt.cfg = cfg2
        opt.run(store=store)
        saved = store.latest()
        assert saved.last_change_iter is not None, "array was not persisted"
        assert len(saved.last_change_iter) == len(parcels)
        assert saved.last_change_iter.max() > 0, "no changes recorded at all"

        fresh = engine.TiledOptimizer(
            parcels, t2p, adj, cfg2, neighborhood_ids=seed
        )
        fresh.load_checkpoint(saved)
        assert np.array_equal(fresh.last_change_iter, opt.last_change_iter)
        assert fresh._stable_streak == opt._stable_streak
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
