"""Offline smoke tests for the desktop pipeline.

    python tests/test_desktop.py      # plain run, prints a report
    pytest tests/test_desktop.py      # same checks under pytest

No network and no parcel file required -- everything is synthesised. The
network sources are monkeypatched, so what is exercised is the local half of
the pipeline: tiling, the parcel/tile join, adjacency, seeding, the
consolidation pass, annealing, checkpointing and resume.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine  # noqa: E402
import mi  # noqa: E402
import pipeline  # noqa: E402
import sources  # noqa: E402
import tiger  # noqa: E402
import tiles as T  # noqa: E402
from checkpoints import CheckpointStore  # noqa: E402
from config import RunConfig  # noqa: E402
from geo import crs_is_feet, describe_crs, pick_feet_crs  # noqa: E402
from render import build_polygons, neighborhood_colors  # noqa: E402

CRS = 2237  # NAD83 / Florida West (ftUS)
X0, Y0, W, H = 700_000.0, 700_000.0, 8_000.0, 8_000.0


# ==========================================================================
# Fixtures (plain functions so the file runs with or without pytest)
# ==========================================================================


def make_area() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": [box(X0, Y0, X0 + W, Y0 + H)]}, crs=CRS)


def make_parcels(n: int = 4000, seed: int = 7) -> gpd.GeoDataFrame:
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
            # spatially structured, so MI has real signal to find
            "dist_to_open_water": (px - X0) * 0.5 + rng.normal(0, 200, n),
            "assr_market_value": rng.lognormal(12.2, 0.5, n),
        },
        geometry=gpd.points_from_xy(px, py).buffer(25, cap_style=3),
        crs=CRS,
    )


def make_roads(seed: int = 11) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(seed)
    lines = [LineString([(X0, Y0 + y), (X0 + W, Y0 + y)])
             for y in np.linspace(500, H - 500, 7)]
    lines += [LineString([(X0 + x, Y0), (X0 + x, Y0 + H)])
              for x in np.linspace(500, W - 500, 7)]
    lines += [
        LineString([(x, y), (x + rng.uniform(400, 2500), y + rng.normal(0, 80))])
        for x, y in zip(rng.uniform(X0, X0 + W, 60), rng.uniform(Y0, Y0 + H, 60))
    ]
    return gpd.GeoDataFrame({"geometry": lines}, crs=CRS)


def base_config(**over) -> RunConfig:
    cfg = RunConfig(
        n_neighborhoods=15,
        grid_size_ft=1000.0,
        adjacency_threshold_ft=100.0,
        max_iterations=400,
        checkpoint_every=100,
        refresh_every=50,
        random_seed=42,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def build_state(cfg: RunConfig):
    area, parcels, roads = make_area(), make_parcels(), make_roads()
    tileset = T.build_tileset(
        area, None, roads, None, None, CRS,
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
    return parcels, adj_tiles, t2p, adj, seed


# ==========================================================================
# Tests
# ==========================================================================


def test_feet_crs_selection():
    """Every study area must get a projected CRS measured in feet."""
    cases = {
        "Fort Myers FL": ((-82.0, 26.5, -81.7, 26.75), 2237),
        "Brooklyn NY": ((-74.05, 40.57, -73.83, 40.74), 2263),
        "Seattle WA": ((-122.44, 47.49, -122.22, 47.73), 2285),
    }
    for name, (bounds, expected_epsg) in cases.items():
        crs = pick_feet_crs(bounds)
        assert crs_is_feet(crs), f"{name}: {describe_crs(crs)} is not feet-based"
        assert crs.to_epsg() == expected_epsg, f"{name}: got {crs.to_epsg()}"


def test_tileset_and_join():
    cfg = base_config()
    parcels, adj_tiles, t2p, adj, _ = build_state(cfg)
    assert len(t2p) > 30, "suspiciously few populated tiles"
    assert parcels["tile_id"].notna().all(), "a parcel was left without a tile"
    assert set(t2p).issubset(set(adj_tiles.index))
    assert sum(len(v) for v in t2p.values()) == len(parcels), "parcels lost in the join"
    assert any(adj.values()), "adjacency graph has no edges"


def test_orphan_parcels_become_virtual_tiles():
    """Parcels outside the shatter get a tile of their own (SPEC_TILED 2)."""
    cfg = base_config()
    parcels = make_parcels()
    left_half = gpd.GeoDataFrame(
        {"geometry": [box(X0, Y0, X0 + W / 2, Y0 + H)]}, crs=CRS
    )
    tileset = T.build_tileset(
        left_half, None, None, None, None, CRS, grid_size_ft=1000.0, clip_water=False
    )
    joined, all_tiles, t2p = T.assign_parcels_to_tiles(parcels, tileset)

    n_virtual = int((joined["tile_id"] >= len(tileset)).sum())
    assert n_virtual > 100, "expected orphans on the uncovered half"
    assert joined["tile_id"].notna().all()
    assert all_tiles.index.is_unique and all_tiles.geometry.notna().all()
    assert set(t2p).issubset(set(all_tiles.index))

    rebuilt = pipeline._tiles_for(joined, tileset, CRS)
    assert rebuilt.index.equals(all_tiles.index)
    assert rebuilt.geometry.geom_equals(all_tiles.geometry).all()


def test_optimizer_improves_and_bookkeeping_stays_exact():
    cfg = base_config()
    parcels, _, t2p, adj, seed = build_state(cfg)
    opt = engine.TiledOptimizer(parcels, t2p, adj, cfg, neighborhood_ids=seed)

    opt.consolidate_mixed_tiles()
    before = float(np.mean(opt.scores))

    snaps = []
    tmp = Path(tempfile.mkdtemp())
    try:
        opt.run(store=CheckpointStore(tmp, keep=0),
                snapshot_cb=lambda ids, s: snaps.append(s.iteration))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    after = float(np.mean(opt.scores))
    assert opt.accepted > 0, "no moves were ever accepted"
    assert after > before, f"score did not improve ({before:.6f} -> {after:.6f})"
    assert len(snaps) >= 5, "snapshot callback never fired"

    # The incremental count tables must agree exactly with a full recompute.
    ct = mi.build_count_tables(
        parcels, cfg.weights, opt.n_neighborhoods, opt.parcel_n_ids,
        exact_mi=cfg.exact_mi,
    )
    drift = abs(float(np.mean(mi.all_scores(ct))) - after)
    assert drift < 1e-9, f"incremental bookkeeping drifted by {drift:.3e}"


def test_checkpoint_resume_is_bit_identical():
    """Stopping at N and resuming must follow the same trajectory as not stopping."""
    cfg = base_config()
    parcels, _, t2p, adj, seed = build_state(cfg)
    tmp = Path(tempfile.mkdtemp())
    try:
        # uninterrupted
        a = engine.TiledOptimizer(parcels, t2p, adj, cfg, neighborhood_ids=seed)
        a.consolidate_mixed_tiles()
        a.run()

        # stop at 200 ...
        cfg_b = base_config(max_iterations=200)
        b = engine.TiledOptimizer(parcels, t2p, adj, cfg_b, neighborhood_ids=seed)
        b.consolidate_mixed_tiles()
        store = CheckpointStore(tmp, keep=0)
        b.run(store=store)
        assert b.iteration == 200

        # ... and pick it back up
        cp = store.latest()
        assert cp is not None and cp.rng_state is not None
        c = engine.TiledOptimizer(parcels, t2p, adj, cfg, neighborhood_ids=seed)
        c.load_checkpoint(cp)
        assert c.iteration == 200
        c.run()

        assert (a.parcel_n_ids == c.parcel_n_ids).all(), (
            "resume forked the trajectory "
            f"(agreement {float((a.parcel_n_ids == c.parcel_n_ids).mean()):.2%})"
        )
        assert abs(float(np.mean(a.scores)) - float(np.mean(c.scores))) < 1e-12
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_checkpoint_pruning():
    cfg = base_config(max_iterations=600, checkpoint_every=100)
    parcels, _, t2p, adj, seed = build_state(cfg)
    tmp = Path(tempfile.mkdtemp())
    try:
        opt = engine.TiledOptimizer(parcels, t2p, adj, cfg, neighborhood_ids=seed)
        opt.consolidate_mixed_tiles()
        store = CheckpointStore(tmp, keep=3)
        opt.run(store=store)
        rolling = [c for c in store.list() if "final" not in c["path"]]
        assert len(rolling) <= 3, f"prune kept {len(rolling)} checkpoints"
        assert any("final" in c["path"] for c in store.list()), "no final checkpoint"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pipeline_prepare_caches_and_resumes(tmp_path=None):
    """Cold prepare writes artefacts; warm prepare reuses them and resumes."""
    work = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    area_4326 = make_area().to_crs(4326)
    jur = sources.Jurisdiction(
        name="Test City", layer_id=28, layer_name="Incorporated Place",
        geoid="1299999", state_fips="12", geometry=area_4326.geometry.iloc[0],
    )
    empty = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    roads_4326 = make_roads().to_crs(4326)

    saved = {name: getattr(sources, name) for name in
             ("get_jurisdiction", "fetch_census_blocks", "fetch_osm_roads",
              "fetch_osm_waterway_lines", "fetch_osm_water_areas")}
    sources.get_jurisdiction = lambda q, progress=None: jur
    sources.fetch_census_blocks = lambda j, progress=None, **k: empty
    sources.fetch_osm_roads = lambda j, progress=None, **k: roads_4326
    sources.fetch_osm_waterway_lines = lambda j, progress=None, **k: empty
    sources.fetch_osm_water_areas = lambda j, progress=None, **k: empty
    # The bulk path must be stubbed too, or this test quietly downloads an
    # entire state's block shapefile on any machine that has a network.
    saved_tiger = tiger.fetch_blocks
    tiger.fetch_blocks = lambda j, progress=None, **k: empty

    parcel_file = work / "parcels.parquet"
    make_parcels().to_parquet(parcel_file)

    try:
        cfg = base_config(
            jurisdiction_query="Test City, FL",
            parcel_path=str(parcel_file),
            work_dir=str(work / "runs"),
            max_iterations=200,
            checkpoint_every=100,
        )
        prep = pipeline.prepare(cfg, jurisdiction=jur)
        for f in ("jurisdiction.parquet", "tiles.parquet", "parcels_prepared.parquet",
                  "initial_neighborhoods.npz", "run_config.json"):
            assert (prep.run_dir / f).exists(), f"cold prepare did not write {f}"

        geoms = prep.tile_geometries()
        assert len(geoms) == prep.optimizer.n_tiles and geoms.notna().all()

        prep.optimizer.consolidate_mixed_tiles()
        prep.optimizer.run(store=prep.store)
        ids = prep.optimizer.parcel_n_ids.copy()
        score = float(np.mean(prep.optimizer.scores))

        warm = pipeline.prepare(cfg, run_dir=prep.run_dir)
        assert warm.run_dir == prep.run_dir
        assert warm.optimizer.n_tiles == prep.optimizer.n_tiles
        warm.optimizer.load_checkpoint(warm.store.latest())
        assert warm.optimizer.iteration == prep.optimizer.iteration
        assert (warm.optimizer.parcel_n_ids == ids).all()
        assert abs(float(np.mean(warm.optimizer.scores)) - score) < 1e-9
    finally:
        for name, fn in saved.items():
            setattr(sources, name, fn)
        tiger.fetch_blocks = saved_tiger
        if tmp_path is None:
            shutil.rmtree(work, ignore_errors=True)


def test_render_helpers():
    cfg = base_config()
    parcels, adj_tiles, t2p, adj, seed = build_state(cfg)
    opt = engine.TiledOptimizer(parcels, t2p, adj, cfg, neighborhood_ids=seed)
    geoms = adj_tiles.geometry.reindex(opt.tile_ids)
    verts, owner = build_polygons(geoms, simplify_tolerance=10.0)
    assert len(verts) == len(owner) > 0
    assert owner.max() < opt.n_tiles
    colors = neighborhood_colors(opt.tile_n_ids)
    assert colors.shape == (opt.n_tiles, 3)
    assert colors[owner].shape == (len(verts), 3)


# ==========================================================================


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        name = fn.__name__
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}\n      {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
