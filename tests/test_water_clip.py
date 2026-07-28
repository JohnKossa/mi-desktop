"""Regression tests for water clipping.

    python tests/test_water_clip.py
    pytest tests/test_water_clip.py

The bug these guard against: clipping water out of the *finished* tiles with
``base.geometry.difference(mask.union_all())`` is an elementwise difference
against one giant MultiPolygon, so every tile pays for all the water in the
county. At Lee County scale that measured ~11 minutes and looked like a hang.

Water is now removed from the study-area polygon before the shatter, so it costs
one difference rather than 380,000.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import LineString, box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiles as T  # noqa: E402

CRS = 2237
X0, Y0, W, H = 700_000.0, 700_000.0, 20_000.0, 20_000.0


def make_area() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": [box(X0, Y0, X0 + W, Y0 + H)]}, crs=CRS)


def make_water(n_canals: int = 300, n_lakes: int = 80, seed: int = 5):
    """A river, a canal network and lakes -- shaped like Lee County's reality."""
    rng = np.random.default_rng(seed)
    river = LineString([(X0, Y0 + H / 2), (X0 + W, Y0 + H / 2 + 800)]).buffer(500)
    canals = [
        LineString(
            [(x, y), (x + rng.uniform(400, 2500), y + rng.normal(0, 40))]
        ).buffer(60)
        for x, y in zip(
            rng.uniform(X0, X0 + W, n_canals), rng.uniform(Y0, Y0 + H, n_canals)
        )
    ]
    lakes = [
        box(x, y, x + rng.uniform(300, 900), y + rng.uniform(300, 900))
        for x, y in zip(
            rng.uniform(X0, X0 + W, n_lakes), rng.uniform(Y0, Y0 + H, n_lakes)
        )
    ]
    return gpd.GeoDataFrame({"geometry": [river] + canals + lakes}, crs=CRS)


def water_fraction(tiles: gpd.GeoDataFrame, water_union) -> np.ndarray:
    shapely.prepare(water_union)
    inter = shapely.intersection(tiles.geometry.to_numpy(), water_union)
    return shapely.area(inter) / tiles.geometry.area.to_numpy()


# ==========================================================================


def test_tiles_do_not_overlap_water():
    water = make_water()
    wu = water.geometry.union_all()
    tiles = T.build_tileset(
        make_area(), None, None, None, water, CRS,
        grid_size_ft=500.0, clip_water=True,
    )
    assert len(tiles) > 100

    frac = water_fraction(tiles, wu)
    # Shorelines are cut lines, so tiles should stop at them cleanly. Allow a
    # whisker for floating-point noise on shared edges.
    assert (frac > 0.01).sum() == 0, (
        f"{int((frac > 0.01).sum())} tiles have >1% of their area in water "
        f"(worst {frac.max():.3f})"
    )


def test_clip_water_off_leaves_water_covered():
    """Control: without clipping, tiles do sit on top of water."""
    water = make_water()
    wu = water.geometry.union_all()
    tiles = T.build_tileset(
        make_area(), None, None, None, water, CRS,
        grid_size_ft=500.0, clip_water=False,
    )
    frac = water_fraction(tiles, wu)
    assert (frac > 0.5).sum() > 0, (
        "unclipped run has no water-covered tiles, so the clipped test is vacuous"
    )


def test_clipping_is_not_quadratic_in_tile_count():
    """Runtime must not scale with tiles x mask complexity.

    A 4x finer grid is ~16x the tiles. The old post-hoc clip scaled linearly in
    tiles *and* in mask parts, so this ratio blew up; removing water from the
    study area up front makes the water handling a fixed cost.
    """
    water = make_water()
    area = make_area()

    t = time.perf_counter()
    coarse = T.build_tileset(area, None, None, None, water, CRS,
                             grid_size_ft=2000.0, clip_water=True)
    coarse_s = time.perf_counter() - t

    t = time.perf_counter()
    fine = T.build_tileset(area, None, None, None, water, CRS,
                           grid_size_ft=500.0, clip_water=True)
    fine_s = time.perf_counter() - t

    tile_ratio = len(fine) / max(len(coarse), 1)
    time_ratio = fine_s / max(coarse_s, 1e-3)
    assert tile_ratio > 4, f"fixture didn't scale up ({tile_ratio:.1f}x tiles)"
    # Generous bound: we only need to catch a return to per-tile clipping, which
    # would make time grow at least as fast as tile count.
    assert time_ratio < tile_ratio, (
        f"time grew {time_ratio:.1f}x for {tile_ratio:.1f}x tiles -- clipping "
        "looks per-tile again"
    )


def test_water_covering_everything_does_not_empty_the_tileset():
    """Degenerate input must warn, not produce zero tiles or raise."""
    area = make_area()
    everything = gpd.GeoDataFrame(
        {"geometry": [box(X0 - 1000, Y0 - 1000, X0 + W + 1000, Y0 + H + 1000)]},
        crs=CRS,
    )
    logged = []
    tiles = T.build_tileset(
        area, None, None, None, everything, CRS,
        grid_size_ft=2000.0, clip_water=True, progress=logged.append,
    )
    assert len(tiles) > 0, "study area was clipped out of existence"
    assert any("entire study area" in m for m in logged), logged


def test_clip_out_helper_matches_the_naive_union_difference():
    """The retained helper must agree with the slow one-liner it replaced."""
    water = make_water(n_canals=60, n_lakes=20)
    tiles = T.build_tileset(
        make_area(), None, None, None, None, CRS,
        grid_size_ft=2000.0, clip_water=False,
    )
    fast = T.clip_out(tiles, water, min_area_sqft=0.0)

    naive = tiles.geometry.difference(water.geometry.union_all())
    naive = naive[~naive.is_empty & naive.notna()]
    naive = gpd.GeoDataFrame({"geometry": naive}, crs=CRS).explode(
        ignore_index=True, index_parts=False
    )

    assert len(fast) == len(naive), (len(fast), len(naive))
    assert abs(fast.geometry.area.sum() - naive.geometry.area.sum()) < 1.0


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
