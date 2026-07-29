"""Tests for parcel-level adjacency and the line-of-sight rule.

    python tests/test_adjacency.py
    pytest tests/test_adjacency.py

Geometry is hand-built so every expected edge can be reasoned about exactly,
rather than asserted against whatever a random fixture happens to produce.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import adjacency  # noqa: E402


def lots(*specs):
    """Rectangles from (x, y, w, h) tuples."""
    return np.array([box(x, y, x + w, y + h) for x, y, w, h in specs], dtype=object)


def edge_set(result) -> set:
    return {(int(a), int(b)) for a, b in zip(result.left, result.right)}


# ==========================================================================
# The core rule
# ==========================================================================


def test_intervening_lot_blocks_but_the_gap_beyond_it_does_not():
    """Three lots in a row: A|B|C. A-C must be cut, A-B and B-C kept.

    This is the case a distance threshold gets wrong -- lots are narrower than
    the threshold, so A and C are "within 100 ft" despite B sitting between them.
    """
    #    A: 0-50      B: 60-110     C: 120-170     (10 ft gaps)
    g = lots((0, 0, 50, 100), (60, 0, 50, 100), (120, 0, 50, 100))
    res = adjacency.build_parcel_adjacency(g, threshold_ft=100.0, obstacle_geoms=g)
    assert edge_set(res) == {(0, 1), (1, 2)}, edge_set(res)
    assert res.n_candidates == 3, "A-C should have been a candidate"
    assert res.n_blocked == 1


def test_clear_gap_is_crossed():
    """Two lots facing each other across unparceled ground stay adjacent."""
    g = lots((0, 0, 50, 100), (130, 0, 50, 100))   # 80 ft of nothing between
    res = adjacency.build_parcel_adjacency(g, threshold_ft=100.0, obstacle_geoms=g)
    assert edge_set(res) == {(0, 1)}
    assert res.n_blocked == 0


def test_touching_pairs_are_kept_unconditionally():
    """Their shortest line is a zero-length point, often where 3+ lots meet.

    Testing that point for intersections finds the third lot and would delete a
    legitimate edge -- which is what pushed the measured drop rate from 24% to
    80% before this case was special-cased.
    """
    # four lots meeting at the corner (100, 100)
    g = lots((0, 0, 100, 100), (100, 0, 100, 100),
             (0, 100, 100, 100), (100, 100, 100, 100))
    res = adjacency.build_parcel_adjacency(g, threshold_ft=100.0, obstacle_geoms=g)
    assert res.n_touching == res.n_candidates, "all six pairs touch"
    assert res.n_blocked == 0
    assert len(edge_set(res)) == 6, "every pair must survive"


def test_transparent_obstacle_does_not_block():
    """A canal or right-of-way parcel between two lots must not sever them."""
    homes = lots((0, 0, 50, 100), (130, 0, 50, 100))
    canal = lots((55, -50, 70, 200))          # sits squarely between them
    blocked = adjacency.build_parcel_adjacency(
        homes, 100.0, obstacle_geoms=np.concatenate([homes, canal])
    )
    assert blocked.n_blocked == 1, "with the canal as an obstacle, expect a cut"
    clear = adjacency.build_parcel_adjacency(homes, 100.0, obstacle_geoms=homes)
    assert clear.n_blocked == 0, "with the canal transparent, expect no cut"
    assert edge_set(clear) == {(0, 1)}


def test_line_of_sight_can_be_disabled():
    g = lots((0, 0, 50, 100), (60, 0, 50, 100), (120, 0, 50, 100))
    res = adjacency.build_parcel_adjacency(
        g, 100.0, obstacle_geoms=g, require_line_of_sight=False
    )
    assert edge_set(res) == {(0, 1), (0, 2), (1, 2)}, "A-C should survive"
    assert res.n_blocked == 0


def test_edges_are_undirected_with_no_self_pairs():
    rng = np.random.default_rng(3)
    g = lots(*[(x, y, 40, 40) for x, y in
               zip(rng.uniform(0, 600, 60), rng.uniform(0, 600, 60))])
    res = adjacency.build_parcel_adjacency(g, 100.0, obstacle_geoms=g)
    assert (res.left < res.right).all(), "must be canonical (left < right)"
    assert len(edge_set(res)) == res.n_edges, "duplicate edges present"


def test_epsilon_is_relative_so_slivers_do_not_invert():
    """Survey slivers give sub-inch gaps; a flat shrink would flip the segment."""
    g = lots((0, 0, 50, 100), (50.01, 0, 50, 100))  # 0.01 ft apart
    res = adjacency.build_parcel_adjacency(
        g, 100.0, obstacle_geoms=g, epsilon_ft=0.5
    )
    assert edge_set(res) == {(0, 1)}, "a hairline gap must not be cut"


def test_threshold_is_respected():
    g = lots((0, 0, 50, 100), (200, 0, 50, 100))   # 150 ft apart
    assert adjacency.build_parcel_adjacency(g, 100.0, obstacle_geoms=g).n_edges == 0
    assert adjacency.build_parcel_adjacency(g, 200.0, obstacle_geoms=g).n_edges == 1


# ==========================================================================
# Lifting to tiles
# ==========================================================================


def test_tile_lift_ignores_intra_tile_pairs_and_keeps_all_tiles():
    g = lots((0, 0, 50, 100), (60, 0, 50, 100), (120, 0, 50, 100))
    res = adjacency.build_parcel_adjacency(g, 100.0, obstacle_geoms=g)
    # parcels 0 and 1 share tile 7; parcel 2 is in tile 9
    adj = res.to_tile_adjacency(np.array([7, 7, 9]), all_tiles=[7, 9, 11])
    assert adj[7] == {9} and adj[9] == {7}
    assert adj[11] == set(), "an isolated tile must still appear as a key"
    # 0-1 was intra-tile and must not have produced a self-edge
    assert 7 not in adj[7]


def test_tile_lift_is_symmetric():
    rng = np.random.default_rng(5)
    g = lots(*[(x, y, 40, 40) for x, y in
               zip(rng.uniform(0, 400, 50), rng.uniform(0, 400, 50))])
    res = adjacency.build_parcel_adjacency(g, 100.0, obstacle_geoms=g)
    tiles = rng.integers(0, 8, len(g))
    adj = res.to_tile_adjacency(tiles)
    for a, neighbours in adj.items():
        for b in neighbours:
            assert a in adj[b], f"{a}->{b} not mirrored"


# ==========================================================================
# Obstacle selection / jurisdiction portability
# ==========================================================================


def test_transparent_keywords_match_by_substring_not_exact_name():
    """Land-class vocabularies differ between jurisdictions."""
    classes = pd.Series([
        "SINGLE FAMILY RESIDENTIAL",     # blocks
        "RIGHT-OF-WAY",                  # Lee County wording
        "Road R/W",                      # some other county's wording
        "RIVERS, LAKES, SUBMERGED LAND",
        "VACANT AGRICULTURAL",           # blocks
        "utility easement",              # lowercase
    ])
    mask = adjacency.transparent_land_class_mask(classes)
    assert list(mask) == [False, True, True, True, False, True], list(mask)


def test_no_keyword_matches_is_reported_not_silent():
    logged = []
    classes = pd.Series(["FARMSTEAD", "DRYLAND CROP", "PASTURE"])
    mask = adjacency.transparent_land_class_mask(classes, progress=logged.append)
    assert not mask.any()
    assert any("No land classes matched" in m for m in logged), logged


def test_obstacle_modes_select_the_right_geometry():
    modeled = lots((0, 0, 10, 10), (20, 0, 10, 10))
    extra = lots((40, 0, 10, 10), (60, 0, 10, 10))
    every = np.concatenate([modeled, extra])
    classes = pd.Series(["HOUSE", "HOUSE", "RIGHT-OF-WAY", "WAREHOUSE"])

    assert len(adjacency.select_obstacles(modeled, "modeled")) == 2
    assert len(adjacency.select_obstacles(
        modeled, "all", all_geoms=every, all_land_class=classes)) == 4
    # all_except drops only the right-of-way row
    assert len(adjacency.select_obstacles(
        modeled, "all_except", all_geoms=every, all_land_class=classes)) == 3


def test_missing_land_class_degrades_to_all_with_a_warning():
    modeled = lots((0, 0, 10, 10))
    every = lots((0, 0, 10, 10), (20, 0, 10, 10))
    logged = []
    out = adjacency.select_obstacles(
        modeled, "all_except", all_geoms=every, all_land_class=None,
        progress=logged.append,
    )
    assert len(out) == 2, "should fall back to blocking on every parcel"
    assert any("land-class" in m for m in logged), logged


def test_unknown_obstacle_mode_is_rejected():
    try:
        adjacency.select_obstacles(lots((0, 0, 1, 1)), "sometimes")
    except ValueError as exc:
        assert "obstacle_mode" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_missing_unfiltered_file_falls_back_rather_than_crashing():
    modeled = lots((0, 0, 10, 10), (20, 0, 10, 10))
    logged = []
    out = adjacency.select_obstacles(
        modeled, "all", all_geoms=None, progress=logged.append
    )
    assert len(out) == 2
    assert any("falling back" in m for m in logged), logged


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
