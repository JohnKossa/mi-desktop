"""Edge classification and fragment counting on hand-checkable geometry."""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import diagnostics  # noqa: E402
import render  # noqa: E402


def _lattice(n=3, size=100.0, gap=0.0):
    """An n x n grid of squares, optionally separated by ``gap`` feet."""
    step = size + gap
    geoms, ids = [], []
    for r in range(n):
        for c in range(n):
            geoms.append(box(c * step, r * step, c * step + size, r * step + size))
            ids.append(r * n + c)
    return gpd.GeoSeries(geoms, index=ids, crs="EPSG:2236"), np.array(ids)


def _all_pairs_within(gdf, dist):
    import shapely

    tree = shapely.STRtree(gdf.values)
    pairs = tree.query(gdf.values, predicate="dwithin", distance=dist)
    adj = {int(i): set() for i in gdf.index}
    for a, b in zip(pairs[0], pairs[1]):
        if a != b:
            adj[int(gdf.index[a])].add(int(gdf.index[b]))
    return adj


def test_touching_lattice_splits_into_rook_and_corner():
    """A 3x3 of abutting squares: 12 shared borders and 8 corner-only diagonals.

    This is the whole problem in miniature -- 40% of the edges a distance rule
    finds here are diagonals, and nothing downstream can tell them apart.
    """
    geoms, ids = _lattice(3, 100.0, gap=0.0)
    adj = _all_pairs_within(geoms, 1.0)
    d = diagnostics.analyse(geoms, adj, ids)

    n_rook = int((d.edge_class == diagnostics.ROOK).sum())
    n_corner = int((d.edge_class == diagnostics.CORNER).sum())
    n_gap = int((d.edge_class == diagnostics.GAP).sum())
    assert (n_rook, n_corner, n_gap) == (12, 8, 0)


def test_separated_lattice_is_all_gap_bridged():
    # 40 ft apart orthogonally, so 56.6 ft on the diagonal: a 50 ft threshold
    # admits the first and not the second.
    geoms, ids = _lattice(3, 100.0, gap=40.0)
    adj = _all_pairs_within(geoms, 50.0)
    d = diagnostics.analyse(geoms, adj, ids)

    assert int((d.edge_class == diagnostics.GAP).sum()) == d.n_edges == 12
    assert np.allclose(d.gap_ft[d.edge_class == diagnostics.GAP], 40.0)


def test_checkerboard_is_one_component_by_corner_and_nine_by_border():
    """The exact failure mode: a neighborhood joined only at its corners."""
    geoms, ids = _lattice(5, 100.0, gap=0.0)
    adj = _all_pairs_within(geoms, 1.0)
    d = diagnostics.analyse(geoms, adj, ids)

    # Colour the 5x5 like a checkerboard; neighborhood 0 owns the 13 "black"
    # squares, which touch each other only at corners.
    rc = np.array([(i // 5, i % 5) for i in ids])
    tile_n = ((rc[:, 0] + rc[:, 1]) % 2).astype(np.int64)

    _, corner_ok = d.components(tile_n, (diagnostics.ROOK, diagnostics.CORNER))
    _, rook_only = d.components(tile_n, (diagnostics.ROOK,))
    assert corner_ok == 2          # two neighborhoods, each "connected"
    assert rook_only == 25         # ...and in truth, 25 separate squares

    weak = d.weak_tiles(tile_n)
    assert weak.all()              # nothing shares a border with its own colour


def test_solid_block_has_no_weak_tiles():
    geoms, ids = _lattice(4, 100.0, gap=0.0)
    adj = _all_pairs_within(geoms, 1.0)
    d = diagnostics.analyse(geoms, adj, ids)
    tile_n = np.zeros(len(ids), dtype=np.int64)

    assert not d.weak_tiles(tile_n).any()
    _, rook_only = d.components(tile_n, (diagnostics.ROOK,))
    assert rook_only == 1


def test_edges_skip_tiles_the_optimizer_does_not_own():
    """Adjacency entries outside ``tile_ids`` are dropped, as the engine does."""
    geoms, ids = _lattice(3, 100.0, gap=0.0)
    adj = _all_pairs_within(geoms, 1.0)
    adj[0].add(999)                       # a tile the optimizer never compacted
    adj[999] = {0}

    keep = ids[:8]
    left, right = diagnostics.edges_from_adjacency(adj, keep)
    assert left.max() < len(keep) and right.max() < len(keep)


def test_summary_reports_without_an_assignment():
    geoms, ids = _lattice(3, 100.0, gap=0.0)
    d = diagnostics.analyse(geoms, _all_pairs_within(geoms, 1.0), ids)
    text = d.summary()
    assert "corner only" in text
    assert "neighborhoods" not in text     # nothing to say without an assignment


def test_edge_rgba_hides_unselected_classes():
    cls = np.array([diagnostics.ROOK, diagnostics.CORNER, diagnostics.GAP])
    visible = np.array([False, True, True])
    rgba = render.edge_rgba(cls, diagnostics.CLASS_COLORS, visible, alpha=0.5)

    assert rgba[0, 3] == 0.0
    assert rgba[1, 3] == 0.5
    assert tuple(rgba[1, :3]) == diagnostics.CLASS_COLORS[diagnostics.CORNER]


def test_defect_colors_mark_only_weak_tiles():
    colors = render.defect_colors(np.array([True, False, True]))
    assert tuple(colors[0]) == render.DEFECT_FILL
    assert tuple(colors[1]) == render.OK_FILL
