"""Why does a neighborhood look like confetti? Classify the adjacency graph.

The optimizer's notion of "contiguous" is whatever ``tile_adjacency`` says, and
that graph is built from a distance threshold (plus, in parcel mode, a sightline
test). A distance rule cannot distinguish three very different relationships:

* **shared border** -- the two tiles run along each other for some distance.
  This is what a person means by "next to".
* **corner only** -- they meet at a point, or along a hairline. Diagonal
  neighbours on the 500 ft lattice land here. Trading across one produces the
  checkerboard: the optimizer believes the pieces are joined, and on screen they
  visibly are not.
* **gap bridged** -- they never touch at all, and are joined purely by the
  ``adjacency_threshold_ft`` slack. Legitimate across-the-street neighbours look
  exactly like a hop over an intervening block.

Every consumer of the graph inherits the conflation: the annealer trades across
these edges, ``enforce_contiguity`` gates against them, and
``split_severed_neighborhoods`` decides severance with them. So a run can report
itself perfectly contiguous while the map is visibly shattered -- the gate is
measuring against the same permissive graph that allowed the damage.

This module re-derives the geometry behind each edge so the difference can be
counted and drawn. It is deliberately Qt-free: the GUI imports it, but so can a
script or a test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import shapely

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


# Edge classes. Ordered by how much they should be trusted.
ROOK, CORNER, GAP = 0, 1, 2
CLASS_NAMES = ("shared border", "corner only", "gap bridged")
#: Drawing colours, matched to the names above: green / red / amber.
CLASS_COLORS = ((0.13, 0.63, 0.30), (0.84, 0.15, 0.16), (0.95, 0.60, 0.07))

#: Below this, a "shared border" is a survey sliver or a rounding artefact
#: rather than a real party line. Two tiles meeting at a corner sometimes
#: produce a few hundredths of a foot of overlap instead of a clean point.
MIN_BORDER_FT = 1.0


# ==========================================================================
# Construction
# ==========================================================================


def edges_from_adjacency(
    adjacency: Dict[int, set], tile_ids: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten an id -> set-of-ids dict into undirected (left, right) positions.

    Positions index into ``tile_ids``, matching the order the optimizer compacts
    its tiles into, so the results line up with ``tile_n_ids`` without a further
    lookup. Ids the caller did not list are skipped -- the engine does the same
    when it builds its CSR, so including them here would describe a graph the
    optimizer never actually used.
    """
    pos = {int(t): i for i, t in enumerate(tile_ids)}
    left: list = []
    right: list = []
    for a, neighbours in adjacency.items():
        ia = pos.get(int(a))
        if ia is None:
            continue
        for b in neighbours:
            ib = pos.get(int(b))
            if ib is not None and ia < ib:  # undirected, no self-pairs
                left.append(ia)
                right.append(ib)
    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
    )


def classify_edges(
    geoms: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    min_border_ft: float = MIN_BORDER_FT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Label each edge ROOK / CORNER / GAP and measure its gap width."""
    if not len(left):
        return np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.float64)

    gap_ft = shapely.distance(geoms[left], geoms[right])
    touching = gap_ft <= 1e-9

    edge_class = np.full(len(left), GAP, dtype=np.int8)
    if touching.any():
        idx = np.where(touching)[0]
        # Only the touching pairs need an intersection, which is the expensive
        # part; the rest are settled by the distance alone.
        shared = shapely.length(shapely.intersection(geoms[left[idx]], geoms[right[idx]]))
        edge_class[idx] = np.where(shared > min_border_ft, ROOK, CORNER)
    return edge_class, gap_ft


# ==========================================================================
# Result
# ==========================================================================


@dataclass
class TileDiagnostics:
    """A classified adjacency graph, plus the geometry needed to draw it."""

    tile_ids: np.ndarray        # (N,) tile ids, in optimizer order
    left: np.ndarray            # (E,) endpoint positions into tile_ids
    right: np.ndarray           # (E,)
    edge_class: np.ndarray      # (E,) ROOK / CORNER / GAP
    gap_ft: np.ndarray          # (E,) 0.0 where the tiles touch
    points: np.ndarray          # (N, 2) a point inside each tile, for drawing
    seconds: float = 0.0

    @property
    def n_tiles(self) -> int:
        return len(self.tile_ids)

    @property
    def n_edges(self) -> int:
        return len(self.left)

    def class_mask(self, *classes: int) -> np.ndarray:
        return np.isin(self.edge_class, np.asarray(classes, dtype=np.int8))

    def segments(self) -> np.ndarray:
        """(E, 2, 2) line segments joining each edge's two tiles."""
        return np.stack([self.points[self.left], self.points[self.right]], axis=1)

    # ------------------------------------------------------------------
    def components(
        self, tile_n_ids: np.ndarray, classes: Sequence[int] = (ROOK,)
    ) -> Tuple[np.ndarray, int]:
        """Label each tile with its connected component, counting only ``classes``.

        Two tiles share a component only if they are in the same neighborhood
        *and* joined by an edge of an accepted class. Returns (labels, count of
        populated components).
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        tile_n_ids = np.asarray(tile_n_ids, dtype=np.int64)
        mask = self.class_mask(*classes)
        same = tile_n_ids[self.left[mask]] == tile_n_ids[self.right[mask]]
        a, b = self.left[mask][same], self.right[mask][same]
        n = self.n_tiles
        graph = coo_matrix(
            (np.ones(len(a), dtype=np.int8), (a, b)), shape=(n, n)
        )
        _, labels = connected_components(graph, directed=False)
        # A "component" is only interesting per neighborhood: the same label
        # cannot span two neighborhoods by construction, so counting distinct
        # (neighborhood, label) pairs counts fragments.
        pairs = np.unique(np.stack([tile_n_ids, labels], axis=1), axis=0)
        return labels, len(pairs)

    def weak_tiles(self, tile_n_ids: np.ndarray) -> np.ndarray:
        """Tiles sharing no real border with any tile of their own neighborhood.

        These are the ones held in place purely by a corner or a gap hop -- the
        visible checkerboard squares.
        """
        tile_n_ids = np.asarray(tile_n_ids, dtype=np.int64)
        same = tile_n_ids[self.left] == tile_n_ids[self.right]
        rook_same = same & (self.edge_class == ROOK)
        any_same = same

        deg_rook = np.zeros(self.n_tiles, dtype=np.int64)
        np.add.at(deg_rook, self.left[rook_same], 1)
        np.add.at(deg_rook, self.right[rook_same], 1)
        deg_any = np.zeros(self.n_tiles, dtype=np.int64)
        np.add.at(deg_any, self.left[any_same], 1)
        np.add.at(deg_any, self.right[any_same], 1)
        return (deg_rook == 0) & (deg_any > 0)

    # ------------------------------------------------------------------
    def summary(self, tile_n_ids: Optional[np.ndarray] = None) -> str:
        """A human-readable report, for the log and the diagnostics panel."""
        lines = [
            f"{self.n_tiles:,} tiles, {self.n_edges:,} edges "
            f"(mean degree {2 * self.n_edges / max(self.n_tiles, 1):.2f})",
            "",
            "Edge classes:",
        ]
        for cls in (ROOK, CORNER, GAP):
            n = int((self.edge_class == cls).sum())
            pct = 100.0 * n / max(self.n_edges, 1)
            lines.append(f"    {CLASS_NAMES[cls]:<16} {n:>9,}  {pct:5.1f}%")

        gaps = self.gap_ft[self.edge_class == GAP]
        if len(gaps):
            lines.append(
                f"    gap width: median {np.median(gaps):,.0f} ft, "
                f"max {gaps.max():,.0f} ft"
            )

        if tile_n_ids is None:
            return "\n".join(lines)

        tile_n_ids = np.asarray(tile_n_ids, dtype=np.int64)
        n_hoods = len(np.unique(tile_n_ids))
        _, all_c = self.components(tile_n_ids, (ROOK, CORNER, GAP))
        _, nodiag_c = self.components(tile_n_ids, (ROOK, GAP))
        _, rook_c = self.components(tile_n_ids, (ROOK,))
        weak = int(self.weak_tiles(tile_n_ids).sum())

        lines += [
            "",
            f"{n_hoods:,} neighborhoods. Fragments, by what counts as joined:",
            f"    the optimizer's own graph   {all_c:>9,}  "
            f"({all_c / max(n_hoods, 1):.1f} per neighborhood)",
            f"    if corners did not count    {nodiag_c:>9,}  "
            f"({nodiag_c / max(n_hoods, 1):.1f})",
            f"    shared borders only         {rook_c:>9,}  "
            f"({rook_c / max(n_hoods, 1):.1f})",
            "",
            f"    fragments hidden by corner edges: {nodiag_c - all_c:,}",
            f"    fragments hidden by gap edges:    "
            f"{self._nocorner_count(tile_n_ids) - all_c:,}",
            f"    tiles with no shared border to their own neighborhood: "
            f"{weak:,} ({100 * weak / max(self.n_tiles, 1):.1f}%)",
        ]
        return "\n".join(lines)

    def _nocorner_count(self, tile_n_ids: np.ndarray) -> int:
        _, c = self.components(tile_n_ids, (ROOK, CORNER))
        return c


# ==========================================================================
# Entry point
# ==========================================================================


def analyse(
    geometries,
    adjacency: Dict[int, set],
    tile_ids: Sequence[int],
    progress: Progress = _noop,
) -> TileDiagnostics:
    """Classify the adjacency graph the optimizer is actually using.

    ``geometries`` must be a GeoSeries in ``tile_ids`` order (what
    ``PreparedRun.tile_geometries()`` returns).
    """
    started = time.time()
    tile_ids = np.asarray(tile_ids, dtype=np.int64)
    geoms = np.asarray(geometries.values if hasattr(geometries, "values") else geometries)

    progress(f"Diagnostics: {len(tile_ids):,} tiles; flattening the graph...")
    left, right = edges_from_adjacency(adjacency, tile_ids)
    progress(f"Diagnostics: {len(left):,} undirected edges; classifying...")
    edge_class, gap_ft = classify_edges(geoms, left, right)

    # representative_point, not centroid: a tile clipped into a crescent by the
    # water pass can have its centroid outside itself, and an edge drawn from
    # there points at nothing.
    points = shapely.get_coordinates(shapely.point_on_surface(geoms))

    result = TileDiagnostics(
        tile_ids=tile_ids,
        left=left,
        right=right,
        edge_class=edge_class,
        gap_ft=gap_ft,
        points=points,
        seconds=time.time() - started,
    )
    progress(f"Diagnostics: done in {result.seconds:.1f}s")
    return result
