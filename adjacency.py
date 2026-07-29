"""Parcel-level adjacency with a line-of-sight rule.

A distance threshold alone cannot tell these two situations apart:

* two lots facing each other across a canal or a street -- genuinely neighbours;
* two lots on the same street with a third lot between them -- not neighbours,
  but within 100 ft of each other because lots are only 60 ft wide.

What distinguishes them is whether there is *parcel* in the way. Streets and
canals are generally unparceled (in Lee County only 598 of 558,688 parcels are
RIGHT-OF-WAY and 338 are submerged land), so a line cast across one hits
nothing, while a line cast over an intervening lot hits it.

So: take candidate pairs within the threshold, then keep an edge only if the
shortest segment between the two parcels is clear of any obstacle.

Two details that matter more than they look:

* **Touching pairs are kept unconditionally.** Their shortest line is a
  zero-length point, and in a subdivision that point usually sits where three or
  four lots meet -- so an intersection test would "find" a third parcel there and
  delete a legitimate edge. Measured on Lee County, 30% of candidate pairs touch,
  and not special-casing them pushed the drop rate from 24% to 80%.
* **The segment is pulled in from both ends.** A shortest line's endpoints lie
  *on* both parcels' boundaries by construction, so it always touches its own two
  parcels and anything meeting them at that point. Shrinking by a hair removes
  that, and once shrunk the segment cannot touch either endpoint parcel at all --
  which is why no pair-exclusion bookkeeping is needed and the obstacle set can be
  an unrelated frame.

Which parcels count as obstacles is the real control, and it subsumes the
question of whether to bridge across empty ground:

* ``"modeled"``   -- only the parcels being optimized. Agricultural land, condos
  and vacant lots are transparent, so single-family pockets separated by a "sea"
  of farmland reconnect. This is empty-tile bridging, but geometric and targeted.
* ``"all"``       -- every parcel in the file blocks. Farmland severs.
* ``"all_except"``-- every parcel except classes that sit in the gaps between
  homes (right-of-way, submerged land, utility, common elements...). Streets and
  canals stay transparent; everything else blocks.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import shapely

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


#: Substrings, not exact class names -- land-class vocabularies differ between
#: jurisdictions, so "RIGHT-OF-WAY" here may be "ROW" or "Road R/W" elsewhere.
#: Matched case-insensitively against the land-class column.
DEFAULT_TRANSPARENT_KEYWORDS: Tuple[str, ...] = (
    "right-of-way", "right of way", "row",
    "submerged", "river", "lake", "canal", "ditch", "drain", "waterway",
    "utility", "subsurface", "sewage", "waste land",
    "common element", "common area",
    "street", "road", "highway", "easement",
)


def transparent_land_class_mask(
    land_class: pd.Series,
    keywords: Sequence[str] = DEFAULT_TRANSPARENT_KEYWORDS,
    progress: Progress = _noop,
) -> np.ndarray:
    """Which parcels sit in the gaps and should not block a sightline.

    Keyword matching rather than an exact list, so this transfers to a
    jurisdiction whose land-class vocabulary we have never seen. The classes it
    actually matched are logged, because that is the first thing to check when
    the results look wrong somewhere new.
    """
    text = land_class.astype(str).fillna("")
    pattern = re.compile(
        "|".join(re.escape(k) for k in keywords if k), re.IGNORECASE
    )
    mask = text.str.contains(pattern, regex=True, na=False).to_numpy()

    if mask.any():
        matched = (
            pd.Series(text[mask]).value_counts().head(12)
        )
        progress(f"Treating {int(mask.sum()):,} parcels as transparent (in-gap):")
        for name, count in matched.items():
            progress(f"    {count:>8,}  {name}")
        extra = int(pd.Series(text[mask]).nunique()) - len(matched)
        if extra > 0:
            progress(f"    ... and {extra} more class(es)")
    else:
        progress(
            "No land classes matched the transparent keywords -- every parcel "
            "will block a sightline. Check the land-class column and keywords "
            "if this jurisdiction parcelises its streets or canals."
        )
    return mask


# ==========================================================================
# Core builder
# ==========================================================================


@dataclass
class AdjacencyResult:
    """Undirected parcel edges, as positional indices into the modeled parcels."""

    left: np.ndarray
    right: np.ndarray
    n_candidates: int = 0
    n_touching: int = 0
    n_blocked: int = 0
    seconds: float = 0.0

    @property
    def n_edges(self) -> int:
        return len(self.left)

    def summary(self) -> str:
        pct = 100 * self.n_blocked / self.n_candidates if self.n_candidates else 0.0
        return (
            f"{self.n_candidates:,} candidate pairs -> {self.n_edges:,} edges "
            f"({self.n_touching:,} touching kept, {self.n_blocked:,} blocked by an "
            f"intervening parcel = {pct:.1f}%) in {self.seconds:.1f}s"
        )

    def to_tile_adjacency(
        self, parcel_tile_ids: np.ndarray, all_tiles: Optional[Sequence[int]] = None
    ) -> Dict[int, set]:
        """Lift parcel edges to tile edges.

        Two tiles are adjacent iff any parcel in one is adjacent to any parcel in
        the other. Pairs inside a single tile carry no information -- the
        optimizer already moves a tile's parcels as one unit.
        """
        tiles = np.asarray(parcel_tile_ids, dtype=np.int64)
        tl, tr = tiles[self.left], tiles[self.right]
        cross = tl != tr
        lo = np.minimum(tl[cross], tr[cross])
        hi = np.maximum(tl[cross], tr[cross])

        adj: Dict[int, set] = {}
        if all_tiles is not None:
            for t in all_tiles:
                adj[int(t)] = set()
        for a, b in np.unique(np.stack([lo, hi], axis=1), axis=0):
            a, b = int(a), int(b)
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        return adj


def build_parcel_adjacency(
    modeled_geoms: np.ndarray,
    threshold_ft: float = 100.0,
    obstacle_geoms: Optional[np.ndarray] = None,
    require_line_of_sight: bool = True,
    epsilon_ft: float = 0.5,
    progress: Progress = _noop,
) -> AdjacencyResult:
    """Adjacency among ``modeled_geoms``, optionally filtered by line of sight."""
    started = time.time()
    n = len(modeled_geoms)
    if n == 0:
        return AdjacencyResult(np.zeros(0, np.int64), np.zeros(0, np.int64))

    progress(f"Adjacency: {n:,} parcels, {threshold_ft:.0f} ft threshold...")
    tree = shapely.STRtree(modeled_geoms)
    pairs = tree.query(modeled_geoms, predicate="dwithin", distance=threshold_ft)
    left, right = pairs[0], pairs[1]
    upper = left < right          # undirected, and drops self-pairs
    left, right = left[upper], right[upper]
    n_candidates = len(left)
    progress(f"Adjacency: {n_candidates:,} candidate pairs")

    if not require_line_of_sight or obstacle_geoms is None or not len(obstacle_geoms):
        return AdjacencyResult(
            left, right, n_candidates=n_candidates,
            seconds=time.time() - started,
        )

    gap = shapely.distance(modeled_geoms[left], modeled_geoms[right]) > 1e-6
    n_touching = int((~gap).sum())
    gl, gr = left[gap], right[gap]
    progress(
        f"Adjacency: {n_touching:,} touching pairs kept outright; "
        f"testing sightlines for {len(gl):,}"
    )

    kept_gapped = np.zeros(0, dtype=bool)
    if len(gl):
        seg = shapely.shortest_line(modeled_geoms[gl], modeled_geoms[gr])
        coords = shapely.get_coordinates(seg).reshape(-1, 2, 2)
        a, b = coords[:, 0, :], coords[:, 1, :]
        length = np.linalg.norm(b - a, axis=1)
        safe = np.where(length > 0, length, 1.0)
        unit = (b - a) / safe[:, None]
        # Relative, not fixed: survey slivers produce sub-inch gaps and a flat
        # shrink would invert those segments.
        eps = np.minimum(epsilon_ft, safe * 0.25)[:, None]
        inner = shapely.linestrings(
            np.stack([a + unit * eps, b - unit * eps], axis=1)
        )

        obstacles = shapely.STRtree(obstacle_geoms)
        hits = obstacles.query(inner, predicate="intersects")
        blocked = np.zeros(len(gl), dtype=bool)
        blocked[np.unique(hits[0])] = True
        kept_gapped = ~blocked

    left = np.concatenate([left[~gap], gl[kept_gapped]])
    right = np.concatenate([right[~gap], gr[kept_gapped]])
    order = np.lexsort((right, left))

    result = AdjacencyResult(
        left=left[order], right=right[order],
        n_candidates=n_candidates, n_touching=n_touching,
        n_blocked=int((~kept_gapped).sum()) if len(kept_gapped) else 0,
        seconds=time.time() - started,
    )
    progress("Adjacency: " + result.summary())
    return result


# ==========================================================================
# Obstacle assembly
# ==========================================================================


OBSTACLE_MODES = ("modeled", "all", "all_except")


def select_obstacles(
    modeled_geoms: np.ndarray,
    mode: str = "modeled",
    all_geoms: Optional[np.ndarray] = None,
    all_land_class: Optional[pd.Series] = None,
    keywords: Sequence[str] = DEFAULT_TRANSPARENT_KEYWORDS,
    progress: Progress = _noop,
) -> np.ndarray:
    """Assemble the geometries that block a sightline, per ``mode``."""
    if mode not in OBSTACLE_MODES:
        raise ValueError(f"obstacle_mode must be one of {OBSTACLE_MODES}, got {mode!r}")

    if mode == "modeled" or all_geoms is None or not len(all_geoms):
        if mode != "modeled":
            progress(
                f"obstacle_mode={mode!r} needs the unfiltered parcel file; "
                "falling back to the modeled parcels only."
            )
        progress(f"Obstacles: the {len(modeled_geoms):,} modeled parcels")
        return modeled_geoms

    if mode == "all":
        progress(f"Obstacles: all {len(all_geoms):,} parcels")
        return all_geoms

    if all_land_class is None:
        progress(
            "obstacle_mode='all_except' needs a land-class column; no such column "
            "was found, so all parcels will block."
        )
        return all_geoms

    transparent = transparent_land_class_mask(
        all_land_class, keywords, progress=progress
    )
    kept = all_geoms[~transparent]
    progress(
        f"Obstacles: {len(kept):,} of {len(all_geoms):,} parcels "
        f"({int(transparent.sum()):,} treated as in-gap and transparent)"
    )
    return kept
