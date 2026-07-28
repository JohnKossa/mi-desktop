"""Decomposing the tileset into independently optimizable pieces.

Adjacency is computed only between tiles that contain parcels, and a real county
is nothing like one connected blob: barrier islands, wide rivers and undeveloped
stretches sever the graph outright. Lee County splits into ~1,000 connected
components, the largest holding under 30% of the tiles.

Because a tile move only ever writes

  * the tile's own parcels and neighborhood label,
  * boundary-set entries for the tile and its one-hop neighbours,
  * the count-table rows of the two neighborhoods involved,

two moves in different components are non-interacting -- *provided no
neighborhood spans a component boundary*. KMeans seeds on lat/long/distance
without any notion of adjacency, so out of the box more than half of them do:
Lee County's seeding puts single neighborhoods on both banks of the
Caloosahatchee.

Splitting those is worth doing on its own merits. A neighborhood straddling a
severance can never be repaired by the optimizer, because trades only happen
along boundaries and there is no boundary between components -- so those parcels
are locked together for the whole run no matter how long it anneals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


# ==========================================================================
# Connected components
# ==========================================================================


@dataclass
class Components:
    """Connected components of the populated-tile adjacency graph."""

    tile_ids: np.ndarray        # (n_tiles,) sorted tile ids
    tile_label: np.ndarray      # (n_tiles,) component index per tile
    n_components: int
    tile_counts: np.ndarray     # (n_components,) tiles per component
    parcel_counts: np.ndarray   # (n_components,) parcels per component

    def order_by_size(self) -> np.ndarray:
        """Component indices, largest first (by parcel count)."""
        return np.argsort(-self.parcel_counts)

    def tiles_in(self, component: int) -> np.ndarray:
        return self.tile_ids[self.tile_label == component]

    def summary(self) -> str:
        largest = int(self.parcel_counts.max()) if self.n_components else 0
        total = int(self.parcel_counts.sum()) or 1
        ceiling = total / largest if largest else 1.0
        return (
            f"{len(self.tile_ids):,} populated tiles in {self.n_components:,} "
            f"components; largest holds {largest:,} parcels "
            f"({100 * largest / total:.1f}%), so parallel speedup tops out "
            f"around {ceiling:.1f}x"
        )


def find_components(
    tile_to_parcels: Dict[int, np.ndarray],
    tile_adjacency: Dict[int, set],
    progress: Progress = _noop,
) -> Components:
    """Label each populated tile with its connected component."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as _cc

    tile_ids = np.array(sorted(tile_to_parcels.keys()), dtype=np.int64)
    pos = {int(t): i for i, t in enumerate(tile_ids)}
    n = len(tile_ids)

    rows: List[int] = []
    cols: List[int] = []
    for t, neighbours in tile_adjacency.items():
        i = pos.get(int(t))
        if i is None:
            continue
        for other in neighbours:
            j = pos.get(int(other))
            if j is not None:
                rows.append(i)
                cols.append(j)

    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n)
    )
    n_components, labels = _cc(graph, directed=False)

    tile_counts = np.bincount(labels, minlength=n_components)
    weights = np.array(
        [len(tile_to_parcels[int(t)]) for t in tile_ids], dtype=np.int64
    )
    parcel_counts = np.bincount(
        labels, weights=weights, minlength=n_components
    ).astype(np.int64)

    comps = Components(
        tile_ids=tile_ids,
        tile_label=labels.astype(np.int64),
        n_components=int(n_components),
        tile_counts=tile_counts.astype(np.int64),
        parcel_counts=parcel_counts,
    )
    progress(comps.summary())
    return comps


def parcel_components(
    comps: Components, tile_to_parcels: Dict[int, np.ndarray], n_parcels: int
) -> np.ndarray:
    """Component index for every parcel (via the tile it sits in)."""
    out = np.full(n_parcels, -1, dtype=np.int64)
    for t, label in zip(comps.tile_ids, comps.tile_label):
        out[tile_to_parcels[int(t)]] = label
    return out


# ==========================================================================
# Neighborhood splitting
# ==========================================================================


def split_neighborhoods(
    parcel_n_ids: np.ndarray,
    parcel_component: np.ndarray,
    progress: Progress = _noop,
) -> Tuple[np.ndarray, int]:
    """Give each (neighborhood, component) pair its own neighborhood id.

    Returns the relabelled ids (contiguous from 0) and how many of the original
    neighborhoods had to be split.
    """
    n_ids = np.asarray(parcel_n_ids, dtype=np.int64)
    comp = np.asarray(parcel_component, dtype=np.int64)
    if len(n_ids) != len(comp):
        raise ValueError("parcel_n_ids and parcel_component must align")

    pairs = np.stack([n_ids, comp], axis=1)
    _, inverse = np.unique(pairs, axis=0, return_inverse=True)
    new_ids = inverse.astype(np.int64).reshape(-1)

    before = len(np.unique(n_ids))
    after = int(new_ids.max()) + 1 if len(new_ids) else 0

    # How many originals actually fragmented (as opposed to merely being
    # renumbered)?
    split = 0
    for k in np.unique(n_ids):
        if len(np.unique(comp[n_ids == k])) > 1:
            split += 1

    if split:
        progress(
            f"Split {split:,} of {before:,} seeded neighborhoods that straddled a "
            f"severed component ({before:,} -> {after:,} neighborhoods). Those "
            "could never have been separated by trading, since there is no "
            "boundary between components."
        )
    else:
        progress(f"No seeded neighborhood spans a component ({before:,} intact)")
    return new_ids, split


def any_spanning(
    parcel_n_ids: np.ndarray, parcel_component: np.ndarray
) -> bool:
    """Fast predicate: does *any* neighborhood straddle a component?

    ``spanning_neighborhoods`` builds a boolean mask per neighborhood, which is
    O(neighborhoods x parcels) -- half a billion operations at county scale. This
    answers the yes/no question with two ``np.unique`` passes, so it is cheap
    enough to consult before every parallel run.
    """
    n_ids = np.asarray(parcel_n_ids, dtype=np.int64)
    comp = np.asarray(parcel_component, dtype=np.int64)
    if not len(n_ids):
        return False
    pairs = np.unique(np.stack([n_ids, comp], axis=1), axis=0)
    _, counts = np.unique(pairs[:, 0], return_counts=True)
    return bool((counts > 1).any())


def spanning_neighborhoods(
    parcel_n_ids: np.ndarray, parcel_component: np.ndarray
) -> List[int]:
    """Neighborhood ids present in more than one component (diagnostic)."""
    n_ids = np.asarray(parcel_n_ids)
    comp = np.asarray(parcel_component)
    return [
        int(k) for k in np.unique(n_ids)
        if len(np.unique(comp[n_ids == k])) > 1
    ]


# ==========================================================================
# Work scheduling
# ==========================================================================


def group_components(
    weights: Sequence[int], n_groups: int
) -> List[List[int]]:
    """Longest-processing-time bin packing of components into worker groups.

    The largest component can't be subdivided, so it sets the wall clock; LPT
    gets the remainder close to evenly spread behind it, which is within a few
    percent of optimal for this shape of input.
    """
    n_groups = max(1, int(n_groups))
    order = np.argsort(-np.asarray(weights, dtype=np.int64))
    groups: List[List[int]] = [[] for _ in range(n_groups)]
    loads = np.zeros(n_groups, dtype=np.int64)
    for idx in order:
        if weights[idx] <= 0:
            continue
        target = int(np.argmin(loads))
        groups[target].append(int(idx))
        loads[target] += int(weights[idx])
    return [g for g in groups if g]


def useful_worker_count(
    weights: Sequence[int], requested: int, cap: int = 8
) -> int:
    """How many workers are worth starting.

    Past the point where the largest component dominates, extra processes only
    add overhead -- so the request is clamped to the theoretical ceiling.
    """
    w = np.asarray(weights, dtype=np.int64)
    w = w[w > 0]
    if len(w) <= 1:
        return 1
    ceiling = float(w.sum()) / float(w.max())
    if requested and requested > 0:
        return max(1, min(int(requested), len(w)))
    return max(1, min(int(np.floor(ceiling + 0.5)), len(w), cap))


def describe_grouping(
    groups: Sequence[Sequence[int]], weights: Sequence[int]
) -> str:
    loads = [sum(int(weights[c]) for c in g) for g in groups]
    total = sum(loads) or 1
    return (
        f"{len(groups)} worker group(s); parcel loads "
        f"{['{:,}'.format(l) for l in loads]}; "
        f"expected speedup {total / max(loads):.2f}x"
    )
