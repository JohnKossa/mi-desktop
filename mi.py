"""Mutual-information scoring on count tables.

Same maths as ``main_tiled.ipynb`` sections 5/9, with two structural changes
that matter a lot for interactive use:

1. **One flat bin space.** All scored fields share a single count vector: field
   *f*'s bins occupy ``offsets[f]:offsets[f+1]``, and ``bin_weights`` carries the
   field weight for each slot. Because ``p(in)`` and ``p(out)`` depend only on
   the neighborhood size -- not on the field -- the entire weighted MI sum
   collapses to one dot product over ~150 numbers instead of one NumPy pass per
   field. That is the difference between ~6 and ~1 vectorised operations per
   candidate move.

2. **Scratch-buffer simulation.** Evaluating a move no longer allocates a copy
   of the count row; it writes into a reusable buffer, scores it, and the caller
   applies the delta in place only if the move is actually accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Scores handed to degenerate neighborhoods so donations can empty them out
# (SPEC.md, "Scoring for Absorption").
SCORE_SINGLETON = 1.0
SCORE_EMPTY = 2.0

#: (bin indices, counts) contributed by one tile, in the flat bin space.
Delta = Tuple[np.ndarray, np.ndarray]


@dataclass
class CountTables:
    fields: List[str]
    weights: np.ndarray          # (n_fields,)
    codes: List[np.ndarray]      # per field, per parcel: local bin index
    offsets: np.ndarray          # (n_fields + 1,) into the flat bin space
    n_bins_total: int
    bin_weights: np.ndarray      # (n_bins_total,) weight of the owning field
    global_counts: np.ndarray    # (n_bins_total,) int64
    neigh_counts: np.ndarray     # (n_neigh, n_bins_total) int64
    totals: np.ndarray           # (n_neigh,) int64
    total_g: int
    exact_mi: bool = False

    # cached constants -----------------------------------------------------
    log_total_g: float = 0.0
    log_global: np.ndarray = None  # type: ignore[assignment]
    _scratch: np.ndarray = None    # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.log_total_g = float(np.log(self.total_g)) if self.total_g > 0 else 0.0
        with np.errstate(divide="ignore"):
            self.log_global = np.log(self.global_counts.astype(np.float64))
        self._scratch = np.empty(self.n_bins_total, dtype=np.float64)

    @property
    def n_neighborhoods(self) -> int:
        return int(self.totals.shape[0])

    def flat_codes(self, parcel_idx: np.ndarray) -> np.ndarray:
        """Flat bin indices for a set of parcels, across every field."""
        return np.concatenate(
            [
                self.codes[fi][parcel_idx].astype(np.int64) + int(self.offsets[fi])
                for fi in range(len(self.fields))
            ]
        )

    def delta_for(self, parcel_idx: np.ndarray) -> Delta:
        bins, counts = np.unique(self.flat_codes(parcel_idx), return_counts=True)
        return bins.astype(np.int64), counts.astype(np.int64)


# ==========================================================================
# Construction
# ==========================================================================


def build_count_tables(
    df: pd.DataFrame,
    weights: Dict[str, float],
    n_neighborhoods: int,
    neighborhood_ids: np.ndarray,
    exact_mi: bool = False,
) -> CountTables:
    fields = list(weights.keys())
    w = np.array([float(weights[f]) for f in fields], dtype=np.float64)

    codes: List[np.ndarray] = []
    n_bins: List[int] = []
    for f in fields:
        vals = pd.to_numeric(df[f], errors="coerce")
        if vals.isna().any():
            # qcut with duplicates='drop' leaves NaN wherever the source value
            # was missing; give those their own bin so no parcel silently
            # vanishes from the count tables.
            _, arr = np.unique(vals.fillna(-1).to_numpy(), return_inverse=True)
            arr = arr.astype(np.int32)
        else:
            arr = vals.to_numpy().astype(np.int32)
            if arr.min() != 0:
                arr = arr - arr.min()
        codes.append(arr)
        n_bins.append(int(arr.max()) + 1)

    offsets = np.zeros(len(fields) + 1, dtype=np.int64)
    np.cumsum(n_bins, out=offsets[1:])
    n_total = int(offsets[-1])

    bin_weights = np.zeros(n_total, dtype=np.float64)
    for fi in range(len(fields)):
        bin_weights[offsets[fi] : offsets[fi + 1]] = w[fi]

    nids = np.asarray(neighborhood_ids, dtype=np.int64)
    total_g = len(df)

    global_counts = np.zeros(n_total, dtype=np.int64)
    neigh_counts = np.zeros((n_neighborhoods, n_total), dtype=np.int64)
    for fi in range(len(fields)):
        lo, b = int(offsets[fi]), n_bins[fi]
        global_counts[lo : lo + b] = np.bincount(codes[fi], minlength=b)
        flat = np.bincount(nids * b + codes[fi], minlength=n_neighborhoods * b)
        neigh_counts[:, lo : lo + b] = flat.reshape(n_neighborhoods, b)

    totals = np.bincount(nids, minlength=n_neighborhoods).astype(np.int64)

    return CountTables(
        fields=fields,
        weights=w,
        codes=codes,
        offsets=offsets,
        n_bins_total=n_total,
        bin_weights=bin_weights,
        global_counts=global_counts,
        neigh_counts=neigh_counts,
        totals=totals,
        total_g=total_g,
        exact_mi=exact_mi,
    )


# ==========================================================================
# Scoring
# ==========================================================================


def weighted_mi(ct: CountTables, counts: np.ndarray, total_n: int) -> float:
    """Weighted MI over every field at once.

    Expanding ``p log(p / (p_in p_val))`` in terms of raw counts turns the
    per-bin term into ``c * (log c + log G - log n - log c_total)``, so the only
    per-call transcendentals are two elementwise logs over the flat bin vector;
    ``log c_total`` and ``log G`` are cached on the table.
    """
    if total_n <= 0:
        return SCORE_EMPTY
    if total_n == 1:
        return SCORE_SINGLETON
    if total_n >= ct.total_g:
        return 0.0

    c_in = counts.astype(np.float64, copy=False)
    c_out = ct.global_counts - c_in

    k_in = ct.log_total_g - float(np.log(total_n))
    k_out = ct.log_total_g - float(np.log(ct.total_g - total_n))

    with np.errstate(divide="ignore", invalid="ignore"):
        t_in = c_in * (np.log(c_in) + k_in - ct.log_global)
        t_out = c_out * (np.log(c_out) + k_out - ct.log_global)

    occupied = c_in > 0
    t_in = np.where(occupied, t_in, 0.0)
    # The notebook only visits bins the neighborhood occupies, so bins with
    # c_in == 0 contribute nothing at all -- not even their "outside" term.
    out_mask = (c_out > 0) if ct.exact_mi else (occupied & (c_out > 0))
    t_out = np.where(out_mask, t_out, 0.0)

    return float(ct.bin_weights @ (t_in + t_out)) / ct.total_g


def score_neighborhood(ct: CountTables, n_id: int, total_n: Optional[int] = None) -> float:
    if total_n is None:
        total_n = int(ct.totals[n_id])
    return weighted_mi(ct, ct.neigh_counts[n_id], total_n)


def all_scores(ct: CountTables) -> np.ndarray:
    return np.array(
        [score_neighborhood(ct, n) for n in range(ct.n_neighborhoods)],
        dtype=np.float64,
    )


# ==========================================================================
# Move simulation
# ==========================================================================


def score_with_delta(
    ct: CountTables,
    n_id: int,
    remove: Optional[Delta],
    add: Optional[Delta],
    new_total: int,
) -> float:
    """Score ``n_id`` as if the deltas had been applied. Allocates nothing."""
    if new_total <= 0:
        return SCORE_EMPTY
    if new_total == 1:
        return SCORE_SINGLETON

    scratch = ct._scratch
    np.copyto(scratch, ct.neigh_counts[n_id])
    if remove is not None:
        scratch[remove[0]] -= remove[1]
    if add is not None:
        scratch[add[0]] += add[1]
    return weighted_mi(ct, scratch, new_total)


def apply_delta(
    ct: CountTables, n_id: int, remove: Optional[Delta], add: Optional[Delta]
) -> None:
    """Commit a move's count changes in place."""
    row = ct.neigh_counts[n_id]
    if remove is not None:
        row[remove[0]] -= remove[1]
    if add is not None:
        row[add[0]] += add[1]


def tile_deltas(ct: CountTables, tile_to_parcels: Dict[int, np.ndarray]) -> Dict[int, Delta]:
    """Pre-compute each tile's contribution to the flat bin space."""
    return {int(t): ct.delta_for(p) for t, p in tile_to_parcels.items()}
