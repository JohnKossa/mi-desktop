"""The tile-level simulated-annealing optimizer, made pausable and resumable.

This is ``run_tiled_optimization`` from ``main_tiled.ipynb`` restructured so it
can drive a GUI:

* tile ids are compacted to contiguous positions and adjacency is stored CSR-
  style, so the hot loop touches NumPy arrays instead of dicts of sets;
* the batch of candidate boundary edges is capped (``max_batch``) so one
  iteration takes a predictable amount of time even on a county-sized boundary;
* ``pause_event`` / ``stop_event`` let the UI interrupt cleanly;
* ``snapshot_cb`` hands the UI a tile -> neighborhood array every
  ``refresh_every`` iterations;
* checkpoints capture the full RNG + annealing state, so resuming continues the
  identical trajectory rather than restarting the cooling schedule.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import mi
from checkpoints import Checkpoint, CheckpointStore
from config import DERIVED_RATIOS, RunConfig

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


# ==========================================================================
# Preparation
# ==========================================================================


def add_derived_columns(df: pd.DataFrame, progress: Progress = _noop) -> pd.DataFrame:
    """Add assr_impr_ppsf / assr_land_ppsf when their inputs are present."""
    out = df.copy()
    for new_col, num, den in DERIVED_RATIOS:
        if new_col in out.columns:
            continue
        if num in out.columns and den in out.columns:
            denom = pd.to_numeric(out[den], errors="coerce")
            denom = denom.where(denom > 0)
            out[new_col] = pd.to_numeric(out[num], errors="coerce") / denom
            progress(f"Derived {new_col} = {num} / {den}")
    return out


def bin_continuous_fields(
    df: pd.DataFrame, fields: Sequence[str], max_bins: int, progress: Progress = _noop
) -> pd.DataFrame:
    out = df.copy()
    for f in fields:
        if f not in out.columns:
            progress(f"Skipping binning for missing column '{f}'")
            continue
        binned = pd.qcut(
            pd.to_numeric(out[f], errors="coerce"),
            q=max_bins,
            labels=False,
            duplicates="drop",
        )
        if binned.isna().all():
            # qcut returns all-NaN (no exception) for a constant or all-null
            # column. Left unsaid, that field would quietly contribute zero
            # signal to the MI score instead of failing loudly.
            progress(
                f"WARNING: '{f}' is constant or all-null, so '{f}_binned' has a "
                "single bin and contributes no information to the score."
            )
        out[f"{f}_binned"] = binned
    return out


def seed_neighborhoods(
    df: pd.DataFrame,
    fields: Sequence[str],
    n_neighborhoods: int,
    random_state: int = 42,
    progress: Progress = _noop,
) -> np.ndarray:
    usable = [f for f in fields if f in df.columns]
    if not usable:
        raise ValueError(f"None of the seed fields {list(fields)} exist in the parcels.")
    if len(usable) != len(fields):
        progress(f"Seeding on {usable} (missing: {set(fields) - set(usable)})")

    feats = df[usable].apply(pd.to_numeric, errors="coerce")
    feats = feats.fillna(feats.median(numeric_only=True))
    scaled = StandardScaler().fit_transform(feats.to_numpy())

    k = min(int(n_neighborhoods), len(df))
    progress(f"KMeans seeding {k} neighborhoods on {len(df):,} parcels...")
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    return km.fit_predict(scaled).astype(np.int64)


# ==========================================================================
# Optimizer
# ==========================================================================


@dataclass
class OptimizerStats:
    iteration: int = 0
    temperature: float = 0.0
    mean_score: float = 0.0
    boundary_size: int = 0
    accepted: int = 0
    rejected: int = 0
    active_neighborhoods: int = 0
    elapsed_s: float = 0.0
    iterations_this_run: int = 0
    # Parcel-weighted mean score. The plain mean is badly misleading once
    # severed components have been split apart: singletons score 1.0 and empties
    # 2.0 by design (SPEC.md, absorption), so a county with hundreds of
    # one-parcel island neighborhoods reports a mean ~60x its real value.
    weighted_score: float = 0.0
    score_num: float = 0.0   # sum(score * parcels), for aggregating across workers
    score_den: float = 0.0   # sum(parcels)
    message: str = ""

    @property
    def rate(self) -> float:
        """Iterations per second for the current ``run()`` call."""
        return self.iterations_this_run / self.elapsed_s if self.elapsed_s > 0 else 0.0


class TiledOptimizer:
    """Resumable tile-level simulated annealing over MI count tables."""

    #: How often the boundary sampling pool is re-materialised from the set.
    POOL_REFRESH = 100

    def __init__(
        self,
        parcels: pd.DataFrame,
        tile_to_parcels: Dict[int, np.ndarray],
        tile_adjacency: Dict[int, set],
        cfg: RunConfig,
        neighborhood_ids: Optional[np.ndarray] = None,
        progress: Progress = _noop,
    ) -> None:
        self.cfg = cfg
        self.parcels = parcels
        self.progress = progress

        # --- compact tile ids -------------------------------------------
        self.tile_ids = np.array(sorted(tile_to_parcels.keys()), dtype=np.int64)
        self.tile_pos = {int(t): i for i, t in enumerate(self.tile_ids)}
        self.n_tiles = len(self.tile_ids)
        self.tile_parcels: List[np.ndarray] = [
            tile_to_parcels[int(t)] for t in self.tile_ids
        ]
        # Every parcel this optimizer owns, cached: recomputing it per stats call
        # would mean concatenating tens of thousands of index arrays.
        self._owned_parcels = (
            np.concatenate(self.tile_parcels) if self.tile_parcels
            else np.zeros(0, dtype=np.int64)
        )

        # --- adjacency as CSR over tile positions ------------------------
        indptr = np.zeros(self.n_tiles + 1, dtype=np.int64)
        neigh_lists: List[List[int]] = []
        for i, t in enumerate(self.tile_ids):
            ns = [
                self.tile_pos[int(n)]
                for n in tile_adjacency.get(int(t), ())
                if int(n) in self.tile_pos
            ]
            neigh_lists.append(ns)
            indptr[i + 1] = indptr[i] + len(ns)
        self.adj_indptr = indptr
        self.adj_indices = np.fromiter(
            (n for ns in neigh_lists for n in ns), dtype=np.int64, count=int(indptr[-1])
        )

        # --- neighborhood assignment -------------------------------------
        if neighborhood_ids is None:
            neighborhood_ids = parcels["neighborhood_id"].to_numpy()
        self.parcel_n_ids = np.asarray(neighborhood_ids, dtype=np.int64).copy()
        self.n_neighborhoods = int(self.parcel_n_ids.max()) + 1

        # --- count tables -------------------------------------------------
        self.ct = mi.build_count_tables(
            parcels,
            cfg.weights,
            self.n_neighborhoods,
            self.parcel_n_ids,
            exact_mi=cfg.exact_mi,
        )
        self.n_fields = len(self.ct.fields)
        self._build_deltas()

        self.scores = mi.all_scores(self.ct)

        # --- annealing state ----------------------------------------------
        self.tile_n_ids = np.zeros(self.n_tiles, dtype=np.int64)
        self._recompute_tile_ids()
        self.boundary: set = set()
        self._rebuild_boundary()

        # neighborhood id -> set of tile positions, for the contiguity gate.
        self.n_to_tiles: Dict[int, set] = {}
        self._rebuild_n_to_tiles()
        if cfg.enforce_contiguity:
            self._report_initial_connectivity()

        # Per-parcel iteration of last label change, for the convergence test.
        # Seeded to 0 so a fresh run reads as "everything just changed" and
        # cannot exit before it has any history.
        self.last_change_iter = np.zeros(len(self.parcel_n_ids), dtype=np.int64)
        # Cumulative parcel-relabel events, and periodic marks of it, so the
        # convergence test can ask "how many of the parcels I touched recently
        # were touched for the first time?" rather than measuring raw volume.
        self.touch_events = 0
        self._event_marks: List[Tuple[int, int]] = []
        self._stable_streak = 0
        self._last_batch = cfg.max_batch
        self.blocked_batches = 0  # batches where every candidate broke contiguity

        self.iteration = 0
        self.temperature = float(cfg.initial_temp)
        self.stability_counter = 0
        self.accepted = 0
        self.rejected = 0
        self.rng = np.random.default_rng(cfg.random_seed)

        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self._paused_saved = False
        self._boundary_list: List[Tuple[int, int]] = sorted(self.boundary)
        self._boundary_stamp = -1

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _build_deltas(self) -> None:
        """Each tile's (flat bin indices, counts) contribution, cached once."""
        self.tile_delta_list: List[mi.Delta] = [
            self.ct.delta_for(p) for p in self.tile_parcels
        ]

    def _recompute_tile_ids(self) -> None:
        for i, pidx in enumerate(self.tile_parcels):
            vals = self.parcel_n_ids[pidx]
            self.tile_n_ids[i] = vals[0] if len(vals) else -1

    def _tile_is_mixed(self, i: int) -> bool:
        vals = self.parcel_n_ids[self.tile_parcels[i]]
        return len(vals) > 1 and bool((vals != vals[0]).any())

    def _rebuild_boundary(self) -> None:
        boundary: set = set()
        indptr, indices, tn = self.adj_indptr, self.adj_indices, self.tile_n_ids
        for i in range(self.n_tiles):
            ni = tn[i]
            for k in range(indptr[i], indptr[i + 1]):
                j = int(indices[k])
                if ni != tn[j]:
                    boundary.add((i, j) if i < j else (j, i))
            if self._tile_is_mixed(i):
                boundary.add((i, i))
        self.boundary = boundary

    def _neighbors(self, i: int) -> np.ndarray:
        return self.adj_indices[self.adj_indptr[i] : self.adj_indptr[i + 1]]

    # ------------------------------------------------------------------
    # Contiguity
    # ------------------------------------------------------------------

    def _rebuild_n_to_tiles(self) -> None:
        out: Dict[int, set] = {}
        for pos in range(self.n_tiles):
            out.setdefault(int(self.tile_n_ids[pos]), set()).add(pos)
        self.n_to_tiles = out

    def _components_of(self, tiles: set) -> int:
        """Number of connected components in a set of tile positions."""
        remaining = set(tiles)
        count = 0
        while remaining:
            count += 1
            stack = [remaining.pop()]
            while stack:
                cur = stack.pop()
                for nb in self._neighbors(cur):
                    nb = int(nb)
                    if nb in remaining:
                        remaining.discard(nb)
                        stack.append(nb)
        return count

    def _report_initial_connectivity(self) -> None:
        counts = {n: self._components_of(t) for n, t in self.n_to_tiles.items()}
        broken = {n: c for n, c in counts.items() if c > 1}
        total = sum(counts.values())
        self.progress(
            f"Contiguity: {len(counts):,} neighborhoods in {total:,} components; "
            f"{len(broken):,} already disconnected."
        )
        if broken:
            worst = sorted(broken.items(), key=lambda kv: -kv[1])[:5]
            self.progress(
                f"  Worst (id, components): {worst}. Pre-existing exclaves are "
                "preserved -- the gate only prevents NEW disconnection."
            )

    def _removal_disconnects(self, n_id: int, tile: int) -> bool:
        """Would removing ``tile`` from ``n_id`` split what's left of it?

        A local articulation test: every tile of ``n_id`` adjacent to ``tile`` was
        reachable through it, so if they are still mutually reachable without it,
        nothing fragmented. Anything already disconnected elsewhere in the
        neighborhood is untouched by this and stays that way.
        """
        owned = self.n_to_tiles.get(n_id)
        if not owned:
            return False
        anchors = [int(n) for n in self._neighbors(tile) if int(n) in owned]
        if len(anchors) <= 1:
            return False

        remaining = owned - {tile}
        needed = set(anchors[1:])
        visited = {anchors[0]}
        queue = [anchors[0]]
        while queue and needed:
            cur = queue.pop()
            for nb in self._neighbors(cur):
                nb = int(nb)
                if nb in remaining and nb not in visited:
                    visited.add(nb)
                    needed.discard(nb)
                    if not needed:
                        return False
                    queue.append(nb)
        return bool(needed)

    def _addition_creates_island(self, n_id: int, leaving: int, arriving: int) -> bool:
        """Would ``arriving`` land in ``n_id`` with no neighbour left in it?

        Only swaps can do this: ``leaving`` may have been the arriving tile's only
        anchor, and it departs in the same move. A donation always travels along a
        boundary edge, so the recipient is adjacent by construction.
        """
        owned = self.n_to_tiles.get(n_id)
        if not owned:
            return True
        for nb in self._neighbors(arriving):
            nb = int(nb)
            if nb != leaving and nb in owned:
                return False
        return True

    def _move_breaks_contiguity(self, move: tuple) -> bool:
        if not self.cfg.enforce_contiguity:
            return False
        if move[0] == "donation":
            _, tile, n_donor, _n_recip, _s_d, _s_r = move
            return self._removal_disconnects(n_donor, tile)
        _, ti, tj, n_i, n_j, _s_i, _s_j = move
        return (
            self._removal_disconnects(n_i, ti)
            or self._removal_disconnects(n_j, tj)
            or self._addition_creates_island(n_i, ti, tj)
            or self._addition_creates_island(n_j, tj, ti)
        )

    def _touch_boundary(self, i: int) -> None:
        tn = self.tile_n_ids
        if self._tile_is_mixed(i):
            self.boundary.add((i, i))
        else:
            self.boundary.discard((i, i))
        for j in self._neighbors(i):
            j = int(j)
            edge = (i, j) if i < j else (j, i)
            if tn[i] != tn[j]:
                self.boundary.add(edge)
            else:
                self.boundary.discard(edge)

    # ------------------------------------------------------------------
    # Consolidation pass (SPEC_TILED section 5.1)
    # ------------------------------------------------------------------

    def consolidate_mixed_tiles(self) -> int:
        """Winner-takes-all: force every mixed tile into one neighborhood."""
        ct = self.ct
        mixed = 0
        for i in range(self.n_tiles):
            pidx = self.tile_parcels[i]
            n_here = self.parcel_n_ids[pidx]
            uniq = np.unique(n_here)
            if len(uniq) < 2:
                continue
            mixed += 1

            best_n, best_score = None, -np.inf
            for cand in uniq:
                moving = pidx[n_here != cand]
                if len(moving) == 0:
                    continue
                score = mi.score_with_delta(
                    ct, int(cand), None, ct.delta_for(moving),
                    int(ct.totals[cand]) + len(moving),
                )
                if score > best_score:
                    best_score, best_n = score, int(cand)

            if best_n is None:
                continue

            for n_from in uniq:
                if n_from == best_n:
                    continue
                moving = pidx[n_here == n_from]
                if len(moving) == 0:
                    continue
                delta = ct.delta_for(moving)
                mi.apply_delta(ct, int(n_from), delta, None)
                mi.apply_delta(ct, int(best_n), None, delta)
                ct.totals[n_from] -= len(moving)
                ct.totals[best_n] += len(moving)

            self.parcel_n_ids[pidx] = best_n
            self.tile_n_ids[i] = best_n

        self.scores = mi.all_scores(ct)
        self._rebuild_boundary()
        self._rebuild_n_to_tiles()
        self._boundary_list = sorted(self.boundary)
        self.progress(f"Consolidation pass: {mixed:,} mixed tiles resolved")
        return mixed

    # ------------------------------------------------------------------
    # Move evaluation
    # ------------------------------------------------------------------

    def _eval_donation(
        self, tile: int, n_donor: int, n_recip: int
    ) -> Optional[tuple]:
        if n_donor == n_recip:
            return None
        ct = self.ct
        delta = self.tile_delta_list[tile]
        num_p = len(self.tile_parcels[tile])

        s_donor = mi.score_with_delta(
            ct, n_donor, delta, None, int(ct.totals[n_donor]) - num_p
        )
        s_recip = mi.score_with_delta(
            ct, n_recip, None, delta, int(ct.totals[n_recip]) + num_p
        )

        gain = (s_donor - self.scores[n_donor]) + (s_recip - self.scores[n_recip])
        return gain, ("donation", tile, n_donor, n_recip, s_donor, s_recip)

    def _eval_swap(self, ti: int, tj: int) -> Optional[tuple]:
        ct = self.ct
        n_i, n_j = int(self.tile_n_ids[ti]), int(self.tile_n_ids[tj])
        if n_i == n_j:
            return None
        d_i, d_j = self.tile_delta_list[ti], self.tile_delta_list[tj]
        num_i, num_j = len(self.tile_parcels[ti]), len(self.tile_parcels[tj])

        s_i = mi.score_with_delta(
            ct, n_i, d_i, d_j, int(ct.totals[n_i]) - num_i + num_j
        )
        s_j = mi.score_with_delta(
            ct, n_j, d_j, d_i, int(ct.totals[n_j]) - num_j + num_i
        )

        gain = (s_i - self.scores[n_i]) + (s_j - self.scores[n_j])
        return gain, ("swap", ti, tj, n_i, n_j, s_i, s_j)

    def _best_move(self, edges: Sequence[Tuple[int, int]]) -> Optional[tuple]:
        """Best *legal* move in this batch.

        The contiguity gate is applied after ranking rather than to every
        candidate. Filtering first and then taking the best is identical to
        taking the best that passes, but the second form only gates until it
        finds a legal move -- about 1.05 checks per iteration at the measured 5%
        block rate, instead of one per candidate (~1,500, which cost 18 ms and
        halved throughput).
        """
        candidates: List[tuple] = []
        for ti, tj in edges:
            if (ti, tj) not in self.boundary:
                continue  # stale entry from a cached boundary list

            if ti == tj:
                # Mixed tile: consider every neighborhood it or its neighbours hold.
                n_primary = int(self.tile_n_ids[ti])
                recipients = set(
                    int(v) for v in np.unique(self.parcel_n_ids[self.tile_parcels[ti]])
                )
                for j in self._neighbors(ti):
                    recipients.add(int(self.tile_n_ids[int(j)]))
                for n_recip in recipients:
                    res = self._eval_donation(ti, n_primary, n_recip)
                    if res:
                        candidates.append(res)
                continue

            n_i, n_j = int(self.tile_n_ids[ti]), int(self.tile_n_ids[tj])
            for donor, n_donor, n_recip in ((ti, n_i, n_j), (tj, n_j, n_i)):
                res = self._eval_donation(donor, n_donor, n_recip)
                if res:
                    candidates.append(res)
            res = self._eval_swap(ti, tj)
            if res:
                candidates.append(res)

        if not candidates:
            return None
        # Sort on gain alone (not the whole tuple) so ties keep insertion order
        # and the choice stays reproducible.
        candidates.sort(key=lambda c: -c[0])
        if not self.cfg.enforce_contiguity:
            return candidates[0]
        for gain, move in candidates:
            if not self._move_breaks_contiguity(move):
                return gain, move
        self.blocked_batches += 1
        return None

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def _commit(self, move: tuple) -> None:
        ct = self.ct
        if move[0] == "donation":
            _, tile, n_donor, n_recip, s_d, s_r = move
            pidx = self.tile_parcels[tile]
            delta = self.tile_delta_list[tile]
            num_p = len(pidx)

            self.parcel_n_ids[pidx] = n_recip
            self.last_change_iter[pidx] = self.iteration
            self.touch_events += len(pidx)
            self.tile_n_ids[tile] = n_recip
            owned = self.n_to_tiles
            owned.setdefault(n_donor, set()).discard(tile)
            owned.setdefault(n_recip, set()).add(tile)
            mi.apply_delta(ct, n_donor, delta, None)
            mi.apply_delta(ct, n_recip, None, delta)
            ct.totals[n_donor] -= num_p
            ct.totals[n_recip] += num_p
            self.scores[n_donor], self.scores[n_recip] = s_d, s_r
            self._touch_boundary(tile)
        else:
            _, ti, tj, n_i, n_j, s_i, s_j = move
            p_i, p_j = self.tile_parcels[ti], self.tile_parcels[tj]
            d_i, d_j = self.tile_delta_list[ti], self.tile_delta_list[tj]
            num_i, num_j = len(p_i), len(p_j)

            self.parcel_n_ids[p_i] = n_j
            self.parcel_n_ids[p_j] = n_i
            self.last_change_iter[p_i] = self.iteration
            self.last_change_iter[p_j] = self.iteration
            self.touch_events += len(p_i) + len(p_j)
            self.tile_n_ids[ti], self.tile_n_ids[tj] = n_j, n_i
            owned = self.n_to_tiles
            owned.setdefault(n_i, set()).discard(ti)
            owned.setdefault(n_j, set()).add(ti)
            owned.setdefault(n_j, set()).discard(tj)
            owned.setdefault(n_i, set()).add(tj)
            mi.apply_delta(ct, n_i, d_i, d_j)
            mi.apply_delta(ct, n_j, d_j, d_i)
            ct.totals[n_i] = ct.totals[n_i] - num_i + num_j
            ct.totals[n_j] = ct.totals[n_j] - num_j + num_i
            self.scores[n_i], self.scores[n_j] = s_i, s_j
            self._touch_boundary(ti)
            self._touch_boundary(tj)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def recent_change_fraction(self) -> float:
        """Share of this optimizer's parcels relabelled in the last window.

        Diagnostic only -- see progress_ratio() for the convergence test. This
        number is not comparable across dataset sizes.
        """
        window = int(self.cfg.assignment_stability_iters)
        owned = self._owned_parcels
        if window <= 0 or not len(owned):
            return 1.0
        age = self.iteration - self.last_change_iter[owned]
        return float(np.count_nonzero(age < window)) / len(owned)

    def _events_in_window(self, window: int) -> Optional[int]:
        """Relabel events in the last ``window`` iterations, or None if unknown."""
        cutoff = self.iteration - window
        base = None
        for it, cum in self._event_marks:
            if it <= cutoff:
                base = cum
        if base is None:
            return None  # not a full window of history yet
        return self.touch_events - base

    def progress_ratio(self, window: int) -> Optional[float]:
        """Distinct parcels changed / relabel events, over the window.

        ~1.0 while the optimizer keeps reaching fresh ground; toward 0 once it is
        just shuffling the same tiles between the same neighborhoods. Both terms
        scale together with dataset size, so the ratio does not.
        """
        events = self._events_in_window(window)
        if events is None:
            return None
        if events <= 0:
            return 0.0  # nothing moved at all
        owned = self._owned_parcels
        if not len(owned):
            return 0.0
        age = self.iteration - self.last_change_iter[owned]
        distinct = int(np.count_nonzero(age < window))
        return distinct / events

    def _assignment_converged(self) -> bool:
        """True once the map has stopped changing, not merely stopped improving.

        Annealing can keep accepting marginal moves forever without the
        assignment actually moving, which is why a rejected-move count is a weak
        signal. This asks the direct question, and because it is a *fraction of
        my own parcels* it means the same thing for a 30-tile island as for a
        15,000-tile mainland.
        """
        cfg = self.cfg
        window = int(cfg.assignment_stability_iters)
        if not window:
            return False

        self._event_marks.append((self.iteration, self.touch_events))
        while len(self._event_marks) > 2 and \
                self.iteration - self._event_marks[0][0] > 2 * window:
            self._event_marks.pop(0)

        # Needs a full window of history first, or the initial all-zero
        # last_change_iter would read as "nothing changed recently".
        if self.iteration < window:
            self._stable_streak = 0
            return False

        ratio = self.progress_ratio(window)
        if ratio is None or ratio >= cfg.assignment_progress_ratio:
            self._stable_streak = 0
            return False

        self._stable_streak += 1
        if self._stable_streak < cfg.assignment_stability_streak:
            return False
        self.progress(
            f"Converged: only {100 * ratio:.1f}% of recent relabels reached "
            f"parcels not already touched, for "
            f"{cfg.assignment_stability_streak} consecutive {window:,}-iteration "
            "windows -- the map is churning, not improving."
        )
        return True

    def owned_neighborhoods(self) -> np.ndarray:
        """Neighborhood ids this optimizer is responsible for.

        For a serial run that's all of them. For one worker in a parallel run it
        is only the ids present in its own tiles -- the count tables deliberately
        cover every neighborhood so bin codes and global statistics match the
        serial path exactly, but the worker must not report on rows it never
        touches.
        """
        return np.unique(self.parcel_n_ids[self._owned_parcels])

    def weighted_score(self) -> Tuple[float, float]:
        """(sum of score x parcels, sum of parcels) over owned neighborhoods."""
        owned = self.owned_neighborhoods()
        if not len(owned):
            return 0.0, 0.0
        sizes = self.ct.totals[owned].astype(np.float64)
        keep = sizes > 0
        if not keep.any():
            return 0.0, 0.0
        return (
            float(np.sum(self.scores[owned][keep] * sizes[keep])),
            float(np.sum(sizes[keep])),
        )

    def stats(self, message: str = "", started: float = 0.0) -> OptimizerStats:
        active = int((self.ct.totals > 0).sum())
        num, den = self.weighted_score()
        return OptimizerStats(
            iteration=self.iteration,
            temperature=self.temperature,
            mean_score=float(np.mean(self.scores)) if len(self.scores) else 0.0,
            boundary_size=len(self.boundary),
            accepted=self.accepted,
            rejected=self.rejected,
            active_neighborhoods=active,
            elapsed_s=(time.time() - started) if started else 0.0,
            iterations_this_run=self.iteration - getattr(self, "_start_iteration", 0),
            weighted_score=(num / den) if den else 0.0,
            score_num=num,
            score_den=den,
            message=message,
        )

    def run(
        self,
        store: Optional[CheckpointStore] = None,
        stats_cb: Optional[Callable[[OptimizerStats], None]] = None,
        snapshot_cb: Optional[Callable[[np.ndarray, OptimizerStats], None]] = None,
    ) -> np.ndarray:
        cfg = self.cfg
        started = time.time()
        last_snapshot = -1
        # Rates are reported over this call's work, not since iteration zero,
        # so a resumed run doesn't claim an absurd it/s.
        self._start_iteration = self.iteration
        self._paused_saved = False

        while self.iteration < cfg.max_iterations:
            if self.stop_event.is_set():
                break
            if self.pause_event.is_set():
                # Checkpoint here, from the worker thread, at a point where the
                # state is consistent. Doing it from the UI thread the moment
                # Pause is pressed can catch the loop mid-commit and produce a
                # checkpoint whose assignments, counters and RNG state disagree.
                if store and not self._paused_saved:
                    self.save_checkpoint(store)
                    self._paused_saved = True
                time.sleep(0.1)
                continue
            self._paused_saved = False
            if not self.boundary:
                self.progress("Boundary set is empty; nothing left to trade.")
                break

            # Refresh the sampling pool periodically instead of after every
            # accepted move (the notebook rebuilt it each time, which is O(|B|)).
            # Sorted, not insertion-ordered: a set rebuilt on resume iterates in
            # a different order than one grown incrementally, which would make
            # rng.choice pick different edges and silently fork the trajectory.
            # The refresh schedule keys off the iteration number alone (never
            # off "iterations since the last rebuild"), so a resumed run hits
            # the same rebuild points as an uninterrupted one.
            due = self.iteration % self.POOL_REFRESH == 0
            if (due and self._boundary_stamp != self.iteration) or not self._boundary_list:
                self._boundary_list = sorted(self.boundary)
                self._boundary_stamp = self.iteration

            pool = self._boundary_list
            n_pool = len(pool)
            if n_pool == 0:
                self._boundary_list = sorted(self.boundary)
                if not self._boundary_list:
                    break
                continue

            size = max(cfg.min_batch, n_pool // max(cfg.batch_divisor, 1))
            size = int(min(size, cfg.max_batch, n_pool))
            self._last_batch = size
            picks = self.rng.choice(n_pool, size=size, replace=False)
            edges = [pool[int(k)] for k in picks]

            result = self._best_move(edges)
            if result is None:
                self.iteration += 1
                self.temperature *= cfg.cooling_rate
                continue

            gain, move = result
            accepted = gain > 0
            if not accepted and self.temperature > 0.001:
                accepted = bool(self.rng.random() < np.exp(min(gain / self.temperature, 0.0)))

            if accepted:
                self._commit(move)
                self.accepted += 1
                if self.temperature < 0.001:
                    self.stability_counter = 0
            else:
                self.rejected += 1
                if self.temperature < 0.001:
                    self.stability_counter += 1
                    if self.stability_counter >= cfg.max_stability:
                        self.progress(
                            f"Converged: no accepted moves in "
                            f"{cfg.max_stability:,} iterations at T~0."
                        )
                        break

            self.iteration += 1
            self.temperature *= cfg.cooling_rate

            if self.iteration % 100 == 0:
                if stats_cb:
                    stats_cb(self.stats(started=started))
                if self._assignment_converged():
                    break

            if snapshot_cb and self.iteration % max(cfg.refresh_every, 1) == 0:
                if self.iteration != last_snapshot:
                    last_snapshot = self.iteration
                    snapshot_cb(self.tile_n_ids.copy(), self.stats(started=started))

            if store and self.iteration % max(cfg.checkpoint_every, 1) == 0:
                self.save_checkpoint(store)

        if store:
            self.save_checkpoint(store, final=True)
        if snapshot_cb:
            snapshot_cb(self.tile_n_ids.copy(), self.stats("finished", started))
        return self.parcel_n_ids

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, store: CheckpointStore, final: bool = False) -> str:
        cp = Checkpoint(
            iteration=self.iteration,
            temperature=self.temperature,
            stability_counter=self.stability_counter,
            accepted=self.accepted,
            rejected=self.rejected,
            mean_score=float(np.mean(self.scores)) if len(self.scores) else 0.0,
            n_neighborhoods=self.n_neighborhoods,
            parcel_n_ids=self.parcel_n_ids,
            rng_state=self.rng.bit_generator.state,
            last_change_iter=self.last_change_iter,
            assignment_stable_streak=self._stable_streak,
        )
        path = store.save(cp, final=final)
        self.progress(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, cp: Checkpoint) -> None:
        if len(cp.parcel_n_ids) != len(self.parcel_n_ids):
            raise ValueError(
                f"Checkpoint has {len(cp.parcel_n_ids):,} parcels but this run has "
                f"{len(self.parcel_n_ids):,}. The parcel file or tileset changed."
            )
        self.parcel_n_ids = np.asarray(cp.parcel_n_ids, dtype=np.int64).copy()
        self.n_neighborhoods = max(
            int(self.parcel_n_ids.max()) + 1, int(cp.n_neighborhoods)
        )
        self.ct = mi.build_count_tables(
            self.parcels,
            self.cfg.weights,
            self.n_neighborhoods,
            self.parcel_n_ids,
            exact_mi=self.cfg.exact_mi,
        )
        self._build_deltas()
        self.scores = mi.all_scores(self.ct)
        self._recompute_tile_ids()
        self._rebuild_boundary()
        self._rebuild_n_to_tiles()
        self._boundary_list = sorted(self.boundary)
        self._boundary_stamp = -1
        # Never zeros: see Checkpoint.restore_last_change. A zeroed array would
        # read as "nothing changed recently" and exit on the first check.
        self.last_change_iter = cp.restore_last_change(len(self.parcel_n_ids))
        self._stable_streak = int(cp.assignment_stable_streak)

        self.iteration = int(cp.iteration)
        self.temperature = float(cp.temperature)
        self.stability_counter = int(cp.stability_counter)
        self.accepted = int(cp.accepted)
        self.rejected = int(cp.rejected)
        if cp.rng_state:
            try:
                self.rng.bit_generator.state = cp.rng_state
            except Exception:
                self.progress("Could not restore RNG state; continuing with a fresh one.")
        self.progress(
            f"Resumed from iteration {self.iteration:,} (T={self.temperature:.5f})"
        )

    # ------------------------------------------------------------------

    def result_frame(self, parcels: Optional[gpd.GeoDataFrame] = None) -> gpd.GeoDataFrame:
        base = parcels if parcels is not None else self.parcels
        out = base.copy()
        out["neighborhood_id"] = self.parcel_n_ids
        return out
