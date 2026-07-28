"""Annealing several severed components at once, in separate processes.

Why processes and not threads: the inner loop is NumPy on ~150-element arrays
plus Python set/dict work on the boundary set. That is squarely GIL-bound, so
threads would buy nothing. Processes suit the problem better anyway -- after
``partition.split_neighborhoods`` there is no shared mutable state at all:

* ``global_counts`` and ``total_g`` are read-only, so every worker computes
  bit-identical move gains to a serial run;
* each worker owns a disjoint set of tiles, hence disjoint parcels;
* each worker owns a disjoint set of neighborhood ids, hence disjoint
  count-table rows.

Workers hold the *full* parcel arrays (a few MB) rather than a slice. That keeps
bin codes, global counts and neighborhood numbering identical to the serial path
by construction -- slicing the frame first would make ``build_count_tables``
re-derive subset-dependent bin codes and silently change the objective.

The spawn start method is used explicitly on every platform: forking a process
that already has Qt and matplotlib loaded is unsafe, and spawn is also what
Windows would do anyway, so behaviour matches everywhere.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as _queue
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


# ==========================================================================
# Compact, picklable description of one worker's slice
# ==========================================================================


@dataclass
class GroupTask:
    group_id: int
    components: List[int]
    tile_ids: np.ndarray          # (n_tiles,) tile ids this worker owns
    tp_indptr: np.ndarray         # CSR: tile -> parcel positional indices
    tp_indices: np.ndarray
    adj_indptr: np.ndarray        # CSR: tile -> neighbouring tile *local* index
    adj_indices: np.ndarray

    def n_tiles(self) -> int:
        return len(self.tile_ids)

    def n_parcels(self) -> int:
        return len(self.tp_indices)

    def to_dicts(self) -> Tuple[Dict[int, np.ndarray], Dict[int, set]]:
        """Rebuild the dict forms TiledOptimizer expects."""
        tile_to_parcels = {
            int(self.tile_ids[i]): self.tp_indices[
                self.tp_indptr[i] : self.tp_indptr[i + 1]
            ]
            for i in range(len(self.tile_ids))
        }
        adjacency = {
            int(self.tile_ids[i]): {
                int(self.tile_ids[j])
                for j in self.adj_indices[
                    self.adj_indptr[i] : self.adj_indptr[i + 1]
                ]
            }
            for i in range(len(self.tile_ids))
        }
        return tile_to_parcels, adjacency


def build_group_tasks(
    groups: Sequence[Sequence[int]],
    comps,
    tile_to_parcels: Dict[int, np.ndarray],
    tile_adjacency: Dict[int, set],
) -> List[GroupTask]:
    """Turn component groupings into compact CSR payloads."""
    tasks: List[GroupTask] = []
    for gid, components in enumerate(groups):
        wanted = set(int(c) for c in components)
        mask = np.isin(comps.tile_label, list(wanted))
        tile_ids = comps.tile_ids[mask]
        local = {int(t): i for i, t in enumerate(tile_ids)}

        tp_indptr = np.zeros(len(tile_ids) + 1, dtype=np.int64)
        tp_chunks: List[np.ndarray] = []
        adj_indptr = np.zeros(len(tile_ids) + 1, dtype=np.int64)
        adj_chunks: List[np.ndarray] = []

        for i, t in enumerate(tile_ids):
            parcels = np.asarray(tile_to_parcels[int(t)], dtype=np.int64)
            tp_chunks.append(parcels)
            tp_indptr[i + 1] = tp_indptr[i] + len(parcels)

            # Edges leaving the group cannot exist (components are severed), but
            # filter defensively so a mislabelled tile can't leak across.
            neigh = [
                local[int(n)] for n in tile_adjacency.get(int(t), ())
                if int(n) in local
            ]
            adj_chunks.append(np.asarray(neigh, dtype=np.int64))
            adj_indptr[i + 1] = adj_indptr[i] + len(neigh)

        tasks.append(
            GroupTask(
                group_id=gid,
                components=[int(c) for c in components],
                tile_ids=tile_ids.astype(np.int64),
                tp_indptr=tp_indptr,
                tp_indices=(
                    np.concatenate(tp_chunks) if tp_chunks
                    else np.zeros(0, dtype=np.int64)
                ),
                adj_indptr=adj_indptr,
                adj_indices=(
                    np.concatenate(adj_chunks) if adj_chunks
                    else np.zeros(0, dtype=np.int64)
                ),
            )
        )
    return tasks


# ==========================================================================
# Worker
# ==========================================================================


def _worker(
    task: GroupTask,
    cfg_dict: dict,
    parcels_path: str,
    parcel_n_ids: np.ndarray,
    n_neighborhoods: int,
    resume: Optional[dict],
    out_q,
    stop_event,
    pause_event,
) -> None:
    """Anneal one group of components. Runs in a spawned child process."""
    # Spawned children start with a bare sys.path; the flat module layout means
    # the project directory has to be importable explicitly.
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)

    try:
        import pandas as pd

        import engine
        from config import RunConfig

        cfg = RunConfig(**cfg_dict)
        # Only the scored columns are needed; geometry stays in the parent.
        frame = pd.read_parquet(parcels_path, columns=list(cfg.weights))

        tile_to_parcels, adjacency = task.to_dicts()
        opt = engine.TiledOptimizer(
            frame, tile_to_parcels, adjacency, cfg,
            neighborhood_ids=np.asarray(parcel_n_ids, dtype=np.int64),
            progress=lambda m: out_q.put(
                {"type": "log", "group": task.group_id, "text": m}
            ),
        )
        # mp.Event quacks like threading.Event for is_set(), which is all the
        # optimizer ever asks of these.
        opt.stop_event = stop_event
        opt.pause_event = pause_event

        if resume:
            opt.iteration = int(resume.get("iteration", 0))
            opt.temperature = float(resume.get("temperature", cfg.initial_temp))
            opt.stability_counter = int(resume.get("stability_counter", 0))
            opt.accepted = int(resume.get("accepted", 0))
            opt.rejected = int(resume.get("rejected", 0))
            opt._stable_streak = int(resume.get("assignment_stable_streak", 0))
            # Group state is scalars only, so the per-parcel change history isn't
            # in it. Leaving the zeros from __init__ would make every parcel look
            # `iteration` steps stale and trip instant false convergence, so fill
            # with "everything just changed" instead -- same safe default as
            # Checkpoint.restore_last_change.
            opt.last_change_iter[:] = opt.iteration
        else:
            opt.consolidate_mixed_tiles()

        def on_snapshot(tile_n_ids, stats):
            out_q.put({
                "type": "snapshot",
                "group": task.group_id,
                "tile_ids": task.tile_ids,
                "tile_n_ids": np.asarray(tile_n_ids, dtype=np.int32),
                "iteration": stats.iteration,
                "temperature": stats.temperature,
                "accepted": stats.accepted,
                "rejected": stats.rejected,
                "boundary": stats.boundary_size,
                "rate": stats.rate,
                # Sent as numerator/denominator so the parent can form the exact
                # global parcel-weighted score, not an average of averages.
                "score_num": stats.score_num,
                "score_den": stats.score_den,
            })

        opt.run(store=None, stats_cb=None, snapshot_cb=on_snapshot)

        owned = task.tp_indices
        num, den = opt.weighted_score()
        out_q.put({
            "type": "result",
            "group": task.group_id,
            "parcel_idx": owned,
            "parcel_n_ids": opt.parcel_n_ids[owned].astype(np.int32),
            "tile_ids": task.tile_ids,
            "tile_n_ids": opt.tile_n_ids.astype(np.int32),
            "iteration": opt.iteration,
            "temperature": opt.temperature,
            "stability_counter": opt.stability_counter,
            "accepted": opt.accepted,
            "rejected": opt.rejected,
            "boundary": len(opt.boundary),
            "score_num": num,
            "score_den": den,
        })
    except Exception:  # noqa: BLE001 - reported to the parent
        out_q.put({
            "type": "error",
            "group": getattr(task, "group_id", -1),
            "traceback": traceback.format_exc(),
        })


# ==========================================================================
# Parent-side coordinator
# ==========================================================================


@dataclass
class ParallelStats:
    iteration: int = 0            # max across groups
    temperature: float = 0.0      # max across groups
    weighted_score: float = 0.0   # parcel-weighted, exact across workers
    mean_score: float = 0.0       # kept for display compatibility
    boundary_size: int = 0        # summed
    accepted: int = 0             # summed
    rejected: int = 0             # summed
    active_neighborhoods: int = 0
    elapsed_s: float = 0.0
    iterations_this_run: int = 0
    groups_running: int = 0
    message: str = ""

    @property
    def rate(self) -> float:
        return self.iterations_this_run / self.elapsed_s if self.elapsed_s else 0.0


class ParallelRunner:
    """Runs one TiledOptimizer per component group in its own process."""

    def __init__(
        self,
        cfg,
        parcels_path: str,
        tasks: Sequence[GroupTask],
        parcel_n_ids: np.ndarray,
        n_neighborhoods: int,
        progress: Progress = _noop,
    ) -> None:
        self.cfg = cfg
        self.parcels_path = str(parcels_path)
        self.tasks = list(tasks)
        self.parcel_n_ids = np.asarray(parcel_n_ids, dtype=np.int64).copy()
        self.n_neighborhoods = int(n_neighborhoods)
        self.progress = progress

        self.ctx = mp.get_context("spawn")
        self.stop_event = self.ctx.Event()
        self.pause_event = self.ctx.Event()

        # Merged tile view, for the live map.
        all_tiles = np.concatenate([t.tile_ids for t in self.tasks]) if self.tasks \
            else np.zeros(0, dtype=np.int64)
        self.tile_ids = np.sort(all_tiles)
        self._tile_pos = {int(t): i for i, t in enumerate(self.tile_ids)}
        self.tile_n_ids = np.zeros(len(self.tile_ids), dtype=np.int64)
        self._group_state: Dict[int, dict] = {}
        # Kept so a force-quit can terminate them; see kill_workers().
        self._procs: List = []

    # ------------------------------------------------------------------

    def _merge_tiles(self, tile_ids: np.ndarray, values: np.ndarray) -> None:
        idx = np.fromiter(
            (self._tile_pos[int(t)] for t in tile_ids),
            dtype=np.int64, count=len(tile_ids),
        )
        self.tile_n_ids[idx] = values

    # ------------------------------------------------------------------

    def run(
        self,
        resume: Optional[Dict[int, dict]] = None,
        stats_cb: Optional[Callable[[ParallelStats], None]] = None,
        snapshot_cb: Optional[Callable[[np.ndarray, ParallelStats], None]] = None,
        checkpoint_cb: Optional[Callable[[np.ndarray, dict], None]] = None,
    ) -> np.ndarray:
        """Anneal every group; returns the merged parcel assignment vector."""
        if not self.tasks:
            return self.parcel_n_ids

        started = time.time()
        out_q = self.ctx.Queue()
        cfg_dict = asdict(self.cfg)
        base_iteration = 0

        procs = []
        for task in self.tasks:
            p = self.ctx.Process(
                target=_worker,
                args=(
                    task, cfg_dict, self.parcels_path, self.parcel_n_ids,
                    self.n_neighborhoods,
                    (resume or {}).get(task.group_id),
                    out_q, self.stop_event, self.pause_event,
                ),
                daemon=False,
            )
            p.start()
            procs.append(p)
        self._procs = procs

        self.progress(
            f"Started {len(procs)} worker process(es) over "
            f"{sum(len(t.components) for t in self.tasks):,} components"
        )

        live = {t.group_id for t in self.tasks}
        latest: Dict[int, dict] = {}
        errors: List[str] = []
        last_checkpoint = time.time()

        try:
            while live:
                try:
                    msg = out_q.get(timeout=0.25)
                except _queue.Empty:
                    # A worker dying without reporting would otherwise hang us.
                    for p, task in zip(procs, self.tasks):
                        if task.group_id in live and not p.is_alive() \
                                and task.group_id not in latest:
                            live.discard(task.group_id)
                            errors.append(
                                f"worker {task.group_id} exited with code "
                                f"{p.exitcode} without reporting"
                            )
                    continue

                kind = msg.get("type")
                if kind == "log":
                    self.progress(f"[w{msg['group']}] {msg['text']}")

                elif kind == "error":
                    live.discard(msg["group"])
                    errors.append(msg["traceback"])

                elif kind in ("snapshot", "result"):
                    self._merge_tiles(msg["tile_ids"], msg["tile_n_ids"])
                    self._group_state[msg["group"]] = {
                        "iteration": int(msg.get("iteration", 0)),
                        "temperature": float(msg.get("temperature", 0.0)),
                        "stability_counter": int(msg.get("stability_counter", 0)),
                        "accepted": int(msg.get("accepted", 0)),
                        "rejected": int(msg.get("rejected", 0)),
                        "boundary": int(msg.get("boundary", 0)),
                        "score_num": float(msg.get("score_num", 0.0)),
                        "score_den": float(msg.get("score_den", 0.0)),
                    }
                    latest[msg["group"]] = self._group_state[msg["group"]]

                    if kind == "result":
                        self.parcel_n_ids[msg["parcel_idx"]] = msg["parcel_n_ids"]
                        live.discard(msg["group"])
                        self.progress(
                            f"[w{msg['group']}] finished at iteration "
                            f"{msg['iteration']:,}"
                        )

                    stats = self._stats(started, base_iteration, len(live))
                    if kind == "snapshot" and snapshot_cb:
                        snapshot_cb(self.tile_n_ids.copy(), stats)
                    if stats_cb:
                        stats_cb(stats)

                    if checkpoint_cb and (
                        time.time() - last_checkpoint > 30 or kind == "result"
                    ):
                        last_checkpoint = time.time()
                        checkpoint_cb(self.tile_n_ids.copy(), dict(self._group_state))
        finally:
            self.stop_event.set()
            for p in procs:
                p.join(timeout=60)
                if p.is_alive():
                    self.progress(f"Terminating unresponsive worker (pid {p.pid})")
                    p.terminate()
                    p.join(timeout=10)

        if errors:
            raise RuntimeError(
                "One or more annealing workers failed:\n\n" + "\n\n".join(errors)
            )
        return self.parcel_n_ids

    # ------------------------------------------------------------------

    def _stats(
        self, started: float, base_iteration: int, groups_running: int
    ) -> ParallelStats:
        states = list(self._group_state.values()) or [{}]
        iters = [int(s.get("iteration", 0)) for s in states]
        num = sum(float(s.get("score_num", 0.0)) for s in states)
        den = sum(float(s.get("score_den", 0.0)) for s in states)
        return ParallelStats(
            weighted_score=(num / den) if den else 0.0,
            iteration=max(iters) if iters else 0,
            temperature=max(float(s.get("temperature", 0.0)) for s in states),
            boundary_size=sum(int(s.get("boundary", 0)) for s in states),
            accepted=sum(int(s.get("accepted", 0)) for s in states),
            rejected=sum(int(s.get("rejected", 0)) for s in states),
            elapsed_s=time.time() - started,
            # Summed across groups: this is throughput, not a position in one
            # cooling schedule.
            iterations_this_run=sum(iters) - base_iteration,
            groups_running=groups_running,
        )

    # ------------------------------------------------------------------

    def group_state(self) -> Dict[int, dict]:
        return dict(self._group_state)

    def stop(self) -> None:
        self.stop_event.set()

    def kill_workers(self, grace_s: float = 1.0) -> None:
        """Terminate the worker processes outright.

        For force-quit only. Spawned children are not tied to the parent's
        lifetime, so a parent that exits via ``os._exit`` (which skips atexit
        handlers) would otherwise leave them running with no window attached.
        """
        self.stop_event.set()
        deadline = time.time() + grace_s
        for p in self._procs:
            remaining = max(0.0, deadline - time.time())
            if remaining:
                p.join(timeout=remaining)
        for p in self._procs:
            if p.is_alive():
                p.terminate()
        for p in self._procs:
            p.join(timeout=2)
            if p.is_alive() and hasattr(p, "kill"):
                p.kill()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self.pause_event.set()
        else:
            self.pause_event.clear()


# ==========================================================================


def default_worker_count(requested: int = 0, cap: int = 8) -> int:
    if requested and requested > 0:
        return int(requested)
    return max(1, min(cap, (os.cpu_count() or 2)))
