"""Checkpoint storage.

The notebook wrote the whole parcel GeoDataFrame to parquet every 1000
iterations, which is tens of megabytes a pop and dominates the loop once the
dataset is large. A checkpoint only needs the assignment vector plus the
annealing state -- count tables, boundary set and scores are all derivable from
it -- so we store a small ``.npz`` (assignments) next to a ``.json`` (state).

A run directory looks like::

    runs/fort_myers_fl_20260727_141530/
        run_config.json
        tiles.parquet
        parcels_prepared.parquet
        checkpoints/
            checkpoint_000000.npz + .json
            checkpoint_005000.npz + .json
            checkpoint_final.npz  + .json
        optimized_neighborhoods_tiled.parquet
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

CHECKPOINT_RE = re.compile(r"checkpoint_(\d+|final)\.json$")


@dataclass
class Checkpoint:
    iteration: int
    temperature: float
    stability_counter: int
    accepted: int
    rejected: int
    mean_score: float
    n_neighborhoods: int
    parcel_n_ids: np.ndarray
    rng_state: Optional[dict] = None
    extra: Dict[str, object] = field(default_factory=dict)

    def meta(self) -> dict:
        return {
            "iteration": int(self.iteration),
            "temperature": float(self.temperature),
            "stability_counter": int(self.stability_counter),
            "accepted": int(self.accepted),
            "rejected": int(self.rejected),
            "mean_score": float(self.mean_score),
            "n_neighborhoods": int(self.n_neighborhoods),
            "n_parcels": int(len(self.parcel_n_ids)),
            "rng_state": _jsonify(self.rng_state) if self.rng_state else None,
            "extra": self.extra,
        }


def _jsonify(obj):
    """numpy scalars/arrays inside the RNG state dict -> plain Python."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _unjsonify(obj):
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            return np.array(obj["__ndarray__"], dtype=obj.get("dtype", None))
        return {k: _unjsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unjsonify(v) for v in obj]
    return obj


class CheckpointStore:
    """Reads and writes checkpoints inside one run directory."""

    def __init__(self, run_dir: Path, keep: int = 20) -> None:
        self.run_dir = Path(run_dir)
        self.dir = self.run_dir / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep = int(keep)

    # ------------------------------------------------------------------
    def save(self, cp: Checkpoint, final: bool = False) -> str:
        tag = "final" if final else f"{cp.iteration:06d}"
        npz = self.dir / f"checkpoint_{tag}.npz"
        meta = self.dir / f"checkpoint_{tag}.json"
        np.savez_compressed(
            npz, parcel_n_ids=np.asarray(cp.parcel_n_ids, dtype=np.int32)
        )
        meta.write_text(json.dumps(cp.meta(), indent=2), encoding="utf-8")
        if not final:
            self._prune()
        return str(npz)

    def _prune(self) -> None:
        if self.keep <= 0:
            return
        entries = [p for p in self.dir.glob("checkpoint_*.json") if "final" not in p.name]
        if len(entries) <= self.keep:
            return
        entries.sort(key=lambda p: _iteration_of(p))
        for p in entries[: len(entries) - self.keep]:
            p.with_suffix(".npz").unlink(missing_ok=True)
            p.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    def list(self) -> List[dict]:
        """Available checkpoints, newest last."""
        out = []
        for meta_path in self.dir.glob("checkpoint_*.json"):
            npz = meta_path.with_suffix(".npz")
            if not npz.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            meta["path"] = str(meta_path)
            meta["label"] = _label(meta_path, meta)
            out.append(meta)
        out.sort(key=lambda m: (m.get("iteration", 0), "final" in m["path"]))
        return out

    def load(self, meta_path: Path | str) -> Checkpoint:
        meta_path = Path(meta_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(meta_path.with_suffix(".npz")) as data:
            ids = data["parcel_n_ids"].astype(np.int64)
        return Checkpoint(
            iteration=int(meta["iteration"]),
            temperature=float(meta["temperature"]),
            stability_counter=int(meta.get("stability_counter", 0)),
            accepted=int(meta.get("accepted", 0)),
            rejected=int(meta.get("rejected", 0)),
            mean_score=float(meta.get("mean_score", 0.0)),
            n_neighborhoods=int(meta.get("n_neighborhoods", int(ids.max()) + 1)),
            parcel_n_ids=ids,
            rng_state=_unjsonify(meta.get("rng_state")) or None,
            extra=meta.get("extra", {}),
        )

    def latest(self) -> Optional[Checkpoint]:
        entries = self.list()
        if not entries:
            return None
        return self.load(entries[-1]["path"])


def _iteration_of(path: Path) -> int:
    m = CHECKPOINT_RE.search(path.name)
    if not m:
        return -1
    return 10**12 if m.group(1) == "final" else int(m.group(1))


def _label(path: Path, meta: dict) -> str:
    it = meta.get("iteration", 0)
    score = meta.get("mean_score", 0.0)
    kind = "final" if "final" in path.name else f"iter {it:,}"
    return f"{kind} — score {score:.6f}, T={meta.get('temperature', 0):.5f}"


def find_runs(root: Path) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and (p / "run_config.json").exists()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs
