"""Tests for durable writes and the shutdown escape hatch.

    python tests/test_shutdown.py
    pytest tests/test_shutdown.py

The app offers "Exit Anyway" while tiling or annealing is in flight, which means
a hard process exit can land in the middle of any cache write. These tests pin
the two properties that make that safe: writes are atomic, and a damaged cache
degrades to "not cached" rather than crashing the next run.

The Qt dialog itself isn't covered here (no display in CI); what's covered is
everything the force-quit path depends on.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline  # noqa: E402
from checkpoints import Checkpoint, CheckpointStore  # noqa: E402


# ==========================================================================
# Atomic writes
# ==========================================================================


def test_atomic_write_keeps_the_npz_suffix_last():
    """np.savez_compressed appends '.npz' to any name lacking it.

    A temp name of 'x.npz.part' is therefore written as 'x.npz.part.npz' and the
    rename silently targets a file that was never created. This is the exact bug
    that made the first version of the atomic-write helper break every
    checkpoint.
    """
    tmp = Path(tempfile.mkdtemp())
    try:
        target = tmp / "seeding.npz"
        pipeline._atomic_write(
            lambda p: np.savez_compressed(p, neighborhood_id=np.arange(5)), target
        )
        assert target.exists(), "atomic write produced nothing"
        with np.load(target) as d:
            assert np.array_equal(d["neighborhood_id"], np.arange(5))
        leftovers = [p.name for p in tmp.iterdir() if ".part" in p.name]
        assert not leftovers, f"temp files left behind: {leftovers}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_atomic_write_leaves_no_partial_file_on_failure():
    tmp = Path(tempfile.mkdtemp())
    try:
        target = tmp / "tiles.parquet"

        def explode(p):
            Path(p).write_bytes(b"half a file")
            raise RuntimeError("boom")

        try:
            pipeline._atomic_write(explode, target)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected the writer's exception to propagate")

        assert not target.exists(), "a failed write must not create the target"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_atomic_write_notices_a_writer_that_produced_nothing():
    tmp = Path(tempfile.mkdtemp())
    try:
        try:
            pipeline._atomic_write(lambda p: None, tmp / "x.parquet")
        except OSError:
            pass
        else:
            raise AssertionError("should have complained about the missing temp")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# Damaged caches
# ==========================================================================


def test_truncated_cache_is_rebuilt_not_fatal():
    tmp = Path(tempfile.mkdtemp())
    try:
        good = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs=2237)
        path = tmp / "tiles.parquet"
        good.to_parquet(path)
        # simulate a kill part-way through the write
        raw = path.read_bytes()
        path.write_bytes(raw[: len(raw) // 2])

        logged = []
        out = pipeline._read_cached(
            gpd.read_parquet, path, logged.append, "tileset"
        )
        assert out is None, "damaged cache must read as absent"
        assert any("unreadable" in m for m in logged), logged
        assert not path.exists(), "damaged cache should be removed so it rebuilds"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_intact_cache_is_returned_untouched():
    tmp = Path(tempfile.mkdtemp())
    try:
        good = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs=2237)
        path = tmp / "tiles.parquet"
        good.to_parquet(path)
        out = pipeline._read_cached(gpd.read_parquet, path, print, "tileset")
        assert out is not None and len(out) == 1
        assert path.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# Checkpoints survive a hard exit
# ==========================================================================


def _checkpoint(iteration: int, n: int = 50) -> Checkpoint:
    return Checkpoint(
        iteration=iteration, temperature=0.5, stability_counter=0,
        accepted=1, rejected=0, mean_score=0.1, n_neighborhoods=4,
        parcel_n_ids=np.arange(n) % 4,
        last_change_iter=np.full(n, iteration, dtype=np.int64),
    )


def test_checkpoint_write_is_atomic_and_round_trips():
    tmp = Path(tempfile.mkdtemp())
    try:
        store = CheckpointStore(tmp, keep=0)
        store.save(_checkpoint(100))
        back = store.latest()
        assert back is not None and back.iteration == 100
        assert back.last_change_iter is not None
        leftovers = [p.name for p in store.dir.iterdir() if "_writing_" in p.name]
        assert not leftovers, f"temp files left behind: {leftovers}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_in_progress_checkpoint_files_are_invisible_to_listing():
    """A kill mid-write must not leave a half-file that looks like a checkpoint."""
    tmp = Path(tempfile.mkdtemp())
    try:
        store = CheckpointStore(tmp, keep=5)
        store.save(_checkpoint(100))
        # emulate an interrupted write of the next one
        (store.dir / "_writing_000200.npz").write_bytes(b"truncated")
        (store.dir / "_writing_000200.json").write_text("{ not json")

        entries = store.list()
        assert len(entries) == 1, [e["path"] for e in entries]
        assert entries[0]["iteration"] == 100
        store._prune()  # must not trip over the debris
        assert store.latest().iteration == 100
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_orphaned_npz_without_json_is_ignored():
    """Interruption between the two renames hides the checkpoint, safely."""
    tmp = Path(tempfile.mkdtemp())
    try:
        store = CheckpointStore(tmp, keep=5)
        store.save(_checkpoint(100))
        # npz lands, json never does
        np.savez_compressed(
            store.dir / "checkpoint_000200.npz", parcel_n_ids=np.arange(50) % 4
        )
        entries = store.list()
        assert [e["iteration"] for e in entries] == [100], entries
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_corrupt_checkpoint_json_does_not_break_listing():
    tmp = Path(tempfile.mkdtemp())
    try:
        store = CheckpointStore(tmp, keep=5)
        store.save(_checkpoint(100))
        store.save(_checkpoint(200))
        (store.dir / "checkpoint_000200.json").write_text("{ truncated")
        entries = store.list()
        assert [e["iteration"] for e in entries] == [100], entries
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
