"""Tests for src.gpu_lock.file_lock (Linux fcntl).

Uses tmp_path-equivalent + multiprocessing to verify cross-process semantics.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import time
import unittest
from pathlib import Path


def _hold_lock(lock_path: str, hold_seconds: float, ready_evt) -> None:
    os.environ["SR_GPU_LOCK_FILE"] = lock_path
    from src.gpu_lock import cross_process_gpu_lock
    with cross_process_gpu_lock(timeout=5.0):
        ready_evt.set()
        time.sleep(hold_seconds)


class GPULockTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._lock = Path(self._tmp.name) / "gpu.lock"
        os.environ["SR_GPU_LOCK_FILE"] = str(self._lock)

    def tearDown(self):
        os.environ.pop("SR_GPU_LOCK_FILE", None)
        self._tmp.cleanup()

    def test_acquires_and_releases(self):
        from src.gpu_lock import cross_process_gpu_lock
        with cross_process_gpu_lock(timeout=1.0):
            pass
        # second acquisition must succeed (no leaked lock)
        with cross_process_gpu_lock(timeout=1.0):
            pass

    def test_release_after_exception(self):
        from src.gpu_lock import cross_process_gpu_lock
        with self.assertRaises(RuntimeError):
            with cross_process_gpu_lock(timeout=1.0):
                raise RuntimeError("boom")
        # subsequent acquisition still works
        with cross_process_gpu_lock(timeout=1.0):
            pass

    def test_times_out_when_held_by_other_process(self):
        ctx = mp.get_context("spawn")
        ready = ctx.Event()
        proc = ctx.Process(
            target=_hold_lock,
            args=(str(self._lock), 3.0, ready),
        )
        proc.start()
        try:
            self.assertTrue(ready.wait(5.0), "child failed to acquire lock")
            from src.gpu_lock import cross_process_gpu_lock
            t0 = time.monotonic()
            with self.assertRaises(TimeoutError):
                with cross_process_gpu_lock(timeout=0.5, poll=0.05):
                    self.fail("should not have acquired")
            elapsed = time.monotonic() - t0
            # We expect roughly 0.5s timeout; allow generous slack on slow CI.
            self.assertGreaterEqual(elapsed, 0.4)
            self.assertLess(elapsed, 3.0)
        finally:
            proc.join(10.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(5.0)


if __name__ == "__main__":
    unittest.main()
