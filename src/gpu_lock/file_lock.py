"""Cross-process GPU lock using POSIX fcntl.flock (Linux).

Lock file path is taken from env SR_GPU_LOCK_FILE (default /tmp/gpu.lock).
Acquisition is non-blocking with a polling timeout, so a stalled holder
does not deadlock the rest of the system. Used by VACE inference (Phase 2)
to avoid GPU contention with the AudioSeparator service on :83 sharing the
same RTX 4070 Ti Super.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import logging
import os
import time
from pathlib import Path
from typing import Iterator

logger = logging.getLogger("gpu_lock")


def _lock_path() -> Path:
    """Resolve the lock-file path at *call* time so tests can monkeypatch env."""
    return Path(os.environ.get("SR_GPU_LOCK_FILE", "/tmp/gpu.lock"))


@contextlib.contextmanager
def cross_process_gpu_lock(timeout: float = 300.0,
                           poll: float = 0.5) -> Iterator[None]:
    """Acquire an exclusive flock on the lock file, or raise TimeoutError.

    Holds the lock for the duration of the with-block. Always releases on
    exit, including via exception. Safe across process boundaries because
    the kernel cleans flock automatically when the holding fd is closed.
    """
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use a+ so the file is created if missing and we don't truncate.
    fp = open(path, "a+")
    deadline = time.monotonic() + timeout
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError as e:
                # EWOULDBLOCK / EAGAIN means another holder; keep polling.
                if e.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"gpu lock timeout after {timeout}s "
                        f"(file={path})"
                    )
                time.sleep(poll)
        logger.debug("gpu lock acquired: %s pid=%d", path, os.getpid())
        yield
    finally:
        if acquired:
            try:
                fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        fp.close()
