"""VACE task execution.

Phase 1 (dry-run): write a manifest.json and copy the source video to the
output path so /download still serves something. No model load, no GPU work.

Phase 2 (real): invoke Wan2.1-VACE-1.3B inference with chunk-based
processing, gated by SR_VACE_DRY_RUN=0. Implementation lands in commit 3
of docs/vace_integration_plan.md.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict
from typing import Callable, Optional

from .. import config
from .config import VACEProfile, get_profile

logger = logging.getLogger("vace.runner")


def process_vace(task, progress_cb: Optional[Callable[[float], None]] = None) -> None:
    """Execute a VACE task. Dispatches by SR_VACE_DRY_RUN."""
    profile = get_profile(task.profile)
    if progress_cb:
        progress_cb(0.05)

    if config.VACE_DRY_RUN:
        _run_dry(task, profile, progress_cb)
    else:
        _run_real(task, profile, progress_cb)


def _run_dry(task, profile: VACEProfile, progress_cb) -> None:
    manifest = {
        "task_id": task.task_id,
        "kind": "vace",
        "dry_run": True,
        "input_path": task.input_path,
        "output_path": task.output_path,
        "prompt": task.prompt,
        "negative_prompt": task.negative_prompt,
        "mask_mode": task.mask_mode,
        "roi_spec": task.roi_spec,
        "mask_path": task.mask_path,
        "seed": task.seed,
        "profile": asdict(profile),
        "created_at": task.created_at,
        "completed_at": time.time(),
    }
    manifest_path = task.output_path + ".manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if progress_cb:
        progress_cb(0.5)

    shutil.copyfile(task.input_path, task.output_path)
    logger.info("vace dry-run done: task=%s manifest=%s",
                task.task_id, manifest_path)
    if progress_cb:
        progress_cb(1.0)


def _run_real(task, profile: VACEProfile, progress_cb) -> None:
    raise NotImplementedError(
        "real VACE inference not yet implemented; set SR_VACE_DRY_RUN=1 or "
        "wait for Phase 2 (commit 3 of docs/vace_integration_plan.md)"
    )
