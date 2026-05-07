"""VACE task execution.

Phase 1 (dry-run): write a manifest.json and copy the source video to the
output path so /download still serves something. No model load, no GPU work.

Phase 2 (real): invoke external Wan2.1-VACE-1.3B inference via the subprocess
bridge already used by src.inpainting.vace_adapter. The SR venv does NOT
install wan2.1 — the call goes to a separate VACE checkout pointed to by
SR_VACE_SCRIPT / SR_VACE_CKPT_DIR / SR_VACE_PYTHON.

Real mode is gated by SR_VACE_DRY_RUN=0. See docs/vace_integration_plan.md
§7.2 for deployment steps.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
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


def _validate_external_vace() -> None:
    if not config.VACE_SCRIPT:
        raise RuntimeError(
            "SR_VACE_SCRIPT is not set; phase 2 needs an external VACE checkout. "
            "See docs/vace_integration_plan.md §7.2."
        )
    if not os.path.isfile(config.VACE_SCRIPT):
        raise RuntimeError(f"SR_VACE_SCRIPT does not exist: {config.VACE_SCRIPT}")
    if not config.VACE_CKPT_DIR:
        raise RuntimeError("SR_VACE_CKPT_DIR is not set")
    if not os.path.isdir(config.VACE_CKPT_DIR):
        raise RuntimeError(f"SR_VACE_CKPT_DIR does not exist: {config.VACE_CKPT_DIR}")


def _resolve_mask(task, info, workdir: str) -> str:
    """Return a path to a mask video. Caller owns workdir cleanup."""
    if task.mask_mode == "mask_file":
        if not task.mask_path or not os.path.isfile(task.mask_path):
            raise RuntimeError(f"mask_file missing: {task.mask_path}")
        return task.mask_path
    if task.mask_mode == "roi":
        if not task.roi_spec:
            raise RuntimeError("mask_mode=roi requires roi_spec")
        from ..detection.roi_detector import parse_roi
        from ..inpainting.vace_adapter import VaceSubtitleRemover
        roi = parse_roi(task.roi_spec, info.width, info.height)
        mask_path = os.path.join(workdir, "src_mask.mp4")
        VaceSubtitleRemover()._write_mask_video(
            task.input_path, mask_path, roi, info,
        )
        return mask_path
    # mask_mode == "none"
    raise RuntimeError(
        "mask_mode=none is not supported in real mode; "
        "use mask_mode=roi or mask_mode=mask_file"
    )


def _build_cmd(task, profile: VACEProfile, mask_path: str, save_path: str) -> list:
    """Build a Wan2.1 generate.py command line.

    generate.py uses --task (not --model_name) and Wan2.1 SIZE_CONFIGS keys
    like '832*480' (not '480p'). It has no --negative_prompt argument; that
    field on Task is accepted for forward compat but is currently dropped.
    """
    cmd = [
        config.VACE_PYTHON,
        config.VACE_SCRIPT,
        "--task", profile.model_name,
        "--ckpt_dir", config.VACE_CKPT_DIR,
        "--src_video", task.input_path,
        "--src_mask", mask_path,
        "--prompt", task.prompt or "",
        "--size", profile.size,
        "--frame_num", str(profile.frame_num),
        "--sample_steps", str(profile.sample_steps),
        "--save_file", save_path,
    ]
    # SR_VACE_LOW_MEM=1 (default) forces low-memory flags regardless of the
    # profile's offload_model/t5_cpu — required when the GPU is shared with
    # audio :83 (16 GB total) and we cannot pre-empt it.
    if profile.offload_model or config.VACE_LOW_MEM:
        cmd.extend(["--offload_model", "True"])
    if profile.t5_cpu or config.VACE_LOW_MEM:
        cmd.append("--t5_cpu")
    if task.seed is not None and task.seed >= 0:
        cmd.extend(["--base_seed", str(task.seed)])
    return cmd


def _run_real(task, profile: VACEProfile, progress_cb) -> None:
    _validate_external_vace()

    if task.mask_mode == "none":
        raise RuntimeError(
            "mask_mode=none is not supported in real mode; "
            "use mask_mode=roi or mask_mode=mask_file"
        )

    from ..pipeline import probe_video
    info = probe_video(task.input_path)
    if progress_cb:
        progress_cb(0.10)

    workdir = tempfile.mkdtemp(prefix="vace_x_")
    try:
        mask_path = _resolve_mask(task, info, workdir)
        if progress_cb:
            progress_cb(0.25)

        save_path = os.path.join(workdir, "vace_out.mp4")
        cmd = _build_cmd(task, profile, mask_path, save_path)
        logger.info("running VACE (X path): %s", " ".join(cmd))

        # PyTorch CUDA allocator hint to reduce fragmentation under tight VRAM,
        # per the OOM message emitted by torch when memory is fragmented.
        env = {**os.environ,
               "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
                   "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")}

        if config.VACE_GPU_LOCK_ENABLED:
            from ..gpu_lock import cross_process_gpu_lock
            timeout = int(os.getenv("SR_GPU_LOCK_TIMEOUT", "1200"))
            with cross_process_gpu_lock(timeout=timeout):
                subprocess.run(cmd, check=True, env=env,
                               timeout=config.VACE_TIMEOUT_SEC)
        else:
            subprocess.run(cmd, check=True, env=env,
                           timeout=config.VACE_TIMEOUT_SEC)

        if not os.path.exists(save_path):
            raise RuntimeError(f"VACE finished but did not produce {save_path}")
        shutil.copyfile(save_path, task.output_path)
        if progress_cb:
            progress_cb(0.95)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if progress_cb:
        progress_cb(1.0)
