"""External VACE subtitle inpainting adapter.

This module intentionally does not import VACE. It prepares the video mask and
delegates generation to a separately managed VACE checkout/venv.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from .. import config

logger = logging.getLogger("inpainting.vace")

VACE_METHODS = {"vace", "vace-1.3b", "vace-1.3B"}


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def is_vace_method(method: str) -> bool:
    return method.strip().lower() in {m.lower() for m in VACE_METHODS}


@dataclass
class VaceConfig:
    python_bin: str = config.VACE_PYTHON
    script_path: str = config.VACE_SCRIPT
    checkpoint_dir: str = config.VACE_CKPT_DIR
    model_name: str = config.VACE_MODEL_NAME
    size: str = config.VACE_SIZE
    frame_num: int = config.VACE_FRAME_NUM
    prompt: str = config.VACE_PROMPT
    sample_steps: int = config.VACE_SAMPLE_STEPS
    offload_model: bool = _parse_bool(config.VACE_OFFLOAD_MODEL)
    t5_cpu: bool = _parse_bool(config.VACE_T5_CPU)
    max_duration_sec: float = config.VACE_MAX_DURATION_SEC
    timeout_sec: int = config.VACE_TIMEOUT_SEC

    @classmethod
    def from_env(cls) -> "VaceConfig":
        return cls(
            python_bin=os.getenv("SR_VACE_PYTHON", config.VACE_PYTHON),
            script_path=os.getenv("SR_VACE_SCRIPT", config.VACE_SCRIPT),
            checkpoint_dir=os.getenv("SR_VACE_CKPT_DIR", config.VACE_CKPT_DIR),
            model_name=os.getenv("SR_VACE_MODEL_NAME", config.VACE_MODEL_NAME),
            size=os.getenv("SR_VACE_SIZE", config.VACE_SIZE),
            frame_num=int(os.getenv("SR_VACE_FRAME_NUM", str(config.VACE_FRAME_NUM))),
            prompt=os.getenv("SR_VACE_PROMPT", config.VACE_PROMPT),
            sample_steps=int(os.getenv("SR_VACE_SAMPLE_STEPS", str(config.VACE_SAMPLE_STEPS))),
            offload_model=_parse_bool(os.getenv("SR_VACE_OFFLOAD_MODEL", config.VACE_OFFLOAD_MODEL)),
            t5_cpu=_parse_bool(os.getenv("SR_VACE_T5_CPU", config.VACE_T5_CPU)),
            max_duration_sec=float(
                os.getenv("SR_VACE_MAX_DURATION_SEC", str(config.VACE_MAX_DURATION_SEC))
            ),
            timeout_sec=int(os.getenv("SR_VACE_TIMEOUT_SEC", str(config.VACE_TIMEOUT_SEC))),
        )

    def validate(self) -> None:
        missing = []
        if not self.script_path:
            missing.append("SR_VACE_SCRIPT")
        if not self.checkpoint_dir:
            missing.append("SR_VACE_CKPT_DIR")
        if missing:
            raise RuntimeError(
                "VACE is not configured. Set SR_VACE_SCRIPT and "
                "SR_VACE_CKPT_DIR to an external VACE checkout and model."
            )
        if not os.path.isfile(self.script_path):
            raise RuntimeError(f"SR_VACE_SCRIPT does not exist: {self.script_path}")
        if not os.path.isdir(self.checkpoint_dir):
            raise RuntimeError(f"SR_VACE_CKPT_DIR does not exist: {self.checkpoint_dir}")

    def build_command(self, *, src_video: str, src_mask: str, save_file: str) -> list[str]:
        cmd = [
            self.python_bin,
            self.script_path,
            "--model_name",
            self.model_name,
            "--ckpt_dir",
            self.checkpoint_dir,
            "--src_video",
            src_video,
            "--src_mask",
            src_mask,
            "--prompt",
            self.prompt,
            "--size",
            self.size,
            "--frame_num",
            str(self.frame_num),
            "--sample_steps",
            str(self.sample_steps),
            "--save_file",
            save_file,
        ]
        if self.offload_model:
            cmd.extend(["--offload_model", "True"])
        if self.t5_cpu:
            cmd.append("--t5_cpu")
        return cmd


class VaceSubtitleRemover:
    def __init__(self, cfg: Optional[VaceConfig] = None):
        self.cfg = cfg or VaceConfig.from_env()

    def remove(
        self,
        input_path: str,
        output_path: str,
        roi: Tuple[int, int, int, int],
        info,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> None:
        if info.duration > self.cfg.max_duration_sec:
            raise ValueError(
                "VACE POC only supports clips up to "
                f"{self.cfg.max_duration_sec:g}s; got {info.duration:.1f}s"
            )

        self.cfg.validate()
        workdir = tempfile.mkdtemp(prefix="vace_")
        try:
            mask_path = os.path.join(workdir, "src_mask.mp4")
            generated_path = os.path.join(workdir, "vace_out.mp4")
            if progress_cb:
                progress_cb(0.05)
            self._write_mask_video(input_path, mask_path, roi, info)
            if progress_cb:
                progress_cb(0.20)

            cmd = self.cfg.build_command(
                src_video=input_path,
                src_mask=mask_path,
                save_file=generated_path,
            )
            logger.info("running VACE: %s", " ".join(cmd))
            subprocess.run(cmd, check=True, timeout=self.cfg.timeout_sec)
            if not os.path.exists(generated_path):
                raise RuntimeError(f"VACE finished but did not create {generated_path}")
            shutil.copyfile(generated_path, output_path)
            if progress_cb:
                progress_cb(0.95)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _write_mask_video(
        self,
        input_path: str,
        mask_path: str,
        roi: Tuple[int, int, int, int],
        info,
    ) -> None:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"could not open input video for VACE mask: {input_path}")

        fps = info.fps or 25
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(mask_path, fourcc, fps, (info.width, info.height), True)
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"could not create VACE mask video: {mask_path}")

        x, y, w, h = roi
        frames = 0
        try:
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                mask = np.zeros((info.height, info.width, 3), dtype=np.uint8)
                mask[y : y + h, x : x + w] = 255
                writer.write(mask)
                frames += 1
        finally:
            cap.release()
            writer.release()

        if frames == 0:
            raise RuntimeError("could not read any frames while creating VACE mask")
