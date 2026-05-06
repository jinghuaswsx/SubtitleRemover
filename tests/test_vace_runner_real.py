"""Tests for VACE runner in real (non-dry-run) mode.

Subprocess and probe_video are mocked; no actual VACE/ffmpeg is invoked.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import patch

import importlib


@dataclass
class _FakeTask:
    task_id: str = "t1"
    input_path: str = ""
    output_path: str = ""
    roi_spec: Optional[str] = None
    kind: str = "vace"
    detection: str = "auto"
    ocr_engine: str = "easyocr"
    inpaint: str = "opencv"
    vsr: str = "off"
    prompt: Optional[str] = "edit"
    negative_prompt: str = ""
    mask_mode: str = "mask_file"
    mask_path: Optional[str] = None
    profile: str = "rtx4070tis_balanced"
    seed: int = -1
    state: str = "queued"
    progress: float = 0.0
    error: Optional[str] = None
    created_at: float = 0.0
    finished_at: Optional[float] = None


def _reload_with_env(**overrides):
    for k, v in overrides.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    from src import config as cfg
    importlib.reload(cfg)
    from src.vace import runner as r
    importlib.reload(r)
    return cfg, r


class VACERunnerRealTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        td = self._tmp.name
        self._inp = os.path.join(td, "in.mp4")
        self._out = os.path.join(td, "out.mp4")
        self._mask = os.path.join(td, "mask.mp4")
        self._script = os.path.join(td, "vace_wan_inference.py")
        self._ckpt = os.path.join(td, "ckpt")
        os.makedirs(self._ckpt, exist_ok=True)
        for p in (self._inp, self._mask, self._script):
            open(p, "wb").close()
        # baseline env: dry-run off, real path on
        self._cfg, self._r = _reload_with_env(
            SR_VACE_DRY_RUN="0",
            SR_VACE_ENABLED="1",
            SR_VACE_SCRIPT=self._script,
            SR_VACE_CKPT_DIR=self._ckpt,
            SR_VACE_PYTHON="/usr/bin/true",
            SR_VACE_TIMEOUT_SEC="60",
            SR_GPU_LOCK_ENABLED="0",
        )

    def tearDown(self):
        self._tmp.cleanup()
        # restore default env so other test modules don't see real-mode VACE
        for k in ("SR_VACE_DRY_RUN", "SR_VACE_ENABLED", "SR_VACE_SCRIPT",
                  "SR_VACE_CKPT_DIR", "SR_VACE_PYTHON",
                  "SR_VACE_TIMEOUT_SEC", "SR_GPU_LOCK_ENABLED"):
            os.environ.pop(k, None)

    def _task(self, **over):
        t = _FakeTask(input_path=self._inp, output_path=self._out,
                      mask_path=self._mask)
        for k, v in over.items():
            setattr(t, k, v)
        return t

    def test_missing_script_raises(self):
        os.environ["SR_VACE_SCRIPT"] = "/no/such/path"
        cfg, r = _reload_with_env()
        with self.assertRaises(RuntimeError) as cm:
            r.process_vace(self._task())
        self.assertIn("SR_VACE_SCRIPT", str(cm.exception))

    def test_mask_none_in_real_mode_raises(self):
        with self.assertRaises(RuntimeError) as cm:
            self._r.process_vace(self._task(mask_mode="none"))
        self.assertIn("mask_mode=none", str(cm.exception))

    def test_invokes_subprocess_with_profile_args(self):
        captured = {}

        def fake_run(cmd, check, timeout):
            captured["cmd"] = cmd
            # produce expected save_file
            for i, a in enumerate(cmd):
                if a == "--save_file":
                    open(cmd[i + 1], "wb").close()
                    break

            class _R:
                returncode = 0
            return _R()

        with patch("subprocess.run", side_effect=fake_run), \
             patch.object(self._r, "probe_video", create=True,
                          return_value=type("VI", (), {
                              "width": 832, "height": 480,
                              "fps": 24.0, "n_frames": 81,
                              "duration": 3.4, "has_audio": False,
                          })()):
            # patch probe_video import lazily
            from src import pipeline
            with patch.object(pipeline, "probe_video",
                              return_value=type("VI", (), {
                                  "width": 832, "height": 480,
                                  "fps": 24.0, "n_frames": 81,
                                  "duration": 3.4, "has_audio": False,
                              })()):
                self._r.process_vace(self._task(mask_mode="mask_file",
                                                profile="rtx4070tis_quality"))

        cmd = captured["cmd"]
        # generate.py uses --task, not --model_name
        self.assertNotIn("--model_name", cmd)
        self.assertIn("--task", cmd)
        self.assertEqual(cmd[cmd.index("--task") + 1], "vace-1.3B")
        # vace-1.3B SUPPORTED_SIZES is only 480*832 / 832*480
        self.assertEqual(cmd[cmd.index("--size") + 1], "832*480")
        self.assertEqual(cmd[cmd.index("--frame_num") + 1], "81")
        self.assertEqual(cmd[cmd.index("--sample_steps") + 1], "30")
        # generate.py has no --negative_prompt argument
        self.assertNotIn("--negative_prompt", cmd)
        # 4070 Ti Super profile keeps offload off + t5 on GPU
        self.assertNotIn("--offload_model", cmd)
        self.assertNotIn("--t5_cpu", cmd)
        # mask path passed through
        self.assertIn(self._mask, cmd)
        # output copied
        self.assertTrue(os.path.exists(self._out))


if __name__ == "__main__":
    unittest.main()
