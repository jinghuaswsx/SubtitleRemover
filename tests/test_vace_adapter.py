import os
import tempfile
import unittest

from src.inpainting.vace_adapter import (
    VaceConfig,
    VaceSubtitleRemover,
    is_vace_method,
)
from src.pipeline import VideoInfo


class VaceAdapterTests(unittest.TestCase):
    def test_is_vace_method_accepts_supported_aliases(self):
        self.assertTrue(is_vace_method("vace"))
        self.assertTrue(is_vace_method("vace-1.3b"))
        self.assertTrue(is_vace_method("VACE-1.3B"))
        self.assertFalse(is_vace_method("lama"))

    def test_build_command_includes_offload_and_t5_cpu_flags(self):
        cfg = VaceConfig(
            python_bin="/venv/bin/python",
            script_path="/opt/VACE/vace/vace_wan_inference.py",
            checkpoint_dir="/models/Wan2.1-VACE-1.3B",
            model_name="vace-1.3B",
            size="480p",
            frame_num=81,
            prompt="Remove subtitles.",
            sample_steps=30,
            offload_model=True,
            t5_cpu=True,
        )

        cmd = cfg.build_command(
            src_video="/tmp/input.mp4",
            src_mask="/tmp/mask.mp4",
            save_file="/tmp/out.mp4",
        )

        self.assertEqual(cmd[0], "/venv/bin/python")
        self.assertIn("--model_name", cmd)
        self.assertIn("vace-1.3B", cmd)
        self.assertIn("--offload_model", cmd)
        self.assertIn("True", cmd)
        self.assertIn("--t5_cpu", cmd)
        self.assertIn("--save_file", cmd)
        self.assertIn("/tmp/out.mp4", cmd)

    def test_missing_config_raises_clear_error(self):
        cfg = VaceConfig(script_path="", checkpoint_dir="")

        with self.assertRaisesRegex(RuntimeError, "SR_VACE_SCRIPT.*SR_VACE_CKPT_DIR"):
            cfg.validate()

    def test_remove_rejects_long_video_before_running_vace(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "vace_wan_inference.py")
            ckpt = os.path.join(tmp, "ckpt")
            os.mkdir(ckpt)
            with open(script, "w", encoding="utf-8") as fh:
                fh.write("print('stub')\n")

            cfg = VaceConfig(
                script_path=script,
                checkpoint_dir=ckpt,
                max_duration_sec=5,
            )
            remover = VaceSubtitleRemover(cfg)
            info = VideoInfo(
                width=16,
                height=16,
                fps=25,
                n_frames=250,
                duration=10,
                has_audio=False,
            )

            with self.assertRaisesRegex(ValueError, "VACE POC only supports clips"):
                remover.remove(
                    "input.mp4",
                    os.path.join(tmp, "out.mp4"),
                    (0, 8, 16, 8),
                    info,
                )


if __name__ == "__main__":
    unittest.main()
