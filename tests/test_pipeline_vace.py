import os
import tempfile
import unittest
from unittest import mock

from src import api_server
from src.pipeline import VideoInfo, process_video


class PipelineVaceTests(unittest.TestCase):
    def test_api_exposes_vace_inpaint_mode(self):
        self.assertIn("vace", api_server._ALLOWED_INPAINT)
        self.assertIn("vace-1.3b", api_server._ALLOWED_INPAINT)

    def test_process_video_dispatches_vace_branch(self):
        info = VideoInfo(
            width=16,
            height=16,
            fps=25,
            n_frames=5,
            duration=0.2,
            has_audio=False,
        )
        roi = (0, 8, 16, 8)

        with tempfile.TemporaryDirectory() as tmp:
            output_path = os.path.join(tmp, "out.mp4")

            def fake_remove(input_path, video_only, got_roi, got_info, progress_cb=None):
                self.assertEqual(input_path, "input.mp4")
                self.assertEqual(got_roi, roi)
                self.assertEqual(got_info, info)
                with open(video_only, "wb") as fh:
                    fh.write(b"vace-video")

            with mock.patch("src.pipeline.probe_video", return_value=info), \
                    mock.patch("src.pipeline._resolve_roi", return_value=roi), \
                    mock.patch("src.pipeline.VaceSubtitleRemover") as remover_cls:
                remover_cls.return_value.remove.side_effect = fake_remove

                result = process_video(
                    "input.mp4",
                    output_path,
                    detection="roi",
                    inpaint="vace-1.3b",
                    vsr="off",
                )

        self.assertEqual(result, info)
        remover_cls.return_value.remove.assert_called_once()

    def test_vace_rejects_vsr_for_first_poc(self):
        info = VideoInfo(
            width=16,
            height=16,
            fps=25,
            n_frames=5,
            duration=0.2,
            has_audio=False,
        )

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("src.pipeline.probe_video", return_value=info), \
                mock.patch("src.pipeline._resolve_roi", return_value=(0, 8, 16, 8)):
            with self.assertRaisesRegex(ValueError, "VACE POC does not support VSR"):
                process_video(
                    "input.mp4",
                    os.path.join(tmp, "out.mp4"),
                    detection="roi",
                    inpaint="vace-1.3b",
                    vsr="real-esrgan",
                )


if __name__ == "__main__":
    unittest.main()
