"""VACE /vace-edit route dry-run tests (Phase 1).

Drives the FastAPI app with TestClient. Re-imports `src.config` and
`src.api_server` once with VACE enabled + dry-run, then mutates
`config.VACE_ENABLED` per-test for the disabled-503 case.
"""

from __future__ import annotations

import importlib
import os
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _seed_sample(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(
            b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42mp41"
            b"\x00\x00\x00\x08free"
        )


class VACERouteDryRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["SR_VACE_ENABLED"] = "1"
        os.environ["SR_VACE_DRY_RUN"] = "1"
        # keep GPU fraction modest in case CUDA is touched at import
        os.environ.setdefault("SR_GPU_FRACTION", "0.05")

        from src import config as cfg
        importlib.reload(cfg)
        from src import api_server
        importlib.reload(api_server)
        cls.cfg = cfg
        cls.api_server = api_server

        from fastapi.testclient import TestClient
        cls.client = TestClient(api_server.app)

        cls._sample = ROOT / "tests" / "_sample_vace.mp4"
        _seed_sample(cls._sample)

    def _post_vace(self, **data):
        with self._sample.open("rb") as f:
            files = {"file": ("a.mp4", f, "video/mp4")}
            return self.client.post("/vace-edit", files=files, data=data)

    def _wait(self, task_id: str, timeout: float = 30.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            r = self.client.get(f"/status/{task_id}")
            self.assertEqual(r.status_code, 200)
            s = r.json()
            if s["state"] in ("done", "failed"):
                return s
            time.sleep(0.05)
        self.fail(f"task {task_id} did not finish in {timeout}s")

    def test_health_reports_vace_enabled_true(self):
        body = self.client.get("/health").json()
        self.assertTrue(body.get("vace_enabled"))

    def test_info_lists_vace_profiles(self):
        body = self.client.get("/info").json()
        self.assertIn("vace_profile", body)
        self.assertEqual(
            set(body["vace_profile"]),
            {"rtx4070tis_fast", "rtx4070tis_balanced", "rtx4070tis_quality"},
        )
        self.assertTrue(body.get("vace_enabled"))

    def test_vace_edit_requires_prompt(self):
        # FastAPI returns 422 from missing required Form field
        with self._sample.open("rb") as f:
            r = self.client.post("/vace-edit",
                                 files={"file": ("a.mp4", f, "video/mp4")},
                                 data={})
        self.assertEqual(r.status_code, 422)

    def test_vace_edit_unknown_profile(self):
        r = self._post_vace(prompt="hello", profile="nope")
        self.assertEqual(r.status_code, 400)

    def test_vace_edit_unknown_mask_mode(self):
        r = self._post_vace(prompt="hello", mask_mode="weird")
        self.assertEqual(r.status_code, 400)

    def test_vace_edit_roi_mode_requires_roi(self):
        r = self._post_vace(prompt="hello", mask_mode="roi")
        self.assertEqual(r.status_code, 422)

    def test_vace_edit_dry_run_completes_and_downloads(self):
        r = self._post_vace(
            prompt="remove background distractions",
            profile="rtx4070tis_balanced",
        )
        self.assertEqual(r.status_code, 200, msg=r.text)
        task_id = r.json()["task_id"]
        s = self._wait(task_id)
        self.assertEqual(s["state"], "done", msg=s)

        d = self.client.get(f"/download/{task_id}")
        self.assertEqual(d.status_code, 200)
        self.assertEqual(d.headers["content-type"], "video/mp4")

        # manifest sidecar should exist
        from src import config as cfg
        manifest = Path(cfg.OUTPUT_DIR) / f"{task_id}.mp4.manifest.json"
        self.assertTrue(manifest.exists(), msg=str(manifest))

    def test_serial_with_subtitle_via_kind_field(self):
        """Both task kinds share a single QUEUE; FIFO ordering is preserved."""
        # Enqueue a vace task; verify Task.kind on the in-memory record.
        r = self._post_vace(prompt="x", profile="rtx4070tis_fast")
        self.assertEqual(r.status_code, 200, msg=r.text)
        task_id = r.json()["task_id"]
        task = self.api_server.TASKS[task_id]
        self.assertEqual(task.kind, "vace")
        self.assertEqual(task.profile, "rtx4070tis_fast")
        self.assertEqual(task.prompt, "x")
        # Wait for completion to keep the worker idle for later tests.
        self._wait(task_id)

    def test_vace_disabled_returns_503(self):
        self.cfg.VACE_ENABLED = False
        try:
            r = self._post_vace(prompt="x")
        finally:
            self.cfg.VACE_ENABLED = True
        self.assertEqual(r.status_code, 503)


if __name__ == "__main__":
    unittest.main()
