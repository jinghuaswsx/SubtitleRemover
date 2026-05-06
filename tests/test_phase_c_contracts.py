import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PhaseCContractsTest(unittest.TestCase):
    def test_roi_specs_are_clamped_and_masked(self):
        from src.detection.roi_detector import build_mask, parse_roi

        self.assertEqual(parse_roi("bottom_20%", 100, 50), (0, 40, 100, 10))
        self.assertEqual(parse_roi("90,45,20,20", 100, 50), (90, 45, 10, 5))

        mask = build_mask(100, 50, (90, 45, 10, 5))

        self.assertEqual(mask.shape, (50, 100))
        self.assertEqual(int((mask == 255).sum()), 50)
        self.assertEqual(int(mask[44, 90]), 0)
        self.assertEqual(int(mask[45, 90]), 255)

    def test_config_reads_service_port_from_sr_port(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        env["SR_PORT"] = "84"

        out = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "from src import config; print(config.SERVICE_PORT)",
            ],
            cwd=ROOT,
            env=env,
            text=True,
        ).strip()

        self.assertEqual(out, "84")

    def test_api_health_reports_phase_c(self):
        from src import api_server

        self.assertEqual(
            api_server.health(),
            {
                "status": "ok",
                "phase": "C",
                "queued": 0,
                "tasks_in_memory": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
