"""VACE profile registry tests (Phase 1)."""
import unittest


class VACEProfileTest(unittest.TestCase):
    def test_4070tis_profiles_registered(self):
        from src.vace.config import PROFILES
        self.assertIn("rtx4070tis_fast", PROFILES)
        self.assertIn("rtx4070tis_balanced", PROFILES)
        self.assertIn("rtx4070tis_quality", PROFILES)

    def test_4070tis_balanced_uses_1_3b_480p(self):
        from src.vace.config import get_profile
        p = get_profile("rtx4070tis_balanced")
        self.assertEqual(p.model_name, "vace-1.3B")
        self.assertEqual(p.size, "480p")
        self.assertEqual(p.frame_num, 81)
        self.assertFalse(p.offload_model)
        self.assertFalse(p.t5_cpu)

    def test_4070tis_quality_720p(self):
        from src.vace.config import get_profile
        p = get_profile("rtx4070tis_quality")
        self.assertEqual(p.size, "720p")
        self.assertEqual(p.max_long_edge, 1280)
        self.assertEqual(p.max_short_edge, 720)

    def test_4070tis_fallback_chain(self):
        from src.vace.config import fallback_profile, get_profile
        quality = get_profile("rtx4070tis_quality")
        balanced = fallback_profile(quality)
        self.assertIsNotNone(balanced)
        self.assertEqual(balanced.name, "rtx4070tis_balanced")
        fast = fallback_profile(balanced)
        self.assertIsNotNone(fast)
        self.assertEqual(fast.name, "rtx4070tis_fast")
        self.assertIsNone(fallback_profile(fast))

    def test_get_profile_unknown_raises(self):
        from src.vace.config import get_profile
        with self.assertRaises(KeyError):
            get_profile("rtx9999_super")


if __name__ == "__main__":
    unittest.main()
