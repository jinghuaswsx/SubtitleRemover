"""VACE profile registry tests (Phase 1)."""
import unittest


class VACEProfileTest(unittest.TestCase):
    def test_4070tis_profiles_registered(self):
        from src.vace.config import PROFILES
        self.assertIn("rtx4070tis_fast", PROFILES)
        self.assertIn("rtx4070tis_balanced", PROFILES)
        self.assertIn("rtx4070tis_quality", PROFILES)

    def test_4070tis_fast_uses_small_frame_num_for_shared_gpu(self):
        from src.vace.config import get_profile
        p = get_profile("rtx4070tis_fast")
        # frame_num=17 keeps VAE decode peak fitting alongside audio :83 on 16GB
        self.assertEqual(p.frame_num, 17)

    def test_4070tis_balanced_uses_1_3b_480p(self):
        from src.vace.config import get_profile
        p = get_profile("rtx4070tis_balanced")
        self.assertEqual(p.model_name, "vace-1.3B")
        # Wan2.1 SUPPORTED_SIZES['vace-1.3B'] = ('480*832', '832*480')
        self.assertEqual(p.size, "832*480")
        self.assertEqual(p.frame_num, 81)
        self.assertFalse(p.offload_model)
        self.assertFalse(p.t5_cpu)

    def test_4070tis_quality_keeps_1_3b(self):
        from src.vace.config import get_profile
        p = get_profile("rtx4070tis_quality")
        # vace-1.3B doesn't support 720p; quality differs by sample_steps only.
        self.assertEqual(p.model_name, "vace-1.3B")
        self.assertEqual(p.size, "832*480")
        self.assertEqual(p.sample_steps, 30)

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
