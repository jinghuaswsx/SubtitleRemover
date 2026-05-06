"""Real-ESRGAN: per-frame super-resolution / detail restoration.

Used to sharpen the inpainted subtitle region after deep inpainting. We keep
this single-image (not temporal) — for restoring an isolated patch this is
both cheaper and more reliable than full-video VSR.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger("vsr.real_esrgan")

# Official Real-ESRGAN x4plus weights (basicsr-compatible)
WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
    "RealESRGAN_x4plus.pth"
)
WEIGHTS_NAME = "RealESRGAN_x4plus.pth"


def _ensure_weights(model_dir: str) -> str:
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, WEIGHTS_NAME)
    if os.path.exists(path):
        return path
    import requests
    logger.info("downloading Real-ESRGAN weights -> %s", path)
    r = requests.get(WEIGHTS_URL, stream=True, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return path


class RealESRGAN:
    """Real-ESRGAN x4 wrapper.

    Build once, call enhance(frame_bgr) per frame. Internally tile-processes
    to keep VRAM bounded (default tile=512 with pad=10).
    """

    def __init__(self, model_dir: str, device: str = "cuda",
                 tile: int = 512, tile_pad: int = 10, half: bool = True):
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        weights = _ensure_weights(model_dir)
        arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                       num_grow_ch=32, scale=4)
        self.engine = RealESRGANer(
            scale=4,
            model_path=weights,
            model=arch,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=0,
            half=half and device == "cuda",
            device=device,
        )
        logger.info("Real-ESRGAN ready (device=%s, tile=%d, half=%s)", device, tile, half)

    def enhance(self, frame_bgr: np.ndarray, outscale: float = 1.0) -> np.ndarray:
        """Enhance a BGR frame. outscale=1.0 keeps original resolution; the x4 model
        super-resolves internally then downsamples back."""
        out, _ = self.engine.enhance(frame_bgr, outscale=outscale)
        return out

    def enhance_region(self, frame_bgr: np.ndarray,
                       region: tuple[int, int, int, int],
                       feather: int = 8) -> np.ndarray:
        """Run Real-ESRGAN only on a sub-region and feather-blend it back into the frame.

        Args:
            region: (x, y, w, h) ROI to enhance.
            feather: width of the soft blend border in pixels.
        """
        x, y, w, h = region
        h_frame, w_frame = frame_bgr.shape[:2]
        x = max(0, x); y = max(0, y)
        w = min(w, w_frame - x); h = min(h, h_frame - y)
        if w <= 0 or h <= 0:
            return frame_bgr

        crop = frame_bgr[y:y + h, x:x + w]
        enhanced = self.enhance(crop, outscale=1.0)
        # may differ in size by 1-2 px; resize back
        if enhanced.shape[:2] != (h, w):
            enhanced = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # feathered alpha for seamless blend
        alpha = np.ones((h, w), dtype=np.float32)
        f = max(1, min(feather, w // 4, h // 4))
        ramp = np.linspace(0, 1, f, dtype=np.float32)
        alpha[:f, :] *= ramp[:, None]
        alpha[-f:, :] *= ramp[::-1][:, None]
        alpha[:, :f] *= ramp[None, :]
        alpha[:, -f:] *= ramp[None, ::-1]
        alpha3 = alpha[:, :, None]

        out = frame_bgr.copy()
        blended = (enhanced.astype(np.float32) * alpha3 +
                   crop.astype(np.float32) * (1.0 - alpha3))
        out[y:y + h, x:x + w] = blended.astype(np.uint8)
        return out
