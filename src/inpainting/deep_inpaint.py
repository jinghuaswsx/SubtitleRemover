"""Deep inpainting via LaMa.

Wraps simple-lama-inpainting. LaMa is single-shot per frame (no temporal
context), which is fine because we only repaint the subtitle region.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("inpainting.deep")

_LAMA = None  # lazy-loaded SimpleLama instance


def _lama(device: str = "cuda"):
    global _LAMA
    if _LAMA is None:
        import os
        # SimpleLama reads LAMA_MODEL env or downloads automatically
        os.environ.setdefault("LAMA_MODEL", "")
        from simple_lama_inpainting import SimpleLama
        logger.info("loading LaMa (device=%s)", device)
        _LAMA = SimpleLama(device=device)
    return _LAMA


def inpaint_frame_lama(frame: np.ndarray, mask: np.ndarray,
                       device: str = "cuda") -> np.ndarray:
    """Inpaint one BGR frame with LaMa. Mask is (H, W) uint8 with 255 over removal regions."""
    lama = _lama(device=device)
    # SimpleLama expects PIL RGB images
    rgb = Image.fromarray(frame[:, :, ::-1])
    mask_pil = Image.fromarray(mask)
    out_rgb = lama(rgb, mask_pil)
    if out_rgb.size != rgb.size:
        out_rgb = out_rgb.resize(rgb.size, Image.BILINEAR)
    out = np.asarray(out_rgb)
    return out[:, :, ::-1].copy()  # back to BGR
