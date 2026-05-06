"""Fixed-region subtitle detection.

For Phase B: parse a ROI spec (e.g. "bottom_20%", "x,y,w,h") into a pixel rect
and produce a binary mask suitable for inpainting.
"""

from __future__ import annotations

import re
from typing import Tuple

import numpy as np


def parse_roi(spec: str, width: int, height: int) -> Tuple[int, int, int, int]:
    """Parse a ROI spec into (x, y, w, h) in pixels.

    Supported forms:
      - "bottom_20%"  → bottom 20% strip across full width
      - "top_15%"     → top 15% strip
      - "x,y,w,h"     → absolute pixel rect (clamped to frame)
    """
    spec = spec.strip().lower()
    m = re.fullmatch(r"(top|bottom)_(\d+)%", spec)
    if m:
        side, pct = m.group(1), int(m.group(2))
        strip_h = max(1, height * pct // 100)
        y = 0 if side == "top" else height - strip_h
        return 0, y, width, strip_h

    parts = [p.strip() for p in spec.split(",")]
    if len(parts) == 4:
        x, y, w, h = (int(p) for p in parts)
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        return x, y, w, h

    raise ValueError(f"unrecognized ROI spec: {spec!r}")


def build_mask(width: int, height: int, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Build a binary mask (H, W) uint8 with 255 inside the ROI."""
    mask = np.zeros((height, width), dtype=np.uint8)
    x, y, w, h = roi
    mask[y : y + h, x : x + w] = 255
    return mask
