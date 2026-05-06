"""OpenCV inpainting — fast subtitle removal via cv2.inpaint."""

from __future__ import annotations

import cv2
import numpy as np


def inpaint_frame(frame: np.ndarray, mask: np.ndarray, radius: int = 3,
                  method: str = "telea") -> np.ndarray:
    """Remove the masked region from a single frame using OpenCV inpainting.

    Args:
        frame: BGR uint8 (H, W, 3)
        mask:  binary uint8 (H, W) with 255 in regions to remove
        method: "telea" (cv2.INPAINT_TELEA) or "ns" (cv2.INPAINT_NS)
    """
    flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA
    return cv2.inpaint(frame, mask, radius, flag)
