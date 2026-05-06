"""OCR-based subtitle region detection.

Two backends:
    - "easyocr"  — works offline once models are cached, light footprint
    - "paddle"   — PaddleOCR; better for dense Chinese text

To stay fast we sample frames at a stride and union the detected boxes into a
single mask reused for neighboring frames. Subtitles drift slowly relative to
frame rate, so an inflated static mask is harmless for inpainting.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("detection.ocr")

_READER_EASY = None
_READER_PADDLE = None


def _easyocr_reader(langs: Sequence[str], gpu: bool):
    global _READER_EASY
    if _READER_EASY is None:
        import easyocr
        logger.info("loading EasyOCR (langs=%s, gpu=%s)", langs, gpu)
        _READER_EASY = easyocr.Reader(list(langs), gpu=gpu, verbose=False)
    return _READER_EASY


def _paddle_reader():
    global _READER_PADDLE
    if _READER_PADDLE is None:
        from paddleocr import PaddleOCR
        logger.info("loading PaddleOCR (lang=ch)")
        _READER_PADDLE = PaddleOCR(lang="ch", use_textline_orientation=False)
    return _READER_PADDLE


def detect_text_boxes(frame: np.ndarray, *,
                      engine: str = "easyocr",
                      langs: Sequence[str] = ("ch_sim", "en"),
                      gpu: bool = True,
                      min_conf: float = 0.3) -> List[Tuple[int, int, int, int]]:
    """Return text bounding boxes (x, y, w, h) for one BGR frame."""
    boxes: List[Tuple[int, int, int, int]] = []
    if engine == "easyocr":
        reader = _easyocr_reader(langs, gpu)
        rgb = frame[:, :, ::-1]
        for poly, _text, conf in reader.readtext(rgb, detail=1, paragraph=False):
            if conf < min_conf:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x, y = int(min(xs)), int(min(ys))
            w, h = int(max(xs) - x), int(max(ys) - y)
            if w > 0 and h > 0:
                boxes.append((x, y, w, h))
    elif engine == "paddle":
        reader = _paddle_reader()
        # paddleocr 3.x: predict() returns a list of OCRResult dicts. We use
        # rec_polys + rec_scores so only boxes that passed text recognition
        # contribute (filters out spurious detections).
        results = reader.predict(frame)
        for res in results or []:
            polys = res.get("rec_polys") or res.get("dt_polys", [])
            scores = res.get("rec_scores") or [1.0] * len(polys)
            for poly, conf in zip(polys, scores):
                if conf < min_conf:
                    continue
                poly_arr = np.asarray(poly)
                x = int(poly_arr[:, 0].min())
                y = int(poly_arr[:, 1].min())
                w = int(poly_arr[:, 0].max() - x)
                h = int(poly_arr[:, 1].max() - y)
                if w > 0 and h > 0:
                    boxes.append((x, y, w, h))
    else:
        raise ValueError(f"unknown OCR engine: {engine!r}")
    return boxes


def boxes_to_mask(boxes: Iterable[Tuple[int, int, int, int]], width: int, height: int,
                  pad: int = 4) -> np.ndarray:
    """Rasterize text boxes into a binary mask (H, W) uint8 with 255 in text regions."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for x, y, w, h in boxes:
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(width, x + w + pad)
        y1 = min(height, y + h + pad)
        mask[y0:y1, x0:x1] = 255
    return mask


def detect_static_subtitle_band(samples: Sequence[np.ndarray],
                                roi_hint: Optional[Tuple[int, int, int, int]] = None,
                                *,
                                engine: str = "easyocr",
                                langs: Sequence[str] = ("ch_sim", "en"),
                                gpu: bool = True) -> Tuple[int, int, int, int]:
    """Locate a stable subtitle band by OCR-ing several sampled frames.

    Returns the union bbox of all detected text boxes. If `roi_hint` is given,
    detections outside it are dropped — useful when the user knows subtitles
    sit in the bottom strip and wants to reject other on-screen text.
    """
    h, w = samples[0].shape[:2]
    union: Optional[List[int]] = None  # [x0, y0, x1, y1]
    for frame in samples:
        for x, y, bw, bh in detect_text_boxes(frame, engine=engine, langs=langs, gpu=gpu):
            if roi_hint is not None:
                rx, ry, rw, rh = roi_hint
                if x < rx or y < ry or (x + bw) > (rx + rw) or (y + bh) > (ry + rh):
                    continue
            x0, y0, x1, y1 = x, y, x + bw, y + bh
            if union is None:
                union = [x0, y0, x1, y1]
            else:
                union[0] = min(union[0], x0)
                union[1] = min(union[1], y0)
                union[2] = max(union[2], x1)
                union[3] = max(union[3], y1)
    if union is None:
        # nothing found — fall back to the hint or bottom 20%
        if roi_hint is not None:
            return roi_hint
        strip = max(1, h // 5)
        return 0, h - strip, w, strip
    return union[0], union[1], union[2] - union[0], union[3] - union[1]
