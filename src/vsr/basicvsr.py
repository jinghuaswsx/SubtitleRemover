"""
VSR Module: BasicVSR++ for 1080p / 2K video super-resolution.

Enhances inpainted subtitle regions to restore visual quality.
Supports tile-based processing for 2K inputs to fit GPU memory.
"""

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger("vsr.basicvsr")

MODEL_URLS = {
    "basicvsr_plusplus_reds4": "https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_reds4_20220916-a3d03b54.pth",
}

MODEL_DIR = os.getenv("SR_MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "models"))


def download_model(name: str = "basicvsr_plusplus_reds4") -> str:
    """Download BasicVSR++ pretrained model."""
    import requests
    url = MODEL_URLS.get(name)
    if not url:
        raise ValueError(f"Unknown model: {name}")

    path = os.path.join(MODEL_DIR, f"{name}.pth")
    if os.path.exists(path):
        logger.info(f"Model exists: {path}")
        return path

    logger.info(f"Downloading {name}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(f"Downloaded: {path}")
    return path


class BasicVSRPlusPlus:
    """
    BasicVSR++ video super-resolution.

    Supports:
      - 1080p (1920x1080) — full-frame processing
      - 2K (2560x1440) — tile-based processing to fit GPU memory

    Usage:
        vsr = BasicVSRPlusPlus(device="cuda", tile_size=512)
        enhanced = vsr.enhance(low_res_frames)  # list of np.ndarray (H, W, 3)
    """

    def __init__(self, device: str = "cuda", model_name: str = "basicvsr_plusplus_reds4",
                 tile_size: int = 0, tile_pad: int = 10,
                 upscale_factor: int = 2):
        """
        Args:
            tile_size: Tile size for processing. 0 = full-frame.
                       For 2K video, set to 512-1024 to avoid OOM.
            tile_pad: Overlap padding between tiles.
            upscale_factor: 2 or 4.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.upscale_factor = upscale_factor
        self.model = None
        logger.info(f"BasicVSR++ initialized (device={self.device}, tile={tile_size}, upscale={upscale_factor})")

    def load_model(self):
        """Load the BasicVSR++ model."""
        raise NotImplementedError(
            "BasicVSR++ requires MMEditing. Install: pip install mmedit\n"
            "See: https://github.com/open-mmlab/mmediting"
        )

    def enhance(self, frames):
        """
        Enhance a sequence of frames.

        For 2K video (2560x1440), automatically uses tile processing.
        For 1080p (1920x1080), uses full-frame processing.

        Args:
            frames: list of np.ndarray (H, W, 3) RGB

        Returns:
            list of np.ndarray enhanced frames
        """
        if self.model is None:
            self.load_model()

        h, w = frames[0].shape[:2]

        # Auto-select processing mode based on resolution
        if self.tile_size > 0 and (h > 1080 or w > 1920):
            logger.info(f"2K detected ({w}x{h}), using tile processing (tile={self.tile_size})")
            return self._enhance_tiled(frames)
        else:
            logger.info(f"Using full-frame processing ({w}x{h})")
            return self._enhance_full(frames)

    def _enhance_full(self, frames):
        """Full-frame processing for up to 1080p."""
        raise NotImplementedError("Full-frame enhance not yet implemented")

    def _enhance_tiled(self, frames):
        """
        Tile-based processing for 2K+ resolution.

        Splits each frame into overlapping tiles, processes each tile
        through VSR, then stitches back with blended overlaps.
        """
        raise NotImplementedError("Tile-based enhance not yet implemented")


def infer_resolution_level(width: int, height: int) -> str:
    """Classify resolution for logging / config."""
    if width >= 3840 or height >= 2160:
        return "4K"
    elif width >= 2560 or height >= 1440:
        return "2K"
    elif width >= 1920 or height >= 1080:
        return "1080p"
    else:
        return "SD"
