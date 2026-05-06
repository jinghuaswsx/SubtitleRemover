"""BasicVSR++ video super-resolution.

Uses the basicsr implementation (`basicsr.archs.basicvsrpp_arch.BasicVSRPlusPlus`).
Inference runs over short sliding windows (default 5 frames) so VRAM stays
bounded; for 2K input set `tile_size>0` to additionally crop spatially.

Checkpoint compatibility:
    The released mmediting checkpoint wraps weights as
    ``{"state_dict": {"generator.xxx": ...}}``. We strip that prefix at load
    time so basicsr's arch can consume it directly.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger("vsr.basicvsr")

MODEL_URLS = {
    "basicvsr_plusplus_reds4": (
        "https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/"
        "basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth"
    ),
}


def download_model(name: str, model_dir: str) -> str:
    import requests
    url = MODEL_URLS.get(name)
    if not url:
        raise ValueError(f"unknown BasicVSR++ checkpoint: {name}")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{name}.pth")
    if os.path.exists(path):
        return path
    logger.info("downloading %s -> %s", name, path)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return path


def _strip_prefix(state_dict: dict, prefixes: tuple) -> dict:
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
                break
        out[nk] = v
    return out


def _remap_mmediting_to_basicsr(sd: dict) -> dict:
    """Convert mmediting BasicVSR++ checkpoint keys to basicsr arch keys.

    Differences:
      - SPyNet conv layers in mmediting wrap the Conv2d as `.conv.weight`;
        basicsr stores them flat: `.weight`.
      - Upsampling: mmediting `upsample1.upsample_conv.*` → basicsr `upconv1.*`,
        same for `upsample2`.
      - mmediting also stores a `step_counter` scalar — drop it.
    """
    out = {}
    for k, v in sd.items():
        if k == "step_counter":
            continue
        nk = k
        if nk.startswith("spynet."):
            nk = nk.replace(".conv.weight", ".weight").replace(".conv.bias", ".bias")
        if nk.startswith("upsample1.upsample_conv"):
            nk = nk.replace("upsample1.upsample_conv", "upconv1")
        elif nk.startswith("upsample2.upsample_conv"):
            nk = nk.replace("upsample2.upsample_conv", "upconv2")
        out[nk] = v
    return out


class BasicVSRPlusPlus:
    """Sliding-window BasicVSR++ inference.

    Args:
        device: "cuda" or "cpu".
        model_dir: where to cache downloaded weights.
        model_name: key in MODEL_URLS.
        window: number of frames per inference batch (must be >=2).
        tile_size: spatial tile size (0 = full frame). Use 480-512 for 2K.
        tile_pad: overlap in pixels.
        upscale_factor: model upscale (REDS4 = x4).
    """

    def __init__(self, device: str = "cuda",
                 model_dir: Optional[str] = None,
                 model_name: str = "basicvsr_plusplus_reds4",
                 window: int = 5, tile_size: int = 0, tile_pad: int = 16,
                 upscale_factor: int = 4):
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.model_dir = model_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "models")
        self.model_name = model_name
        self.window = max(2, window)
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.upscale_factor = upscale_factor
        self.model = None

    def load_model(self) -> None:
        """Load BasicVSR++ weights into a basicsr arch.

        NOTE: The publicly available openmmlab BasicVSR++ checkpoint targets
        mmediting's SPyNet, whose internal Conv2d channel widths differ from
        basicsr's SPyNet at deeper levels (basic_module.4/5). After key
        remapping ~210/274 tensors load cleanly, but the SPyNet flow estimator
        cannot be loaded due to shape mismatch. We bail out with a clear
        message rather than silently produce garbage. Use ``vsr=real-esrgan``
        instead, or supply a basicsr-native BasicVSR++ checkpoint via
        ``SR_VSR_CHECKPOINT``.
        """
        from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus as _Arch
        custom = os.getenv("SR_VSR_CHECKPOINT")
        if custom:
            ckpt_path = custom
        else:
            ckpt_path = download_model(self.model_name, self.model_dir)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "params" in ckpt:
            sd = ckpt["params"]
        else:
            sd = ckpt
        sd = _strip_prefix(sd, ("generator.", "module."))
        sd = _remap_mmediting_to_basicsr(sd)
        net = _Arch(mid_channels=64, num_blocks=7, max_residue_magnitude=10,
                    is_low_res_input=True, spynet_path=None)
        try:
            missing, unexpected = net.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                "BasicVSR++ checkpoint is not compatible with the basicsr arch "
                "(mmediting and basicsr SPyNet have divergent channel widths). "
                "Use vsr=real-esrgan, or provide a basicsr-native checkpoint via "
                "SR_VSR_CHECKPOINT=<path>."
            ) from e
        self.model = net.to(self.device).eval()
        logger.info("BasicVSR++ ready (device=%s, window=%d, tile=%d)",
                    self.device, self.window, self.tile_size)

    @torch.no_grad()
    def _run_window(self, frames_bgr: List[np.ndarray]) -> List[np.ndarray]:
        """Run one sliding window through the network. Returns same number of frames."""
        # BGR uint8 → RGB float32 NCHW in [0,1], then add temporal dim
        arr = np.stack([f[:, :, ::-1] for f in frames_bgr]).astype(np.float32) / 255.0
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).to(self.device)
        if self.device == "cuda":
            x = x.contiguous()
        y = self.model(x)  # (1, T, 3, H*s, W*s)
        y = y.squeeze(0).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        out = (y * 255.0).round().astype(np.uint8)
        return [f[:, :, ::-1].copy() for f in out]  # back to BGR

    def enhance(self, frames_bgr: List[np.ndarray]) -> List[np.ndarray]:
        """Process a clip of BGR frames. Returns enhanced BGR frames at the same
        temporal length but spatially upscaled by `upscale_factor`."""
        if self.model is None:
            self.load_model()
        if not frames_bgr:
            return []
        if self.tile_size > 0:
            return self._enhance_tiled(frames_bgr)
        return self._enhance_sliding(frames_bgr)

    def _enhance_sliding(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        n = len(frames)
        i = 0
        while i < n:
            j = min(n, i + self.window)
            chunk = frames[i:j]
            if len(chunk) < 2:  # arch needs at least 2 frames; pad by repeating
                chunk = chunk + [chunk[-1]]
                enhanced = self._run_window(chunk)[:1]
            else:
                enhanced = self._run_window(chunk)
            out.extend(enhanced)
            i = j
        return out

    def _enhance_tiled(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Spatial tile + sliding window. Used for 2K+ inputs."""
        h, w = frames[0].shape[:2]
        s = self.upscale_factor
        ts, pad = self.tile_size, self.tile_pad
        out_h, out_w = h * s, w * s

        # accumulate output frames
        out_frames = [np.zeros((out_h, out_w, 3), dtype=np.uint8) for _ in frames]
        for y0 in range(0, h, ts):
            for x0 in range(0, w, ts):
                y1 = min(h, y0 + ts)
                x1 = min(w, x0 + ts)
                py0 = max(0, y0 - pad); px0 = max(0, x0 - pad)
                py1 = min(h, y1 + pad); px1 = min(w, x1 + pad)
                tile_clip = [f[py0:py1, px0:px1] for f in frames]
                enhanced_clip = self._enhance_sliding(tile_clip)
                # crop the un-padded center
                ty0 = (y0 - py0) * s
                tx0 = (x0 - px0) * s
                ty1 = ty0 + (y1 - y0) * s
                tx1 = tx0 + (x1 - x0) * s
                for k, ef in enumerate(enhanced_clip):
                    out_frames[k][y0 * s:y1 * s, x0 * s:x1 * s] = \
                        ef[ty0:ty1, tx0:tx1]
        return out_frames


def infer_resolution_level(width: int, height: int) -> str:
    if width >= 3840 or height >= 2160:
        return "4K"
    if width >= 2560 or height >= 1440:
        return "2K"
    if width >= 1920 or height >= 1080:
        return "1080p"
    return "SD"
