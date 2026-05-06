"""Video processing pipeline.

Modes (selected per-request or via env):

  detection:
    - "roi": fixed band (e.g. bottom_20%)
    - "ocr": OCR a few sample frames, take the union of text boxes
  inpainting:
    - "opencv": cv2.inpaint TELEA — fast, low quality
    - "lama":   LaMa deep inpaint — slower, much better quality
  vsr:
    - "off":          no super-resolution
    - "real-esrgan":  per-frame Real-ESRGAN on the ROI region only (default)
    - "basicvsr++":   full-frame temporal VSR (heavy; upscales the output)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from . import config
from .detection.roi_detector import build_mask, parse_roi
from .inpainting.opencv_inpaint import inpaint_frame as opencv_inpaint
from .inpainting.vace_adapter import VaceSubtitleRemover, is_vace_method

logger = logging.getLogger("pipeline")


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    n_frames: int
    duration: float
    has_audio: bool


def probe_video(path: str) -> VideoInfo:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_streams", "-show_format", path]
    data = json.loads(subprocess.check_output(cmd, text=True))
    v = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        raise ValueError(f"no video stream in {path}")
    has_audio = any(s["codec_type"] == "audio" for s in data["streams"])
    num, den = (int(x) for x in v.get("avg_frame_rate", "0/1").split("/"))
    fps = num / den if den else 0.0
    duration = float(data["format"].get("duration", 0.0))
    n_frames = int(v.get("nb_frames") or round(fps * duration))
    return VideoInfo(int(v["width"]), int(v["height"]), fps, n_frames, duration, has_audio)


def _sample_frames(input_path: str, info: VideoInfo, n: int = 8) -> List[np.ndarray]:
    """Decode `n` evenly-spaced frames for OCR-based detection."""
    cap = cv2.VideoCapture(input_path)
    total = info.n_frames or 1
    indices = np.linspace(0, max(0, total - 1), num=n, dtype=int).tolist()
    frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def _resolve_roi(input_path: str, info: VideoInfo, mode: str,
                 roi_spec: Optional[str], device: str,
                 ocr_engine: str = "easyocr") -> Tuple[int, int, int, int]:
    """Pick a (x, y, w, h) subtitle region using the chosen detection mode."""
    spec = roi_spec or config.SUBTITLE_ROI
    hint_roi = parse_roi(spec, info.width, info.height)
    if mode == "roi":
        return hint_roi
    if mode == "ocr":
        from .detection.ocr_detector import detect_static_subtitle_band
        samples = _sample_frames(input_path, info, n=6)
        if not samples:
            return hint_roi
        return detect_static_subtitle_band(
            samples, roi_hint=hint_roi, engine=ocr_engine,
            gpu=(device == "cuda"))
    if mode == "auto":
        try:
            return _resolve_roi(input_path, info, "ocr", roi_spec, device, ocr_engine)
        except Exception as e:
            logger.warning("OCR detection failed (%s), falling back to ROI", e)
            return hint_roi
    raise ValueError(f"unknown detection mode: {mode}")


def _make_inpaint_fn(method: str, device: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    if method == "opencv":
        return lambda f, m: opencv_inpaint(f, m, method="telea")
    if method == "lama":
        from .inpainting.deep_inpaint import inpaint_frame_lama
        return lambda f, m: inpaint_frame_lama(f, m, device=device)
    raise ValueError(f"unknown inpaint method: {method}")


def _make_vsr(model: str, device: str, info: VideoInfo):
    """Returns (kind, engine) where kind in {'off','region','frame'}."""
    if model in ("off", "none", ""):
        return "off", None
    if model in ("real-esrgan", "realesrgan"):
        from .vsr.real_esrgan import RealESRGAN
        engine = RealESRGAN(model_dir=config.MODEL_DIR, device=device,
                            tile=512 if max(info.width, info.height) > 1920 else 0)
        return "region", engine
    if model in ("basicvsr++", "basicvsrpp", "basicvsr"):
        from .vsr.basicvsr import BasicVSRPlusPlus
        engine = BasicVSRPlusPlus(
            device=device, model_dir=config.MODEL_DIR,
            tile_size=480 if max(info.width, info.height) > 1920 else 0,
            window=config.BATCH_SIZE if config.BATCH_SIZE >= 2 else 5,
        )
        return "frame", engine
    raise ValueError(f"unknown VSR model: {model}")


def _vsr_disabled(model: str) -> bool:
    return model in ("off", "none", "")


def process_video(
    input_path: str,
    output_path: str,
    *,
    detection: str = "auto",
    ocr_engine: str = "easyocr",
    roi_spec: Optional[str] = None,
    inpaint: str = "opencv",
    vsr: str = "off",
    progress_cb: Optional[Callable[[float], None]] = None,
) -> VideoInfo:
    info = probe_video(input_path)
    logger.info("probed %s: %dx%d @ %.2ffps, %d frames, audio=%s",
                input_path, info.width, info.height, info.fps,
                info.n_frames, info.has_audio)

    if info.width > config.MAX_WIDTH or info.height > config.MAX_HEIGHT:
        raise ValueError(
            f"resolution {info.width}x{info.height} exceeds max "
            f"{config.MAX_WIDTH}x{config.MAX_HEIGHT}")
    if info.duration / 60.0 > config.MAX_DURATION_MIN:
        raise ValueError(f"duration {info.duration / 60:.1f}min exceeds max")

    device = config.DEVICE
    roi = _resolve_roi(input_path, info, detection, roi_spec, device, ocr_engine)
    logger.info("detection=%s(engine=%s) → roi=%s, inpaint=%s, vsr=%s",
                detection, ocr_engine, roi, inpaint, vsr)

    vace_mode = is_vace_method(inpaint)
    if vace_mode and not _vsr_disabled(vsr):
        raise ValueError("VACE POC does not support VSR; use vsr=off")

    if not vace_mode:
        mask = build_mask(info.width, info.height, roi)
        inpaint_fn = _make_inpaint_fn(inpaint, device)
        vsr_kind, vsr_engine = _make_vsr(vsr, device, info)
    else:
        mask = None
        inpaint_fn = None
        vsr_kind, vsr_engine = "off", None

    workdir = tempfile.mkdtemp(prefix="sr_")
    try:
        audio_path = os.path.join(workdir, "audio.aac")
        if info.has_audio:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", input_path,
                 "-vn", "-acodec", "copy", audio_path], check=True)

        video_only = os.path.join(workdir, "video.mp4")
        if vace_mode:
            VaceSubtitleRemover().remove(input_path, video_only, roi, info, progress_cb)
        else:
            decode = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-i", input_path,
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                stdout=subprocess.PIPE, bufsize=10**8)

            # When BasicVSR++ runs on whole frames at x4, output dims grow.
            out_w, out_h = info.width, info.height
            upscale = 1
            if vsr_kind == "frame":
                upscale = vsr_engine.upscale_factor
                out_w *= upscale
                out_h *= upscale
                logger.info("BasicVSR++ frame mode: output upscaled to %dx%d",
                            out_w, out_h)

            encode = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{out_w}x{out_h}", "-r", str(info.fps or 25), "-i", "-",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
                 video_only],
                stdin=subprocess.PIPE, bufsize=10**8)

            frame_bytes = info.width * info.height * 3
            idx = 0
            # for BasicVSR++ we must batch frames; for region/off we go frame-by-frame.
            try:
                if vsr_kind == "frame":
                    _process_frame_vsr(decode, encode, info, frame_bytes, mask,
                                       inpaint_fn, vsr_engine, progress_cb)
                else:
                    while True:
                        buf = decode.stdout.read(frame_bytes)
                        if len(buf) < frame_bytes:
                            break
                        frame = np.frombuffer(buf, dtype=np.uint8).reshape(
                            info.height, info.width, 3).copy()
                        cleaned = inpaint_fn(frame, mask)
                        if vsr_kind == "region":
                            cleaned = vsr_engine.enhance_region(cleaned, roi)
                        encode.stdin.write(cleaned.tobytes())
                        idx += 1
                        if progress_cb and info.n_frames and idx % 30 == 0:
                            progress_cb(min(idx / info.n_frames, 0.99))
            finally:
                if encode.stdin:
                    encode.stdin.close()
                decode.wait()
                encode.wait()

            if encode.returncode != 0:
                raise RuntimeError(f"encode failed (rc={encode.returncode})")

        if info.has_audio:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-i", video_only, "-i", audio_path,
                 "-c:v", "copy", "-c:a", "aac", "-shortest", output_path],
                check=True)
        else:
            shutil.move(video_only, output_path)

        if progress_cb:
            progress_cb(1.0)
        return info
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _process_frame_vsr(decode, encode, info, frame_bytes, mask, inpaint_fn,
                       vsr_engine, progress_cb):
    """BasicVSR++ branch: buffer windows, run network, write enhanced frames."""
    # mask is at original res; VSR upscales, so we upscale mask only once.
    s = vsr_engine.upscale_factor
    win = vsr_engine.window
    buf: List[np.ndarray] = []
    written = 0
    while True:
        b = decode.stdout.read(frame_bytes)
        if len(b) < frame_bytes:
            break
        frame = np.frombuffer(b, dtype=np.uint8).reshape(
            info.height, info.width, 3).copy()
        cleaned = inpaint_fn(frame, mask)
        buf.append(cleaned)
        if len(buf) >= win:
            enhanced = vsr_engine.enhance(buf)
            for ef in enhanced:
                encode.stdin.write(ef.tobytes())
            written += len(enhanced)
            buf.clear()
            if progress_cb and info.n_frames:
                progress_cb(min(written / info.n_frames, 0.99))
    if buf:
        enhanced = vsr_engine.enhance(buf)
        for ef in enhanced:
            encode.stdin.write(ef.tobytes())
