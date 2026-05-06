"""Global configuration for SubtitleRemover."""

import os

# Device
DEVICE = os.getenv("SR_DEVICE", "cuda")
GPU_MEMORY_FRACTION = float(os.getenv("SR_GPU_FRACTION", "0.4"))  # leave room for audio separator

# Video — support 1080p and 2K
MAX_WIDTH = int(os.getenv("SR_MAX_WIDTH", "2560"))      # max input width (1080p=1920, 2K=2560)
MAX_HEIGHT = int(os.getenv("SR_MAX_HEIGHT", "1440"))     # max input height
MAX_DURATION_MIN = int(os.getenv("SR_MAX_DURATION", "60"))  # max video duration in minutes
SUPPORTED_INPUTS = os.getenv("SR_INPUT_FORMATS", "mp4,mkv,avi,mov,webm").split(",")
BATCH_SIZE = int(os.getenv("SR_BATCH_SIZE", "4"))        # VSR batch size (reduce for 2K)

# Detection
DETECTION_MODE = os.getenv("SR_DETECTION_MODE", "auto")  # auto / roi / ocr
SUBTITLE_ROI = os.getenv("SR_SUBTITLE_ROI", "bottom_20%")  # default: bottom 20%

# Inpainting
INPAINT_METHOD = os.getenv("SR_INPAINT_METHOD", "opencv")  # opencv / lama

# VSR
VSR_MODEL = os.getenv("SR_VSR_MODEL", "basicvsr++")
VSR_UPSCALE_FACTOR = int(os.getenv("SR_VSR_UPSCALE", "2"))  # upscale inpainted region only
VSR_TILE_SIZE = int(os.getenv("SR_VSR_TILE_SIZE", "0"))    # tile processing for 2K (0=disabled)
VSR_TILE_PAD = int(os.getenv("SR_VSR_TILE_PAD", "10"))
MODEL_DIR = os.getenv("SR_MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))

# API
SERVICE_PORT = int(os.getenv("SR_PORT", "8082"))
UPLOAD_DIR = os.getenv("SR_UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "..", "uploads"))
OUTPUT_DIR = os.getenv("SR_OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "..", "output"))

# VACE — see docs/vace_integration_plan.md
def _envflag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")

VACE_ENABLED = _envflag("SR_VACE_ENABLED", "0")
VACE_DRY_RUN = _envflag("SR_VACE_DRY_RUN", "1")
VACE_PROFILE = os.getenv("SR_VACE_PROFILE", "rtx4070tis_balanced")
VACE_GPU_FRACTION = float(os.getenv("SR_VACE_GPU_FRACTION", "0.9"))
VACE_MODEL_DIR = os.getenv(
    "SR_VACE_MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "..", "models", "vace"),
)
VACE_GPU_LOCK_ENABLED = _envflag("SR_GPU_LOCK_ENABLED", "0")
VACE_GPU_LOCK_FILE = os.getenv("SR_GPU_LOCK_FILE", "/tmp/gpu.lock")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VACE_MODEL_DIR, exist_ok=True)
