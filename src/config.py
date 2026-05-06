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

# Experimental VACE integration. VACE runs in a separate checkout/venv; this
# service only prepares inputs and invokes the external CLI.
VACE_PYTHON = os.getenv("SR_VACE_PYTHON", "python")
VACE_SCRIPT = os.getenv("SR_VACE_SCRIPT", "")
VACE_CKPT_DIR = os.getenv("SR_VACE_CKPT_DIR", "")
VACE_MODEL_NAME = os.getenv("SR_VACE_MODEL_NAME", "vace-1.3B")
VACE_SIZE = os.getenv("SR_VACE_SIZE", "480p")
VACE_FRAME_NUM = int(os.getenv("SR_VACE_FRAME_NUM", "81"))
VACE_PROMPT = os.getenv(
    "SR_VACE_PROMPT",
    "Remove the subtitles and reconstruct the background naturally.",
)
VACE_SAMPLE_STEPS = int(os.getenv("SR_VACE_SAMPLE_STEPS", "30"))
VACE_OFFLOAD_MODEL = os.getenv("SR_VACE_OFFLOAD_MODEL", "true")
VACE_T5_CPU = os.getenv("SR_VACE_T5_CPU", "true")
VACE_MAX_DURATION_SEC = float(os.getenv("SR_VACE_MAX_DURATION_SEC", "6"))
VACE_TIMEOUT_SEC = int(os.getenv("SR_VACE_TIMEOUT_SEC", "3600"))

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

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
