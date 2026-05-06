"""Global configuration for SubtitleRemover."""

import os

# Device
DEVICE = os.getenv("SR_DEVICE", "cuda")
GPU_MEMORY_FRACTION = float(os.getenv("SR_GPU_FRACTION", "0.4"))  # leave room for audio separator

# Video
MAX_RESOLUTION = int(os.getenv("SR_MAX_RESOLUTION", "1920"))  # max width
BATCH_SIZE = int(os.getenv("SR_BATCH_SIZE", "4"))

# Detection
DETECTION_MODE = os.getenv("SR_DETECTION_MODE", "auto")  # auto / roi / ocr
SUBTITLE_ROI = os.getenv("SR_SUBTITLE_ROI", "bottom_20%")  # default: bottom 20%

# Inpainting
INPAINT_METHOD = os.getenv("SR_INPAINT_METHOD", "opencv")  # opencv / lama

# VSR
VSR_MODEL = os.getenv("SR_VSR_MODEL", "basicvsr++")
VSR_UPSCALE_FACTOR = int(os.getenv("SR_VSR_UPSCALE", "2"))
MODEL_DIR = os.getenv("SR_MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))

# API
SERVICE_PORT = int(os.getenv("SR_PORT", "8082"))
UPLOAD_DIR = os.getenv("SR_UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "..", "uploads"))
OUTPUT_DIR = os.getenv("SR_OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "..", "output"))

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
