"""
DashClarity AI - Configuration Module
Centralized settings for file management, hardware, model paths, and processing parameters.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights" / "STCD"
MODELS_DIR = BASE_DIR / "models" / "STCD"
UPLOAD_DIR = BASE_DIR / "input_videos"
OUTPUT_DIR = BASE_DIR / "output_videos"
LOGS_DIR = BASE_DIR / "logs"

# Ensure runtime directories exist
for folder in [UPLOAD_DIR, OUTPUT_DIR, LOGS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Pretrained Model Checkpoint
STCD_CHECKPOINT_PATH = WEIGHTS_DIR / "indoor_G.pth"

# Model Architecture Settings (matching IJCAI STCD specifications)
MODEL_CONFIG = {
    "which_model_G": "low_light_transformer42",
    "nf": 64,
    "nframes": 5,
    "groups": 8,
    "front_RBs": 1,
    "back_RBs": 1,
    "center": None,
    "predeblur": True,
    "HR_in": True,
    "w_TSA": True,
}

# Upload Constraints
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
MAX_CONTENT_LENGTH = 250 * 1024 * 1024  # 250 MB limit
DEFAULT_MAX_FRAMES = 900  # Default limit for responsive demo (~30s at 30fps)

# Processing Defaults
DEVICE = "cuda"  # fallback to 'cpu' handled automatically in engine
DEFAULT_BATCH_SIZE = 1  # Sequential temporal pairs for STCD cross-frame attention
MAX_VIDEO_DIMENSION = 1920  # Max dimension supported before automatic safe scaling

# Flask App Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "dashclarity-ai-academic-secret-key-2026")
DEBUG = False
PORT = 5000
HOST = "127.0.0.1"

