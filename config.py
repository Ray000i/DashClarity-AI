"""
DashClarity AI - Configuration Module
Centralized settings for file management, hardware, model paths, logging, and processing parameters.
"""

import os
import secrets
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Base Project Root
BASE_DIR = Path(__file__).resolve().parent

# Runtime Storage Paths
TEST_MODE = os.environ.get("TEST_MODE", "").lower() in ("1", "true", "yes")

if TEST_MODE:
    import tempfile
    _test_storage = Path(tempfile.gettempdir()) / "dashclarity_test_storage"
    UPLOAD_DIR = _test_storage / "input_videos"
    OUTPUT_DIR = _test_storage / "output_videos"
    LOGS_DIR = _test_storage / "logs"
else:
    UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", str(BASE_DIR / "input_videos")))
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(BASE_DIR / "output_videos")))
    LOGS_DIR = Path(os.environ.get("LOGS_DIR", str(BASE_DIR / "logs")))

# Ensure runtime directories exist
for folder in [UPLOAD_DIR, OUTPUT_DIR, LOGS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

def set_test_mode_dir(temp_dir: Path):
    """Dynamically redirects storage paths for unit test isolation."""
    global UPLOAD_DIR, OUTPUT_DIR, LOGS_DIR, TEST_MODE
    TEST_MODE = True
    UPLOAD_DIR = temp_dir / "input_videos"
    OUTPUT_DIR = temp_dir / "output_videos"
    LOGS_DIR = temp_dir / "logs"
    for folder in [UPLOAD_DIR, OUTPUT_DIR, LOGS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

# Model & Architecture Directories
WEIGHTS_BASE_DIR = BASE_DIR / "weights"
MODELS_DIR = BASE_DIR / "models" / "STCD"
STCD_CHECKPOINT_PATH = WEIGHTS_BASE_DIR / "STCD" / "indoor_G.pth"

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
DEFAULT_MAX_FRAMES = 900  # Default limit (~30s at 30fps)

# Processing Defaults
DEVICE = os.environ.get("DEVICE", "cuda")
DEFAULT_BATCH_SIZE = 1
MAX_VIDEO_DIMENSION = 1920

# Flask App Configuration
_env_secret = os.environ.get("SECRET_KEY")
if not _env_secret:
    # Generate ephemeral secure key if not explicitly set in environment
    SECRET_KEY = secrets.token_hex(32)
else:
    SECRET_KEY = _env_secret

DEBUG = os.environ.get("FLASK_DEBUG", os.environ.get("DEBUG", "0")).lower() in ("1", "true", "yes")
PORT = int(os.environ.get("PORT", "5000"))
HOST = os.environ.get("HOST", "127.0.0.1")

# Configure Structured Logging with Rotation
def setup_logging(log_filename: str = "dashclarity.log", log_level: int = logging.INFO):
    """Configures root logger with console and rotating file handler."""
    log_file = LOGS_DIR / log_filename
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers on re-initialization
    if not any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        root_logger.addHandler(console_handler)

setup_logging()
