"""
DashClarity AI - Utility Functions
Handles FFmpeg path resolution, secure ID generation, and thread-safe job management.
"""

import os
import uuid
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from werkzeug.utils import secure_filename

# Logger setup
logger = logging.getLogger("DashClarity.Utils")

def get_ffmpeg_binary() -> str:
    """
    Locates the FFmpeg binary.
    Prioritizes bundled imageio_ffmpeg executable, then PATH.
    """
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_exe and os.path.exists(ffmpeg_exe):
            return ffmpeg_exe
    except ImportError:
        pass
    
    # Fallback to system PATH
    import shutil
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
        
    raise RuntimeError("FFmpeg executable could not be found via imageio_ffmpeg or system PATH.")

def generate_job_id() -> str:
    """Generate a unique job ID for processing isolation."""
    return uuid.uuid4().hex[:12]

def sanitize_filename(filename: str) -> str:
    """Produce a secure ASCII-safe filename with original extension preserved."""
    safe_name = secure_filename(filename)
    if not safe_name:
        safe_name = f"upload_{uuid.uuid4().hex[:8]}.mp4"
    return safe_name

def is_allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Check if the uploaded file has a permitted video extension."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in allowed_extensions

