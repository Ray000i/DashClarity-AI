"""
DashClarity AI - Utility Functions and Subpackage Root
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

def ensure_web_compatible_video(video_path: str) -> str:
    """
    Ensures the video is encoded in an HTML5-compliant container and codec (H.264/avc1 + yuv420p + faststart).
    If the file is already compliant, returns video_path directly.
    Otherwise, transcodes a cached web version and returns the cached path.
    """
    p = Path(video_path)
    if not p.exists():
        return video_path
    
    ffmpeg = get_ffmpeg_binary()
    try:
        import subprocess
        res = subprocess.run([ffmpeg, '-i', str(p)], capture_output=True, text=True, timeout=5)
        err = res.stderr.lower()
        if ('video: h264' in err or 'video: avc1' in err) and 'yuv420p' in err:
            return str(p)
    except Exception as e:
        logger.warning(f"Failed to probe video codec for {video_path}: {e}")
        return str(p)

    # Need web-compatible transcode
    web_dir = p.parent / ".web_cache"
    web_dir.mkdir(parents=True, exist_ok=True)
    web_path = web_dir / f"{p.stem}_web.mp4"

    if web_path.exists() and web_path.stat().st_size > 1000:
        return str(web_path)

    try:
        import subprocess
        cmd = [
            ffmpeg, '-y',
            '-i', str(p),
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            str(web_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        logger.info(f"Transcoded {p.name} to web-compatible H.264: {web_path.name}")
        return str(web_path)
    except Exception as e:
        logger.warning(f"Could not transcode {video_path} to web H.264: {e}")
        return str(p)

