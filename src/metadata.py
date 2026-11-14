"""
DashClarity AI - Video Metadata Extraction Module
Extracts video container and stream properties using OpenCV and FFmpeg.
"""

import os
import cv2
import json
import subprocess
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from src.utils import get_ffmpeg_binary

logger = logging.getLogger("DashClarity.Metadata")

def format_file_size(size_bytes: int) -> str:
    """Format bytes into human-readable string (KB, MB, GB)."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def format_duration(seconds: float) -> str:
    """Format duration in seconds to MM:SS or HH:MM:SS."""
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"

def extract_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Extracts comprehensive metadata from a video file.
    
    Args:
        video_path: Path to the input video file.
        
    Returns:
        Dictionary containing metadata properties.
    """
    path_obj = Path(video_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")
        
    file_size_bytes = path_obj.stat().st_size
    formatted_size = format_file_size(file_size_bytes)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video container: {video_path}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Handle potentially missing or variable FPS values
    if fps <= 0 or fps > 240:
        fps = 30.0
        
    duration_sec = total_frames / fps if fps > 0 and total_frames > 0 else 0.0
    aspect_ratio = f"{width}:{height}"
    if height > 0:
        # Simplify common aspect ratios
        ratio_val = width / height
        if abs(ratio_val - (16/9)) < 0.05:
            aspect_ratio = "16:9"
        elif abs(ratio_val - (4/3)) < 0.05:
            aspect_ratio = "4:3"
        elif abs(ratio_val - (21/9)) < 0.05:
            aspect_ratio = "21:9"
        else:
            aspect_ratio = f"{ratio_val:.2f}:1"
            
    # Probe audio stream existence using FFmpeg
    has_audio = check_audio_stream(video_path)
    
    return {
        "filename": path_obj.name,
        "filepath": str(path_obj.resolve()),
        "file_size_bytes": file_size_bytes,
        "file_size_formatted": formatted_size,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "aspect_ratio": aspect_ratio,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_seconds": round(duration_sec, 2),
        "duration_formatted": format_duration(duration_sec),
        "has_audio": has_audio,
        "codec": "h264 / mp4v"
    }

def check_audio_stream(video_path: str) -> bool:
    """
    Checks if the video file contains an audio stream using FFmpeg.
    """
    try:
        ffmpeg_bin = get_ffmpeg_binary()
        cmd = [
            ffmpeg_bin,
            "-i", str(video_path),
            "-vn",
            "-af", "astats",
            "-f", "null",
            "-"
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=8
        )
        # Check stderr for Audio stream indicators
        output = result.stderr.lower()
        if "audio:" in output or "stream #0:1" in output or "stream #0:0(eng): audio" in output:
            return True
        return False
    except Exception as e:
        logger.warning(f"Audio probe check non-fatal exception: {e}")
        return False
