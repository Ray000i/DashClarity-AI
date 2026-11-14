"""
DashClarity AI - Video Reconstruction Module
Reconstructs enhanced frames into a high-compatibility MP4 video container.
Preserves FPS, frame sequence, aspect ratio, and remuxes original audio.
"""

import os
import cv2
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Optional, List
from src.utils import get_ffmpeg_binary

logger = logging.getLogger("DashClarity.Reconstructor")

class VideoReconstructor:
    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        audio_path: Optional[str] = None
    ):
        self.output_path = Path(output_path)
        self.fps = fps if fps > 0 else 30.0
        self.width = width
        self.height = height
        self.audio_path = Path(audio_path) if audio_path and os.path.exists(audio_path) else None
        self.ffmpeg_bin = get_ffmpeg_binary()

    def assemble_from_frame_dir(self, frames_dir: str, frame_pattern: str = "frame_%06d.png") -> bool:
        """
        Uses FFmpeg to stitch frames from a directory into a browser-ready MP4.
        
        Args:
            frames_dir: Directory containing ordered PNG/JPG frames.
            frame_pattern: Formatting pattern for frames (e.g. frame_%06d.png).
            
        Returns:
            True if successful.
        """
        frames_path = Path(frames_dir) / frame_pattern
        temp_video_path = self.output_path.with_suffix(".temp.mp4")
        
        # Primary FFmpeg command for H.264 video encoding
        # -pix_fmt yuv420p is mandatory for HTML5 browser video playback
        # -movflags +faststart moves the moov atom to beginning of file for instant web playback
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-framerate", str(self.fps),
            "-i", str(frames_path),
        ]
        
        if self.audio_path and self.audio_path.exists():
            cmd.extend([
                "-i", str(self.audio_path),
                "-c:a", "aac",
                "-b:a", "192k"
            ])
            
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",  # Visually lossless/high quality
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(self.output_path)
        ])
        
        logger.info(f"Running video assembly FFmpeg command: {' '.join(cmd)}")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
        
        if res.returncode != 0:
            logger.error(f"FFmpeg assembly failed: {res.stderr}")
            # Fallback attempt without audio if audio mux failed
            if self.audio_path:
                logger.info("Attempting video-only fallback assembly...")
                cmd_fallback = [
                    self.ffmpeg_bin,
                    "-y",
                    "-framerate", str(self.fps),
                    "-i", str(frames_path),
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(self.output_path)
                ]
                res_fb = subprocess.run(cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
                if res_fb.returncode == 0:
                    return True
            raise RuntimeError(f"FFmpeg reconstruction failed with code {res.returncode}: {res.stderr}")
            
        return True

