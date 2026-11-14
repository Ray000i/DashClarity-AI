"""
DashClarity AI - Video Loader Module
Reads frames sequentially and handles audio extraction for reconstruction.
"""

import os
import cv2
import subprocess
import logging
from typing import Generator, Tuple, Optional
from pathlib import Path
from src.utils import get_ffmpeg_binary

logger = logging.getLogger("DashClarity.VideoLoader")

class VideoLoader:
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")
            
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0 or self.fps > 240:
            self.fps = 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame_count(self) -> int:
        return self.total_frames

    def get_fps(self) -> float:
        return self.fps

    def get_dimensions(self) -> Tuple[int, int]:
        return self.width, self.height

    def stream_frames(self, max_frames: Optional[int] = None) -> Generator[Tuple[int, cv2.Mat], None, None]:
        """
        Yields (frame_index, frame_bgr) sequentially.
        Ensures resources are cleaned up on finish or generator exit.
        """
        frame_idx = 0
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                if max_frames and frame_idx >= max_frames:
                    break
                    
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break
                    
                yield frame_idx, frame
                frame_idx += 1
        finally:
            self.cap.release()

    def extract_audio(self, output_audio_path: str) -> bool:
        """
        Extracts the audio track from the video file into an AAC/M4A audio file.
        Returns True if successful, False if no audio exists or extraction failed.
        """
        try:
            ffmpeg_bin = get_ffmpeg_binary()
            cmd = [
                ffmpeg_bin,
                "-y",  # Overwrite
                "-i", str(self.video_path),
                "-vn", # Disable video
                "-acodec", "copy",
                str(output_audio_path)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            if res.returncode == 0 and os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 100:
                return True
                
            # If copy failed (e.g. incompatible audio format for container), try re-encoding to aac
            cmd_reencode = [
                ffmpeg_bin,
                "-y",
                "-i", str(self.video_path),
                "-vn",
                "-acodec", "aac",
                "-b:a", "192k",
                str(output_audio_path)
            ]
            res2 = subprocess.run(cmd_reencode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            return res2.returncode == 0 and os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 100
        except Exception as e:
            logger.warning(f"Audio extraction skipped or failed: {e}")
            return False

    def close(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
