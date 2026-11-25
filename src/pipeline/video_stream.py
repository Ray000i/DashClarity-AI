"""
DashClarity AI - Streaming Video Engine
Frame-by-frame streaming processing with direct FFmpeg stdin piping.
Preserves FPS, duration, audio synchronization, and never loads entire videos into RAM.
"""

import os
import cv2
import time
import shutil
import logging
import subprocess
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple

from src.utils import get_ffmpeg_binary

logger = logging.getLogger("DashClarity.Pipeline.VideoStream")


class StreamingVideoProcessor:
    """
    Manages low-memory streaming enhancement.
    Reads frames with cv2.VideoCapture and writes directly to an active FFmpeg sub-process pipe.
    """
    def __init__(
        self,
        input_path: str,
        output_path: str,
        max_seconds: Optional[float] = None,
        frame_skip: int = 1
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_seconds = max_seconds
        self.frame_skip = max(1, frame_skip)

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input video file not found: {self.input_path}")

        self.cap = cv2.VideoCapture(str(self.input_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open input video container: {self.input_path}")

        self.in_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.in_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0 or self.fps > 240:
            self.fps = 30.0

        self.total_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.max_seconds and self.max_seconds > 0:
            max_limit = int(self.max_seconds * self.fps)
            target = min(self.total_video_frames, max_limit) // self.frame_skip
        else:
            target = self.total_video_frames // self.frame_skip
        self.total_target_frames = max(1, target)

        self.ffmpeg_bin = get_ffmpeg_binary()
        self.audio_temp_path = self.output_path.with_suffix(".temp_audio.aac")
        self.audio_extracted = False
        self.audio_warning = None

    def extract_audio(self) -> bool:
        """
        Extracts original audio track for final multiplexing.
        Catches specific Subprocess/OS errors and logs clearly without silent bare excepts.
        """
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i", str(self.input_path),
            "-vn",
            "-acodec", "aac",
            "-b:a", "192k",
            str(self.audio_temp_path)
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
            if res.returncode == 0 and self.audio_temp_path.exists() and self.audio_temp_path.stat().st_size > 100:
                self.audio_extracted = True
                logger.info("Audio stream successfully extracted to %s", self.audio_temp_path)
                return True
            else:
                self.audio_warning = "Video contains no audio stream or audio format is unsupported."
                logger.info("No audio extracted: %s", self.audio_warning)
                return False
        except subprocess.TimeoutExpired:
            self.audio_warning = "Audio extraction timed out after 45 seconds."
            logger.warning(self.audio_warning)
            return False
        except (subprocess.SubprocessError, OSError) as e:
            self.audio_warning = f"Audio extraction failed: {str(e)}"
            logger.warning(self.audio_warning)
            return False

    def process(
        self,
        frame_transform_fn: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, Dict[str, Any]]],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Streams frames through the enhancement function directly into FFmpeg stdin.
        """
        # 1. Extract audio in background
        self.extract_audio()

        # Read first frame to determine enhanced output dimensions
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = self.cap.read()
        if not ret or first_frame is None:
            raise ValueError("Unable to read first video frame.")

        test_enh, _ = frame_transform_fn(first_frame, None)
        raw_out_h, raw_out_w = test_enh.shape[:2]
        # Ensure dimensions are even numbers for libx264 yuv420p
        out_w = (raw_out_w // 2) * 2
        out_h = (raw_out_h // 2) * 2
        if raw_out_w != out_w or raw_out_h != out_h:
            test_enh = cv2.resize(test_enh, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        temp_video_out = self.output_path.with_suffix(".temp_enh.mp4")
        ffmpeg_log_path = self.output_path.with_suffix(".ffmpeg.log")
        ffmpeg_log_file = open(ffmpeg_log_path, "w", encoding="utf-8")

        # 2. Launch FFmpeg sub-process reading raw BGR24 frames from stdin pipe
        ffmpeg_cmd = [
            self.ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{out_w}x{out_h}",
            "-pix_fmt", "bgr24",
            "-r", str(self.fps / self.frame_skip),
            "-i", "-",  # Read from pipe
            # Convert RGB->YUV with the BT.709 matrix and tag the stream accordingly. Without this
            # FFmpeg uses BT.601 and leaves the file untagged, so players (which assume BT.709 for
            # HD) shift the colours of the enhanced video.
            "-vf", "scale=out_color_matrix=bt709:out_range=tv",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-colorspace", "bt709",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-color_range", "tv",
            "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709",
            "-movflags", "+faststart",
            str(temp_video_out)
        ]

        logger.info("Starting FFmpeg stream: %s", " ".join(ffmpeg_cmd))
        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=ffmpeg_log_file
        )

        # 3. Rolling frame queue (holds at most 5 frames in memory for STCD or temporal window)
        window = deque(maxlen=5)
        window.append(first_frame)

        sample_orig_frames = [first_frame]
        sample_enh_frames = [test_enh]

        frames_written = 0
        t_start = time.time()
        sample_interval = max(1, self.total_target_frames // 10)

        try:
            # Process first frame
            proc.stdin.write(test_enh.tobytes())
            frames_written += 1

            raw_idx = 0
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break
                raw_idx += 1

                if self.frame_skip > 1 and (raw_idx % self.frame_skip != 0):
                    continue

                if self.max_seconds and (frames_written >= self.total_target_frames):
                    break

                prev_frame = window[-1] if len(window) > 0 else frame
                window.append(frame)

                # Execute pipeline enhancement
                enhanced, _ = frame_transform_fn(frame, prev_frame)

                # Ensure dimensions match out_w x out_h exactly
                if enhanced.shape[1] != out_w or enhanced.shape[0] != out_h:
                    enhanced = cv2.resize(enhanced, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

                # Write directly to FFmpeg pipe
                proc.stdin.write(enhanced.tobytes())
                frames_written += 1

                # Sample frames for metric calculations
                if frames_written % sample_interval == 0:
                    sample_orig_frames.append(frame)
                    sample_enh_frames.append(enhanced)

                # Telemetry update
                elapsed = time.time() - t_start
                fps = frames_written / elapsed if elapsed > 0 else 0.0
                remaining = max(0, self.total_target_frames - frames_written)
                eta = remaining / fps if fps > 0 else 0.0
                pct = min(99.0, (frames_written / max(1, self.total_target_frames)) * 100.0)

                if progress_callback:
                    progress_callback({
                        "frames_processed": frames_written,
                        "total_frames": self.total_target_frames,
                        "fps": round(fps, 2),
                        "eta_seconds": round(eta, 1),
                        "elapsed_seconds": round(elapsed, 1),
                        "progress_percent": round(pct, 1),
                        "stage": f"Enhancing frame {frames_written}/{self.total_target_frames} ({fps:.1f} FPS)"
                    })

            proc.stdin.close()
            proc.wait(timeout=60)
            ffmpeg_log_file.close()
            if ffmpeg_log_path.exists():
                try:
                    ffmpeg_log_path.unlink()
                except OSError:
                    pass
        except BaseException as e:
            # BaseException: a user cancel arrives as KeyboardInterrupt from the progress callback.
            # The encoder must be stopped and the partial files removed in every case; previously a
            # cancel skipped this block, leaving FFmpeg running and .temp_* files on disk.
            if proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            ffmpeg_log_file.close()
            err_details = ""
            if ffmpeg_log_path.exists():
                try:
                    with open(ffmpeg_log_path, "r", encoding="utf-8") as f:
                        err_details = f.read()[-500:]
                except Exception:
                    pass
            for partial in (temp_video_out, self.audio_temp_path, ffmpeg_log_path):
                try:
                    partial.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove partial file %s", partial)
            if isinstance(e, KeyboardInterrupt):
                raise
            raise RuntimeError(f"Streaming enhancement pipeline failed: {e}. FFmpeg output: {err_details}")
        finally:
            self.cap.release()

        # 4. Final mux. The audio is cut to the length of the enhanced video: when only part of
        # the clip is processed (max_seconds) the full-length track would otherwise extend the
        # file and freeze the last frame. The original track is stream-copied when the
        # container allows it (no second lossy encode); otherwise the extracted AAC is used.
        video_duration = frames_written / (self.fps / self.frame_skip)
        mux_res = None
        if self.audio_extracted:
            attempts = [
                ["-i", str(self.input_path), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy"],
                ["-i", str(self.audio_temp_path), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k"],
            ]
            for extra in attempts:
                cmd = [self.ffmpeg_bin, "-y", "-i", str(temp_video_out), *extra,
                       "-t", f"{video_duration:.3f}", "-shortest", "-movflags", "+faststart", str(self.output_path)]
                mux_res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if mux_res.returncode == 0:
                    break
        if mux_res is None or mux_res.returncode != 0:
            if mux_res is not None:
                self.audio_warning = "Audio could not be multiplexed; the enhanced video was saved without audio."
                logger.warning("%s FFmpeg: %s", self.audio_warning, mux_res.stderr.decode("utf-8", "ignore")[-300:])
            mux_res = subprocess.run([self.ffmpeg_bin, "-y", "-i", str(temp_video_out), "-c:v", "copy",
                                      "-movflags", "+faststart", str(self.output_path)],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if mux_res.returncode != 0:
                shutil.move(str(temp_video_out), str(self.output_path))

        # Cleanup temp video and audio
        if temp_video_out.exists():
            temp_video_out.unlink()
        if self.audio_temp_path.exists():
            self.audio_temp_path.unlink()

        total_time = time.time() - t_start
        return {
            "sample_orig_frames": sample_orig_frames,
            "sample_enh_frames": sample_enh_frames,
            "total_frames_processed": frames_written,
            "total_time_seconds": total_time,
            "processing_fps": frames_written / total_time if total_time > 0 else 0.0,
            "audio_warning": self.audio_warning,
            "out_width": out_w,
            "out_height": out_h
        }
