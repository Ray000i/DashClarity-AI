"""
DashClarity AI - Video Processor Coordinator
Manages asynchronous enhancement jobs, thread-safe progress reporting,
real-time execution logs, side-by-side comparison rendering, job history, and single-GPU queueing.
"""

import os
import json
import time
import shutil
import logging
import threading
import subprocess
import queue
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np
import torch

import config
from src.pipeline.enhancer import VideoEnhancementPipeline
from src.pipeline.video_stream import StreamingVideoProcessor
from src.utils.metrics import evaluate_video_metrics
from src.utils import get_ffmpeg_binary

logger = logging.getLogger("DashClarity.Processor")

TEST_JOB_PREFIXES = ("test_", "sample", "clip_test", "night_upload", "cli_test", "dummy")


def is_test_job(job_id: str, clip_name: str) -> bool:
    jid = job_id.lower()
    cname = clip_name.lower()
    return any(
        jid.startswith(p) or cname.startswith(p) or f"_{p}" in jid or f"_{p}" in cname
        for p in TEST_JOB_PREFIXES
    )


class JobState:
    """Thread-safe representation of an enhancement job's progress and state."""
    def __init__(self, job_id: str, input_path: str):
        self.job_id = job_id
        self.input_path = input_path
        self.output_path = None
        self.comparison_path = None
        self.status = "ready"  # ready, queued, initializing, processing_frames, reconstructing, completed, failed, cancelled
        self.current_stage = "Uploaded and ready for enhancement"
        self.progress_percent = 0.0
        self.frames_processed = 0
        self.total_frames = 0
        self.current_fps = 0.0
        self.eta_seconds = 0.0
        self.elapsed_seconds = 0.0
        self.start_time = 0.0
        self.error_message = None
        self.audio_warning = None
        self.cancel_requested = False
        self.queue_position = 0
        self.metrics = {}
        self.logs: List[str] = []
        self.lock = threading.Lock()
        self.add_log("Job initialized and ready.")

    def add_log(self, message: str):
        with self.lock:
            ts = time.strftime("%H:%M:%S")
            self.logs.append(f"[{ts}] {message}")
            if len(self.logs) > 300:
                self.logs = self.logs[-300:]

    def update(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "job_id": self.job_id,
                "status": self.status,
                "stage": self.current_stage,
                "progress_percent": round(self.progress_percent, 1),
                "frames_processed": self.frames_processed,
                "total_frames": self.total_frames,
                "current_fps": round(self.current_fps, 2),
                "eta_seconds": round(self.eta_seconds, 1),
                "elapsed_seconds": round(self.elapsed_seconds, 1),
                "queue_position": self.queue_position,
                "error": self.error_message,
                "audio_warning": self.audio_warning,
                "metrics": self.metrics,
                "logs": list(self.logs),
                "has_enhanced_video": self.output_path is not None and os.path.exists(self.output_path),
                "has_comparison_video": self.comparison_path is not None and os.path.exists(self.comparison_path)
            }


class ProcessingManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.jobs: Dict[str, JobState] = {}
        self.jobs_lock = threading.Lock()
        self.pipeline = None

        # GPU Job Queue for single-GPU safe serial processing
        self.job_queue: queue.Queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.active_job_id: Optional[str] = None
        self._shutdown_event = threading.Event()

        # Clean up any leftover temporary files from prior interrupted runs
        self._cleanup_interrupted_artifacts()

    def _cleanup_interrupted_artifacts(self):
        """Cleans orphaned .temp_* files left by interrupted processes or server restarts."""
        try:
            for temp_file in config.OUTPUT_DIR.glob("*.temp_*"):
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            for log_file in config.OUTPUT_DIR.glob("*.ffmpeg.log"):
                try:
                    log_file.unlink()
                except OSError:
                    pass
        except Exception as e:
            logger.warning("Error cleaning temporary artifacts: %s", e)

    def get_pipeline(self) -> VideoEnhancementPipeline:
        if self.pipeline is None:
            self.pipeline = VideoEnhancementPipeline()
        return self.pipeline

    def get_job(self, job_id: str) -> Optional[JobState]:
        with self.jobs_lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                # Check for dead worker thread while job was supposed to be running
                if (
                    job.status in ["initializing", "processing_frames", "reconstructing"]
                    and (self.worker_thread is None or not self.worker_thread.is_alive())
                    and self.active_job_id != job_id
                ):
                    job.update(
                        status="failed",
                        error_message="Worker thread terminated unexpectedly.",
                        current_stage="Processing failed: Worker thread died unexpectedly."
                    )
                return job

            # Check if completed video exists on disk from prior run
            out_candidate = config.OUTPUT_DIR / f"enhanced_{job_id}.mp4"
            metrics_candidate = config.OUTPUT_DIR / f"enhanced_{job_id}.json"
            comp_candidate = config.OUTPUT_DIR / f"comparison_{job_id}.mp4"

            if out_candidate.exists():
                input_matches = list(config.UPLOAD_DIR.glob(f"*{job_id}*"))
                if not input_matches:
                    input_matches = list(config.UPLOAD_DIR.glob("sample_*.mp4"))
                input_path = str(input_matches[0]) if input_matches else str(out_candidate)

                job = JobState(job_id, input_path)
                job.output_path = str(out_candidate)
                if comp_candidate.exists():
                    job.comparison_path = str(comp_candidate)
                job.status = "completed"
                job.progress_percent = 100.0
                job.current_stage = "Enhancement complete. Ready for review and download."

                if metrics_candidate.exists():
                    try:
                        with open(metrics_candidate, "r", encoding="utf-8") as f:
                            job.metrics = json.load(f)
                    except Exception as e:
                        logger.warning("Could not read metrics JSON for job %s: %s", job_id, e)
                else:
                    job.metrics = {}

                job.add_log("Job loaded from disk cache.")
                self.jobs[job_id] = job
                return job

            return None

    def create_job(self, job_id: str, input_path: str) -> JobState:
        with self.jobs_lock:
            job = JobState(job_id, input_path)
            self.jobs[job_id] = job
            return job

    def cancel_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job and job.status not in ["completed", "failed", "cancelled"]:
            job.cancel_requested = True
            job.status = "cancelling"
            job.current_stage = "Cancellation requested..."
            job.add_log("Cancellation requested by user.")
            return True
        return False

    def delete_job(self, job_id: str) -> bool:
        """Deletes all generated artifacts and uploaded files associated with a job."""
        with self.jobs_lock:
            job = self.jobs.pop(job_id, None)

        files_to_remove = [
            config.OUTPUT_DIR / f"enhanced_{job_id}.mp4",
            config.OUTPUT_DIR / f"enhanced_{job_id}.json",
            config.OUTPUT_DIR / f"comparison_{job_id}.mp4",
            config.OUTPUT_DIR / f"thumb_{job_id}.jpg"
        ]
        files_to_remove.extend(config.UPLOAD_DIR.glob(f"*{job_id}*"))

        deleted_any = False
        for f in files_to_remove:
            if f.exists():
                try:
                    f.unlink()
                    deleted_any = True
                except OSError as e:
                    logger.warning("Could not delete %s: %s", f, e)

        return deleted_any or (job is not None)

    def get_queue_position(self, job_id: str) -> int:
        """Returns 1-based queue position. 0 if currently active or completed."""
        if self.active_job_id == job_id:
            return 0
        # Count position in current queue
        pos = 1 if self.active_job_id is not None else 0
        items = list(self.job_queue.queue)
        for idx, item in enumerate(items):
            if item[0].job_id == job_id:
                return pos + idx + 1
        return 0

    def start_enhancement_async(
        self,
        job_id: str,
        preset: str = "balanced",
        strength: float = 1.0,
        max_seconds: Optional[float] = None,
        frame_skip: int = 1,
        restore_at_original_res: Optional[bool] = None,
        stage_overrides: Optional[Dict[str, bool]] = None,
        **kwargs
    ):
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found.")

        # Prevent double submit
        if job.status in ["queued", "initializing", "processing_frames", "reconstructing"]:
            logger.info("Job %s is already active in state '%s'", job_id, job.status)
            return

        if "max_frames" in kwargs and max_seconds is None:
            max_seconds = float(kwargs["max_frames"]) / 30.0

        # Calculate initial queue position
        q_pos = self.get_queue_position(job_id)
        job.update(
            status="queued",
            current_stage=f"Queued for processing (Queue position: {max(1, q_pos)})...",
            queue_position=max(1, q_pos)
        )
        job.add_log(f"Enqueued in GPU enhancement queue (Preset: {preset}).")

        self.job_queue.put((
            job, preset, strength, max_seconds, frame_skip, restore_at_original_res, stage_overrides
        ))

        self._ensure_worker_running()

    def _ensure_worker_running(self):
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._queue_worker, daemon=True, name="DashClarity-GPUWorker")
            self.worker_thread.start()

    def _queue_worker(self):
        """Serial GPU queue worker thread. Executes one enhancement job at a time."""
        logger.info("GPU Queue Worker started.")
        while not self._shutdown_event.is_set():
            try:
                task = self.job_queue.get(timeout=2.0)
            except queue.Empty:
                continue

            job, preset, strength, max_seconds, frame_skip, restore_at_original_res, stage_overrides = task
            self.active_job_id = job.job_id
            job.update(queue_position=0)

            try:
                if not job.cancel_requested:
                    self._run_pipeline(
                        job, preset, strength, max_seconds, frame_skip, restore_at_original_res, stage_overrides
                    )
                else:
                    job.update(status="cancelled", current_stage="Job cancelled before execution started.")
                    job.add_log("Job cancelled while queued.")
            except Exception as e:
                logger.exception("Unexpected error executing job %s: %s", job.job_id, e)
                job.update(
                    status="failed",
                    error_message=str(e),
                    current_stage=f"Processing failed: {str(e)}"
                )
            finally:
                self.active_job_id = None
                if self.pipeline is not None:
                    self.pipeline.release_cached_memory()  # give unused VRAM back to the desktop / other apps
                self.job_queue.task_done()

    def _run_pipeline(
        self,
        job: JobState,
        preset: str,
        strength: float,
        max_seconds: Optional[float],
        frame_skip: int,
        restore_at_original_res: Optional[bool],
        stage_overrides: Optional[Dict[str, bool]] = None
    ):
        job.update(
            status="initializing",
            current_stage=f"Initializing {preset} enhancement pipeline on GPU...",
            start_time=time.time(),
            progress_percent=2.0
        )
        job.add_log(f"Starting pipeline with preset: {preset}, blend strength: {strength:.2f}")

        output_filename = f"enhanced_{job.job_id}.mp4"
        final_output_path = config.OUTPUT_DIR / output_filename
        metrics_json_path = config.OUTPUT_DIR / f"enhanced_{job.job_id}.json"

        try:
            pipeline = self.get_pipeline()
            pipeline.reset_temporal()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()  # peak VRAM must describe this job only
            job.add_log(f"Compute hardware initialized: {pipeline.device} (fp16={pipeline.fp16})")

            # Initialize Streaming Processor
            streamer = StreamingVideoProcessor(
                input_path=job.input_path,
                output_path=str(final_output_path),
                max_seconds=max_seconds,
                frame_skip=frame_skip
            )

            job.update(
                total_frames=streamer.total_target_frames,
                status="processing_frames",
                current_stage=f"Executing {preset} pipeline ({streamer.fps:.1f} FPS target)...",
                progress_percent=5.0
            )
            job.add_log(
                f"Ingested stream: {streamer.in_width}x{streamer.in_height} @ {streamer.fps:.1f} FPS "
                f"({streamer.total_target_frames} frames to process)"
            )

            last_log_frame = 0

            def on_progress(telemetry):
                nonlocal last_log_frame
                if job.cancel_requested:
                    raise KeyboardInterrupt("Cancelled by user")
                job.update(
                    frames_processed=telemetry["frames_processed"],
                    current_fps=telemetry["fps"],
                    eta_seconds=telemetry["eta_seconds"],
                    elapsed_seconds=telemetry["elapsed_seconds"],
                    progress_percent=telemetry["progress_percent"],
                    current_stage=telemetry["stage"]
                )
                done = telemetry["frames_processed"]
                if done - last_log_frame >= 60 or done == streamer.total_target_frames:
                    job.add_log(f"Processed {done}/{streamer.total_target_frames} frames ({telemetry['fps']:.2f} FPS, ETA: {telemetry['eta_seconds']:.1f}s)")
                    last_log_frame = done

            def frame_fn(frame, prev):
                return pipeline.enhance_frame(
                    frame_bgr=frame,
                    frame_neighbor=prev,
                    preset=preset,
                    strength=strength,
                    restore_at_original_res=restore_at_original_res
                )

            # Execute streaming video processing
            result_info = streamer.process(
                frame_transform_fn=frame_fn,
                progress_callback=on_progress
            )

            if streamer.audio_extracted:
                job.add_log("Original audio track preserved and trimmed to the enhanced video length.")
            else:
                job.add_log(f"Audio note: {streamer.audio_warning or 'No audio stream.'}")

            sample_origs = result_info["sample_orig_frames"]
            sample_enhs = result_info["sample_enh_frames"]

            # Output-differs-from-input guard check on final output
            if sample_origs and sample_enhs:
                pipeline.verify_output_transformation(sample_origs[0], sample_enhs[0])
                job.add_log("Quality guard verified: Output confirmed distinct from input.")

            # Compute authentic real metrics and save directly to metrics.json
            job.update(
                current_stage="Calculating visual quality & stability metrics...",
                progress_percent=95.0
            )
            job.add_log("Calculating photometric, noise, and stability metrics...")

            orig_size = os.path.getsize(job.input_path) if os.path.exists(job.input_path) else 0
            enh_size = os.path.getsize(str(final_output_path)) if final_output_path.exists() else 0

            real_metrics = evaluate_video_metrics(
                sample_orig_frames=sample_origs,
                sample_enh_frames=sample_enhs,
                processing_time=result_info["total_time_seconds"],
                orig_file_size=orig_size,
                enh_file_size=enh_size,
                orig_res=f"{streamer.in_width}x{streamer.in_height}",
                enh_res=f"{result_info['out_width']}x{result_info['out_height']}",
                orig_fps=streamer.fps,
                metrics_json_path=str(metrics_json_path),
                frames_processed=result_info["total_frames_processed"]
            )

            # Generate thumbnail for history view
            self.get_or_create_thumbnail(job.job_id)

            job.update(
                status="completed",
                output_path=str(final_output_path),
                current_stage="Enhancement complete. Ready for review and download.",
                progress_percent=100.0,
                metrics=real_metrics,
                audio_warning=result_info.get("audio_warning")
            )
            if torch.cuda.is_available():
                job.add_log(f"GPU memory: peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MB allocated, "
                            f"{torch.cuda.max_memory_reserved() / 2**20:.0f} MB reserved "
                            f"(budget {pipeline.profile.get('vram_budget_mb')} MB, {pipeline.profile.get('tier')}).")
            job.add_log(f"Job completed successfully in {result_info['total_time_seconds']:.2f}s ({result_info['processing_fps']:.2f} FPS avg).")
            logger.info("Job %s completed successfully in %.2fs", job.job_id, result_info["total_time_seconds"])

        except KeyboardInterrupt:
            job.update(status="cancelled", current_stage="Job cancelled by user.")
            job.add_log("Job was cancelled by the user.")
            if final_output_path.exists():
                try:
                    final_output_path.unlink()
                except OSError:
                    pass
        except Exception as e:
            logger.exception("Pipeline error in job %s: %s", job.job_id, e)
            job.add_log(f"Error during processing: {str(e)}")
            job.update(
                status="failed",
                error_message=str(e),
                current_stage=f"Processing failed: {str(e)}"
            )

    def generate_comparison_clip(self, job_id: str) -> Optional[str]:
        """Creates a synchronized side-by-side MP4 comparison using FFmpeg hstack."""
        job = self.get_job(job_id)
        if not job or not job.output_path or not os.path.exists(job.output_path):
            return None

        comp_path = config.OUTPUT_DIR / f"comparison_{job_id}.mp4"
        if comp_path.exists() and comp_path.stat().st_size > 1000:
            job.comparison_path = str(comp_path)
            return str(comp_path)

        ffmpeg_bin = get_ffmpeg_binary()
        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(job.input_path),
            "-i", str(job.output_path),
            "-filter_complex",
            "[0:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2[v0];"
            "[1:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2[v1];"
            "[v0][v1]hstack=inputs=2[outv]",
            "-map", "[outv]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-movflags", "+faststart",
            str(comp_path)
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
            if res.returncode == 0 and comp_path.exists() and comp_path.stat().st_size > 1000:
                job.comparison_path = str(comp_path)
                return str(comp_path)
            else:
                logger.error("FFmpeg hstack failed: %s", res.stderr.decode("utf-8", errors="ignore"))
                return None
        except Exception as e:
            logger.error("Exception generating comparison clip: %s", e)
            return None

    def get_or_create_thumbnail(self, job_id: str) -> Optional[str]:
        """Extracts and saves a 320x180 JPG thumbnail for history listings."""
        thumb_path = config.OUTPUT_DIR / f"thumb_{job_id}.jpg"
        if thumb_path.exists():
            return str(thumb_path)

        job = self.get_job(job_id)
        if not job:
            return None

        video_path = job.output_path if job.output_path and os.path.exists(job.output_path) else job.input_path
        if not video_path or not os.path.exists(video_path):
            return None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_f = min(30, max(0, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            thumb = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(thumb_path), thumb)
            return str(thumb_path)

        return None

    def list_history(self, show_all: bool = False) -> List[Dict[str, Any]]:
        """Returns completed past enhancement jobs. Hides test jobs unless show_all=True."""
        history_items = []
        enhanced_files = list(config.OUTPUT_DIR.glob("enhanced_*.mp4"))
        enhanced_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        for enh_file in enhanced_files:
            job_id = enh_file.stem.replace("enhanced_", "")
            json_file = config.OUTPUT_DIR / f"enhanced_{job_id}.json"

            input_matches = list(config.UPLOAD_DIR.glob(f"*{job_id}*"))
            orig_name = Path(input_matches[0]).stem.replace(f"{job_id}_", "") if input_matches else f"clip_{job_id[:8]}"

            if not show_all and is_test_job(job_id, orig_name):
                continue

            metrics = {}
            if json_file.exists():
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                except Exception:
                    pass

            tech = metrics.get("technical_summary", {})
            created_dt = time.strftime("%b %d, %H:%M", time.localtime(enh_file.stat().st_mtime))
            self.get_or_create_thumbnail(job_id)

            proc_time = tech.get("total_processing_time_sec")
            raw_fps = tech.get("processing_fps")
            if (raw_fps is None or raw_fps <= 0) and proc_time and proc_time > 0:
                frames = tech.get("total_frames_processed", 0)
                if frames > 0:
                    raw_fps = frames / proc_time

            fps_display = f"{raw_fps:.2f} FPS" if (raw_fps is not None and raw_fps > 0) else "--"
            duration_display = f"{proc_time:.2f}s" if proc_time is not None else "--"

            history_items.append({
                "job_id": job_id,
                "clip_name": orig_name,
                "created_at": created_dt,
                "resolution": tech.get("enhanced_resolution", "720p"),
                "duration": f"{duration_display} Processing time",
                "fps": fps_display,
                "condition": tech.get("scene_lighting_condition", "Low-light scene"),
                "thumbnail_url": f"/media/thumbnail/{job_id}",
                "result_url": f"/results/{job_id}",
                "download_url": f"/api/download/{job_id}"
            })

        return history_items
