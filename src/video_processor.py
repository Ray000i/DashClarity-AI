"""
DashClarity AI - Video Processor Coordinator
Manages asynchronous enhancement jobs, thread-safe progress reporting,
and end-to-end pipeline execution from raw upload to final enhanced MP4.
"""

import os
import time
import shutil
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np

from config import UPLOAD_DIR, OUTPUT_DIR, DEFAULT_MAX_FRAMES
from src.video_loader import VideoLoader
from src.enhancement_engine import STCDEnhancer
from src.video_reconstructor import VideoReconstructor
from src.quality_metrics import calculate_enhancement_metrics
from src.metadata import extract_video_metadata

logger = logging.getLogger("DashClarity.Processor")

class JobState:
    """Thread-safe representation of an enhancement job's progress and state."""
    def __init__(self, job_id: str, input_path: str):
        self.job_id = job_id
        self.input_path = input_path
        self.output_path = None
        self.status = "queued"  # queued, processing_frames, reconstructing, completed, failed, cancelled
        self.current_stage = "Queued in processing pipeline"
        self.progress_percent = 0.0
        self.frames_processed = 0
        self.total_frames = 0
        self.current_fps = 0.0
        self.eta_seconds = 0.0
        self.elapsed_seconds = 0.0
        self.start_time = 0.0
        self.error_message = None
        self.cancel_requested = False
        self.metrics = {}
        self.lock = threading.Lock()

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
                "error": self.error_message,
                "metrics": self.metrics,
                "has_enhanced_video": self.output_path is not None and os.path.exists(self.output_path)
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

    def get_job(self, job_id: str) -> Optional[JobState]:
        with self.jobs_lock:
            return self.jobs.get(job_id)

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
            return True
        return False

    def start_enhancement_async(
        self,
        job_id: str,
        max_frames: Optional[int] = DEFAULT_MAX_FRAMES,
        super_res: bool = False
    ):
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found.")

        thread = threading.Thread(
            target=self._run_pipeline,
            args=(job, max_frames, super_res),
            daemon=True
        )
        thread.start()

    def _run_pipeline(self, job: JobState, max_frames: Optional[int], super_res: bool):
        job.update(
            status="initializing",
            current_stage="Initializing STCD neural model on GPU...",
            start_time=time.time(),
            progress_percent=2.0
        )
        
        temp_dir = OUTPUT_DIR / f"temp_{job.job_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_audio = temp_dir / "audio_track.aac"
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"enhanced_{job.job_id}.mp4"
        final_output_path = OUTPUT_DIR / output_filename
        
        loader = None
        try:
            # 1. Video Loading and Metadata
            loader = VideoLoader(job.input_path)
            total_video_frames = loader.get_frame_count()
            target_frames = min(total_video_frames, max_frames) if max_frames else total_video_frames
            fps = loader.get_fps()
            orig_w, orig_h = loader.get_dimensions()
            
            job.update(
                total_frames=target_frames,
                current_stage="Extracting audio track for synchronized reconstruction...",
                progress_percent=5.0
            )
            
            # 2. Audio Extraction
            has_audio = loader.extract_audio(str(temp_audio))
            
            # 3. Model Engine Setup
            enhancer = STCDEnhancer.get_instance()
            
            job.update(
                status="processing_frames",
                current_stage="Executing STCD spatial-temporal video decomposition...",
                progress_percent=8.0
            )
            
            # 4. Sequential Frame Processing with Cross-Frame Attention
            frames_buffer = []
            for idx, frame in loader.stream_frames(max_frames=target_frames):
                frames_buffer.append(frame)
                
            actual_frame_count = len(frames_buffer)
            if actual_frame_count == 0:
                raise ValueError("Could not decode any valid video frames from input.")
                
            job.update(total_frames=actual_frame_count)
            
            sample_orig_frames = []
            sample_enh_frames = []
            sample_interval = max(1, actual_frame_count // 10)
            
            t_process_start = time.time()
            
            for i in range(actual_frame_count):
                if job.cancel_requested:
                    job.update(status="cancelled", current_stage="Job cancelled by user.")
                    return
                    
                frame_current = frames_buffer[i]
                
                # Pair with next frame, or previous frame for final index
                if i < actual_frame_count - 1:
                    frame_neighbor = frames_buffer[i + 1]
                else:
                    frame_neighbor = frames_buffer[i - 1] if actual_frame_count > 1 else frame_current
                    
                enh_dict = enhancer.enhance_frame_pair(frame_current, frame_neighbor)
                frame_enh = enh_dict["enhanced"]
                
                # Optional Super-Resolution (Module C)
                if super_res:
                    frame_enh = enhancer.apply_super_resolution(frame_enh, scale=2.0)
                    
                # Save frame
                frame_path = frames_dir / f"frame_{i:06d}.png"
                cv2.imwrite(str(frame_path), frame_enh)
                
                # Collect metrics samples
                if i % sample_interval == 0:
                    sample_orig_frames.append(frame_current)
                    sample_enh_frames.append(frame_enh)
                    
                # Progress calculation
                elapsed = time.time() - t_process_start
                frames_done = i + 1
                curr_fps = frames_done / elapsed if elapsed > 0 else 0.0
                remaining_frames = actual_frame_count - frames_done
                eta = remaining_frames / curr_fps if curr_fps > 0 else 0.0
                pct = 8.0 + (frames_done / actual_frame_count) * 75.0
                
                job.update(
                    frames_processed=frames_done,
                    current_fps=curr_fps,
                    eta_seconds=eta,
                    elapsed_seconds=elapsed,
                    progress_percent=pct,
                    current_stage=f"Enhancing frame {frames_done}/{actual_frame_count} ({curr_fps:.1f} FPS)"
                )
                
            total_processing_time = time.time() - t_process_start
            
            # 5. Reconstruct Video
            job.update(
                status="reconstructing",
                current_stage="Reconstructing enhanced video with H.264 & audio muxing...",
                progress_percent=86.0
            )
            
            enh_h, enh_w = (orig_h * 2, orig_w * 2) if super_res else (orig_h, orig_w)
            reconstructor = VideoReconstructor(
                output_path=str(final_output_path),
                fps=fps,
                width=enh_w,
                height=enh_h,
                audio_path=str(temp_audio) if has_audio else None
            )
            reconstructor.assemble_from_frame_dir(str(frames_dir))
            
            # 6. Quality Metrics Analysis
            job.update(
                current_stage="Computing authentic visual quality and stability metrics...",
                progress_percent=95.0
            )
            
            orig_size = os.path.getsize(job.input_path)
            enh_size = os.path.getsize(str(final_output_path))
            
            metrics = calculate_enhancement_metrics(
                sample_orig_frames=sample_orig_frames,
                sample_enh_frames=sample_enh_frames,
                processing_time=total_processing_time,
                orig_file_size=orig_size,
                enh_file_size=enh_size,
                orig_res=f"{orig_w}x{orig_h}",
                enh_res=f"{enh_w}x{enh_h}",
                orig_fps=fps
            )
            
            job.update(
                status="completed",
                output_path=str(final_output_path),
                current_stage="Enhancement complete. Ready for review and download.",
                progress_percent=100.0,
                metrics=metrics
            )
            logger.info(f"Job {job.job_id} successfully completed in {total_processing_time:.2f}s.")
            
        except Exception as e:
            logger.exception(f"Pipeline error in job {job.job_id}: {e}")
            job.update(
                status="failed",
                error_message=str(e),
                current_stage=f"Processing failed: {str(e)}"
            )
        finally:
            if loader:
                loader.close()
            # Clean up intermediate frame png files to conserve disk space
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")
