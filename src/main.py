"""
DashClarity AI - Command Line Interface (CLI)
AI-Powered Dashcam Video Enhancement Platform.
Usage:
  .venv\\Scripts\\python -m src.main --input in.mp4 --output out.mp4 --preset fast --strength 0.8 --max_seconds 10 --device cuda
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

import cv2
import torch
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline.enhancer import VideoEnhancementPipeline
from src.pipeline.video_stream import StreamingVideoProcessor
from src.utils.metrics import evaluate_video_metrics
from src.metadata import extract_video_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DashClarity.CLI")


def parse_args():
    parser = argparse.ArgumentParser(
        description="DashClarity AI — AI-Powered Dashcam Footage Enhancement CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to input dashcam video file (MP4, AVI, MOV, MKV)"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Path for enhanced output video file (e.g., enhanced.mp4)"
    )
    parser.add_argument(
        "--preset", "-p", type=str, default="fast",
        choices=["fast", "balanced", "quality", "full", "stcd_comparison", "hybrid_scunet_stcd", "classical_baseline"],
        help="Enhancement pipeline preset"
    )
    parser.add_argument(
        "--strength", "-s", type=float, default=0.8,
        help="Enhancement strength factor (0.1 to 1.0)"
    )
    parser.add_argument(
        "--max_seconds", "-m", type=float, default=10.0,
        help="Maximum clip duration in seconds to process (0 or negative for all frames)"
    )
    parser.add_argument(
        "--device", "-d", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Inference compute device"
    )
    parser.add_argument(
        "--frame_skip", type=int, default=1,
        help="Frame skip interval (1 = process every frame, 2 = every 2nd frame)"
    )
    parser.add_argument(
        "--restore_at_original_res", action="store_true", default=None,
        help="Downscale super-resolution back to input resolution"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        logger.error(f"Input video file not found: {input_path}")
        sys.exit(1)
        
    print("=" * 72)
    print(" DashClarity AI — Video Enhancement CLI")
    print("=" * 72)
    print(f" Input Video:     {input_path.name}")
    print(f" Output Video:    {output_path.name}")
    print(f" Selected Preset: {args.preset.upper()}")
    print(f" Strength Factor: {args.strength * 100:.0f}%")
    max_sec = args.max_seconds if args.max_seconds > 0 else "Full Clip (All Frames)"
    print(f" Duration Limit:  {max_sec}")
    
    # Device configuration
    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f" Compute Device:  {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU Mode'})")
    print("-" * 72)
    
    # Extract metadata
    try:
        meta = extract_video_metadata(str(input_path))
        print(f" Source Specs:    {meta['width']}x{meta['height']} @ {meta['fps']:.1f} FPS, {meta['total_frames']} frames ({meta['duration_seconds']:.1f}s)")
    except Exception as e:
        logger.warning(f"Could not read complete metadata: {e}")
        
    # Initialize Pipeline
    print(f"\n[1/3] Loading neural models for preset '{args.preset}' on {device}...")
    t_start = time.time()
    try:
        pipeline = VideoEnhancementPipeline(preset=args.preset)
        pipeline.device = device
        pipeline.reset_temporal()
    except Exception as e:
        logger.exception(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
        
    # Initialize Streaming Processor
    max_sec_val = args.max_seconds if args.max_seconds > 0 else None
    streamer = StreamingVideoProcessor(
        input_path=str(input_path),
        output_path=str(output_path),
        max_seconds=max_sec_val,
        frame_skip=args.frame_skip
    )
    
    print(f"[2/3] Processing video stream ({streamer.total_target_frames} target frames)...")
    
    def on_progress(telemetry):
        pct = telemetry["progress_percent"]
        fps = telemetry["fps"]
        eta = telemetry["eta_seconds"]
        cur = telemetry["frames_processed"]
        tot = telemetry["total_frames"]
        sys.stdout.write(f"\r  -> Progress: {pct:5.1f}% | Frame: {cur:4d}/{tot:4d} | Speed: {fps:5.1f} FPS | ETA: {eta:5.1f}s  ")
        sys.stdout.flush()
        
    def frame_fn(frame, prev):
        return pipeline.enhance_frame(
            frame_bgr=frame,
            frame_neighbor=prev,
            preset=args.preset,
            strength=args.strength,
            restore_at_original_res=args.restore_at_original_res
        )
        
    try:
        res = streamer.process(
            frame_transform_fn=frame_fn,
            progress_callback=on_progress
        )
        sys.stdout.write("\n")
    except Exception as e:
        sys.stdout.write("\n")
        logger.exception(f"Processing error: {e}")
        sys.exit(1)
        
    # Verify Output Guard
    sample_origs = res["sample_orig_frames"]
    sample_enhs = res["sample_enh_frames"]
    if sample_origs and sample_enhs:
        try:
            mad = pipeline.verify_output_transformation(sample_origs[0], sample_enhs[0])
            print(f"  -> Quality Guard: Verified (MAD = {mad:.2f} >= threshold)")
        except RuntimeError as e:
            logger.error(str(e))
            sys.exit(1)
            
    # Compute Metrics
    print("\n[3/3] Evaluating authentic visual quality metrics...")
    metrics_path = str(output_path.with_suffix(".json"))
    orig_size = os.path.getsize(str(input_path)) if input_path.exists() else 0
    enh_size = os.path.getsize(str(output_path)) if output_path.exists() else 0
    
    metrics = evaluate_video_metrics(
        sample_orig_frames=sample_origs,
        sample_enh_frames=sample_enhs,
        processing_time=res["total_time_seconds"],
        orig_file_size=orig_size,
        enh_file_size=enh_size,
        orig_res=f"{streamer.in_width}x{streamer.in_height}",
        enh_res=f"{res['out_width']}x{res['out_height']}",
        orig_fps=streamer.fps,
        metrics_json_path=metrics_path
    )
    
    tot_time = time.time() - t_start
    print("\n" + "=" * 72)
    print(" ENHANCEMENT COMPLETE")
    print("=" * 72)
    print(f" Output Video File:    {output_path} ({enh_size / 1024 / 1024:.2f} MB)")
    print(f" Metrics JSON:         {metrics_path}")
    print(f" Total Runtime:        {tot_time:.2f}s (Processing Speed: {len(sample_origs) / res['total_time_seconds']:.1f} FPS)")
    
    ill = metrics.get("illumination_analysis", {})
    tech = metrics.get("technical_summary", {})
    no_ref = metrics.get("no_reference_quality", {})
    
    print("-" * 72)
    print(f" Luminance Recovery:   {ill.get('mean_luminance_original', 0):.1f} -> {ill.get('mean_luminance_enhanced', 0):.1f} ({ill.get('illumination_gain_percent', 0):+.1f}%)")
    print(f" Contrast (CII):       {ill.get('contrast_improvement_index', 1.0):.2f}x")
    print(f" Temporal Stability:   {ill.get('temporal_stability_index', 0):.1f} / 100")
    print(f" Noise (Sigma):        {ill.get('noise_sigma_original', 0):.2f} -> {ill.get('noise_sigma_enhanced', 0):.2f}")
    if no_ref.get("niqe") is not None:
        print(f" NIQE / BRISQUE:       {no_ref.get('niqe')} / {no_ref.get('brisque')}")
    if tech.get("peak_vram_mb"):
        print(f" Peak GPU VRAM:        {tech.get('peak_vram_mb')} MB")
    if res.get("audio_warning"):
        print(f" Audio Notice:         {res.get('audio_warning')}")
    print("=" * 72)
    print(" Note: Enhanced footage is not original evidence; generative upscaling can hallucinate details.\n")
    return 0


if __name__ == "__main__":
    main()
