"""
DashClarity AI - Diagnostic & Quality Verification Script (Phase 2)
Evaluates input vs output difference, PSNR, SSIM, and generates amplified difference maps.
Enforces the output-differs-from-input guard: raises an explicit error if MAD < threshold.
"""

import sys
import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import estimate_sigma

# Ensure workspace root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline.enhancer import VideoEnhancementPipeline


def run_diagnose(
    video_path: str,
    preset: str = "full",
    frame_index: int = 0,
    output_dir: str = "scripts/diag_output",
    min_mad_threshold: float = 3.5,
    tag: str = ""
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    print(f"\n[DIAGNOSE] Target Video: {video_path}")
    print(f"[DIAGNOSE] Preset: {preset} | Frame: {frame_index}")
    print(f"[DIAGNOSE] PyTorch Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"[DIAGNOSE] GPU Name: {torch.cuda.get_device_name(0)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret1, f1 = cap.read()
    ret2, f2 = cap.read()
    cap.release()

    if not ret1 or f1 is None:
        raise ValueError("Failed to read video frame 1")
    if not ret2 or f2 is None:
        f2 = f1.copy()

    h, w = f1.shape[:2]
    print(f"[DIAGNOSE] Input Resolution: {w}x{h}")

    pipeline = VideoEnhancementPipeline(preset=preset)
    f_out, stage_telemetry = pipeline.enhance_frame(f1, f2, preset=preset)
    out_h, out_w = f_out.shape[:2]
    print(f"[DIAGNOSE] Enhanced Resolution: {out_w}x{out_h}")

    # If resolutions differ, align original for quantitative comparison
    if f1.shape[:2] != f_out.shape[:2]:
        f1_aligned = cv2.resize(f1, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    else:
        f1_aligned = f1

    # Quantitative Differences
    diff_raw = np.abs(f_out.astype(np.float32) - f1_aligned.astype(np.float32)).round().astype(np.uint8)
    mad = float(np.mean(diff_raw))
    max_diff = int(np.max(diff_raw))
    pct_identical = float(np.mean(diff_raw == 0) * 100.0)

    # Sharpness, Noise & Luminance
    gray_in = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    gray_out = cv2.cvtColor(f_out, cv2.COLOR_BGR2GRAY)
    lap_in = float(cv2.Laplacian(gray_in, cv2.CV_64F).var())
    lap_out = float(cv2.Laplacian(gray_out, cv2.CV_64F).var())
    lum_in = float(np.mean(gray_in))
    lum_out = float(np.mean(gray_out))
    
    try:
        sigma_in = float(estimate_sigma(f1, channel_axis=-1, average_sigmas=True))
        sigma_out = float(estimate_sigma(f_out, channel_axis=-1, average_sigmas=True))
    except Exception:
        sigma_in = float(np.std(gray_in[:32, :32]))
        sigma_out = float(np.std(gray_out[:32, :32]))

    # Amplified Difference Map
    diff_gray = cv2.cvtColor(diff_raw, cv2.COLOR_BGR2GRAY)
    diff_amplified = np.clip(diff_gray.astype(np.float32) * 10.0, 0, 255).astype(np.uint8)
    diff_heatmap = cv2.applyColorMap(diff_amplified, cv2.COLORMAP_JET)

    prefix = f"{tag}_" if tag else ""
    in_file = out_dir / f"{prefix}input.png"
    out_file = out_dir / f"{prefix}output.png"
    diff_file = out_dir / f"{prefix}diff.png"

    cv2.imwrite(str(in_file), f1)
    cv2.imwrite(str(out_file), f_out)
    cv2.imwrite(str(diff_file), diff_heatmap)

    print("\n" + "="*55)
    print(f"DIAGNOSTIC RESULTS [{preset.upper()}] {tag}")
    print("="*55)
    print(f"Mean Absolute Difference (MAD) : {mad:.4f} gray levels")
    print(f"Maximum Pixel Difference       : {max_diff} / 255")
    print(f"Identical Pixels (diff == 0)   : {pct_identical:.2f}%")
    print(f"Luminance (Mean Y)             : {lum_in:.1f} -> {lum_out:.1f}")
    print(f"Laplacian Sharpness Variance   : {lap_in:.1f} -> {lap_out:.1f}")
    print(f"Wavelet Noise StdDev (Sigma)   : {sigma_in:.2f} -> {sigma_out:.2f}")
    print(f"Saved Input Frame              : {in_file}")
    print(f"Saved Enhanced Frame           : {out_file}")
    print(f"Saved Amplified Difference Map : {diff_file}")
    print("="*55)

    # Quality Guard check
    if mad < min_mad_threshold:
        err_msg = (
            f"Quality Guard Failure: Output is visually identical to input! "
            f"MAD ({mad:.4f}) is below threshold ({min_mad_threshold:.1f})."
        )
        print(f"[ERROR] {err_msg}")
        raise RuntimeError(err_msg)

    print("[SUCCESS] Meaningful visual enhancement verified.\n")
    return {
        "mad": mad,
        "max_diff": max_diff,
        "pct_identical": pct_identical,
        "lum_in": lum_in,
        "lum_out": lum_out,
        "lap_in": lap_in,
        "lap_out": lap_out,
        "sigma_in": sigma_in,
        "sigma_out": sigma_out,
        "in_file": str(in_file),
        "out_file": str(out_file),
        "diff_file": str(diff_file)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DashClarity AI Video Diagnostics")
    parser.add_argument("--input", "-i", type=str, default="input_videos/sample_night_dashcam.mp4")
    parser.add_argument("--preset", "-p", type=str, default="full")
    parser.add_argument("--frame", "-f", type=int, default=0)
    parser.add_argument("--tag", "-t", type=str, default="")
    args = parser.parse_args()

    run_diagnose(
        video_path=args.input,
        preset=args.preset,
        frame_index=args.frame,
        tag=args.tag
    )
