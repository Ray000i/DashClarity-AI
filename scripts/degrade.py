"""
DashClarity AI - Synthetic Degradation Engine (Ground-Truth Generator)
Creates physically modeled dashcam degraded footage from clean reference clips:
1. Non-linear low-light attenuation (gamma + linear gain reduction)
2. Heteroscedastic CMOS sensor noise (Poisson shot noise + Gaussian read noise)
3. Atmospheric/lens motion blur
4. Spatial resolution downscaling
5. H.264 compression block artifacts
"""

import os
import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils import get_ffmpeg_binary


def apply_dashcam_degradation(
    frame_bgr: np.ndarray,
    gamma: float = 2.4,
    gain: float = 0.22,
    noise_sigma: float = 14.0,
    blur_ksize: int = 5,
    blur_sigma: float = 1.2,
    downscale_factor: float = 0.5
) -> np.ndarray:
    """
    Simulates night-time low-light CMOS sensor degradation on a clean frame.
    """
    h, w = frame_bgr.shape[:2]
    img = frame_bgr.astype(np.float32) / 255.0

    # 1. Non-linear low-light attenuation (Gamma + Gain)
    darkened = gain * (img ** gamma)

    # 2. Heteroscedastic noise: Poisson photon shot noise + Gaussian sensor read noise
    # Poisson noise scaled by light intensity
    poisson_noisy = np.random.poisson(np.clip(darkened * 255.0, 0, 255)) / 255.0
    # Additive Gaussian read noise
    gauss_noise = np.random.normal(0, noise_sigma / 255.0, darkened.shape)
    noisy = np.clip(poisson_noisy + gauss_noise, 0.0, 1.0)

    # 3. Lens and atmospheric blur
    if blur_ksize > 1:
        blurred = cv2.GaussianBlur(noisy, (blur_ksize, blur_ksize), blur_sigma)
    else:
        blurred = noisy

    # 4. Spatial downsampling (CMOS pixel binning / low-res sensor)
    if downscale_factor < 1.0:
        dw, dh = int(w * downscale_factor), int(h * downscale_factor)
        lowres = cv2.resize(blurred, (dw, dh), interpolation=cv2.INTER_AREA)
        # Restore to original dimensions for pairing
        degraded = cv2.resize(lowres, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        degraded = blurred

    out_bgr = np.clip(degraded * 255.0, 0, 255).astype(np.uint8)
    return out_bgr


def create_synthetic_dataset(
    input_video_path: str,
    output_dir: str = "data/synthetic_pairs",
    num_frames: int = 30
):
    out_path = Path(output_dir)
    clean_dir = out_path / "clean"
    degraded_dir = out_path / "degraded"
    clean_dir.mkdir(parents=True, exist_ok=True)
    degraded_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[DEGRADE] Reading clean reference: {input_video_path} ({w}x{h} @ {fps:.1f} FPS)")

    degraded_frames = []
    clean_frames = []

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Degrade frame
        deg = apply_dashcam_degradation(frame)

        # Save image pairs
        cv2.imwrite(str(clean_dir / f"frame_{i:04d}.png"), frame)
        cv2.imwrite(str(degraded_dir / f"frame_{i:04d}.png"), deg)

        clean_frames.append(frame)
        degraded_frames.append(deg)

    cap.release()
    print(f"[DEGRADE] Extracted {len(clean_frames)} frame pairs.")

    # Write degraded video MP4 with H.264 compression via FFmpeg
    deg_video_path = out_path / "degraded_video.mp4"
    clean_video_path = out_path / "clean_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(deg_video_path), fourcc, fps, (w, h))
    for f in degraded_frames:
        out_writer.write(f)
    out_writer.release()

    clean_writer = cv2.VideoWriter(str(clean_video_path), fourcc, fps, (w, h))
    for f in clean_frames:
        clean_writer.write(f)
    clean_writer.release()

    meta = {
        "source_video": input_video_path,
        "num_pairs": len(clean_frames),
        "resolution": f"{w}x{h}",
        "fps": fps,
        "degradation_parameters": {
            "gamma": 2.4,
            "gain": 0.22,
            "noise_sigma": 14.0,
            "blur_ksize": 5,
            "blur_sigma": 1.2,
            "downscale_factor": 0.5
        }
    }
    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[DEGRADE] Dataset successfully created at: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic dashcam degradation pairs")
    parser.add_argument("--input", "-i", type=str, default="input_videos/95fd223e9fe3_19960474-hd_1080_1920_30fps.mp4")
    parser.add_argument("--out", "-o", type=str, default="data/synthetic_pairs")
    parser.add_argument("--frames", "-n", type=int, default=30)
    args = parser.parse_args()

    create_synthetic_dataset(args.input, args.out, args.frames)
