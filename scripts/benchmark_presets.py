"""
DashClarity AI - Preset Benchmark on RTX 4060 Laptop GPU
Measures FPS, processing latency, and peak VRAM consumption across all presets at both 720p and 1080p.
Updates results/results.json with measured performance metrics.
"""

import sys
import os
import json
import time
from pathlib import Path
import torch
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline.enhancer import VideoEnhancementPipeline


def benchmark_resolution(resolution_name: str, width: int, height: int):
    print(f"\n=======================================================", flush=True)
    print(f"BENCHMARKING RESOLUTION: {resolution_name} ({width}x{height})", flush=True)
    print(f"=======================================================", flush=True)

    presets = [
        "fast",
        "balanced",
        "quality",
        "stcd_comparison",
        "hybrid_scunet_stcd",
        "classical_baseline"
    ]
    results = {}

    # Synthetic realistic dashcam frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:height//2, :] = (20, 20, 30)
    frame[height//2:, :] = (40, 40, 40)
    cv2.line(frame, (width//2, height//2), (width//4, height), (180, 180, 180), 4)
    cv2.line(frame, (width//2, height//2), (3*width//4, height), (180, 180, 180), 4)

    for preset_name in presets:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Adjust frame count based on preset throughput to maintain quick evaluation
        if preset_name in ("fast", "classical_baseline"):
            num_frames = 15
        elif preset_name == "balanced":
            num_frames = 10
        else:
            num_frames = 5 if width <= 1280 else 3

        try:
            pipeline = VideoEnhancementPipeline(preset=preset_name)

            # Warmup pass
            _ = pipeline.enhance_frame(frame, frame, preset=preset_name)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(num_frames):
                _ = pipeline.enhance_frame(frame, frame, preset=preset_name)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_time = time.time() - t0

            fps = round(num_frames / total_time, 2)
            peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if torch.cuda.is_available() else 0.0
            peak_vram_gb = round(peak_vram_mb / 1024.0, 2)

            results[preset_name] = {
                "fps": fps,
                "latency_ms": round((total_time / num_frames) * 1000.0, 1),
                "vram_mb": peak_vram_mb,
                "vram_gb": peak_vram_gb
            }
            print(f"[{preset_name.upper():<20}] {fps:5.2f} FPS | Peak VRAM: {peak_vram_mb:6.1f} MB ({peak_vram_gb:.2f} GB) | Latency: {results[preset_name]['latency_ms']:6.1f} ms", flush=True)
        except Exception as e:
            print(f"[{preset_name.upper():<20}] FAILED: {e}", flush=True)
            results[preset_name] = {"fps": 0.0, "latency_ms": 0.0, "vram_mb": 0.0, "vram_gb": 0.0}

    return results


def main():
    res_720p = benchmark_resolution("720p Standard Dashcam", 1280, 720)
    res_1080p = benchmark_resolution("1080p Full HD Dashcam", 1920, 1080)

    print("\n" + "="*85, flush=True)
    print("CONSOLIDATED BENCHMARK REPORT (NVIDIA GeForce RTX 4060 Laptop GPU)", flush=True)
    print("="*85, flush=True)
    print(f"{'PRESET':<20} | {'720p FPS':<10} | {'720p VRAM':<12} | {'1080p FPS':<10} | {'1080p VRAM':<12}", flush=True)
    print("-" * 85, flush=True)
    all_keys = list(res_720p.keys())
    for k in all_keys:
        r7 = res_720p.get(k, {})
        r10 = res_1080p.get(k, {})
        print(f"{k:<20} | {r7.get('fps', 0):<10.2f} | {r7.get('vram_mb', 0):>6.1f} MB ({r7.get('vram_gb', 0):.2f} GB) | {r10.get('fps', 0):<10.2f} | {r10.get('vram_mb', 0):>6.1f} MB ({r10.get('vram_gb', 0):.2f} GB)", flush=True)
    print("="*85, flush=True)

    # Save to results/results.json
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "results.json"

    data = {}
    if results_file.exists():
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

    if "profiling" not in data:
        data["profiling"] = {}
    data["profiling"]["presets"] = {
        "720p": res_720p,
        "1080p": res_1080p
    }
    data["profiling"]["hardware"] = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\n[OK] Updated measured profiling data saved to {results_file}", flush=True)


if __name__ == "__main__":
    main()
