"""
DashClarity AI - Preset Benchmark on RTX 4060 Laptop GPU
Measures FPS, processing latency, and peak VRAM consumption across all presets at both 720p and 1080p.
"""

import sys
import time
import torch
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline.enhancer import VideoEnhancementPipeline

def benchmark_resolution(resolution_name: str, width: int, height: int, num_frames: int = 15):
    print(f"\n=======================================================")
    print(f"BENCHMARKING RESOLUTION: {resolution_name} ({width}x{height})")
    print(f"=======================================================")
    
    presets = ["low_light", "rain_noise", "upscale_only", "full", "stcd_comparison", "classical_baseline"]
    results = []
    
    # Synthetic frame representative of dashcam content
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Road horizon and lines
    frame[height//2:, :] = 40
    cv2.line(frame, (width//2, height//2), (width//4, height), (200, 200, 200), 4)
    cv2.line(frame, (width//2, height//2), (3*width//4, height), (200, 200, 200), 4)
    
    for preset_name in presets:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            pipeline = VideoEnhancementPipeline(preset=preset_name)
            
            # Warmup
            _ = pipeline.enhance_frame(frame, frame, preset=preset_name)
            torch.cuda.synchronize()
            
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(num_frames):
                _ = pipeline.enhance_frame(frame, frame, preset=preset_name)
            torch.cuda.synchronize()
            total_time = time.time() - t0
            
            fps = num_frames / total_time
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            results.append({"preset": preset_name, "fps": round(fps, 2), "vram_mb": round(peak_vram_mb, 1)})
            print(f"[{preset_name.upper():<18}] {fps:5.2f} FPS | Peak VRAM: {peak_vram_mb:6.1f} MB")
        except Exception as e:
            print(f"[{preset_name.upper():<18}] FAILED: {e}")
            results.append({"preset": preset_name, "fps": 0.0, "vram_mb": 0.0})
            
    return results

if __name__ == "__main__":
    res_720p = benchmark_resolution("720p Standard Dashcam", 1280, 720, num_frames=15)
    res_1080p = benchmark_resolution("1080p Full HD Dashcam", 1920, 1080, num_frames=10)
    
    print("\n" + "="*65)
    print("CONSOLIDATED BENCHMARK REPORT (NVIDIA RTX 4060 Laptop GPU)")
    print("="*65)
    print(f"{'PRESET':<20} | {'720p FPS':<10} | {'720p VRAM':<10} | {'1080p FPS':<10} | {'1080p VRAM':<10}")
    print("-" * 65)
    for r7, r10 in zip(res_720p, res_1080p):
        print(f"{r7['preset']:<20} | {r7['fps']:<10.2f} | {r7['vram_mb']:<7.1f} MB | {r10['fps']:<10.2f} | {r10['vram_mb']:<7.1f} MB")
    print("="*65)
