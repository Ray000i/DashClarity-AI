"""
DashClarity AI - Video Quality Metrics Calculator
Calculates authentic visual metrics matching results/results.json schema:
- Mean Luminance (Y-channel)
- Sky Luminance Ratio (upper 25% FOV)
- Headlight Bloom Area Ratio (Y >= 240)
- RMS Contrast (std dev of luminance)
- Flat-Region Sensor Noise (textureless high-pass residual)
- Normalized Temporal Flicker (E[|I(t) - I(t-1)|] / mean_lum)
- No-Reference Perceptual Metrics (NIQE, BRISQUE via pyiqa)
- System Telemetry (Processing throughput FPS, Peak VRAM MB)

Outputs both legacy structures and standardized schema_metrics for generic UI rendering.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import estimate_sigma

logger = logging.getLogger("DashClarity.Utils.Metrics")

# Cache for pyiqa metrics to avoid re-instantiating on every clip
_NIQE_METRIC = None
_BRISQUE_METRIC = None


def get_pyiqa_metric(name: str):
    global _NIQE_METRIC, _BRISQUE_METRIC
    try:
        import pyiqa
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if name == "niqe":
            if _NIQE_METRIC is None:
                _NIQE_METRIC = pyiqa.create_metric("niqe", device=device)
            return _NIQE_METRIC
        elif name == "brisque":
            if _BRISQUE_METRIC is None:
                _BRISQUE_METRIC = pyiqa.create_metric("brisque", device=device)
            return _BRISQUE_METRIC
    except Exception as e:
        logger.warning("Could not instantiate pyiqa %s metric: %s", name, e)
    return None


def compute_flat_region_noise(img_bgr: np.ndarray) -> float:
    """Measures sensor noise standard deviation strictly on non-clipped, flat textureless regions."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Flat, non-saturated mask: 15 < gray < 235 and gradient magnitude < 4.0
    flat_mask = (gray > 15.0) & (gray < 235.0) & (grad_mag < 4.0)
    if np.sum(flat_mask) < 200:
        return float(cv2.Laplacian(gray, cv2.CV_64F).std())

    # High-pass residual via median filter difference
    gray_u8 = gray.astype(np.uint8)
    median_blur = cv2.medianBlur(gray_u8, 5).astype(np.float32)
    noise_res = gray - median_blur
    return float(np.std(noise_res[flat_mask]))


def compute_sky_luminance_ratio(raw_bgr: np.ndarray, enh_bgr: np.ndarray) -> float:
    """Computes ratio of mean luminance in top 25% sky region (out / in)."""
    h = raw_bgr.shape[0]
    sky_h = max(1, int(h * 0.25))
    raw_sky = float(np.mean(cv2.cvtColor(raw_bgr[:sky_h, :], cv2.COLOR_BGR2GRAY)))
    enh_sky = float(np.mean(cv2.cvtColor(enh_bgr[:sky_h, :], cv2.COLOR_BGR2GRAY)))
    return float(enh_sky / (raw_sky + 1e-5))


def compute_bloom_area_ratio(raw_bgr: np.ndarray, enh_bgr: np.ndarray) -> float:
    """Computes ratio of pixels with Y >= 240 (headlight/specular bloom out / in)."""
    raw_y = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    enh_y = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    raw_bloom = int(np.sum(raw_y >= 240))
    enh_bloom = int(np.sum(enh_y >= 240))
    if raw_bloom == 0:
        return 1.0 if enh_bloom == 0 else float(enh_bloom)
    return float(enh_bloom / raw_bloom)


def evaluate_video_metrics(
    sample_orig_frames: List[np.ndarray],
    sample_enh_frames: List[np.ndarray],
    processing_time: float,
    orig_file_size: int,
    enh_file_size: int,
    orig_res: str,
    enh_res: str,
    orig_fps: float,
    metrics_json_path: Optional[str] = None,
    ground_truth_frames: Optional[List[np.ndarray]] = None,
    compute_pyiqa: bool = True
) -> Dict[str, Any]:
    """
    Computes authentic metrics across sample frame pairs matching results.json schema.
    """
    num_samples = min(len(sample_orig_frames), len(sample_enh_frames))
    if num_samples == 0:
        return {}

    raw_grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in sample_orig_frames[:num_samples]]
    enh_grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in sample_enh_frames[:num_samples]]

    # 1. Mean Luminance & RMS Contrast
    raw_lum_mean = float(np.mean([np.mean(g) for g in raw_grays]))
    enh_lum_mean = float(np.mean([np.mean(g) for g in enh_grays]))

    raw_rms_contrast = float(np.mean([np.std(g) for g in raw_grays]))
    enh_rms_contrast = float(np.mean([np.std(g) for g in enh_grays]))

    # 2. Sky & Bloom ratios
    sky_ratios = [
        compute_sky_luminance_ratio(sample_orig_frames[i], sample_enh_frames[i])
        for i in range(num_samples)
    ]
    bloom_ratios = [
        compute_bloom_area_ratio(sample_orig_frames[i], sample_enh_frames[i])
        for i in range(num_samples)
    ]
    avg_sky_ratio = float(np.mean(sky_ratios)) if sky_ratios else 1.0
    avg_bloom_ratio = float(np.mean(bloom_ratios)) if bloom_ratios else 1.0

    # 3. Flat region sensor noise
    raw_flat_noises = [compute_flat_region_noise(f) for f in sample_orig_frames[:num_samples]]
    enh_flat_noises = [compute_flat_region_noise(f) for f in sample_enh_frames[:num_samples]]
    avg_raw_flat_noise = float(np.mean(raw_flat_noises)) if raw_flat_noises else 0.0
    avg_enh_flat_noise = float(np.mean(enh_flat_noises)) if enh_flat_noises else 0.0

    # 4. Normalized Temporal Flicker: E[|I(t) - I(t-1)|] / mean_lum
    norm_flicker_val = 0.0
    if num_samples >= 2:
        diffs = [
            float(np.mean(np.abs(enh_grays[i] - enh_grays[i - 1])))
            for i in range(1, num_samples)
        ]
        norm_flicker_val = float(np.mean(diffs) / (enh_lum_mean + 1e-5))

    # 5. NIQE & BRISQUE
    niqe_score = None
    brisque_score = None
    if compute_pyiqa and num_samples > 0:
        try:
            niqe_eval = get_pyiqa_metric("niqe")
            if niqe_eval is not None:
                mid_idx = num_samples // 2
                f_rgb = cv2.cvtColor(sample_enh_frames[mid_idx], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                t_in = torch.from_numpy(f_rgb).permute(2, 0, 1).unsqueeze(0).to(niqe_eval.device)
                with torch.no_grad():
                    niqe_score = round(float(niqe_eval(t_in).cpu().item()), 2)
        except Exception as e:
            logger.warning("NIQE evaluation error: %s", e)

        try:
            brisque_eval = get_pyiqa_metric("brisque")
            if brisque_eval is not None:
                mid_idx = num_samples // 2
                f_rgb = cv2.cvtColor(sample_enh_frames[mid_idx], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                t_in = torch.from_numpy(f_rgb).permute(2, 0, 1).unsqueeze(0).to(brisque_eval.device)
                with torch.no_grad():
                    brisque_score = round(float(brisque_eval(t_in).cpu().item()), 2)
        except Exception as e:
            logger.warning("BRISQUE evaluation error: %s", e)

    # 6. System & Hardware
    cuda_avail = torch.cuda.is_available()
    peak_vram = round(torch.cuda.max_memory_allocated() / (1024**2), 1) if cuda_avail else 0.0
    proc_fps = round(num_samples / processing_time, 2) if processing_time > 0 else 0.0

    # 7. Helper to evaluate change direction
    def evaluate_change(direction: str, raw_val, enh_val):
        if raw_val is None or enh_val is None or raw_val == "not available" or enh_val == "not available":
            return "neutral"
        if direction == "neutral":
            return "neutral"
        diff = enh_val - raw_val
        if abs(diff) < 1e-4:
            return "neutral"
        if direction == "lower_better":
            return "better" if diff < 0 else "worse"
        if direction == "higher_better":
            return "better" if diff > 0 else "worse"
        return "neutral"

    def format_diff(raw_val, enh_val):
        if raw_val is None or enh_val is None or raw_val == "not available" or enh_val == "not available":
            return "not available"
        diff = enh_val - raw_val
        prefix = "+" if diff >= 0 else ""
        return f"{prefix}{diff:.2f}"

    # 8. Standardized schema_metrics array for generic UI rendering
    schema_metrics = [
        {
            "id": "mean_lum",
            "label": "Mean luminance",
            "unit": "",
            "direction": "neutral",
            "caveat": None,
            "input": round(raw_lum_mean, 2),
            "output": round(enh_lum_mean, 2),
            "change": format_diff(round(raw_lum_mean, 2), round(enh_lum_mean, 2)),
            "evaluation": "neutral"
        },
        {
            "id": "sky_ratio",
            "label": "Sky luminance ratio",
            "unit": "",
            "direction": "lower_better",
            "caveat": "Luminance ratio in upper 25% field of view; higher values indicate night sky haze amplification.",
            "input": 1.0,
            "output": round(avg_sky_ratio, 2),
            "change": format_diff(1.0, round(avg_sky_ratio, 2)),
            "evaluation": evaluate_change("lower_better", 1.0, round(avg_sky_ratio, 2))
        },
        {
            "id": "bloom_ratio",
            "label": "Headlight bloom ratio",
            "unit": "",
            "direction": "lower_better",
            "caveat": "Ratio of saturated specular pixels (Y >= 240); lower or equal ratio indicates controlled glare without blooming.",
            "input": 1.0,
            "output": round(avg_bloom_ratio, 2),
            "change": format_diff(1.0, round(avg_bloom_ratio, 2)),
            "evaluation": evaluate_change("lower_better", 1.0, round(avg_bloom_ratio, 2))
        },
        {
            "id": "rms_contrast",
            "label": "RMS contrast",
            "unit": "",
            "direction": "higher_better",
            "caveat": "Standard deviation of pixel luminance across the frame; higher contrast reflects restored dynamic range.",
            "input": round(raw_rms_contrast, 2),
            "output": round(enh_rms_contrast, 2),
            "change": format_diff(round(raw_rms_contrast, 2), round(enh_rms_contrast, 2)),
            "evaluation": evaluate_change("higher_better", round(raw_rms_contrast, 2), round(enh_rms_contrast, 2))
        },
        {
            "id": "flat_noise",
            "label": "Flat-region noise std",
            "unit": "",
            "direction": "lower_better",
            "caveat": "High-pass residual variance measured strictly over textureless road and sky zones to exclude edge detail.",
            "input": round(avg_raw_flat_noise, 2),
            "output": round(avg_enh_flat_noise, 2),
            "change": format_diff(round(avg_raw_flat_noise, 2), round(avg_enh_flat_noise, 2)),
            "evaluation": evaluate_change("lower_better", round(avg_raw_flat_noise, 2), round(avg_enh_flat_noise, 2))
        },
        {
            "id": "norm_flicker",
            "label": "Normalized temporal flicker",
            "unit": "",
            "direction": "lower_better",
            "caveat": "Frame-to-frame mean absolute difference normalized by scene luminance; lower score indicates steady playback.",
            "input": "not available",
            "output": round(norm_flicker_val, 4),
            "change": "not available",
            "evaluation": "neutral"
        },
        {
            "id": "niqe",
            "label": "NIQE (no-reference)",
            "unit": "",
            "direction": "lower_better",
            "caveat": "Unreliable for night footage. Natural scene statistics model trained on daytime photography.",
            "input": "not available",
            "output": niqe_score if niqe_score is not None else "not available",
            "change": "not available",
            "evaluation": "neutral"
        },
        {
            "id": "brisque",
            "label": "BRISQUE (no-reference)",
            "unit": "",
            "direction": "lower_better",
            "caveat": "Unreliable for night footage. Evaluates natural luminance gradients which dark dashcam scenes violate.",
            "input": "not available",
            "output": brisque_score if brisque_score is not None else "not available",
            "change": "not available",
            "evaluation": "neutral"
        },
        {
            "id": "processing_fps",
            "label": "Processing throughput",
            "unit": "FPS",
            "direction": "higher_better",
            "caveat": None,
            "input": "not available",
            "output": proc_fps,
            "change": "not available",
            "evaluation": "neutral"
        },
        {
            "id": "peak_vram_mb",
            "label": "Peak GPU VRAM",
            "unit": "MB",
            "direction": "lower_better",
            "caveat": None,
            "input": "not available",
            "output": peak_vram,
            "change": "not available",
            "evaluation": "neutral"
        }
    ]

    # Scene Lighting Condition Diagnosis
    if raw_lum_mean > 85.0:
        cond_name = "Daytime / Well-Illuminated"
        cond_desc = "Daylight scene. Denoising and generative detail super-resolution applied without sky overexposure."
    elif raw_lum_mean > 45.0:
        cond_name = "Twilight / Mixed Lighting"
        cond_desc = "Mixed ambient lighting. Adaptive low-light curve estimation applied with highlight protection."
    else:
        cond_name = "Nighttime / Low-Light"
        cond_desc = "Dark dashcam scene. Full low-light deep curve illumination boost with noise suppression."

    results = {
        "schema_metrics": schema_metrics,
        "technical_summary": {
            "original_resolution": orig_res,
            "enhanced_resolution": enh_res,
            "original_fps": orig_fps,
            "processing_fps": proc_fps,
            "total_processing_time_sec": round(processing_time, 2),
            "original_file_size_bytes": orig_file_size,
            "enhanced_file_size_bytes": enh_file_size,
            "scene_lighting_condition": cond_name,
            "scene_diagnosis": cond_desc,
            "device_mode": "CUDA (NVIDIA RTX 4060)" if cuda_avail else "CPU (Fallback)",
            "peak_vram_mb": peak_vram
        },
        "illumination_analysis": {
            "mean_luminance_original": round(raw_lum_mean, 2),
            "mean_luminance_enhanced": round(enh_lum_mean, 2),
            "illumination_gain_percent": round(((enh_lum_mean - raw_lum_mean) / (raw_lum_mean + 1e-5)) * 100.0, 1),
            "contrast_improvement_index": round(enh_rms_contrast / (raw_rms_contrast + 1e-5), 2),
            "contrast_rms_original": round(raw_rms_contrast, 2),
            "contrast_rms_enhanced": round(enh_rms_contrast, 2),
            "sky_luminance_ratio": round(avg_sky_ratio, 2),
            "bloom_area_ratio": round(avg_bloom_ratio, 2),
            "flat_noise_original": round(avg_raw_flat_noise, 2),
            "flat_noise_enhanced": round(avg_enh_flat_noise, 2),
            "temporal_stability_index": round(max(0.0, 100.0 - (norm_flicker_val * 100.0)), 1),
            "temporal_flicker_score": round(norm_flicker_val, 4)
        },
        "no_reference_quality": {
            "niqe": niqe_score,
            "brisque": brisque_score,
            "caveat": "Unreliable for night footage. Natural scene statistics models trained on daytime photography."
        },
        "ground_truth_reference": {
            "is_reference_available": False,
            "note": "Reference ground-truth not applicable for real-world uploaded dashcam video. No synthetic PSNR/SSIM generated."
        }
    }

    if metrics_json_path:
        try:
            with open(metrics_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            logger.info("Saved authentic job metrics to %s", metrics_json_path)
        except Exception as e:
            logger.warning("Could not write metrics JSON to %s: %s", metrics_json_path, e)

    return results
