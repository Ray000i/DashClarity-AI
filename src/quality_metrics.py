"""
DashClarity AI - Video Quality & Technical Metrics Module
Calculates authentic visual enhancement statistics, illumination gain,
contrast improvement, temporal stability, and performance benchmarks.
Adheres strictly to scientific standards: PSNR/SSIM are only reported
when paired ground-truth reference footage is present.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List

def compute_frame_luminance_stats(frame_bgr: np.ndarray) -> Dict[str, float]:
    """
    Computes luminance mean, standard deviation (RMS contrast), and percentile metrics.
    """
    # Convert BGR to YCrCb and take Y (luminance) channel
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0].astype(np.float32)
    
    mean_lum = float(np.mean(y_channel))
    std_lum = float(np.std(y_channel))
    p10 = float(np.percentile(y_channel, 10))
    p90 = float(np.percentile(y_channel, 90))
    dynamic_range = p90 - p10
    
    return {
        "mean_luminance": mean_lum,
        "contrast_rms": std_lum,
        "dynamic_range": dynamic_range
    }

def calculate_enhancement_metrics(
    sample_orig_frames: List[np.ndarray],
    sample_enh_frames: List[np.ndarray],
    processing_time: float,
    orig_file_size: int,
    enh_file_size: int,
    orig_res: str,
    enh_res: str,
    orig_fps: float,
    ground_truth_frames: Optional[List[np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Calculates technical metrics between original and enhanced video samples.
    """
    num_samples = min(len(sample_orig_frames), len(sample_enh_frames))
    if num_samples == 0:
        return {}
        
    orig_means = []
    enh_means = []
    orig_contrasts = []
    enh_contrasts = []
    
    for i in range(num_samples):
        s_orig = compute_frame_luminance_stats(sample_orig_frames[i])
        s_enh = compute_frame_luminance_stats(sample_enh_frames[i])
        orig_means.append(s_orig["mean_luminance"])
        enh_means.append(s_enh["mean_luminance"])
        orig_contrasts.append(s_orig["contrast_rms"])
        enh_contrasts.append(s_enh["contrast_rms"])
        
    avg_orig_lum = float(np.mean(orig_means))
    avg_enh_lum = float(np.mean(enh_means))
    avg_orig_contrast = float(np.mean(orig_contrasts))
    avg_enh_contrast = float(np.mean(enh_contrasts))
    
    # Illumination Gain percentage
    lum_gain_pct = ((avg_enh_lum - avg_orig_lum) / (avg_orig_lum + 1e-5)) * 100.0
    
    # Contrast Improvement Index (CII)
    cii = avg_enh_contrast / (avg_orig_contrast + 1e-5)
    
    # Temporal Consistency Metric across samples
    # Measures smooth inter-frame luminance gradient continuity
    temporal_flicker_score = 0.0
    if num_samples >= 2:
        diffs = []
        for i in range(1, num_samples):
            d_orig = cv2.absdiff(sample_orig_frames[i], sample_orig_frames[i-1])
            d_enh = cv2.absdiff(sample_enh_frames[i], sample_enh_frames[i-1])
            diffs.append(float(np.mean(cv2.absdiff(d_enh, d_orig))))
        temporal_flicker_score = float(np.mean(diffs))
        
    processing_fps = round(num_samples / processing_time, 2) if processing_time > 0 else 0.0
    
    results = {
        "technical_summary": {
            "original_resolution": orig_res,
            "enhanced_resolution": enh_res,
            "original_fps": orig_fps,
            "processing_fps": processing_fps,
            "total_processing_time_sec": round(processing_time, 2),
            "original_file_size_bytes": orig_file_size,
            "enhanced_file_size_bytes": enh_file_size,
        },
        "illumination_analysis": {
            "mean_luminance_original": round(avg_orig_lum, 2),
            "mean_luminance_enhanced": round(avg_enh_lum, 2),
            "illumination_gain_percent": round(lum_gain_pct, 1),
            "contrast_improvement_index": round(cii, 2),
            "temporal_stability_index": round(max(0.0, 100.0 - temporal_flicker_score), 1)
        },
        "ground_truth_reference": {
            "is_reference_available": False,
            "note": "Reference Ground-Truth Not Applicable for real-world uploaded dashcam video. No fake PSNR/SSIM generated."
        }
    }
    
    # If paired ground truth is provided, compute real PSNR / SSIM
    if ground_truth_frames and len(ground_truth_frames) >= num_samples:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        psnr_vals = []
        ssim_vals = []
        for i in range(num_samples):
            gt = ground_truth_frames[i]
            enh = sample_enh_frames[i]
            psnr_vals.append(peak_signal_noise_ratio(gt, enh))
            ssim_vals.append(structural_similarity(gt, enh, channel_axis=2))
            
        results["ground_truth_reference"] = {
            "is_reference_available": True,
            "psnr_db": round(float(np.mean(psnr_vals)), 2),
            "ssim": round(float(np.mean(ssim_vals)), 4),
            "note": "Evaluated against paired ground-truth reference frames."
        }
        
    return results

