"""
DashClarity AI - Video Quality & Technical Metrics Module
Calculates authentic visual enhancement statistics, illumination gain,
contrast improvement, temporal stability, color saturation retention,
sharpness preservation, and highlight protection.
Adheres strictly to scientific standards: PSNR/SSIM are only reported
when paired ground-truth reference footage is present.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List

def compute_frame_luminance_stats(frame_bgr: np.ndarray) -> Dict[str, float]:
    """
    Computes luminance mean, standard deviation (RMS contrast), percentile metrics,
    Laplacian sharpness, HSV saturation, and highlight clipping ratio.
    """
    # Convert BGR to YCrCb and take Y (luminance) channel
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0].astype(np.float32)
    
    mean_lum = float(np.mean(y_channel))
    std_lum = float(np.std(y_channel))
    p10 = float(np.percentile(y_channel, 10))
    p90 = float(np.percentile(y_channel, 90))
    dynamic_range = p90 - p10
    
    # Sharpness (Laplacian variance on grayscale)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # Saturation (mean S channel in HSV)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[:, :, 1].mean())
    
    # Highlight clipping (> 240 in all or any channel)
    overexp_pct = float(np.mean(frame_bgr > 240) * 100.0)
    
    return {
        "mean_luminance": mean_lum,
        "contrast_rms": std_lum,
        "dynamic_range": dynamic_range,
        "sharpness_laplacian": lap_var,
        "mean_saturation": mean_sat,
        "overexposed_pct": overexp_pct
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
    orig_sharpness = []
    enh_sharpness = []
    orig_sats = []
    enh_sats = []
    orig_overexps = []
    enh_overexps = []
    
    for i in range(num_samples):
        s_orig = compute_frame_luminance_stats(sample_orig_frames[i])
        s_enh = compute_frame_luminance_stats(sample_enh_frames[i])
        orig_means.append(s_orig["mean_luminance"])
        enh_means.append(s_enh["mean_luminance"])
        orig_contrasts.append(s_orig["contrast_rms"])
        enh_contrasts.append(s_enh["contrast_rms"])
        orig_sharpness.append(s_orig["sharpness_laplacian"])
        enh_sharpness.append(s_enh["sharpness_laplacian"])
        orig_sats.append(s_orig["mean_saturation"])
        enh_sats.append(s_enh["mean_saturation"])
        orig_overexps.append(s_orig["overexposed_pct"])
        enh_overexps.append(s_enh["overexposed_pct"])
        
    avg_orig_lum = float(np.mean(orig_means))
    avg_enh_lum = float(np.mean(enh_means))
    avg_orig_contrast = float(np.mean(orig_contrasts))
    avg_enh_contrast = float(np.mean(enh_contrasts))
    avg_orig_sharp = float(np.mean(orig_sharpness))
    avg_enh_sharp = float(np.mean(enh_sharpness))
    avg_orig_sat = float(np.mean(orig_sats))
    avg_enh_sat = float(np.mean(enh_sats))
    avg_orig_overexp = float(np.mean(orig_overexps))
    avg_enh_overexp = float(np.mean(enh_overexps))
    
    # Illumination Gain percentage
    lum_gain_pct = ((avg_enh_lum - avg_orig_lum) / (avg_orig_lum + 1e-5)) * 100.0
    
    # Contrast Improvement Index (CII)
    cii = avg_enh_contrast / (avg_orig_contrast + 1e-5)
    
    # Temporal Consistency Metric across samples
    temporal_flicker_score = 0.0
    if num_samples >= 2:
        diffs = []
        for i in range(1, num_samples):
            d_orig = cv2.absdiff(sample_orig_frames[i], sample_orig_frames[i-1])
            d_enh = cv2.absdiff(sample_enh_frames[i], sample_enh_frames[i-1])
            # Ensure spatial dimensions match before calculating difference
            if d_orig.shape[:2] != d_enh.shape[:2]:
                d_orig = cv2.resize(d_orig, (d_enh.shape[1], d_enh.shape[0]), interpolation=cv2.INTER_LINEAR)
            diffs.append(float(np.mean(cv2.absdiff(d_enh, d_orig))))
        temporal_flicker_score = float(np.mean(diffs)) if diffs else 0.0
        
    processing_fps = round(num_samples / processing_time, 2) if processing_time > 0 else 0.0
    
    # Lighting condition diagnosis
    if avg_orig_lum > 85.0:
        condition_name = "Daytime / Well-Illuminated"
        condition_desc = "Optimal daylight exposure detected. Highlights preserved with adaptive dynamic range balancing."
    elif avg_orig_lum > 45.0:
        condition_name = "Twilight / Mixed Lighting"
        condition_desc = "Mixed ambient lighting detected. STCD spatial-temporal decomposition applied with highlight protection."
    else:
        condition_name = "Nighttime / Low-Light"
        condition_desc = "Low-light conditions detected. Full STCD spatial-temporal neural enhancement applied."
    
    results = {
        "technical_summary": {
            "original_resolution": orig_res,
            "enhanced_resolution": enh_res,
            "original_fps": orig_fps,
            "processing_fps": processing_fps,
            "total_processing_time_sec": round(processing_time, 2),
            "original_file_size_bytes": orig_file_size,
            "enhanced_file_size_bytes": enh_file_size,
            "scene_lighting_condition": condition_name,
            "scene_diagnosis": condition_desc
        },
        "illumination_analysis": {
            "mean_luminance_original": round(avg_orig_lum, 2),
            "mean_luminance_enhanced": round(avg_enh_lum, 2),
            "illumination_gain_percent": round(lum_gain_pct, 1),
            "contrast_improvement_index": round(cii, 2),
            "temporal_stability_index": round(max(0.0, 100.0 - temporal_flicker_score), 1),
            "color_saturation_original": round(avg_orig_sat, 1),
            "color_saturation_enhanced": round(avg_enh_sat, 1),
            "sharpness_laplacian_original": round(avg_orig_sharp, 1),
            "sharpness_laplacian_enhanced": round(avg_enh_sharp, 1),
            "overexposed_pixels_original_pct": round(avg_orig_overexp, 2),
            "overexposed_pixels_enhanced_pct": round(avg_enh_overexp, 2)
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
            if gt.shape[:2] != enh.shape[:2]:
                gt = cv2.resize(gt, (enh.shape[1], enh.shape[0]), interpolation=cv2.INTER_LINEAR)
            psnr_vals.append(peak_signal_noise_ratio(gt, enh))
            ssim_vals.append(structural_similarity(gt, enh, channel_axis=2))
            
        results["ground_truth_reference"] = {
            "is_reference_available": True,
            "psnr_db": round(float(np.mean(psnr_vals)), 2),
            "ssim": round(float(np.mean(ssim_vals)), 4),
            "note": "Evaluated against paired ground-truth reference frames."
        }
        
    return results
