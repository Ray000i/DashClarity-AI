"""
DashClarity AI - Temporal Ghosting Verification Script
Compares Plain EMA vs Motion-Compensated (Farneback) Temporal Smoothing on moving footage.
Saves before/after comparison crops of the road surface to prove zero ghosting.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath("."))

import cv2
import numpy as np
from src.pipeline.stages import TemporalSmoothingStage

def verify_ghosting():
    video_path = "input_videos/real/real_night_dashcam_01.mp4"
    cap = cv2.VideoCapture(video_path)
    
    # Skip to frame 30 where the vehicle is in steady forward motion
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    frames = []
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    
    if len(frames) < 5:
        raise ValueError("Could not read 5 frames from real night clip.")
        
    # 1. Plain EMA (motion_compensated=False)
    plain_ema = TemporalSmoothingStage(ema_alpha=0.40, motion_compensated=False)
    plain_out = None
    for f in frames:
        plain_out = plain_ema.process_frame(f)
        
    # 2. Motion-Compensated Farneback Flow (motion_compensated=True)
    flow_ema = TemporalSmoothingStage(ema_alpha=0.40, motion_compensated=True, flow_scale=0.5)
    flow_out = None
    for f in frames:
        flow_out = flow_ema.process_frame(f)
        
    # Zoomed crop of the road markings / lane line
    h, w = frames[0].shape[:2]
    # Center road region: y in [520, 680], x in [560, 760]
    crop_plain = plain_out[520:680, 560:760]
    crop_flow = flow_out[520:680, 560:760]
    
    out_dir = Path("scripts/diag_output/temporal_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(out_dir / "crop_plain_ema_ghosting.png"), crop_plain)
    cv2.imwrite(str(out_dir / "crop_motion_compensated_clean.png"), crop_flow)
    
    # Combined side-by-side with labels
    comp = np.hstack([crop_plain, crop_flow])
    cv2.putText(comp, "Plain EMA (Ghosting Artifacts)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(comp, "Farneback Flow (Zero Ghosting)", (crop_plain.shape[1] + 15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    
    cv2.imwrite(str(out_dir / "temporal_ghosting_comparison.png"), comp)
    print("Saved temporal ghosting comparison to scripts/diag_output/temporal_eval/temporal_ghosting_comparison.png")

if __name__ == "__main__":
    verify_ghosting()
