"""
Test script for verifying end-to-end video enhancement pipeline.
Uses TEST_MODE to isolate files from production history and storage.
"""
import sys
import os
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))
os.environ["TEST_MODE"] = "1"

import config
import time
import json
import cv2
import numpy as np
from src.metadata import extract_video_metadata
from src.video_processor import ProcessingManager


def test_pipeline():
    temp_dir = tempfile.mkdtemp(prefix="dashclarity_pipe_test_")
    config.set_test_mode_dir(Path(temp_dir))

    try:
        # Create a tiny 15-frame test clip
        input_video = str(config.UPLOAD_DIR / "sample_pipe_dashcam.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(input_video, fourcc, 30.0, (640, 360))
        for _ in range(15):
            f = np.ones((360, 640, 3), dtype=np.uint8) * 30
            cv2.circle(f, (320, 180), 40, (120, 120, 100), -1)
            out.write(f)
        out.release()

        print("Testing metadata extraction...")
        meta = extract_video_metadata(input_video)
        print("Metadata extracted successfully:")
        print(json.dumps(meta, indent=2))
        
        print("\nStarting video enhancement pipeline on RTX 4060...")
        pm = ProcessingManager.get_instance()
        job = pm.create_job("test_dash_001", input_video)
        pm.start_enhancement_async("test_dash_001", max_frames=15, preset="balanced")
        
        for _ in range(60):
            status = job.to_dict()
            print(f"[{status['status'].upper()}] {status['progress_percent']}% | {status['stage']} | FPS: {status['current_fps']}")
            if status['status'] in ['completed', 'failed', 'cancelled']:
                break
            time.sleep(1.0)
            
        print("\n=== PIPELINE FINISHED ===")
        print(json.dumps(status, indent=2))
        assert status['status'] == 'completed', f"Job failed: {status.get('error')}"
        assert status['has_enhanced_video'], "Enhanced output MP4 file was not created!"
        print("\nALL ASSERTIONS PASSED! Pipeline verified successfully.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_pipeline()
