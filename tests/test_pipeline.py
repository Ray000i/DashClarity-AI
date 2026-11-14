"""
Test script for verifying end-to-end video enhancement pipeline.
"""
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import time
import json
from src.metadata import extract_video_metadata
from src.video_processor import ProcessingManager

def test_pipeline():
    input_video = "input_videos/sample_night_dashcam.mp4"
    print("Testing metadata extraction...")
    meta = extract_video_metadata(input_video)
    print("Metadata extracted successfully:")
    print(json.dumps(meta, indent=2))
    
    print("\nStarting video enhancement pipeline on RTX 4060...")
    pm = ProcessingManager.get_instance()
    job = pm.create_job("test_dash_001", input_video)
    pm.start_enhancement_async("test_dash_001", max_frames=30, super_res=False)
    
    while True:
        status = job.to_dict()
        print(f"[{status['status'].upper()}] {status['progress_percent']}% | {status['stage']} | FPS: {status['current_fps']}")
        if status['status'] in ['completed', 'failed', 'cancelled']:
            break
        time.sleep(1.0)
        
    print("\n=== PIPELINE FINISHED ===")
    print(json.dumps(status, indent=2))
    assert status['status'] == 'completed', f"Job failed: {status.get('error')}"
    assert status['has_enhanced_video'], "Enhanced output MP4 file was not created!"
    print("\nALL ASSERTIONS PASSED! Milestone 2 verified successfully.")

if __name__ == "__main__":
    test_pipeline()
