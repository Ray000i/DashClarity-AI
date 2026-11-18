import sys
import os
sys.path.insert(0, os.path.abspath("."))
import time
import json
from src.video_processor import ProcessingManager

pm = ProcessingManager.get_instance()
job = pm.create_job("95fd223e9fe3", "input_videos/95fd223e9fe3_19960474-hd_1080_1920_30fps.mp4")
pm.start_enhancement_async("95fd223e9fe3", max_frames=60, super_res=False)

while True:
    st = job.to_dict()
    print(f"[{st['status'].upper()}] {st['progress_percent']}% | {st['stage']} | FPS: {st['current_fps']}")
    if st['status'] in ['completed', 'failed', 'cancelled']:
        break
    time.sleep(2.0)

print("\nProcessing completed successfully!")
print(json.dumps(st['metrics'], indent=2))
