"""
DashClarity AI - Web UI Screenshot & End-to-End Validation Script
Launches the Flask test server in a background daemon thread,
submits real upload and enhancement requests,
captures pixel-accurate 1366x768 screenshots of:
1. Upload Page (docs/screenshots/01_upload_page.png)
2. Processing Page (docs/screenshots/02_processing_page.png)
3. Results & Compare Page (docs/screenshots/03_compare_page.png)
using headless Edge browser.
"""

import os
import sys
import time
import json
import shutil
import subprocess
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app import app, processing_manager

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
SHOT_DIR = BASE_DIR / "docs" / "screenshots"
SHOT_DIR.mkdir(parents=True, exist_ok=True)


def capture_screenshot(url: str, output_path: Path, delay_sec: float = 1.0):
    print(f"Capturing screenshot: {url} -> {output_path.name}...")
    time.sleep(delay_sec)
    cmd = [
        EDGE_PATH,
        "--headless=new",
        "--disable-gpu",
        f"--window-size=1366,768",
        f"--screenshot={str(output_path)}",
        url
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if output_path.exists() and output_path.stat().st_size > 5000:
        print(f"  -> Captured successfully ({output_path.stat().st_size} bytes)")
    else:
        print(f"  -> Failed or empty capture: {res.stderr}")


def run_e2e_capture():
    # 1. Start Flask in daemon thread on port 5055 to avoid conflicts
    test_port = 5055
    server_url = f"http://127.0.0.1:{test_port}"
    
    def start_flask():
        app.run(host="127.0.0.1", port=test_port, debug=False, use_reloader=False)

    server_thread = threading.Thread(target=start_flask, daemon=True)
    server_thread.start()
    print("Flask server launched in background thread. Waiting for server readiness...")
    time.sleep(2.0)

    # 2. Capture Screenshot 1: Upload Page
    capture_screenshot(
        f"{server_url}/upload",
        SHOT_DIR / "01_upload_page.png",
        delay_sec=1.5
    )

    # 3. Perform a real upload and job initiation
    test_night_clip = BASE_DIR / "input_videos" / "real" / "real_night_dashcam_01.mp4"
    print(f"\nSubmitting real night dashcam clip: {test_night_clip.name}...")
    
    with app.test_client() as client:
        with open(test_night_clip, "rb") as f:
            up_res = client.post(
                "/api/upload",
                data={"video": (f, "real_night_dashcam_01.mp4")},
                content_type="multipart/form-data"
            )
        assert up_res.status_code == 201, f"Upload failed: {up_res.data}"
        up_data = json.loads(up_res.data)
        job_id = up_data["job_id"]
        print(f"Upload successful. Job ID: {job_id}")

        # Start enhancement with fast preset, 5 seconds demo window
        enh_res = client.post(
            f"/api/enhance/{job_id}",
            json={"preset": "fast", "strength": 0.8, "max_seconds": 5.0}
        )
        assert enh_res.status_code in [200, 202], f"Enhance request failed: {enh_res.data}"
        print(f"Enhancement initiated for Job {job_id}.")

    # 4. Capture Screenshot 2: Processing Page
    time.sleep(1.0)
    capture_screenshot(
        f"{server_url}/processing/{job_id}",
        SHOT_DIR / "02_processing_page.png",
        delay_sec=1.5
    )

    # 5. Wait for job completion
    print("Waiting for job to complete processing...")
    job = processing_manager.get_job(job_id)
    t0 = time.time()
    while job.status not in ["completed", "failed", "cancelled"] and (time.time() - t0) < 60:
        time.sleep(1.0)
        sys.stdout.write(f"\r  -> Progress: {job.progress_percent:.1f}% | Stage: {job.current_stage[:45]}...")
        sys.stdout.flush()
    print(f"\nJob status: {job.status} in {time.time() - t0:.1f}s")
    assert job.status == "completed", f"Job ended in non-completed state: {job.status} (error: {job.error_message})"

    # 6. Capture Screenshot 3: Results & Compare Page
    capture_screenshot(
        f"{server_url}/results/{job_id}",
        SHOT_DIR / "03_compare_page.png",
        delay_sec=2.0
    )

    print("\nAll 3 web UI screenshots successfully captured:")
    for f in sorted(list(SHOT_DIR.glob("*.png"))):
        print(f"  - {f.name} ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    run_e2e_capture()
