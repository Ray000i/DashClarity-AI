"""
DashClarity AI - Automated UI Screenshot Capture Script
Uses Microsoft Edge headless to capture pixel-accurate screenshots
at 1366x768 (Desktop) and 390x844 (Mobile) for all redesigned screens.
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path

# Add project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app import app, processing_manager

EDGE_BIN = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
SHOTS_DIR = BASE_DIR / "docs" / "screenshots"
SHOTS_DIR.mkdir(parents=True, exist_ok=True)

sample_job_id = "87b5a0d729f6"
if not processing_manager.get_job(sample_job_id):
    jobs = processing_manager.list_history()
    sample_job_id = jobs[0]["job_id"] if jobs else "test_ui_dummy_job"

# Seed a realistic active processing job so processing.html can be screenshotted live
from src.video_processor import JobState
proc_job = JobState(job_id="demo_processing", input_path="input_videos/real/real_night_dashcam_01.mp4")
proc_job.status = "processing"
proc_job.progress = 68.0
proc_job.current_stage = "neural_denoise"
proc_job.processed_frames = 102
proc_job.total_frames = 150
proc_job.fps = 23.4
proc_job.eta_seconds = 2.1
proc_job.logs = [
    "Initializing pipeline with preset: balanced",
    "Stage 1: Zero-DCE++ low-light curve estimation complete",
    "Stage 2: SCUNet neural denoising frame 102/150...",
    "VRAM allocation: 2.18 GB / 8.00 GB (NVIDIA RTX 4060 Laptop)"
]
processing_manager.jobs["demo_processing"] = proc_job

print(f"Using Job ID for results captures: {sample_job_id}")

screens = [
    ("desktop_01_index.png", "http://127.0.0.1:5000/", 1366, 768),
    ("desktop_02_upload.png", "http://127.0.0.1:5000/upload", 1366, 768),
    ("desktop_03_processing.png", "http://127.0.0.1:5000/processing/demo_processing", 1366, 768),
    ("desktop_04_results.png", f"http://127.0.0.1:5000/results/{sample_job_id}", 1366, 768),
    ("desktop_04b_results_side.png", f"http://127.0.0.1:5000/results/{sample_job_id}?mode=side", 1366, 768),
    ("desktop_05_history.png", "http://127.0.0.1:5000/history", 1366, 768),
    ("desktop_06_about.png", "http://127.0.0.1:5000/about", 1366, 768),
    ("mobile_01_results.png", f"http://127.0.0.1:5000/results/{sample_job_id}", 390, 844),
    ("mobile_02_upload.png", "http://127.0.0.1:5000/upload", 390, 844),
]

def run_server():
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(2.5)

print("Server started. Capturing screenshots via Edge headless...")

for filename, url, width, height in screens:
    out_path = SHOTS_DIR / filename
    cmd = [
        EDGE_BIN,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--window-size={width},{height}",
        f"--screenshot={str(out_path)}",
        "--hide-scrollbars",
        url
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        if out_path.exists() and out_path.stat().st_size > 500:
            print(f"Captured: {filename} ({width}x{height}) - {out_path.stat().st_size} bytes")
        else:
            print(f"Warning: {filename} failed to capture (returncode {res.returncode})")
    except Exception as e:
        print(f"Error capturing {filename}: {e}")

print("Screenshot capture sequence finished.")
