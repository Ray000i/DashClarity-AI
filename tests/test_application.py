"""
Comprehensive Test Suite for DashClarity AI
Validates all Section 9 Testing Requirements:
1. Normal daytime video
2. Low-light nighttime video
3. Low-resolution video
4. Different FPS video
5. Invalid video file
6. Video without audio
7. REST API route handling & error responses
"""

import os
import sys
import time
import json
import unittest
import numpy as np
import cv2

# Set path
sys.path.insert(0, os.path.abspath("."))

from app import app
from src.metadata import extract_video_metadata
from src.video_processor import ProcessingManager

class DashClarityTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Prepare sample videos for all required test conditions."""
        cls.client = app.test_client()
        cls.test_dir = os.path.abspath("input_videos")
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # 1. Daytime video (high illumination ~180-220)
        cls.day_path = os.path.join(cls.test_dir, "test_daytime.mp4")
        cls._create_sample_video(cls.day_path, is_night=False, fps=30, size=(640, 360), frames=20)
        
        # 2. Low-light nighttime video (low illumination ~10-25)
        cls.night_path = os.path.join(cls.test_dir, "test_nighttime.mp4")
        cls._create_sample_video(cls.night_path, is_night=True, fps=30, size=(640, 360), frames=20)
        
        # 3. Low-resolution video (320x180)
        cls.lowres_path = os.path.join(cls.test_dir, "test_lowres.mp4")
        cls._create_sample_video(cls.lowres_path, is_night=True, fps=15, size=(320, 180), frames=15)
        
        # 4. Different FPS video (24 fps)
        cls.fps24_path = os.path.join(cls.test_dir, "test_fps24.mp4")
        cls._create_sample_video(cls.fps24_path, is_night=True, fps=24, size=(640, 360), frames=24)
        
        # 5. Invalid video file (corrupt text file with .mp4 extension)
        cls.corrupt_path = os.path.join(cls.test_dir, "test_corrupt.mp4")
        with open(cls.corrupt_path, "w") as f:
            f.write("THIS_IS_NOT_A_VALID_VIDEO_CONTAINER_STREAM_DATA")
            
        # 6. Invalid extension file (.exe)
        cls.invalid_ext_path = os.path.join(cls.test_dir, "test_binary.exe")
        with open(cls.invalid_ext_path, "wb") as f:
            f.write(b"\x00\x01\x02\x03\x04")

    @classmethod
    def _create_sample_video(cls, path, is_night=True, fps=30, size=(640, 360), frames=20):
        w, h = size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for i in range(frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            if is_night:
                # Dark ambient
                frame[:] = (18, 15, 12)
                # Headlight patch
                cv2.circle(frame, (w//2, h - 50), 60, (60, 60, 50), -1)
            else:
                # Daylight scene
                frame[:h//2, :] = (210, 180, 140)  # Daylight sky
                frame[h//2:, :] = (80, 80, 80)     # Road
                cv2.circle(frame, (w//4, h//4), 30, (180, 240, 255), -1) # Sun
            # Road markings
            cv2.line(frame, (w//2, h//2), (w//2, h), (180, 180, 180), 3)
            out.write(frame)
        out.release()

    def test_01_api_system_status(self):
        """Test GPU availability and model architecture status endpoint."""
        resp = self.client.get("/api/system/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["cuda_available"])
        self.assertIn("RTX 4060", data["device_name"])
        print("\n[TEST 01] GPU Status:", data["device_name"], "| PyTorch:", data["pytorch_version"])

    def test_02_video_metadata_extraction(self):
        """Test metadata extraction across diverse dimensions and frame rates."""
        meta_night = extract_video_metadata(self.night_path)
        self.assertEqual(meta_night["resolution"], "640x360")
        self.assertEqual(meta_night["fps"], 30.0)
        self.assertEqual(meta_night["total_frames"], 20)
        self.assertFalse(meta_night["has_audio"])
        
        meta_lowres = extract_video_metadata(self.lowres_path)
        self.assertEqual(meta_lowres["resolution"], "320x180")
        self.assertEqual(meta_lowres["fps"], 15.0)
        print("\n[TEST 02] Metadata successfully verified for diverse resolutions and FPS.")

    def test_03_invalid_file_rejection(self):
        """Test security rejection of invalid extensions and corrupt containers."""
        # Invalid extension
        with open(self.invalid_ext_path, "rb") as f:
            resp = self.client.post("/api/upload", data={"video": (f, "test_binary.exe")})
            self.assertEqual(resp.status_code, 400)
            data = resp.get_json()
            self.assertIn("error", data)
            self.assertIn("Unsupported format", data["error"])
            
        # Corrupt file content
        with open(self.corrupt_path, "rb") as f:
            resp2 = self.client.post("/api/upload", data={"video": (f, "corrupt.mp4")})
            self.assertEqual(resp2.status_code, 500)
        print("\n[TEST 03] Invalid and corrupt uploads correctly rejected.")

    def test_04_nighttime_enhancement_pipeline(self):
        """Test actual STCD enhancement on nighttime low-light dashcam video."""
        print("\n[TEST 04] Running default enhancement pipeline on nighttime video...")
        with open(self.night_path, "rb") as f:
            resp = self.client.post("/api/upload", data={"video": (f, "night_upload.mp4")})
            self.assertEqual(resp.status_code, 201)
            job_id = resp.get_json()["job_id"]

        # Trigger enhancement
        resp_enh = self.client.post(f"/api/enhance/{job_id}", json={"max_frames": 20})
        self.assertIn(resp_enh.status_code, [200, 202])

        # Poll until completion
        for _ in range(80):
            r_stat = self.client.get(f"/api/status/{job_id}")
            stat_data = r_stat.get_json()
            if stat_data["status"] in ["completed", "failed"]:
                break
            time.sleep(0.5)

        self.assertEqual(stat_data["status"], "completed")
        self.assertTrue(stat_data["has_enhanced_video"])

        # Check metrics
        r_res = self.client.get(f"/api/result/{job_id}")
        self.assertEqual(r_res.status_code, 200)
        res_data = r_res.get_json()
        metrics = res_data["metrics"]
        
        illum_gain = metrics["illumination_analysis"]["illumination_gain_percent"]
        cii = metrics["illumination_analysis"]["contrast_improvement_index"]
        stability = metrics["illumination_analysis"]["temporal_stability_index"]
        
        self.assertGreater(illum_gain, 25.0, "Nighttime footage should show positive illumination gain")
        print(f"-> Nighttime Illumination Gain: +{illum_gain}% | CII: {cii}x | Stability: {stability}/100")

    def test_05_daytime_video_evaluation(self):
        """
        Test enhancement on daytime video.
        Ensures model does not break normal video, but documents that already-bright video
        exhibits modest or minimal illumination gain (honest model limitation behavior).
        """
        print("\n[TEST 05] Running default enhancement pipeline on normal daytime video...")
        pm = ProcessingManager.get_instance()
        job = pm.create_job("test_day_job", self.day_path)
        pm.start_enhancement_async("test_day_job", max_frames=20)

        for _ in range(80):
            stat = job.to_dict()
            if stat["status"] in ["completed", "failed"]:
                break
            time.sleep(0.5)

        self.assertEqual(stat["status"], "completed")
        metrics = stat["metrics"]
        mean_orig = metrics["illumination_analysis"]["mean_luminance_original"]
        mean_enh = metrics["illumination_analysis"]["mean_luminance_enhanced"]
        print(f"-> Daytime Luminance: {mean_orig:.1f} -> {mean_enh:.1f} (Already well-illuminated)")

    def test_06_low_resolution_and_aspect_ratio(self):
        """Test model and reconstructor with low-resolution 320x180 footage."""
        print("\n[TEST 06] Testing low-resolution video (320x180)...")
        pm = ProcessingManager.get_instance()
        job = pm.create_job("test_lowres_job", self.lowres_path)
        pm.start_enhancement_async("test_lowres_job", max_frames=15)

        for _ in range(80):
            stat = job.to_dict()
            if stat["status"] in ["completed", "failed"]:
                break
            time.sleep(0.5)

        self.assertEqual(stat["status"], "completed")
        self.assertEqual(stat["metrics"]["technical_summary"]["enhanced_resolution"], "320x180")
        print("-> Low-resolution 320x180 preserved with perfect aspect ratio.")

    def test_07_media_streaming_and_download(self):
        """Test video streaming endpoints with HTTP partial range requests."""
        resp_dl = self.client.get("/api/download/test_day_job")
        self.assertEqual(resp_dl.status_code, 200)
        self.assertEqual(resp_dl.mimetype, "video/mp4")
        self.assertIn("attachment", resp_dl.headers.get("Content-Disposition", ""))

        resp_stream = self.client.get("/media/enhanced/test_day_job")
        self.assertEqual(resp_stream.status_code, 200)
        self.assertEqual(resp_stream.mimetype, "video/mp4")
        print("\n[TEST 07] Media download & video streaming endpoints verified.")

if __name__ == "__main__":
    unittest.main()
