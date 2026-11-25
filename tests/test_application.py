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

Isolated to temporary storage directory via TEST_MODE.
"""

import os
import sys
import time
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2

# Set path and TEST_MODE prior to any app imports
sys.path.insert(0, os.path.abspath("."))
os.environ["TEST_MODE"] = "1"

import config
from app import app
from src.metadata import extract_video_metadata
from src.video_processor import ProcessingManager


class DashClarityTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Prepare sample videos in an isolated temporary directory."""
        cls.temp_dir = tempfile.mkdtemp(prefix="dashclarity_app_test_")
        config.set_test_mode_dir(Path(cls.temp_dir))

        app.config["TESTING"] = True
        cls.client = app.test_client()
        cls.test_dir = str(config.UPLOAD_DIR)
        
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
    def tearDownClass(cls):
        """Removes temporary test directory completely."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

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
        """Test GPU availability, adaptive profile, and hardware detection status endpoint."""
        import torch
        resp = self.client.get("/api/system/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("cuda_available", data)
        self.assertIn("adaptive_profile", data)
        if torch.cuda.is_available():
            self.assertTrue(data["cuda_available"])
            self.assertEqual(data["device_name"], torch.cuda.get_device_name(0))
            self.assertGreater(data["total_vram_mb"], 0)
        print("\n[TEST 01] GPU Status:", data["device_name"], "| VRAM:", data.get("total_vram_mb"), "MB | Tier:", data["adaptive_profile"]["tier"])

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
            
        # Corrupt file content rejected with friendly message and 400
        with open(self.corrupt_path, "rb") as f:
            resp2 = self.client.post("/api/upload", data={"video": (f, "corrupt.mp4")})
            self.assertEqual(resp2.status_code, 400)
            self.assertIn("Corrupt or invalid video container", resp2.get_json()["error"])
        print("\n[TEST 03] Invalid and corrupt uploads correctly rejected.")

    def test_04_nighttime_enhancement_pipeline(self):
        """Test actual enhancement on nighttime low-light dashcam video."""
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
        resp_dl.close()

        resp_stream = self.client.get("/media/enhanced/test_day_job")
        self.assertEqual(resp_stream.status_code, 200)
        self.assertEqual(resp_stream.mimetype, "video/mp4")
        self.assertEqual(resp_stream.headers.get("Accept-Ranges"), "bytes")
        resp_stream.close()
        print("\n[TEST 07] Media download & video streaming endpoints verified.")

    def test_08_adaptive_vram_profiles(self):
        """Verify adaptive profiles select optimal configs for RTX 4050 (6GB), RTX 4060 (8GB), and CPU."""
        from src.utils.hardware import select_gpu_profile
        p4050 = select_gpu_profile(6.0, cuda_available=True)
        self.assertEqual(p4050["tier"], "mid_vram_6gb")
        self.assertEqual(p4050["tile_size"], 384)
        self.assertTrue(p4050["sequential_offload"])
        self.assertLessEqual(p4050["sr_max_out_dim"], 2560)

        p4060 = select_gpu_profile(8.0, cuda_available=True)
        self.assertEqual(p4060["tier"], "high_vram_8gb")
        self.assertEqual(p4060["tile_size"], 512)
        self.assertGreaterEqual(p4060["tile_batch_size"], 2)

        p_cpu = select_gpu_profile(0.0, cuda_available=False)
        self.assertEqual(p_cpu["tier"], "cpu_fallback")
        self.assertFalse(p_cpu["fp16"])
        print("\n[TEST 08] Adaptive RTX 4050 (6GB) / RTX 4060 (8GB) / CPU profiles verified.")

    def test_09_odd_resolution_and_cpu_lowlight_safety(self):
        """Verify odd frame dimensions produce even H.264 output and CPU LowLightStage handles float32 cleanly."""
        import torch
        from src.pipeline.stages import LowLightStage
        from src.pipeline.video_stream import StreamingVideoProcessor

        # CPU LowLightStage float32 safety check (no .half() crash on CPU)
        ll_cpu = LowLightStage(torch.device("cpu"), model_type="zero_dce_plus", fp16=False)
        dark_odd = np.ones((181, 323, 3), dtype=np.uint8) * 18
        out_cpu, info = ll_cpu.process_frame(dark_odd)
        self.assertEqual(out_cpu.shape, (181, 323, 3))
        self.assertTrue(info["applied"])

        # Odd resolution stream processing -> even output resolution for libx264
        odd_in = os.path.join(self.test_dir, "test_odd_323x181.mp4")
        odd_out = os.path.join(self.test_dir, "test_odd_out.mp4")
        out_w_writer = cv2.VideoWriter(odd_in, cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (323, 181))
        for _ in range(5):
            out_w_writer.write(dark_odd.copy())
        out_w_writer.release()

        sp = StreamingVideoProcessor(odd_in, odd_out)
        res = sp.process(lambda f, p: ll_cpu.process_frame(f, p))
        self.assertEqual(res["out_width"] % 2, 0)
        self.assertEqual(res["out_height"] % 2, 0)
        self.assertEqual(res["total_frames_processed"], 5)
        self.assertTrue(os.path.exists(odd_out))
        print("\n[TEST 09] Odd resolution H.264 encoding and CPU float32 LowLight safety verified.")

    def test_10_stage_overrides_and_fps_metrics(self):
        """Verify UI stage_overrides are honored by pipeline and metrics report total_frames FPS."""
        pm = ProcessingManager.get_instance()
        pipeline = pm.get_pipeline()
        dark_frame = np.ones((240, 320, 3), dtype=np.uint8) * 20

        # Disable neural denoise via stage_overrides on balanced preset
        _, telemetry = pipeline.enhance_frame(
            dark_frame, dark_frame, preset="balanced",
            stage_overrides={"enable_denoise": False, "enable_low_light": True, "enable_super_res": False, "enable_temporal": False}
        )
        self.assertNotIn("neural_denoise", telemetry)
        self.assertIn("low_light", telemetry)
        self.assertNotIn("temporal", telemetry)

        # Verify total_frames_processed governs FPS calculation in evaluate_video_metrics
        from src.utils.metrics import evaluate_video_metrics
        orig_samples = [dark_frame] * 5
        enh_samples = [dark_frame + 40] * 5
        m = evaluate_video_metrics(
            sample_orig_frames=orig_samples,
            sample_enh_frames=enh_samples,
            processing_time=10.0,
            orig_file_size=1000,
            enh_file_size=2000,
            orig_res="320x240",
            enh_res="320x240",
            orig_fps=30.0,
            compute_pyiqa=False,
            total_frames_processed=300
        )
        self.assertAlmostEqual(m["technical_summary"]["processing_fps"], 30.0, places=2)
        print("\n[TEST 10] stage_overrides enforcement and true FPS telemetry verified.")


if __name__ == "__main__":
    unittest.main()
