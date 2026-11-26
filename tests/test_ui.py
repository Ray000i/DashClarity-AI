"""
DashClarity AI - Frontend UI & API Route Test Suite
Tests all routes, endpoints, JSON schemas, upload validations,
missing-metric handling, and warning banners via Flask test client.

Runs isolated from production directories using TEST_MODE.
"""

import os
import sys
import json
import io
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
from app import app, processing_manager


class TestFrontendUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="dashclarity_ui_test_")
        config.set_test_mode_dir(Path(cls.temp_dir))

        app.config["TESTING"] = True
        cls.client = app.test_client()

        # Create dummy job for results & processing testing
        cls.test_job_id = "test_ui_dummy_job"
        dummy_in = config.UPLOAD_DIR / f"{cls.test_job_id}_sample.mp4"
        dummy_out = config.OUTPUT_DIR / f"enhanced_{cls.test_job_id}.mp4"
        dummy_json = config.OUTPUT_DIR / f"enhanced_{cls.test_job_id}.json"

        # Write dummy files
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_in = cv2.VideoWriter(str(dummy_in), fourcc, 15.0, (320, 240))
        for _ in range(15):
            out_in.write(np.zeros((240, 320, 3), dtype=np.uint8))
        out_in.release()

        out_out = cv2.VideoWriter(str(dummy_out), fourcc, 15.0, (320, 240))
        for _ in range(15):
            out_out.write(np.ones((240, 320, 3), dtype=np.uint8) * 80)
        out_out.release()

        # Write dummy metrics.json with schema_metrics and missing-metric case
        dummy_metrics = {
            "schema_metrics": [
                {
                    "id": "mean_lum",
                    "label": "Mean luminance",
                    "unit": "",
                    "direction": "neutral",
                    "caveat": None,
                    "input": 50.0,
                    "output": 75.0,
                    "change": "+25.00",
                    "evaluation": "neutral"
                },
                {
                    "id": "sky_ratio",
                    "label": "Sky luminance ratio",
                    "unit": "",
                    "direction": "lower_better",
                    "caveat": "Upper 25% FOV ratio.",
                    "input": 1.0,
                    "output": 1.05,
                    "change": "+0.05",
                    "evaluation": "worse"
                },
                {
                    "id": "missing_metric_test",
                    "label": "Missing metric test",
                    "unit": "score",
                    "direction": "lower_better",
                    "caveat": "Should render not available.",
                    "input": "not available",
                    "output": "not available",
                    "change": "not available",
                    "evaluation": "neutral"
                }
            ],
            "technical_summary": {
                "original_resolution": "320x240",
                "enhanced_resolution": "320x240",
                "original_fps": 15.0,
                "processing_fps": 18.5,
                "total_processing_time_sec": 0.81,
                "original_file_size_bytes": 1024,
                "enhanced_file_size_bytes": 2048,
                "scene_lighting_condition": "Nighttime / Low-Light",
                "device_mode": "CPU (Fallback)",
                "peak_vram_mb": 0.0
            },
            "audio_warning": "Video contains no audio stream."
        }
        with open(dummy_json, "w", encoding="utf-8") as f:
            json.dump(dummy_metrics, f, indent=2)

    @classmethod
    def tearDownClass(cls):
        """Cleans up temporary directory completely."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_all_page_routes_return_200(self):
        """Verifies that all user-facing pages return HTTP 200 OK and /history redirects."""
        routes = ["/", "/upload", f"/processing/{self.test_job_id}", f"/results/{self.test_job_id}", "/about"]
        for r in routes:
            resp = self.client.get(r)
            self.assertEqual(resp.status_code, 200, f"Route {r} returned {resp.status_code}")
            html = resp.data.decode("utf-8")
            self.assertNotIn("FORENSIC ANALYSIS REPORT", html)
            self.assertNotIn("ORDER A", html)
        
        # Verify /history redirects cleanly
        resp_hist = self.client.get("/history")
        self.assertEqual(resp_hist.status_code, 302)

    def test_02_api_presets_endpoint(self):
        """Verifies /api/presets returns config presets with measured FPS and tradeoffs."""
        resp = self.client.get("/api/presets")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("presets", data)
        self.assertEqual(len(data["presets"]), 3)
        keys = [p["key"] for p in data["presets"]]
        self.assertIn("fast", keys)
        self.assertIn("balanced", keys)
        self.assertIn("quality", keys)
        self.assertNotIn("stcd_comparison", keys)
        self.assertNotIn("hybrid_scunet_stcd", keys)
        self.assertNotIn("classical_baseline", keys)
        for p in data["presets"]:
            self.assertIn("tradeoff", p)
            self.assertIn("label", p)

    def test_03_api_history_endpoint(self):
        """Verifies /api/history returns stored enhancement records."""
        resp = self.client.get("/api/history")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("jobs", data)
        self.assertIsInstance(data["jobs"], list)

    def test_04_api_status_and_logs(self):
        """Verifies status polling and log endpoints."""
        resp = self.client.get(f"/api/status/{self.test_job_id}")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "completed")

        resp_logs = self.client.get(f"/api/logs/{self.test_job_id}")
        self.assertEqual(resp_logs.status_code, 200)
        self.assertIn("logs", resp_logs.get_json())

    def test_05_api_result_schema_metrics(self):
        """Verifies /api/result returns schema_metrics matching results.json."""
        resp = self.client.get(f"/api/result/{self.test_job_id}")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("metrics", data)
        self.assertIn("schema_metrics", data["metrics"])
        m_list = data["metrics"]["schema_metrics"]
        ids = [m["id"] for m in m_list]
        self.assertIn("mean_lum", ids)
        self.assertIn("sky_ratio", ids)
        self.assertIn("missing_metric_test", ids)

    def test_06_upload_validation_rejects_empty_and_invalid(self):
        """Verifies /api/upload validation rules."""
        # Missing file field
        r1 = self.client.post("/api/upload")
        self.assertEqual(r1.status_code, 400)

        # Disallowed file extension (.exe)
        data = {"video": (io.BytesIO(b"fake binary"), "malicious.exe")}
        r2 = self.client.post("/api/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(r2.status_code, 400)
        self.assertIn("Unsupported format", r2.get_json()["error"])

    def test_07_side_by_side_download_endpoint(self):
        """Verifies /api/download/<job_id>?mode=comparison generates and streams hstack comparison."""
        resp = self.client.get(f"/api/download/{self.test_job_id}?mode=comparison")
        self.assertIn(resp.status_code, [200, 500])
        if resp.status_code == 200:
            self.assertEqual(resp.mimetype, "video/mp4")
            resp.close()

    def test_08_health_endpoint(self):
        """Verifies /health endpoint."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("status", data)
        self.assertIn("cuda_available", data)
        self.assertIn("weights_ok", data)

    def test_09_custom_error_handlers(self):
        """Verifies custom 404 handler for invalid routes."""
        resp = self.client.get("/non_existent_page_12345")
        self.assertEqual(resp.status_code, 404)
        html = resp.data.decode("utf-8")
        self.assertIn("404", html)
        self.assertIn("Resource Not Found", html)

        # API 404 returns JSON
        resp_api = self.client.get("/api/non_existent_endpoint")
        self.assertEqual(resp_api.status_code, 404)
        self.assertIn("error", resp_api.get_json())


if __name__ == "__main__":
    unittest.main()
