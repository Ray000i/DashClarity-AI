"""
Unified regression tests for combined RTX 4050 (6 GB) & RTX 4060 (8 GB) optimizations:
- Adaptive GPU profiles & free-VRAM budget bounds
- SCUNet SDPA / channels-last equivalence (`src.models.scunet_fast`)
- Tiling OOM persistence (`tile_state`)
- LowLightStage continuous Retinex exposure ramp & GPU TemporalSmoothingStage tensor chaining
- BT.709 FFmpeg color tagging, audio trim muxing, and cancel partial-file cleanup
"""

import math
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from src.pipeline.tiling import run_tiled_inference
from src.pipeline.video_stream import StreamingVideoProcessor
from src.utils import get_ffmpeg_binary
from src.utils.hardware import detect_system_hardware, select_gpu_profile


def _ffprobe_info(path: str) -> str:
    res = subprocess.run([get_ffmpeg_binary(), "-i", path], capture_output=True, text=True)
    return res.stderr


class TestUnifiedAdaptiveHardware(unittest.TestCase):
    def test_profile_tiers_cover_4050_4060_and_cpu(self):
        p8 = select_gpu_profile(8.0, cuda_available=True, free_vram_gb=7.0)
        p6_free = select_gpu_profile(6.0, cuda_available=True, free_vram_gb=5.1)
        p6_tight = select_gpu_profile(6.0, cuda_available=True, free_vram_gb=2.5)
        p4 = select_gpu_profile(4.0, cuda_available=True, free_vram_gb=3.5)
        pcpu = select_gpu_profile(0.0, cuda_available=False)

        self.assertEqual(p8["tier"], "high_vram_8gb")
        self.assertGreater(p8["scunet_full_frame_max_px"], 1280 * 720)
        self.assertEqual(p6_free["tier"], "mid_vram_6gb")
        self.assertTrue(p6_free["sequential_offload"])
        self.assertGreater(p6_free["scunet_full_frame_max_px"], 1280 * 720)
        self.assertEqual(p6_tight["scunet_full_frame_max_px"], 0)
        self.assertEqual(p4["tier"], "low_vram")
        self.assertEqual(pcpu["tier"], "cpu_fallback")
        self.assertFalse(pcpu["fp16"])

    def test_hardware_detection_driver_field(self):
        hw = detect_system_hardware()
        self.assertIn("profile", hw)
        self.assertIn("driver_version", hw)


class TestTilingStatePersistence(unittest.TestCase):
    def test_tile_state_remembers_oom_reduction(self):
        calls = {"n": 0}

        class OOMOnLarge(torch.nn.Module):
            def forward(self, x):
                calls["n"] += 1
                if max(x.shape[2:]) > 300:
                    raise torch.cuda.OutOfMemoryError("simulated")
                return x

        state = {}
        x = torch.rand(1, 3, 400, 400)
        out = run_tiled_inference(
            OOMOnLarge(), x, tile_size=512, tile_pad=8, scale=1, fp16=False, batch_size=2, tile_state=state
        )
        self.assertTrue(torch.allclose(out, x))
        self.assertLessEqual(state["tile_size"], 300)

        before = calls["n"]
        run_tiled_inference(
            OOMOnLarge(), x, tile_size=512, tile_pad=8, scale=1, fp16=False, batch_size=2, tile_state=state
        )
        expected_tiles = math.ceil(400 / state["tile_size"]) ** 2
        self.assertLessEqual(calls["n"] - before, expected_tiles)


class TestExposureAndTemporalTensor(unittest.TestCase):
    def test_continuous_exposure_near_thresholds(self):
        from src.pipeline.stages import LowLightStage

        st = LowLightStage.__new__(LowLightStage)
        st._prev_adap_strength = None
        y1 = torch.full((1, 1, 64, 64), 79.9 / 255.0)
        y2 = torch.full((1, 1, 64, 64), 80.1 / 255.0)
        s1, _, _, _ = st._compute_scene_exposure(y1, 1.0)
        st._prev_adap_strength = None
        s2, _, _, _ = st._compute_scene_exposure(y2, 1.0)
        self.assertLess(abs(s1 - s2), 0.01)

    def test_temporal_process_frame_tensor(self):
        from src.pipeline.stages import TemporalSmoothingStage

        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        stage = TemporalSmoothingStage(ema_alpha=0.20, motion_compensated=True, flow_scale=0.5, device=dev)
        dtype = torch.float16 if dev.type == "cuda" else torch.float32
        f1 = torch.rand(1, 3, 128, 160, device=dev, dtype=dtype)
        f2 = (f1 + 0.03).clamp(0.0, 1.0)
        o1 = stage.process_frame_tensor(f1, strength=1.0)
        o2 = stage.process_frame_tensor(f2, strength=1.0)
        self.assertEqual(o1.shape, f1.shape)
        self.assertEqual(o2.shape, f2.shape)
        self.assertEqual(o2.dtype, dtype)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
class TestSCUNetFastEquivalence(unittest.TestCase):
    def test_patched_attention_matches_reference(self):
        import spandrel
        from src.models.scunet_fast import accelerate_scunet, verify_equivalence
        from src.models.weights_manager import get_weights_path

        m = spandrel.ModelLoader().load_from_file(str(get_weights_path("scunet_color_real_psnr"))).to("cuda").eval()
        self.assertGreater(accelerate_scunet(m), 0)
        self.assertLess(verify_equivalence(m, (136, 200), torch.float32), 1e-3)


class TestVideoMuxAndCancel(unittest.TestCase):
    def test_partial_processing_trims_audio_and_tags_bt709(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            silent = d / "v.mp4"
            w = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 180))
            for i in range(100):
                w.write(np.full((180, 320, 3), i, np.uint8))
            w.release()

            src = d / "with_audio.mp4"
            subprocess.run(
                [get_ffmpeg_binary(), "-y", "-loglevel", "error", "-i", str(silent),
                 "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
                 "-c:v", "copy", "-c:a", "aac", "-shortest", str(src)],
                check=True,
            )

            out = d / "out.mp4"
            res = StreamingVideoProcessor(str(src), str(out), max_seconds=1.0).process(lambda f, p: (f, {}))
            self.assertEqual(res["total_frames_processed"], 25)
            info = _ffprobe_info(str(out))
            self.assertIn("Duration: 00:00:01.0", info)
            self.assertIn("Audio:", info)
            self.assertIn("bt709", info)

    def test_cancel_removes_partial_files(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            src = d / "in.mp4"
            w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 180))
            for i in range(60):
                w.write(np.full((180, 320, 3), i, np.uint8))
            w.release()

            out = d / "out.mp4"
            sp = StreamingVideoProcessor(str(src), str(out))
            calls = {"n": 0}

            def cancel_after_five(_telemetry):
                calls["n"] += 1
                if calls["n"] >= 5:
                    raise KeyboardInterrupt("Cancelled by user")

            with self.assertRaises(KeyboardInterrupt):
                sp.process(lambda f, p: (f, {}), progress_callback=cancel_after_five)
            leftovers = [p.name for p in d.iterdir() if p.name != "in.mp4"]
            self.assertEqual(leftovers, [], f"Partial files left behind: {leftovers}")


if __name__ == "__main__":
    unittest.main()
