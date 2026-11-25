"""
Regression tests for the RTX 4050 / 4060 GPU work (adaptive profile, SCUNet acceleration,
exposure smoothing, tiling OOM memory, video muxing).

    python -m unittest tests/test_gpu_optimizations.py
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))
os.environ.setdefault("TEST_MODE", "1")

from src.hardware import select_profile  # noqa: E402
from src.pipeline.tiling import run_tiled_inference  # noqa: E402
from src.utils import get_ffmpeg_binary  # noqa: E402


def _probe(path: str) -> str:
    return subprocess.run([get_ffmpeg_binary(), "-hide_banner", "-i", path], capture_output=True, text=True).stderr


class TestAdaptiveProfile(unittest.TestCase):
    def _p(self, total, free):
        return select_profile({"cuda_available": True, "vram_total_mb": total, "vram_free_mb": free, "fp16_tensor_cores": True})

    def test_rtx4050_class_runs_1080p_untiled_when_memory_is_free(self):
        p = self._p(6141, 4900)
        self.assertIn("6gb_class", p["tier"])
        self.assertGreater(p["scunet_full_frame_max_px"], 1920 * 1088)
        self.assertTrue(p["unload_unused_models"])

    def test_busy_rtx4050_falls_back_to_tiles_for_1080p(self):
        p = self._p(6141, 3000)
        self.assertLess(p["scunet_full_frame_max_px"], 1920 * 1088)
        self.assertGreaterEqual(p["scunet_full_frame_max_px"], 1280 * 736)

    def test_rtx4060_class_gets_a_larger_budget(self):
        p6, p8 = self._p(6141, 4900), self._p(8188, 7000)
        self.assertIn("8gb_class", p8["tier"])
        self.assertGreater(p8["vram_budget_mb"], p6["vram_budget_mb"])
        self.assertFalse(p8["unload_unused_models"])

    def test_budget_never_exceeds_free_memory(self):
        p = self._p(8188, 2000)
        self.assertLessEqual(p["vram_budget_mb"], 2000)

    def test_cpu_profile(self):
        self.assertEqual(select_profile({"cuda_available": False})["device"], "cpu")


class TestTilingMemory(unittest.TestCase):
    def test_tile_state_is_remembered_after_oom(self):
        calls = {"n": 0}

        class OOMOnLarge(torch.nn.Module):
            def forward(self, x):
                calls["n"] += 1
                if max(x.shape[2:]) > 300:
                    raise torch.cuda.OutOfMemoryError("simulated")
                return x

        state = {}
        x = torch.rand(1, 3, 400, 400)
        out = run_tiled_inference(OOMOnLarge(), x, tile_size=512, tile_pad=8, scale=1, fp16=False, tile_state=state)
        self.assertTrue(torch.allclose(out, x))
        self.assertLessEqual(state["tile_size"], 300)
        before = calls["n"]
        run_tiled_inference(OOMOnLarge(), x, tile_size=512, tile_pad=8, scale=1, fp16=False, tile_state=state)
        self.assertEqual(calls["n"] - before, int(np.ceil(400 / state["tile_size"])) ** 2,
                         "second frame should start at the remembered tile size without new OOMs")


class TestExposureSmoothing(unittest.TestCase):
    def test_strength_is_continuous_near_thresholds(self):
        from src.pipeline.stages import LowLightStage
        st = LowLightStage.__new__(LowLightStage)  # no weights needed for the exposure logic
        st.smooth_exposure, st._lum_ema = False, None
        jump = abs(st._exposure(79.9, 1.0)[0] - st._exposure(80.1, 1.0)[0])
        self.assertGreater(jump, 0.3, "the original step mapping jumps at 80")
        st.smooth_exposure = True
        st._lum_ema = None
        a = st._exposure(79.9, 1.0)[0]
        b = st._exposure(80.1, 1.0)[0]
        self.assertLess(abs(a - b), 0.01)

    def test_scene_cut_resets_history(self):
        from src.pipeline.stages import LowLightStage
        st = LowLightStage.__new__(LowLightStage)
        st.smooth_exposure, st._lum_ema = True, None
        for _ in range(20):
            st._exposure(30.0, 1.0)
        st._exposure(140.0, 1.0)
        self.assertEqual(st._lum_ema, 140.0)


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class TestSCUNetAcceleration(unittest.TestCase):
    def test_patched_attention_matches_reference(self):
        import spandrel
        from src.models.scunet_fast import accelerate_scunet, verify_equivalence
        from src.models.weights_manager import get_weights_path
        m = spandrel.ModelLoader().load_from_file(str(get_weights_path("scunet_color_real_psnr"))).to("cuda").eval()
        self.assertGreater(accelerate_scunet(m), 0)
        self.assertLess(verify_equivalence(m, (136, 200), torch.float32), 1e-3)


class TestVideoMux(unittest.TestCase):
    def test_partial_processing_trims_audio_and_tags_bt709(self):
        from src.pipeline.video_stream import StreamingVideoProcessor
        d = Path(tempfile.mkdtemp())
        silent = d / "v.mp4"
        w = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 180))
        for i in range(100):
            w.write(np.full((180, 320, 3), i, np.uint8))
        w.release()
        src = d / "with_audio.mp4"
        subprocess.run([get_ffmpeg_binary(), "-y", "-loglevel", "error", "-i", str(silent), "-f", "lavfi", "-i",
                        "sine=frequency=440:duration=4", "-c:v", "copy", "-c:a", "aac", "-shortest", str(src)], check=True)
        out = d / "out.mp4"
        res = StreamingVideoProcessor(str(src), str(out), max_seconds=1.0).process(lambda f, p: (f, {}))
        self.assertEqual(res["total_frames_processed"], 25)
        info = _probe(str(out))
        self.assertIn("Duration: 00:00:01.0", info)
        self.assertIn("Audio:", info)
        self.assertIn("bt709", info)


class TestCancelCleanup(unittest.TestCase):
    def test_cancel_removes_partial_files(self):
        from src.pipeline.video_stream import StreamingVideoProcessor
        d = Path(tempfile.mkdtemp())
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
        self.assertEqual(leftovers, [], f"partial files left behind: {leftovers}")


if __name__ == "__main__":
    unittest.main()
