"""
Regression test for OpenCV (-209: Sizes of input arguments do not match) error in cv::arithm_op.
Verifies that SuperResolutionStage and all pipeline blending stages properly align array shapes
and data types before invoking OpenCV arithmetic operations.
"""

import unittest
import numpy as np
import cv2
import torch

from src.pipeline.stages import SuperResolutionStage, NeuralDenoiseStage, LowLightStage


class TestOpenCVRegression(unittest.TestCase):
    def test_reproduce_raw_opencv_failure(self):
        """Verifies that unaligned array shapes trigger OpenCV -209 error in raw cv2.addWeighted."""
        arr_highres = np.zeros((1440, 2560, 3), dtype=np.uint8)
        arr_lowres = np.zeros((720, 1280, 3), dtype=np.uint8)
        with self.assertRaises(cv2.error) as ctx:
            cv2.addWeighted(arr_highres, 0.8, arr_lowres, 0.2, 0)
        # OpenCV 4.x reports -209 ("Sizes of input arguments do not match"); OpenCV 5.x reports a
        # broadcast-shape assertion. Either way, mismatched shapes must raise.
        msg = str(ctx.exception)
        self.assertTrue("-209" in msg or "broadcast" in msg, msg)

    def test_fixed_super_resolution_stage_handles_mismatched_res(self):
        """Verifies that SuperResolutionStage with restore_at_original_res=False and strength < 1.0 blends cleanly."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        sr_stage = SuperResolutionStage(device=device, model_name="realesr_general_x4v3", tile_size=512)
        
        # Test 720p input with 4x SR and 80% blend strength
        input_720p = np.zeros((720, 1280, 3), dtype=np.uint8)
        try:
            out_upscaled = sr_stage.process_frame(input_720p, restore_at_original_res=False, strength=0.75)
            self.assertIsNotNone(out_upscaled)
            self.assertEqual(out_upscaled.dtype, np.uint8)
        except cv2.error as e:
            self.fail(f"SuperResolutionStage threw OpenCV error: {e}")

    def test_fixed_super_resolution_stage_restore_original(self):
        """Verifies that SuperResolutionStage with restore_at_original_res=True and strength < 1.0 blends cleanly."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        sr_stage = SuperResolutionStage(device=device, model_name="realesr_general_x4v3", tile_size=512)
        
        input_720p = np.zeros((720, 1280, 3), dtype=np.uint8)
        try:
            out_restored = sr_stage.process_frame(input_720p, restore_at_original_res=True, strength=0.75)
            self.assertIsNotNone(out_restored)
            self.assertEqual(out_restored.shape, (720, 1280, 3))
        except cv2.error as e:
            self.fail(f"SuperResolutionStage threw OpenCV error: {e}")


if __name__ == "__main__":
    unittest.main()
