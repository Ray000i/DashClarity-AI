"""
DashClarity AI - Modular Video Enhancement Pipeline
Orchestrates sequential execution of modular restoration stages:
Denoising (SCUNet / Classical) -> Low-Light (Zero-DCE++ / STCD) -> Super-Resolution -> Temporal Smoothing.
Enforces the output-differs-from-input quality guard.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch

from src.pipeline.stages import (
    LowLightStage,
    NeuralDenoiseStage,
    ClassicalDenoiseStage,
    SuperResolutionStage,
    TemporalSmoothingStage
)
from src.pipeline.tiling import GPUOutOfMemoryFallback

logger = logging.getLogger("DashClarity.Pipeline.Enhancer")
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class VideoEnhancementPipeline:
    """
    Modular enhancement pipeline coordinator.
    Loads models once on the target hardware (CUDA default) and applies configurable stages.
    """
    def __init__(self, config_path: Optional[str] = None, preset: str = "full"):
        if config_path is None:
            config_path = str(BASE_DIR / "config.yaml")

        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        device_name = self.cfg.get("hardware", {}).get("device", "cuda")
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and device_name == "cuda" else "cpu"
        )
        self.fp16 = self.cfg.get("hardware", {}).get("fp16", True) and self.device.type == "cuda"
        self.tile_size = self.cfg.get("hardware", {}).get("tile_size", 256)
        self.tile_pad = self.cfg.get("hardware", {}).get("tile_pad", 16)

        logger.info("Initializing VideoEnhancementPipeline on %s (fp16=%s)", self.device, self.fp16)

        # Stage instances
        self.neural_denoise_stage = None
        self.classical_denoise_stage = ClassicalDenoiseStage(
            strength=self.cfg.get("models", {}).get("denoise_strength", 0.85)
        )
        self.low_light_stage = None
        self.stcd_stage = None
        self.super_res_stage = None
        self.temporal_stage = TemporalSmoothingStage(
            ema_alpha=self.cfg.get("models", {}).get("temporal_ema_alpha", 0.20),
            motion_compensated=self.cfg.get("models", {}).get("motion_compensated", True),
            flow_scale=self.cfg.get("models", {}).get("flow_scale", 0.5)
        )

        self.current_preset = preset
        self._load_models_for_preset(preset)

    def _load_models_for_preset(self, preset_name: str):
        presets = self.cfg.get("presets", {})
        preset_cfg = presets.get(preset_name, presets.get("full", {}))

        # 1. Neural Denoiser setup (SCUNet via spandrel)
        if preset_cfg.get("enable_neural_denoise", False) or preset_cfg.get("enable_denoise", False):
            fast_m = preset_cfg.get("neural_denoise_fast_mode", False)
            if self.neural_denoise_stage is None or getattr(self.neural_denoise_stage, "fast_mode", None) != fast_m:
                self.neural_denoise_stage = NeuralDenoiseStage(
                    device=self.device,
                    tile_size=self.tile_size,
                    tile_pad=self.tile_pad,
                    fp16=self.fp16,
                    fast_mode=fast_m
                )

        # Update temporal flow scale if preset configures it
        if hasattr(self, "temporal_stage"):
            self.temporal_stage.flow_scale = preset_cfg.get("temporal_flow_scale", self.cfg.get("models", {}).get("flow_scale", 0.5))

        # 2. Low-light model setup
        if preset_cfg.get("enable_low_light", False):
            model_key = preset_cfg.get("low_light_model", self.cfg.get("models", {}).get("low_light", "zero_dce_plus"))
            if model_key == "stcd_lowlight":
                if self.stcd_stage is None:
                    self.stcd_stage = LowLightStage(self.device, model_type="stcd_lowlight")
            else:
                if self.low_light_stage is None:
                    self.low_light_stage = LowLightStage(self.device, model_type="zero_dce_plus")

        # 3. Super-Resolution model setup
        if preset_cfg.get("enable_super_res", False):
            sr_key = preset_cfg.get("super_res_model", self.cfg.get("models", {}).get("super_res", "realesr_general_x4v3"))
            if self.super_res_stage is None or self.super_res_stage.model_name != sr_key:
                self.super_res_stage = SuperResolutionStage(
                    device=self.device,
                    model_name=sr_key,
                    tile_size=self.tile_size,
                    tile_pad=self.tile_pad,
                    fp16=self.fp16
                )

    def reset_temporal(self):
        """Clears temporal motion buffer between video clips."""
        self.temporal_stage.reset()

    def enhance_frame(
        self,
        frame_bgr: np.ndarray,
        frame_neighbor: Optional[np.ndarray] = None,
        preset: Optional[str] = None,
        strength: float = 1.0,
        restore_at_original_res: Optional[bool] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes a single video frame through the active pipeline stages with GPU OOM fallback.
        """
        try:
            return self._enhance_frame_internal(
                frame_bgr=frame_bgr,
                frame_neighbor=frame_neighbor,
                preset=preset,
                strength=strength,
                restore_at_original_res=restore_at_original_res
            )
        except (torch.cuda.OutOfMemoryError, GPUOutOfMemoryFallback) as oom_err:
            torch.cuda.empty_cache()
            curr = preset or self.current_preset
            logger.warning("GPU OOM encountered with preset '%s': %s. Activating safety fallback.", curr, oom_err)

            # Demote preset based on hierarchy: quality -> balanced -> fast
            if curr in ("quality", "full"):
                fallback_preset = "balanced"
            elif curr == "balanced":
                fallback_preset = "fast"
            else:
                fallback_preset = "fast"

            logger.warning("Demoted preset from '%s' to '%s' to prevent crash.", curr, fallback_preset)
            self.current_preset = fallback_preset
            self._load_models_for_preset(fallback_preset)

            res, telemetry = self._enhance_frame_internal(
                frame_bgr=frame_bgr,
                frame_neighbor=frame_neighbor,
                preset=fallback_preset,
                strength=strength,
                restore_at_original_res=restore_at_original_res
            )
            telemetry["gpu_oom_fallback"] = f"Demoted from {curr} to {fallback_preset}"
            return res, telemetry

    def _enhance_frame_internal(
        self,
        frame_bgr: np.ndarray,
        frame_neighbor: Optional[np.ndarray] = None,
        preset: Optional[str] = None,
        strength: float = 1.0,
        restore_at_original_res: Optional[bool] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if preset and preset != self.current_preset:
            self.current_preset = preset
            self._load_models_for_preset(preset)

        preset_cfg = self.cfg.get("presets", {}).get(self.current_preset, self.cfg.get("presets", {}).get("full", {}))
        eff_strength = strength * preset_cfg.get("strength", 1.0)

        current = frame_bgr.copy()
        stage_telemetry = {}

        # 1. Denoise Stage FIRST (Clean CMOS sensor noise before amplifying illumination)
        if (preset_cfg.get("enable_neural_denoise", False) or preset_cfg.get("enable_denoise", False)) and self.neural_denoise_stage is not None:
            current = self.neural_denoise_stage.process_frame(current, strength=eff_strength)
            stage_telemetry["neural_denoise"] = "scunet"
        elif preset_cfg.get("enable_classical_denoise", False):
            current = self.classical_denoise_stage.process_frame(current, strength=eff_strength)
            stage_telemetry["classical_denoise"] = "bilateral"

        # 2. Low-Light Stage (Zero-DCE++ or STCD with gray-world neutral balancing and CLAHE)
        if preset_cfg.get("enable_low_light", False):
            is_stcd = preset_cfg.get("low_light_model") == "stcd_lowlight"
            stage = self.stcd_stage if is_stcd else self.low_light_stage
            if stage is not None:
                current, ll_info = stage.process_frame(current, frame_neighbor=frame_neighbor, strength=eff_strength)
                stage_telemetry["low_light"] = ll_info

        # 3. Super-Resolution / Detail Synthesis Stage (Anti-ringing Real-ESRGAN)
        if preset_cfg.get("enable_super_res", False) and self.super_res_stage is not None:
            auto_down = self.cfg.get("super_resolution_policy", {}).get("auto_1080p_downscale", True)
            down_policy = restore_at_original_res if restore_at_original_res is not None else auto_down
            current = self.super_res_stage.process_frame(
                current,
                restore_at_original_res=down_policy,
                strength=eff_strength
            )
            stage_telemetry["super_res"] = {
                "model": self.super_res_stage.model_name,
                "output_shape": current.shape
            }

        # 4. Motion-Compensated Temporal Smoothing Stage (Farneback optical flow)
        if preset_cfg.get("enable_temporal", False):
            current = self.temporal_stage.process_frame(current, strength=eff_strength)
            stage_telemetry["temporal"] = "farneback_motion_compensated"

        return current, stage_telemetry

    def verify_output_transformation(
        self,
        orig_sample: np.ndarray,
        enh_sample: np.ndarray,
        min_mad_threshold: Optional[float] = None
    ) -> float:
        """
        Enforces the quality guard: computes MAD and raises an error if output is identical to input.
        """
        if min_mad_threshold is None:
            min_mad_threshold = self.cfg.get("quality_guard", {}).get("min_mad_threshold", 0.05)

        # If resolutions differ (e.g. upscaled), resize original for MAD comparison
        if orig_sample.shape[:2] != enh_sample.shape[:2]:
            orig_comp = cv2.resize(orig_sample, (enh_sample.shape[1], enh_sample.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            orig_comp = orig_sample

        diff = np.abs(orig_comp.astype(np.float32) - enh_sample.astype(np.float32))
        mad = float(np.mean(diff))

        if mad < min_mad_threshold and self.cfg.get("quality_guard", {}).get("raise_on_zero_diff", True):
            err_msg = (
                f"Quality Guard Failure: Output is visually identical to input! "
                f"Mean Absolute Difference ({mad:.2f}) < threshold ({min_mad_threshold:.2f}). "
                f"Pipeline failed to apply meaningful visual enhancement."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        return mad
