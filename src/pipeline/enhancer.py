"""
DashClarity AI - Adaptive Modular Video Enhancement Pipeline
Orchestrates sequential execution of modular restoration stages:
Denoising (SCUNet / Classical) -> Low-Light (Zero-DCE++ / STCD) -> Super-Resolution -> Temporal Smoothing.

Optimizations & Compatibility:
- Automatically detects GPU VRAM (e.g. RTX 4050 6GB vs RTX 4060 8GB) via src.utils.hardware
  and selects optimal tile size, tile batching, SR output bounding, and sequential offloading.
- Chains GPU stages directly via GPU-resident tensors to eliminate redundant CPU-GPU roundtrips.
- Enforces UI stage_overrides and dynamic CUDA OOM preset adaptation.
"""

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
    TemporalSmoothingStage,
    bgr_to_gpu_rgb_tensor,
    gpu_rgb_tensor_to_bgr,
)
from src.pipeline.tiling import GPUOutOfMemoryFallback
from src.utils.hardware import detect_system_hardware, safe_cuda_cleanup

logger = logging.getLogger("DashClarity.Pipeline.Enhancer")
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class VideoEnhancementPipeline:
    """
    Modular enhancement pipeline coordinator.
    Detects hardware capabilities, configures memory-safe GPU stages, and supports
    zero-roundtrip tensor chaining across SCUNet -> Zero-DCE++ -> Real-ESRGAN.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        preset: str = "full",
        override_vram_gb: Optional[float] = None
    ):
        if config_path is None:
            config_path = str(BASE_DIR / "config.yaml")

        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # Probe actual system GPU and RAM (supports RTX 4050 6GB, RTX 4060 8GB, and CPU fallback)
        self.hardware_info = detect_system_hardware(override_vram_gb=override_vram_gb)
        hw_profile = self.hardware_info["profile"]

        cfg_device = self.cfg.get("hardware", {}).get("device", "cuda")
        cuda_ok = self.hardware_info["cuda_available"] and cfg_device == "cuda"
        self.device = torch.device("cuda:0" if cuda_ok else "cpu")

        if self.device.type == "cuda":
            # Keep cudnn.benchmark=False: autotuner allocates up to 5.5 GB workspace on dynamic/depthwise kernels
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.profile = hw_profile
        self.fp16 = bool(self.cfg.get("hardware", {}).get("fp16", True) and hw_profile["fp16"] and cuda_ok)
        self.tile_size = int(hw_profile["tile_size"])
        self.tile_pad = int(self.cfg.get("hardware", {}).get("tile_pad", hw_profile["tile_pad"]))
        self.tile_batch_size = int(hw_profile["tile_batch_size"])
        self.scunet_fast_scale = float(hw_profile["scunet_fast_scale"])
        self.scunet_fast_scale_1080p = float(hw_profile["scunet_fast_scale_1080p"])
        self.scunet_full_frame_max_px = int(hw_profile.get("scunet_full_frame_max_px", 0))
        self.sr_full_frame_max_px = int(hw_profile.get("sr_full_frame_max_px", 0))
        self.sr_max_out_dim = int(hw_profile["sr_max_out_dim"])
        self.sequential_offload = bool(hw_profile["sequential_offload"])
        self.gpu_tier = hw_profile["tier"]

        logger.info(
            "Pipeline initialized | GPU=%s (%.2f GB VRAM) | Tier=%s | fp16=%s | tile=%d (batch=%d) | max_sr_dim=%d | offload=%s",
            self.hardware_info["gpu_name"],
            self.hardware_info["total_vram_gb"],
            self.gpu_tier,
            self.fp16,
            self.tile_size,
            self.tile_batch_size,
            self.sr_max_out_dim,
            self.sequential_offload
        )

        # Stage instances (loaded lazily per preset / overrides)
        self.neural_denoise_stage: Optional[NeuralDenoiseStage] = None
        self.classical_denoise_stage = ClassicalDenoiseStage(
            strength=self.cfg.get("models", {}).get("denoise_strength", 0.85)
        )
        self.low_light_stage: Optional[LowLightStage] = None
        self.stcd_stage: Optional[LowLightStage] = None
        self.super_res_stage: Optional[SuperResolutionStage] = None
        self.temporal_stage = TemporalSmoothingStage(
            ema_alpha=self.cfg.get("models", {}).get("temporal_ema_alpha", 0.18),
            motion_compensated=self.cfg.get("models", {}).get("motion_compensated", True),
            flow_scale=self.cfg.get("models", {}).get("flow_scale", 0.5),
            device=self.device
        )

        self.current_preset = preset
        self._warmed_up = False
        self._load_models_for_preset(preset)

    def _resolve_effective_preset(
        self,
        preset_name: str,
        stage_overrides: Optional[Dict[str, Optional[bool]]] = None
    ) -> Dict[str, Any]:
        presets = self.cfg.get("presets", {})
        base = dict(presets.get(preset_name, presets.get("full", {})))

        if stage_overrides:
            # Map frontend checkbox toggles onto modular stage switches cleanly
            if stage_overrides.get("enable_denoise") is not None:
                want_denoise = bool(stage_overrides["enable_denoise"])
                if not want_denoise:
                    base["enable_neural_denoise"] = False
                    base["enable_denoise"] = False
                    base["enable_classical_denoise"] = False
                else:
                    # Only activate classical denoise if fast preset explicitly uses classical, else neural SCUNet
                    if not base.get("enable_neural_denoise", False) and not base.get("enable_classical_denoise", False):
                        if preset_name == "fast":
                            base["enable_classical_denoise"] = True
                        else:
                            base["enable_neural_denoise"] = True
            if stage_overrides.get("enable_low_light") is not None:
                base["enable_low_light"] = bool(stage_overrides["enable_low_light"])
            if stage_overrides.get("enable_super_res") is not None:
                base["enable_super_res"] = bool(stage_overrides["enable_super_res"])
            if stage_overrides.get("enable_temporal") is not None:
                base["enable_temporal"] = bool(stage_overrides["enable_temporal"])

        return base

    def _prune_unused_stages(self, preset_cfg: Dict[str, Any]):
        """Unloads heavy unused models on constrained GPUs (e.g., 6 GB RTX 4050) to reclaim VRAM."""
        need_neural_denoise = preset_cfg.get("enable_neural_denoise", False) or preset_cfg.get("enable_denoise", False)
        need_stcd = preset_cfg.get("enable_low_light", False) and preset_cfg.get("low_light_model") == "stcd_lowlight"
        need_sr = preset_cfg.get("enable_super_res", False)

        freed = False
        if not need_stcd and self.stcd_stage is not None:
            self.stcd_stage = None
            freed = True
        if self.sequential_offload:
            if not need_neural_denoise and self.neural_denoise_stage is not None:
                self.neural_denoise_stage = None
                freed = True
            if not need_sr and self.super_res_stage is not None:
                self.super_res_stage = None
                freed = True
        if freed:
            safe_cuda_cleanup(collect_gc=True)

    def _load_models_for_preset(
        self,
        preset_name: str,
        stage_overrides: Optional[Dict[str, Optional[bool]]] = None
    ):
        preset_cfg = self._resolve_effective_preset(preset_name, stage_overrides)
        self._prune_unused_stages(preset_cfg)

        # 1. Neural Denoiser setup (SCUNet via spandrel)
        if preset_cfg.get("enable_neural_denoise", False) or preset_cfg.get("enable_denoise", False):
            fast_m = preset_cfg.get("neural_denoise_fast_mode", False)
            if self.neural_denoise_stage is None or getattr(self.neural_denoise_stage, "fast_mode", None) != fast_m:
                self.neural_denoise_stage = NeuralDenoiseStage(
                    device=self.device,
                    tile_size=self.tile_size,
                    tile_pad=self.tile_pad,
                    fp16=self.fp16,
                    fast_mode=fast_m,
                    tile_batch_size=self.tile_batch_size,
                    fast_scale=self.scunet_fast_scale,
                    fast_scale_1080p=self.scunet_fast_scale_1080p,
                    full_frame_max_px=self.scunet_full_frame_max_px
                )

        if hasattr(self, "temporal_stage"):
            self.temporal_stage.flow_scale = preset_cfg.get(
                "temporal_flow_scale",
                self.cfg.get("models", {}).get("flow_scale", 0.5)
            )

        # 2. Low-light model setup
        if preset_cfg.get("enable_low_light", False):
            model_key = preset_cfg.get("low_light_model", self.cfg.get("models", {}).get("low_light", "zero_dce_plus"))
            if model_key == "stcd_lowlight":
                if self.stcd_stage is None:
                    self.stcd_stage = LowLightStage(self.device, model_type="stcd_lowlight", fp16=self.fp16)
            else:
                if self.low_light_stage is None:
                    self.low_light_stage = LowLightStage(self.device, model_type="zero_dce_plus", fp16=self.fp16)

        # 3. Super-Resolution model setup
        if preset_cfg.get("enable_super_res", False):
            sr_key = preset_cfg.get("super_res_model", self.cfg.get("models", {}).get("super_res", "realesr_general_x4v3"))
            if self.super_res_stage is None or self.super_res_stage.model_name != sr_key:
                self.super_res_stage = SuperResolutionStage(
                    device=self.device,
                    model_name=sr_key,
                    tile_size=self.tile_size,
                    tile_pad=self.tile_pad,
                    fp16=self.fp16,
                    tile_batch_size=self.tile_batch_size,
                    max_out_dim=self.sr_max_out_dim,
                    full_frame_max_px=self.sr_full_frame_max_px
                )

    def warmup(self):
        """Runs a tiny dummy forward pass once so cuDNN/CUDA graphs are warm before the first live clip."""
        if self._warmed_up:
            return
        try:
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            _ = self.enhance_frame(dummy, dummy, preset=self.current_preset)
            self.reset_temporal()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self._warmed_up = True
            logger.info("GPU pipeline warmup completed cleanly on %s.", self.device)
        except Exception as e:
            logger.warning("Pipeline warmup skipped due to non-fatal error: %s", e)

    def release_cached_memory(self):
        """Releases cached GPU allocator blocks between queued jobs while retaining loaded models."""
        safe_cuda_cleanup(collect_gc=False)

    def reset_temporal(self):
        """Clears temporal motion and exposure history buffers between separate video clips."""
        self.temporal_stage.reset()
        if self.low_light_stage is not None:
            self.low_light_stage.reset()

    def enhance_frame(
        self,
        frame_bgr: np.ndarray,
        frame_neighbor: Optional[np.ndarray] = None,
        preset: Optional[str] = None,
        strength: float = 1.0,
        restore_at_original_res: Optional[bool] = None,
        stage_overrides: Optional[Dict[str, Optional[bool]]] = None
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
                restore_at_original_res=restore_at_original_res,
                stage_overrides=stage_overrides
            )
        except (torch.cuda.OutOfMemoryError, GPUOutOfMemoryFallback) as oom_err:
            safe_cuda_cleanup(collect_gc=True)
            curr = preset or self.current_preset
            logger.warning("GPU OOM encountered with preset '%s': %s. Activating safety fallback.", curr, oom_err)

            if curr in ("quality", "full"):
                fallback_preset = "balanced"
            else:
                fallback_preset = "fast"

            logger.warning("Demoted preset from '%s' to '%s' to prevent crash.", curr, fallback_preset)
            self.current_preset = fallback_preset
            self._load_models_for_preset(fallback_preset, stage_overrides=None)

            res, telemetry = self._enhance_frame_internal(
                frame_bgr=frame_bgr,
                frame_neighbor=frame_neighbor,
                preset=fallback_preset,
                strength=strength,
                restore_at_original_res=restore_at_original_res,
                stage_overrides=None
            )
            telemetry["gpu_oom_fallback"] = f"Demoted from {curr} to {fallback_preset}"
            return res, telemetry

    def _enhance_frame_internal(
        self,
        frame_bgr: np.ndarray,
        frame_neighbor: Optional[np.ndarray] = None,
        preset: Optional[str] = None,
        strength: float = 1.0,
        restore_at_original_res: Optional[bool] = None,
        stage_overrides: Optional[Dict[str, Optional[bool]]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if preset and preset != self.current_preset:
            self.current_preset = preset

        self._load_models_for_preset(self.current_preset, stage_overrides=stage_overrides)
        preset_cfg = self._resolve_effective_preset(self.current_preset, stage_overrides=stage_overrides)
        eff_strength = strength * preset_cfg.get("strength", 1.0)

        stage_telemetry: Dict[str, Any] = {"gpu_tier": self.gpu_tier}
        current_np = frame_bgr
        gpu_tensor: Optional[torch.Tensor] = None

        use_neural_denoise = (
            (preset_cfg.get("enable_neural_denoise", False) or preset_cfg.get("enable_denoise", False))
            and self.neural_denoise_stage is not None
        )
        use_classical_denoise = preset_cfg.get("enable_classical_denoise", False) and not use_neural_denoise
        use_low_light = preset_cfg.get("enable_low_light", False)
        is_stcd = use_low_light and (preset_cfg.get("low_light_model") == "stcd_lowlight")
        use_sr = preset_cfg.get("enable_super_res", False) and self.super_res_stage is not None

        # 1. Denoising Stage FIRST (SCUNet on GPU or Bilateral on CPU)
        if use_neural_denoise:
            gpu_tensor = bgr_to_gpu_rgb_tensor(current_np, self.device, self.fp16)
            gpu_tensor = self.neural_denoise_stage.process_frame_tensor(gpu_tensor, strength=eff_strength)
            stage_telemetry["neural_denoise"] = "scunet"
        elif use_classical_denoise:
            current_np = self.classical_denoise_stage.process_frame(current_np, strength=eff_strength)
            stage_telemetry["classical_denoise"] = "bilateral"

        # 2. Low-Light Illumination Stage (chains directly on GPU tensor when using Zero-DCE++)
        if use_low_light:
            if is_stcd and self.stcd_stage is not None:
                if gpu_tensor is not None:
                    current_np = gpu_rgb_tensor_to_bgr(gpu_tensor)
                    gpu_tensor = None
                current_np, ll_info = self.stcd_stage.process_frame(current_np, frame_neighbor=frame_neighbor, strength=eff_strength)
                stage_telemetry["low_light"] = ll_info
            elif self.low_light_stage is not None:
                if gpu_tensor is None:
                    gpu_tensor = bgr_to_gpu_rgb_tensor(current_np, self.device, self.fp16)
                gpu_tensor, ll_info = self.low_light_stage.process_frame_tensor(gpu_tensor, strength=eff_strength)
                stage_telemetry["low_light"] = ll_info

        # 3. Super-Resolution Stage (chains directly on GPU tensor with bounded output resolution)
        if use_sr:
            if gpu_tensor is None:
                gpu_tensor = bgr_to_gpu_rgb_tensor(current_np, self.device, self.fp16)
            auto_down = self.cfg.get("super_resolution_policy", {}).get("auto_1080p_downscale", True)
            down_policy = restore_at_original_res if restore_at_original_res is not None else auto_down
            anti_ring = float(self.cfg.get("super_resolution_policy", {}).get("anti_ringing_blend", 0.22))
            gpu_tensor = self.super_res_stage.process_frame_tensor(
                gpu_tensor,
                restore_at_original_res=down_policy,
                strength=eff_strength,
                anti_ringing_blend=anti_ring
            )
            stage_telemetry["super_res"] = {
                "model": self.super_res_stage.model_name,
                "output_shape": tuple(gpu_tensor.shape[-2:])
            }

        # 4. Motion-Compensated Temporal Smoothing Stage (Farneback flow + GPU warp/blend)
        if preset_cfg.get("enable_temporal", False):
            if gpu_tensor is not None:
                gpu_tensor = self.temporal_stage.process_frame_tensor(gpu_tensor, strength=eff_strength)
            else:
                current_np = self.temporal_stage.process_frame(current_np, strength=eff_strength)
            stage_telemetry["temporal"] = "farneback_motion_compensated"

        # Transfer final enhanced GPU tensor back to CPU ONCE at the very end
        if gpu_tensor is not None:
            current_np = gpu_rgb_tensor_to_bgr(gpu_tensor)
            del gpu_tensor

        return current_np, stage_telemetry

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
