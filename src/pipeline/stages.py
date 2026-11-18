"""
DashClarity AI - Modular Enhancement Pipeline Stages
Encapsulates individual neural restoration and generative enhancement stages:
1. LowLightStage: Adaptive Zero-DCE++ with gray-world neutral balancing, CLAHE, and gain limiting
2. NeuralDenoiseStage: Pretrained SCUNet neural denoiser loaded via spandrel (tiled, fp16)
3. ClassicalDenoiseStage: Bilateral filter labeled classical baseline for ablation
4. SuperResolutionStage: Real-ESRGAN with anti-ringing Lanczos blending
5. TemporalSmoothingStage: Farneback dense optical flow motion-compensated temporal consistency
"""

import logging
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.models.zerodce import ZeroDCEPlus
from src.models.srvgg import SRVGGNetCompact
from src.models.rrdb import RRDBNet
from src.models.stcd import STCDWrapper
from src.models.weights_manager import get_weights_path
from src.pipeline.tiling import run_tiled_inference

logger = logging.getLogger("DashClarity.Pipeline.Stages")


class LowLightStage:
    """
    Adaptive Low-Light Illumination Enhancement.
    Estimates scene illumination. For bright scenes (daytime), attenuates boost to prevent blowout.
    Integrates gray-world chromaticity balancing (prevents blue/purple cast) and L-channel CLAHE.
    """
    def __init__(self, device: torch.device, model_type: str = "zero_dce_plus"):
        self.device = device
        self.model_type = model_type
        self.model = None

        if model_type == "zero_dce_plus":
            w_path = get_weights_path("zero_dce_plus")
            self.model = ZeroDCEPlus().to(self.device).eval()
            ckpt = torch.load(str(w_path), map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt, strict=True)
            logger.info("Zero-DCE++ loaded successfully on %s", device)
        elif model_type == "stcd_lowlight":
            w_path = get_weights_path("stcd_indoor")
            self.model = STCDWrapper(str(w_path), self.device)
            logger.info("STCD low-light wrapper loaded successfully on %s", device)
        else:
            raise ValueError(f"Unknown low-light model type: {model_type}")

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_neighbor: Optional[np.ndarray] = None,
        strength: float = 1.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))

        # Adaptive exposure thresholding:
        # Prevents overexposure in daylight while boosting nighttime scenes
        if mean_lum >= 120.0:
            adaptive_strength = 0.05 * strength
            condition = "daytime_bright"
        elif mean_lum >= 80.0:
            adaptive_strength = 0.22 * strength
            condition = "daytime_moderate"
        elif mean_lum >= 45.0:
            adaptive_strength = 0.60 * strength
            condition = "twilight_mixed"
        else:
            adaptive_strength = 1.0 * strength
            condition = "nighttime_lowlight"

        if adaptive_strength <= 0.01:
            return frame_bgr.copy(), {"condition": condition, "lum": mean_lum, "applied": False}

        if self.model_type == "stcd_lowlight":
            neigh = frame_neighbor if frame_neighbor is not None else frame_bgr
            raw_enh = self.model.enhance(frame_bgr, neigh)
            if raw_enh.shape[:2] != frame_bgr.shape[:2]:
                raw_enh = cv2.resize(raw_enh, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
            if raw_enh.dtype != frame_bgr.dtype:
                raw_enh = raw_enh.astype(frame_bgr.dtype)
            blended = cv2.addWeighted(raw_enh, adaptive_strength, frame_bgr, 1.0 - adaptive_strength, 0)
            return blended, {"condition": condition, "lum": mean_lum, "applied": True}

        # Zero-DCE++ path (fully vectorized on GPU with fp16)
        h, w = frame_bgr.shape[:2]
        t_in = torch.from_numpy(frame_bgr).to(self.device, non_blocking=True).permute(2, 0, 1).unsqueeze(0)
        t_rgb = (t_in[:, [2, 1, 0], :, :].half()) * (1.0 / 255.0)

        if not hasattr(self, "_model_fp16_checked") and self.device.type == "cuda":
            self.model = self.model.half()
            self._model_fp16_checked = True

        with torch.inference_mode():
            enh_t, _ = self.model(t_rgb)

            # Global smooth Gray-World color balance (pure scalar multipliers):
            # Normalizes red/blue sensor biases without any per-pixel thresholds,
            # eliminating contour bands, steps, or topographic ripple artifacts.
            r_m = enh_t[:, 0].mean()
            g_m = enh_t[:, 1].mean()
            b_m = enh_t[:, 2].mean()
            luma = 0.299 * r_m + 0.587 * g_m + 0.114 * b_m
            kr = torch.clamp(luma / (r_m + 1e-4), 0.45, 1.45)
            kg = torch.clamp(luma / (g_m + 1e-4), 0.70, 1.45)
            kb = torch.clamp(luma / (b_m + 1e-4), 0.70, 1.45)
            enh_clean = torch.cat([enh_t[:, 0:1] * kr, enh_t[:, 1:2] * kg, enh_t[:, 2:3] * kb], dim=1).clamp(0.0, 1.0)

            # Modulate with adaptive exposure strength
            if adaptive_strength < 1.0:
                out_t = t_rgb + (enh_clean - t_rgb) * adaptive_strength
            else:
                out_t = enh_clean

            out_bgr_t = out_t[:, [2, 1, 0], :, :].clamp(0.0, 1.0) * 255.0
            out_bgr = np.ascontiguousarray(out_bgr_t.squeeze(0).byte().permute(1, 2, 0).cpu().numpy())

        return out_bgr, {"condition": condition, "lum": mean_lum, "applied": True}


class NeuralDenoiseStage:
    """
    Pretrained Neural Denoiser: SCUNet loaded via spandrel.
    Removes high-ISO CMOS sensor noise and compression block artifacts.
    Runs tiled with fp16 to maintain VRAM < 1 GB.
    Includes fast_mode (half-resolution luma inference) and organic texture preservation.
    """
    def __init__(
        self,
        device: torch.device,
        tile_size: int = 512,
        tile_pad: int = 16,
        fp16: bool = True,
        fast_mode: bool = False
    ):
        self.device = device
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.fp16 = fp16
        self.fast_mode = fast_mode

        w_path = get_weights_path("scunet_color_real_psnr")
        import spandrel
        loader = spandrel.ModelLoader()
        self.model = loader.load_from_file(str(w_path)).to(self.device).eval()
        if self.fp16 and self.device.type == 'cuda':
            self.model = self.model.half()
        logger.info("SCUNet neural denoiser loaded on %s (tile_size=%d, fast_mode=%s)", device, tile_size, fast_mode)

    def process_frame(self, frame_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
        if strength <= 0.05:
            return frame_bgr

        h, w = frame_bgr.shape[:2]

        if self.fast_mode and (h > 720 or w > 1280):
            # For 1080p+ videos in fast mode: downsample untiled to maintain fast throughput
            sh, sw = (h // 2 // 2) * 2, (w // 2 // 2) * 2
            small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
            rgb_s = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t_in = torch.from_numpy(rgb_s).permute(2, 0, 1).unsqueeze(0).to(self.device)
            if self.fp16 and self.device.type == 'cuda':
                t_in = t_in.half()
            with torch.inference_mode():
                out_tensor = self.model(t_in)
            out_s = out_tensor.squeeze(0).clamp(0.0, 1.0).permute(1, 2, 0).float().cpu().numpy()
            out_s_bgr = cv2.cvtColor((out_s * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
            denoised_bgr = cv2.resize(out_s_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
        elif self.fast_mode:
            # For <= 720p (e.g. 480p/720p dashcam footage): run native 1x resolution padded to 32
            # Eliminates severe downsampling blur while running faster on modern GPUs
            pad_h = (32 - (h % 32)) % 32
            pad_w = (32 - (w % 32)) % 32
            padded = cv2.copyMakeBorder(frame_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT) if (pad_h > 0 or pad_w > 0) else frame_bgr
            rgb_pad = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t_in = torch.from_numpy(rgb_pad).permute(2, 0, 1).unsqueeze(0).to(self.device)
            if self.fp16 and self.device.type == 'cuda':
                t_in = t_in.half()
            with torch.inference_mode():
                out_tensor = self.model(t_in)
            out_np = (out_tensor.squeeze(0).clamp(0.0, 1.0).permute(1, 2, 0).float().cpu().numpy() * 255.0).round().astype(np.uint8)
            denoised_bgr = cv2.cvtColor(out_np[:h, :w], cv2.COLOR_RGB2BGR)
        else:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
            if self.fp16 and self.device.type == 'cuda':
                tensor = tensor.half()

            with torch.inference_mode():
                out_tensor = run_tiled_inference(
                    model=self.model,
                    tensor=tensor,
                    tile_size=self.tile_size,
                    tile_pad=self.tile_pad,
                    scale=1,
                    fp16=self.fp16
                )

            out_np = out_tensor.squeeze(0).clamp(0.0, 1.0).permute(1, 2, 0).float().cpu().numpy()
            denoised_bgr = (out_np * 255.0).round().astype(np.uint8)
            denoised_bgr = cv2.cvtColor(denoised_bgr, cv2.COLOR_RGB2BGR)

        if strength < 1.0:
            if denoised_bgr.shape != frame_bgr.shape:
                denoised_bgr = cv2.resize(denoised_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
            return cv2.addWeighted(denoised_bgr, strength, frame_bgr, 1.0 - strength, 0)
        return denoised_bgr


class ClassicalDenoiseStage:
    """
    Classical Bilateral baseline for comparison and ablation studies.
    """
    def __init__(self, strength: float = 1.0):
        self.strength = strength

    def process_frame(self, frame_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
        eff = self.strength * strength
        if eff <= 0.05:
            return frame_bgr
        sigma_c = int(22 * eff)
        sigma_s = int(6 * eff)
        return cv2.bilateralFilter(frame_bgr, d=5, sigmaColor=sigma_c, sigmaSpace=sigma_s)


class SuperResolutionStage:
    """
    Real-ESRGAN Generative Super-Resolution & Detail Restoration.
    Applies anti-ringing Lanczos blending (0.78 SR + 0.22 Lanczos) to prevent haloing around
    lane lines, road signs, and license plates.
    """
    def __init__(
        self,
        device: torch.device,
        model_name: str = "realesr_general_x4v3",
        tile_size: int = 512,
        tile_pad: int = 16,
        fp16: bool = True
    ):
        self.device = device
        self.model_name = model_name
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.fp16 = fp16

        if model_name == "realesr_x2plus":
            w_path = get_weights_path("realesr_x2plus")
            self.model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=23, num_grow_ch=32)
            ckpt = torch.load(str(w_path), map_location="cpu", weights_only=False)
            state_dict = ckpt['params_ema'] if 'params_ema' in ckpt else (ckpt['params'] if 'params' in ckpt else ckpt)
            self.model.load_state_dict(state_dict, strict=True)
            self.scale = 2
        elif model_name == "realesr_general_x4v3":
            w_path = get_weights_path("realesr_general_x4v3")
            self.model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            ckpt = torch.load(str(w_path), map_location="cpu", weights_only=False)
            state_dict = ckpt['params'] if 'params' in ckpt else ckpt
            self.model.load_state_dict(state_dict, strict=True)
            self.scale = 4
        else:
            raise ValueError(f"Unknown SuperResolution model: {model_name}")

        self.model = self.model.to(self.device).eval()
        if self.fp16 and self.device.type == 'cuda':
            self.model = self.model.half()
        logger.info("SuperResolution (%s, scale=%dx) loaded on %s", model_name, self.scale, device)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        restore_at_original_res: bool = False,
        strength: float = 1.0
    ) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

        if self.fp16 and self.device.type == 'cuda':
            tensor = tensor.half()

        with torch.no_grad():
            out_tensor = run_tiled_inference(
                model=self.model,
                tensor=tensor,
                tile_size=self.tile_size,
                tile_pad=self.tile_pad,
                scale=self.scale,
                fp16=self.fp16
            )

        out_np = out_tensor.squeeze(0).clamp(0.0, 1.0).permute(1, 2, 0).float().cpu().numpy()
        out_bgr = (out_np * 255.0).round().astype(np.uint8)
        out_bgr = cv2.cvtColor(out_bgr, cv2.COLOR_RGB2BGR)

        # Anti-ringing blend with Lanczos upscaled original
        lanczos_ref = cv2.resize(frame_bgr, (out_bgr.shape[1], out_bgr.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        if lanczos_ref.dtype != out_bgr.dtype:
            lanczos_ref = lanczos_ref.astype(out_bgr.dtype)
        out_bgr = cv2.addWeighted(out_bgr, 0.78, lanczos_ref, 0.22, 0)

        if restore_at_original_res or (h >= 1080 or w >= 1920):
            out_bgr = cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
            if strength < 1.0:
                if frame_bgr.shape != out_bgr.shape:
                    frame_ref = cv2.resize(frame_bgr, (out_bgr.shape[1], out_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                else:
                    frame_ref = frame_bgr
                if frame_ref.dtype != out_bgr.dtype:
                    frame_ref = frame_ref.astype(out_bgr.dtype)
                out_bgr = cv2.addWeighted(out_bgr, strength, frame_ref, 1.0 - strength, 0)
        elif strength < 1.0:
            if lanczos_ref.shape != out_bgr.shape:
                lanczos_ref = cv2.resize(lanczos_ref, (out_bgr.shape[1], out_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
            if lanczos_ref.dtype != out_bgr.dtype:
                lanczos_ref = lanczos_ref.astype(out_bgr.dtype)
            out_bgr = cv2.addWeighted(out_bgr, strength, lanczos_ref, 1.0 - strength, 0)

        return out_bgr


class TemporalSmoothingStage:
    """
    Motion-compensated temporal consistency using Farneback dense optical flow.
    Warps previous enhanced frame forward along the camera motion vector field.
    Completely eliminates ghosting on moving asphalt, passing cars, and dashcam motion.
    """
    def __init__(
        self,
        ema_alpha: float = 0.15,
        motion_compensated: bool = True,
        flow_scale: float = 0.5
    ):
        self.ema_alpha = ema_alpha
        self.motion_compensated = motion_compensated
        self.flow_scale = flow_scale
        self.prev_output_bgr = None
        self.prev_gray = None

    def reset(self):
        self.prev_output_bgr = None
        self.prev_gray = None

    def process_frame(self, frame_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.prev_output_bgr is None or strength <= 0.05:
            self.prev_output_bgr = frame_bgr.copy()
            self.prev_gray = curr_gray.copy()
            return frame_bgr

        if self.prev_output_bgr.shape != frame_bgr.shape:
            self.prev_output_bgr = frame_bgr.copy()
            self.prev_gray = curr_gray.copy()
            return frame_bgr

        h, w = frame_bgr.shape[:2]

        if self.motion_compensated and self.prev_gray is not None:
            # Downscale for high-speed Farneback optical flow calculation
            sw, sh = max(16, int(w * self.flow_scale)), max(16, int(h * self.flow_scale))
            prev_small = cv2.resize(self.prev_gray, (sw, sh), interpolation=cv2.INTER_LINEAR)
            curr_small = cv2.resize(curr_gray, (sw, sh), interpolation=cv2.INTER_LINEAR)

            flow_small = cv2.calcOpticalFlowFarneback(
                prev_small, curr_small, None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.1, flags=0
            )
            flow = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR) * (1.0 / self.flow_scale)

            # Forward-warp the previous enhanced output
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (grid_x - flow[..., 0]).astype(np.float32)
            map_y = (grid_y - flow[..., 1]).astype(np.float32)
            warped_prev = cv2.remap(self.prev_output_bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            # Photometric error mask: avoid ghosting in disoccluded or rapid-motion zones
            photometric_err = np.mean(np.abs(frame_bgr.astype(np.float32) - warped_prev.astype(np.float32)), axis=2)
            valid_mask = np.clip((24.0 - photometric_err) / 24.0, 0.0, 1.0)[:, :, None]

            # Glare & specular light guard: prevent ghosting/motion streaks on headlights and streetlights
            glare_mask = (curr_gray > 190)[:, :, None].astype(np.float32)
            alpha = (self.ema_alpha * strength) * valid_mask * (1.0 - 0.85 * glare_mask)

            out_f = (1.0 - alpha) * frame_bgr.astype(np.float32) + alpha * warped_prev.astype(np.float32)
            out_bgr = np.clip(out_f, 0, 255).astype(np.uint8)
        else:
            diff = np.abs(frame_bgr.astype(np.float32) - self.prev_output_bgr.astype(np.float32))
            diff_gray = np.mean(diff, axis=2)
            motion_weight = np.clip(diff_gray / 30.0, 0.0, 1.0)[:, :, None]
            alpha = (self.ema_alpha * strength) * (1.0 - motion_weight)
            out_f = (1.0 - alpha) * frame_bgr.astype(np.float32) + alpha * self.prev_output_bgr.astype(np.float32)
            out_bgr = np.clip(out_f, 0, 255).astype(np.uint8)

        self.prev_output_bgr = out_bgr.copy()
        self.prev_gray = curr_gray.copy()
        return out_bgr
