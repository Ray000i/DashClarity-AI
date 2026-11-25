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
from src.models.scunet_fast import accelerate_scunet

_CL = torch.channels_last


def _bgr_to_tensor(frame_bgr: np.ndarray, device: torch.device, half: bool) -> torch.Tensor:
    """uint8 BGR HxWx3 -> float RGB 1x3xHxW in [0,1] on `device` (conversion done on the GPU)."""
    t = torch.from_numpy(np.ascontiguousarray(frame_bgr)).to(device, non_blocking=True)
    t = t.permute(2, 0, 1).flip(0).unsqueeze(0)
    t = (t.half() if half else t.float()) * (1.0 / 255.0)
    return t.contiguous(memory_format=_CL) if device.type == "cuda" else t.contiguous()


def _tensor_to_bgr(t: torch.Tensor, h: int, w: int) -> np.ndarray:
    """float RGB 1x3xHxW in [0,1] -> uint8 BGR HxWx3 (rounding, channel flip and crop on the GPU)."""
    t = (t[:, :, :h, :w].float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return t.squeeze(0).flip(0).permute(1, 2, 0).contiguous().cpu().numpy()

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

        # Tunables (defaults chosen from scripts/benchmark_gpu.py measurements, see docs/GPU_OPTIMIZATION.md)
        # Measured in scripts/quality_experiments.py (balanced preset, identical inputs):
        #  - smooth_exposure: real-clip flicker 2.48 -> 1.26, no other metric changes
        #  - curve_downscale 0.25 (as in the published Zero-DCE++): 37 -> 8 ms per 720p frame,
        #    quality neutral (SSIM +0.002, LPIPS +0.006, NIQE 8.49 -> 8.38, noise lower)
        #  - color_balance 0.5: LPIPS 0.470 -> 0.451, PSNR +0.21 dB (full gray-world tints scenes
        #    whose true average is not grey); half strength keeps the guard against sensor casts
        #  - clahe_clip: rejected by default (temporal error +35-62%, noise x2), kept as an option
        self.curve_downscale = 0.25
        self.color_balance = 0.5
        self.clahe_clip = 0.0
        self.smooth_exposure = True
        self._lum_ema: Optional[float] = None

    def reset(self) -> None:
        """Forget the exposure history (call between videos)."""
        self._lum_ema = None

    # Original step thresholds -> strength: <45: 1.0, 45-80: 0.60, 80-120: 0.22, >=120: 0.05.
    # A continuous curve through the middle of each step removes the frame-to-frame jumps when
    # the scene brightness hovers near a threshold (a measured source of flicker).
    _LUM_KNOTS = (35.0, 62.5, 100.0, 130.0)
    _STR_KNOTS = (1.00, 0.60, 0.22, 0.05)

    def _exposure(self, mean_lum: float, strength: float) -> Tuple[float, str]:
        lum = mean_lum
        if self.smooth_exposure:
            if self._lum_ema is None or abs(mean_lum - self._lum_ema) > 25.0:   # first frame / scene cut
                self._lum_ema = mean_lum
            else:
                self._lum_ema = 0.9 * self._lum_ema + 0.1 * mean_lum
            lum = self._lum_ema
            base = float(np.interp(lum, self._LUM_KNOTS, self._STR_KNOTS))
        else:
            base = 0.05 if lum >= 120.0 else 0.22 if lum >= 80.0 else 0.60 if lum >= 45.0 else 1.0
        condition = ("daytime_bright" if lum >= 120.0 else "daytime_moderate" if lum >= 80.0
                     else "twilight_mixed" if lum >= 45.0 else "nighttime_lowlight")
        return base * strength, condition

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_neighbor: Optional[np.ndarray] = None,
        strength: float = 1.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))

        # Adaptive exposure: prevents overexposure in daylight while boosting nighttime scenes
        adaptive_strength, condition = self._exposure(mean_lum, strength)

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
        if self.device.type == "cuda":
            t_rgb = (t_in[:, [2, 1, 0], :, :].half()) * (1.0 / 255.0)
            if not hasattr(self, "_model_fp16_checked"):
                self.model = self.model.half()
                self._model_fp16_checked = True
        else:
            t_rgb = (t_in[:, [2, 1, 0], :, :].float()) * (1.0 / 255.0)

        with torch.inference_mode():
            if self.curve_downscale < 1.0:
                # Zero-DCE++ as published: estimate the curve maps on a downscaled frame and
                # upsample them (smoother curves, far less compute); apply at full resolution.
                small = F.interpolate(t_rgb, scale_factor=self.curve_downscale, mode="bilinear",
                                      align_corners=False, antialias=True)
                _, A = self.model(small)
                A = F.interpolate(A, size=(h, w), mode="bilinear", align_corners=False)
                enh_t = self.model.enhance(t_rgb, A)
            else:
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
            if self.color_balance < 1.0:
                cb = self.color_balance
                kr, kg, kb = (1 + cb * (kr - 1)), (1 + cb * (kg - 1)), (1 + cb * (kb - 1))
            enh_clean = torch.cat([enh_t[:, 0:1] * kr, enh_t[:, 1:2] * kg, enh_t[:, 2:3] * kb], dim=1).clamp(0.0, 1.0)

            # Modulate with adaptive exposure strength
            if adaptive_strength < 1.0:
                out_t = t_rgb + (enh_clean - t_rgb) * adaptive_strength
            else:
                out_t = enh_clean

            out_bgr_t = out_t[:, [2, 1, 0], :, :].float().clamp(0.0, 1.0) * 255.0
            # round (the previous .byte() truncated, darkening every pixel by 0.5 level on average)
            out_bgr = np.ascontiguousarray(out_bgr_t.round().squeeze(0).byte().permute(1, 2, 0).cpu().numpy())

        if self.clahe_clip > 0:
            lab = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2LAB)
            if not hasattr(self, "_clahe") or self._clahe_clip_used != self.clahe_clip:
                self._clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
                self._clahe_clip_used = self.clahe_clip
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            out_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return out_bgr, {"condition": condition, "lum": mean_lum, "applied": True, "strength": round(adaptive_strength, 3)}


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
        fast_mode: bool = False,
        full_frame_max_px: int = 0
    ):
        self.device = device
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.fp16 = fp16
        self.fast_mode = fast_mode
        # Largest frame (in pixels) that may run untiled within this GPU's VRAM budget.
        # Untiled is faster and seam-free: 512 px tiles + 2x16 px overlap, padded by SCUNet to a
        # multiple of 64, roughly double the computed area of a 720p frame.
        self.full_frame_max_px = full_frame_max_px
        self.tile_state: Dict[str, Any] = {}

        w_path = get_weights_path("scunet_color_real_psnr")
        import spandrel
        loader = spandrel.ModelLoader()
        self.model = loader.load_from_file(str(w_path)).to(self.device).eval()
        self.patched_modules = accelerate_scunet(self.model)
        if self.device.type == 'cuda':
            if self.fp16:
                self.model = self.model.half()
            self.model.model.to(memory_format=_CL)
        logger.info("SCUNet neural denoiser loaded on %s (tile_size=%d, fast_mode=%s, full_frame_max_px=%d, "
                    "accelerated modules=%d)", device, tile_size, fast_mode, full_frame_max_px, self.patched_modules)

    def _infer(self, tensor: torch.Tensor) -> torch.Tensor:
        """Full frame when it fits the VRAM budget, otherwise tiled; an OOM degrades to smaller tiles."""
        h, w = tensor.shape[2:]
        untiled = (h * w <= self.full_frame_max_px) and not self.tile_state.get("tile_size")
        tile = max(h, w) if untiled else self.tile_size
        with torch.inference_mode():
            return run_tiled_inference(self.model, tensor, tile_size=tile, tile_pad=self.tile_pad,
                                       scale=1, fp16=self.fp16, tile_state=self.tile_state)

    def process_frame(self, frame_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
        if strength <= 0.05:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        half = self.fp16 and self.device.type == 'cuda'

        if self.fast_mode and (h > 720 or w > 1280):
            # For 1080p+ videos in fast mode: denoise at half resolution to maintain fast throughput
            sh, sw = (h // 2 // 2) * 2, (w // 2 // 2) * 2
            small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
            out_s_bgr = _tensor_to_bgr(self._infer(_bgr_to_tensor(small, self.device, half)), sh, sw)
            denoised_bgr = cv2.resize(out_s_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
        else:
            # Native resolution (SCUNet pads internally to a multiple of 64 with reflection)
            denoised_bgr = _tensor_to_bgr(self._infer(_bgr_to_tensor(frame_bgr, self.device, half)), h, w)

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
        if self.device.type == 'cuda':
            if self.fp16:
                self.model = self.model.half()
            self.model = self.model.to(memory_format=_CL)
        self.full_frame_max_px = 0
        self.tile_state: Dict[str, Any] = {}
        logger.info("SuperResolution (%s, scale=%dx) loaded on %s", model_name, self.scale, device)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        restore_at_original_res: bool = False,
        strength: float = 1.0
    ) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if self.device.type == 'cuda':
            return self._process_frame_gpu(frame_bgr, restore_at_original_res, strength)
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

    def _process_frame_gpu(self, frame_bgr: np.ndarray, restore_at_original_res: bool, strength: float) -> np.ndarray:
        """Same output policy as the CPU path, but the x2/x4 frame never leaves the GPU.

        The reference path copied the full x4 frame (e.g. 5120x2880 for 720p) to the CPU as
        float32, Lanczos-upscaled the input to that size for the 0.78/0.22 anti-ringing blend
        and then Lanczos-downscaled the result (~280 ms of CPU work per 720p frame). Blending is
        linear, so 'blend at x4 then downscale' equals 'downscale SR, then blend with the
        input'; the downscale uses antialiased bicubic filtering, which (unlike a Lanczos
        decimation without prefiltering) does not alias.
        """
        h, w = frame_bgr.shape[:2]
        half = self.fp16
        x = _bgr_to_tensor(frame_bgr, self.device, half)
        untiled = (h * w <= self.full_frame_max_px) and not self.tile_state.get("tile_size")
        tile = max(h, w) if untiled else self.tile_size
        with torch.inference_mode():
            sr = run_tiled_inference(self.model, x, tile_size=tile, tile_pad=self.tile_pad,
                                     scale=self.scale, fp16=self.fp16, tile_state=self.tile_state).float()
            src = x.float()
            if restore_at_original_res or (h >= 1080 or w >= 1920):
                sr = F.interpolate(sr, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
                out = 0.78 * sr + 0.22 * src
                if strength < 1.0:
                    out = strength * out + (1.0 - strength) * src
                return _tensor_to_bgr(out, h, w)
            ref = F.interpolate(src, size=sr.shape[2:], mode="bicubic", align_corners=False)
            out = 0.78 * sr + 0.22 * ref
            if strength < 1.0:
                out = strength * out + (1.0 - strength) * ref
            return _tensor_to_bgr(out, out.shape[2], out.shape[3])


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
        flow_scale: float = 0.5,
        device: Optional[torch.device] = None
    ):
        self.ema_alpha = ema_alpha
        self.motion_compensated = motion_compensated
        self.flow_scale = flow_scale
        self.device = device
        self.prev_output_bgr = None
        self.prev_gray = None
        self._prev_t = None
        self._grid_cache = None

    def reset(self):
        self.prev_output_bgr = None
        self.prev_gray = None
        self._prev_t = None

    def _flow(self, curr_gray: np.ndarray, h: int, w: int) -> np.ndarray:
        sw, sh = max(16, int(w * self.flow_scale)), max(16, int(h * self.flow_scale))
        prev_small = cv2.resize(self.prev_gray, (sw, sh), interpolation=cv2.INTER_LINEAR)
        curr_small = cv2.resize(curr_gray, (sw, sh), interpolation=cv2.INTER_LINEAR)
        return cv2.calcOpticalFlowFarneback(
            prev_small, curr_small, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.1, flags=0
        )

    def _process_gpu(self, frame_bgr: np.ndarray, curr_gray: np.ndarray, strength: float) -> np.ndarray:
        """Same algorithm as the CPU path: Farneback runs on the CPU at reduced scale (cheap);
        the full-resolution flow upsampling, backward warp, photometric/glare masks and the
        blend run on the GPU (they were ~75 ms of NumPy work per 720p frame)."""
        h, w = frame_bgr.shape[:2]
        dev = self.device
        with torch.inference_mode():
            cur = torch.from_numpy(frame_bgr).to(dev, non_blocking=True).permute(2, 0, 1).unsqueeze(0).float()
            flow_small = self._flow(curr_gray, h, w)
            flow = torch.from_numpy(flow_small).to(dev).permute(2, 0, 1).unsqueeze(0)
            flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False) * (1.0 / self.flow_scale)
            if self._grid_cache is None or self._grid_cache[0] != (h, w):
                gy, gx = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                                        torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
                self._grid_cache = ((h, w), gx, gy)
            _, gx, gy = self._grid_cache
            map_x = gx - flow[0, 0]
            map_y = gy - flow[0, 1]
            grid = torch.stack((map_x * (2.0 / max(w - 1, 1)) - 1.0, map_y * (2.0 / max(h - 1, 1)) - 1.0), dim=-1).unsqueeze(0)
            warped = F.grid_sample(self._prev_t, grid, mode="bilinear", padding_mode="reflection", align_corners=True)

            err = (cur - warped).abs().mean(dim=1, keepdim=True)
            valid = ((24.0 - err) / 24.0).clamp(0.0, 1.0)
            glare = (torch.from_numpy(curr_gray).to(dev) > 190).float()[None, None]
            alpha = (self.ema_alpha * strength) * valid * (1.0 - 0.85 * glare)
            out = ((1.0 - alpha) * cur + alpha * warped).clamp(0, 255).to(torch.uint8)  # truncation, as np.astype
            self._prev_t = out.float()
            return out.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()

    def process_frame(self, frame_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if (self.device is not None and self.device.type == "cuda" and self.motion_compensated
                and self._prev_t is not None and strength > 0.05
                and tuple(self._prev_t.shape[2:]) == frame_bgr.shape[:2]):
            out_bgr = self._process_gpu(frame_bgr, curr_gray, strength)
            self.prev_output_bgr = out_bgr
            self.prev_gray = curr_gray
            return out_bgr

        if self.prev_output_bgr is None or strength <= 0.05:
            self.prev_output_bgr = frame_bgr.copy()
            self.prev_gray = curr_gray.copy()
            self._store_prev_gpu(frame_bgr)
            return frame_bgr

        if self.prev_output_bgr.shape != frame_bgr.shape:
            self.prev_output_bgr = frame_bgr.copy()
            self.prev_gray = curr_gray.copy()
            self._store_prev_gpu(frame_bgr)
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
        self._store_prev_gpu(out_bgr)
        return out_bgr

    def _store_prev_gpu(self, out_bgr: np.ndarray) -> None:
        if self.device is not None and self.device.type == "cuda" and self.motion_compensated:
            self._prev_t = torch.from_numpy(out_bgr).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
