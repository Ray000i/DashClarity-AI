"""
DashClarity AI - Modular Enhancement Pipeline Stages
Encapsulates neural restoration and generative enhancement stages with GPU-resident tensor chaining:
1. NeuralDenoiseStage: Pretrained SCUNet with adaptive multi-scale fast mode & luma detail preservation
2. LowLightStage: Retinex Luminance-Preserving Zero-DCE++ with C-infinity smooth toe & highlight guards
3. ClassicalDenoiseStage: Bilateral filter classical baseline for ablation
4. SuperResolutionStage: Real-ESRGAN with GPU anti-ringing blending & adaptive output resolution ceiling
5. TemporalSmoothingStage: Farneback backward-flow motion-compensated temporal consistency
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
from src.models.scunet_fast import accelerate_scunet
from src.models.weights_manager import get_weights_path
from src.pipeline.tiling import run_tiled_inference

_CL = torch.channels_last
logger = logging.getLogger("DashClarity.Pipeline.Stages")


def bgr_to_gpu_rgb_tensor(frame_bgr: np.ndarray, device: torch.device, fp16: bool) -> torch.Tensor:
    """
    Uploads uint8 BGR HWC numpy frame directly to target device (4x smaller PCIe transfer than float32)
    and converts to normalized BCHW RGB float/half tensor.
    """
    t_u8 = torch.from_numpy(np.ascontiguousarray(frame_bgr)).to(device, non_blocking=True)
    t_rgb = t_u8[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0)
    if fp16 and device.type == "cuda":
        return t_rgb.half() * (1.0 / 255.0)
    return t_rgb.float() * (1.0 / 255.0)


def gpu_rgb_tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    """Converts normalized BCHW RGB GPU tensor back to contiguous HWC uint8 BGR numpy array."""
    out_bgr_t = (tensor[:, [2, 1, 0], :, :].float().clamp(0.0, 1.0) * 255.0).round()
    out_bgr = out_bgr_t.squeeze(0).byte().permute(1, 2, 0).cpu().numpy()
    return np.ascontiguousarray(out_bgr)


class LowLightStage:
    """
    Adaptive Low-Light Illumination Enhancement using Zero-DCE++ with Retinex Luminance-Ratio
    Chromaticity Preservation and C-infinity smooth photometric guards:
    - Eliminates per-channel blue/magenta bias and Gray-World green/cyan failures.
    - Uses spatially smoothed local luminance field (double avg-pool) so night skies stay
      crisp without triggering discrete quantization contour bands.
    - Continuous scene luminance curve + temporal exposure EMA prevents headlight step-flicker.
    """

    def __init__(self, device: torch.device, model_type: str = "zero_dce_plus", fp16: bool = True):
        self.device = device
        self.model_type = model_type
        self.fp16 = bool(fp16 and device.type == "cuda")
        self.model = None
        self._prev_adap_strength: Optional[float] = None

        if model_type == "zero_dce_plus":
            w_path = get_weights_path("zero_dce_plus")
            self.model = ZeroDCEPlus().to(self.device).eval()
            ckpt = torch.load(str(w_path), map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt, strict=True)
            if self.fp16:
                self.model = self.model.half()
            logger.info("Zero-DCE++ loaded successfully on %s (fp16=%s)", device, self.fp16)
        elif model_type == "stcd_lowlight":
            w_path = get_weights_path("stcd_indoor")
            self.model = STCDWrapper(str(w_path), self.device)
            logger.info("STCD low-light wrapper loaded successfully on %s", device)
        else:
            raise ValueError(f"Unknown low-light model type: {model_type}")

    def reset(self):
        """Resets temporal exposure state between separate clips."""
        self._prev_adap_strength = None

    def _compute_scene_exposure(self, y_tensor: torch.Tensor, strength: float) -> Tuple[float, float, float, str]:
        """Computes continuous scene exposure strength & statistics with single batched GPU->CPU scalar read."""
        y_u8 = y_tensor * 255.0
        # Coarse grid (16x downsample -> ~80x45) reduces sort/kthvalue cost to <0.05ms while preserving distribution
        if min(y_u8.shape[-2:]) >= 64:
            y_coarse = F.avg_pool2d(y_u8, kernel_size=16, stride=16).flatten()
        elif min(y_u8.shape[-2:]) >= 16:
            y_coarse = F.avg_pool2d(y_u8, kernel_size=4, stride=4).flatten()
        else:
            y_coarse = y_u8.flatten()

        mean_t = y_u8.mean()
        n_el = y_coarse.numel()
        if n_el > 10:
            sorted_y, _ = torch.sort(y_coarse)
            p50_t = sorted_y[min(n_el - 1, int(0.50 * n_el))]
            p95_t = sorted_y[min(n_el - 1, int(0.95 * n_el))]
            mean_lum, p50_lum, p95_lum = torch.stack([mean_t, p50_t, p95_t]).tolist()
        else:
            mean_lum = float(mean_t.item())
            p50_lum, p95_lum = mean_lum, mean_lum

        scene_lum = 0.5 * mean_lum + 0.5 * p50_lum

        # Continuous ramp instead of discrete steps: avoids temporal exposure pumping across thresholds
        raw_adap = max(0.05, min(1.0, 1.0 - max(0.0, scene_lum - 42.0) / 88.0))
        if self._prev_adap_strength is not None:
            adap_ramp = 0.85 * self._prev_adap_strength + 0.15 * raw_adap
        else:
            adap_ramp = raw_adap
        self._prev_adap_strength = adap_ramp

        effective_strength = adap_ramp * strength
        if scene_lum >= 115.0:
            condition = "daytime_bright"
        elif scene_lum >= 80.0:
            condition = "daytime_moderate"
        elif scene_lum >= 45.0:
            condition = "twilight_mixed"
        else:
            condition = "nighttime_lowlight"

        return effective_strength, scene_lum, p95_lum, condition

    def process_frame_tensor(
        self,
        t_rgb: torch.Tensor,
        strength: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        GPU-resident Retinex-preserving Zero-DCE++ forward pass.
        Accepts and returns (1, 3, H, W) RGB tensor on self.device in [0, 1].
        """
        if self.model_type != "zero_dce_plus":
            raise NotImplementedError("process_frame_tensor supports zero_dce_plus only.")

        with torch.inference_mode():
            t_in = t_rgb.half() if (self.fp16 and t_rgb.dtype != torch.float16) else (
                t_rgb.float() if (not self.fp16 and t_rgb.dtype != torch.float32) else t_rgb
            )
            t_f = t_rgb.float()
            y_in = 0.299 * t_f[:, 0:1] + 0.587 * t_f[:, 1:2] + 0.114 * t_f[:, 2:3]

            adap_str, scene_lum, p95_lum, condition = self._compute_scene_exposure(y_in, strength)
            if adap_str <= 0.01:
                return t_rgb, {"condition": condition, "lum": round(scene_lum, 2), "applied": False}

            enh_t, _ = self.model(t_in)
            e_f = enh_t.float().clamp(0.0, 1.0)
            y_dce = 0.299 * e_f[:, 0:1] + 0.587 * e_f[:, 1:2] + 0.114 * e_f[:, 2:3]

            # Progressive gamma assist applies smoothly ONLY when entire frame lacks bright illumination
            dark_assist = max(0.0, min(1.0, (25.0 - scene_lum) / 14.0)) * max(0.0, min(1.0, (95.0 - p95_lum) / 30.0))
            if dark_assist > 0.01:
                eff_gamma = 1.0 - (1.0 - 0.72) * dark_assist
                y_dce_adj = torch.pow(y_dce.clamp(min=1e-4), eff_gamma)
            else:
                y_dce_adj = y_dce

            # C1-smooth local illumination field via cascaded 15x15 average pools:
            # Eliminates 8-bit macroblock steps so black-level sky protection causes zero contour banding
            y_local = F.avg_pool2d(y_in, kernel_size=15, stride=1, padding=7)
            y_local = F.avg_pool2d(y_local, kernel_size=15, stride=1, padding=7)
            y_smooth = 0.60 * y_local + 0.40 * y_in

            # Smooth C-infinity rational toe mask (preserves deep black night skies; relaxes on uniform dark scenes)
            eff_tau = 0.036 * (1.0 - 0.50 * dark_assist)
            w_toe = (y_smooth ** 2) / (y_smooth ** 2 + (eff_tau ** 2))

            # Smoothstep highlight anti-bloom knee (protects oncoming vehicle headlights & street lamps)
            t_hi = torch.clamp((0.90 - y_in) / 0.30, 0.0, 1.0)
            w_hi = t_hi * t_hi * (3.0 - 2.0 * t_hi)

            y_target = y_in + (y_dce_adj - y_in) * (w_toe * w_hi * adap_str)

            # Chromaticity-preserving Retinex gain scaling: scales RGB identically -> zero hue distortion!
            gain = torch.clamp(y_target / (y_in + 2e-3), 0.95, 7.2)

            # Smooth near-black IR dark-current chroma stabilization (only softens black sensor noise y_in < 0.035)
            chroma_keep = 0.35 + 0.65 * ((y_in ** 2) / (y_in ** 2 + 0.028 ** 2))
            t_neutral = y_in + (t_f - y_in) * chroma_keep

            out_f = t_neutral * gain

            # Soft highlight rolloff shoulder (>0.88) to retain specular edge texture without harsh clipping
            excess = F.relu(out_f - 0.88)
            out_f = (out_f - excess * (excess / (excess + 0.15))).clamp(0.0, 1.0)

            out_t = out_f.half() if self.fp16 else out_f
            return out_t, {"condition": condition, "lum": round(scene_lum, 2), "applied": True}

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_neighbor: Optional[np.ndarray] = None,
        strength: float = 1.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.model_type == "stcd_lowlight":
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            mean_lum = float(np.mean(gray))
            adaptive_strength = max(0.05, min(1.0, 1.0 - max(0.0, mean_lum - 42.0) / 88.0)) * strength
            condition = "nighttime_lowlight" if mean_lum < 45.0 else "twilight_mixed"
            if adaptive_strength <= 0.01:
                return frame_bgr.copy(), {"condition": condition, "lum": mean_lum, "applied": False}

            neigh = frame_neighbor if frame_neighbor is not None else frame_bgr
            raw_enh = self.model.enhance(frame_bgr, neigh)
            if raw_enh.shape[:2] != frame_bgr.shape[:2]:
                raw_enh = cv2.resize(raw_enh, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
            if raw_enh.dtype != frame_bgr.dtype:
                raw_enh = raw_enh.astype(frame_bgr.dtype)
            blended = cv2.addWeighted(raw_enh, adaptive_strength, frame_bgr, 1.0 - adaptive_strength, 0)
            return blended, {"condition": condition, "lum": mean_lum, "applied": True}

        t_rgb = bgr_to_gpu_rgb_tensor(frame_bgr, self.device, self.fp16)
        out_t, info = self.process_frame_tensor(t_rgb, strength=strength)
        if not info.get("applied", True):
            return frame_bgr.copy(), info
        return gpu_rgb_tensor_to_bgr(out_t), info


class NeuralDenoiseStage:
    """
    Pretrained Neural Denoiser: SCUNet loaded via spandrel.
    - Eliminates CMOS Poisson-Gaussian sensor noise and compression blockiness.
    - Preserves chroma 100% denoised while retaining signal-gated luma micro-texture
      (prevents synthetic waxy/oil-paint surfaces on foliage and road asphalt).
    - Fast mode uses GPU-adaptive resolution scaling with edge-aware high-frequency detail
      preservation + OOM safe fallback to tiled execution.
    """

    def __init__(
        self,
        device: torch.device,
        tile_size: int = 384,
        tile_pad: int = 16,
        fp16: bool = True,
        fast_mode: bool = False,
        tile_batch_size: int = 1,
        fast_scale: float = 0.67,
        fast_scale_1080p: float = 0.55,
        full_frame_max_px: int = 0
    ):
        self.device = device
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.fp16 = bool(fp16 and device.type == "cuda")
        self.fast_mode = fast_mode
        self.tile_batch_size = max(1, tile_batch_size)
        self.fast_scale = fast_scale
        self.fast_scale_1080p = fast_scale_1080p
        self.full_frame_max_px = max(0, int(full_frame_max_px))
        self.tile_state: Dict[str, Any] = {}

        w_path = get_weights_path("scunet_color_real_psnr")
        import spandrel
        loader = spandrel.ModelLoader()
        self.model = loader.load_from_file(str(w_path)).to(self.device).eval()
        self.patched_modules = accelerate_scunet(self.model)
        if self.device.type == "cuda":
            if self.fp16:
                self.model = self.model.half()
            if hasattr(self.model, "model"):
                self.model.model.to(memory_format=_CL)
            else:
                self.model.to(memory_format=_CL)
        logger.info(
            "SCUNet neural denoiser loaded on %s (tile_size=%d, batch=%d, fast_mode=%s, fast_scale=%.2f, max_untiled_px=%d, sdpa_blocks=%d)",
            device, tile_size, self.tile_batch_size, fast_mode, fast_scale, self.full_frame_max_px, self.patched_modules
        )

    def process_frame_tensor(self, t_rgb: torch.Tensor, strength: float = 1.0, detail_keep: float = 0.20) -> torch.Tensor:
        """
        Executes GPU-resident SCUNet denoising on (1, 3, H, W) RGB tensor in [0, 1].
        """
        if strength <= 0.05:
            return t_rgb

        _, _, h, w = t_rgb.shape
        t_in = t_rgb.half() if (self.fp16 and t_rgb.dtype != torch.float16) else (
            t_rgb.float() if (not self.fp16 and t_rgb.dtype != torch.float32) else t_rgb
        )

        denoised = None
        if self.fast_mode:
            scale_down = self.fast_scale_1080p if (h >= 1000 or w >= 1800) else self.fast_scale
            sh = max(32, ((int(h * scale_down) + 31) // 32) * 32)
            sw = max(32, ((int(w * scale_down) + 31) // 32) * 32)
            try:
                with torch.inference_mode():
                    t_small = F.interpolate(t_in, size=(sh, sw), mode="bilinear", align_corners=False)
                    if self.device.type == "cuda":
                        t_small = t_small.contiguous(memory_format=_CL)
                    out_s = self.model(t_small)
                    den_up = F.interpolate(out_s.float(), size=(h, w), mode="bilinear", align_corners=False)

                    # Edge-aware high-frequency detail transfer restores sharp text, plates & lanes without noise
                    t_f = t_in.float()
                    hf_in = t_f - F.avg_pool2d(t_f, kernel_size=3, stride=1, padding=1)
                    hf_lum = torch.abs(0.299 * hf_in[:, 0:1] + 0.587 * hf_in[:, 1:2] + 0.114 * hf_in[:, 2:3])
                    edge_w = (hf_lum ** 2) / (hf_lum ** 2 + 0.030 ** 2)
                    denoised = (den_up + hf_in * edge_w * 0.65).clamp(0.0, 1.0)
            except torch.cuda.OutOfMemoryError:
                logger.warning("CUDA OOM in SCUNet fast_mode (%dx%d). Falling back to tiled 1x inference.", sw, sh)
                torch.cuda.empty_cache()
                denoised = None

        if denoised is None:
            untiled = (self.full_frame_max_px > 0 and (h * w <= self.full_frame_max_px) and not self.tile_state.get("tile_size"))
            eff_tile = max(h, w) if untiled else self.tile_size
            t_in_exec = t_in.contiguous(memory_format=_CL) if self.device.type == "cuda" else t_in
            denoised = run_tiled_inference(
                model=self.model,
                tensor=t_in_exec,
                tile_size=eff_tile,
                tile_pad=self.tile_pad,
                scale=1,
                fp16=self.fp16,
                batch_size=self.tile_batch_size,
                tile_state=self.tile_state
            ).float().clamp(0.0, 1.0)

        # Preserve calibrated natural luma micro-texture outside dark flat noise areas while keeping chroma 100% denoised
        in_f = t_rgb.float()
        if detail_keep > 0.0:
            y_den = 0.299 * denoised[:, 0:1] + 0.587 * denoised[:, 1:2] + 0.114 * denoised[:, 2:3]
            y_in = 0.299 * in_f[:, 0:1] + 0.587 * in_f[:, 1:2] + 0.114 * in_f[:, 2:3]
            sig_w = (y_in ** 2) / (y_in ** 2 + 0.065 ** 2)
            y_mix = y_den + (y_in - y_den) * (detail_keep * sig_w)
            ratio = torch.clamp(y_mix / (y_den + 1e-4), 0.70, 1.40)
            denoised = (denoised * ratio).clamp(0.0, 1.0)

        if strength < 1.0:
            denoised = in_f + (denoised - in_f) * strength

        return denoised.half() if self.fp16 else denoised

    def process_frame(self, frame_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
        if strength <= 0.05:
            return frame_bgr
        t_rgb = bgr_to_gpu_rgb_tensor(frame_bgr, self.device, self.fp16)
        out_t = self.process_frame_tensor(t_rgb, strength=strength)
        return gpu_rgb_tensor_to_bgr(out_t)


class ClassicalDenoiseStage:
    """Classical Bilateral baseline for fast preview and ablation comparison."""

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
    Performs GPU-resident upscaling + bicubic-antialias resolution capping & anti-ringing
    blending directly on GPU before CPU transfer (eliminates 8K ~400MB RAM/PCIe spikes).
    """

    def __init__(
        self,
        device: torch.device,
        model_name: str = "realesr_general_x4v3",
        tile_size: int = 384,
        tile_pad: int = 16,
        fp16: bool = True,
        tile_batch_size: int = 1,
        max_out_dim: int = 3840,
        full_frame_max_px: int = 0
    ):
        self.device = device
        self.model_name = model_name
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.fp16 = bool(fp16 and device.type == "cuda")
        self.tile_batch_size = max(1, tile_batch_size)
        self.max_out_dim = max_out_dim
        self.full_frame_max_px = max(0, int(full_frame_max_px))
        self.tile_state: Dict[str, Any] = {}

        if model_name == "realesr_x2plus":
            w_path = get_weights_path("realesr_x2plus")
            self.model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=23, num_grow_ch=32)
            ckpt = torch.load(str(w_path), map_location="cpu", weights_only=False)
            state_dict = ckpt["params_ema"] if "params_ema" in ckpt else (ckpt["params"] if "params" in ckpt else ckpt)
            self.model.load_state_dict(state_dict, strict=True)
            self.scale = 2
        elif model_name == "realesr_general_x4v3":
            w_path = get_weights_path("realesr_general_x4v3")
            self.model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")
            ckpt = torch.load(str(w_path), map_location="cpu", weights_only=False)
            state_dict = ckpt["params"] if "params" in ckpt else ckpt
            self.model.load_state_dict(state_dict, strict=True)
            self.scale = 4
        else:
            raise ValueError(f"Unknown SuperResolution model: {model_name}")

        self.model = self.model.to(self.device).eval()
        if self.device.type == "cuda":
            if self.fp16:
                self.model = self.model.half()
            self.model = self.model.to(memory_format=_CL)
        logger.info(
            "SuperResolution (%s, scale=%dx, tile=%d, max_out=%d, max_untiled_px=%d) loaded on %s",
            model_name, self.scale, tile_size, max_out_dim, self.full_frame_max_px, device
        )

    def process_frame_tensor(
        self,
        t_rgb: torch.Tensor,
        restore_at_original_res: bool = False,
        strength: float = 1.0,
        anti_ringing_blend: float = 0.22
    ) -> torch.Tensor:
        """
        GPU-resident Super-Resolution pass with anti-ringing blend and resolution bounding.
        """
        _, _, h, w = t_rgb.shape
        t_in = t_rgb.half() if (self.fp16 and t_rgb.dtype != torch.float16) else (
            t_rgb.float() if (not self.fp16 and t_rgb.dtype != torch.float32) else t_rgb
        )
        untiled = (self.full_frame_max_px > 0 and (h * w <= self.full_frame_max_px) and not self.tile_state.get("tile_size"))
        eff_tile = max(h, w) if untiled else self.tile_size
        t_in_exec = t_in.contiguous(memory_format=_CL) if self.device.type == "cuda" else t_in

        with torch.inference_mode():
            out_sr = run_tiled_inference(
                model=self.model,
                tensor=t_in_exec,
                tile_size=eff_tile,
                tile_pad=self.tile_pad,
                scale=self.scale,
                fp16=self.fp16,
                batch_size=self.tile_batch_size,
                tile_state=self.tile_state
            )

            # Determine final target spatial dimensions cleanly on GPU
            if restore_at_original_res or (h >= 1080 or w >= 1920):
                target_h, target_w = h, w
            else:
                sr_h, sr_w = out_sr.shape[-2:]
                max_curr = max(sr_h, sr_w)
                if self.max_out_dim > 0 and max_curr > self.max_out_dim:
                    scale_cap = self.max_out_dim / float(max_curr)
                    target_h = max(2, (int(sr_h * scale_cap) // 2) * 2)
                    target_w = max(2, (int(sr_w * scale_cap) // 2) * 2)
                else:
                    target_h, target_w = sr_h, sr_w

            # Perform downscaling and anti-ringing reference blending entirely on GPU
            # Avoids copying giant 5K/8K tensors to CPU RAM!
            if out_sr.shape[-2:] != (target_h, target_w):
                sr_resized = F.interpolate(
                    out_sr.float(), size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=True
                ).clamp(0.0, 1.0)
            else:
                sr_resized = out_sr.float().clamp(0.0, 1.0)

            if t_in.shape[-2:] != (target_h, target_w):
                ref_resized = F.interpolate(t_in.float(), size=(target_h, target_w), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
            else:
                ref_resized = t_in.float()

            # Anti-ringing blend (22% smooth reference) + user strength modulation
            sr_clean = (1.0 - anti_ringing_blend) * sr_resized + anti_ringing_blend * ref_resized
            if strength < 1.0:
                sr_clean = ref_resized + (sr_clean - ref_resized) * strength

            return sr_clean.half() if self.fp16 else sr_clean

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        restore_at_original_res: bool = False,
        strength: float = 1.0
    ) -> np.ndarray:
        t_rgb = bgr_to_gpu_rgb_tensor(frame_bgr, self.device, self.fp16)
        out_t = self.process_frame_tensor(
            t_rgb,
            restore_at_original_res=restore_at_original_res,
            strength=strength
        )
        return gpu_rgb_tensor_to_bgr(out_t)


class TemporalSmoothingStage:
    """
    Motion-compensated temporal consistency using Farneback backward dense optical flow.
    - Runs reduced-scale Farneback optical flow on CPU (capped at 360p height for 4K responsiveness).
    - Performs flow upsampling, F.grid_sample backward warping, photometric gating,
      glare suppression, and EMA blending directly on GPU tensors to stay zero-copy in pipeline.
    """

    def __init__(
        self,
        ema_alpha: float = 0.18,
        motion_compensated: bool = True,
        flow_scale: float = 0.5,
        device: Optional[torch.device] = None
    ):
        self.ema_alpha = ema_alpha
        self.motion_compensated = motion_compensated
        self.flow_scale = flow_scale
        self.device = device
        self.prev_output_bgr: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None
        self._prev_t: Optional[torch.Tensor] = None
        self._grid_shape: Optional[Tuple[int, int]] = None
        self._grid_x: Optional[np.ndarray] = None
        self._grid_y: Optional[np.ndarray] = None
        self._gpu_grid_cache: Optional[Tuple[Tuple[int, int, str], torch.Tensor, torch.Tensor]] = None

    def reset(self):
        self.prev_output_bgr = None
        self.prev_gray = None
        self._prev_t = None

    def _get_grid(self, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._grid_shape != (h, w) or self._grid_x is None or self._grid_y is None:
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            self._grid_x = gx
            self._grid_y = gy
            self._grid_shape = (h, w)
        return self._grid_x, self._grid_y

    def _get_gpu_grid(self, h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (h, w, str(device))
        if self._gpu_grid_cache is None or self._gpu_grid_cache[0] != cache_key:
            gy, gx = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing="ij"
            )
            self._gpu_grid_cache = (cache_key, gx, gy)
        return self._gpu_grid_cache[1], self._gpu_grid_cache[2]

    def process_frame_tensor(self, t_rgb: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """
        GPU-resident motion-compensated temporal smoothing on (1, 3, H, W) RGB tensor in [0, 1].
        Eliminates extra full-frame CPU<->GPU transfers by computing Farneback flow on compact gray luma
        and running full-resolution warping and gating directly on t_rgb.device.
        """
        _, _, h, w = t_rgb.shape
        dev = t_rgb.device
        out_dtype = t_rgb.dtype
        cur_f = t_rgb.float()

        with torch.inference_mode():
            gray_u8 = (
                (0.299 * cur_f[:, 0] + 0.587 * cur_f[:, 1] + 0.114 * cur_f[:, 2])
                .clamp(0.0, 1.0) * 255.0
            ).round().byte().squeeze(0).cpu().numpy()

            if (
                self._prev_t is None
                or strength <= 0.05
                or tuple(self._prev_t.shape[-2:]) != (h, w)
                or self._prev_t.device != dev
            ):
                self._prev_t = (cur_f.clamp(0.0, 1.0) * 255.0).round() / 255.0
                self.prev_gray = gray_u8
                return t_rgb

            if self.motion_compensated and self.prev_gray is not None:
                eff_scale = min(self.flow_scale, 360.0 / max(1, h))
                sw = max(16, int(w * eff_scale))
                sh = max(16, int(h * eff_scale))

                prev_small = cv2.resize(self.prev_gray, (sw, sh), interpolation=cv2.INTER_AREA)
                curr_small = cv2.resize(gray_u8, (sw, sh), interpolation=cv2.INTER_AREA)

                flow_small = cv2.calcOpticalFlowFarneback(
                    curr_small, prev_small, None,
                    pyr_scale=0.5, levels=3, winsize=13, iterations=3, poly_n=5, poly_sigma=1.1, flags=0
                )

                flow_t = torch.from_numpy(np.ascontiguousarray(flow_small)).to(dev, non_blocking=True)
                flow_t = flow_t.permute(2, 0, 1).unsqueeze(0)
                flow_up = F.interpolate(flow_t, size=(h, w), mode="bilinear", align_corners=False)

                gx, gy = self._get_gpu_grid(h, w, dev)
                map_x = gx + flow_up[0, 0] * (float(w) / float(sw))
                map_y = gy + flow_up[0, 1] * (float(h) / float(sh))

                norm_x = map_x * (2.0 / max(w - 1, 1)) - 1.0
                norm_y = map_y * (2.0 / max(h - 1, 1)) - 1.0
                grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)

                warped = F.grid_sample(self._prev_t, grid, mode="bilinear", padding_mode="reflection", align_corners=True)

                photometric_err = (cur_f - warped).abs().mean(dim=1, keepdim=True) * 255.0
                valid_mask = ((22.0 - photometric_err) / 22.0).clamp(0.0, 1.0)

                y_u8 = (0.299 * cur_f[:, 0:1] + 0.587 * cur_f[:, 1:2] + 0.114 * cur_f[:, 2:3]) * 255.0
                glare_weight = 1.0 - 0.88 * ((y_u8 - 180.0) / 50.0).clamp(0.0, 1.0)
                alpha = (self.ema_alpha * strength) * valid_mask * glare_weight
                out_f = (cur_f + alpha * (warped - cur_f)).clamp(0.0, 1.0)
            else:
                diff_gray = (cur_f - self._prev_t).abs().mean(dim=1, keepdim=True) * 255.0
                motion_weight = (diff_gray / 28.0).clamp(0.0, 1.0)
                alpha = (self.ema_alpha * strength) * (1.0 - motion_weight)
                out_f = (cur_f + alpha * (self._prev_t - cur_f)).clamp(0.0, 1.0)

            self._prev_t = (out_f * 255.0).round() / 255.0
            self.prev_gray = gray_u8
            return out_f.half() if out_dtype == torch.float16 else out_f

    def process_frame(self, frame_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
        if self.device is not None and self.device.type == "cuda":
            t_rgb = bgr_to_gpu_rgb_tensor(frame_bgr, self.device, fp16=False)
            out_t = self.process_frame_tensor(t_rgb, strength=strength)
            out_bgr = gpu_rgb_tensor_to_bgr(out_t)
            self.prev_output_bgr = out_bgr
            return out_bgr

        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.prev_output_bgr is None or strength <= 0.05:
            self.prev_output_bgr = frame_bgr.copy()
            self.prev_gray = curr_gray
            return frame_bgr

        if self.prev_output_bgr.shape != frame_bgr.shape:
            self.prev_output_bgr = frame_bgr.copy()
            self.prev_gray = curr_gray
            return frame_bgr

        h, w = frame_bgr.shape[:2]

        if self.motion_compensated and self.prev_gray is not None:
            # Cap Farneback calculation height (~360p equivalent) so 1440p/4K outputs don't stall on CPU
            eff_scale = min(self.flow_scale, 360.0 / max(1, h))
            sw = max(16, int(w * eff_scale))
            sh = max(16, int(h * eff_scale))

            prev_small = cv2.resize(self.prev_gray, (sw, sh), interpolation=cv2.INTER_AREA)
            curr_small = cv2.resize(curr_gray, (sw, sh), interpolation=cv2.INTER_AREA)

            # Backward flow from curr -> prev directly indexes prev at destination pixel (x, y)
            flow_small = cv2.calcOpticalFlowFarneback(
                curr_small, prev_small, None,
                pyr_scale=0.5, levels=3, winsize=13, iterations=3, poly_n=5, poly_sigma=1.1, flags=0
            )
            scale_x = float(w) / float(sw)
            scale_y = float(h) / float(sh)
            flow = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR)
            flow[..., 0] *= scale_x
            flow[..., 1] *= scale_y

            grid_x, grid_y = self._get_grid(w, h)
            map_x = grid_x + flow[..., 0]
            map_y = grid_y + flow[..., 1]
            warped_prev = cv2.remap(self.prev_output_bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            # Photometric error gate: drop temporal weight smoothly wherever alignment error or disocclusion occurs
            curr_f = frame_bgr.astype(np.float32)
            warp_f = warped_prev.astype(np.float32)
            photometric_err = np.mean(np.abs(curr_f - warp_f), axis=2)
            valid_mask = np.clip((22.0 - photometric_err) / 22.0, 0.0, 1.0)[:, :, None]

            # Glare & specular light guard: avoid ghosting streaks on oncoming headlights and signs
            glare_weight = 1.0 - 0.88 * np.clip((curr_gray.astype(np.float32) - 180.0) / 50.0, 0.0, 1.0)[:, :, None]
            alpha = (self.ema_alpha * strength) * valid_mask * glare_weight

            out_f = curr_f + alpha * (warp_f - curr_f)
            out_bgr = np.clip(out_f, 0, 255).round().astype(np.uint8)
        else:
            curr_f = frame_bgr.astype(np.float32)
            prev_f = self.prev_output_bgr.astype(np.float32)
            diff_gray = np.mean(np.abs(curr_f - prev_f), axis=2)
            motion_weight = np.clip(diff_gray / 28.0, 0.0, 1.0)[:, :, None]
            alpha = (self.ema_alpha * strength) * (1.0 - motion_weight)
            out_f = curr_f + alpha * (prev_f - curr_f)
            out_bgr = np.clip(out_f, 0, 255).round().astype(np.uint8)

        self.prev_output_bgr = out_bgr
        self.prev_gray = curr_gray
        return out_bgr
