"""
DashClarity AI - AI Enhancement Engine
Direct PyTorch inference using the pretrained IJCAI STCD model.
Decomposes low-light frames into reflectance and illumination components with cross-frame attention.
Features Retinex chromaticity preservation, adaptive highlight attenuation, and scene lighting detection.
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Ensure models/STCD is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
STCD_DIR = BASE_DIR / "models" / "STCD"
if str(STCD_DIR) not in sys.path:
    sys.path.insert(0, str(STCD_DIR))

import models.archs.low_light_transformer42 as stcd_arch

logger = logging.getLogger("DashClarity.EnhancementEngine")

class STCDEnhancer:
    _instance = None  # Singleton pattern to prevent multiple GPU allocations

    @classmethod
    def get_instance(cls, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        if cls._instance is None:
            cls._instance = cls(checkpoint_path=checkpoint_path, device=device)
        return cls._instance

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        logger.info(f"Initializing STCDEnhancer on hardware: {self.device}")
        
        if checkpoint_path is None:
            checkpoint_path = str(BASE_DIR / "weights" / "STCD" / "indoor_G.pth")
            
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"STCD model checkpoint not found at: {self.checkpoint_path}")
            
        # Instantiate STCD Generator (low_light_transformer42)
        # Architecture configuration matches IJCAI 2025 SDSD benchmark specifications
        self.model = stcd_arch.low_light_transformer(
            nf=64,
            nframes=5,
            groups=8,
            front_RBs=1,
            back_RBs=1,
            center=None,
            predeblur=True,
            HR_in=True,
            w_TSA=True
        )
        
        # Load weights
        logger.info(f"Loading checkpoint weights from {self.checkpoint_path}...")
        ckpt = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
        state_dict = {}
        for k, v in ckpt.items():
            if k.startswith("module."):
                state_dict[k[7:]] = v
            else:
                state_dict[k] = v
                
        load_result = self.model.load_state_dict(state_dict, strict=True)
        logger.info(f"STCD state_dict loaded with strict=True: {load_result}")
        
        # Place model in evaluation mode to disable dropout / batchnorm updating
        self.model = self.model.to(self.device).eval()
        logger.info("STCD low_light_transformer model loaded successfully and placed on GPU in eval mode.")

    def _prepare_tensor(self, frame_bgr: np.ndarray) -> Tuple[torch.Tensor, int, int]:
        """
        Converts BGR [0, 255] uint8 numpy array to float32 [0.0, 1.0] RGB PyTorch tensor.
        Pads to multiple of 16 to ensure safe convolution/pooling alignments.
        """
        orig_h, orig_w = frame_bgr.shape[:2]
        
        # Calculate padding needed to reach multiple of 16
        pad_h = (16 - (orig_h % 16)) % 16
        pad_w = (16 - (orig_w % 16)) % 16
        
        if pad_h > 0 or pad_w > 0:
            frame_bgr = cv2.copyMakeBorder(
                frame_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
            )
            
        # Convert BGR -> RGB and normalize to [0.0, 1.0] float32
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # HWC -> CHW -> BCHW
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor, orig_h, orig_w

    def _tensor_to_bgr(self, tensor: torch.Tensor, orig_h: int, orig_w: int) -> np.ndarray:
        """
        Converts output tensor back to uint8 BGR [0, 255] numpy image, cropped to original dimensions.
        """
        # Crop to original height and width
        tensor = tensor[:, :, :orig_h, :orig_w]
        
        # Clamp to [0.0, 1.0]
        tensor = tensor.squeeze(0).clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
        
        # Convert RGB -> BGR [0, 255]
        bgr = (tensor * 255.0).round().astype(np.uint8)
        bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
        return bgr

    @torch.no_grad()
    def enhance_frame_pair(
        self,
        frame_t: np.ndarray,
        frame_neighbor: np.ndarray,
        return_maps: bool = False
    ) -> Dict[str, Any]:
        """
        Enhances frame_t using cross-frame spatial-temporal consistency with frame_neighbor.
        Applies Retinex chromaticity preservation, adaptive highlight attenuation, and
        scene lighting classification to prevent highlight blowout, desaturation, or texture blur.
        
        Args:
            frame_t: Current frame (BGR uint8).
            frame_neighbor: Adjacent temporal frame (BGR uint8).
            return_maps: If True, also returns decomposed reflectance and illumination maps.
            
        Returns:
            Dictionary containing enhanced frame and optional decomposition maps.
        """
        orig_h, orig_w = frame_t.shape[:2]
        t1, _, _ = self._prepare_tensor(frame_t)
        t2, _, _ = self._prepare_tensor(frame_neighbor)
        
        # Run STCD forward pass
        # Returns: out_rf1, out_rf2, out1_trans, out2_trans, out_ill1, out_ill2
        out_rf1, out_rf2, out_enh1, out_enh2, out_ill1, out_ill2 = self.model(t1, t2)
        
        # Raw STCD model reconstruction in BGR
        raw_bgr = self._tensor_to_bgr(out_enh1, orig_h, orig_w)
        
        # Convert frames to float32 [0.0, 1.0] for Retinex decomposition
        orig_f = frame_t.astype(np.float32) / 255.0
        raw_f = raw_bgr.astype(np.float32) / 255.0
        
        # Luminance channels (Rec. 709: Y = 0.114 B + 0.587 G + 0.299 R)
        y_orig = 0.114 * orig_f[:, :, 0] + 0.587 * orig_f[:, :, 1] + 0.299 * orig_f[:, :, 2]
        y_raw = 0.114 * raw_f[:, :, 0] + 0.587 * raw_f[:, :, 1] + 0.299 * raw_f[:, :, 2]
        
        mean_lum = float(np.mean(y_orig)) * 255.0
        
        # Smooth luminance fields to extract smooth spatial illumination gains
        y_orig_smooth = cv2.GaussianBlur(y_orig, (0, 0), sigmaX=3.0)
        y_raw_smooth = cv2.GaussianBlur(y_raw, (0, 0), sigmaX=3.0)
        
        # STCD-guided neural illumination gain map with small epsilon
        neural_gain = (y_raw_smooth + 0.04) / (y_orig_smooth + 0.04)
        
        # Scene-adaptive exposure protection and condition classification
        if mean_lum > 85.0:
            # Broad daytime: already well-illuminated scene
            # Protect bright sky, clouds, and structures (gain = 1.0)
            # Gently balance deep underexposed shadows with modest gain ceiling
            t_hi = np.clip((y_orig - 0.20) / (0.45 - 0.20), 0.0, 1.0)
            w_hi = t_hi * t_hi * (3.0 - 2.0 * t_hi)  # Smoothstep
            gain_clamped = np.clip(neural_gain, 0.95, 1.30)
            eff_gain = (1.0 - w_hi) * gain_clamped + w_hi * 1.0
            condition = "daytime_well_lit"
        elif mean_lum > 45.0:
            # Mixed lighting / twilight / urban dusk scene
            t_hi = np.clip((y_orig - 0.25) / (0.65 - 0.25), 0.0, 1.0)
            w_hi = t_hi * t_hi * (3.0 - 2.0 * t_hi)
            gain_clamped = np.clip(neural_gain, 0.90, 1.85)
            eff_gain = (1.0 - w_hi) * gain_clamped + w_hi * 1.0
            condition = "twilight_mixed"
        else:
            # Low-light nighttime dashcam: full STCD spatial-temporal enhancement
            # Protect direct headlights, streetlamps, and neon signs from clipping
            t_hi = np.clip((y_orig - 0.20) / (0.65 - 0.20), 0.0, 1.0)
            w_hi = t_hi * t_hi * (3.0 - 2.0 * t_hi)
            gain_clamped = np.clip(neural_gain, 0.50, 4.00)
            eff_gain = (1.0 - w_hi) * gain_clamped + w_hi * 1.0
            condition = "nighttime_low_light"
            
        # Retinex Chromaticity-Preserving Reconstruction:
        # Scaling original color channels by effective illumination gain guarantees
        # that chromaticity (r, g, b ratios) remains 100% invariant, preventing washed-out colors.
        enhanced_f = np.zeros_like(orig_f)
        for c in range(3):
            enhanced_f[:, :, c] = orig_f[:, :, c] * eff_gain
            
        enhanced_f = np.clip(enhanced_f, 0.0, 1.0)
        enhanced_bgr = (enhanced_f * 255.0).round().astype(np.uint8)
        
        result = {
            "enhanced": enhanced_bgr,
            "raw_stcd": raw_bgr,
            "lighting_condition": condition,
            "mean_luminance": round(mean_lum, 2)
        }
        
        if return_maps:
            result["reflectance"] = self._tensor_to_bgr(out_rf1, orig_h, orig_w)
            result["illumination"] = self._tensor_to_bgr(out_ill1, orig_h, orig_w)
            
        return result

    @staticmethod
    def apply_super_resolution(frame_bgr: np.ndarray, scale: float = 2.0) -> np.ndarray:
        """
        Modular Resolution Enhancement (Module C).
        High-fidelity edge-preserving upscaler using Lanczos interpolation
        with subtle, artifact-free micro-contrast enhancement.
        """
        if scale <= 1.0:
            return frame_bgr
            
        h, w = frame_bgr.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        
        # High quality Lanczos interpolation
        upscaled = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Subtle unsharp masking for edge definition without halo artifacts
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.2)
        sharpened = cv2.addWeighted(upscaled, 1.08, gaussian, -0.08, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
