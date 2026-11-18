"""
STCD Model Wrapper (Optional Stage: stcd_lowlight)
Loads the IJCAI 2025 STCD architecture and correctly outputs the neural reconstruction.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import cv2
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
STCD_DIR = BASE_DIR / "models" / "STCD"
if str(STCD_DIR) not in sys.path:
    sys.path.insert(0, str(STCD_DIR))

import models.archs.low_light_transformer42 as stcd_arch

logger = logging.getLogger("DashClarity.Models.STCD")


class STCDWrapper:
    """Wrapper around low_light_transformer42 with direct, unfettered tensor inference."""
    def __init__(self, weights_path: str, device: torch.device):
        self.device = device
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"STCD weights not found: {weights_path}")

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

        ckpt = torch.load(str(self.weights_path), map_location="cpu", weights_only=False)
        state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt.items()}
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device).eval()
        logger.info("STCD model initialized successfully on %s", self.device)

    def _prepare_tensor(self, frame_bgr: np.ndarray) -> Tuple[torch.Tensor, int, int]:
        orig_h, orig_w = frame_bgr.shape[:2]
        pad_h = (16 - (orig_h % 16)) % 16
        pad_w = (16 - (orig_w % 16)) % 16
        if pad_h > 0 or pad_w > 0:
            frame_bgr = cv2.copyMakeBorder(frame_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor, orig_h, orig_w

    def _tensor_to_bgr(self, tensor: torch.Tensor, orig_h: int, orig_w: int) -> np.ndarray:
        tensor = tensor[:, :, :orig_h, :orig_w].squeeze(0).clamp(0.0, 1.0)
        img = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    @torch.no_grad()
    def enhance(self, frame_t: np.ndarray, frame_neighbor: np.ndarray) -> np.ndarray:
        """
        Runs STCD forward pass and returns the REAL neural output (raw_bgr / out_enh1).
        """
        t1, orig_h, orig_w = self._prepare_tensor(frame_t)
        t2, _, _ = self._prepare_tensor(frame_neighbor)
        out_rf1, out_rf2, out_enh1, out_enh2, out_ill1, out_ill2 = self.model(t1, t2)
        return self._tensor_to_bgr(out_enh1, orig_h, orig_w)
