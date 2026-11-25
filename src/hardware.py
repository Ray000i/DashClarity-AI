"""
DashClarity AI - Hardware detection and adaptive GPU profile.

Reads the real device at start-up (never hard-coded) and derives the processing settings
that bound peak VRAM: spatial tile size for SCUNet / Real-ESRGAN, whether full-frame
(untiled) inference is allowed, precision and the allocator budget.

The same code adapts to a 6 GB RTX 4050 Laptop GPU and an 8 GB RTX 4060 Laptop GPU:
the tier is chosen from the VRAM actually available, not from the GPU name.

Override with environment variables when needed:
    DASHCLARITY_TILE_SIZE=384      force a tile size
    DASHCLARITY_VRAM_BUDGET_MB=4500 force the VRAM budget used for tier selection
    DASHCLARITY_FP16=0             disable half precision
"""
from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger("DashClarity.Hardware")

# Measured on an RTX 4050 Laptop GPU (fp16, channels-last, scripts/benchmark_gpu.py):
#   SCUNet          ~1.5-1.6 KB of activation memory per input pixel (1080p frame: ~2.9 GB)
#   Real-ESRGAN x4  ~0.5-0.55 KB per input pixel                       (1080p frame: ~1.0 GB)
SCUNET_BYTES_PER_PX = 1700      # rounded up for safety
SR_BYTES_PER_PX = 600
RESERVE_MB = 700                # all model weights (~200 MB) + cuDNN workspaces + fragmentation
TILE_CHOICES = (512, 384, 256, 192)

# Tier names are informational (logged / shown in the API); the behaviour is derived from
# the measured VRAM budget so that any GPU between the tiers is handled sensibly.
TIER_NAMES = ((7500, "8gb_class (e.g. RTX 4060 Laptop)"), (5500, "6gb_class (e.g. RTX 4050 Laptop)"),
              (3500, "4gb_class"), (0, "low_vram"))


def _nvidia_smi() -> Dict[str, Any]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
        name, driver, total, used = [x.strip() for x in out[0].split(",")]
        return {"smi_name": name, "driver": driver, "smi_total_mb": float(total), "smi_used_mb": float(used)}
    except Exception:
        return {}


def detect_hardware() -> Dict[str, Any]:
    """Facts about the runtime: PyTorch build, CUDA availability, GPU and its free memory."""
    info: Dict[str, Any] = {
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    info.update(_nvidia_smi())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free_b, total_b = torch.cuda.mem_get_info(0)
        info.update(
            gpu=props.name,
            compute_capability=f"{props.major}.{props.minor}",
            vram_total_mb=round(total_b / 2 ** 20),
            vram_free_mb=round(free_b / 2 ** 20),
            fp16_tensor_cores=props.major >= 7,
        )
    elif info.get("smi_name") and torch.version.cuda is None:
        info["warning"] = (
            f"An NVIDIA GPU ({info['smi_name']}) is present but PyTorch {torch.__version__} is a CPU-only build. "
            "Install a CUDA build of PyTorch (see README) to use the GPU."
        )
    return info


def select_profile(hw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Derive tile sizes / full-frame limits / precision from the VRAM that is actually free."""
    hw = hw or detect_hardware()
    fp16_env = os.environ.get("DASHCLARITY_FP16", "1").lower() not in ("0", "false", "no")
    if not hw.get("cuda_available"):
        return {"tier": "cpu", "device": "cpu", "fp16": False, "tile_size": 512, "sr_tile_size": 512,
                "scunet_full_frame_max_px": 1920 * 1088, "sr_full_frame_max_px": 1920 * 1088, "vram_budget_mb": 0,
                "unload_unused_models": False}

    total, free = hw["vram_total_mb"], hw["vram_free_mb"]
    # Leave room for the desktop compositor (WDDM) and other applications: 85% of the card,
    # and never more than what is free right now minus 400 MB.
    budget = min(int(total * 0.85), int(free - 400))
    if os.environ.get("DASHCLARITY_VRAM_BUDGET_MB"):
        budget = int(os.environ["DASHCLARITY_VRAM_BUDGET_MB"])
    work = max(256, budget - RESERVE_MB) * 2 ** 20  # bytes available for one model's activations

    tile = TILE_CHOICES[-1]
    for t in TILE_CHOICES:  # largest tile whose padded footprint fits comfortably (<= 50% of work)
        padded = -(-(t + 32) // 64) * 64
        if padded * padded * SCUNET_BYTES_PER_PX <= 0.5 * work:
            tile = t
            break
    tier = next(name for min_mb, name in TIER_NAMES if total >= min_mb)
    profile = {
        "tier": tier, "device": "cuda", "fp16": bool(fp16_env and hw.get("fp16_tensor_cores", True)),
        "vram_total_mb": total, "vram_free_at_start_mb": free, "vram_budget_mb": budget,
        "tile_size": tile, "sr_tile_size": tile,
        "scunet_full_frame_max_px": int(work / SCUNET_BYTES_PER_PX),
        "sr_full_frame_max_px": int(work / SR_BYTES_PER_PX),
        # Keep only the active preset's models resident on 6 GB-class GPUs and below
        "unload_unused_models": budget < 6000,
    }
    if os.environ.get("DASHCLARITY_TILE_SIZE"):
        profile["tile_size"] = profile["sr_tile_size"] = int(os.environ["DASHCLARITY_TILE_SIZE"])
        profile["scunet_full_frame_max_px"] = profile["sr_full_frame_max_px"] = 0  # force tiling
    return profile


def configure_torch(profile: Dict[str, Any]) -> None:
    """Global PyTorch settings for inference."""
    if profile.get("device") != "cuda":
        return
    torch.backends.cudnn.benchmark = True          # fixed frame/tile shapes -> autotuned conv kernels
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        # Cap this process at the budget. On Windows (WDDM) exceeding physical VRAM silently
        # spills into shared system memory and runs many times slower; with the cap an
        # over-allocation raises an OOM instead, which the tiling code recovers from.
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
        torch.cuda.set_per_process_memory_fraction(min(0.95, max(0.3, profile["vram_budget_mb"] / total)), 0)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not set the per-process memory fraction: %s", exc)


def vram_snapshot() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    free_b, total_b = torch.cuda.mem_get_info(0)
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 2 ** 20, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 2 ** 20, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "device_free_mb": round(free_b / 2 ** 20, 1),
        "device_total_mb": round(total_b / 2 ** 20, 1),
    }
