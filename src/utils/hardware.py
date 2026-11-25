"""
DashClarity AI - Adaptive Hardware Detection & VRAM Profiling Module
Dynamically detects CUDA GPU model, VRAM capacity, system RAM, and configures optimal inference
parameters for both RTX 4050 (6 GB VRAM) and RTX 4060 (8 GB VRAM) without hardcoded assumptions.
"""

import os
import gc
import sys
import ctypes
import logging
import subprocess
from typing import Dict, Any, Optional

import torch

logger = logging.getLogger("DashClarity.Utils.Hardware")


def _probe_nvidia_driver() -> Optional[str]:
    """Queries NVIDIA driver version cleanly via nvidia-smi when available."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        return lines[0] if lines else None
    except Exception:
        return None


def get_system_ram_stats() -> Dict[str, float]:
    """Detects total and available physical system RAM in GB."""
    try:
        if sys.platform == "win32":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return {
                "total_gb": round(stat.ullTotalPhys / (1024 ** 3), 2),
                "available_gb": round(stat.ullAvailPhys / (1024 ** 3), 2),
                "used_pct": float(stat.dwMemoryLoad),
            }
        else:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                lines = f.readlines()
            mem = {}
            for line in lines:
                parts = line.split(":")
                if len(parts) == 2:
                    val = float(parts[1].strip().split()[0]) / (1024 ** 2)
                    mem[parts[0].strip()] = val
            total_gb = round(mem.get("MemTotal", 0.0), 2)
            avail_gb = round(mem.get("MemAvailable", mem.get("MemFree", 0.0)), 2)
            used_pct = round(((total_gb - avail_gb) / max(1e-5, total_gb)) * 100.0, 1)
            return {"total_gb": total_gb, "available_gb": avail_gb, "used_pct": used_pct}
    except Exception as e:
        logger.warning("System RAM detection fallback triggered: %s", e)
        return {"total_gb": 0.0, "available_gb": 0.0, "used_pct": 0.0}


def select_gpu_profile(
    total_vram_gb: float,
    cuda_available: bool = True,
    free_vram_gb: Optional[float] = None
) -> Dict[str, Any]:
    """
    Selects adaptive processing parameters based on actual detected VRAM capacity and free headroom.
    Designed to keep identical perceptual enhancement quality across RTX 4050 (6GB)
    and RTX 4060 (8GB) while maintaining safe GPU memory bounds and enabling fast untiled SCUNet
    inference when sufficient VRAM headroom exists.
    """
    if not cuda_available or total_vram_gb <= 0.0:
        return {
            "tier": "cpu_fallback",
            "tile_size": 256,
            "tile_pad": 16,
            "tile_batch_size": 1,
            "fp16": False,
            "scunet_fast_scale": 0.50,
            "scunet_fast_scale_1080p": 0.50,
            "scunet_full_frame_max_px": 0,
            "sr_full_frame_max_px": 0,
            "sr_max_out_dim": 1920,
            "sequential_offload": True,
            "vram_headroom_mb": 0.0,
            "vram_budget_mb": 0.0,
        }

    total_mb = total_vram_gb * 1024.0
    if free_vram_gb is not None and free_vram_gb > 0.0:
        free_mb = min(total_mb, free_vram_gb * 1024.0)
        budget_mb = max(512.0, min(total_mb * 0.85, free_mb - 450.0))
    else:
        budget_mb = max(512.0, total_mb * 0.80)

    # Activation + PyTorch allocator reservation margins (~2.1 KB/px SCUNet, ~0.75 KB/px Real-ESRGAN, 750 MB reserve)
    # Ensures 720p runs untiled on 6GB RTX 4050 (~1.75 GB resv) while keeping 1080p safely tiled on 6GB (and untiled on 8GB RTX 4060).
    work_bytes = max(256.0, budget_mb - 750.0) * (1024.0 ** 2)
    scunet_max_px = int(work_bytes / 2100.0)
    sr_max_px = int(work_bytes / 750.0)

    if total_vram_gb >= 7.2:
        # 8 GB+ GPUs (e.g., NVIDIA GeForce RTX 4060 Laptop GPU)
        return {
            "tier": "high_vram_8gb",
            "tile_size": 512,
            "tile_pad": 16,
            "tile_batch_size": 2,
            "fp16": True,
            "scunet_fast_scale": 0.67,
            "scunet_fast_scale_1080p": 0.60,
            "scunet_full_frame_max_px": scunet_max_px,
            "sr_full_frame_max_px": sr_max_px,
            "sr_max_out_dim": 3840,
            "sequential_offload": False,
            "vram_headroom_mb": 1024.0,
            "vram_budget_mb": round(budget_mb, 1),
        }
    elif total_vram_gb >= 4.8:
        # ~6 GB GPUs (e.g., NVIDIA GeForce RTX 4050 Laptop GPU)
        # Allow untiled 720p (~0.92 MPx) when >= 3.2 GB VRAM is free; fall back to 384px tiles otherwise
        return {
            "tier": "mid_vram_6gb",
            "tile_size": 384,
            "tile_pad": 16,
            "tile_batch_size": 1,
            "fp16": True,
            "scunet_fast_scale": 0.60,
            "scunet_fast_scale_1080p": 0.50,
            "scunet_full_frame_max_px": scunet_max_px if budget_mb >= 2600.0 else 0,
            "sr_full_frame_max_px": sr_max_px if budget_mb >= 2600.0 else 0,
            "sr_max_out_dim": 2560,
            "sequential_offload": True,
            "vram_headroom_mb": 768.0,
            "vram_budget_mb": round(budget_mb, 1),
        }
    else:
        # <= 4 GB GPUs
        return {
            "tier": "low_vram",
            "tile_size": 256,
            "tile_pad": 16,
            "tile_batch_size": 1,
            "fp16": True,
            "scunet_fast_scale": 0.50,
            "scunet_fast_scale_1080p": 0.50,
            "scunet_full_frame_max_px": 0,
            "sr_full_frame_max_px": 0,
            "sr_max_out_dim": 1920,
            "sequential_offload": True,
            "vram_headroom_mb": 512.0,
            "vram_budget_mb": round(budget_mb, 1),
        }


def detect_system_hardware(override_vram_gb: Optional[float] = None) -> Dict[str, Any]:
    """
    Probes actual GPU hardware, CUDA runtime, system RAM, and generates the active
    adaptive profile. Respects optional `override_vram_gb` or `DASHCLARITY_VRAM_GB_OVERRIDE`
    only for deterministic VRAM-tier simulation/testing.
    """
    cuda_avail = torch.cuda.is_available()
    env_override = os.environ.get("DASHCLARITY_VRAM_GB_OVERRIDE", "").strip()
    if override_vram_gb is None and env_override:
        try:
            override_vram_gb = float(env_override)
        except ValueError:
            override_vram_gb = None

    gpu_name = "CPU Fallback"
    total_vram_mb = 0.0
    free_vram_mb = 0.0
    total_vram_gb = 0.0
    compute_cap = None
    cuda_ver = None
    cudnn_ver = None

    if cuda_avail:
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        free_b, total_b = torch.cuda.mem_get_info(0)
        total_vram_mb = round(total_b / (1024 ** 2), 1)
        free_vram_mb = round(free_b / (1024 ** 2), 1)
        total_vram_gb = round(total_b / (1024 ** 3), 2)
        compute_cap = f"{props.major}.{props.minor}"
        cuda_ver = torch.version.cuda
        cudnn_ver = torch.backends.cudnn.version()

    effective_vram_gb = override_vram_gb if override_vram_gb is not None else total_vram_gb
    free_gb_for_profile = (
        round(free_vram_mb / 1024.0, 2)
        if (override_vram_gb is None and cuda_avail)
        else (round(override_vram_gb * 0.85, 2) if override_vram_gb is not None else None)
    )
    profile = select_gpu_profile(
        effective_vram_gb,
        cuda_available=cuda_avail,
        free_vram_gb=free_gb_for_profile,
    )
    ram = get_system_ram_stats()
    driver_ver = _probe_nvidia_driver() if cuda_avail else None

    return {
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "device_count": torch.cuda.device_count() if cuda_avail else 0,
        "compute_capability": compute_cap,
        "driver_version": driver_ver,
        "total_vram_mb": total_vram_mb,
        "total_vram_gb": total_vram_gb,
        "free_vram_mb": free_vram_mb,
        "effective_vram_gb_for_profile": effective_vram_gb,
        "pytorch_version": torch.__version__,
        "cuda_version": cuda_ver,
        "cudnn_version": cudnn_ver,
        "system_ram_total_gb": ram["total_gb"],
        "system_ram_available_gb": ram["available_gb"],
        "profile": profile,
    }


def get_vram_snapshot() -> Dict[str, float]:
    """Returns current PyTorch CUDA memory usage in MB."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0, "free_mb": 0.0}
    free_b, _ = torch.cuda.mem_get_info(0)
    return {
        "allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1),
        "reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 ** 2), 1),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 1),
        "free_mb": round(free_b / (1024 ** 2), 1),
    }


def safe_cuda_cleanup(collect_gc: bool = False):
    """Safely clears unused CUDA cached blocks without blocking unless required."""
    if collect_gc:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
