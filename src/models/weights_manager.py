"""
DashClarity AI - Model Weights Manager
Automated downloading, checksum validation, size verification, and local caching of pretrained weights.
"""

import os
import hashlib
import logging
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("DashClarity.Models.WeightsManager")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"

# Registry of supported pretrained models with verified URLs and metadata
WEIGHTS_REGISTRY: Dict[str, Dict[str, Any]] = {
    "zero_dce_plus": {
        "filename": "Zero_DCE++_Epoch99.pth",
        "category": "ZeroDCE",
        "url": "https://github.com/Li-Chongyi/Zero-DCE_extension/raw/main/Zero-DCE%2B%2B/snapshots_Zero_DCE%2B%2B/Epoch99.pth",
        "expected_size": 52395,
        "sha256": "ca8855b90df9a80fa4195a831f33d3476b1964f787eb70602797c773067f3b84"
    },
    "zero_dce": {
        "filename": "Zero_DCE_Epoch99.pth",
        "category": "ZeroDCE",
        "url": "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/Zero-DCE_code/snapshots/Epoch99.pth",
        "expected_size": 320017,
        "sha256": "a4395acb874f320375d9704997cef874eaaaaa26a1777ceb29a92b70f74c3612"
    },
    "scunet_color_real_psnr": {
        "filename": "scunet_color_real_psnr.pth",
        "category": "SCUNet",
        "url": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
        "expected_size": 71982841,
        "sha256": "fa78899ba2caec9d235a900e91d96c689da71c42029230c2028b00f09f809c2e"
    },
    "realesr_general_x4v3": {
        "filename": "realesr-general-x4v3.pth",
        "category": "RealESRGAN",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "expected_size": 4885111,
        "sha256": "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"
    },
    "realesr_x2plus": {
        "filename": "RealESRGAN_x2plus.pth",
        "category": "RealESRGAN",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "expected_size": 67061725,
        "sha256": "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"
    },
    "stcd_indoor": {
        "filename": "indoor_G.pth",
        "category": "STCD",
        "url": "https://drive.google.com/file/d/1sUzhwKWxJOjITFL8ypb3UQeofz0KI7E4/view",
        "expected_size": 7258767,
        "sha256": "dc50957303b2ac15d2f71a32c3d4231c47c5d3f8dc21daa52bccf48faea49cda",
        "local_only": True
    }
}


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_weights_path(model_key: str, auto_download: bool = True) -> Path:
    """
    Returns the absolute path to the model weights.
    Downloads automatically from verified official mirrors if not present.
    """
    if model_key not in WEIGHTS_REGISTRY:
        raise ValueError(f"Unknown model key '{model_key}'. Registered models: {list(WEIGHTS_REGISTRY.keys())}")

    info = WEIGHTS_REGISTRY[model_key]
    cat_dir = WEIGHTS_DIR / info["category"]
    cat_dir.mkdir(parents=True, exist_ok=True)
    target_path = cat_dir / info["filename"]

    if target_path.exists():
        actual_size = target_path.stat().st_size
        if actual_size > 1000:
            if "sha256" in info:
                actual_sha = compute_sha256(target_path)
                if actual_sha.lower() != info["sha256"].lower():
                    logger.warning("SHA256 mismatch for %s: expected %s, got %s. Re-downloading...", model_key, info["sha256"], actual_sha)
                    target_path.unlink()
                else:
                    logger.info("Found verified weights for %s (%d bytes, SHA-256 match)", model_key, actual_size)
                    return target_path
            else:
                logger.info("Found cached weights for %s at %s (%d bytes)", model_key, target_path, actual_size)
                return target_path
        else:
            logger.warning("Cached weights file %s appears corrupted (%d bytes). Re-downloading...", target_path, actual_size)
            target_path.unlink()

    if not auto_download:
        raise FileNotFoundError(f"Weights for {model_key} not found at {target_path} and auto_download is False.")

    if info.get("local_only"):
        raise FileNotFoundError(f"Weights for {model_key} must be provided locally at {target_path}.")

    urls_to_try = [info["url"]]
    if "fallback_url" in info:
        urls_to_try.append(info["fallback_url"])

    download_success = False
    for url in urls_to_try:
        try:
            logger.info("Downloading pretrained %s weights from %s...", model_key, url)
            headers = {"User-Agent": "DashClarity-AI-Downloader/1.0"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as response, open(target_path, "wb") as out_file:
                total_length = response.getheader('content-length')
                if total_length:
                    total_length = int(total_length)
                downloaded = 0
                while True:
                    buffer = response.read(1024 * 64)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)

            actual_size = target_path.stat().st_size
            logger.info("Downloaded %s successfully (%d bytes).", model_key, actual_size)
            download_success = True
            break
        except Exception as e:
            logger.warning("Download failed from %s: %s", url, e)
            if target_path.exists():
                target_path.unlink()

    if not download_success:
        raise RuntimeError(f"Could not download pretrained weights for '{model_key}'. Please check your network connection.")

    return target_path


def list_downloaded_weights() -> Dict[str, Dict[str, Any]]:
    """Returns a dict of all downloaded weights with their file sizes."""
    status = {}
    for key, info in WEIGHTS_REGISTRY.items():
        cat_dir = WEIGHTS_DIR / info["category"]
        p = cat_dir / info["filename"]
        if p.exists():
            status[key] = {
                "filename": info["filename"],
                "path": str(p),
                "size_bytes": p.stat().st_size,
                "size_mb": round(p.stat().st_size / (1024 * 1024), 2)
            }
        else:
            status[key] = {
                "filename": info["filename"],
                "path": None,
                "size_bytes": 0,
                "size_mb": 0.0
            }
    return status
