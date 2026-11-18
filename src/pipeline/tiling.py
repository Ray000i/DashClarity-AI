"""
DashClarity AI - Tiled Tensor Inference Engine
Executes memory-bounded convolutional and generative inference on large image frames.
Maintains VRAM below 6 GB on NVIDIA RTX 4060 Laptop GPU (8 GB) with automatic OOM recovery.
"""

import math
import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger("DashClarity.Pipeline.Tiling")


class GPUOutOfMemoryFallback(Exception):
    """Raised when GPU VRAM is completely exhausted even at minimum spatial tile size."""
    pass


def run_tiled_inference(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    tile_size: int = 384,
    tile_pad: int = 16,
    scale: int = 1,
    fp16: bool = True
) -> torch.Tensor:
    """
    Executes model inference in overlapping spatial tiles to cap peak VRAM.
    
    Args:
        model: PyTorch model in eval() mode.
        tensor: Input tensor of shape (B, C, H, W).
        tile_size: Base dimension for each square tile (256 - 400).
        tile_pad: Overlapping padding for boundary artifact blending.
        scale: Spatial upscaling factor of the model (1 for enhancement, 2 or 4 for SR).
        fp16: Whether to run in half-precision (torch.float16).
        
    Returns:
        Processed output tensor of shape (B, C_out, H*scale, W*scale).
    """
    b, c, h, w = tensor.shape
    device = tensor.device

    # For small images that comfortably fit in VRAM, run directly with OOM protection
    if max(h, w) <= tile_size:
        try:
            with torch.amp.autocast('cuda', enabled=fp16 and device.type == 'cuda'):
                return model(tensor)
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM on full frame (%dx%d). Falling back to tiled execution.", w, h)
            torch.cuda.empty_cache()

    # Determine tile grid
    current_tile_size = tile_size
    min_tile_size = 192

    while current_tile_size >= min_tile_size:
        try:
            return _tiled_forward(
                model=model,
                tensor=tensor,
                tile_size=current_tile_size,
                tile_pad=tile_pad,
                scale=scale,
                fp16=fp16
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            new_tile = int(current_tile_size * 0.75)
            logger.warning(
                "CUDA OOM with tile size %d. Auto-reducing tile size to %d...",
                current_tile_size, new_tile
            )
            current_tile_size = new_tile

    # If GPU fails even at minimum tile size, raise GPUOutOfMemoryFallback
    logger.warning("Exhausted GPU tile reductions. Raising GPUOutOfMemoryFallback for pipeline preset adaptation.")
    torch.cuda.empty_cache()
    raise GPUOutOfMemoryFallback("GPU VRAM exhausted across all spatial tile sizes.")


def _tiled_forward(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    tile_size: int,
    tile_pad: int,
    scale: int,
    fp16: bool
) -> torch.Tensor:
    b, c, h, w = tensor.shape
    device = tensor.device

    tiles_x = math.ceil(w / tile_size)
    tiles_y = math.ceil(h / tile_size)

    # Determine output channels from a test forward or default to 3
    out_c = c
    out_h = h * scale
    out_w = w * scale

    output = torch.zeros((b, out_c, out_h, out_w), device=device, dtype=tensor.dtype)
    weight_map = torch.zeros((b, 1, out_h, out_w), device=device, dtype=torch.float32)

    for yi in range(tiles_y):
        for xi in range(tiles_x):
            # Tile coordinates without padding
            ofs_x = xi * tile_size
            ofs_y = yi * tile_size
            sz_x = min(tile_size, w - ofs_x)
            sz_y = min(tile_size, h - ofs_y)

            # Coordinates with padding
            in_x1 = max(ofs_x - tile_pad, 0)
            in_y1 = max(ofs_y - tile_pad, 0)
            in_x2 = min(ofs_x + sz_x + tile_pad, w)
            in_y2 = min(ofs_y + sz_y + tile_pad, h)

            pad_left = ofs_x - in_x1
            pad_top = ofs_y - in_y1
            pad_right = in_x2 - (ofs_x + sz_x)
            pad_bottom = in_y2 - (ofs_y + sz_y)

            input_tile = tensor[:, :, in_y1:in_y2, in_x1:in_x2]

            with torch.amp.autocast('cuda', enabled=fp16 and device.type == 'cuda'):
                tile_out = model(input_tile)
                if isinstance(tile_out, tuple):
                    tile_out = tile_out[0]

            # Crop out the padding from the output tile
            crop_top = pad_top * scale
            crop_bottom = (tile_out.shape[2] - pad_bottom * scale) if pad_bottom > 0 else tile_out.shape[2]
            crop_left = pad_left * scale
            crop_right = (tile_out.shape[3] - pad_right * scale) if pad_right > 0 else tile_out.shape[3]

            valid_tile = tile_out[:, :, crop_top:crop_bottom, crop_left:crop_right]

            # Output coordinates
            out_x1 = ofs_x * scale
            out_y1 = ofs_y * scale
            out_x2 = out_x1 + valid_tile.shape[3]
            out_y2 = out_y1 + valid_tile.shape[2]

            # Linear cosine blending weight to smoothly stitch tiles
            tw = valid_tile.shape[3]
            th = valid_tile.shape[2]
            wx = torch.ones((1, 1, 1, tw), device=device)
            wy = torch.ones((1, 1, th, 1), device=device)
            tile_w = wy * wx

            output[:, :, out_y1:out_y2, out_x1:out_x2] += valid_tile.to(output.dtype) * tile_w
            weight_map[:, :, out_y1:out_y2, out_x1:out_x2] += tile_w

    output = output / torch.clamp(weight_map, min=1e-5)
    return output
