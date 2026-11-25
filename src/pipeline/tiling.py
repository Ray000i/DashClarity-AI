"""
DashClarity AI - Adaptive Tiled Tensor Inference Engine
Executes memory-bounded convolutional and generative inference on large image frames.
Adapts tile sizing and micro-batching dynamically for both RTX 4050 (6 GB) and RTX 4060 (8 GB)
with automatic CUDA OOM recovery and zero redundant overlap weight-map allocations.
"""

import math
import logging
import torch

logger = logging.getLogger("DashClarity.Pipeline.Tiling")


class GPUOutOfMemoryFallback(Exception):
    """Raised when GPU VRAM is exhausted even at minimum spatial tile size."""
    pass


def run_tiled_inference(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    tile_size: int = 384,
    tile_pad: int = 16,
    scale: int = 1,
    fp16: bool = True,
    batch_size: int = 1
) -> torch.Tensor:
    """
    Executes model inference in context-padded non-overlapping spatial tiles to cap peak VRAM.

    Args:
        model: PyTorch model in eval() mode.
        tensor: Input tensor of shape (B, C, H, W).
        tile_size: Base dimension for each square tile (adapted to GPU VRAM).
        tile_pad: Context border padding around each tile to prevent boundary seams.
        scale: Spatial upscaling factor (1 for denoising, 2 or 4 for SR).
        fp16: Whether to run half-precision (torch.float16) on CUDA.
        batch_size: Max tiles processed per batched forward pass (adapted to VRAM).

    Returns:
        Processed output tensor of shape (B, C_out, H*scale, W*scale).
    """
    b, c, h, w = tensor.shape
    device = tensor.device
    use_amp = fp16 and device.type == "cuda"

    # For small images that fit within a single tile, run directly with OOM fallback
    if max(h, w) <= tile_size:
        try:
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_amp):
                out = model(tensor)
                return out[0] if isinstance(out, tuple) else out
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM on single-pass frame (%dx%d). Switching to tiled execution.", w, h)
            torch.cuda.empty_cache()

    current_tile_size = tile_size
    current_batch = max(1, batch_size)
    min_tile_size = 160

    while current_tile_size >= min_tile_size:
        try:
            return _tiled_forward(
                model=model,
                tensor=tensor,
                tile_size=current_tile_size,
                tile_pad=tile_pad,
                scale=scale,
                use_amp=use_amp,
                batch_size=current_batch
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current_batch > 1:
                # First try reducing batch size to 1 before shrinking spatial tile size
                logger.warning("CUDA OOM with tile_batch=%d, tile_size=%d. Reducing tile_batch to 1...", current_batch, current_tile_size)
                current_batch = 1
            else:
                new_tile = max(128, (int(current_tile_size * 0.75) // 32) * 32)
                if new_tile >= current_tile_size:
                    new_tile = current_tile_size - 64
                logger.warning(
                    "CUDA OOM with tile size %d. Auto-reducing tile size to %d...",
                    current_tile_size, new_tile
                )
                current_tile_size = new_tile

    logger.warning("Exhausted GPU tile reductions. Raising GPUOutOfMemoryFallback for preset adaptation.")
    torch.cuda.empty_cache()
    raise GPUOutOfMemoryFallback("GPU VRAM exhausted across all spatial tile sizes.")


def _tiled_forward(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    tile_size: int,
    tile_pad: int,
    scale: int,
    use_amp: bool,
    batch_size: int = 1
) -> torch.Tensor:
    b, c, h, w = tensor.shape
    device = tensor.device

    tiles_x = math.ceil(w / tile_size)
    tiles_y = math.ceil(h / tile_size)

    out_h = h * scale
    out_w = w * scale
    out_dtype = torch.float16 if (use_amp or tensor.dtype == torch.float16) else tensor.dtype
    output = torch.empty((b, c, out_h, out_w), device=device, dtype=out_dtype)

    # Collect tile metadata; group identical padded shapes for batched execution when batch_size > 1
    tile_specs = []
    for yi in range(tiles_y):
        for xi in range(tiles_x):
            ofs_x = xi * tile_size
            ofs_y = yi * tile_size
            sz_x = min(tile_size, w - ofs_x)
            sz_y = min(tile_size, h - ofs_y)

            in_x1 = max(ofs_x - tile_pad, 0)
            in_y1 = max(ofs_y - tile_pad, 0)
            in_x2 = min(ofs_x + sz_x + tile_pad, w)
            in_y2 = min(ofs_y + sz_y + tile_pad, h)

            pad_left = ofs_x - in_x1
            pad_top = ofs_y - in_y1
            pad_right = in_x2 - (ofs_x + sz_x)
            pad_bottom = in_y2 - (ofs_y + sz_y)

            tile_specs.append((
                ofs_x, ofs_y, sz_x, sz_y,
                in_x1, in_y1, in_x2, in_y2,
                pad_left, pad_top, pad_right, pad_bottom
            ))

    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_amp):
        if batch_size > 1 and b == 1 and len(tile_specs) > 1:
            # Group by (in_h, in_w) to allow clean tensor stacking without dynamic shape padding bugs
            groups = {}
            for spec in tile_specs:
                shape_key = (spec[7] - spec[5], spec[6] - spec[4])
                groups.setdefault(shape_key, []).append(spec)

            for shape_key, group_specs in groups.items():
                for start in range(0, len(group_specs), batch_size):
                    chunk = group_specs[start:start + batch_size]
                    batch_in = torch.cat(
                        [tensor[:, :, s[5]:s[7], s[4]:s[6]] for s in chunk],
                        dim=0
                    )
                    batch_out = model(batch_in)
                    if isinstance(batch_out, tuple):
                        batch_out = batch_out[0]

                    for idx, s in enumerate(chunk):
                        ofs_x, ofs_y, sz_x, sz_y, _, _, _, _, pad_left, pad_top, pad_right, pad_bottom = s
                        tile_out = batch_out[idx:idx + 1]
                        crop_top = pad_top * scale
                        crop_bottom = (tile_out.shape[2] - pad_bottom * scale) if pad_bottom > 0 else tile_out.shape[2]
                        crop_left = pad_left * scale
                        crop_right = (tile_out.shape[3] - pad_right * scale) if pad_right > 0 else tile_out.shape[3]

                        valid_tile = tile_out[:, :, crop_top:crop_bottom, crop_left:crop_right]
                        out_x1 = ofs_x * scale
                        out_y1 = ofs_y * scale
                        output[:, :, out_y1:out_y1 + valid_tile.shape[2], out_x1:out_x1 + valid_tile.shape[3]] = valid_tile.to(out_dtype)
        else:
            for s in tile_specs:
                ofs_x, ofs_y, sz_x, sz_y, in_x1, in_y1, in_x2, in_y2, pad_left, pad_top, pad_right, pad_bottom = s
                input_tile = tensor[:, :, in_y1:in_y2, in_x1:in_x2]
                tile_out = model(input_tile)
                if isinstance(tile_out, tuple):
                    tile_out = tile_out[0]

                crop_top = pad_top * scale
                crop_bottom = (tile_out.shape[2] - pad_bottom * scale) if pad_bottom > 0 else tile_out.shape[2]
                crop_left = pad_left * scale
                crop_right = (tile_out.shape[3] - pad_right * scale) if pad_right > 0 else tile_out.shape[3]

                valid_tile = tile_out[:, :, crop_top:crop_bottom, crop_left:crop_right]
                out_x1 = ofs_x * scale
                out_y1 = ofs_y * scale
                output[:, :, out_y1:out_y1 + valid_tile.shape[2], out_x1:out_x1 + valid_tile.shape[3]] = valid_tile.to(out_dtype)

    return output
