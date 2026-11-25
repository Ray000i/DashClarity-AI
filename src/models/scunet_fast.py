"""
DashClarity AI - Faster window attention for the pretrained SCUNet denoiser (inference only).

Profiling SCUNet at 1280x736 (fp16, RTX 4050) showed the reference window-attention code
spends most of its time outside the attention maths itself:

  * `sim + relative_embedding` is a broadcast add over a (heads, b, windows, 64, 64) tensor
    executed by a slow non-vectorised kernel (~26% of total GPU time), and
  * the shifted-window mask is rebuilt from scratch on every call as a
    (h_windows x w_windows x 64 x 64) boolean tensor, then rearranged.

This module replaces `WMSA.forward` on an already-loaded model with a mathematically
equivalent implementation that

  1. calls `torch.nn.functional.scaled_dot_product_attention` (fused kernel) with the
     relative-position bias passed as the additive attention bias,
  2. caches that bias once per module / dtype (the weights are frozen at inference), and
  3. exploits the structure of the SW-MSA mask: only windows in the last window-row and
     last window-column are masked, and within each of those three groups (last row,
     last column, corner) the mask is identical. All windows are computed with the plain
     bias, then the few border windows are recomputed with their group mask.

The checkpoint and its weights are untouched; `verify_equivalence` compares the patched and
the reference forward on the same input.
"""
from __future__ import annotations

import logging
import types
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger("DashClarity.Models.SCUNetFast")


def _bias_cache(self, dtype: torch.dtype, device: torch.device) -> Dict[str, torch.Tensor]:
    key = (dtype, device)
    cache = getattr(self, "_dc_bias_cache", None)
    if cache is not None and cache.get("key") == key:
        return cache
    ws = self.window_size
    n = ws * ws
    rel = self.relative_embedding().to(device=device, dtype=dtype)          # (heads, n, n)
    neg = torch.finfo(dtype).min if dtype != torch.float32 else float("-inf")
    cache = {"key": key, "rel": rel}
    if self.type != "W":
        s = ws - ws // 2
        idx = torch.arange(n, device=device)
        r, c = idx // ws, idx % ws                                               # pixel row / col inside a window
        # same rules as WMSA.generate_mask: queries and keys on different sides of the
        # wrap-around seam (row index < s vs >= s, and/or column index < s vs >= s) are masked
        row_cut = (r[:, None] < s) != (r[None, :] < s)
        col_cut = (c[:, None] < s) != (c[None, :] < s)
        for name, m in (("row", row_cut), ("col", col_cut), ("both", row_cut | col_cut)):
            b = rel.clone()
            b.masked_fill_(m[None], neg)
            cache[name] = b
    self._dc_bias_cache = cache
    return cache


def _sdpa(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """x: (B, n, C) tokens of B windows -> (B, n, C_out)."""
    bsz, n, _ = x.shape
    qkv = self.embedding_layer(x)                                                # (B, n, 3*heads*hd)
    qkv = qkv.view(bsz, n, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]                                             # (B, heads, n, hd)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale)
    out = out.transpose(1, 2).reshape(bsz, n, self.n_heads * self.head_dim)
    return self.linear(out)


def _fast_wmsa_forward(self, x: torch.Tensor) -> torch.Tensor:
    ws = self.window_size
    shift = ws // 2
    if self.type != "W":
        x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))
    b, H, W, C = x.shape
    h1, w1 = H // ws, W // ws
    n = ws * ws
    win = x.view(b, h1, ws, w1, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(b, h1, w1, n, C)
    cache = _bias_cache(self, x.dtype, x.device)

    out = _sdpa(self, win.reshape(b * h1 * w1, n, C), cache["rel"]).view(b, h1, w1, n, -1)
    if self.type != "W":
        # border windows: recompute with the group mask and overwrite
        if w1 > 1:
            out[:, h1 - 1, : w1 - 1] = _sdpa(self, win[:, h1 - 1, : w1 - 1].reshape(-1, n, C), cache["row"]).view(b, w1 - 1, n, -1)
        if h1 > 1:
            out[:, : h1 - 1, w1 - 1] = _sdpa(self, win[:, : h1 - 1, w1 - 1].reshape(-1, n, C), cache["col"]).view(b, h1 - 1, n, -1)
        out[:, h1 - 1, w1 - 1] = _sdpa(self, win[:, h1 - 1, w1 - 1].reshape(-1, n, C), cache["both"]).view(b, n, -1)

    c_out = out.shape[-1]
    out = out.view(b, h1, w1, ws, ws, c_out).permute(0, 1, 3, 2, 4, 5).reshape(b, H, W, c_out)
    if self.type != "W":
        out = torch.roll(out, shifts=(shift, shift), dims=(1, 2))
    return out


def _fast_convtrans_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Same computation as ConvTransBlock.forward, but the transformer branch gets a
    contiguous channels-last tensor. The reference passes a permuted view of a channel
    slice, so every LayerNorm / residual add / GELU inside the block ran on strided memory
    (non-vectorised kernels, ~40% of SCUNet's GPU time)."""
    conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
    conv_x = self.conv_block(conv_x) + conv_x
    trans_x = self.trans_block(trans_x.permute(0, 2, 3, 1).contiguous())
    trans_x = trans_x.permute(0, 3, 1, 2)
    res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
    return x + res


def accelerate_scunet(model: torch.nn.Module) -> int:
    """Patch WMSA and ConvTransBlock modules inside a loaded SCUNet (spandrel ModelDescriptor
    or nn.Module). Returns the number of patched modules."""
    net = getattr(model, "model", model)
    count = 0
    for mod in net.modules():
        name = type(mod).__name__
        if name == "WMSA" and hasattr(mod, "relative_embedding"):
            fast = _fast_wmsa_forward
        elif name == "ConvTransBlock" and hasattr(mod, "trans_block"):
            fast = _fast_convtrans_forward
        else:
            continue
        if not hasattr(mod, "_dc_reference_forward"):
            mod._dc_reference_forward = mod.forward
        mod.forward = types.MethodType(fast, mod)
        count += 1
    return count


def restore_reference(model: torch.nn.Module) -> None:
    net = getattr(model, "model", model)
    for mod in net.modules():
        if hasattr(mod, "_dc_reference_forward"):
            mod.forward = mod._dc_reference_forward
            del mod._dc_reference_forward


@torch.inference_mode()
def verify_equivalence(model: torch.nn.Module, shape: Tuple[int, int] = (200, 328), dtype=torch.float32) -> float:
    """Max |patched - reference| on a random image (use fp32 for a strict check)."""
    net = getattr(model, "model", model)
    p = next(net.parameters())
    x = torch.rand(1, 3, *shape, device=p.device, dtype=dtype)
    fast = net(x).float()
    restore_reference(model)
    ref = net(x).float()
    accelerate_scunet(model)
    return float((fast - ref).abs().max())
