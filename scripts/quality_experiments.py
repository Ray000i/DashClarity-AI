"""
DashClarity AI - Controlled quality experiments for the low-light stage.

Each variant changes one LowLightStage setting; everything else (inputs, degradation seed,
preset, frame count) is identical. Results: results/gpu_benchmark/quality_experiments.json

    python scripts/quality_experiments.py --preset balanced
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
warnings.filterwarnings("ignore")

from scripts.benchmark_gpu import (CLEAN, REAL_720P, flicker, lpips_score, luma, mean_of, no_ref,  # noqa: E402
                                   psnr, read_frames, run_clip, ssim, temporal_error)
from scripts.degrade import apply_dashcam_degradation  # noqa: E402
from src.pipeline.enhancer import VideoEnhancementPipeline  # noqa: E402
from src.utils.metrics import compute_flat_region_noise  # noqa: E402

VARIANTS = {
    "V0_original_steps":       dict(smooth_exposure=False, curve_downscale=1.0, color_balance=1.0, clahe_clip=0.0),
    "V1_smooth_exposure":      dict(smooth_exposure=True,  curve_downscale=1.0, color_balance=1.0, clahe_clip=0.0),
    "V2_curves_at_1/12":       dict(smooth_exposure=True,  curve_downscale=1 / 12, color_balance=1.0, clahe_clip=0.0),
    "V3_curves_at_1/4":        dict(smooth_exposure=True,  curve_downscale=0.25, color_balance=1.0, clahe_clip=0.0),
    "V4_colorbalance_0.5":     dict(smooth_exposure=True,  curve_downscale=1.0, color_balance=0.5, clahe_clip=0.0),
    "V5_colorbalance_off":     dict(smooth_exposure=True,  curve_downscale=1.0, color_balance=0.0, clahe_clip=0.0),
    "V6_clahe_1.5":            dict(smooth_exposure=True,  curve_downscale=1.0, color_balance=1.0, clahe_clip=1.5),
    "V7_clahe_2.5":            dict(smooth_exposure=True,  curve_downscale=1.0, color_balance=1.0, clahe_clip=2.5),
    "V8_adopted_defaults":     dict(smooth_exposure=True,  curve_downscale=0.25, color_balance=0.5, clahe_clip=0.0),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="balanced")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--real-frames", type=int, default=40)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()

    pipe = VideoEnhancementPipeline(preset=args.preset)
    clean_sets, low_sets = [], []
    for clip in CLEAN:
        clean = read_frames(clip, args.frames)
        np.random.seed(1234)
        clean_sets.append(clean)
        low_sets.append([apply_dashcam_degradation(c) for c in clean])
    real = read_frames(REAL_720P, args.real_frames)
    out_dir = BASE_DIR / "results" / "gpu_benchmark" / "quality_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in args.variants:
        cfg = VARIANTS[name]
        pipe.enhance_frame(real[0], real[0], preset=args.preset)  # make sure the stage exists
        for k, v in cfg.items():
            setattr(pipe.low_light_stage, k, v)
        rows = []
        for clean, low in zip(clean_sets, low_sets):
            outs, _ = run_clip(pipe, args.preset, low, restore=True)
            idx = list(range(0, len(outs), max(1, len(outs) // 5)))[:5]
            rows.append({
                "psnr": float(np.mean([psnr(clean[i], outs[i]) for i in idx])),
                "ssim": float(np.mean([ssim(clean[i], outs[i]) for i in idx])),
                "lpips": mean_of([{"v": lpips_score(clean[i], outs[i])} for i in idx], "v"),
                "temporal_error": temporal_error(outs, clean),
            })
            if clean is clean_sets[0]:
                cv2.imwrite(str(out_dir / f"synthetic_{name.replace('/', '-')}.png"), outs[len(outs) // 2])
        outs, stats = run_clip(pipe, args.preset, real)
        sel = outs[:: max(1, len(outs) // 6)][:6]
        res = {k: mean_of(rows, k) for k in ("psnr", "ssim", "lpips", "temporal_error")}
        res.update({
            "real_niqe": mean_of([{"v": no_ref("niqe", o)} for o in sel], "v"),
            "real_brisque": mean_of([{"v": no_ref("brisque", o)} for o in sel], "v"),
            "real_flat_noise": mean_of([{"v": compute_flat_region_noise(o)} for o in sel], "v"),
            "real_rms_contrast": round(float(np.mean([luma(o).std() for o in sel])), 2),
            "real_mean_luma": round(float(np.mean([luma(o).mean() for o in sel])), 2),
            "real_flicker": round(flicker(outs), 4),
            "real_lowlight_ms": stats["stage_ms_mean"].get("lowlight_zerodce"),
        })
        results[name] = res
        cv2.imwrite(str(out_dir / f"real_{name.replace('/', '-')}.png"), outs[len(outs) // 2])
        print(f"{name:<22} " + "  ".join(f"{k}={v}" for k, v in res.items()), flush=True)

    sel = real[:: max(1, len(real) // 6)][:6]
    results["input"] = {
        "real_niqe": mean_of([{"v": no_ref("niqe", o)} for o in sel], "v"),
        "real_brisque": mean_of([{"v": no_ref("brisque", o)} for o in sel], "v"),
        "real_flat_noise": mean_of([{"v": compute_flat_region_noise(o)} for o in sel], "v"),
        "real_rms_contrast": round(float(np.mean([luma(o).std() for o in sel])), 2),
        "real_mean_luma": round(float(np.mean([luma(o).mean() for o in sel])), 2),
        "real_flicker": round(flicker(real), 4),
    }
    print("input", results["input"])
    tag = "" if len(args.variants) == len(VARIANTS) else "_" + "_".join(v.split("_")[0] for v in args.variants)
    path = BASE_DIR / "results" / "gpu_benchmark" / f"quality_experiments_{args.preset}{tag}.json"
    path.write_text(json.dumps(results, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
