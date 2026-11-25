"""
DashClarity AI - Reproducible GPU benchmark (speed, memory and quality).

Runs real input clips through the enhancement presets and records measured values only:

  * speed:   per-stage latency (CUDA-synchronised), frame latency, FPS
  * memory:  PyTorch peak allocated / reserved VRAM, device-wide VRAM from nvidia-smi
             (includes other processes; the idle baseline is recorded separately), process RAM
  * quality: synthetic pairs (clean daytime clip -> scripts/degrade.py night degradation -> enhance)
             scored against the clean frames with PSNR, SSIM, LPIPS (if the `lpips` package is
             installed) and a temporal-consistency error; real night clips (no ground truth) scored
             with no-reference NIQE / BRISQUE (if `pyiqa` is installed), flat-region noise,
             mean luminance and frame-to-frame flicker.

Usage:
    python scripts/benchmark_gpu.py --tag baseline
    python scripts/benchmark_gpu.py --tag improved --presets fast balanced quality

Results: results/gpu_benchmark/<tag>.json and sample frames in results/gpu_benchmark/<tag>/
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("TEST_MODE", "1")  # keep benchmark files out of the app's history

from scripts.degrade import apply_dashcam_degradation  # noqa: E402
from src.pipeline.enhancer import VideoEnhancementPipeline  # noqa: E402
from src.utils.metrics import compute_flat_region_noise  # noqa: E402

OUT_DIR = BASE_DIR / "results" / "gpu_benchmark"
REAL_720P = BASE_DIR / "input_videos" / "real" / "real_night_dashcam_01.mp4"
REAL_1080P = next(iter(sorted((BASE_DIR / "input_videos").glob("*1920_1080*.mp4"))), None)
CLEAN = sorted((BASE_DIR / "input_videos" / "clean_sources").glob("*.mp4"))[:3]
STAGE_ATTRS = {
    "neural_denoise_stage": "denoise_scunet",
    "classical_denoise_stage": "denoise_bilateral",
    "low_light_stage": "lowlight_zerodce",
    "stcd_stage": "lowlight_stcd",
    "super_res_stage": "super_resolution",
    "temporal_stage": "temporal_farneback",
}


# ------------------------------------------------------------------ helpers
def read_frames(path: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def nvidia_smi_used_mb() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        return float(out[0])
    except Exception:
        return None


class Sampler:
    """Samples device VRAM (nvidia-smi) and process RSS in the background."""

    def __init__(self, period: float = 0.25):
        self.period, self.stop_ev = period, threading.Event()
        self.peak_smi, self.peak_rss = 0.0, 0.0
        try:
            import psutil
            self.proc = psutil.Process()
        except Exception:
            self.proc = None

    def _run(self) -> None:
        while not self.stop_ev.is_set():
            v = nvidia_smi_used_mb()
            if v is not None:
                self.peak_smi = max(self.peak_smi, v)
            if self.proc is not None:
                self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss / 2 ** 20)
            time.sleep(self.period)

    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *exc):
        self.stop_ev.set()
        self.t.join(timeout=5)


TIMINGS: dict = defaultdict(list)  # shared by all wrapped stages; cleared before each measured run


def instrument(pipeline: VideoEnhancementPipeline, timings: dict = TIMINGS) -> None:
    """Wrap every stage's process_frame with a CUDA-synchronised timer (idempotent)."""
    for attr, label in STAGE_ATTRS.items():
        stage = getattr(pipeline, attr, None)
        if stage is None or getattr(stage.process_frame, "_timed", False):
            continue
        orig = stage.process_frame

        def timed(*a, _orig=orig, _label=label, **k):
            sync()
            t0 = time.perf_counter()
            out = _orig(*a, **k)
            sync()
            TIMINGS[_label].append((time.perf_counter() - t0) * 1000.0)
            return out

        timed._timed = True
        stage.process_frame = timed


def luma(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(luma(a), luma(b), data_range=255.0))


_LPIPS = None


def lpips_score(a: np.ndarray, b: np.ndarray) -> float | None:
    global _LPIPS
    try:
        import lpips
    except ImportError:
        return None
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _LPIPS is None:
        _LPIPS = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()

    def t(x):
        return torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1

    with torch.inference_mode():
        return float(_LPIPS(t(a), t(b)).item())


_NR = {}


def no_ref(name: str, img: np.ndarray) -> float | None:
    try:
        import pyiqa
    except ImportError:
        return None
    try:
        if name not in _NR:
            _NR[name] = pyiqa.create_metric(name, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 255.0
        with torch.inference_mode():
            return float(_NR[name](x).item())
    except Exception as exc:
        print(f"  [warn] {name} failed: {exc}")
        return None


def temporal_error(outs: list[np.ndarray], refs: list[np.ndarray]) -> float:
    """Mean |(o_t - o_t-1) - (r_t - r_t-1)| on luma: flicker relative to the true scene motion."""
    errs = [np.mean(np.abs((luma(outs[i]) - luma(outs[i - 1])) - (luma(refs[i]) - luma(refs[i - 1]))))
            for i in range(1, len(outs))]
    return float(np.mean(errs)) if errs else 0.0


def flicker(frames: list[np.ndarray]) -> float:
    """Mean absolute change of global mean luminance between consecutive frames."""
    m = [float(luma(f).mean()) for f in frames]
    return float(np.mean(np.abs(np.diff(m)))) if len(m) > 1 else 0.0


def mean_of(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(float(np.mean(vals)), 4) if vals else None


# ------------------------------------------------------------------ runs
def run_clip(pipeline, preset: str, frames: list[np.ndarray], restore: bool | None = None):
    timings = TIMINGS
    instrument(pipeline)
    pipeline.reset_temporal()
    outs, lat = [], []
    prev = None
    # warm-up frame (cudnn autotune, lazy allocations) is excluded from timing
    pipeline.enhance_frame(frames[0], frames[0], preset=preset, restore_at_original_res=restore)
    pipeline.reset_temporal()
    instrument(pipeline)  # stages loaded lazily by the warm-up get wrapped too
    timings.clear()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for f in frames:
        sync()
        t0 = time.perf_counter()
        out, tel = pipeline.enhance_frame(f, prev if prev is not None else f, preset=preset, restore_at_original_res=restore)
        sync()
        lat.append((time.perf_counter() - t0) * 1000.0)
        prev = f
        outs.append(out)
        if "gpu_oom_fallback" in tel:
            print(f"  [OOM FALLBACK] {tel['gpu_oom_fallback']}")
    stats = {
        "frames": len(frames),
        "latency_ms_mean": round(float(np.mean(lat)), 2),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
        "fps": round(1000.0 / float(np.mean(lat)), 3),
        "stage_ms_mean": {k: round(float(np.mean(v)), 2) for k, v in timings.items() if v},
        "output_shape": list(outs[0].shape),
    }
    if torch.cuda.is_available():
        stats["torch_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        stats["torch_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)
    return outs, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--presets", nargs="+", default=["fast", "balanced", "quality"])
    ap.add_argument("--frames-720p", type=int, default=60)
    ap.add_argument("--frames-1080p", type=int, default=30)
    ap.add_argument("--quality-frames", type=int, default=20)
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--skip-1080p", action="store_true")
    args = ap.parse_args()

    out_dir = OUT_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"tag": args.tag, "time": time.strftime("%Y-%m-%d %H:%M:%S")}

    hw = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        hw.update(gpu=p.name, vram_total_mb=round(p.total_memory / 2 ** 20), compute_capability=f"{p.major}.{p.minor}",
                  cudnn=torch.backends.cudnn.version())
    hw["nvidia_smi_idle_used_mb"] = nvidia_smi_used_mb()
    report["hardware"] = hw
    print(json.dumps(hw, indent=2))

    pipeline = VideoEnhancementPipeline(preset=args.presets[0])
    report["pipeline"] = {"device": str(pipeline.device), "fp16": pipeline.fp16, "tile_size": pipeline.tile_size}
    if hasattr(pipeline, "profile"):
        report["pipeline"]["profile"] = pipeline.profile

    # ---------------- speed + memory on real footage
    clips = [("720p_real_night", REAL_720P, args.frames_720p)]
    if REAL_1080P is not None and not args.skip_1080p:
        clips.append(("1080p_real", REAL_1080P, args.frames_1080p))
    report["speed"] = {}
    for clip_name, path, n in clips:
        frames = read_frames(path, n)
        print(f"\n== {clip_name}: {path.name} ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]})")
        for preset in args.presets:
            n_use = frames if not (preset in ("quality", "full") and "1080p" in clip_name) else frames[: max(5, n // 3)]
            with Sampler() as s:
                try:
                    outs, stats = run_clip(pipeline, preset, n_use)
                except Exception as exc:  # report and continue
                    print(f"  {preset:<10} FAILED: {type(exc).__name__}: {exc}")
                    report["speed"][f"{clip_name}/{preset}"] = {"error": f"{type(exc).__name__}: {exc}"}
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
            stats["nvidia_smi_peak_used_mb"] = s.peak_smi or None
            stats["process_peak_rss_mb"] = round(s.peak_rss, 1) if s.peak_rss else None
            report["speed"][f"{clip_name}/{preset}"] = stats
            print(f"  {preset:<10} {stats['fps']:7.2f} FPS  {stats['latency_ms_mean']:8.1f} ms  "
                  f"alloc {stats.get('torch_peak_allocated_mb')} MB  reserved {stats.get('torch_peak_reserved_mb')} MB  "
                  f"smi {stats['nvidia_smi_peak_used_mb']} MB  stages {stats['stage_ms_mean']}")
            if clip_name == "720p_real_night":
                mid = len(outs) // 2
                cv2.imwrite(str(out_dir / f"{clip_name}_{preset}_f{mid}.png"), outs[mid])
                if preset == args.presets[0]:
                    cv2.imwrite(str(out_dir / f"{clip_name}_input_f{mid}.png"), n_use[mid])
                # no-reference quality on real footage
                sel = outs[:: max(1, len(outs) // 6)][:6]
                report.setdefault("real_quality", {})[preset] = {
                    "niqe": mean_of([{"v": no_ref("niqe", o)} for o in sel], "v"),
                    "brisque": mean_of([{"v": no_ref("brisque", o)} for o in sel], "v"),
                    "flat_noise": mean_of([{"v": compute_flat_region_noise(o)} for o in sel], "v"),
                    "mean_luma": round(float(np.mean([luma(o).mean() for o in sel])), 2),
                    "flicker_luma_delta": round(flicker(outs), 4),
                }
            del outs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if clip_name == "720p_real_night":
            sel = frames[:: max(1, len(frames) // 6)][:6]
            report.setdefault("real_quality", {})["input"] = {
                "niqe": mean_of([{"v": no_ref("niqe", o)} for o in sel], "v"),
                "brisque": mean_of([{"v": no_ref("brisque", o)} for o in sel], "v"),
                "flat_noise": mean_of([{"v": compute_flat_region_noise(o)} for o in sel], "v"),
                "mean_luma": round(float(np.mean([luma(o).mean() for o in sel])), 2),
                "flicker_luma_delta": round(flicker(frames), 4),
            }

    # ---------------- full-reference quality on synthetic night degradation
    if not args.skip_quality:
        report["synthetic_quality"] = {}
        rng_seed = 1234
        for preset in ["input"] + args.presets:
            rows = []
            for clip in CLEAN:
                clean = read_frames(clip, args.quality_frames)
                np.random.seed(rng_seed)  # identical degradation for every preset and every run
                low = [apply_dashcam_degradation(c) for c in clean]
                if preset == "input":
                    outs = low
                else:
                    outs, _ = run_clip(pipeline, preset, low, restore=True)
                    outs = [o if o.shape == c.shape else cv2.resize(o, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_AREA)
                            for o, c in zip(outs, clean)]
                idx = list(range(0, len(outs), max(1, len(outs) // 5)))[:5]
                rows.append({
                    "psnr": float(np.mean([psnr(clean[i], outs[i]) for i in idx])),
                    "ssim": float(np.mean([ssim(clean[i], outs[i]) for i in idx])),
                    "lpips": mean_of([{"v": lpips_score(clean[i], outs[i])} for i in idx], "v"),
                    "temporal_error": temporal_error(outs, clean),
                })
                if clip == CLEAN[0]:
                    cv2.imwrite(str(out_dir / f"synthetic_{preset}.png"), outs[len(outs) // 2])
                    if preset == "input":
                        cv2.imwrite(str(out_dir / "synthetic_clean.png"), clean[len(clean) // 2])
            report["synthetic_quality"][preset] = {k: mean_of(rows, k) for k in ("psnr", "ssim", "lpips", "temporal_error")}
            print(f"  quality {preset:<10} {report['synthetic_quality'][preset]}")

    path = OUT_DIR / f"{args.tag}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
