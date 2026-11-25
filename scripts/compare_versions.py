"""
DashClarity AI - End-to-end comparison of two code versions on the same video.

Runs a repository's own pipeline + streaming writer (so each version is measured exactly as it
ships) and records wall time, FPS, peak VRAM and the output file. Run once per version:

    <repo>/.venv/Scripts/python.exe scripts/compare_versions.py --root <repo_dir> --input clip.mp4 \
        --output out.mp4 --preset balanced --json run.json

Then compare the two outputs:

    python scripts/compare_versions.py --diff old.mp4 new.mp4 --reference clip.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def run(root: Path, inp: str, out: str, preset: str, js: str) -> None:
    sys.path.insert(0, str(root))
    os.chdir(root)
    os.environ["TEST_MODE"] = "1"
    import torch
    from src.pipeline.enhancer import VideoEnhancementPipeline
    from src.pipeline.video_stream import StreamingVideoProcessor

    pipe = VideoEnhancementPipeline(preset=preset)
    pipe.reset_temporal()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    sp = StreamingVideoProcessor(inp, out)
    info = sp.process(lambda f, prev: pipe.enhance_frame(f, prev, preset=preset))
    wall = time.perf_counter() - t0
    res = {"root": str(root), "preset": preset, "frames": info["total_frames_processed"], "wall_seconds": round(wall, 2),
           "fps": round(info["total_frames_processed"] / wall, 3),
           "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if torch.cuda.is_available() else None,
           "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2 ** 20, 1) if torch.cuda.is_available() else None}
    Path(js).write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


def frames(path: str):
    import cv2
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def diff(old: str, new: str, ref: str) -> None:
    import cv2
    a, b, r = frames(old), frames(new), frames(ref)
    n = min(len(a), len(b))
    ps = []
    for i in range(n):
        mse = np.mean((a[i].astype(np.float64) - b[i].astype(np.float64)) ** 2)
        ps.append(99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse))

    def lum(fs):
        return [float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()) for f in fs]

    def flick(fs):
        m = lum(fs)
        return float(np.mean(np.abs(np.diff(m))))

    print(json.dumps({"frames_old": len(a), "frames_new": len(b), "frames_input": len(r),
                      "psnr_old_vs_new_mean": round(float(np.mean(ps)), 2), "psnr_old_vs_new_min": round(float(np.min(ps)), 2),
                      "flicker_input": round(flick(r), 4), "flicker_old": round(flick(a), 4), "flicker_new": round(flick(b), 4),
                      "mean_luma_input": round(float(np.mean(lum(r))), 2), "mean_luma_old": round(float(np.mean(lum(a))), 2),
                      "mean_luma_new": round(float(np.mean(lum(b))), 2)}, indent=2))
    # side-by-side video for visual review: input | old | new
    h, w = a[0].shape[:2]
    ff = subprocess.run([sys.executable, "-c", "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"],
                        capture_output=True, text=True).stdout.strip()
    side = str(Path(new).with_name("side_by_side_input_old_new.mp4"))
    cmd = [ff, "-y", "-loglevel", "error", "-i", ref, "-i", old, "-i", new, "-filter_complex",
           "[0:v]scale=640:-2,drawtext=text='input':x=10:y=10:fontcolor=white:box=1:boxcolor=black@0.6[v0];"
           "[1:v]scale=640:-2,drawtext=text='original code':x=10:y=10:fontcolor=white:box=1:boxcolor=black@0.6[v1];"
           "[2:v]scale=640:-2,drawtext=text='improved code':x=10:y=10:fontcolor=white:box=1:boxcolor=black@0.6[v2];"
           "[v0][v1][v2]hstack=inputs=3[o]", "-map", "[o]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", side]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:  # drawtext needs a font; fall back to an unlabeled stack
        cmd[cmd.index("-filter_complex") + 1] = ("[0:v]scale=640:-2[v0];[1:v]scale=640:-2[v1];[2:v]scale=640:-2[v2];"
                                                 "[v0][v1][v2]hstack=inputs=3[o]")
        r = subprocess.run(cmd, capture_output=True, text=True)
    print("side-by-side:", side if r.returncode == 0 else r.stderr[-300:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--preset", default="balanced")
    ap.add_argument("--json", default="run.json")
    ap.add_argument("--diff", nargs=2)
    ap.add_argument("--reference")
    a = ap.parse_args()
    if a.diff:
        diff(a.diff[0], a.diff[1], a.reference)
    else:
        run(Path(a.root).resolve(), str(Path(a.input).resolve()), str(Path(a.output).resolve()), a.preset, a.json)
