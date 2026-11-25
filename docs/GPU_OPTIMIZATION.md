# RTX 4050 GPU Bug Fixing & Performance Optimisation Report

All numbers below were measured on this machine with `scripts/benchmark_gpu.py`
(raw results: `results/gpu_benchmark/baseline.json`, `results/gpu_benchmark/improved.json`).
Nothing is estimated except where explicitly marked.

## 1. Detected environment

| Item | Detected value |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU (compute capability 8.9) |
| VRAM | 6141 MiB total; ~850–1450 MiB used by Windows/desktop apps when idle |
| NVIDIA driver | 581.86 (supports CUDA 13.0) |
| System RAM | 15.7 GB |
| Python | 3.14.5 |
| PyTorch (system Python, before) | 2.15.0**+cpu** — CPU-only build, `torch.version.cuda = None` |
| PyTorch (project `.venv`, after) | 2.14.1+cu130, cuDNN 9.24 |
| OpenCV / FFmpeg | OpenCV 5.1.0, FFmpeg from `imageio-ffmpeg` |

## 2. Bug report

| # | Bug | Root cause | Fix | Files |
|---|---|---|---|---|
| 1 | App ran on the CPU (0.15 FPS) although an RTX 4050 is installed | Installed PyTorch was a CPU-only wheel; `run_app.bat` expects a `.venv` that did not exist | Created `.venv` with PyTorch 2.14.1+cu130; GPU now used (4.3 FPS on the same job) | environment, `requirements.txt` |
| 2 | Partial-clip jobs produced a file as long as the whole clip, last frame frozen (10 s processed → 22.5 s file) | Full-length audio muxed without trimming | Mux trims audio to the processed duration (`-t`, `-shortest`) | `src/pipeline/video_stream.py` |
| 3 | Audio re-encoded twice (AAC→AAC) | Audio extracted to AAC then re-encoded in the mux | Original audio track stream-copied; re-encode only as fallback | `src/pipeline/video_stream.py` |
| 4 | Colour shift in enhanced videos | RGB→YUV used FFmpeg's default BT.601 matrix and the file was untagged; players decode HD as BT.709 | Encode with the BT.709 matrix and write BT.709 colour tags | `src/pipeline/video_stream.py` |
| 5 | Reported throughput wrong (0.15 FPS shown for a 4.33 FPS job) | FPS computed from the 11 metric sample frames instead of all processed frames | Pass the processed frame count | `src/utils/metrics.py`, `src/video_processor.py` |
| 6 | Results always said "NVIDIA RTX 4060" | Hard-coded string | Real device name from PyTorch | `src/utils/metrics.py` |
| 7 | Peak VRAM per job could include earlier jobs | Peak counter never reset | Reset at job start | `src/video_processor.py` |
| 8 | Job log said "No audio stream" even when audio existed | Logged before audio extraction ran | Logged after processing | `src/video_processor.py` |
| 9 | Fresh installs fail on balanced/quality presets | `spandrel` (SCUNet loader) missing from `requirements.txt` | Added | `requirements.txt` |
| 10 | After a CUDA OOM the reduced tile size was forgotten, so every frame hit the same OOM again | No state kept between frames | `tile_state` remembers the working tile size | `src/pipeline/tiling.py`, `stages.py` |
| 11 | Two tests failed on non-4060 hardware / OpenCV 5 | Asserted "RTX 4060" in the GPU name and an OpenCV 4.x error string | Hardware- and version-neutral assertions | `tests/test_application.py`, `tests/test_opencv_regression.py` |
| 12 | Enhanced video flickered more than the input (real clip: brightness jumps of 24 levels between consecutive frames) | Low-light strength switched in hard steps (1.0 / 0.60 / 0.22 / 0.05) on each frame's own mean brightness; a scene near a threshold (the test clip averages 80.3, threshold 80) toggled every few frames | Continuous strength curve through the same values, driven by a temporally smoothed brightness with scene-cut reset | `src/pipeline/stages.py`, `enhancer.py` |
| 13 | Low-light output 0.5 grey level too dark on average | `.byte()` truncated instead of rounding | Round before converting | `src/pipeline/stages.py` |
| 14 | Cancelling a job left FFmpeg running and `.temp_*` files on disk | Cancel is raised as `KeyboardInterrupt`, which `except Exception` did not catch | Catch it, stop FFmpeg, delete partial files, re-raise | `src/pipeline/video_stream.py` |
| 15 | Failed jobs left `.temp_*` files until the next restart | Cleanup only at start-up | Partial files removed on any failure | `src/pipeline/video_stream.py` |
| 16 | Upload errors exposed the server's file system path to the browser | Exception text (containing the path) returned in the JSON error | Generic message to the client, details in the server log | `app.py` |
| 17 | `python app.py` with the system Python ran on the CPU | System Python has a CPU-only PyTorch | `app.py` restarts itself with the project `.venv` when that environment has CUDA (opt out: `DASHCLARITY_NO_VENV_RELAUNCH=1`) | `app.py` |

## 3. Model audit

| Model | Checkpoint | Device / precision (after) | Status |
|---|---|---|---|
| Zero-DCE++ | `weights/ZeroDCE/Zero_DCE++_Epoch99.pth` (SHA-256 verified by `weights_manager`) | CUDA, fp16 | Loads strictly; output valid. fp16 is the fastest option measured (fp32 59 ms, fp16 32 ms at 720p) |
| SCUNet (color, real, PSNR) | `weights/SCUNet/scunet_color_real_psnr.pth` | CUDA, fp16, channels-last | Loads via spandrel; accelerated (see §4) |
| Real-ESRGAN general x4v3 | `weights/RealESRGAN/realesr-general-x4v3.pth` | CUDA, fp16, channels-last | Loads strictly |
| STCD (indoor_G) | `weights/STCD/indoor_G.pth` | CUDA, fp32 | Loads and runs (`stcd_comparison`, `hybrid_scunet_stcd`); not optimised |
| Farnebäck optical flow | OpenCV | flow on CPU at reduced scale; warp/blend now on GPU | Valid |

No weights were changed, replaced or downloaded. Presets that use each model are unchanged.

## 4. Optimisations

1. **SCUNet attention** (`src/models/scunet_fast.py`): profiling showed 26% of SCUNet time in a slow
   broadcast add of the relative-position bias and the shifted-window mask rebuilt every call. The window
   attention now uses PyTorch's fused `scaled_dot_product_attention` with a cached bias, and the mask is
   applied only to the border windows that need it. The transformer branch receives a contiguous tensor
   (LayerNorm/residuals had been running on strided memory). Output matches the reference within
   1.5e-4 (fp32), i.e. < 0.04 of one 8-bit grey level. **1.28–1.36x faster, ~8% less VRAM.**
2. **channels-last memory format** for SCUNet and Real-ESRGAN: Real-ESRGAN 321 → 219 ms, SCUNet a further ~1.2x.
3. **Memory-aware full-frame inference**: tiles of 512 px + overlap, padded to SCUNet's multiple of 64,
   roughly doubled the computed area of a 720p frame. Frames now run untiled when they fit the measured
   VRAM budget, tiled otherwise (no tile seams as a side benefit).
4. **Real-ESRGAN post-processing on the GPU**: the x4 frame (5120x2880 for 720p) was copied to the CPU as
   float32, blended with a Lanczos-upscaled copy and Lanczos-downscaled (~280 ms CPU work). The blend is
   linear, so it is now done after an antialiased GPU downscale. **SR stage 597 → 210 ms at 720p.**
5. **Temporal smoothing warp/blend on the GPU**: Farnebäck stays on the CPU at reduced scale; flow upsampling,
   warping, masks and blending moved to the GPU. **89 → 17–35 ms at 720p.**
6. **cuDNN autotuning, TF32**, and `inference_mode` around all inference.

## 5. Adaptive RTX 4050 / RTX 4060 support (`src/hardware.py`)

At start-up the pipeline reads the GPU and the VRAM that is free *at that moment* and derives:

* VRAM budget = min(85% of total, free − 400 MB)
* untiled frame limits from measured activation memory (SCUNet ≈ 1.6 KB/px, Real-ESRGAN ≈ 0.55 KB/px)
* tile size for frames above the limit, fp16, cuDNN autotune
* a per-process memory cap at the budget, so an over-allocation raises a recoverable OOM instead of
  spilling into shared system memory (which is very slow on Windows laptops)

| GPU state | Tier | Budget | SCUNet untiled up to | Behaviour |
|---|---|---|---|---|
| RTX 4050, 5.1 GB free (measured) | 6gb_class | 4673 MB | 2.45 MPx | 720p and 1080p run untiled |
| RTX 4050, only 3.0 GB free (simulated) | 6gb_class | 2600 MB | 1.17 MPx | 720p untiled, 1080p tiled |
| RTX 4060 8 GB, 7.0 GB free (simulated) | 8gb_class | 6600 MB | 3.64 MPx | 1080p untiled |

The RTX 4060 row is computed by the same code from typical values; **it was not tested on an RTX 4060.**
Overrides: `DASHCLARITY_TILE_SIZE`, `DASHCLARITY_VRAM_BUDGET_MB`, `DASHCLARITY_FP16=0`.

## 6. RTX 4050 benchmarks (same clips, frames and settings before and after)

Inputs: `input_videos/real/real_night_dashcam_01.mp4` (1280x720, 40 frames) and the 1920x1080 25 fps clip
(15 frames; quality preset 5 frames). One warm-up frame excluded. VRAM = PyTorch peak allocated / reserved.

| Clip / preset | FPS before | FPS after | Speed-up | ms/frame before → after | VRAM before (MB) | VRAM after (MB) |
|---|---|---|---|---|---|---|
| 720p fast | 24.61 | 24.67 | 1.00x | 41 → 41 | 576 / 618 | 576 / 598 |
| 720p balanced | 0.91 | 1.28 | 1.40x | 1098 → 784 | 1479 / 1832 | 1434 / 1870 |
| 720p quality | 0.55 | 0.99 | 1.80x | 1821 → 1014 | 665 / 1440 | 1474 / 1934 |
| 1080p fast | 10.81 | 10.86 | 1.00x | 92 → 92 | 1378 / 1420 | 1384 / 1436 |
| 1080p balanced | 1.27 | 1.87 | 1.47x | 788 → 536 | 1413 / 1996 | 1446 / 1686 |
| 1080p quality | 0.24 | 0.46 | 1.94x | 4244 → 2182 | 1449 / 2868 | 3115 / 4162 |

**Peak VRAM on the RTX 4050:** 4162 MB reserved by PyTorch (1080p quality, untiled), within the 4673 MB budget;
device-wide peak 5401 of 6141 MiB including the desktop. End-to-end app job (1080p quality, 4 s): 0.45 FPS,
peak 3108 MB allocated. Process RAM peaked at 1.4–2.6 GB.

Stage times after (720p quality): SCUNet 729 ms, Real-ESRGAN 210 ms, Zero-DCE++ 38 ms, temporal 35 ms.
With the quality changes of §7b, Zero-DCE++ dropped further to 8.8 ms per 720p frame (curves estimated at 1/4 resolution).

## 7. Quality (no regressions)

Synthetic pairs: 3 clean daytime clips x 16 frames, degraded with `scripts/degrade.py` (fixed seed), scored
against the clean frames (restore to original resolution). LPIPS uses AlexNet (`lpips` package).

| Preset | PSNR before → after | SSIM before → after | LPIPS before → after (lower better) | Temporal error before → after (lower better) |
|---|---|---|---|---|
| degraded input | 7.03 | 0.197 | 0.855 | 4.27 |
| fast | 13.18 → 13.18 | 0.642 → 0.642 | 0.850 → 0.850 | 7.19 → 7.19 |
| balanced | 12.72 → 12.71 | 0.707 → 0.707 | 0.472 → 0.472 | 3.84 → 3.84 |
| quality | 12.74 → 12.78 | 0.710 → 0.716 | 0.447 → 0.442 | 3.68 → 3.61 |

Real night clip (no ground truth): flat-region noise for the quality preset dropped from 0.45 to 0.19
(tile seams and aliased decimation removed); luminance and flicker unchanged. Old vs new output frames:
45–57 dB PSNR, mean difference < 1 grey level. NIQE/BRISQUE were not computed (`pyiqa` not installed;
the project's own metric notes flag them as unreliable for night footage).

Low absolute PSNR (~13 dB) is expected: the degradation darkens frames by a gamma of 2.4, so brightness
mismatch dominates; use these numbers to compare presets and versions, not as absolute quality.

Visual comparison: `results/gpu_benchmark/comparison_before_after.png`.

## 7b. Enhancement quality improvements (low-light stage)

Controlled experiments (`scripts/quality_experiments.py`, balanced preset, identical inputs; one setting
changed per variant; results in `results/gpu_benchmark/quality_experiments_balanced*.json`):

| Variant | PSNR | SSIM | LPIPS ↓ | Temporal err ↓ | Real NIQE ↓ | Real BRISQUE ↓ | Real noise ↓ | Real flicker ↓ | Zero-DCE ms |
|---|---|---|---|---|---|---|---|---|---|
| V0 original (step exposure) | 12.77 | 0.7099 | 0.4699 | 3.849 | 8.46 | 70.86 | 0.284 | 2.48 | 37.4 |
| V1 smooth exposure | 12.77 | 0.7099 | 0.4699 | 3.849 | 8.49 | 70.81 | 0.290 | **1.26** | 37.4 |
| V2 curves at 1/12 res | 12.92 | 0.7088 | 0.4760 | 3.851 | 8.47 | 72.50 | 0.257 | 1.27 | 8.2 |
| V3 curves at 1/4 res | 12.84 | 0.7119 | 0.4759 | 3.862 | 8.38 | 71.13 | 0.275 | 1.27 | 8.2 |
| V4 gray-world 50% | 12.98 | 0.7102 | **0.4509** | 3.836 | 8.40 | 70.52 | 0.293 | 1.25 | 37.4 |
| V5 gray-world off | 13.17 | 0.7105 | 0.4523 | 3.840 | 8.41 | 70.66 | 0.291 | 1.26 | 37.1 |
| V6 CLAHE 1.5 | 14.47 | 0.7487 | 0.4682 | 5.190 | 7.16 | 47.58 | 0.536 | 0.97 | 40.5 |
| V7 CLAHE 2.5 | 14.90 | 0.7407 | 0.4895 | 6.226 | 6.75 | 42.17 | 0.632 | 0.85 | 43.8 |
| **V8 adopted (V1 + V3 + V4)** | **13.05** | **0.7123** | **0.4569** | 3.854 | **8.36** | 71.10 | **0.275** | **1.27** | **8.8** |
| input (real clip) | – | – | – | – | 7.16 | 62.03 | 0.288 | 1.61 | – |

Adopted defaults (configurable in `config.yaml` → `models.low_light_tuning`): smooth exposure, curves estimated
at 1/4 resolution (the published Zero-DCE++ design), gray-world at 50%. CLAHE raised NIQE/BRISQUE/PSNR but
increased temporal error by 35–62% and doubled noise, so it is off by default (kept as an option).
Honest note: NIQE/BRISQUE of the real clip are better for the untouched input than for any enhanced output;
the input is already moderately exposed (mean luma 80) and brightening it raises these naturalness scores.

**Full-video comparison** (150-frame real night clip, each version's own pipeline + encoder, `scripts/compare_versions.py`):

| Preset | Original code | Improved code | Flicker original → improved (input 0.87) | Mean luma |
|---|---|---|---|---|
| balanced | 161.9 s (0.93 FPS) | 113.9 s (1.32 FPS), 1.42x | 1.21 → **0.66** | 102.6 → 102.3 |
| quality | 273.7 s (0.55 FPS) | 186.3 s (0.81 FPS), 1.47x | 1.21 → **0.66** | 103.3 → 103.0 |

At frame 22 the input brightens 79.1 → 80.2 and the original output *darkens* 117 → 93 (the step bug);
the improved output follows the input (107 → 108 → 113 → 116). Side-by-side videos (input | original |
improved): `results/gpu_benchmark/video_comparison/side_by_side_*.mp4`; stills: `most_different_frames.png`.
A first improved balanced run measured 143 s right after a long GPU workload; re-run under the same thermal
conditions as the original it took 113.9 s — laptop GPU clocks vary, so compare runs made back to back.

## 7c. Batch size, model loading and memory monitoring

Measured throughput on the RTX 4050 (fp16, channels-last, 720p):

| Model | batch 1 | batch 2 | batch 4 |
|---|---|---|---|
| SCUNet | 1.68 fps / 1364 MB | 1.57 fps / 2680 MB | 1.15 fps / 5312 MB |
| Real-ESRGAN x4 | 5.19 fps / 306 MB | 5.20 fps / 564 MB | 5.21 fps / 1082 MB |
| Zero-DCE++ (1/4-res curves) | 656 fps | 953 fps | 704 fps |

A single 720p frame already saturates this GPU, so larger batches add VRAM without adding throughput (SCUNet
gets slower). Batch size 1 is therefore the measured optimum on 6 GB and is kept; the temporal stage also
needs frames in order. Memory is adapted through tile size / untiled limits instead (§5).

* **Model unloading**: on GPUs with a budget below 6 GB (`unload_unused_models` in the profile) models not used
  by the active preset are released when the preset changes; cached allocator memory is returned to the
  driver after every job (`release_cached_memory`), so the desktop and other apps get it back.
* **Monitoring**: `GET /api/system/status` now returns live VRAM (`allocated/reserved/peak/device free`), the
  driver, the CUDA build and the adaptive profile; each job log records its peak allocated / reserved VRAM and
  the budget.

## 8. Video reliability — end-to-end upload tests

`scripts/make_test_videos.py` builds 14 test files from footage in the repo; `scripts/e2e_upload_check.py`
uploads each through the web API, enhances it, and checks the preview, streaming (HTTP 206), download,
side-by-side comparison, thumbnail and the output file itself with FFmpeg. Final run:
**23 / 23 cases passed** (`results/e2e_upload_check.json`).

| Case | Preset / options | Output check (all passed) |
|---|---|---|
| 720p MP4 + AAC | balanced, fast (strength 0.5), classical_baseline, stcd_comparison, hybrid_scunet_stcd | 90 frames, 30 fps, 3.00 s, audio 3.00 s |
| 720p MP4 + AAC | frame_skip 2 | 45 frames at 15 fps, 3.00 s, audio in sync |
| 720p MP4 + AAC | max_seconds 1 | 30 frames, video 1.00 s, audio 0.98 s |
| 720p MP4 + AAC | quality, restore_at_original_res=false | 5120x2880 (x4), 0.5 s, audio kept |
| 720p MP4 + AAC | cancel after 4 frames | cancelled, FFmpeg stopped, no partial files |
| 641x361 4:4:4 H.264 + AAC | quality | 640x360 (even size required by H.264 4:2:0), 2.00 s |
| Phone video rotated 90° | fast | 720x1280 upright, no leftover rotation tag |
| AVI (MJPEG + PCM) | balanced | preview transcoded to H.264; PCM audio kept |
| MOV + AAC / MKV + Opus | classical_baseline / balanced | audio stream-copied, lengths match |
| Variable frame rate + MP3 | fast | 2.90 s video vs 2.95 s input, audio 2.88 s |
| 2-frame clip | quality | 2 frames out |
| HEVC 10-bit | fast | decoded, preview transcoded to H.264 |
| Daylight 720p | balanced | not over-brightened (adaptive exposure) |
| 1080p 25 fps + AAC | quality | 50 frames, 25 fps, 2.00 s, untiled SCUNet |
| 4K 3840x2160 | balanced, quality | correct size and duration; SCUNet tiled automatically |
| Truncated MP4 / text file named .mp4 | upload | rejected with HTTP 400, no server path in the message |

Peak VRAM over the whole run (including 4K): **3543 MB allocated** (budget 4673 MB).
Processing speed reported in the app for these jobs: fast 47–108 FPS, balanced 1.4–5.6 FPS depending on
resolution, quality 0.37 FPS at 1080p, 0.1 FPS at 4K.

Earlier single-job checks: 30 → 30 fps / 300 → 300 frames (WhatsApp clip, 10 s of 22.5 s with audio trimmed),
25 → 25 fps / 100 → 100 frames (1080p), BT.709 colour round-trip error 2–4 levels.

## 9. Tests

`python -m unittest tests/test_opencv_regression.py tests/test_application.py tests/test_ui.py tests/test_gpu_optimizations.py`
→ **30 passed, 0 failed** (on the `.venv` CUDA build). New file `tests/test_gpu_optimizations.py` covers the
adaptive profile (6 GB, busy 6 GB, 8 GB, CPU), OOM tile memory, exposure continuity and scene-cut reset,
SCUNet patch equivalence, audio trimming + BT.709 tags, and cleanup on cancel.
End-to-end: 23 / 23 upload cases (§8).

## 10. Remaining limitations

* SCUNet is still ~70–75% of frame time in balanced/quality. Remaining cost is spread across LayerNorm
  (memory-bound at 64 channels), concatenation and convolutions; `torch.compile` could fuse these but
  needs Triton, which is not available for this Windows/Python 3.14 setup.
* Batch size stays 1: measured larger batches give no throughput gain on this GPU (§7c).
* STCD was not optimised (fp32, untiled).
* The OOM fallback can still demote a preset as a last resort (after tile sizes down to 192 px fail).
* RTX 4060 behaviour is derived from the same measured per-pixel memory figures, not tested on an RTX 4060.
* `templates/about.html` still quotes the original author's RTX 4060 measurements.
* NIQE / BRISQUE are installed through `pyiqa` without its full dependency tree (installed with `--no-deps`
  plus only the modules it imports); they are optional and only used by the benchmark scripts.
* Laptop GPU clocks vary with temperature (2.0–2.7 GHz observed); compare timings measured back to back.

## 11. Modified and new files

Modified: `app.py`, `config.yaml`, `requirements.txt`, `README.md`, `src/pipeline/enhancer.py`,
`src/pipeline/stages.py`, `src/pipeline/tiling.py`, `src/pipeline/video_stream.py`, `src/utils/metrics.py`,
`src/video_processor.py`, `tests/test_application.py`, `tests/test_opencv_regression.py`.

New: `src/hardware.py`, `src/models/scunet_fast.py`, `scripts/benchmark_gpu.py`, `scripts/quality_experiments.py`,
`scripts/compare_versions.py`, `scripts/make_test_videos.py`, `scripts/e2e_upload_check.py`,
`tests/test_gpu_optimizations.py`, this report, and results under `results/gpu_benchmark/` and `results/e2e_upload_check*.json`.

Original code backup: `../backup_before_gpu_work_2026-11-25/` (taken at commit 5c3da41 plus the then-uncommitted
`stages.py` change).
