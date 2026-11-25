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

## 8. Video reliability (verified with FFmpeg on real outputs)

| Check | Result |
|---|---|
| Frame rate | preserved (30 → 30, 25 → 25) |
| Frame count | 300 → 300, 100 → 100 decoded |
| Resolution | preserved (626x360, 1920x1080) |
| Duration | equals processed frames / fps (10.00 s, 4.00 s) |
| Audio | original track copied (AAC 128 kb/s kept), trimmed to video length |
| Colour | BT.709 matrix and tags; round-trip error 2–4 levels (H.264 4:2:0) |

## 9. Tests

`python -m unittest tests/test_opencv_regression.py tests/test_application.py tests/test_ui.py` →
**19 passed, 0 failed** (on the `.venv` CUDA build).

## 10. Remaining limitations

* SCUNet is still ~70–75% of frame time in balanced/quality. Remaining cost is spread across LayerNorm
  (memory-bound at 64 channels), concatenation and convolutions; `torch.compile` could fuse these but
  needs Triton, which is not available for this Windows/Python 3.14 setup.
* Frames are processed one at a time; batching is not used because the temporal stage depends on the
  previous output frame, and single 720p/1080p frames already fill the GPU.
* Zero-DCE++ (38 ms at 720p, 88 ms at 1080p) and STCD were not optimised.
* The OOM fallback can still demote a preset as a last resort (after tile sizes down to 192 px fail).
* RTX 4060 behaviour is derived, not measured.
* `templates/about.html` still quotes the original author's RTX 4060 measurements.
* The app must be started from `.venv` (`run_app.bat`); `python app.py` with the system Python runs on the CPU.
