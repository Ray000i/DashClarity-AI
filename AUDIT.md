# DashClarity AI - Comprehensive Codebase & Architecture Audit
**Audit Date:** 2026-11-18  
**Target Environment:** Windows 11, NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6), PyTorch 2.5.1+cu121, Python 3.10  
**Phase:** Phase 0 - Full System Audit (Pre-Remediation)

---

## 1. Test Suite & Runtime Baseline Execution

### Unit Test Execution Log
The test suite was executed via:
`.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"`

- **Results:** Ran 17 tests in 26.414s — `OK`.
- **Recorded Errors & Warnings:**
  1. `ResourceWarning: unclosed file <...output_videos\enhanced_test_day_job.mp4>` in `test_07_media_streaming_and_download`.
  2. `ResourceWarning: unclosed file <...output_videos\comparison_test_ui_dummy_job.mp4>` in `test_07_side_by_side_download_endpoint`.
  3. FFmpeg stream warning: `[mov,mp4,m4a,3gp,3g2,mj2 @ ...] moov atom not found` emitted during OpenCV `VideoCapture` probing of synthetic dummy video files.
  4. Real directory pollution: Tests write test video uploads (`night_upload.mp4`, `test_corrupt.mp4`, `test_binary.exe`, `test_daytime.mp4`, etc.) into `input_videos/` and completed test outputs (`enhanced_test_day_job.mp4`, `enhanced_test_lowres_job.mp4`, etc.) into `output_videos/`, causing them to show up in the production history page.

### Application Startup & Health Probe
The application was probed via Flask test client:
- `/api/system/status` -> HTTP 200 OK:
  `{"cuda_available": True, "device_name": "NVIDIA GeForce RTX 4060 Laptop GPU", "allocated_vram_mb": 0.0, "max_vram_mb": 0.0, "model_architecture": "Modular: SCUNet + Zero-DCE++ + Real-ESRGAN + Farneback"}`
- `/history` -> HTTP 200 OK (emits `moov atom not found` when OpenCV attempts to generate thumbnails for dummy test files).

---

## 2. Comprehensive Issue Audit Table

| File | Issue | Severity | Planned Fix |
| :--- | :--- | :---: | :--- |
| `tests/test_application.py` | Tests write directly to production `input_videos/` and `output_videos/`. | **Blocker** | Introduce `TEST_MODE=1` environment variable and `tempfile.TemporaryDirectory` redirection so test runs never touch real storage. Clean up all junk test jobs. |
| `tests/test_ui.py` | Writes `test_ui_dummy_job_sample.mp4` and `enhanced_test_ui_dummy_job.*` to real folders; unclosed file warnings on `send_file`. | **Blocker** | Redirect tests to isolated temporary folders and properly close file handles. |
| `src/video_processor.py` | No GPU concurrency control: concurrent `/api/enhance` requests launch parallel GPU threads leading to CUDA OOM. | **Blocker** | Implement a thread-safe single-GPU job queue (`queue.Queue`) with real-time queue position reporting (`Queue position: X`). |
| `src/video_processor.py` | Server restart leaves active jobs in `processing_frames` or `queued` state forever (infinite spinner for users). | **Blocker** | At app startup, sweep jobs and mark any incomplete jobs as `interrupted` or `failed` with clear status messaging. |
| `src/video_processor.py` | If worker thread terminates unexpectedly (e.g., uncaught exception or fatal crash), frontend polls indefinitely. | **Blocker** | Add worker thread heartbeat/liveness check in `get_job` / `to_dict` that marks dead worker jobs as `failed`. |
| `src/video_processor.py` | History rate displays integer/unformatted duration (`0.8s processed`) and rate is missing proper two-decimal precision. | **Major** | Calculate `proc_fps = frames_processed / processing_seconds`, serialize it, format to 2 decimals (`0.57 FPS`), and rename label to "Processing time". |
| `src/video_processor.py` | No delete-job capability; junk test files clutter history forever. | **Major** | Add `/api/job/<job_id>` `DELETE` endpoint and frontend delete button with confirmation dialog. |
| `app.py` | Path traversal risk on `/media/*`, `/api/result/*`, `/api/download/*` routes from unsanitized `job_id`. | **Blocker** | Validate `job_id` with strict regex (`^[a-f0-9]{12}$` or alphanumeric safeguard) before resolving filesystem paths. |
| `app.py` | No custom error handlers for HTTP 404, 500, or 413 (File Too Large). Raw errors/white screens appear. | **Major** | Implement `@app.errorhandler` for 400, 404, 413, and 500 rendering consistent dark-theme pages or JSON error payloads. |
| `app.py` | Debug mode defaults to hardcoded `False` in entrypoint without environment variable control. | **Minor** | Read `FLASK_DEBUG` / `DEBUG` from environment variables, defaulting to `False`. |
| `app.py` | Hardcoded secret key fallback in code. | **Minor** | Check `.env` and generate a secure random secret key if absent. |
| `src/pipeline/tiling.py` | Tile OOM recovery falls back directly to CPU rather than dynamically reducing tiles or falling back to lighter preset. | **Major** | Add structured `torch.cuda.OutOfMemoryError` handler: clear cache, attempt smaller tiles, and if exhausted, gracefully fall back to a lighter preset (e.g., disable SR or switch to fast denoise) with user notification. |
| `src/pipeline/video_stream.py` | Odd dimensions, portrait/rotated videos, and clips < 1s can cause FFmpeg pipeline errors or division-by-zero. | **Major** | Enforce even dimensions `(w//2)*2` and `(h//2)*2` across all frames; clamp minimum target frames to 1; sanitize rotation metadata. |
| `src/utils/metrics.py` | Metric helper functions (`compute_sky_luminance_ratio`, `compute_bloom_area_ratio`, `compute_flat_region_noise`) lack individual try/except guards. | **Major** | Wrap each metric calculation in try/except blocks returning safe fallback ("n/a" or neutral) so no visual metric can crash a job. |
| `src/models/weights_manager.py` | Startup does not perform preflight check of required weights; missing weights only detected during enhancement. | **Major** | Add validation function at app startup that verifies weight presence and checksums, with clear error messages and exact download commands. |
| `models/STCD` vs `src/models/stcd.py` | STCD architecture and options are stored under `models/STCD` while wrapper is in `src/models/stcd.py`; potential path duplication. | **Minor** | Clarify and standardize STCD model import paths; verify weights path (`weights/STCD/indoor_G.pth`). |
| `config.yaml` vs `README.md` | **Stage Order Contradiction:** Code executes Denoise (SCUNet) -> Low-Light (Zero-DCE++) -> SR -> Flow, whereas `README.md` flowchart depicts Zero-DCE++ -> SCUNet. | **Major** | Align documentation, flowchart, and UI to match the real theoretical order executed in code (SCUNet denoising first to avoid amplifying Poisson noise). |
| `scripts/benchmark_presets.py` | Loops over obsolete presets (`low_light`, `rain_noise`, `upscale_only`) that do not exist in `config.yaml`. | **Major** | Update benchmark script to benchmark the exact active presets (`fast`, `balanced`, `quality`, `stcd_comparison`, `hybrid_scunet_stcd`, `classical_baseline`) with unbuffered stdout and measure authentic FPS and peak VRAM. |
| `templates/upload.html` | Missing estimated processing time calculation (`duration_cap * fps / preset_fps`) before starting job. | **Major** | Add client-side estimated processing duration badge next to Enhance button based on clip duration and selected preset FPS. |
| `templates/processing.html` | Missing visible Cancel button; user cannot cancel a running job from the UI. | **Major** | Add Cancel button connected to `/api/cancel/<job_id>`. |
| `templates/results.html` | Missing explicit legal forensic warning ("Enhanced footage is not original evidence"). | **Minor** | Add visible disclaimer banner and verify interactive shortcuts (Space, arrows, S, L). |
| `templates/base.html` | Missing system hardware/health indicator chip in navbar. | **Minor** | Add `/health` endpoint and navbar status chip (GPU status, weights status). |
| `scripts/preflight.py` | Script does not exist; no automated pre-demo verification checklist. | **Major** | Create `scripts/preflight.py` checking Python, PyTorch, CUDA, GPU VRAM, FFmpeg, weights, sample videos, and disk space. |
| `run_app.bat` | Does not execute preflight checks or automatically launch the default browser. | **Minor** | Update `run_app.bat` to run preflight, start app, and auto-open browser on `http://127.0.0.1:5000`. |
| `LICENSE` | File is missing from repository root despite being referenced in `README.md`. | **Minor** | Add standard MIT LICENSE file with user's attribution. |

---

## 3. Critical Architectural & Documentation Discrepancies

### 1. Stage Order Justification (Denoise vs Illumination)
- **Current Code (`src/pipeline/enhancer.py` & `REPORT.md` Section 1.1):**  
  `SCUNet Denoise -> Zero-DCE++ Illumination -> Real-ESRGAN Super-Resolution -> Farneback Temporal Flow`  
  *Justification:* Denoising first removes sensor shot noise before non-linear illumination gain. If Zero-DCE++ is applied first, additive noise $\epsilon$ is multiplied by quadratic curve factors $\mathcal{A}(1 - 2I_0)$, causing catastrophic noise amplification and chroma boiling.
- **Outdated Docs (`README.md` Diagram & Overview):**  
  Showed `Zero-DCE++ -> SCUNet`.
- **Planned Fix (Phase 2):**  
  Standardize `README.md`, `templates/about.html`, and `templates/index.html` to reflect the actual, scientifically defended order: `Denoise -> Illumination -> Super-Resolution -> Temporal Smoothing`.

### 2. Preset Definitions & Naming
- In `config.yaml`, presets are: `fast`, `balanced`, `quality`, `full` (alias), `stcd_comparison`, `hybrid_scunet_stcd`, `classical_baseline`.
- In `scripts/benchmark_presets.py`, presets were: `low_light`, `rain_noise`, `upscale_only`, `full`, `stcd_comparison`, `classical_baseline`.
- In `README.md`, only `fast`, `balanced`, `quality`, and `stcd_comparison` are listed in the table.
- **Planned Fix (Phase 2):** Define a single authoritative Python dictionary in `config.py` containing preset metadata, active stages, descriptions, and measured benchmarks, from which all UI and documentation tables derive.

---

## 4. Questions for Me (User Clarifications)

1. **License Attribution Name:**  
   What name should be used on the MIT license file? (e.g., "Mohammad / Ray000i" or a specific full name?)
2. **Demo Clips Selection:**  
   For Phase 4 `DEMO_MODE`, which 2-3 sample clips in `input_videos/` should be pre-processed and pinned in History? (We recommend `real_night_dashcam_01.mp4`, `sample_night_dashcam.mp4`, and `19960474-hd_1080_1920_30fps.mp4`).
3. **Stage Order Defense in Presentation:**  
   Does your teacher/syllabus prefer the terminology "Denoise -> Illumination" (the code's actual mathematically justified order) or did your project proposal state "Zero-DCE++ -> SCUNet"? (Our viva script in `docs/VIVA_QA.md` and `docs/REPORT.md` will clearly explain why denoising must precede illumination to prevent noise amplification).
