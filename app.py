"""
DashClarity AI - Flask Web Application
AI-Powered Dashcam Footage Enhancement for Incident Review and Insurance Claims.
Academic End-Semester Generative AI Platform.
"""

import os
import json
import time
import logging
from pathlib import Path
from flask import (
    Flask, render_template, request, jsonify,
    send_file, redirect, url_for, abort, Response
)
import torch

from config import (
    UPLOAD_DIR, OUTPUT_DIR, ALLOWED_EXTENSIONS,
    MAX_CONTENT_LENGTH, SECRET_KEY, PORT, HOST
)
from src.utils import generate_job_id, sanitize_filename, is_allowed_file
from src.metadata import extract_video_metadata
from src.video_processor import ProcessingManager, JobState

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DashClarity.App")

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

processing_manager = ProcessingManager.get_instance()

@app.context_processor
def inject_system_info():
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "CPU Mode"
    max_size_mb = MAX_CONTENT_LENGTH // (1024 * 1024)
    return dict(
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        max_upload_size_mb=max_size_mb
    )

# -----------------------------------------------------------------------------
# Frontend Page Routes
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    """Landing dashboard page with project overview, statistics, and quick-start CTA."""
    return render_template(
        "index.html",
        active_page="home"
    )

@app.route("/upload")
def upload_page():
    """Video upload and preview page."""
    return render_template("upload.html", active_page="upload")

@app.route("/processing/<job_id>")
def processing_page(job_id):
    """Real-time processing dashboard for active enhancement jobs."""
    job = processing_manager.get_job(job_id)
    if not job:
        return redirect(url_for("upload_page"))
    return render_template(
        "processing.html",
        job_id=job_id,
        active_page="processing"
    )

@app.route("/results/<job_id>")
def results_page(job_id):
    """Comparison and visual quality results page."""
    job = processing_manager.get_job(job_id)
    if not job or job.status != "completed":
        return redirect(url_for("processing_page", job_id=job_id))
    return render_template(
        "results.html",
        job_id=job_id,
        active_page="results"
    )

@app.route("/history")
def history_page():
    """Past enhancement jobs history page."""
    jobs = processing_manager.list_history()
    return render_template("history.html", active_page="history", jobs=jobs)

@app.route("/about")
def about_page():
    """Academic project documentation, architectural diagrams, and research context."""
    return render_template("about.html", active_page="about")

# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------

@app.route("/api/presets", methods=["GET"])
def api_presets():
    """
    Returns active pipeline presets dynamically loaded from config.yaml and measured benchmarks from results.json.
    """
    import yaml
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    presets_cfg = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            presets_cfg = cfg.get("presets", {})

    # Load benchmark measured throughput if available
    bench_path = Path(__file__).resolve().parent / "results" / "results.json"
    profiling = {}
    if bench_path.exists():
        try:
            with open(bench_path, "r", encoding="utf-8") as f:
                res_data = json.load(f)
                profiling = res_data.get("profiling", {}).get("presets", {})
        except Exception:
            pass

    tradeoffs = {
        "fast": "Light bilateral denoise only; high-ISO CMOS noise is preserved.",
        "balanced": "Fast neural denoiser and motion compensation; 6x longer processing latency.",
        "quality": "Full neural SCUNet + Real-ESRGAN super-resolution; slow execution (0.6 FPS).",
        "stcd_comparison": "Experimental indoor-trained model; slower inference and potential color domain shift.",
        "hybrid_scunet_stcd": "Research ablation combining neural denoiser and spatial-temporal transformer.",
        "classical_baseline": "Non-deep learning classical bilateral baseline for academic comparison."
    }
    labels = {
        "fast": "Fast (real-time demo)",
        "balanced": "Balanced (recommended)",
        "quality": "Quality (full pipeline)",
        "stcd_comparison": "STCD comparison (ablation)",
        "hybrid_scunet_stcd": "Hybrid (SCUNet -> STCD)",
        "classical_baseline": "Classical baseline"
    }

    preset_list = []
    for key, p in presets_cfg.items():
        if key == "full":
            continue
        p_720 = profiling.get("720p", {}).get(key, {})
        p_1080 = profiling.get("1080p", {}).get(key, {})
        preset_list.append({
            "key": key,
            "label": labels.get(key, key.replace("_", " ").title()),
            "description": p.get("description", ""),
            "tradeoff": tradeoffs.get(key, "Standard processing parameters."),
            "measured_fps_720p": p_720.get("fps", None),
            "measured_fps_1080p": p_1080.get("fps", None),
            "enable_neural_denoise": p.get("enable_neural_denoise", False),
            "enable_classical_denoise": p.get("enable_classical_denoise", False),
            "enable_low_light": p.get("enable_low_light", False),
            "enable_super_res": p.get("enable_super_res", False),
            "enable_temporal": p.get("enable_temporal", False),
            "strength": p.get("strength", 1.0)
        })

    return jsonify({"presets": preset_list})

@app.route("/api/history", methods=["GET"])
def api_history():
    """Returns list of past enhancement jobs."""
    items = processing_manager.list_history()
    return jsonify({"jobs": items, "count": len(items)})

@app.route("/api/logs/<job_id>", methods=["GET"])
def api_job_logs(job_id):
    """Returns real-time execution logs for an active or completed job."""
    job = processing_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({"job_id": job_id, "logs": job.logs})

@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    Accepts video uploads, validates file type and size, extracts metadata,
    and initializes a processing job record.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided in request."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not is_allowed_file(file.filename, ALLOWED_EXTENSIONS):
        return jsonify({
            "error": f"Unsupported format. Permitted video formats: {', '.join(ALLOWED_EXTENSIONS).upper()}"
        }), 400

    try:
        job_id = generate_job_id()
        safe_name = sanitize_filename(file.filename)
        saved_filename = f"{job_id}_{safe_name}"
        saved_path = UPLOAD_DIR / saved_filename

        file.save(str(saved_path))
        logger.info(f"Uploaded video saved to {saved_path} (Job ID: {job_id})")

        # Extract metadata
        metadata = extract_video_metadata(str(saved_path))

        # Register job in processing manager
        processing_manager.create_job(job_id, str(saved_path))

        return jsonify({
            "success": True,
            "job_id": job_id,
            "filename": safe_name,
            "metadata": metadata,
            "preview_url": f"/media/original/{job_id}"
        }), 201

    except Exception as e:
        logger.exception(f"Upload failed: {e}")
        return jsonify({"error": f"Failed to process uploaded video: {str(e)}"}), 500

@app.route("/api/video/<job_id>", methods=["GET"])
def api_video_info(job_id):
    """Returns metadata for an uploaded video."""
    job = processing_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    try:
        meta = extract_video_metadata(job.input_path)
        return jsonify({"job_id": job_id, "metadata": meta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/enhance/<job_id>", methods=["POST"])
def api_start_enhance(job_id):
    """
    Initiates model enhancement for a validated job.
    Accepts JSON body with modular pipeline parameters:
      preset: fast, balanced, quality, stcd_comparison, hybrid_scunet_stcd, classical_baseline
      strength: 0.1 to 1.0 (default 1.0)
      max_seconds: optional float cap (e.g. 10.0 for fast demo)
      frame_skip: 1 for every frame, 2 for 2x faster evaluation
      restore_at_original_res: whether to downscale SR output back to original resolution
      stage_overrides: optional dict with boolean stage toggles
    """
    job = processing_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job.status in ["processing_frames", "initializing", "reconstructing"]:
        return jsonify({"message": "Job is already actively running.", "job_id": job_id}), 200

    data = request.get_json(silent=True) or {}
    preset = str(data.get("preset", "balanced")).lower()
    strength = float(data.get("strength", 1.0))

    max_seconds = data.get("max_seconds", None)
    if max_seconds is not None:
        try:
            max_seconds = float(max_seconds)
            if max_seconds <= 0:
                max_seconds = None
        except ValueError:
            max_seconds = None

    frame_skip = int(data.get("frame_skip", 1))
    restore_at_original_res = data.get("restore_at_original_res", None)

    # Optional stage overrides if provided
    stage_overrides = None
    if any(k in data for k in ["enable_denoise", "enable_low_light", "enable_super_res", "enable_temporal"]):
        stage_overrides = {
            "enable_denoise": data.get("enable_denoise"),
            "enable_low_light": data.get("enable_low_light"),
            "enable_super_res": data.get("enable_super_res"),
            "enable_temporal": data.get("enable_temporal")
        }

    processing_manager.start_enhancement_async(
        job_id=job_id,
        preset=preset,
        strength=strength,
        max_seconds=max_seconds,
        frame_skip=frame_skip,
        restore_at_original_res=restore_at_original_res,
        stage_overrides=stage_overrides
    )

    return jsonify({
        "success": True,
        "message": f"Enhancement job initiated with preset '{preset}'.",
        "job_id": job_id
    }), 202

@app.route("/api/status/<job_id>", methods=["GET"])
def api_job_status(job_id):
    """Returns real-time processing status, frame counts, FPS, logs, and progress."""
    job = processing_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job.to_dict())

@app.route("/api/cancel/<job_id>", methods=["POST"])
def api_job_cancel(job_id):
    """Requests safe cancellation of an ongoing enhancement job."""
    success = processing_manager.cancel_job(job_id)
    if success:
        return jsonify({"success": True, "message": "Cancellation request submitted."})
    return jsonify({"success": False, "message": "Job cannot be cancelled (not active)."}), 400

@app.route("/api/result/<job_id>", methods=["GET"])
def api_job_result(job_id):
    """Returns final metrics and media download links for a completed job."""
    job = processing_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job.status != "completed":
        return jsonify({"error": f"Job is in state '{job.status}', not completed."}), 400

    return jsonify({
        "job_id": job_id,
        "status": job.status,
        "metrics": job.metrics,
        "logs": job.logs,
        "audio_warning": job.audio_warning,
        "original_url": f"/media/original/{job_id}",
        "enhanced_url": f"/media/enhanced/{job_id}",
        "download_url": f"/api/download/{job_id}",
        "comparison_download_url": f"/api/download/{job_id}?mode=comparison"
    })

@app.route("/api/download/<job_id>", methods=["GET"])
def api_download_video(job_id):
    """Downloads enhanced MP4 or generated side-by-side comparison video."""
    job = processing_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    mode = request.args.get("mode", "enhanced").lower()
    orig_name = Path(job.input_path).stem.replace(f"{job_id}_", "")

    if mode == "comparison":
        comp_file = processing_manager.generate_comparison_clip(job_id)
        if not comp_file or not os.path.exists(comp_file):
            return jsonify({"error": "Failed to generate side-by-side comparison video."}), 500
        return send_file(
            comp_file,
            as_attachment=True,
            download_name=f"DashClarity_Comparison_{orig_name}.mp4",
            mimetype="video/mp4"
        )
    else:
        if not job.output_path or not os.path.exists(job.output_path):
            return jsonify({"error": "Enhanced video file not found."}), 404
        return send_file(
            job.output_path,
            as_attachment=True,
            download_name=f"DashClarity_Enhanced_{orig_name}.mp4",
            mimetype="video/mp4"
        )

@app.route("/api/system/status", methods=["GET"])
def api_system_status():
    """Returns GPU and deep learning runtime status."""
    cuda_avail = torch.cuda.is_available()
    info = {
        "cuda_available": cuda_avail,
        "pytorch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0) if cuda_avail else "CPU",
        "device_count": torch.cuda.device_count() if cuda_avail else 0,
        "allocated_vram_mb": round(torch.cuda.memory_allocated() / (1024**2), 2) if cuda_avail else 0.0,
        "max_vram_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2) if cuda_avail else 0.0,
        "model_architecture": "Modular: SCUNet + Zero-DCE++ + Real-ESRGAN + Farneback",
        "weights_status": "Loaded"
    }
    return jsonify(info)

# -----------------------------------------------------------------------------
# Media Streaming Routes with Partial Content (HTTP 206)
# -----------------------------------------------------------------------------

@app.route("/media/original/<job_id>")
def stream_original_video(job_id):
    """Streams original video with Range request support for HTML5 playback."""
    job = processing_manager.get_job(job_id)
    if not job or not os.path.exists(job.input_path):
        abort(404)
    return send_file(job.input_path, mimetype="video/mp4", conditional=True)

@app.route("/media/enhanced/<job_id>")
def stream_enhanced_video(job_id):
    """Streams enhanced video with Range request support for HTML5 playback."""
    job = processing_manager.get_job(job_id)
    if not job or not job.output_path or not os.path.exists(job.output_path):
        abort(404)
    return send_file(job.output_path, mimetype="video/mp4", conditional=True)

@app.route("/media/thumbnail/<job_id>")
def stream_thumbnail(job_id):
    """Returns video thumbnail JPG."""
    thumb = processing_manager.get_or_create_thumbnail(job_id)
    if not thumb or not os.path.exists(thumb):
        abort(404)
    return send_file(thumb, mimetype="image/jpeg")

# -----------------------------------------------------------------------------
# Application Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(f"Starting DashClarity AI on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)
