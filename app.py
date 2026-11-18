"""
DashClarity AI - Flask Web Application
AI-Powered Dashcam Footage Enhancement for Incident Review and Insurance Claims.
Academic End-Semester Generative AI Platform.
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from flask import (
    Flask, render_template, request, jsonify,
    send_file, redirect, url_for, abort, Response
)
import cv2
import torch

import config
from src.utils import generate_job_id, sanitize_filename, is_allowed_file, ensure_web_compatible_video
from src.metadata import extract_video_metadata
from src.video_processor import ProcessingManager, JobState
from src.models.weights_manager import validate_weights_at_startup

logger = logging.getLogger("DashClarity.App")

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

processing_manager = ProcessingManager.get_instance()

JOB_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{6,64}$")


def is_valid_job_id(job_id: str) -> bool:
    """Validates job ID format against strict alphanumeric pattern to prevent path traversal."""
    return bool(job_id and JOB_ID_REGEX.match(job_id))


@app.context_processor
def inject_system_info():
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "CPU Mode"
    max_size_mb = config.MAX_CONTENT_LENGTH // (1024 * 1024)
    return dict(
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        max_upload_size_mb=max_size_mb
    )


# -----------------------------------------------------------------------------
# Error Handlers
# -----------------------------------------------------------------------------

@app.errorhandler(400)
def bad_request_error(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": str(getattr(e, "description", "Bad Request"))}), 400
    return render_template("404.html", active_page=""), 400


@app.errorhandler(404)
def not_found_error(e):
    if request.path.startswith("/api/") or request.path.startswith("/media/"):
        return jsonify({"error": "Resource not found."}), 404
    return render_template("404.html", active_page=""), 404


@app.errorhandler(413)
def request_entity_too_large(e):
    msg = f"Uploaded file exceeds maximum server limit of {config.MAX_CONTENT_LENGTH // (1024 * 1024)} MB."
    if request.path.startswith("/api/"):
        return jsonify({"error": msg}), 413
    return render_template("500.html", active_page=""), 413


@app.errorhandler(500)
def internal_server_error(e):
    logger.exception("Internal 500 error encountered: %s", e)
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server processing error occurred."}), 500
    return render_template("500.html", active_page=""), 500


# -----------------------------------------------------------------------------
# System Health & Diagnostic Endpoints
# -----------------------------------------------------------------------------

@app.route("/health")
def health_status():
    """Health status endpoint returning GPU and model weights readiness."""
    cuda_avail = torch.cuda.is_available()
    weights_status = validate_weights_at_startup()
    all_weights_ok = all(weights_status.values())
    overall_status = "ok" if (cuda_avail and all_weights_ok) else ("degraded" if cuda_avail or all_weights_ok else "unhealthy")

    return jsonify({
        "status": overall_status,
        "cuda_available": cuda_avail,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_avail else "CPU Mode",
        "weights_ok": all_weights_ok,
        "weights_detail": weights_status
    }), 200


@app.route("/api/system/status", methods=["GET"])
def api_system_status():
    """Returns detailed GPU and deep learning runtime metrics."""
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
# Frontend Page Routes
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    """Landing dashboard page with project overview, statistics, and quick-start CTA."""
    return render_template("index.html", active_page="home")


@app.route("/upload")
def upload_page():
    """Video upload and preview page."""
    return render_template("upload.html", active_page="upload")


@app.route("/processing/<job_id>")
def processing_page(job_id):
    """Real-time processing dashboard for active enhancement jobs."""
    if not is_valid_job_id(job_id):
        abort(400, description="Invalid Job ID format.")

    job = processing_manager.get_job(job_id)
    if not job:
        return redirect(url_for("upload_page"))
    return render_template("processing.html", job_id=job_id, active_page="processing")


@app.route("/results/<job_id>")
def results_page(job_id):
    """Comparison and visual quality results page."""
    if not is_valid_job_id(job_id):
        abort(400, description="Invalid Job ID format.")

    job = processing_manager.get_job(job_id)
    if not job or job.status != "completed":
        return redirect(url_for("processing_page", job_id=job_id))
    return render_template("results.html", job_id=job_id, active_page="results")


@app.route("/history")
def history_page():
    """Past enhancement jobs history page."""
    show_all = request.args.get("show_all", "0").lower() in ("1", "true", "yes")
    jobs = processing_manager.list_history(show_all=show_all)
    return render_template("history.html", active_page="history", jobs=jobs, show_all=show_all)


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
    Returns active pipeline presets dynamically loaded from config.yaml and measured benchmarks.
    """
    import yaml
    cfg_path = config.BASE_DIR / "config.yaml"
    presets_cfg = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            presets_cfg = cfg.get("presets", {})

    bench_path = config.BASE_DIR / "results" / "results.json"
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
    show_all = request.args.get("show_all", "0").lower() in ("1", "true", "yes")
    items = processing_manager.list_history(show_all=show_all)
    return jsonify({"jobs": items, "count": len(items)})


@app.route("/api/logs/<job_id>", methods=["GET"])
def api_job_logs(job_id):
    """Returns real-time execution logs for an active or completed job."""
    if not is_valid_job_id(job_id):
        return jsonify({"error": "Invalid job ID."}), 400

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

    if not is_allowed_file(file.filename, config.ALLOWED_EXTENSIONS):
        return jsonify({
            "error": f"Unsupported format. Permitted video formats: {', '.join(config.ALLOWED_EXTENSIONS).upper()}"
        }), 400

    job_id = generate_job_id()
    safe_name = sanitize_filename(file.filename)
    saved_filename = f"{job_id}_{safe_name}"
    saved_path = config.UPLOAD_DIR / saved_filename

    try:
        file.save(str(saved_path))
        logger.info(f"Uploaded video saved to {saved_path} (Job ID: {job_id})")

        # Extract metadata and validate video container
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

    except (ValueError, cv2.error) as ve:
        logger.warning(f"Upload validation rejected file: {ve}")
        if saved_path.exists():
            try:
                saved_path.unlink()
            except OSError:
                pass
        return jsonify({"error": f"Corrupt or invalid video container: {str(ve)}. Please upload a valid playable video."}), 400
    except Exception as e:
        logger.exception(f"Upload failed: {e}")
        if saved_path.exists():
            try:
                saved_path.unlink()
            except OSError:
                pass
        return jsonify({"error": f"Failed to process uploaded video: {str(e)}"}), 500


@app.route("/api/video/<job_id>", methods=["GET"])
def api_video_info(job_id):
    """Returns metadata for an uploaded video."""
    if not is_valid_job_id(job_id):
        return jsonify({"error": "Invalid job ID."}), 400

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
    Accepts JSON body with modular pipeline parameters.
    """
    if not is_valid_job_id(job_id):
        return jsonify({"error": "Invalid job ID."}), 400

    job = processing_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job.status in ["queued", "processing_frames", "initializing", "reconstructing"]:
        return jsonify({
            "message": "Job is already queued or actively running.",
            "job_id": job_id,
            "status": job.status
        }), 200

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
        "message": f"Enhancement job queued with preset '{preset}'.",
        "job_id": job_id
    }), 202


@app.route("/api/status/<job_id>", methods=["GET"])
def api_job_status(job_id):
    """Returns real-time processing status, frame counts, FPS, logs, and progress."""
    if not is_valid_job_id(job_id):
        return jsonify({"error": "Invalid job ID."}), 400

    job = processing_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job.to_dict())


@app.route("/api/cancel/<job_id>", methods=["POST"])
def api_job_cancel(job_id):
    """Requests safe cancellation of an ongoing enhancement job."""
    if not is_valid_job_id(job_id):
        return jsonify({"error": "Invalid job ID."}), 400

    success = processing_manager.cancel_job(job_id)
    if success:
        return jsonify({"success": True, "message": "Cancellation request submitted."})
    return jsonify({"success": False, "message": "Job cannot be cancelled (not active)."}), 400


@app.route("/api/job/<job_id>", methods=["DELETE"])
def api_delete_job(job_id):
    """Deletes an enhancement job and cleans all associated storage files."""
    if not is_valid_job_id(job_id):
        return jsonify({"error": "Invalid job ID."}), 400

    success = processing_manager.delete_job(job_id)
    if success:
        return jsonify({"success": True, "message": f"Job {job_id} deleted successfully."})
    return jsonify({"error": "Job not found or could not be removed."}), 404


@app.route("/api/result/<job_id>", methods=["GET"])
def api_job_result(job_id):
    """Returns final metrics and media download links for a completed job."""
    if not is_valid_job_id(job_id):
        return jsonify({"error": "Invalid job ID."}), 400

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
    if not is_valid_job_id(job_id):
        abort(400, description="Invalid Job ID format.")

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


# -----------------------------------------------------------------------------
# Media Streaming Routes with Partial Content (HTTP 206) Support
# -----------------------------------------------------------------------------

@app.route("/media/original/<job_id>")
def stream_original_video(job_id):
    """Streams original video with Range request support for HTML5 playback."""
    if not is_valid_job_id(job_id):
        abort(400)

    job = processing_manager.get_job(job_id)
    if not job or not os.path.exists(job.input_path):
        abort(404)
    stream_path = ensure_web_compatible_video(job.input_path)
    response = send_file(stream_path, mimetype="video/mp4", conditional=True)
    response.headers["Accept-Ranges"] = "bytes"
    return response


@app.route("/media/enhanced/<job_id>")
def stream_enhanced_video(job_id):
    """Streams enhanced video with Range request support for HTML5 playback."""
    if not is_valid_job_id(job_id):
        abort(400)

    job = processing_manager.get_job(job_id)
    if not job or not job.output_path or not os.path.exists(job.output_path):
        abort(404)
    response = send_file(job.output_path, mimetype="video/mp4", conditional=True)
    response.headers["Accept-Ranges"] = "bytes"
    return response


@app.route("/media/thumbnail/<job_id>")
def stream_thumbnail(job_id):
    """Returns video thumbnail JPG."""
    if not is_valid_job_id(job_id):
        abort(400)

    thumb = processing_manager.get_or_create_thumbnail(job_id)
    if not thumb or not os.path.exists(thumb):
        abort(404)
    return send_file(thumb, mimetype="image/jpeg")


# -----------------------------------------------------------------------------
# Application Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    validate_weights_at_startup()
    if not torch.cuda.is_available():
        logger.warning("=" * 70)
        logger.warning("[CPU MODE NOTICE] NVIDIA CUDA is unavailable. Processing will execute on CPU.")
        logger.warning("=" * 70)

    logger.info("Starting DashClarity AI on http://%s:%d (debug=%s)", config.HOST, config.PORT, config.DEBUG)
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
