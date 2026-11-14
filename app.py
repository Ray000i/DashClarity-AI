"""
DashClarity AI - Flask Web Application
AI-Powered Dashcam Footage Enhancement for Incident Review and Insurance Claims.
Academic End-Semester Generative AI Platform.
"""

import os
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

# -----------------------------------------------------------------------------
# Frontend Page Routes
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    """Landing dashboard page with project overview, statistics, and quick-start CTA."""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "CPU Mode"
    return render_template(
        "index.html",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
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
    """Comparison and forensic analysis results page."""
    job = processing_manager.get_job(job_id)
    if not job or job.status != "completed":
        return redirect(url_for("processing_page", job_id=job_id))
    return render_template(
        "results.html",
        job_id=job_id,
        active_page="results"
    )

@app.route("/about")
def about_page():
    """Academic project documentation, architectural diagrams, and research context."""
    return render_template("about.html", active_page="about")

# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------

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
    Accepts JSON body with optional parameters: max_frames, super_res.
    """
    job = processing_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
        
    if job.status in ["processing_frames", "initializing", "reconstructing"]:
        return jsonify({"message": "Job is already actively running.", "job_id": job_id}), 200
        
    data = request.get_json(silent=True) or {}
    max_frames = data.get("max_frames", None)
    if max_frames is not None:
        try:
            max_frames = int(max_frames)
            if max_frames <= 0:
                max_frames = None
        except ValueError:
            max_frames = None
            
    super_res = bool(data.get("super_res", False))
    
    processing_manager.start_enhancement_async(
        job_id=job_id,
        max_frames=max_frames,
        super_res=super_res
    )
    
    return jsonify({
        "success": True,
        "message": "Enhancement job initiated successfully.",
        "job_id": job_id
    }), 202

@app.route("/api/status/<job_id>", methods=["GET"])
def api_job_status(job_id):
    """Returns actual real-time processing status, frame counts, FPS, and progress."""
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
        "original_url": f"/media/original/{job_id}",
        "enhanced_url": f"/media/enhanced/{job_id}",
        "download_url": f"/api/download/{job_id}"
    })

@app.route("/api/download/<job_id>", methods=["GET"])
def api_download_video(job_id):
    """Downloads the enhanced MP4 video file."""
    job = processing_manager.get_job(job_id)
    if not job or not job.output_path or not os.path.exists(job.output_path):
        return jsonify({"error": "Enhanced video file not found."}), 404
        
    orig_name = Path(job.input_path).stem
    download_filename = f"DashClarity_Enhanced_{orig_name}.mp4"
    return send_file(
        job.output_path,
        as_attachment=True,
        download_name=download_filename,
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
        "model_architecture": "IJCAI STCD (low_light_transformer42)",
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

# -----------------------------------------------------------------------------
# Application Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(f"Starting DashClarity AI on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)
