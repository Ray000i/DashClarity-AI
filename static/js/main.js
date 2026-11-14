/**
 * DashClarity AI - Frontend Client Architecture
 * Controls video upload, real-time job polling, synchronized comparison players,
 * and interactive before/after split slider.
 */

let activeJobId = null;
let pollingTimer = null;

// ============================================================================
// UPLOAD PAGE CONTROLLERS
// ============================================================================

function initUploadPage() {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("video-file-input");
    const startBtn = document.getElementById("btn-start-enhance");

    if (!dropzone || !fileInput) return;

    // Drag and drop event listeners
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    if (startBtn) {
        startBtn.addEventListener('click', () => {
            if (!activeJobId) {
                alert("Please upload and validate a video file first.");
                return;
            }
            startEnhancement(activeJobId);
        });
    }
}

function handleSelectedFile(file) {
    // Validate file size (< 250 MB)
    const maxSize = 250 * 1024 * 1024;
    if (file.size > maxSize) {
        alert("File size exceeds 250 MB limit. Please select a shorter or lower bitrate recording.");
        return;
    }

    // Show upload progress bar
    const progressContainer = document.getElementById("upload-progress-container");
    const progressBar = document.getElementById("upload-progress-bar");
    const percentText = document.getElementById("upload-percent-text");
    const statusText = document.getElementById("upload-status-text");

    progressContainer.classList.remove("hidden");
    progressBar.style.width = "0%";
    percentText.innerText = "0%";
    statusText.innerText = `Uploading ${file.name}...`;

    const formData = new FormData();
    formData.append("video", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload", true);

    xhr.upload.onprogress = function(e) {
        if (e.lengthComputable) {
            const percent = Math.round((e.loaded / e.total) * 100);
            progressBar.style.width = percent + "%";
            percentText.innerText = percent + "%";
        }
    };

    xhr.onload = function() {
        if (xhr.status === 201) {
            const res = JSON.parse(xhr.responseText);
            activeJobId = res.job_id;
            statusText.innerText = "Upload complete! Video validated.";
            displayVideoPreview(file, res.metadata, res.preview_url);
        } else {
            let err = "Upload failed.";
            try {
                const res = JSON.parse(xhr.responseText);
                err = res.error || err;
            } catch(e) {}
            alert("Error: " + err);
            progressContainer.classList.add("hidden");
        }
    };

    xhr.onerror = function() {
        alert("Network error occurred during upload. Check local server connection.");
        progressContainer.classList.add("hidden");
    };

    xhr.send(formData);
}

function displayVideoPreview(file, meta, previewUrl) {
    const previewSection = document.getElementById("preview-section");
    const previewPlayer = document.getElementById("client-preview-player");
    
    // Set video preview source
    previewPlayer.src = previewUrl;
    previewPlayer.load();

    // Populate metadata card
    document.getElementById("meta-filename").innerText = meta.filename;
    document.getElementById("meta-resolution").innerText = meta.resolution;
    document.getElementById("meta-duration").innerText = meta.duration_formatted;
    document.getElementById("meta-fps").innerText = `${meta.fps} FPS`;
    document.getElementById("meta-frames").innerText = meta.total_frames;
    document.getElementById("meta-size").innerText = meta.file_size_formatted;

    previewSection.classList.remove("hidden");
    previewSection.scrollIntoView({ behavior: 'smooth' });
}

function startEnhancement(jobId) {
    const startBtn = document.getElementById("btn-start-enhance");
    startBtn.disabled = true;
    startBtn.innerHTML = `
        <svg class="w-4 h-4 mr-2 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        <span>Queuing Job on GPU...</span>
    `;

    const maxFramesVal = document.getElementById("param-max-frames").value;
    const superResVal = document.getElementById("param-super-res").checked;

    fetch(`/api/enhance/${jobId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            max_frames: parseInt(maxFramesVal),
            super_res: superResVal
        })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success || data.message) {
            window.location.href = `/processing/${jobId}`;
        } else {
            alert("Error initiating enhancement: " + (data.error || "Unknown"));
            startBtn.disabled = false;
        }
    })
    .catch(err => {
        alert("Network error: " + err);
        startBtn.disabled = false;
    });
}

// ============================================================================
// PROCESSING MONITOR CONTROLLERS
// ============================================================================

function startStatusPolling(jobId) {
    activeJobId = jobId;
    
    function poll() {
        fetch(`/api/status/${jobId}`)
            .then(r => r.json())
            .then(data => {
                updateProcessingUI(data);
                
                if (data.status === "completed") {
                    window.location.href = `/results/${jobId}`;
                } else if (data.status === "failed") {
                    showProcessingError(data.error || "Processing failed.");
                } else if (data.status === "cancelled") {
                    showProcessingError("Job cancelled by user.");
                } else {
                    pollingTimer = setTimeout(poll, 1000);
                }
            })
            .catch(err => {
                console.error("Status polling exception:", err);
                pollingTimer = setTimeout(poll, 2000);
            });
    }

    poll();
}

function updateProcessingUI(data) {
    const pBar = document.getElementById("progress-bar");
    const pText = document.getElementById("metric-percent");
    const stageText = document.getElementById("stage-text");
    const statusText = document.getElementById("status-text");
    const framesText = document.getElementById("metric-frames");
    const fpsText = document.getElementById("metric-fps");
    const etaText = document.getElementById("metric-eta");

    if (pBar) pBar.style.width = `${Math.min(100, Math.max(2, data.progress_percent))}%`;
    if (pText) pText.innerText = `${data.progress_percent.toFixed(1)}%`;
    if (stageText) stageText.querySelector("span").innerText = data.stage || "Processing...";
    if (statusText) statusText.innerText = data.status.toUpperCase();
    if (framesText) framesText.innerText = `${data.frames_processed} / ${data.total_frames}`;
    if (fpsText) fpsText.innerText = `${data.current_fps.toFixed(1)} FPS`;
    if (etaText) {
        if (data.eta_seconds > 0) {
            const mins = Math.floor(data.eta_seconds / 60);
            const secs = Math.floor(data.eta_seconds % 60);
            etaText.innerText = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        } else {
            etaText.innerText = "--:--";
        }
    }
}

function showProcessingError(msg) {
    const errBox = document.getElementById("error-box");
    const errMsg = document.getElementById("error-message");
    const statusText = document.getElementById("status-text");
    const statusBadge = document.getElementById("status-badge");

    if (statusText) statusText.innerText = "FAILED";
    if (statusBadge) {
        statusBadge.className = "px-3 py-1 rounded-full text-xs font-mono font-semibold bg-red-950 text-red-400 border border-red-800 flex items-center space-x-1.5";
    }
    if (errMsg) errMsg.innerText = msg;
    if (errBox) errBox.classList.remove("hidden");
}

function cancelProcessing(jobId) {
    if (!confirm("Are you sure you want to stop this AI enhancement job?")) return;

    fetch(`/api/cancel/${jobId}`, { method: "POST" })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                document.getElementById("btn-cancel").disabled = true;
                document.getElementById("btn-cancel").innerText = "Cancelling...";
            }
        });
}

// ============================================================================
// RESULTS & COMPARISON CONTROLLERS
// ============================================================================

let origPlayer = null;
let enhPlayer = null;
let isPlaying = false;

function initResultsPage(jobId) {
    origPlayer = document.getElementById("player-original");
    enhPlayer = document.getElementById("player-enhanced");
    const scrub = document.getElementById("sync-scrub");

    if (origPlayer && enhPlayer) {
        // Synchronize scrub time updates
        enhPlayer.addEventListener("timeupdate", () => {
            if (!scrub) return;
            const progress = (enhPlayer.currentTime / enhPlayer.duration) * 100;
            if (!isNaN(progress)) {
                scrub.value = progress;
                document.getElementById("time-current").innerText = formatTime(enhPlayer.currentTime);
            }
        });

        enhPlayer.addEventListener("loadedmetadata", () => {
            document.getElementById("time-duration").innerText = formatTime(enhPlayer.duration);
        });

        // Scrub bar user input
        if (scrub) {
            scrub.addEventListener("input", (e) => {
                const targetTime = (e.target.value / 100) * enhPlayer.duration;
                if (!isNaN(targetTime)) {
                    origPlayer.currentTime = targetTime;
                    enhPlayer.currentTime = targetTime;
                }
            });
        }
    }

    // Initialize Split Slider
    initSplitSlider();

    // Fetch and display metrics
    fetchMetrics(jobId);
}

function toggleSyncPlay() {
    if (!origPlayer || !enhPlayer) return;
    const playBtn = document.getElementById("btn-sync-play");
    const playLabel = document.getElementById("label-play");

    if (isPlaying) {
        origPlayer.pause();
        enhPlayer.pause();
        isPlaying = false;
        playLabel.innerText = "Play Both";
    } else {
        // Align timestamps before starting
        origPlayer.currentTime = enhPlayer.currentTime;
        origPlayer.play();
        enhPlayer.play();
        isPlaying = true;
        playLabel.innerText = "Pause Both";
    }
}

function restartSyncVideos() {
    if (!origPlayer || !enhPlayer) return;
    origPlayer.currentTime = 0;
    enhPlayer.currentTime = 0;
    if (isPlaying) {
        origPlayer.play();
        enhPlayer.play();
    }
}

function changeSyncSpeed(speed) {
    if (!origPlayer || !enhPlayer) return;
    origPlayer.playbackRate = parseFloat(speed);
    enhPlayer.playbackRate = parseFloat(speed);
}

function formatTime(seconds) {
    if (isNaN(seconds)) return "00:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// Split Slider Controller
function initSplitSlider() {
    const box = document.getElementById("split-slider-box");
    const clip = document.getElementById("split-slider-clip");
    const handle = document.getElementById("split-slider-handle");
    const splitOrig = document.getElementById("split-original-video");
    const splitEnh = document.getElementById("split-enhanced-video");

    if (!box || !clip || !handle) return;

    let isDragging = false;

    function setSliderPosition(x) {
        const rect = box.getBoundingClientRect();
        let posX = x - rect.left;
        posX = Math.max(0, Math.min(posX, rect.width));
        const percent = (posX / rect.width) * 100;
        
        clip.style.width = `${percent}%`;
        handle.style.left = `${percent}%`;
    }

    box.addEventListener("mousedown", (e) => {
        isDragging = true;
        setSliderPosition(e.clientX);
    });

    window.addEventListener("mousemove", (e) => {
        if (!isDragging) return;
        setSliderPosition(e.clientX);
    });

    window.addEventListener("mouseup", () => {
        isDragging = false;
    });

    // Touch support for mobile/tablets
    box.addEventListener("touchstart", (e) => {
        isDragging = true;
        setSliderPosition(e.touches[0].clientX);
    });

    window.addEventListener("touchmove", (e) => {
        if (!isDragging) return;
        setSliderPosition(e.touches[0].clientX);
    });

    window.addEventListener("touchend", () => {
        isDragging = false;
    });

    // Synchronize split video timestamps
    if (splitOrig && splitEnh) {
        splitEnh.addEventListener("timeupdate", () => {
            if (Math.abs(splitOrig.currentTime - splitEnh.currentTime) > 0.1) {
                splitOrig.currentTime = splitEnh.currentTime;
            }
        });
    }
}

function switchViewMode(mode) {
    const sideCont = document.getElementById("container-side-by-side");
    const splitCont = document.getElementById("container-split-slider");
    const sideBtn = document.getElementById("view-mode-side");
    const splitBtn = document.getElementById("view-mode-split");

    if (mode === "side") {
        sideCont.classList.remove("hidden");
        splitCont.classList.add("hidden");
        sideBtn.className = "px-3 py-1.5 rounded-md bg-slate-800 text-cyan-400 font-semibold transition-all";
        splitBtn.className = "px-3 py-1.5 rounded-md text-slate-400 hover:text-white transition-all";
    } else {
        sideCont.classList.add("hidden");
        splitCont.classList.remove("hidden");
        splitBtn.className = "px-3 py-1.5 rounded-md bg-slate-800 text-cyan-400 font-semibold transition-all";
        sideBtn.className = "px-3 py-1.5 rounded-md text-slate-400 hover:text-white transition-all";
    }
}

function fetchMetrics(jobId) {
    fetch(`/api/result/${jobId}`)
        .then(r => r.json())
        .then(data => {
            const m = data.metrics;
            if (!m) return;

            document.getElementById("metrics-loading").classList.add("hidden");
            document.getElementById("metrics-grid").classList.remove("hidden");

            const ill = m.illumination_analysis || {};
            const tech = m.technical_summary || {};
            const gt = m.ground_truth_reference || {};

            // Populate illumination stats
            document.getElementById("metric-lum-gain").innerText = `+${ill.illumination_gain_percent || 0}%`;
            document.getElementById("metric-contrast-idx").innerText = `${ill.contrast_improvement_index || 1.0}x`;
            document.getElementById("metric-temporal-score").innerText = `${ill.temporal_stability_index || 95}/100`;
            document.getElementById("metric-infer-fps").innerText = `${tech.processing_fps || 0} FPS`;

            // Populate video container specs
            document.getElementById("metric-orig-res").innerText = tech.original_resolution || "-";
            document.getElementById("metric-enh-res").innerText = tech.enhanced_resolution || "-";
            document.getElementById("metric-orig-fps").innerText = `${tech.original_fps || 0} FPS`;
            document.getElementById("metric-proc-time").innerText = `${tech.total_processing_time_sec || 0}s`;
            document.getElementById("metric-orig-size").innerText = formatBytes(tech.original_file_size_bytes || 0);
            document.getElementById("metric-enh-size").innerText = formatBytes(tech.enhanced_file_size_bytes || 0);

            // Ground truth note
            if (gt.is_reference_available) {
                document.getElementById("metric-ground-truth-note").innerText = 
                    `Evaluated against ground truth: PSNR: ${gt.psnr_db} dB, SSIM: ${gt.ssim}.`;
            }
        })
        .catch(err => {
            console.error("Failed to load metrics:", err);
        });
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
