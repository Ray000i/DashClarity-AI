/**
 * DashClarity AI - Video Review Suite Client Controller
 * Neutral surround palette, generic metrics schema rendering,
 * synchronized dual-video transport, frame-stepping, and 3x detail loupe.
 */

let activeJobId = null;
let pollingTimer = null;
let currentViewMode = "split"; // "split" or "side"
let loupeActive = false;
let loadedPresetsMap = {};

// Transport Player References
let origPlayer = null;
let enhPlayer = null;
let isPlaying = false;

// ============================================================================
// UPLOAD PAGE CONTROLLER
// ============================================================================

function initUploadPage() {
    loadPresets();

    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("video-file-input");
    const startBtn = document.getElementById("btn-start-enhance");

    if (!dropzone || !fileInput) return;

    ['dragenter', 'dragover'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        }, false);
    });

    ['dragleave', 'drop'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
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
                alert("Please select and upload a video file first.");
                return;
            }
            startEnhancement(activeJobId);
        });
    }
}

function renderPresetCard(p, isSelected) {
    const fpsLabel = p.measured_fps_720p ? `${p.measured_fps_720p} FPS` : "Custom";
    return `
        <div class="preset-card ${isSelected ? 'selected' : ''}" onclick="selectPreset('${p.key}', this)">
            <input type="radio" name="preset-radio" value="${p.key}" class="preset-radio" ${isSelected ? 'checked' : ''}>
            <div style="flex: 1; min-width: 0;">
                <div class="preset-header">
                    <span class="preset-name">${escapeHtml(p.label)}</span>
                    <span class="preset-fps">${fpsLabel}</span>
                </div>
                <div class="preset-tradeoff">${escapeHtml(p.tradeoff)}</div>
            </div>
        </div>
    `;
}

function loadPresets() {
    const container = document.getElementById("presets-container");
    if (!container) return;

    fetch("/api/presets")
        .then(r => r.json())
        .then(data => {
            const presets = data.presets || [];
            if (presets.length === 0) {
                container.innerHTML = `<div style="padding: 0.5rem; color: var(--text-muted);">No presets found.</div>`;
                return;
            }

            loadedPresetsMap = {};
            presets.forEach(p => { loadedPresetsMap[p.key] = p; });

            const recommendedKeys = ["fast", "balanced", "quality"];
            const recommended = presets.filter(p => recommendedKeys.includes(p.key));
            const ablation = presets.filter(p => !recommendedKeys.includes(p.key));

            let html = "";
            let selectedPresetKey = null;
            // Render Recommended presets
            recommended.forEach((p, index) => {
                const isSelected = p.key === "balanced" || (index === 0 && !presets.some(x => x.key === "balanced"));
                if (isSelected) selectedPresetKey = p.key;
                html += renderPresetCard(p, isSelected);
            });

            // Render Research / ablation presets in collapsed details
            if (ablation.length > 0) {
                html += `
                    <details class="presets-details">
                        <summary class="presets-summary">
                            <span>Research / ablation presets (STCD, Hybrid, Classical)</span>
                            <span style="font-size: 0.72rem; color: var(--accent-amber);">&#9660;</span>
                        </summary>
                        <div class="presets-content">
                            ${ablation.map(p => renderPresetCard(p, false)).join("")}
                        </div>
                    </details>
                `;
            }

            container.innerHTML = html;
            if (selectedPresetKey) syncStageCheckboxes(selectedPresetKey);
        })
        .catch(err => {
            console.error("Failed to load presets:", err);
            container.innerHTML = `<div style="padding: 0.5rem; color: var(--state-alert);">Could not load presets from configuration.</div>`;
        });
}

function syncStageCheckboxes(presetKey) {
    const p = loadedPresetsMap[presetKey];
    if (!p) return;
    const cbDenoise = document.getElementById("toggle-stage-denoise");
    const cbLowLight = document.getElementById("toggle-stage-lowlight");
    const cbSR = document.getElementById("toggle-stage-sr");
    const cbTemporal = document.getElementById("toggle-stage-temporal");

    if (cbDenoise) cbDenoise.checked = Boolean(p.enable_neural_denoise || p.enable_classical_denoise);
    if (cbLowLight) cbLowLight.checked = Boolean(p.enable_low_light);
    if (cbSR) cbSR.checked = Boolean(p.enable_super_res);
    if (cbTemporal) cbTemporal.checked = Boolean(p.enable_temporal);
}

function selectPreset(presetKey, elem) {
    document.querySelectorAll(".preset-card").forEach(c => c.classList.remove("selected"));
    elem.classList.add("selected");
    const radio = elem.querySelector("input[type='radio']");
    if (radio) radio.checked = true;

    const input = document.getElementById("param-preset");
    if (input) input.value = presetKey;
    syncStageCheckboxes(presetKey);
}

function handleSelectedFile(file) {
    const progressContainer = document.getElementById("upload-progress-container");
    const progressBar = document.getElementById("upload-progress-bar");
    const percentText = document.getElementById("upload-percent-text");
    const statusText = document.getElementById("upload-status-text");

    if (progressContainer) progressContainer.classList.remove("hidden");
    if (progressBar) progressBar.style.width = "0%";
    if (percentText) percentText.innerText = "0%";
    if (statusText) statusText.innerText = `Uploading ${file.name}...`;

    const formData = new FormData();
    formData.append("video", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload", true);

    xhr.upload.onprogress = function(e) {
        if (e.lengthComputable && progressBar && percentText) {
            const percent = Math.round((e.loaded / e.total) * 100);
            progressBar.style.width = percent + "%";
            percentText.innerText = percent + "%";
        }
    };

    xhr.onload = function() {
        if (xhr.status === 201) {
            const res = JSON.parse(xhr.responseText);
            activeJobId = res.job_id;
            if (statusText) statusText.innerText = "Upload complete. Video container validated.";
            displayVideoPreview(file, res.metadata, res.preview_url);
            const startBtn = document.getElementById("btn-start-enhance");
            if (startBtn) startBtn.disabled = false;
            const hint = document.getElementById("enhance-hint");
            if (hint) hint.classList.add("hidden");
        } else {
            let err = "Upload failed.";
            try {
                const res = JSON.parse(xhr.responseText);
                err = res.error || err;
            } catch(e) {}
            alert("Error: " + err);
            if (progressContainer) progressContainer.classList.add("hidden");
        }
    };

    xhr.onerror = function() {
        alert("Network error occurred during upload. Check server connection.");
        if (progressContainer) progressContainer.classList.add("hidden");
    };

    xhr.send(formData);
}

function displayVideoPreview(file, meta, previewUrl) {
    const previewSection = document.getElementById("preview-section");
    const previewPlayer = document.getElementById("client-preview-player");

    if (previewPlayer) {
        previewPlayer.src = previewUrl;
        previewPlayer.load();
    }

    const fnEl = document.getElementById("meta-filename");
    if (fnEl) fnEl.innerText = meta.filename || file.name;

    const resEl = document.getElementById("meta-resolution");
    if (resEl) resEl.innerText = meta.resolution || "--";

    const durFpsEl = document.getElementById("meta-duration-fps");
    if (durFpsEl) durFpsEl.innerText = `${meta.duration_formatted} @ ${meta.fps} FPS`;

    const framesEl = document.getElementById("meta-frames");
    if (framesEl) framesEl.innerText = `${meta.total_frames} frames (${meta.file_size_formatted})`;

    if (previewSection) {
        previewSection.classList.remove("hidden");
        previewSection.scrollIntoView({ behavior: 'smooth' });
    }
}

function startEnhancement(jobId) {
    const startBtn = document.getElementById("btn-start-enhance");
    if (startBtn) {
        startBtn.disabled = true;
        startBtn.innerText = "Starting pipeline...";
    }

    const presetVal = document.getElementById("param-preset")?.value || "balanced";
    const strengthVal = parseFloat(document.getElementById("param-strength")?.value || 0.85);
    const maxSecVal = parseFloat(document.getElementById("param-max-seconds")?.value || 10.0);

    const toggleDenoise = document.getElementById("toggle-stage-denoise")?.checked;
    const toggleLowLight = document.getElementById("toggle-stage-lowlight")?.checked;
    const toggleSR = document.getElementById("toggle-stage-sr")?.checked;
    const toggleTemporal = document.getElementById("toggle-stage-temporal")?.checked;

    fetch(`/api/enhance/${jobId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            preset: presetVal,
            strength: strengthVal,
            max_seconds: maxSecVal > 0 ? maxSecVal : null,
            enable_denoise: toggleDenoise,
            enable_low_light: toggleLowLight,
            enable_super_res: toggleSR,
            enable_temporal: toggleTemporal
        })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success || data.message) {
            window.location.href = `/processing/${jobId}`;
        } else {
            alert("Error initiating enhancement: " + (data.error || "Unknown"));
            if (startBtn) startBtn.disabled = false;
        }
    })
    .catch(err => {
        alert("Network error: " + err);
        if (startBtn) startBtn.disabled = false;
    });
}

// ============================================================================
// PROCESSING MONITOR CONTROLLER
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
                    pollingTimer = setTimeout(poll, 700);
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
    const stageText = document.getElementById("stage-text");
    const statusText = document.getElementById("status-text");
    const framesText = document.getElementById("metric-frames");
    const fpsText = document.getElementById("metric-fps");
    const etaText = document.getElementById("metric-eta");
    const elapsedText = document.getElementById("metric-elapsed");
    const logDrawer = document.getElementById("log-drawer");
    const logCount = document.getElementById("log-count");

    if (pBar) pBar.style.width = `${Math.min(100, Math.max(3, data.progress_percent))}%`;
    if (stageText) {
        const span = stageText.querySelector("span");
        if (span) span.innerText = data.stage || "Processing frames...";
    }
    if (statusText) statusText.innerText = data.status;
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

    if (elapsedText && data.elapsed_seconds > 0) {
        const eMins = Math.floor(data.elapsed_seconds / 60);
        const eSecs = Math.floor(data.elapsed_seconds % 60);
        elapsedText.innerText = `${eMins.toString().padStart(2, '0')}:${eSecs.toString().padStart(2, '0')}`;
    }

    // Render Real Logs
    if (logDrawer && data.logs && data.logs.length > 0) {
        if (logCount) logCount.innerText = `${data.logs.length} events`;
        logDrawer.innerHTML = data.logs.map(l => `<div class="log-entry">${escapeHtml(l)}</div>`).join("");
        logDrawer.scrollTop = logDrawer.scrollHeight;
    }
}

function showProcessingError(msg) {
    const errBox = document.getElementById("error-box");
    const errMsg = document.getElementById("error-message");
    const statusText = document.getElementById("status-text");
    const statusDot = document.getElementById("status-dot");

    if (statusText) statusText.innerText = "Failed";
    if (statusDot) statusDot.style.backgroundColor = "var(--state-alert)";
    if (errMsg) errMsg.innerText = msg;
    if (errBox) errBox.classList.remove("hidden");
}

function cancelProcessing(jobId) {
    if (!confirm("Are you sure you want to cancel video enhancement?")) return;

    fetch(`/api/cancel/${jobId}`, { method: "POST" })
        .then(r => r.json())
        .then(data => {
            const btn = document.getElementById("btn-cancel");
            if (btn) {
                btn.disabled = true;
                btn.innerText = "Cancelling...";
            }
        });
}

// ============================================================================
// RESULTS & COMPARISON CONTROLLER
// ============================================================================

function initResultsPage(jobId) {
    activeJobId = jobId;
    origPlayer = document.getElementById("player-original");
    enhPlayer = document.getElementById("player-enhanced");

    initSyncPlayback();
    initSplitSlider();
    initLoupe();
    setupKeyboardShortcuts();
    fetchMetrics(jobId);

    const params = new URLSearchParams(window.location.search);
    if (params.get("mode") === "side") {
        switchViewMode("side");
    }
}

function initSyncPlayback() {
    if (!enhPlayer || !origPlayer) return;

    const scrub = document.getElementById("sync-scrub");
    const cur = document.getElementById("time-current");
    const dur = document.getElementById("time-duration");

    enhPlayer.addEventListener("loadedmetadata", () => {
        if (dur) dur.innerText = formatTimeMs(enhPlayer.duration);
        if (origPlayer) origPlayer.currentTime = enhPlayer.currentTime;
    });

    enhPlayer.addEventListener("timeupdate", () => {
        if (!scrub) return;
        const maxDur = enhPlayer.duration || 1;
        const progress = (enhPlayer.currentTime / maxDur) * 100;
        if (!isNaN(progress)) {
            scrub.value = progress;
            if (cur) cur.innerText = formatTimeMs(enhPlayer.currentTime);
        }
        // Strict lock: enforce original video tracks enhanced video time exactly
        if (origPlayer && Math.abs(origPlayer.currentTime - enhPlayer.currentTime) > 0.04) {
            origPlayer.currentTime = enhPlayer.currentTime;
        }
    });

    // Guard: Prevent raw player from overrunning when enhanced video finishes
    enhPlayer.addEventListener("ended", () => {
        if (origPlayer) {
            origPlayer.pause();
            origPlayer.currentTime = enhPlayer.duration;
        }
        isPlaying = false;
        const playLabel = document.getElementById("label-play");
        if (playLabel) playLabel.innerHTML = "&#9658; Play";
    });

    // Guard: Prevent raw video from advancing beyond the capped enhanced duration
    origPlayer.addEventListener("timeupdate", () => {
        if (enhPlayer && enhPlayer.duration && origPlayer.currentTime > enhPlayer.duration) {
            origPlayer.pause();
            origPlayer.currentTime = enhPlayer.duration;
        }
    });

    if (scrub) {
        scrub.addEventListener("input", (e) => {
            const maxDur = enhPlayer.duration || 1;
            const targetTime = (e.target.value / 100) * maxDur;
            if (!isNaN(targetTime)) {
                if (origPlayer) origPlayer.currentTime = targetTime;
                enhPlayer.currentTime = targetTime;
            }
        });
    }
}

function toggleSyncPlay() {
    if (!enhPlayer || !origPlayer) return;
    const playLabel = document.getElementById("label-play");

    if (isPlaying) {
        origPlayer.pause();
        enhPlayer.pause();
        isPlaying = false;
        if (playLabel) playLabel.innerHTML = "&#9658; Play";
    } else {
        // Auto-rewind if at end
        if (enhPlayer.duration && enhPlayer.currentTime >= (enhPlayer.duration - 0.1)) {
            enhPlayer.currentTime = 0;
            origPlayer.currentTime = 0;
        }
        origPlayer.currentTime = enhPlayer.currentTime;
        const p1 = origPlayer.play().catch(e => console.warn("Original video playback notice:", e));
        const p2 = enhPlayer.play().catch(e => console.warn("Enhanced video playback notice:", e));
        Promise.all([p1, p2]).then(() => {
            isPlaying = true;
            if (playLabel) playLabel.innerHTML = "&#10074;&#10074; Pause";
        });
    }
}

function stepFrame(deltaFrames) {
    if (!enhPlayer || !origPlayer) return;
    const frameDuration = 1.0 / 30.0;
    const target = Math.max(0, Math.min(enhPlayer.duration, enhPlayer.currentTime + (deltaFrames * frameDuration)));

    origPlayer.pause();
    enhPlayer.pause();
    origPlayer.currentTime = target;
    enhPlayer.currentTime = target;
    isPlaying = false;
    const playLabel = document.getElementById("label-play");
    if (playLabel) playLabel.innerHTML = "&#9658; Play";
}

function changeSyncSpeed(speed) {
    const rate = parseFloat(speed);
    if (origPlayer) origPlayer.playbackRate = rate;
    if (enhPlayer) enhPlayer.playbackRate = rate;
}

function formatTimeMs(seconds) {
    if (isNaN(seconds)) return "00:00.00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
}

// Split Slider Controller
function initSplitSlider() {
    const canvas = document.getElementById("viewport-canvas");
    if (!canvas) return;

    let isDragging = false;

    function setSliderPosition(clientX) {
        if (currentViewMode !== "split") return;
        const rect = canvas.getBoundingClientRect();
        let posX = clientX - rect.left;
        posX = Math.max(0, Math.min(posX, rect.width));
        const percent = (posX / rect.width) * 100;
        canvas.style.setProperty("--split-pos", `${percent}%`);
    }

    canvas.addEventListener("mousedown", (e) => {
        if (currentViewMode !== "split") return;
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

    canvas.addEventListener("touchstart", (e) => {
        if (currentViewMode !== "split") return;
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
}

function switchViewMode(mode) {
    currentViewMode = mode;
    const canvas = document.getElementById("viewport-canvas");
    const sideBtn = document.getElementById("view-mode-side");
    const splitBtn = document.getElementById("view-mode-split");

    if (!canvas) return;

    if (mode === "side") {
        canvas.classList.remove("mode-split");
        canvas.classList.add("mode-side");
        if (sideBtn) sideBtn.classList.add("active");
        if (splitBtn) splitBtn.classList.remove("active");
    } else {
        canvas.classList.remove("mode-side");
        canvas.classList.add("mode-split");
        if (splitBtn) splitBtn.classList.add("active");
        if (sideBtn) sideBtn.classList.remove("active");
    }
}

// 3x Detail Loupe Magnifier Controller
function initLoupe() {
    const box = document.getElementById("viewport-canvas");
    const lens = document.getElementById("loupe-lens");
    const canvas = document.getElementById("loupe-canvas");
    if (!box || !lens || !canvas) return;

    const ctx = canvas.getContext("2d");
    canvas.width = 160;
    canvas.height = 160;

    box.addEventListener("mousemove", (e) => {
        if (!loupeActive) return;
        const rect = box.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        lens.style.display = "block";
        lens.style.left = `${mouseX - 80}px`;
        lens.style.top = `${mouseY - 80}px`;

        let activeVid = enhPlayer;
        let relX = mouseX;
        let vidWidth = rect.width;

        if (currentViewMode === "split") {
            const splitPosPercent = parseFloat(getComputedStyle(box).getPropertyValue("--split-pos")) || 50;
            const splitX = (splitPosPercent / 100) * rect.width;
            activeVid = (mouseX < splitX) ? origPlayer : enhPlayer;
        } else {
            vidWidth = rect.width / 2;
            if (mouseX < vidWidth) {
                activeVid = origPlayer;
                relX = mouseX;
            } else {
                activeVid = enhPlayer;
                relX = mouseX - vidWidth;
            }
        }

        if (!activeVid || activeVid.videoWidth === 0) return;

        const scaleX = activeVid.videoWidth / vidWidth;
        const scaleY = activeVid.videoHeight / rect.height;

        const srcW = (160 / 3) * scaleX;
        const srcH = (160 / 3) * scaleY;
        const srcX = (relX * scaleX) - (srcW / 2);
        const srcY = (mouseY * scaleY) - (srcH / 2);

        ctx.clearRect(0, 0, 160, 160);
        try {
            ctx.drawImage(activeVid, srcX, srcY, srcW, srcH, 0, 0, 160, 160);
        } catch(err) {}
    });

    box.addEventListener("mouseleave", () => {
        if (lens) lens.style.display = "none";
    });
}

function toggleLoupe() {
    loupeActive = !loupeActive;
    const btn = document.getElementById("btn-toggle-loupe");
    const lens = document.getElementById("loupe-lens");

    if (btn) {
        if (loupeActive) {
            btn.classList.add("active");
        } else {
            btn.classList.remove("active");
            if (lens) lens.style.display = "none";
        }
    }
}

// Keyboard Shortcuts Listener
function setupKeyboardShortcuts() {
    window.addEventListener("keydown", (e) => {
        // Do not intercept if user is typing in an input field
        if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;

        if (e.code === "Space") {
            e.preventDefault();
            toggleSyncPlay();
        } else if (e.code === "ArrowLeft") {
            e.preventDefault();
            stepFrame(-1);
        } else if (e.code === "ArrowRight") {
            e.preventDefault();
            stepFrame(1);
        } else if (e.code === "KeyS") {
            e.preventDefault();
            switchViewMode(currentViewMode === "split" ? "side" : "split");
        } else if (e.code === "KeyL") {
            e.preventDefault();
            toggleLoupe();
        }
    });
}

function toggleShortcutsModal() {
    const modal = document.getElementById("shortcuts-modal");
    if (modal) {
        modal.classList.toggle("hidden");
    }
}

// Generic Schema-Driven Metrics Fetcher
function fetchMetrics(jobId) {
    fetch(`/api/result/${jobId}`)
        .then(r => r.json())
        .then(data => {
            if (data.audio_warning) {
                const b = document.getElementById("audio-warning-banner");
                const bt = document.getElementById("audio-warning-text");
                if (b && bt) {
                    bt.innerText = data.audio_warning;
                    b.classList.remove("hidden");
                }
            }

            const m = data.metrics || {};
            const tech = m.technical_summary || {};

            if (tech.device_mode && tech.device_mode.includes("CPU")) {
                const cpuB = document.getElementById("cpu-warning-banner");
                if (cpuB) cpuB.classList.remove("hidden");
            }

            renderGenericMetrics(m.schema_metrics);
        })
        .catch(err => {
            console.error("Failed to load metrics:", err);
            const grid = document.getElementById("metrics-grid");
            if (grid) {
                grid.innerHTML = `<div style="grid-column: 1 / -1; padding: 1.5rem; color: var(--state-alert);" class="card">Failed to load metrics from server.</div>`;
            }
        });
}

function renderGenericMetrics(schemaMetrics) {
    const grid = document.getElementById("metrics-grid");
    if (!grid) return;

    if (!schemaMetrics || schemaMetrics.length === 0) {
        grid.innerHTML = `<div style="grid-column: 1 / -1; padding: 1.5rem; color: var(--text-muted);" class="card">No structured metrics available for this job.</div>`;
        return;
    }

    let html = "";
    schemaMetrics.forEach(item => {
        const inVal = (item.input !== null && item.input !== undefined && item.input !== "not available") ? item.input : "not available";
        const outVal = (item.output !== null && item.output !== undefined && item.output !== "not available") ? item.output : "not available";
        const changeVal = (item.change !== null && item.change !== undefined && item.change !== "not available") ? item.change : "";
        const evalWord = item.evaluation || "neutral";
        const unitSuffix = item.unit ? ` ${item.unit}` : "";

        html += `
            <div class="metric-card">
                <div class="metric-header">
                    <span class="metric-label">${escapeHtml(item.label)}</span>
                    <span class="metric-eval">${escapeHtml(evalWord)}</span>
                </div>
                <div class="metric-values">
                    <span class="metric-main-val tabular">${outVal !== "not available" ? outVal + unitSuffix : "not available"}</span>
                    ${changeVal && changeVal !== "not available" ? `<span class="metric-diff tabular">(${changeVal})</span>` : ''}
                </div>
                <div style="font-size: 0.72rem; color: var(--text-dim); margin-top: 0.25rem;" class="tabular">
                    Input: ${inVal !== "not available" ? inVal + unitSuffix : "not available"}
                </div>
                ${item.caveat ? `<div class="metric-caveat">${escapeHtml(item.caveat)}</div>` : ''}
            </div>
        `;
    });

    grid.innerHTML = html;
}

function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// ============================================================================
// HISTORY PAGE CONTROLLER
// ============================================================================

function initHistoryPage() {
    document.querySelectorAll(".history-fps[data-job-id]").forEach(el => {
        const jobId = el.getAttribute("data-job-id");
        if (!jobId) return;
        fetch(`/api/result/${jobId}`)
            .then(r => r.json())
            .then(data => {
                const fps = data?.metrics?.technical_summary?.processing_fps;
                if (typeof fps === "number" && !isNaN(fps)) {
                    el.innerText = `${fps.toFixed(2)} FPS`;
                }
            })
            .catch(() => {});
    });
}
