# DashClarity AI: End-Semester Project Defense Presentation
**10-Slide Presentation Script & Technical Defense Guide**  
**Total Target Duration:** 12–15 Minutes (followed by Q&A)

---

### Slide 1: Title & Project Identity (1 Min)
- **Title:** DashClarity AI — AI-Powered Dashcam Video Enhancement for Incident Review and Insurance Claims
- **Subtitle:** An Adaptive Deep Learning Pipeline for Low-Light, Specular Glare, and Motion-Stabilized Video Restoration
- **Speaker:** ML Engineering Team
- **Key Talking Points:**
  - Modern vehicle fleets and insurance claims adjusters rely on dashcams for dispute resolution and incident liability.
  - Dashcam recordings consistently suffer from low illumination, severe CMOS sensor noise, oncoming headlight glare, and motion blur.
  - DashClarity AI is a demo-ready, modular Generative AI platform engineered to restore forensic detail from degraded footage.

---

### Slide 2: Real-World Problem & Industrial Motivation (1.5 Mins)
- **Visuals:** Side-by-side crop showing degraded night dashcam (license plate obscured by dark asphalt & noise) vs enhanced clear view.
- **Key Talking Points:**
  - **The Three Core Nighttime Hazards:**
    1. *Extreme Photon Starvation & Sensor Read Noise:* High-ISO gain produces boiling chroma noise.
    2. *High Dynamic Range Specular Glare:* Oncoming Xenon/LED headlights blind dashcam CMOS sensors.
    3. *High-Speed Ego-Motion Blur:* Road texture and passing vehicles smear across exposure times.
  - **Why Existing Commercial Solutions Fail:** Simple gamma boosts brighten dark skies into noisy gray haze and blow out headlights into huge white blobs without recovering structural contrast.

---

### Slide 3: Critique of Prior Art & STCD Transformer Limitations (1.5 Mins)
- **Visuals:** Comparison crop showing STCD indoor success vs STCD automotive failure (color boiling on road).
- **Key Talking Points:**
  - In our initial phase, we integrated STCD (Spatial-Temporal Cross-Frame Disentanglement, IJCAI 2025).
  - *The Failure Mode:* STCD was trained on static indoor scenes (SDSD dataset). When deployed on open highways, its cross-frame attention caused severe temporal artifacts and motion smearing on asphalt.
  - *Engineering Decision:* We retained STCD strictly as an ablation comparison and re-architected the system into a robust, domain-specific modular pipeline.

---

### Slide 4: Architectural Blueprint & Theoretical Restoration Order (2 Mins)
- **Visuals:** Modular pipeline flowchart:
  $$\text{Input Frame} \longrightarrow \boxed{\text{SCUNet Denoiser}} \longrightarrow \boxed{\text{Zero-DCE++ Illumination}} \longrightarrow \boxed{\text{Real-ESRGAN SR}} \longrightarrow \boxed{\text{Farneback Motion Flow}} \longrightarrow \text{Output}$$
- **Key Talking Points:**
  - **Why Denoise First?** Quadratic curve estimation exponentially amplifies noise: $\frac{\partial I_1}{\partial \epsilon} \approx 2.0$. By cleaning sensor noise with SCUNet first, Zero-DCE++ estimates clean parameter maps.
  - Verified by our ablation study: `SCUNet -> Zero-DCE++` achieves +1.40 dB higher PSNR, +0.1322 higher SSIM, and 14.1% lower perceptual error (LPIPS) than reversing the order.

---

### Slide 5: Photometric Safeguards & Anti-Blooming Controls (1.5 Mins)
- **Visuals:** Before/after crops of headlight knee compression and dark sky black-level preservation.
- **Key Talking Points:**
  - **Toe Gain Mask:** Protects dark pixels ($I \le 10$). Eliminates the artificial blue-gray haze seen in naive low-light algorithms.
  - **Highlight Knee Compression:** Attenuates gain on intensities $> 242$. Prevents oncoming headlights from blooming and destroying vehicle contours.
  - **Organic Texture Retention:** Blends 15% sensor detail with 85% neural denoiser to prevent "plastic-smoothing" on foliage and asphalt.

---

### Slide 6: Motion-Compensated Temporal Consistency (1.5 Mins)
- **Visuals:** Diagram showing camera motion vector field $\mathbf{u}(x, y)$ vs static EMA ghost trail.
- **Key Talking Points:**
  - Standard temporal filters (EMA) produce ghostly translucent trails behind moving cars and road markings.
  - DashClarity AI computes Farneback dense optical flow and forward-warps previous enhanced frames along the camera trajectory.
  - Gated by a photometric error confidence mask: if optical flow fails (e.g. rapid disocclusion), temporal smoothing drops to 0, completely preventing ghosting.

---

### Slide 7: Quantitative Evaluation & Ablation Frontier (2 Mins)
- **Visuals:** 4-panel ablation bar chart + Speed-vs-Quality Pareto frontier plot.
- **Key Talking Points:**
  - **Synthetic Ablation (50 Paired Frames across 5 Clips):**
    - LPIPS drops from **0.9012** (degraded input) down to **0.6596** (full pipeline), proving strong perceptual clarity gains.
    - Zero-DCE++ boosts illumination with 35.7 FPS throughput in fp16.
  - **Speed-vs-Quality Pareto Trade-Off:**
    - `fast` preset: **31.5 FPS** (Zero-DCE++ + light denoise).
    - `balanced` preset: **3.4 FPS** (Half-res neural SCUNet + Zero-DCE++ + Farneback flow).
    - `quality` preset: **0.6 FPS** (Full 512-tile SCUNet + Real-ESRGAN + Farneback flow).

---

### Slide 8: Real-World No-Reference Benchmark (1.5 Mins)
- **Visuals:** Table of authentic real night clips (NIQE, BRISQUE, Normalized Noise $\sigma/\mu$, Temporal Flicker).
- **Key Talking Points:**
  - Evaluated on 5 authentic real nighttime dashcam videos from the Roadscapes Video Dataset (CC-BY-4.0).
  - *No Fake Metrics:* We do not compute full-reference PSNR/SSIM where no ground-truth exists.
  - Fast preset achieves the best naturalness (lowest NIQE: **6.27**).
  - Balanced preset delivers the best temporal stability (lowest flicker: **6.44**) and lowest normalized noise (**0.0021**).

---

### Slide 9: System Engineering & Web Application Demo (2 Mins)
- **Visuals:** Screenshot / Live demonstration of Flask web interface (`/upload`, `/processing`, `/results`).
- **Key Talking Points:**
  - **Memory Efficiency:** Streaming pipeline pipes frames directly to FFmpeg stdin; holds at most 5 frames in RAM ($< 300\text{ MB}$ memory footprint vs 15 GB naive buffer).
  - **Hardware Reliability:** Tiled tensor inference with automated CUDA OOM recovery guarantees peak VRAM $< 2.4\text{ GB}$ on RTX 4060 Laptop GPU.
  - **Full Container Support:** Preserves original AAC/MP3 audio via FFmpeg demux/remux with zero sync drift.
  - **Intuitive UI:** Interactive slider, live FPS/ETA telemetry, and synchronized split-screen comparison player.

---

### Slide 10: Conclusion & Future Scope (1 Min)
- **Key Talking Points:**
  - DashClarity AI successfully bridges academic deep learning models with real-world automotive video constraints.
  - Demonstrates that informed domain-specific stage ordering and photometric constraints outperform end-to-end black-box networks.
  - **Future Work:**
    - Edge deployment on NVIDIA Jetson Orin via TensorRT fp16/int8 engine export.
    - Integration of downstream incident intelligence: automatic license plate recognition (ALPR) and multi-object tracking (YOLOv11 + ByteTrack) on restored frames.
