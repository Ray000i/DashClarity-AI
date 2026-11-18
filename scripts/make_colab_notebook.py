import json
from pathlib import Path

notebook_dir = Path("notebooks")
notebook_dir.mkdir(parents=True, exist_ok=True)

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# DashClarity AI: AI-Powered Dashcam Video Enhancement\n",
                "### Academic End-Semester Project Demo Notebook\n",
                "**Hardware Acceleration:** Connect to a GPU runtime (Runtime -> Change runtime type -> T4 GPU or higher)\n",
                "\n",
                "This notebook allows interactive evaluation of DashClarity AI on dashcam footage."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 1: Install Dependencies and Verify Hardware"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install required dependencies\n",
                "!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n",
                "!pip install -q opencv-python-headless spandrel pyiqa scikit-image pyyaml\n",
                "!apt-get update -qq && apt-get install -y -qq ffmpeg\n",
                "\n",
                "import torch\n",
                "print('CUDA Available:', torch.cuda.is_available())\n",
                "if torch.cuda.is_available():\n",
                "    print('Device Name:', torch.cuda.get_device_name(0))\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 2: Initialize Modular Enhancement Pipeline"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys, os\n",
                "from pathlib import Path\n",
                "\n",
                "from src.pipeline.enhancer import VideoEnhancementPipeline\n",
                "from src.pipeline.video_stream import StreamingVideoProcessor\n",
                "from src.utils.metrics import evaluate_video_metrics\n",
                "\n",
                "pipeline = VideoEnhancementPipeline()\n",
                "print('Pipeline initialized successfully on:', pipeline.device)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 3: Run Enhancement on Sample Dashcam Footage"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "input_clip = 'input_videos/real/real_night_dashcam_01.mp4'\n",
                "output_clip = 'output_videos/colab_enhanced_demo.mp4'\n",
                "selected_preset = 'balanced'  # Options: fast, balanced, quality\n",
                "\n",
                "print(f'Processing {input_clip} with preset: {selected_preset}')\n",
                "pipeline.reset_temporal()\n",
                "streamer = StreamingVideoProcessor(\n",
                "    input_path=input_clip,\n",
                "    output_path=output_clip,\n",
                "    max_seconds=10.0\n",
                ")\n",
                "\n",
                "def on_progress(telemetry):\n",
                "    if telemetry['frames_processed'] % 30 == 0:\n",
                "        print(f\"Frames: {telemetry['frames_processed']}/{telemetry['total_frames']} | FPS: {telemetry['fps']:.1f}\")\n",
                "\n",
                "res = streamer.process(\n",
                "    frame_transform_fn=lambda f, prev: pipeline.enhance_frame(f, prev, preset=selected_preset),\n",
                "    progress_callback=on_progress\n",
                ")\n",
                "print(f\"Complete in {res['total_time_seconds']:.2f}s!\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 4: Visual Comparison & Quality Metrics"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import cv2\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "orig_sample = res['sample_orig_frames'][0]\n",
                "enh_sample = res['sample_enh_frames'][0]\n",
                "\n",
                "orig_rgb = cv2.cvtColor(orig_sample, cv2.COLOR_BGR2RGB)\n",
                "enh_rgb = cv2.cvtColor(enh_sample, cv2.COLOR_BGR2RGB)\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
                "axes[0].imshow(orig_rgb)\n",
                "axes[0].set_title('Original Degraded Dashcam Frame', fontsize=12)\n",
                "axes[0].axis('off')\n",
                "\n",
                "axes[1].imshow(enh_rgb)\n",
                "axes[1].set_title(f'DashClarity AI Enhanced ({selected_preset.upper()})', fontsize=12)\n",
                "axes[1].axis('off')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = notebook_dir / "colab_demo.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Created notebook:", out_path, "(", out_path.stat().st_size, "bytes)")
