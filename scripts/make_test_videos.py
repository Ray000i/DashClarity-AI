"""
DashClarity AI - Generate a varied set of upload test videos (formats, audio, geometry, timing).

    python scripts/make_test_videos.py --out <folder>

All clips are derived from footage already in the repository; nothing is downloaded.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from src.utils import get_ffmpeg_binary  # noqa: E402

FF = get_ffmpeg_binary()
NIGHT = BASE / "input_videos" / "real" / "real_night_dashcam_01.mp4"
DAY = BASE / "input_videos" / "clean_sources" / "clean_day_01_highway.mp4"
HD = next((BASE / "input_videos").glob("*1920_1080*.mp4"), None)
UHD = next((BASE / "input_videos").glob("*3840x2160*.mp4"), None)
SINE = ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=30"]


def ff(*args: str) -> None:
    subprocess.run([FF, "-y", "-loglevel", "error", *args], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    cases = {
        # name: ffmpeg args producing it
        "01_night_720p_aac.mp4": ["-i", str(NIGHT), *SINE, "-t", "3", "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
                                  "-crf", "20", "-c:a", "aac", "-shortest"],
        "02_odd_size_641x361_aac.mp4": ["-i", str(NIGHT), *SINE, "-t", "2", "-map", "0:v", "-map", "1:a", "-vf", "scale=641:361",
                                        "-c:v", "libx264", "-pix_fmt", "yuv444p", "-c:a", "aac", "-shortest"],
        "03_portrait_rotated.mp4": ["-display_rotation", "90", "-i", str(NIGHT), "-t", "2", "-c", "copy"],
        "04_avi_mjpeg_pcm.avi": ["-i", str(NIGHT), *SINE, "-t", "2", "-map", "0:v", "-map", "1:a", "-vf", "scale=640:360",
                                 "-c:v", "mjpeg", "-q:v", "4", "-c:a", "pcm_s16le", "-shortest"],
        "05_mov_aac.mov": ["-i", str(NIGHT), *SINE, "-t", "2", "-map", "0:v", "-map", "1:a", "-vf", "scale=854:480",
                           "-c:v", "libx264", "-c:a", "aac", "-shortest"],
        "06_mkv_opus.mkv": ["-i", str(NIGHT), *SINE, "-t", "2", "-map", "0:v", "-map", "1:a", "-vf", "scale=854:480",
                            "-c:v", "libx264", "-c:a", "libopus", "-shortest"],
        "07_vfr_mp3.mp4": ["-i", str(NIGHT), *SINE, "-t", "3", "-map", "0:v", "-map", "1:a", "-vf",
                           "scale=640:360,setpts='if(lt(N,30),N/30/TB,(30+(N-30)*2)/30/TB)'", "-fps_mode", "vfr",
                           "-c:v", "libx264", "-c:a", "libmp3lame", "-shortest"],
        "08_two_frames.mp4": ["-i", str(NIGHT), "-frames:v", "2", "-vf", "scale=640:360", "-c:v", "libx264"],
        "09_hevc_10bit.mp4": ["-i", str(NIGHT), "-t", "2", "-vf", "scale=854:480", "-c:v", "libx265", "-pix_fmt",
                              "yuv420p10le", "-tag:v", "hvc1"],
        "10_daylight_720p.mp4": ["-i", str(DAY), "-t", "2", "-c", "copy"],
    }
    if HD:
        cases["11_1080p_25fps_aac.mp4"] = ["-i", str(HD), *SINE, "-t", "2", "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                                           "-c:a", "aac", "-shortest"]
    if UHD:
        cases["12_4k.mp4"] = ["-i", str(UHD), "-t", "1", "-c", "copy"]
    for name, args in cases.items():
        dst = out / name
        try:
            ff(*args, str(dst))
            print("made", name)
        except subprocess.CalledProcessError as exc:
            print("FAILED to make", name, exc)
    (out / "13_corrupt.mp4").write_bytes((out / "01_night_720p_aac.mp4").read_bytes()[:20000])
    (out / "14_not_a_video.mp4").write_bytes(b"this is a text file pretending to be a video\n" * 100)
    print("made 13_corrupt.mp4, 14_not_a_video.mp4")


if __name__ == "__main__":
    main()
