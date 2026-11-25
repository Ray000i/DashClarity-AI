"""
DashClarity AI - End-to-end upload checks against a running server.

Uploads each test video through the web API, runs an enhancement with the given options and
verifies the result exactly as a user would get it: playback/preview endpoints, download,
side-by-side comparison, thumbnail, and the output file itself (codec, frame rate, frame count,
resolution, duration, audio presence and audio/video length, colour tags).

    python scripts/make_test_videos.py --out <dir>
    python scripts/e2e_upload_check.py --base http://127.0.0.1:5000 --videos <dir>
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from src.utils import get_ffmpeg_binary  # noqa: E402

FF = get_ffmpeg_binary()

# (video, enhance options, expectation)
CASES = [
    ("01_night_720p_aac.mp4", {"preset": "balanced"}, {}),
    ("01_night_720p_aac.mp4", {"preset": "fast", "strength": 0.5}, {}),
    ("01_night_720p_aac.mp4", {"preset": "balanced", "frame_skip": 2}, {}),
    ("01_night_720p_aac.mp4", {"preset": "balanced", "max_seconds": 1}, {}),
    ("01_night_720p_aac.mp4", {"preset": "quality", "max_seconds": 0.5, "restore_at_original_res": False}, {"scale": 4}),
    ("01_night_720p_aac.mp4", {"preset": "stcd_comparison", "max_seconds": 1}, {}),
    ("01_night_720p_aac.mp4", {"preset": "hybrid_scunet_stcd", "max_seconds": 1}, {}),
    ("01_night_720p_aac.mp4", {"preset": "classical_baseline"}, {}),
    ("01_night_720p_aac.mp4", {"preset": "balanced"}, {"cancel_after": 4}),
    ("02_odd_size_641x361_aac.mp4", {"preset": "quality"}, {}),
    ("03_portrait_rotated.mp4", {"preset": "fast"}, {"rotated": True}),
    ("04_avi_mjpeg_pcm.avi", {"preset": "balanced"}, {}),
    ("05_mov_aac.mov", {"preset": "classical_baseline"}, {}),
    ("06_mkv_opus.mkv", {"preset": "balanced"}, {}),
    ("07_vfr_mp3.mp4", {"preset": "fast"}, {}),
    ("08_two_frames.mp4", {"preset": "quality"}, {}),
    ("09_hevc_10bit.mp4", {"preset": "fast"}, {}),
    ("10_daylight_720p.mp4", {"preset": "balanced"}, {}),
    ("11_1080p_25fps_aac.mp4", {"preset": "quality"}, {}),
    ("12_4k.mp4", {"preset": "balanced", "max_seconds": 0.2}, {}),
    ("12_4k.mp4", {"preset": "quality", "max_seconds": 0.1}, {}),
    ("13_corrupt.mp4", None, {"reject": True}),
    ("14_not_a_video.mp4", None, {"reject": True}),
]


def probe(path: str) -> dict:
    txt = subprocess.run([FF, "-hide_banner", "-i", path], capture_output=True, text=True, errors="ignore").stderr
    d = {"raw": txt}
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", txt)
    d["duration"] = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else None
    v = re.search(r"Video: (\w+).*?, (\d+)x(\d+).*?, ([\d.]+) fps", txt)
    if v:
        d.update(vcodec=v[1], width=int(v[2]), height=int(v[3]), fps=float(v[4]))
    a = re.search(r"Audio: (\w+)", txt)
    d["acodec"] = a[1] if a else None
    d["bt709"] = "bt709" in txt.split("Video:")[1] if "Video:" in txt else False
    rot = re.search(r"rotation of (-?[\d.]+)", txt)
    d["rotation"] = float(rot[1]) if rot else 0.0
    return d


def stream_duration(path: str, kind: str) -> float | None:
    """Duration of the first audio or video stream by decoding packets (robust to container tags)."""
    sel = "a:0" if kind == "a" else "v:0"
    txt = subprocess.run([FF, "-hide_banner", "-i", path, "-map", f"0:{sel}", "-f", "null", "-"],
                         capture_output=True, text=True, errors="ignore").stderr
    t = re.findall(r"time=(\d+):(\d+):([\d.]+)", txt)
    return int(t[-1][0]) * 3600 + int(t[-1][1]) * 60 + float(t[-1][2]) if t else None


def count_frames(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = 0
    while cap.read()[0]:
        n += 1
    return n


def run_case(base: str, vids: Path, name: str, opts: dict | None, exp: dict, tmp: Path) -> dict:
    s = requests.Session()
    src = vids / name
    res = {"case": f"{name} {json.dumps(opts) if opts else ''}".strip(), "problems": []}
    P = res["problems"]
    with open(src, "rb") as fh:
        r = s.post(f"{base}/api/upload", files={"video": (name, fh)})
    if exp.get("reject"):
        res["result"] = f"HTTP {r.status_code}: {r.json().get('error', '')[:90]}"
        if r.status_code not in (400, 415, 422):
            P.append(f"corrupt/invalid upload not rejected (HTTP {r.status_code})")
        return res
    if r.status_code != 201:
        P.append(f"upload failed HTTP {r.status_code}: {r.text[:200]}")
        return res
    jid = r.json()["job_id"]
    res["job"] = jid
    pv = s.get(f"{base}/media/original/{jid}")
    if pv.status_code != 200:
        P.append(f"original preview HTTP {pv.status_code}")
    else:
        pf = tmp / f"{jid}_preview.mp4"
        pf.write_bytes(pv.content)
        pp = probe(str(pf))
        if pp.get("vcodec") != "h264":
            P.append(f"original preview not browser-playable H.264 (codec {pp.get('vcodec')})")

    r = s.post(f"{base}/api/enhance/{jid}", json=opts)
    if r.status_code not in (200, 202):
        P.append(f"enhance request HTTP {r.status_code}: {r.text[:200]}")
        return res
    t0 = time.time()
    cancelled = False
    while True:
        st = s.get(f"{base}/api/status/{jid}").json()
        if exp.get("cancel_after") and not cancelled and st["frames_processed"] >= exp["cancel_after"]:
            s.post(f"{base}/api/cancel/{jid}")
            cancelled = True
        if st["status"] in ("completed", "failed", "cancelled"):
            break
        if time.time() - t0 > 900:
            P.append("timed out after 15 min")
            return res
        time.sleep(1)
    res["status"] = st["status"]
    res["seconds"] = round(time.time() - t0, 1)
    if exp.get("cancel_after"):
        if st["status"] != "cancelled":
            P.append(f"cancel did not cancel (status {st['status']})")
        leftovers = list((BASE_DIR / "output_videos").glob(f"*{jid}*.temp_*"))
        if leftovers:
            P.append(f"temp files left after cancel: {[p.name for p in leftovers]}")
        return res
    if st["status"] != "completed":
        P.append(f"job {st['status']}: {st.get('error')}")
        return res

    res_json = s.get(f"{base}/api/result/{jid}").json()
    tech = res_json["metrics"].get("technical_summary", {})
    res["fps_reported"] = tech.get("processing_fps")
    res["device"] = tech.get("device_mode")
    if "CPU" in str(tech.get("device_mode")):
        P.append("processed on CPU")

    out = tmp / f"{jid}_enhanced.mp4"
    r = s.get(f"{base}/api/download/{jid}")
    if r.status_code != 200:
        P.append(f"download HTTP {r.status_code}")
        return res
    out.write_bytes(r.content)
    rr = s.get(f"{base}/media/enhanced/{jid}", headers={"Range": "bytes=0-1023"})
    if rr.status_code != 206:
        P.append(f"enhanced stream range request HTTP {rr.status_code}")
    if s.get(f"{base}/media/thumbnail/{jid}").status_code != 200:
        P.append("thumbnail missing")
    cmp_ = s.get(f"{base}/api/download/{jid}?mode=comparison")
    if cmp_.status_code != 200:
        P.append(f"comparison download HTTP {cmp_.status_code}: {cmp_.text[:120]}")

    pin, pout = probe(str(src)), probe(str(out))
    frames = st["frames_processed"]
    skip = int((opts or {}).get("frame_skip", 1))
    dec = count_frames(str(out))
    res.update(frames=frames, decoded=dec, in_size=f"{pin.get('width')}x{pin.get('height')}",
               out_size=f"{pout.get('width')}x{pout.get('height')}", in_fps=pin.get("fps"), out_fps=pout.get("fps"),
               in_dur=pin.get("duration"), out_dur=pout.get("duration"), in_audio=pin.get("acodec"), out_audio=pout.get("acodec"))
    if dec != frames:
        P.append(f"decoded {dec} frames but job reported {frames}")
    if pout.get("vcodec") != "h264":
        P.append(f"output codec {pout.get('vcodec')}")
    if not pout.get("bt709"):
        P.append("output not tagged BT.709")
    # geometry
    w, h = pin.get("width"), pin.get("height")
    if exp.get("rotated") and abs(pin.get("rotation", 0)) in (90.0, 270.0):
        w, h = h, w
    scale = exp.get("scale", 1)
    ew, eh = (w * scale) // 2 * 2, (h * scale) // 2 * 2
    if (pout.get("width"), pout.get("height")) != (ew, eh):
        P.append(f"resolution {pout.get('width')}x{pout.get('height')}, expected {ew}x{eh}")
    if pout.get("rotation"):
        P.append(f"output still carries rotation {pout['rotation']}")
    # timing: the output must cover the same time span as the processed part of the input
    vdur = stream_duration(str(out), "v")
    max_s = (opts or {}).get("max_seconds")
    expected_span = min(pin["duration"], max_s) if max_s else pin["duration"]
    res["out_video_dur"] = vdur
    if vdur is not None and abs(vdur - expected_span) > max(0.12, 1.5 / (pin.get("fps") or 30)):
        P.append(f"video lasts {vdur:.2f}s but covers {expected_span:.2f}s of input (A/V drift / wrong fps)")
    if pin.get("acodec"):
        if not pout.get("acodec"):
            P.append(f"audio lost (input {pin['acodec']})")
        else:
            adur = stream_duration(str(out), "a")
            res["out_audio_dur"] = adur
            if vdur and adur and abs(adur - vdur) > 0.15:
                P.append(f"audio {adur:.2f}s vs video {vdur:.2f}s")
    elif pout.get("acodec"):
        P.append("audio appeared in output although input had none")
    if pout.get("fps") and pin.get("fps") and skip == 1 and abs(pout["fps"] - pin["fps"]) > 0.05 and "vfr" not in name:
        P.append(f"fps changed {pin['fps']} -> {pout['fps']}")
    s.delete(f"{base}/api/job/{jid}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5000")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--json", default=str(BASE_DIR / "results" / "e2e_upload_check.json"))
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="dc_e2e_"))
    results = []
    for name, opts, exp in CASES:
        if a.only and not any(o in name for o in a.only):
            continue
        try:
            r = run_case(a.base, Path(a.videos), name, opts, exp, tmp)
        except Exception as exc:
            r = {"case": f"{name} {opts}", "problems": [f"checker exception: {type(exc).__name__}: {exc}"]}
        results.append(r)
        flag = "PASS" if not r["problems"] else "FAIL"
        extra = {k: r[k] for k in ("status", "frames", "decoded", "in_size", "out_size", "in_fps", "out_fps", "in_dur",
                                   "out_video_dur", "out_audio_dur", "in_audio", "out_audio", "fps_reported", "result", "seconds")
                 if k in r}
        print(f"[{flag}] {r['case']}\n       {extra}" + "".join(f"\n       !! {p}" for p in r["problems"]), flush=True)
    Path(a.json).write_text(json.dumps(results, indent=2, default=str))
    print(f"\n{sum(1 for r in results if not r['problems'])}/{len(results)} cases passed. Saved {a.json}")


if __name__ == "__main__":
    main()
