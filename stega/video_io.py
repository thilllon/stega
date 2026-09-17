"""ffmpeg-based video I/O (RGB24). Using the ffmpeg CLI instead of cv2.VideoCapture handles phone
recordings robustly: HEVC/.mov, rotation metadata, 10-bit/HDR -> 8-bit conversion."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterator

import cv2
import numpy as np


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH (macOS: brew install ffmpeg)")
    return exe


class VideoWriter:
    def __init__(self, path: str, width: int, height: int, fps: float, crf: int = 14, preset: str = "medium"):
        cmd = [
            _ffmpeg(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
                   "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-g", str(int(fps * 2)), "-movflags", "+faststart", path,
        ]  # fmt: skip
        self.size = (width, height)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, rgb: np.ndarray, repeat: int = 1) -> None:
        assert rgb.shape[:2] == (self.size[1], self.size[0]) and rgb.dtype == np.uint8
        buf = np.ascontiguousarray(rgb).tobytes()
        for _ in range(repeat):
            self.proc.stdin.write(buf)

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def probe(path: str) -> dict:
    """Frame size as delivered by ffmpeg (after autorotation), fps and approximate frame count."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such video: {path}")
    proc = subprocess.run(
        [_ffmpeg(), "-nostdin", "-loglevel", "error", "-i", path, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, stdin=subprocess.DEVNULL,
    )  # fmt: skip
    if proc.returncode != 0 or not proc.stdout:
        tail = proc.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["no video stream"]
        raise RuntimeError(f"cannot read video {path}: {tail[0]}")
    png = proc.stdout
    first = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise RuntimeError(f"could not decode a frame from {path}")
    info = {"width": first.shape[1], "height": first.shape[0], "fps": 30.0, "frames": None, "duration": None}
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=avg_frame_rate,nb_frames,duration:format=duration", "-of", "json", path],
            capture_output=True, text=True, stdin=subprocess.DEVNULL,
        ).stdout  # fmt: skip
        try:
            j = json.loads(out)
            st = j["streams"][0]
            num, den = st.get("avg_frame_rate", "30/1").split("/")
            info["fps"] = float(num) / float(den) if float(den) else 30.0
            if st.get("nb_frames", "N/A").isdigit():
                info["frames"] = int(st["nb_frames"])
            dur = st.get("duration") or j.get("format", {}).get("duration")
            info["duration"] = float(dur) if dur not in (None, "N/A") else None
        except (KeyError, ValueError, IndexError, json.JSONDecodeError):
            pass
    return info


def read_frames(path: str, max_width: int = 1920) -> Iterator[np.ndarray]:
    """Yield RGB frames, downscaled so the longest side <= max_width (0 = native)."""
    info = probe(path)
    w, h = info["width"], info["height"]
    if max_width and max(w, h) > max_width:
        s = max_width / max(w, h)
        w, h = int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2
    # -nostdin + DEVNULL: otherwise ffmpeg puts the user's terminal in raw mode and, when we stop early
    # and kill it, never restores it (no echo in the shell afterwards)
    # -fps_mode passthrough: phone recordings (e.g. Samsung) declare r_frame_rate=120 with ~30 fps real
    # frames; without it ffmpeg duplicates every frame to fill a constant 120 fps output
    cmd = [_ffmpeg(), "-nostdin", "-loglevel", "error", "-i", path, "-fps_mode", "passthrough",
           "-vf", f"scale={w}:{h}:flags=area", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]  # fmt: skip
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    n = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.kill()  # kill before closing the pipe, so ffmpeg doesn't log "Broken pipe" mid-progress-bar
        proc.wait()
        proc.stdout.close()
