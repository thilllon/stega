"""Fast tests for the phone re-recording channel simulator (stega.camsim)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pytest

from stega import camsim

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"

W, H, FPS, SECS = 960, 540, 30, 1.5


def _ffprobe(path):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames,codec_name,pix_fmt",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    n, d = s["avg_frame_rate"].split("/")
    s["fps"] = float(n) / float(d)
    s["frames"] = int(s["nb_read_frames"])
    return s


def _read_frames(path, w, h, limit=None):
    proc = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        check=True, capture_output=True,
    )
    arr = np.frombuffer(proc.stdout, np.uint8).reshape(-1, h, w, 3)
    return arr[:limit] if limit else arr


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """~1.5 s 960x540@30 clip: checkerboards of changing size/phase + a frame counter."""
    path = tmp_path_factory.mktemp("camsim") / "in.mp4"
    proc = subprocess.Popen(
        [FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-crf", "12", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE,
    )
    yy, xx = np.mgrid[0:H, 0:W]
    for i in range(int(SECS * FPS)):
        cell = 8 + 4 * (i // 3 % 4)
        board = (((xx + i // 3 * 3) // cell + yy // cell) % 2 * 255).astype(np.uint8)
        img = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img, (380, 220), (580, 320), (0, 0, 0), -1)
        cv2.putText(img, f"{i:03d}", (395, 295), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 5)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    return path


@pytest.mark.parametrize(
    "preset,size",
    [("mild", (960, 540)), ("phone", (1920, 1080)), ("harsh", (960, 540))],
)
def test_presets(clip, tmp_path, preset, size):
    out = tmp_path / f"{preset}.mp4"
    summary = camsim.simulate(str(clip), str(out), preset=preset, seed=7, out_size=size)
    assert out.exists() and out.stat().st_size > 10_000

    info = _ffprobe(out)
    assert (info["width"], info["height"]) == size
    assert info["codec_name"] == "h264" and info["pix_fmt"] == "yuv420p"
    assert abs(info["fps"] - summary["temporal"]["cam_fps"]) < 0.05
    assert 29.9 < info["fps"] < 30.1
    expected = SECS * summary["temporal"]["cam_fps"]
    assert abs(info["frames"] - expected) <= 3, (info["frames"], expected)
    assert summary["frames_out"] == info["frames"]

    # not blank, not a copy of the input: has contrast and differs strongly from the input
    frames = _read_frames(out, *size, limit=10).astype(np.float32)
    assert frames.std() > 20, "output looks blank"
    assert 20 < frames.mean() < 235
    src = _read_frames(clip, W, H, limit=10).astype(np.float32)
    src_rs = np.stack([cv2.resize(f, size, interpolation=cv2.INTER_LINEAR) for f in src])
    assert np.abs(frames - src_rs).mean() > 20, "output too close to the input"

    # sampled parameters within preset ranges
    P = camsim.PRESETS[preset]
    g = summary["geometry"]
    assert P["fill"][0] * 0.8 <= g["fill_width"] <= min(1.0, P["fill"][1] * 1.1)
    assert abs(g["roll_deg"]) <= P["roll"] + 1e-6
    assert P["exposure_ms"][0] - 1e-6 <= summary["temporal"]["exposure_ms"] <= P["exposure_ms"][1] + 1e-6


def test_start_duration_and_reproducible_params(clip, tmp_path):
    out = tmp_path / "seg.mp4"
    s1 = camsim.simulate(str(clip), str(out), preset="phone", seed=3, start=0.5, duration=0.5,
                         out_size=(640, 360), cam_fps=29.97)
    info = _ffprobe(out)
    assert (info["width"], info["height"]) == (640, 360)
    assert abs(info["fps"] - 29.97) < 0.01
    assert 12 <= info["frames"] <= 17
    # same seed -> same sampled channel
    sim_a = camsim.CamSim(camsim.probe(str(clip)), "phone", 3, (640, 360), 29.97)
    sim_b = camsim.CamSim(camsim.probe(str(clip)), "phone", 3, (640, 360), 29.97)
    geo = camsim._jsonable(sim_a.summary["geometry"])
    assert geo == camsim._jsonable(sim_b.summary["geometry"]) == s1["geometry"]
    sim_c = camsim.CamSim(camsim.probe(str(clip)), "phone", 4, (640, 360), 29.97)
    assert camsim._jsonable(sim_c.summary["geometry"]) != geo
    # same seed -> identical output pixels
    out2 = tmp_path / "seg2.mp4"
    camsim.simulate(str(clip), str(out2), preset="phone", seed=3, start=0.5, duration=0.5,
                    out_size=(640, 360), cam_fps=29.97)
    assert np.array_equal(_read_frames(out, 640, 360), _read_frames(out2, 640, 360))


def test_cli(clip, tmp_path):
    out = tmp_path / "cli.mp4"
    dump = tmp_path / "frames"
    res = subprocess.run(
        [sys.executable, "-m", "stega.camsim", str(clip), str(out), "--preset", "harsh", "--seed", "1",
         "--size", "640x360", "--duration", "0.5", "--dump-frames", str(dump), "--dump-every", "5", "--quiet"],
        capture_output=True, text=True, timeout=60,
    )
    assert res.returncode == 0, res.stderr
    summary = json.loads(res.stdout)
    assert summary["preset"] == "harsh" and summary["frames_out"] > 10
    pngs = sorted(dump.glob("*.png"))
    assert len(pngs) >= 2
    assert cv2.imread(str(pngs[0])).shape == (360, 640, 3)
