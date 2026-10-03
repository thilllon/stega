"""Phone-camera re-recording channel simulator ("camsim").

Simulates pointing a phone at a monitor that plays a video and recording it:

    input video (opaque pixels)
      -> monitor: gamma / contrast / black lift / brightness / colour cast,
         LCD response time, occasional frame-pacing glitches (60 Hz refresh)
      -> scene: screen + dark bezel placed in a cluttered room background
      -> geometry: 3D pose (tilt/roll/distance), perspective homography,
         hand shake (smooth drift + tremor + occasional jerks), radial lens
         distortion
      -> temporal: camera clock not locked to the display, exposure integration,
         rolling shutter (per-row sample time), motion blur
      -> optics: defocus (depth-varying), vignetting, glare / veiling glare
      -> sensor: screen sub-pixel structure sampled by an RGGB Bayer mosaic
         (moire), auto-exposure + white-balance drift, light flicker,
         shot + read + row noise
      -> ISP: demosaic, gamma/tone curve, saturation, sharpening halo
      -> H.264 (libx264, yuv420p) at a phone-like bitrate

Everything is randomised from ``seed`` and reproducible.  The input is streamed
through ffmpeg; only a handful of decoded frames are held in memory.

CLI::

    uv run python -m stega.camsim IN.mp4 OUT.mp4 --preset phone --seed 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction

import cv2
import numpy as np

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"

# --------------------------------------------------------------------------------------
# Presets.  Tuples are uniform ranges (lo, hi); scalars are fixed values.
# --------------------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
    "mild": dict(
        # geometry
        fill=(0.80, 0.90), tilt=(0.0, 4.0), roll=1.0, hfov=(64.0, 70.0), k1=0.02,
        bezel=(0.02, 0.03), center_jitter=0.3,
        # hand shake (tripod-like)
        shake_px=(0.2, 0.6), shake_rot_deg=0.02, shake_zoom=0.0005, drift_tau=(1.5, 3.0),
        tremor_px=(0.0, 0.08), jump_rate=0.0, jump_px=(0.0, 0.0),
        # temporal
        exposure_ms=(8.0, 16.0), readout_ms=(12.0, 20.0), lcd_ms=(2.0, 5.0), pacing_glitch=0.0,
        # optics
        defocus=(0.45, 0.75), focus_breath=0.04, af_hunt_rate=0.0, dof=(0.0, 0.1),
        vignette=(0.05, 0.15), glare_prob=0.0, glare_int=(0.0, 0.0), veil=(0.0, 0.004),
        flicker=(0.0, 0.0),
        # display
        disp_gamma=(2.1, 2.3), contrast=(0.97, 1.03), black=(0.001, 0.004), brightness=(0.95, 1.05),
        cast=0.015, native_width=(1920,), moire=(0.03, 0.07), view_falloff=(0.0, 0.1),
        # sensor / ISP
        ae_target=(0.20, 0.26), ae_tau=(0.6, 1.2), ae_drift=0.015, clip_allow=0.92, wb_bias=0.015,
        wb_drift=0.006, shot=(1 / 9000, 1 / 6000), read=(0.0010, 0.0020), row_noise=0.0,
        sharpen=(0.2, 0.4), saturation=(1.0, 1.1), tone_contrast=(0.05, 0.12),
        # encoder
        bitrate="20M", x264_preset="fast",
    ),
    "phone": dict(
        fill=(0.65, 0.85), tilt=(2.0, 12.0), roll=4.0, hfov=(64.0, 74.0), k1=0.04,
        bezel=(0.02, 0.04), center_jitter=0.6,
        shake_px=(1.5, 4.0), shake_rot_deg=0.15, shake_zoom=0.003, drift_tau=(0.7, 2.0),
        tremor_px=(0.2, 0.7), jump_rate=0.12, jump_px=(3.0, 10.0),
        exposure_ms=(10.0, 33.0), readout_ms=(18.0, 30.0), lcd_ms=(3.0, 10.0), pacing_glitch=0.01,
        defocus=(0.8, 1.35), focus_breath=0.10, af_hunt_rate=0.03, dof=(0.1, 0.35),
        vignette=(0.15, 0.35), glare_prob=0.35, glare_int=(0.01, 0.06), veil=(0.002, 0.012),
        flicker=(0.0, 0.01),
        disp_gamma=(2.0, 2.4), contrast=(0.9, 1.05), black=(0.002, 0.010), brightness=(0.85, 1.1),
        cast=0.04, native_width=(1920, 2560, 2880), moire=(0.06, 0.15), view_falloff=(0.03, 0.2),
        ae_target=(0.18, 0.26), ae_tau=(0.3, 0.9), ae_drift=0.05, clip_allow=0.98, wb_bias=0.04,
        wb_drift=0.02, shot=(1 / 3500, 1 / 2000), read=(0.0025, 0.0045), row_noise=0.0008,
        sharpen=(0.4, 0.8), saturation=(1.05, 1.25), tone_contrast=(0.1, 0.25),
        bitrate="12M", x264_preset="veryfast",
    ),
    "harsh": dict(
        fill=(0.50, 0.60), tilt=(12.0, 20.0), roll=6.0, hfov=(66.0, 78.0), k1=0.06,
        bezel=(0.025, 0.04), center_jitter=1.0,
        shake_px=(4.0, 9.0), shake_rot_deg=0.4, shake_zoom=0.008, drift_tau=(0.4, 1.2),
        tremor_px=(0.6, 1.6), jump_rate=0.5, jump_px=(8.0, 28.0),
        exposure_ms=(18.0, 33.0), readout_ms=(24.0, 33.0), lcd_ms=(5.0, 15.0), pacing_glitch=0.03,
        defocus=(1.25, 2.1), focus_breath=0.18, af_hunt_rate=0.12, dof=(0.3, 0.7),
        vignette=(0.3, 0.5), glare_prob=1.0, glare_int=(0.08, 0.30), veil=(0.01, 0.04),
        flicker=(0.02, 0.07),
        disp_gamma=(1.9, 2.5), contrast=(0.8, 1.0), black=(0.006, 0.025), brightness=(0.7, 1.1),
        cast=0.07, native_width=(1920, 2560, 3840), moire=(0.08, 0.20), view_falloff=(0.15, 0.35),
        ae_target=(0.16, 0.30), ae_tau=(0.15, 0.6), ae_drift=0.15, clip_allow=1.12, wb_bias=0.08,
        wb_drift=0.05, shot=(1 / 1200, 1 / 600), read=(0.006, 0.011), row_noise=0.003,
        sharpen=(0.7, 1.2), saturation=(1.1, 1.35), tone_contrast=(0.15, 0.35),
        bitrate="5M", x264_preset="veryfast",
    ),
}

DISPLAY_REFRESH = 60.0
SIG_RATE = 240.0  # Hz, sampling rate of the smooth random processes


def _u(rng: np.random.Generator, r) -> float:
    if isinstance(r, (tuple, list)):
        return float(rng.uniform(r[0], r[1]))
    return float(r)


# --------------------------------------------------------------------------------------
# ffmpeg helpers
# --------------------------------------------------------------------------------------
def probe(path: str) -> dict:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_packets,duration",
         "-of", "json", path],
        check=True, capture_output=True, text=True,
    ).stdout
    s = json.loads(out)["streams"][0]

    def rate(x):
        try:
            n, d = x.split("/")
            return float(n) / float(d) if float(d) else 0.0
        except Exception:
            return 0.0

    fps = rate(s.get("avg_frame_rate", "0/0")) or rate(s.get("r_frame_rate", "0/0"))
    return dict(
        width=int(s["width"]), height=int(s["height"]), fps=fps,
        frames=int(s.get("nb_read_packets", 0) or 0),
        duration=float(s.get("duration", 0) or 0),
    )


class _FrameReader:
    """Streams decoded BGR frames from ffmpeg."""

    def __init__(self, path, w, h, start, duration):
        cmd = [FFMPEG, "-v", "error", "-nostdin"]
        if start and start > 0:
            cmd += ["-ss", f"{start:.6f}"]
        cmd += ["-i", path]
        if duration is not None:
            cmd += ["-t", f"{duration:.6f}"]
        cmd += ["-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.w, self.h = w, h
        self.nbytes = w * h * 3
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=self.nbytes * 2)

    def read(self):
        buf = np.empty(self.nbytes, np.uint8)
        mv = memoryview(buf)
        got = 0
        while got < self.nbytes:
            n = self.proc.stdout.readinto(mv[got:])
            if not n:
                return None
            got += n
        return buf.reshape(self.h, self.w, 3)

    def close(self):
        try:
            self.proc.stdout.close()
        except Exception:
            pass
        try:
            self.proc.kill()
        except Exception:
            pass
        self.proc.wait()


def _fps_fraction(fps: float) -> str:
    fr = Fraction(fps).limit_denominator(1001)
    return f"{fr.numerator}/{fr.denominator}"


def _open_encoder(out_path, w, h, fps, bitrate, preset, threads=1):
    br = _parse_rate(bitrate)
    cmd = [
        FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
        "-framerate", _fps_fraction(fps), "-i", "-",
        "-c:v", "libx264", "-preset", preset, "-profile:v", "high", "-bf", "0", "-g", str(int(round(fps))),
        "-b:v", str(br), "-maxrate", str(int(br * 1.5)), "-bufsize", str(int(br * 2)),
        "-pix_fmt", "yuv420p", "-r", _fps_fraction(fps), "-threads", str(threads),
        "-movflags", "+faststart", out_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _parse_rate(s) -> int:
    if isinstance(s, (int, float)):
        return int(s)
    s = str(s).strip().upper()
    mult = {"K": 1e3, "M": 1e6}.get(s[-1], 1)
    return int(float(s[:-1] if mult != 1 else s) * mult)


# --------------------------------------------------------------------------------------
# Smooth random processes (hand shake, drifts)
# --------------------------------------------------------------------------------------
class _Signals:
    """Unit-variance low-pass noise (two cascaded one-pole filters) per dimension,
    sampled at SIG_RATE and generated lazily (deterministic)."""

    def __init__(self, rng, tau1, tau2):
        self.rng = rng
        dt = 1.0 / SIG_RATE
        self.a1 = np.exp(-dt / np.asarray(tau1, float))
        self.a2 = np.exp(-dt / np.asarray(tau2, float)) - 1e-6
        a1, a2 = self.a1, self.a2
        var = ((1 - a1) * (1 - a2)) ** 2 * (1 + a1 * a2) / ((1 - a1**2) * (1 - a2**2) * (1 - a1 * a2))
        self.gain = 1.0 / np.sqrt(var)
        self.n = len(self.a1)
        self.s1 = np.zeros(self.n)
        self.s2 = np.zeros(self.n)
        # burn in to stationarity
        for _ in range(int(6 * float(np.max(tau1)) * SIG_RATE)):
            self._step()
        self.data = np.zeros((1024, self.n))
        self.len = 0

    def _step(self):
        x = self.rng.standard_normal(self.n)
        self.s1 = self.a1 * self.s1 + (1 - self.a1) * x
        self.s2 = self.a2 * self.s2 + (1 - self.a2) * self.s1
        return self.s2 * self.gain

    def _ensure(self, n):
        while self.len < n:
            if self.len >= len(self.data):
                self.data = np.concatenate([self.data, np.zeros_like(self.data)])
            self.data[self.len] = self._step()
            self.len += 1

    def at(self, t):
        """Values at times t (array, seconds >= 0) -> (len(t), n)."""
        t = np.atleast_1d(np.asarray(t, float))
        idx = np.clip(t, 0, None) * SIG_RATE
        self._ensure(int(np.ceil(idx.max())) + 2)
        i0 = np.floor(idx).astype(int)
        fr = (idx - i0)[:, None]
        return self.data[i0] * (1 - fr) + self.data[i0 + 1] * fr


class _Events:
    """Poisson events with lazy generation."""

    def __init__(self, rng, rate, make):
        self.rng, self.rate, self.make = rng, rate, make
        self.events = []
        self.t_next = rng.exponential(1.0 / rate) if rate > 0 else math.inf

    def upto(self, t):
        while self.t_next <= t:
            self.events.append(self.make(self.rng, self.t_next))
            self.t_next += self.rng.exponential(1.0 / self.rate)
        return self.events


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# --------------------------------------------------------------------------------------
# Scene construction
# --------------------------------------------------------------------------------------
def _display_lut(rng, P):
    """uint8 code -> linear light (per channel, BGR), float32 (1,256,3)."""
    gamma = _u(rng, P["disp_gamma"])
    contrast = _u(rng, P["contrast"])
    black = _u(rng, P["black"])
    bright = _u(rng, P["brightness"])
    cast = np.exp(rng.normal(0, P["cast"], 3))
    v = np.arange(256) / 255.0
    v = np.clip((v - 0.5) * contrast + 0.5 + (1 - contrast) * 0.1, 0, 1)
    lin = black + (1 - black) * v**gamma
    lut = (bright * lin[None, :, None] * cast[None, None, :]).astype(np.float32)
    info = dict(gamma=gamma, contrast=contrast, black_lift=black, brightness=bright,
                color_cast_bgr=cast.round(3).tolist())
    return np.ascontiguousarray(lut), info


def _value_noise(rng, h, w, cells):
    small = rng.random((max(2, int(cells * h / max(h, w))) + 1, max(2, int(cells * w / max(h, w))) + 1))
    return cv2.resize(small.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)


def _render_background(rng, cw, ch, disp_rect, bezel_rect, kind):
    """Linear-light BGR float32 background at half canvas resolution (cluttered room)."""
    h, w = max(8, ch // 2), max(8, cw // 2)
    s = 0.5
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(ang) * xx / w + np.sin(ang) * yy / h)
    grad = (grad - grad.min()) / (np.ptp(grad) + 1e-6)
    c1 = _room_color(rng, "wall")
    c2 = _room_color(rng, "wall") * rng.uniform(0.3, 1.0)
    img = (c1[None, None] * (1 - grad[..., None]) + c2[None, None] * grad[..., None]).astype(np.float32)
    for cells, amp in ((3, 0.5), (9, 0.35), (27, 0.2), (80, 0.12)):
        n = _value_noise(rng, h, w, cells)
        img *= (1 + amp * (n - 0.5))[..., None]

    # wall / desk split below the monitor
    bx0, by0, bx1, by1 = [int(v * s) for v in bezel_rect]
    desk_y = int(by1 + rng.uniform(0.05, 0.4) * (by1 - by0))
    if desk_y < h:
        desk = _room_color(rng, "wood").astype(np.float32)
        grain = _value_noise(rng, h - desk_y, w, 200)
        stripes = 0.5 + 0.5 * np.sin(xx[desk_y:] * rng.uniform(0.02, 0.2) + 6 * grain)
        img[desk_y:] = desk[None, None] * (0.75 + 0.35 * stripes)[..., None]

    # clutter: posters, shelves, window, lamp, cables, books, text-ish blocks
    for _ in range(int(rng.integers(30, 90))):
        col = tuple(float(c) for c in _room_color(rng))
        cxp, cyp = rng.uniform(-0.1, 1.1) * w, rng.uniform(-0.1, 1.1) * h
        sz = w * math.exp(rng.uniform(math.log(0.006), math.log(0.12)))
        t = rng.random()
        if t < 0.35:
            a = rng.uniform(-0.5, 0.5)
            rw, rh = sz, sz * rng.uniform(0.2, 1.6)
            pts = np.array([[-rw, -rh], [rw, -rh], [rw, rh], [-rw, rh]])
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            pts = (pts @ R.T + [cxp, cyp]).astype(np.int32)
            cv2.fillPoly(img, [pts], col, lineType=cv2.LINE_AA)
            if rng.random() < 0.4:  # framed / text-like content
                for k in range(int(rng.integers(3, 12))):
                    yk = cyp - rh + (k + 1) * 2 * rh / 13
                    x0 = cxp - rw * rng.uniform(0.4, 0.9)
                    x1 = cxp + rw * rng.uniform(-0.2, 0.9)
                    cv2.line(img, (int(x0), int(yk)), (int(x1), int(yk)),
                             tuple(c * rng.uniform(0.2, 2.0) for c in col), max(1, int(rh / 20)), cv2.LINE_AA)
        elif t < 0.55:
            cv2.ellipse(img, (int(cxp), int(cyp)), (int(sz), int(sz * rng.uniform(0.3, 1.0))),
                        rng.uniform(0, 180), 0, 360, col, -1, cv2.LINE_AA)
        elif t < 0.8:
            p0 = (int(cxp), int(cyp))
            p1 = (int(cxp + rng.normal(0, sz * 3)), int(cyp + rng.normal(0, sz * 3)))
            cv2.line(img, p0, p1, col, int(rng.integers(1, 8)), cv2.LINE_AA)
        else:
            nb = int(rng.integers(4, 14))
            bw = sz / nb
            for k in range(nb):
                hh = sz * rng.uniform(0.6, 1.2)
                bc = tuple(float(c) for c in rng.uniform(0.01, 0.3, 3))
                cv2.rectangle(img, (int(cxp + k * bw), int(cyp - hh)), (int(cxp + (k + 0.85) * bw), int(cyp)), bc, -1)
    # uneven room lighting / shadows over everything
    for cells, amp in ((2, 0.8), (6, 0.35)):
        img *= np.clip(1 + amp * (_value_noise(rng, h, w, cells) - 0.5), 0.2, 2)[..., None]
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 2.5))

    # monitor body: stand or laptop deck
    dx0, dy0, dx1, dy1 = [int(v * s) for v in disp_rect]
    if kind == "laptop":
        deck = rng.uniform(0.08, 0.3)
        dh = int((by1 - by0) * rng.uniform(0.35, 0.6))
        pts = np.array([[bx0, by1], [bx1, by1], [bx1 + dh * 0.5, by1 + dh], [bx0 - dh * 0.5, by1 + dh]], np.int32)
        cv2.fillPoly(img, [pts], (deck, deck, deck * 1.02), cv2.LINE_AA)
        kw = (bx1 - bx0) / 16
        for r in range(5):
            for c in range(15):
                yk = by1 + dh * (0.08 + r * 0.14)
                cv2.rectangle(img, (int(bx0 + (c + 0.6) * kw + r * 3), int(yk)),
                              (int(bx0 + (c + 1.4) * kw + r * 3), int(yk + dh * 0.1)),
                              (deck * 0.15,) * 3, -1)
    else:
        neck = rng.uniform(0.01, 0.04)
        nw = (bx1 - bx0) * 0.08
        cxm = (bx0 + bx1) / 2
        cv2.rectangle(img, (int(cxm - nw), by1), (int(cxm + nw), int(by1 + (by1 - by0) * 0.3)), (neck,) * 3, -1)

    # bezel (dark plastic with a faint highlight rim), rounded corners
    bez = rng.uniform(0.004, 0.02)
    rad = max(1, int(0.25 * min(dx0 - bx0 + 1, by1 - dy1 + 1)))
    bez_img = np.zeros((h, w), np.uint8)
    cv2.rectangle(bez_img, (bx0 + rad, by0), (bx1 - rad, by1), 255, -1)
    cv2.rectangle(bez_img, (bx0, by0 + rad), (bx1, by1 - rad), 255, -1)
    for cxr, cyr in ((bx0 + rad, by0 + rad), (bx1 - rad, by0 + rad), (bx0 + rad, by1 - rad), (bx1 - rad, by1 - rad)):
        cv2.circle(bez_img, (cxr, cyr), rad, 255, -1, cv2.LINE_AA)
    m = bez_img.astype(np.float32)[..., None] / 255.0
    gy = np.linspace(1.3, 0.8, h, dtype=np.float32)[:, None, None]
    img = img * (1 - m) + (bez * gy) * m
    cv2.rectangle(img, (bx0, by0), (bx1, by1), (bez * 4,) * 3, 1, cv2.LINE_AA)
    # fine grain
    img *= (1 + 0.04 * (rng.random((h, w), dtype=np.float32) - 0.5))[..., None]
    return np.clip(img, 0, None)


def _room_color(rng, kind=None):
    """Plausible indoor surface colour in linear light (relative to screen white ~1)."""
    t = rng.random() if kind is None else {"wall": 0.0, "wood": 0.6}[kind]
    if t < 0.5:  # walls, paper, plastic: near-neutral
        base, tint = rng.uniform(0.02, 0.16), rng.normal(0, 0.08, 3) + [0.0, 0.0, 0.04]
    elif t < 0.75:  # wood / fabric: warm
        base, tint = rng.uniform(0.01, 0.1), np.array([-0.5, -0.1, 0.25]) + rng.normal(0, 0.1, 3)
    elif t < 0.95:  # dark furniture / objects / occasional saturated item
        base, tint = rng.uniform(0.005, 0.06), rng.normal(0, 0.35, 3)
    else:  # bright: window, lamp, white paper in light
        base, tint = rng.uniform(0.25, 0.7), rng.normal(0, 0.05, 3)
    return base * np.exp(tint)


def _invert_lut(lut, lin):
    """linear image (float, BGR) -> uint8 codes such that lut[code] ~= lin."""
    out = np.empty(lin.shape, np.uint8)
    nb = 8192
    for c in range(3):
        table = lut[0, :, c].astype(np.float64)
        top = table[-1]
        mid = (table[1:] + table[:-1]) / 2
        centers = (np.arange(nb) + 0.5) / nb * top
        inv = np.searchsorted(mid, centers).astype(np.uint8)
        q = np.clip(lin[..., c] * (nb / top), 0, nb - 1).astype(np.int32)
        out[..., c] = inv[q]
    return out


# --------------------------------------------------------------------------------------
# Parameter container
# --------------------------------------------------------------------------------------
@dataclass
class _FrameJob:
    k: int
    contribs: list  # [(frame array, weights(H) float32, r0, r1)]
    tx: np.ndarray  # per half-res row
    ty: np.ndarray
    rot: np.ndarray
    zoom: float
    motion: tuple  # (dx, dy)
    sigma: float
    gain_bgr: np.ndarray
    glare_shift: tuple
    glare_scale: float
    rowgain: np.ndarray | None
    noise_seed: int
    extra: dict = field(default_factory=dict)


class CamSim:
    def __init__(self, info, preset, seed, out_size, cam_fps, display_width=None):
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
        P = self.P = PRESETS[preset]
        self.preset, self.seed = preset, seed
        W, H = out_size
        if W % 2 or H % 2:
            raise ValueError("out_size must be even")
        self.W, self.H = W, H
        self.Wd, self.Hd = info["width"], info["height"]
        self.fps_in = info["fps"] or 30.0
        ss = np.random.SeedSequence([int(seed), 0xCA5, sum(map(ord, preset))])
        r_geo, r_disp, r_tmp, r_opt, r_sig, r_ev, r_bg, r_frame = [np.random.default_rng(s) for s in ss.spawn(8)]
        self.frame_ss = ss.spawn(1)[0]
        self.summary = dict(preset=preset, seed=seed, input=info, out_size=[W, H])

        # ---------------- display ----------------
        self.lut, dinfo = _display_lut(r_disp, P)
        native_w = float(r_disp.choice(P["native_width"]))  # always drawn: keeps the seed's other draws stable
        if display_width:  # a specific panel, e.g. the video shown 1:1 on a monitor of its own resolution
            native_w = float(display_width)
        self.pitch = native_w / self.Wd  # native pixels per input pixel
        dinfo["native_width"] = native_w
        self.summary["display"] = dinfo

        # ---------------- temporal ----------------
        self.cam_fps = float(cam_fps) if cam_fps else float(r_tmp.choice([29.97, 30.0, 30.02]) + r_tmp.uniform(-0.003, 0.003))
        self.E = min(_u(r_tmp, P["exposure_ms"]) / 1000.0, 0.98 / self.cam_fps)
        self.R = _u(r_tmp, P["readout_ms"]) / 1000.0
        self.rho = _u(r_tmp, P["lcd_ms"]) / 1000.0
        self.phase = r_tmp.uniform(0, 1.0 / self.fps_in)
        self.flip_readout = bool(r_tmp.random() < 0.25)
        self.clock_jitter = 0.0002
        self.r_tmp = r_tmp
        self.glitch_p = P["pacing_glitch"]
        self.glitches: list[bool] = []
        self.summary["temporal"] = dict(
            cam_fps=self.cam_fps, exposure_ms=self.E * 1e3, readout_ms=self.R * 1e3,
            lcd_response_ms=self.rho * 1e3, phase_ms=self.phase * 1e3,
            readout_bottom_to_top=self.flip_readout, display_refresh_hz=DISPLAY_REFRESH,
        )

        # ---------------- geometry ----------------
        self._build_geometry(r_geo, P)

        # ---------------- optics / sensor ----------------
        self.sigma0 = _u(r_opt, P["defocus"])
        self.dof = _u(r_opt, P["dof"])
        self.vig = _u(r_opt, P["vignette"])
        self.moire_amp = _u(r_opt, P["moire"])
        self.shot = _u(r_opt, P["shot"])
        self.read = _u(r_opt, P["read"])
        self.row_noise = P["row_noise"]
        self.sharpen = _u(r_opt, P["sharpen"])
        self.sharpen_sigma = r_opt.uniform(0.8, 1.4)
        self.saturation = _u(r_opt, P["saturation"])
        self.tone_contrast = _u(r_opt, P["tone_contrast"])
        self.ae_target = _u(r_opt, P["ae_target"])
        self.ae_tau = _u(r_opt, P["ae_tau"])
        self.wb_base = np.exp(r_opt.normal(0, P["wb_bias"], 3))
        self.wb_base /= self.wb_base[1]
        self.flicker_amp = _u(r_opt, P["flicker"])
        self.flicker_hz = float(r_opt.choice([100.0, 120.0]))
        self.flicker_ph = r_opt.uniform(0, 2 * np.pi)
        self.has_glare = bool(r_opt.random() < P["glare_prob"])
        self.glare_int = _u(r_opt, P["glare_int"]) if self.has_glare else 0.0
        self.veil = _u(r_opt, P["veil"])
        self._build_static_maps(r_opt, P)
        self._build_tone_lut()
        self.summary["optics"] = dict(
            defocus_sigma_px=self.sigma0, dof_extra=self.dof, vignette=self.vig, moire_amp=self.moire_amp,
            glare=self.has_glare, glare_intensity=self.glare_int, veiling_glare=self.veil,
            flicker_amp=self.flicker_amp, flicker_hz=self.flicker_hz,
        )
        self.summary["sensor"] = dict(
            shot_var_per_unit=self.shot, read_sigma=self.read, row_noise=self.row_noise,
            sharpen=self.sharpen, saturation=self.saturation, tone_contrast=self.tone_contrast,
            ae_target=self.ae_target, ae_tau_s=self.ae_tau, wb_base_bgr=self.wb_base.round(3).tolist(),
        )

        # ---------------- shake & drifts ----------------
        dtau = _u(r_sig, P["drift_tau"])
        self.shake_px = _u(r_sig, P["shake_px"])
        self.tremor_px = _u(r_sig, P["tremor_px"])
        self.tremor_hz = r_sig.uniform(7.5, 11.5, 2)
        self.tremor_ph = r_sig.uniform(0, 2 * np.pi, 2)
        # dims: tx, ty, rot, zoom, focus, ae, wb_b, wb_r, glare_x, glare_y, tremor_mod
        self.sig = _Signals(
            r_sig,
            tau1=[dtau, dtau * 1.1, dtau, 2.5, 1.0, 1.5, 4.0, 4.0, 2.0, 2.0, 0.5],
            tau2=[0.06, 0.06, 0.06, 0.2, 0.15, 0.2, 0.5, 0.5, 0.3, 0.3, 0.1],
        )
        jp = P["jump_px"]

        def make_jump(rng, t0):
            m = rng.uniform(jp[0], jp[1])
            a = rng.uniform(0, 2 * np.pi)
            return dict(t0=t0, dur=rng.uniform(0.04, 0.15), dx=m * np.cos(a), dy=m * np.sin(a) * 0.7,
                        drot=rng.normal(0, P["shake_rot_deg"]), ret=rng.uniform(0.4, 2.0))

        def make_hunt(rng, t0):
            return dict(t0=t0, dur=rng.uniform(0.2, 0.7), mag=rng.uniform(0.5, 1.5))

        self.jumps = _Events(r_ev, P["jump_rate"], make_jump)
        self.hunts = _Events(r_ev, P["af_hunt_rate"], make_hunt)
        self.summary["shake"] = dict(drift_px_sigma=self.shake_px, tremor_px=self.tremor_px,
                                     tremor_hz=self.tremor_hz.round(2).tolist(), drift_tau_s=dtau,
                                     jump_rate_hz=P["jump_rate"], jump_px=list(jp))
        self.ae_gain = None
        self.summary["encoder"] = dict(bitrate=P["bitrate"], x264_preset=P["x264_preset"])

        # half-res row indices (sensor rows at 2i+0.5)
        self.row_frac_half = (2 * np.arange(H // 2) + 0.5) / H
        self.row_frac = (np.arange(H) + 0.5) / H
        if self.flip_readout:
            self.row_frac_half = 1 - self.row_frac_half
            self.row_frac = 1 - self.row_frac
        self._tls = threading.local()

    # ------------------------------------------------------------------ geometry
    def _build_geometry(self, rng, P):
        W, H, Wd, Hd = self.W, self.H, self.Wd, self.Hd
        hfov = _u(rng, P["hfov"])
        f = (W / 2) / math.tan(math.radians(hfov / 2))
        cx, cy = W / 2 + rng.normal(0, W * 0.003), H / 2 + rng.normal(0, H * 0.003)
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        tilt = _u(rng, P["tilt"])
        psi = rng.uniform(0, 2 * np.pi)
        yaw, pitch = tilt * math.cos(psi), tilt * math.sin(psi) * 0.8
        roll = rng.uniform(-P["roll"], P["roll"])
        y, p, r = np.radians([yaw, pitch, roll])
        Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
        Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
        Rm = Rz @ Rx @ Ry
        k1 = rng.uniform(-P["k1"], P["k1"])
        k2 = -0.3 * k1 * rng.uniform(0, 1)
        self.f, self.cx, self.cy, self.k1, self.k2 = f, cx, cy, k1, k2
        corners = np.array([[0, 0, 1], [Wd, 0, 1], [Wd, Hd, 1], [0, Hd, 1]], float).T
        T0 = np.array([[1, 0, -Wd / 2], [0, 1, -Hd / 2], [0, 0, 1.0]])
        fill = _u(rng, P["fill"])
        margin = 0.012 * W + 2.0 * P["shake_px"][1] + 0.3 * P["jump_px"][1]
        for _attempt in range(40):
            Z = f * Wd / (fill * W)
            for _ in range(6):
                Hm = K @ np.column_stack([Rm[:, 0], Rm[:, 1], [0, 0, Z]]) @ T0
                pc = Hm @ corners
                pc = pc[:2] / pc[2]
                wfrac = np.ptp(pc[0]) / W
                Z *= wfrac / fill
            pc = self._distort_pts(pc)
            lo_x, hi_x = margin - pc[0].min(), (W - margin) - pc[0].max()
            lo_y, hi_y = margin - pc[1].min(), (H - margin) - pc[1].max()
            if lo_x <= hi_x and lo_y <= hi_y:
                break
            fill *= 0.97
        cj = P["center_jitter"]
        sx = rng.uniform(lo_x, hi_x) if lo_x <= hi_x else 0.0
        sy = rng.uniform(lo_y, hi_y) if lo_y <= hi_y else 0.0
        mid_x, mid_y = (lo_x + hi_x) / 2, (lo_y + hi_y) / 2
        sx, sy = mid_x + cj * (sx - mid_x), mid_y + cj * (sy - mid_y)
        Timg = np.array([[1, 0, sx], [0, 1, sy], [0, 0, 1.0]])
        self.Hdc = Timg @ Hm  # display px -> ideal camera px
        self.Hcd = np.linalg.inv(self.Hdc)
        self.Rm, self.Z, self.K = Rm, Z, K
        # plane depth: z = (Rm row2) . [u-Wd/2, v-Hd/2, 0] + Z
        self.depth_coef = (Rm[2, 0], Rm[2, 1], Z - Rm[2, 0] * Wd / 2 - Rm[2, 1] * Hd / 2)
        bez = _u(rng, P["bezel"]) * Wd
        self.bezel_px = [bez, bez * rng.uniform(0.8, 1.2), bez, bez * rng.uniform(1.0, 2.6)]  # L T R B
        self.monitor_kind = "laptop" if rng.random() < 0.4 else "monitor"
        dc = self._distort_pts((self.Hdc @ corners)[:2] / (self.Hdc @ corners)[2])
        self.summary["geometry"] = dict(
            fill_width=float(np.ptp(dc[0]) / W), fill_height=float(np.ptp(dc[1]) / H),
            yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, hfov_deg=hfov, focal_px=f, k1=k1, k2=k2,
            bezel_frac=bez / Wd, monitor=self.monitor_kind, display_corners_px=dc.T.round(1).tolist(),
        )

    def _distort_pts(self, p):
        x, y = (p[0] - self.cx) / self.f, (p[1] - self.cy) / self.f
        r2 = x * x + y * y
        s = 1 + self.k1 * r2 + self.k2 * r2 * r2
        return np.array([self.cx + x * s * self.f, self.cy + y * s * self.f])

    def _build_static_maps(self, rng, P):
        W, H, Wd, Hd = self.W, self.H, self.Wd, self.Hd
        f, cx, cy = self.f, self.cx, self.cy
        xs = 2 * np.arange(W // 2) + 0.5
        ys = 2 * np.arange(H // 2) + 0.5
        XD, YD = np.meshgrid(xs, ys)
        xd, yd = (XD - cx) / f, (YD - cy) / f
        xu, yu = xd.copy(), yd.copy()
        for _ in range(12):
            r2 = xu * xu + yu * yu
            s = 1 + self.k1 * r2 + self.k2 * r2 * r2
            xu, yu = xd / s, yd / s
        self.XU, self.YU = xu * f, yu * f  # relative to (cx, cy)

        # canvas extent (display coords covering the whole camera view + shake margin)
        marg = 1.0 + 0.02 + (3 * P["shake_px"][1] + P["jump_px"][1]) / W
        bx = np.concatenate([self.XU[0], self.XU[-1], self.XU[:, 0], self.XU[:, -1]]) * marg + cx
        by = np.concatenate([self.YU[0], self.YU[-1], self.YU[:, 0], self.YU[:, -1]]) * marg + cy
        q = self.Hcd @ np.vstack([bx, by, np.ones_like(bx)])
        u, v = q[0] / q[2], q[1] / q[2]
        umin, umax = max(u.min(), -3 * Wd) - 8, min(u.max(), 4 * Wd) + 8
        vmin, vmax = max(v.min(), -3 * Hd) - 8, min(v.max(), 4 * Hd) + 8
        umin, vmin = min(umin, -self.bezel_px[0] - 8), min(vmin, -self.bezel_px[1] - 8)
        umax, vmax = max(umax, Wd + self.bezel_px[2] + 8), max(vmax, Hd + self.bezel_px[3] + 8)
        self.ox, self.oy = int(math.ceil(-umin)), int(math.ceil(-vmin))
        cw, ch = int(math.ceil(umax)) + self.ox, int(math.ceil(vmax)) + self.oy
        cw, ch = cw + cw % 2, ch + ch % 2
        disp_rect = (self.ox, self.oy, self.ox + Wd, self.oy + Hd)
        L, T, Rr, B = self.bezel_px
        bezel_rect = (self.ox - L, self.oy - T, self.ox + Wd + Rr, self.oy + Hd + B)
        bg_lin = _render_background(rng, cw, ch, disp_rect, bezel_rect, self.monitor_kind)
        self.bg_mean = float(bg_lin.mean())
        self.canvas = cv2.resize(_invert_lut(self.lut, bg_lin), (cw, ch), interpolation=cv2.INTER_LINEAR)
        del bg_lin
        self.mask_canvas = np.zeros((ch, cw), np.uint8)
        self.mask_canvas[self.oy:self.oy + Hd, self.ox:self.ox + Wd] = 255
        self.summary["geometry"]["canvas_size"] = [cw, ch]

        # static maps (no shake) at full res for vignette / DOF / viewing angle
        mx, my = self._maps(np.zeros(H // 2), np.zeros(H // 2), np.zeros(H // 2), 1.0)
        u0, v0 = mx - self.ox, my - self.oy
        inside = ((u0 >= 0) & (u0 < Wd) & (v0 >= 0) & (v0 < Hd)).astype(np.float32)
        self.fill_area = float(inside.mean())
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        rr = ((xx - cx) ** 2 + (yy - cy) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)
        vign = 1 - self.vig * rr * (0.75 + 0.25 * rr)
        # LCD viewing-angle falloff across the tilted screen
        vf = _u(rng, P["view_falloff"])
        a = rng.uniform(0, 2 * np.pi)
        ramp = (np.cos(a) * (u0 / Wd - 0.5) + np.sin(a) * (v0 / Hd - 0.5)).astype(np.float32)
        view = 1 - vf * np.clip(ramp + 0.5, 0, 1.5) * inside
        self.static_gain = np.ascontiguousarray((vign * view).astype(np.float32))
        # depth-of-field weight: 0 at focus distance (screen centre), 1 far from it / background
        dc0, dc1, dc2 = self.depth_coef
        dz = dc0 * u0 + dc1 * v0 + dc2
        zc = dc0 * Wd / 2 + dc1 * Hd / 2 + dc2
        zr = max(abs(dc0 * uu + dc1 * vv + dc2 - zc) for uu in (0, Wd) for vv in (0, Hd))
        wdof = np.clip(np.abs(dz - zc) / max(zr, 0.08 * zc), 0, 1) * inside + (1 - inside)
        self.wdof = np.ascontiguousarray(cv2.GaussianBlur(wdof.astype(np.float32), (0, 0), 8))
        # Bayer lattice (RGGB): channel index (BGR) & subpixel phase (R,G,B stripes -> 0,1/3,2/3)
        lat = np.empty((H, W), np.float32)
        lat[0::2, 0::2] = 0.0
        lat[0::2, 1::2] = 2 * np.pi / 3
        lat[1::2, 0::2] = 2 * np.pi / 3
        lat[1::2, 1::2] = 4 * np.pi / 3
        self.lat_phase = lat
        # moire amplitude attenuated by camera pixel aperture when subpixels are tiny
        J = self.Hdc[:2, :2]
        disp_px_in_cam = math.sqrt(abs(np.linalg.det(J / (self.Hdc[2] @ [Wd / 2, Hd / 2, 1])))) / self.pitch
        self.moire_eff = self.moire_amp * min(1.0, disp_px_in_cam)
        self.summary["display"]["native_px_in_camera_px"] = disp_px_in_cam

        # glare map (quarter res, linear, relative to screen white)
        gh, gw = H // 4, W // 4
        g = np.zeros((gh, gw), np.float32)
        if self.has_glare:
            gy, gx = np.mgrid[0:gh, 0:gw].astype(np.float32)
            dcx = np.array(self.summary["geometry"]["display_corners_px"])
            for _ in range(int(rng.integers(1, 3))):
                px = rng.uniform(dcx[:, 0].min(), dcx[:, 0].max()) / 4
                py = rng.uniform(dcx[:, 1].min(), dcx[:, 1].max()) / 4
                sxg, syg = rng.uniform(0.06, 0.3) * gw, rng.uniform(0.05, 0.25) * gw
                ang = rng.uniform(0, np.pi)
                ca, sa = math.cos(ang), math.sin(ang)
                X, Y = gx - px, gy - py
                d2 = ((ca * X + sa * Y) / sxg) ** 2 + ((-sa * X + ca * Y) / syg) ** 2
                g += self.glare_int * rng.uniform(0.5, 1.0) * np.exp(-0.5 * d2)
            if rng.random() < 0.6:  # soft window reflection band
                band = np.zeros((gh, gw), np.float32)
                x0 = rng.uniform(0.1, 0.8) * gw
                wdt = rng.uniform(0.08, 0.25) * gw
                sk = rng.uniform(-0.6, 0.6)
                xs_ = gx - x0 - sk * gy
                band[(xs_ > 0) & (xs_ < wdt)] = 1.0
                band = cv2.GaussianBlur(band, (0, 0), rng.uniform(3, 12))
                g += band * self.glare_int * rng.uniform(0.2, 0.5)
        g += self.veil
        self.glare_q = g
        self.glare_color = np.exp(rng.normal(0, 0.08, 3)).astype(np.float32)

    def _maps(self, tx, ty, rot, zoom):
        """Full-res float32 remap maps (camera px -> canvas px) for per-half-row shake."""
        c = np.cos(rot)[:, None]
        s = np.sin(rot)[:, None]
        dx = self.XU - tx[:, None]
        dy = self.YU - ty[:, None]
        x0 = self.cx + (c * dx + s * dy) / zoom
        y0 = self.cy + (-s * dx + c * dy) / zoom
        G = self.Hcd
        den = G[2, 0] * x0 + G[2, 1] * y0 + G[2, 2]
        u = (G[0, 0] * x0 + G[0, 1] * y0 + G[0, 2]) / den + self.ox
        v = (G[1, 0] * x0 + G[1, 1] * y0 + G[1, 2]) / den + self.oy
        size = (self.W, self.H)
        mx = cv2.resize(u.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
        my = cv2.resize(v.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
        return mx, my

    def _build_tone_lut(self):
        x = np.arange(256) / 255.0
        c = self.tone_contrast
        s = x + c * (x - 0.5) * (1 - np.abs(2 * x - 1)) * 1.0  # gentle S-curve
        s = np.clip(s, 0, 1)
        self.tone_lut = np.clip(np.round(s * 255), 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ temporal
    def display_time(self, j: int) -> float:
        """Time at which input frame j starts to be shown (quantised to refresh, with glitches)."""
        while len(self.glitches) <= j:
            self.glitches.append(bool(self.r_tmp.random() < self.glitch_p) if self.glitch_p > 0 else False)
        t = math.ceil((j / self.fps_in) * DISPLAY_REFRESH - 1e-6) / DISPLAY_REFRESH
        if self.glitches[j] and j > 0:
            t += 1.0 / DISPLAY_REFRESH
        return t

    def frame_at(self, t: float) -> int:
        j = max(0, int(math.floor(t * self.fps_in)))
        while j > 0 and self.display_time(j) > t:
            j -= 1
        while self.display_time(j + 1) <= t:
            j += 1
        return j

    def shake_at(self, t):
        """(tx, ty, rot_rad, zoom) arrays for times t."""
        t = np.atleast_1d(np.asarray(t, float))
        sv = self.sig.at(t)
        mod = 0.6 + 0.4 * np.tanh(sv[:, 10])
        tx = self.shake_px * sv[:, 0] + self.tremor_px * mod * np.sin(2 * np.pi * self.tremor_hz[0] * t + self.tremor_ph[0])
        ty = self.shake_px * sv[:, 1] + self.tremor_px * mod * np.sin(2 * np.pi * self.tremor_hz[1] * t + self.tremor_ph[1])
        rot = np.radians(self.P["shake_rot_deg"]) * sv[:, 2]
        zoom = 1 + self.P["shake_zoom"] * sv[:, 3]
        for e in self.jumps.upto(float(t.max())):
            if e["t0"] > t.max():
                continue
            prof = _smoothstep((t - e["t0"]) / e["dur"]) * np.exp(-np.clip(t - e["t0"] - e["dur"], 0, None) / e["ret"])
            tx = tx + e["dx"] * prof
            ty = ty + e["dy"] * prof
            rot = rot + np.radians(e["drot"]) * prof
        return tx, ty, rot, zoom

    # ------------------------------------------------------------------ per-frame job
    def make_job(self, k, frames, n_total, disp_means):
        """Build the job for camera frame k.  frames: dict j->array (must contain the needed ones)."""
        H = self.H
        t_k = self.phase + k / self.cam_fps + self.r_tmp.normal(0, self.clock_jitter)
        a = t_k + self.row_frac * self.R  # per-row exposure start
        b = a + self.E
        j_lo = self.frame_at(max(0.0, a.min() - self.rho - 1e-3))
        j_hi = self.frame_at(b.max())
        rho = self.rho

        def G(x):
            if rho <= 1e-6:
                return np.maximum(x, 0)
            return np.where(x <= 0, 0.0, np.where(x < rho, x * x / (2 * rho), x - rho / 2))

        contribs = []
        for j in range(j_lo, min(j_hi, n_total - 1) + 1):
            Tj = -1e6 if j == 0 else self.display_time(j)
            Tn = 1e6 if j >= n_total - 1 else self.display_time(j + 1)
            w = (G(b - Tj) - G(a - Tj) - G(b - Tn) + G(a - Tn)) / self.E
            nz = np.nonzero(w > 1e-4)[0]
            if len(nz) == 0:
                continue
            r0, r1 = int(nz[0]), int(nz[-1]) + 1
            contribs.append([j, w.astype(np.float32), r0, r1])
        tot = np.zeros(H, np.float32)
        for c in contribs:
            tot += c[1]
        tot = np.maximum(tot, 1e-6)
        for c in contribs:
            c[1] = c[1] / tot
            c[0] = frames[c[0]]

        # shake per half-res row at mid-exposure
        t_rows = t_k + self.row_frac_half * self.R + self.E / 2
        tx, ty, rot, zoom = self.shake_at(t_rows)
        tc = t_k + 0.5 * self.R
        p0 = self.shake_at([tc])
        p1 = self.shake_at([tc + self.E])
        motion = (float(p1[0][0] - p0[0][0]), float(p1[1][0] - p0[1][0]))

        sv = self.sig.at([tc])[0]
        sigma = self.sigma0 * (1 + self.P["focus_breath"] * np.tanh(sv[4]))
        for e in self.hunts.upto(tc):
            if e["t0"] <= tc <= e["t0"] + e["dur"]:
                sigma *= 1 + e["mag"] * math.sin(math.pi * (tc - e["t0"]) / e["dur"])

        # auto exposure (lagged) on centre-weighted scene mean
        jm = self.frame_at(tc + self.E / 2)
        dm = disp_means.get(min(jm, n_total - 1), 0.3)
        scene = self.fill_area * dm + (1 - self.fill_area) * self.bg_mean
        white = float(self.lut[0, 255].mean())
        g_t = min(self.ae_target / max(scene, 1e-3), self.P["clip_allow"] / white, 8.0)
        if self.ae_gain is None:
            self.ae_gain = g_t
        else:
            self.ae_gain += (g_t - self.ae_gain) * (1 - math.exp(-(1 / self.cam_fps) / self.ae_tau))
        ae = self.ae_gain * math.exp(self.P["ae_drift"] * sv[5])
        wb = self.wb_base * np.exp(self.P["wb_drift"] * np.array([sv[6], 0.0, sv[7]]))
        gain_bgr = (ae * wb).astype(np.float32)

        rowgain = None
        if self.flicker_amp > 0:
            w_ = 2 * np.pi * self.flicker_hz
            ar = a + self.flicker_ph / w_
            avg = (np.cos(w_ * ar) - np.cos(w_ * (ar + self.E))) / (w_ * self.E)
            rowgain = (1 + self.flicker_amp * avg).astype(np.float32)

        glare_shift = (float(-0.6 * tx.mean() + 20 * sv[8]) / 4, float(-0.6 * ty.mean() + 20 * sv[9]) / 4)
        return _FrameJob(
            k=k, contribs=contribs, tx=tx, ty=ty, rot=rot, zoom=float(np.mean(zoom)), motion=motion,
            sigma=float(sigma), gain_bgr=gain_bgr, glare_shift=glare_shift, glare_scale=1.0,
            rowgain=rowgain, noise_seed=int(self.frame_ss.generate_state(1)[0] ^ (k * 2654435761 & 0xFFFFFFFF)),
            extra=dict(t=t_k, n_contrib=len(contribs)),
        )

    # ------------------------------------------------------------------ render (thread-safe)
    def render(self, job: _FrameJob) -> np.ndarray:
        W, H = self.W, self.H
        tls = self._tls
        if getattr(tls, "canvas", None) is None:
            tls.canvas = self.canvas.copy()
        canvas = tls.canvas
        mx, my = self._maps(job.tx, job.ty, job.rot, job.zoom)

        # rolling-shutter / exposure blend of display frames, in linear light
        acc = np.zeros((H, W, 3), np.float32)
        oy, ox = self.oy, self.ox
        for frame, w, r0, r1 in job.contribs:
            canvas[oy:oy + self.Hd, ox:ox + self.Wd] = frame
            rem = cv2.remap(canvas, mx[r0:r1], my[r0:r1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            lin = cv2.LUT(rem, self.lut)
            ws = w[r0:r1]
            if ws.min() < 0.9999:
                np.multiply(lin, ws[:, None, None], out=lin)
            acc[r0:r1] += lin

        # motion blur from camera motion during the exposure
        dx, dy = job.motion
        dist = math.hypot(dx, dy)
        if dist >= 0.8:
            n = min(int(math.ceil(dist / 0.7)) + 1, 32)
            out = np.zeros_like(acc)
            for i in range(n):
                s = i / (n - 1) - 0.5
                M = np.float32([[1, 0, s * dx], [0, 1, s * dy]])
                out += cv2.warpAffine(acc, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            acc = out / n

        # defocus, depth varying
        s0 = max(0.3, job.sigma)
        b0 = cv2.GaussianBlur(acc, (0, 0), s0)
        if self.dof > 0.02:
            b1 = cv2.GaussianBlur(acc, (0, 0), s0 * (1 + self.dof))
            b1 -= b0
            b1 *= self.wdof[:, :, None]
            b0 += b1
        del acc

        # Bayer mosaic (RGGB); channel order BGR
        raw = np.empty((H, W), np.float32)
        raw[0::2, 0::2] = b0[0::2, 0::2, 2]
        raw[0::2, 1::2] = b0[0::2, 1::2, 1]
        raw[1::2, 0::2] = b0[1::2, 0::2, 1]
        raw[1::2, 1::2] = b0[1::2, 1::2, 0]
        del b0

        # screen sub-pixel structure sampled by the sensor -> moire
        if self.moire_eff > 1e-4:
            mask = cv2.remap(self.mask_canvas, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            mag = mask.astype(np.float32)
            mag *= self.moire_eff / 255.0
            ang = mx - ox
            ang *= 2 * np.pi * self.pitch
            ang += self.lat_phase
            cu, _ = cv2.polarToCart(mag, ang)
            angv = my - oy
            angv *= 2 * np.pi * self.pitch
            mag *= 0.35
            cv_, _ = cv2.polarToCart(mag, angv)
            cu += cv_
            cu += 1.0
            raw *= cu

        # glare (reflection on the glass; not modulated by sub-pixels)
        if self.glare_q.max() > 0:
            M = np.float32([[1, 0, job.glare_shift[0]], [0, 1, job.glare_shift[1]]])
            gq = cv2.warpAffine(self.glare_q, M, (W // 4, H // 4), borderMode=cv2.BORDER_REFLECT)
            gfull = cv2.resize(gq, (W, H), interpolation=cv2.INTER_LINEAR)
            gc = self.glare_color * float(self.lut[0, 255].mean())
            gfull[0::2, 0::2] *= gc[2]
            gfull[1::2, 1::2] *= gc[0]
            gfull[0::2, 1::2] *= gc[1]
            gfull[1::2, 0::2] *= gc[1]
            raw += gfull

        # exposure, white balance, vignetting, flicker
        raw *= self.static_gain
        g = job.gain_bgr
        raw[0::2, 0::2] *= g[2]
        raw[1::2, 1::2] *= g[0]
        raw[0::2, 1::2] *= g[1]
        raw[1::2, 0::2] *= g[1]
        if job.rowgain is not None:
            raw *= job.rowgain[:, None]

        # sensor noise
        rng = np.random.default_rng(job.noise_seed)
        std = np.maximum(raw, 0)
        std *= self.shot
        std += self.read * self.read
        cv2.sqrt(std, std)
        noise = rng.standard_normal((H, W), dtype=np.float32)
        noise *= std
        raw += noise
        if self.row_noise > 0:
            raw += (rng.standard_normal(H).astype(np.float32) * self.row_noise)[:, None]

        # ISP: gamma, demosaic, sharpen, saturation, tone
        np.clip(raw, 0.0, 1.0, out=raw)
        cv2.pow(raw, 1 / 2.2, raw)
        raw *= 255.0
        raw += 0.5
        raw8 = raw.astype(np.uint8)
        bgr = cv2.cvtColor(raw8, cv2.COLOR_BayerBG2BGR_EA)
        if self.sharpen > 0:
            bl = cv2.GaussianBlur(bgr, (0, 0), self.sharpen_sigma)
            bgr = cv2.addWeighted(bgr, 1 + self.sharpen, bl, -self.sharpen, 0)
        if abs(self.saturation - 1) > 1e-3:
            gray = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            bgr = cv2.addWeighted(bgr, self.saturation, gray, 1 - self.saturation, 0)
        bgr = cv2.LUT(bgr, self.tone_lut)
        return bgr


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def simulate(in_path, out_path, preset="phone", seed=0, start=0.0, duration=None, cam_fps=None,
             out_size=(1920, 1080), progress=False, dump_frames=None, dump_every=30, workers=None,
             encoder_threads=1, bitrate=None, display_width=None) -> dict:
    """Simulate a phone recording of `in_path` played on a monitor; write H.264 MP4 to `out_path`.

    dump_frames/dump_every: write every N-th frame of the *encoded* output as PNG into that dir.
    workers: render threads (default: cpu_count-4, clamped to 2..8).
    encoder_threads: libx264 threads.  1 (default) makes the output bit-exact for a given seed
    (x264 multi-threading is not deterministic here); 0 = auto (faster, not bit-exact).

    Returns a JSON-able summary dict of the sampled channel parameters (+ timing)."""
    t_start = time.perf_counter()
    info = probe(in_path)
    sim = CamSim(info, preset, seed, tuple(out_size), cam_fps, display_width)
    W, H = sim.W, sim.H
    if dump_frames:
        os.makedirs(dump_frames, exist_ok=True)
    workers = workers or max(2, min(8, (os.cpu_count() or 4) - 4))
    reader = _FrameReader(in_path, info["width"], info["height"], start, duration)
    # Preset bitrates are tuned for a 1080p30 recording. A phone raises its bitrate with the pixel rate
    # (4K60 records at ~5-8x the 1080p30 rate), so scale likewise unless an explicit bitrate is given.
    if bitrate is None:
        scale = (W * H * sim.cam_fps) / (1920 * 1080 * 30.0)
        bitrate = int(_parse_rate(sim.P["bitrate"]) * max(1.0, scale))
    sim.summary.setdefault("encoder", {})["bitrate_used"] = int(_parse_rate(bitrate)) if isinstance(bitrate, str) else int(bitrate)
    enc = _open_encoder(out_path, W, H, sim.cam_fps, bitrate, sim.P["x264_preset"], encoder_threads)
    frames: dict[int, np.ndarray] = {}
    disp_means: dict[int, float] = {}
    n_read = 0
    eof = False
    lut_mean = sim.lut[0].mean(axis=1).astype(np.float32)  # code -> mean linear

    def read_one():
        nonlocal n_read, eof
        fr = reader.read()
        if fr is None:
            eof = True
            return False
        frames[n_read] = fr
        small = cv2.resize(fr, (32, 18), interpolation=cv2.INTER_AREA)
        disp_means[n_read] = float(lut_mean[small].mean())
        n_read += 1
        return True

    pending: deque = deque()
    n_out = 0
    t_setup = time.perf_counter() - t_start
    t_loop = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            k = 0
            while True:
                t_k = sim.phase + k / sim.cam_fps
                need = sim.frame_at(t_k + sim.R + sim.E + 0.005) + 1
                while not eof and n_read <= need:
                    read_one()
                if n_read == 0:
                    raise RuntimeError("no frames decoded from input")
                total = n_read if eof else 10**9
                if eof and t_k >= n_read / sim.fps_in:
                    break
                job = sim.make_job(k, frames, total, disp_means)
                pending.append((k, pool.submit(sim.render, job)))
                # drop frames no longer needed
                keep_from = sim.frame_at(max(0.0, t_k - 0.1)) - 1
                for j in [j for j in frames if j < keep_from]:
                    del frames[j]
                    disp_means.pop(j, None)
                while len(pending) >= 2 * workers or (pending and pending[0][1].done() and len(pending) > workers):
                    n_out += _emit(pending, enc, dump_frames, dump_every)
                if progress and k % 30 == 0:
                    el = time.perf_counter() - t_loop
                    print(f"[camsim] frame {k} ({n_out / max(el, 1e-6):.1f} fps)", file=sys.stderr, flush=True)
                k += 1
            while pending:
                n_out += _emit(pending, enc, dump_frames, dump_every)
    finally:
        reader.close()
        try:
            enc.stdin.close()
        except Exception:
            pass
        rc = enc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg encoder failed with code {rc}")
    el = time.perf_counter() - t_loop
    if dump_frames and dump_every > 0:  # snapshots of the *encoded* output (what a decoder sees)
        rd = _FrameReader(out_path, W, H, 0.0, None)
        try:
            i = 0
            while (img := rd.read()) is not None:
                if i % dump_every == 0:
                    cv2.imwrite(os.path.join(dump_frames, f"frame_{i:05d}.png"), img)
                i += 1
        finally:
            rd.close()
    sim.summary["temporal"]["pacing_glitches"] = int(sum(sim.glitches[:n_read]))
    sim.summary["frames_in"] = n_read
    sim.summary["frames_out"] = n_out
    sim.summary["timing"] = dict(setup_s=round(t_setup, 3), render_s=round(el, 3),
                                 fps=round(n_out / max(el, 1e-9), 2), workers=workers)
    return _jsonable(sim.summary)


def _emit(pending, enc, dump_dir, dump_every):
    k, fut = pending.popleft()
    img = fut.result()
    enc.stdin.write(img.tobytes())
    return 1


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float):
        return round(o, 6)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m stega.camsim", description=__doc__.split("\n\n")[0])
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--preset", default="phone", choices=sorted(PRESETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--cam-fps", type=float, default=None)
    ap.add_argument("--display-width", type=int, default=None,
                    help="native width of the simulated panel (default: drawn from the preset); "
                         "set it to the video width to model 1:1 playback")
    ap.add_argument("--size", default="1920x1080", help="output WxH (even)")
    ap.add_argument("--dump-frames", default=None, metavar="DIR", help="write PNG snapshots of output frames")
    ap.add_argument("--dump-every", type=int, default=30, metavar="N")
    ap.add_argument("--workers", type=int, default=None, help="render threads")
    ap.add_argument("--encoder-threads", type=int, default=1,
                    help="libx264 threads; 1 = bit-exact reproducible (default), 0 = auto/faster")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    w, h = (int(x) for x in a.size.lower().split("x"))
    summary = simulate(a.input, a.output, preset=a.preset, seed=a.seed, start=a.start, duration=a.duration,
                       cam_fps=a.cam_fps, out_size=(w, h), progress=not a.quiet, dump_frames=a.dump_frames,
                       dump_every=a.dump_every, workers=a.workers, encoder_threads=a.encoder_threads,
                       display_width=a.display_width)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
