import os
import random

import cv2
import numpy as np
import pytest

from stega import container
from stega.detector import read_frame
from stega.fountain import FountainDecoder, FountainEncoder
from stega.framing import FrameHeader, build_codec
from stega.profiles import PROFILES
from stega.render import render_frame


def test_container_roundtrip_and_integrity():
    data = b"hello " * 1000
    blob = container.pack("a.txt", data)
    assert len(blob) < len(data)  # compressed
    p = container.unpack(blob)
    assert p.name == "a.txt" and p.data == data
    bad = bytearray(blob)
    bad[-1] ^= 1
    with pytest.raises(Exception):
        container.unpack(bytes(bad))


@pytest.mark.parametrize("n_bytes", [1, 5000, 300_000])
def test_fountain_recovers_from_random_subset(n_bytes):
    blob = os.urandom(n_bytes)
    enc = FountainEncoder(blob, symbol_size=1000, gen_size=64, session=42)
    order = enc.schedule(overhead=0.3)
    rng = random.Random(n_bytes)
    rng.shuffle(order)
    kept = order[: int(len(order) * 0.85)]  # lose 15% of the frames, in random order
    dec = FountainDecoder(enc.params, session=42)
    for gen, esi in kept:
        dec.add(gen, esi, enc.symbol(gen, esi))
        if dec.done:
            break
    assert dec.done
    assert dec.recover() == blob


def test_fountain_repair_only():
    blob = os.urandom(20_000)
    enc = FountainEncoder(blob, symbol_size=500, gen_size=16, session=7)
    dec = FountainDecoder(enc.params, session=7)
    for g in range(enc.params.n_gens):
        k = enc.params.gen_k(g)
        esi = k
        while not dec.gens[g].done:  # no systematic symbols at all
            dec.add(g, esi, enc.symbol(g, esi))
            esi += 1
        assert esi - k <= k + 12
    assert dec.recover() == blob


@pytest.mark.parametrize("name", list(PROFILES))
def test_frame_codec_corrects_byte_errors(name):
    codec = build_codec(name)
    sym = os.urandom(codec.symbol_size)
    hdr = FrameHeader(codec.layout.profile.pid, 1, 2, 256, 3, 4)
    cells = codec.encode(hdr, sym)
    from stega.framing import symbols_to_bits

    bpc = codec.layout.profile.bits_per_cell
    stream = np.packbits(symbols_to_bits(cells, bpc)[: codec.layout.capacity_bytes * 8])
    rng = np.random.default_rng(0)
    # exactly 60% of one codeword's correction capacity in EVERY block (uniform random placement would
    # make many-block frames, e.g. 119 blocks at 4K, fail on Poisson tails rather than on capacity)
    per_block = int(0.6 * codec.nsym / 2)
    block_of = np.empty(len(stream), np.int64)  # stream position -> RS block index
    for b, s in enumerate(codec.block_slices()):
        block_of[np.isin(codec.interleave, np.arange(s.start, s.stop))] = b
    for b in range(len(codec.block_lens)):
        idx = rng.choice(np.flatnonzero(block_of == b), per_block, replace=False)
        stream[idx] ^= rng.integers(1, 256, per_block, dtype=np.uint8)
    out = codec.decode(stream, np.ones(len(stream), np.float32))
    assert out is not None
    assert out[0] == hdr and out[1] == sym


def _perspective_capture(img, seed):
    r = np.random.default_rng(seed)
    h, w = img.shape[:2]
    W, H = w, h  # a camera of the screen's own resolution (1080p screen -> 1080p, 4K screen -> 4K)
    px = W / 1920  # geometric jitter scales with resolution; blur/noise stay in camera pixels
    sc = r.uniform(0.7, 0.85)
    base = np.float32([[W * (1 - sc) / 2, H * (1 - sc) / 2], [W * (1 + sc) / 2, H * (1 - sc) / 2],
                       [W * (1 + sc) / 2, H * (1 + sc) / 2], [W * (1 - sc) / 2, H * (1 + sc) / 2]])  # fmt: skip
    dst = base + r.uniform(-70 * px, 70 * px, (4, 2)).astype(np.float32)
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst)
    out = cv2.warpPerspective(img, M, (W, H), borderValue=(30, 30, 30))
    # mild barrel distortion
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx, ny = (xx - W / 2) / (W / 2), (yy - H / 2) / (W / 2)
    k = 1 + 0.03 * (nx**2 + ny**2)
    out = cv2.remap(out, W / 2 + nx * k * W / 2, H / 2 + ny * k * W / 2, cv2.INTER_LINEAR)
    out = cv2.GaussianBlur(out, (0, 0), 1.1)
    out = out.astype(np.float32) * 0.75 + 25 + r.normal(0, 5, out.shape)
    ok, jpg = cv2.imencode(".jpg", np.clip(out, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return cv2.imdecode(jpg, cv2.IMREAD_UNCHANGED)


@pytest.mark.parametrize("name", list(PROFILES))
@pytest.mark.parametrize("rot", [0, 1, 2])
def test_frame_survives_synthetic_capture(name, rot):
    codec = build_codec(name)
    sym = os.urandom(codec.symbol_size)
    hdr = FrameHeader(codec.layout.profile.pid, 99, 1, 256, 0, 5)
    img = render_frame(codec.layout, codec.encode(hdr, sym))
    cap = _perspective_capture(img, seed=rot)
    if rot == 1:
        cap = cv2.rotate(cap, cv2.ROTATE_90_CLOCKWISE)  # phone held in portrait
    elif rot == 2:
        cap = cv2.flip(cap, 1)  # mirrored (front camera)
    r = read_frame(cap, list(PROFILES))
    assert r.status == "ok", (r.status, r.pn_score, r.fixed_ber)
    assert r.profile == name and r.header == hdr and r.symbol == sym


@pytest.mark.parametrize("name", ["balanced", "color4"])
@pytest.mark.parametrize("hidden", [0, 2])
def test_frame_with_one_finder_occluded(name, hidden):
    codec = build_codec(name)
    lay = codec.layout
    sym = os.urandom(codec.symbol_size)
    hdr = FrameHeader(lay.profile.pid, 5, 1, 256, 1, 2)
    img = render_frame(lay, codec.encode(hdr, sym))
    # paint over one finder (e.g. glare or a finger), then capture
    cx, cy = lay.finder_centers[hidden] * lay.profile.cell_px
    r = int(4.5 * lay.profile.cell_px)
    img[int(cy) - r : int(cy) + r, int(cx) - r : int(cx) + r] = 128
    cap = _perspective_capture(img, seed=hidden + 10)
    res = read_frame(cap, list(PROFILES))
    assert res.status == "ok", (res.status, res.pn_score, res.fixed_ber)
    assert res.symbol == sym


def _gray_profile(monkeypatch):
    from stega.profiles import Profile

    p = Profile("t_gray4", pid=200, cell_px=12, bits_per_cell=2, ecc_nsym=48, hold_frames=3, palette="gray")
    monkeypatch.setitem(PROFILES, p.name, p)  # in-process only: spawned decoder workers never see it
    return p.name


@pytest.mark.parametrize("bpc", [2, 3])
def test_gray_palette_is_gray_coded(bpc):
    from stega.render import palette

    P = palette(bpc, "gray")
    by_brightness = np.argsort(P[:, 0])
    for a, b in zip(by_brightness, by_brightness[1:]):
        assert bin(int(a) ^ int(b)).count("1") == 1  # adjacent luma levels differ in exactly one bit
    assert np.allclose(P[:, 0], P[:, 1]) and np.allclose(P[:, 1], P[:, 2])  # pure luma, no chroma


def test_gray_palette_survives_blurred_capture(monkeypatch):
    """Multi-level cells need the equaliser strength matched to the actual blur (a fixed strong
    equaliser throws mid levels to the extremes); the decoder estimates it from known cells."""
    from stega.detector import estimate_beta

    name = _gray_profile(monkeypatch)
    codec = build_codec(name)
    hdr = FrameHeader(codec.layout.profile.pid, 7, 1, 256, 0, 3)
    sym = os.urandom(codec.symbol_size)
    img = render_frame(codec.layout, codec.encode(hdr, sym))
    betas = []
    for sigma in (0.0, 2.5):
        cap = _perspective_capture(img, seed=4)
        if sigma:
            cap = cv2.GaussianBlur(cap, (0, 0), sigma)
        r = read_frame(cap, [name], debug=True)
        assert r.status == "ok" and r.symbol == sym, (sigma, r.status)
        betas.append(estimate_beta(r.debug["N"], codec.layout))
    assert betas[1] > betas[0] + 0.1  # more blur -> stronger equalisation
