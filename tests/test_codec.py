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
    from stega.framing import symbols_to_bits

    codec = build_codec(name)
    bpc = codec.layout.profile.bits_per_cell
    rng = np.random.default_rng(0)
    for b, unit in enumerate(codec.units):  # every band is an independent RS/CRC unit
        sym = os.urandom(unit.symbol_size)
        hdr = FrameHeader(codec.layout.profile.pid, 1, 2, 256, 3, 4 + b)
        stream = np.packbits(symbols_to_bits(unit.encode(hdr, sym), bpc)[: unit.capacity_bytes * 8])
        # exactly 60% of one codeword's correction capacity in EVERY block (uniform random placement would
        # make many-block units fail on Poisson tails rather than on capacity)
        per_block = int(0.6 * unit.nsym / 2)
        block_of = np.empty(len(stream), np.int64)  # stream position -> RS block index
        for k, sl in enumerate(unit.block_slices()):
            block_of[np.isin(unit.interleave, np.arange(sl.start, sl.stop))] = k
        for k in range(len(unit.block_lens)):
            idx = rng.choice(np.flatnonzero(block_of == k), per_block, replace=False)
            stream[idx] ^= rng.integers(1, 256, per_block, dtype=np.uint8)
        out = unit.decode(stream, np.ones(len(stream), np.float32))
        assert out is not None, b
        assert out[0] == hdr and out[1] == sym


def _perspective_capture(img, seed):
    r = np.random.default_rng(seed)
    h, w = img.shape[:2]
    # the camera: 1080p screens are filmed at 1080p (conservative); denser canvases are meant for a 4K/60
    # phone recording, so they are filmed by a 4K camera (a QHD screen then gets ~1.5 camera px per screen px)
    W, H = (w, h) if w <= 1920 else (3840, 2160)
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
    items = [(FrameHeader(codec.layout.profile.pid, 99, 1, 256, 0, 5 + b), os.urandom(codec.symbol_size)) for b in range(codec.bands)]
    img = render_frame(codec.layout, codec.encode_frame(items))
    cap = _perspective_capture(img, seed=rot)
    if rot == 1:
        cap = cv2.rotate(cap, cv2.ROTATE_90_CLOCKWISE)  # phone held in portrait
    elif rot == 2:
        cap = cv2.flip(cap, 1)  # mirrored (front camera)
    r = read_frame(cap, list(PROFILES))
    assert r.status == "ok", (r.status, r.pn_score, r.fixed_ber)
    assert r.profile == name and sorted(r.payloads, key=lambda p: p[0].esi) == items


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


def test_torn_capture_yields_bands_above_and_below_the_tear(monkeypatch):
    """Rolling shutter: a capture across a code-frame change shows frame t on top and t+1 below.
    With horizontal bands, every band that does not straddle the tear still decodes."""
    from stega.profiles import Profile

    p = Profile("t_band8", pid=201, cell_px=12, bits_per_cell=1, ecc_nsym=40, hold_frames=2, bands=8)
    monkeypatch.setitem(PROFILES, p.name, p)
    codec = build_codec(p.name)
    lay = codec.layout

    def frame(esi0):
        items = [(FrameHeader(p.pid, 9, 1, 256, b, esi0 + b), os.urandom(codec.symbol_size)) for b in range(codec.bands)]
        return items, render_frame(lay, codec.encode_frame(items))

    top_items, top = frame(0)
    bottom_items, bottom = frame(100)
    torn = top.copy()
    cut = top.shape[0] * 45 // 100
    torn[cut:] = bottom[cut:]
    r = read_frame(_perspective_capture(torn, seed=5), [p.name])
    want_top = {(h.gen, h.esi): s for h, s in top_items}
    want_bottom = {(h.gen, h.esi): s for h, s in bottom_items}
    got = {(h.gen, h.esi): s for h, s in r.payloads}
    for key, sym in got.items():  # nothing decoded wrongly
        assert {**want_top, **want_bottom}[key] == sym
    n_top = sum(k in want_top for k in got)
    n_bottom = sum(k in want_bottom for k in got)
    assert n_top >= 2 and n_bottom >= 3 and n_top + n_bottom >= codec.bands - 1, (n_top, n_bottom)


def test_schedule_fills_whole_multi_band_frames():
    blob = os.urandom(123_457)
    enc = FountainEncoder(blob, symbol_size=1000, gen_size=64, session=3)
    for b in (1, 3, 8):
        order = enc.schedule(0.1, multiple_of=b)
        assert len(order) % b == 0 and len(set(order)) == len(order)


# Bitstreams recorded before sub-framing existed. Single-band profiles must never change: recordings of
# videos made with earlier versions (including real phone recordings) have to keep decoding.
SINGLE_BAND_GOLDEN = {
    "robust": "00ed5487d076e21f",
    "balanced": "937d43365c94f7f7",
    "fast": "a06091f79694c714",
    "dense": "5d05e9e8c68e9815",
    "color4": "e8b62869b7261ef3",
    "color8": "7aab77984a4b7740",
}


@pytest.mark.parametrize("name", list(SINGLE_BAND_GOLDEN))
def test_single_band_bitstream_is_unchanged(name):
    import hashlib

    codec = build_codec(name)
    assert codec.bands == 1
    sym = hashlib.shake_128(name.encode()).digest(codec.symbol_size)
    cells = codec.encode(FrameHeader(codec.layout.profile.pid, 0xC0FFEE, 123456, 256, 3, 77), sym)
    assert hashlib.sha256(cells.tobytes()).hexdigest()[:16] == SINGLE_BAND_GOLDEN[name]


def test_generation_size_policy():
    from stega.fountain import generation_size

    assert generation_size(1) == 256 and generation_size(873) == 256  # small files: unchanged
    assert generation_size(65234) == 2048  # 270 MB of uhd3 symbols -> 32 generations, not 255
    for k in (1, 900, 9000, 65234, 10**6):
        g = generation_size(k)
        assert 256 <= g <= 2048 and g & (g - 1) == 0


def test_large_generation_recovers_with_loss():
    blob = os.urandom(2048 * 300 - 7)
    enc = FountainEncoder(blob, symbol_size=300, gen_size=2048, session=11)
    rng = random.Random(5)
    dec = FountainDecoder(enc.params, session=11)
    for gen, esi in enc.schedule(0.15):
        if rng.random() < 0.88:  # 12% of symbols lost
            dec.add(gen, esi, enc.symbol(gen, esi))
    assert dec.done and dec.recover() == blob
