"""file -> mp4 -> (simulated phone recording) -> file. Slow-ish (tens of seconds)."""

import os

import pytest

from stega.decoder import decode_video
from stega.encoder import encode_file


def test_encode_decode_direct(tmp_path):
    src = tmp_path / "payload.bin"
    src.write_bytes(os.urandom(40_000))
    video = tmp_path / "code.mp4"
    encode_file(src, video, "balanced", lead_in=0.5)
    rep = decode_video(video, tmp_path / "out.bin", workers=4)
    assert rep.ok, rep.error
    assert (tmp_path / "out.bin").read_bytes() == src.read_bytes()


@pytest.mark.parametrize("preset", ["mild", "phone"])
def test_encode_camsim_decode(tmp_path, preset):
    camsim = pytest.importorskip("stega.camsim")
    src = tmp_path / "notes.txt"
    src.write_bytes(os.urandom(30_000))
    video = tmp_path / "code.mp4"
    encode_file(src, video, "balanced", lead_in=1.0, overhead=0.4)
    rec = tmp_path / "rec.mp4"
    camsim.simulate(str(video), str(rec), preset=preset, seed=11)
    rep = decode_video(rec, tmp_path, workers=6)
    assert rep.ok, (rep.error, rep.stats, rep.symbols)
    assert (tmp_path / "notes.txt").read_bytes() == src.read_bytes()


def test_multi_band_4k_roundtrip(tmp_path):
    """uhd: 8 independently decodable bands per frame, decoded by the real multi-process decoder."""
    src = tmp_path / "big.bin"
    src.write_bytes(os.urandom(300_000))
    video = tmp_path / "uhd.mp4"
    rep = encode_file(src, video, "uhd", lead_in=0.3, overhead=0.12)
    assert rep.code_frames < 30
    dec = decode_video(video, tmp_path / "out.bin", workers=4)
    assert dec.ok, (dec.error, dec.stats)
    assert (tmp_path / "out.bin").read_bytes() == src.read_bytes()
    assert dec.stats.get("bands_ok", 0) > 0
