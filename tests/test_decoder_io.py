"""Decoder robustness: untrusted names, output paths, mixed sessions, split recordings, zlib bombs."""

import hashlib
import os
import struct
import subprocess
import zlib

import pytest

from stega import container
from stega.decoder import decode_video, safe_name
from stega.encoder import encode_file


@pytest.mark.parametrize(
    "stored, expected",
    [("a.txt", "a.txt"), ("../../x", "x"), ("/etc/passwd", "passwd"), ("..", "stega_output.bin"), ("", "stega_output.bin"),
     ("dir\\evil.bin", "evil.bin"), ("한글 파일.zip", "한글 파일.zip")],
)  # fmt: skip
def test_safe_name(stored, expected):
    assert safe_name(stored) == expected


def test_zlib_bomb_is_bounded():
    body = zlib.compress(bytes(64 << 20), 9)  # 64 MiB of zeros in ~65 KB
    hdr = struct.pack(">4sBBHQ32s", b"STGC", 1, container.FLAG_ZLIB, 1, 10, hashlib.sha256(b"0123456789").digest())
    with pytest.raises(ValueError):
        container.unpack(hdr + b"x" + body)


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *args], check=True)


def _cut(src, dst, start, end):
    _ffmpeg("-i", str(src), "-vf", f"trim=start={start}:end={end},setpts=PTS-STARTPTS", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(dst))


@pytest.fixture(scope="module")
def two_encodes(tmp_path_factory):
    d = tmp_path_factory.mktemp("enc")
    out = {}
    for tag in ("a", "b"):
        src = d / f"{tag}.bin"
        src.write_bytes(os.urandom(20_000))
        encode_file(src, d / f"{tag}.mp4", "balanced", lead_in=0.5)
        out[tag] = (src, d / f"{tag}.mp4")
    return d, out


def test_tail_of_other_session_does_not_block_decode(two_encodes, tmp_path):
    d, enc = two_encodes
    a_part = tmp_path / "a_part.mp4"
    _cut(enc["a"][1], a_part, 0.6, 1.6)  # ~10 code frames of transmission A, incomplete
    mixed = tmp_path / "mixed.mp4"
    _ffmpeg("-i", str(a_part), "-i", str(enc["b"][1]), "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]",
            "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(mixed))  # fmt: skip
    rep = decode_video(mixed, f"{tmp_path}/restored/", workers=4)  # trailing slash: create dir
    assert rep.ok, (rep.error, rep.symbols)
    assert (tmp_path / "restored" / "b.bin").read_bytes() == enc["b"][0].read_bytes()


def test_split_recording_clips_are_combined(two_encodes, tmp_path):
    d, enc = two_encodes
    src, video = enc["b"]
    p1, p2 = tmp_path / "p1.mp4", tmp_path / "p2.mp4"
    _cut(video, p1, 0, 1.8)
    _cut(video, p2, 1.6, 10)
    assert not decode_video(p1, tmp_path / "x1.bin", workers=4).ok
    assert not decode_video(p2, tmp_path / "x2.bin", workers=4).ok
    rep = decode_video([p1, p2], tmp_path / "both.bin", workers=4)
    assert rep.ok, (rep.error, rep.symbols)
    assert (tmp_path / "both.bin").read_bytes() == src.read_bytes()


def test_default_output_never_overwrites(two_encodes, tmp_path, monkeypatch):
    d, enc = two_encodes
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.bin").write_bytes(b"precious")
    rep = decode_video(enc["a"][1], None, workers=4)
    assert rep.ok
    assert (tmp_path / "a.bin").read_bytes() == b"precious"
    assert (tmp_path / "a (1).bin").read_bytes() == enc["a"][0].read_bytes()


def test_reader_does_not_duplicate_frames_of_variable_rate_video(tmp_path):
    """Phone recordings declare a high container rate (Samsung: r_frame_rate=120 for ~30 fps). The reader
    must yield each real frame once, or decode time and `stega check` sampling are off by that factor."""
    from stega.video_io import read_frames

    clip = tmp_path / "mixed_rate.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=60:duration=1",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]", "-fps_mode", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip))  # fmt: skip
    assert sum(1 for _ in read_frames(str(clip))) == 90
