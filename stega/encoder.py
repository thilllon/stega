"""file -> MP4 of code frames."""

from __future__ import annotations

import math
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from . import container
from .fountain import FountainEncoder
from .framing import FrameHeader, build_codec
from .profiles import get_profile
from .render import render_banner, render_frame
from .video_io import VideoWriter

GEN_SIZE = 256


@dataclass
class EncodeReport:
    output: str
    profile: str
    file_bytes: int
    container_bytes: int
    symbol_size: int
    source_symbols: int
    code_frames: int
    video_seconds: float
    net_kbps: float


def plan(profile_name: str, n_bytes: int, overhead: float = 0.25, hold: int | None = None) -> dict:
    """Capacity / duration estimate without encoding anything."""
    p = get_profile(profile_name)
    codec = build_codec(profile_name)
    hold = hold or p.hold_frames
    s = codec.symbol_size
    k = max(1, math.ceil(n_bytes / s))
    gens = math.ceil(k / GEN_SIZE)
    frames = sum(g_k + max(4, math.ceil(g_k * overhead)) for g_k in (min(GEN_SIZE, k - g * GEN_SIZE) for g in range(gens)))
    secs = frames * hold / p.fps
    return {
        "profile": p.name,
        "grid": f"{codec.layout.cols}x{codec.layout.rows}",
        "cell_px": p.cell_px,
        "bits_per_cell": p.bits_per_cell,
        "data_cells": codec.layout.n_data_cells,
        "rs_blocks": len(codec.block_lens),
        "ecc": f"{p.ecc_nsym}/{codec.block_lens[0]}",
        "payload_per_frame": s,
        "code_fps": p.fps / hold,
        "frames": frames,
        "seconds": secs,
        "net_kbps": n_bytes * 8 / 1000 / secs,
    }


def encode_file(
    src: str | os.PathLike,
    out: str | os.PathLike,
    profile_name: str = "balanced",
    overhead: float = 0.25,
    hold: int | None = None,
    loops: int = 1,
    lead_in: float = 3.0,
    compress: bool = True,
    session: int | None = None,
    crf: int = 14,
    progress=None,
) -> EncodeReport:
    if loops < 1:
        raise ValueError("loops must be >= 1")
    if hold is not None and hold < 1:
        raise ValueError("hold must be >= 1")
    if overhead < 0 or lead_in < 0:
        raise ValueError("overhead and lead_in must be >= 0")
    src = Path(src)
    data = src.read_bytes()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    blob = container.pack(src.name, data, compress=compress)
    p = get_profile(profile_name)
    codec = build_codec(profile_name)
    layout = codec.layout
    hold = hold or p.hold_frames
    session = secrets.randbits(32) if session is None else session

    fenc = FountainEncoder(blob, codec.symbol_size, GEN_SIZE, session)
    order = fenc.schedule(overhead)
    fp = fenc.params
    if max(esi for _, esi in order) >= 1 << 16:
        raise ValueError("too many repair symbols for a 16-bit ESI; lower --overhead")

    with VideoWriter(str(out), p.width, p.height, p.fps, crf=crf) as vw:
        banner = render_banner(layout, [f"STEGA  {p.name}", f"{src.name}  {len(data):,} bytes", "hold the camera steady"])
        vw.write(banner, repeat=max(1, int(lead_in * p.fps)))
        for loop in range(loops):
            for i, (gen, esi) in enumerate(order):
                hdr = FrameHeader(p.pid, session, fp.total_len, fp.gen_size, gen, esi)
                symbols = codec.encode(hdr, fenc.symbol(gen, esi))
                vw.write(render_frame(layout, symbols), repeat=hold)
                if progress:
                    progress(loop * len(order) + i + 1, loops * len(order))
        vw.write(render_banner(layout, ["END"]), repeat=p.fps)

    n_frames = len(order) * loops
    secs = n_frames * hold / p.fps
    return EncodeReport(
        output=str(out),
        profile=p.name,
        file_bytes=len(data),
        container_bytes=len(blob),
        symbol_size=codec.symbol_size,
        source_symbols=fp.k,
        code_frames=n_frames,
        video_seconds=secs + lead_in + 1,
        net_kbps=len(data) * 8 / 1000 / max(secs, 1e-9),
    )
