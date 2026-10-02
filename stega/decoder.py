"""Recorded video(s) -> file. Frames are read by ffmpeg, processed in a process pool, and every CRC-valid
symbol is fed to a fountain decoder for its session; decoding stops as soon as one session is complete."""

from __future__ import annotations

import collections
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import container
from .detector import read_frame
from .fountain import FountainDecoder, FountainParams
from .framing import FrameHeader
from .profiles import PROFILES
from .video_io import probe, read_frames

_W_PROFILES: list[str] = []
MAX_SESSIONS = 4
FALLBACK_NAME = "stega_output.bin"


@dataclass
class DecodeReport:
    ok: bool
    output: str | None
    name: str | None
    bytes: int
    frames_read: int
    stats: dict
    symbols: tuple[int, int]
    unique_frames: int
    profile: str | None
    session: int | None
    seconds: float
    error: str | None = None
    completed_at_frame: int | None = None  # capture index at which the file first became recoverable
    timeline: list = field(default_factory=list)


def _init_worker(profiles: list[str]):
    global _W_PROFILES
    _W_PROFILES = profiles
    cv2.setNumThreads(1)


def _work(task):
    idx, frame, have = task
    skip = (lambda h: (h.session, h.gen, h.esi) in have) if have else None
    try:
        r = read_frame(frame, _W_PROFILES, skip=skip)
    except Exception as e:  # a single weird frame must never kill the run
        return idx, "error:" + type(e).__name__, None, None, 0.0, 1.0
    payload = (r.header, r.symbol) if r.header else None
    return idx, r.status, r.profile if r.status in ("ok", "dup") else None, payload, r.pn_score, r.fixed_ber


# ---------------------------------------------------------------------------------------------
# output path handling
# ---------------------------------------------------------------------------------------------
def safe_name(name: str) -> str:
    """The file name stored in a transmission is untrusted: keep only a plain base name."""
    base = Path(name.replace("\\", "/")).name.replace("\x00", "").strip()
    return FALLBACK_NAME if base in ("", ".", "..") else base


def resolve_output(out: str | os.PathLike | None) -> tuple[Path | None, bool]:
    """-> (path, is_directory). Directories (existing, or given with a trailing slash) are created
    before decoding starts so a long decode never ends in a write error."""
    if out is None:
        return None, True
    s = os.fspath(out)
    p = Path(s)
    if p.is_dir() or s.endswith(("/", os.sep)):
        p.mkdir(parents=True, exist_ok=True)
        return p, True
    p.parent.mkdir(parents=True, exist_ok=True)
    return p, False


def _unique(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(1, 10_000):
        cand = path.with_name(f"{path.stem} ({i}){path.suffix}")
        if not cand.exists():
            return cand
    raise FileExistsError(path)


# ---------------------------------------------------------------------------------------------
# session bookkeeping
# ---------------------------------------------------------------------------------------------
class Sessions:
    """One fountain decoder per transmission seen in the recording(s). A recording may start with the
    tail of another playback, or combine clips of different encodes; the first complete one wins."""

    def __init__(self):
        self.decoders: dict[tuple, FountainDecoder] = {}
        self.pending: dict[tuple, list] = collections.defaultdict(list)
        self.have: set[tuple[int, int, int]] = set()

    @staticmethod
    def _valid(h: FrameHeader, symbol: bytes) -> bool:
        s = len(symbol)
        if s == 0 or not 1 <= h.gen_size <= 4096 or h.total_len < container.HEADER_SIZE:
            return False
        k = math.ceil(h.total_len / s)
        return h.gen < math.ceil(k / h.gen_size)

    def add(self, h: FrameHeader, symbol: bytes) -> None:
        if not self._valid(h, symbol):
            return
        key = (h.session, h.total_len, h.gen_size, len(symbol))
        self.have.add((h.session, h.gen, h.esi))
        dec = self.decoders.get(key)
        if dec is None:
            self.pending[key].append((h.gen, h.esi, symbol))
            if len({(g, e) for g, e, _ in self.pending[key]}) < 2:
                return  # a single CRC-valid frame is not enough evidence to open a session
            if len(self.decoders) >= MAX_SESSIONS:
                weakest = min(self.decoders, key=lambda k: self.decoders[k].progress()[0])
                del self.decoders[weakest]
            dec = self.decoders[key] = FountainDecoder(FountainParams(h.total_len, len(symbol), h.gen_size), h.session)
            for g, e, sym in self.pending.pop(key):
                dec.add(g, e, sym)
            return
        dec.add(h.gen, h.esi, symbol)

    def done(self) -> FountainDecoder | None:
        return next((d for d in self.decoders.values() if d.done), None)

    def best(self) -> FountainDecoder | None:
        return max(self.decoders.values(), key=lambda d: d.progress()[0] / d.progress()[1], default=None)


# ---------------------------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------------------------
def decode_video(
    videos: str | os.PathLike | list,
    out: str | os.PathLike | None = None,
    profile: str = "auto",
    workers: int | None = None,
    max_width: int = 3840,
    progress=None,
    keep_going: bool = False,
) -> DecodeReport:
    """Decode one transmission from one or more recordings. Uses spawned worker processes, so a script
    calling this must be import-safe: put the call under `if __name__ == "__main__":`."""
    if getattr(mp.current_process(), "_inheriting", False):  # a spawned child is still importing __main__
        raise RuntimeError(
            "decode_video() was called while a worker process was importing your script; "
            'guard the call with `if __name__ == "__main__":`'
        )
    t0 = time.time()
    videos = [videos] if isinstance(videos, (str, os.PathLike)) else list(videos)
    profiles = list(PROFILES) if profile == "auto" else [profile]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    out_path, out_is_dir = resolve_output(out)
    total_frames = sum((probe(str(v)).get("frames") or 0) for v in videos) or None

    stats = collections.Counter()
    sessions = Sessions()
    locked_profile = None
    frames_read = 0
    timeline = []
    error = None
    completed_at = None

    def frame_source():
        nonlocal frames_read
        for v in videos:
            gen = read_frames(str(v), max_width=max_width)
            try:
                for frame in gen:
                    yield frames_read, frame
                    frames_read += 1
            finally:
                gen.close()

    def handle(result):
        nonlocal locked_profile, completed_at
        idx, status, prof, payload, pn, ber = result
        stats[status] += 1
        timeline.append((idx, status, round(pn, 3), round(ber, 4)))
        if prof:
            locked_profile = locked_profile or prof
        if status == "ok" and payload:
            sessions.add(*payload)
            if completed_at is None and sessions.done():
                completed_at = idx

    ctx = mp.get_context("spawn")
    max_inflight = workers * 2
    source = frame_source()
    restarts = 0
    try:
        while True:
            try:
                with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker, initargs=(profiles,)) as ex:
                    inflight = {}
                    exhausted = False
                    while True:
                        while not exhausted and len(inflight) < max_inflight:
                            item = next(source, None)
                            if item is None:
                                exhausted = True
                                break
                            idx, frame = item
                            task = (idx, frame, frozenset(sessions.have))  # snapshot at submit time
                            inflight[ex.submit(_work, task)] = task
                        if not inflight:
                            break
                        finished, _ = wait(inflight, return_when=FIRST_COMPLETED)
                        for fut in finished:
                            inflight.pop(fut)
                            handle(fut.result())
                        if progress:
                            best = sessions.best()
                            progress(frames_read, total_frames, stats, best.progress() if best else (0, 0))
                        if sessions.done() and not keep_going:
                            for fut in inflight:
                                fut.cancel()
                            break
                break
            except BrokenProcessPool:
                # a worker was killed (OOM / native crash): the frames it held are lost, which the
                # fountain code tolerates; restart the pool a few times, then give up cleanly
                restarts += 1
                if not stats:  # nothing ever came back: workers can't even start
                    error = (
                        "worker processes failed to start (see errors above); when calling decode_video() "
                        'from a script, guard it with `if __name__ == "__main__":`'
                    )
                    break
                stats["worker_crash"] += 1
                if restarts > 3:
                    error = "worker processes keep dying (out of memory? try --workers 2 or --max-width 1280)"
                    break
    finally:
        source.close()

    dec = sessions.done() or sessions.best()
    base = dict(
        frames_read=frames_read,
        stats=dict(stats),
        symbols=dec.progress() if dec else (0, 0),
        unique_frames=len(sessions.have),
        profile=locked_profile,
        session=dec.session if dec else None,
        completed_at_frame=completed_at,
        timeline=timeline,
    )
    if dec is None or not dec.done:
        return DecodeReport(False, None, None, 0, seconds=time.time() - t0, error=error or "not enough frames decoded", **base)
    try:
        payload = container.unpack(dec.recover())
    except ValueError as e:
        return DecodeReport(False, None, None, 0, seconds=time.time() - t0, error=str(e), **base)

    name = safe_name(payload.name)
    if out_path is None:
        target = _unique(Path(name))
    elif out_is_dir:
        target = _unique(out_path / name)
    else:
        target = out_path  # explicitly named by the user: overwrite is intended
    target.write_bytes(payload.data)
    return DecodeReport(True, str(target), payload.name, len(payload.data), seconds=time.time() - t0, **base)


def decode_image(img_rgb: np.ndarray, profile: str = "auto"):
    """Convenience for experiments: decode a single RGB frame."""
    return read_frame(img_rgb, list(PROFILES) if profile == "auto" else [profile], debug=True)
