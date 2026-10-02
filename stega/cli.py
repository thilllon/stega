"""stega command line: encode / decode / info / frame."""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time

from .profiles import DEFAULT_PROFILE, PROFILES


def _bar(done: int, total: int | None, extra: str = "") -> None:
    if total:
        w = 30
        f = int(w * min(done, total) / total)
        sys.stderr.write(f"\r[{'#' * f}{'.' * (w - f)}] {done}/{total} {extra}   ")
    else:
        sys.stderr.write(f"\r{done} {extra}   ")
    sys.stderr.flush()


def _positive(v: str) -> int:
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def _nonneg(kind):
    def conv(v: str):
        x = kind(v)
        if x < 0:
            raise argparse.ArgumentTypeError("must be >= 0")
        return x

    return conv


def cmd_info(a) -> int:
    from .encoder import plan

    size = a.size
    print(f"payload = {size:,} bytes, overhead = {a.overhead:.0%}")
    print(f"{'profile':9s} {'grid':>8s} {'cell':>4s} {'bpc':>3s} {'ecc':>7s} {'B/frame':>7s} {'code fps':>8s} {'frames':>6s} {'video':>7s} {'net kbps':>8s}")
    for name in PROFILES:
        p = plan(name, size, a.overhead)
        print(
            f"{name:9s} {p['grid']:>8s} {p['cell_px']:>4d} {p['bits_per_cell']:>3d} {p['ecc']:>7s} "
            f"{p['payload_per_frame']:>7d} {p['code_fps']:>8.1f} {p['frames']:>6d} {p['seconds']:>6.0f}s {p['net_kbps']:>8.1f}"
        )
    return 0


def cmd_encode(a) -> int:
    from .encoder import encode_file, plan

    size = os.path.getsize(a.input)
    pl = plan(a.profile, size, a.overhead, a.hold)
    print(f"profile {a.profile}: {pl['payload_per_frame']} B/frame, ~{pl['frames'] * a.loops} code frames, ~{pl['seconds'] * a.loops:.0f}s of video", file=sys.stderr)
    t = time.time()
    rep = encode_file(
        a.input, a.output, a.profile, overhead=a.overhead, hold=a.hold, loops=a.loops,
        lead_in=a.lead_in, compress=not a.no_compress, crf=a.crf,
        progress=lambda d, n: _bar(d, n, "frames"),
    )  # fmt: skip
    sys.stderr.write("\n")
    print(json.dumps(rep.__dict__ | {"encode_seconds": round(time.time() - t, 1)}, indent=2))
    return 0


def cmd_decode(a) -> int:
    from .decoder import decode_video

    def prog(n, total, stats, sym):
        _bar(n, total, f"ok={stats.get('ok', 0)} dup={stats.get('dup', 0)} rsfail={stats.get('rsfail', 0)} symbols={sym[0]}/{sym[1]}")

    rep = decode_video(a.videos, a.output, profile=a.profile, workers=a.workers, max_width=a.max_width, progress=prog, keep_going=a.keep_going)
    sys.stderr.write("\n")
    d = rep.__dict__.copy()
    timeline = d.pop("timeline")
    if a.timeline:
        with open(a.timeline, "w") as f:
            json.dump(timeline, f)
    print(json.dumps(d, indent=2))
    return 0 if rep.ok else 1


def cmd_check(a) -> int:
    """Rate a short trial recording (or a photo) BEFORE committing to a long transmission."""
    import statistics

    from .detector import read_frame
    from .profiles import PROFILES
    from .video_io import probe, read_frames

    info = probe(a.input)
    total = info.get("frames") or 0
    step = max(1, total // a.frames) if total else 1
    profiles = list(PROFILES) if a.profile == "auto" else [a.profile]
    rows = []
    for i, frame in enumerate(read_frames(a.input, max_width=a.max_width)):
        if i % step:
            continue
        r = read_frame(frame, profiles)
        rows.append(r)
        _bar(len(rows), (total // step) + 1 if total else None, "frames checked")
        if len(rows) >= a.frames:
            break
    sys.stderr.write("\n")
    if not rows:
        print("stega: no frames could be read", file=sys.stderr)
        return 1

    # frames before the first / after the last sighting of the code are just the phone being aimed
    # (e.g. at the terminal before playback started); they say nothing about capture quality
    seen = [i for i, r in enumerate(rows) if r.status != "nofinder"]
    n_outside = len(rows) - (seen[-1] - seen[0] + 1) if seen else 0
    if seen:
        rows = rows[seen[0] : seen[-1] + 1]
    banners = [r for r in rows if r.status == "banner"]
    rows = [r for r in rows if r.status != "banner"] or rows  # the banner carries no data: not a failure
    n_banner = len(banners)
    found = [r for r in rows if r.status != "nofinder"]
    good = [r for r in rows if r.status in ("ok", "dup")]
    bers = [r.fixed_ber for r in found if r.fixed_ber < 1.0]
    n = len(rows)
    prof = collections.Counter(r.profile for r in good + banners if r.profile).most_common(1)
    excluded = [f"{n_banner} banner" if n_banner else "", f"{n_outside} before/after playback" if n_outside else ""]
    excluded = ", ".join(x for x in excluded if x)
    print(f"resolution {info['width']}x{info['height']} @ {info['fps']:.2f} fps, checked {n + n_banner + n_outside} frames"
          + (f" (excluded: {excluded})" if excluded else ""))
    print(f"  markers found   {len(found):>4}/{n}  ({100 * len(found) / n:.0f}%)")
    print(f"  fully decoded   {len(good):>4}/{n}  ({100 * len(good) / n:.0f}%)   profile: {prof[0][0] if prof else '-'}")
    if bers:
        print(f"  cell error rate median {statistics.median(bers):.4f}  worst {max(bers):.4f}  (known marker cells)")

    rate = len(good) / n
    print()
    if rate >= 0.5:
        print("VERDICT: good. Record the real transmission the same way.")
    elif rate >= 0.25:
        print("VERDICT: usable but tight. Record the video twice (or encode with --loops 2) so gaps get filled.")
    elif len(found) / n >= 0.5:
        print("VERDICT: markers are found but cells are misread. Try: lock focus/exposure, more light,")
        print("         fill more of the frame, a bigger-cell profile (-p robust), or --hold 5 when encoding.")
    else:
        print("VERDICT: markers are rarely found. Get closer / steadier, avoid glare and reflections,")
        print("         make sure the whole code (with its white border) is in frame and the player UI is hidden.")
    if info["fps"] < 45:
        print("NOTE: recording at 30 fps. 60 fps roughly doubles the usable captures per code frame.")
    return 0


def cmd_frame(a) -> int:
    """Write one example code frame as PNG (for eyeballing / printing / single-photo experiments)."""
    import cv2

    from .framing import FrameHeader, build_codec
    from .render import render_frame

    codec = build_codec(a.profile)
    hdr = FrameHeader(codec.layout.profile.pid, 0, codec.symbol_size, 256, 0, 0)
    img = render_frame(codec.layout, codec.encode(hdr, os.urandom(codec.symbol_size)))
    if not cv2.imwrite(a.output, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
        print(f"stega: error: could not write {a.output}", file=sys.stderr)
        return 1
    print(a.output)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="stega", description="Transmit a file through a screen -> phone-camera video channel.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    names = list(PROFILES)

    s = sub.add_parser("info", help="capacity / duration of every profile for a payload size")
    s.add_argument("--size", type=int, default=1_048_576)
    s.add_argument("--overhead", type=float, default=0.25)
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("encode", help="file -> mp4")
    s.add_argument("input")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("-p", "--profile", default=DEFAULT_PROFILE, choices=names)
    s.add_argument("--overhead", type=_nonneg(float), default=0.25, help="extra fountain repair frames per generation (0.25 = +25%%)")
    s.add_argument("--hold", type=_positive, default=None, help="video frames per code frame (default: profile)")
    s.add_argument("--loops", type=_positive, default=1, help="repeat the whole frame sequence")
    s.add_argument("--lead-in", type=_nonneg(float), default=3.0, help="seconds of framing banner before data")
    s.add_argument("--crf", type=int, default=14)
    s.add_argument("--no-compress", action="store_true")
    s.set_defaults(fn=cmd_encode)

    s = sub.add_parser("decode", help="recorded video(s) -> file")
    s.add_argument("videos", nargs="+", help="one or more recordings of the same transmission (clips are combined)")
    s.add_argument("-o", "--output", default=None, help="output file, or directory (existing or ending in /); default: stored name in cwd")
    s.add_argument("-p", "--profile", default="auto", choices=["auto", *names])
    s.add_argument("-j", "--workers", type=_positive, default=None)
    s.add_argument("--max-width", type=int, default=3840, help="downscale frames so the longest side <= this (0 = native; keep >= 3840 for 4K profiles)")
    s.add_argument("--keep-going", action="store_true", help="process the whole video even after success (for statistics)")
    s.add_argument("--timeline", default=None, help="write per-frame status JSON here")
    s.set_defaults(fn=cmd_decode)

    s = sub.add_parser("check", help="rate a trial recording/photo of a code video (do this before a long run)")
    s.add_argument("input", help="a few seconds of recording, or a photo of one code frame")
    s.add_argument("-p", "--profile", default="auto", choices=["auto", *names])
    s.add_argument("-n", "--frames", type=_positive, default=40, help="how many frames to sample")
    s.add_argument("--max-width", type=int, default=3840)
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("frame", help="write a sample code frame PNG")
    s.add_argument("-p", "--profile", default=DEFAULT_PROFILE, choices=names)
    s.add_argument("-o", "--output", default="frame.png")
    s.set_defaults(fn=cmd_frame)

    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except KeyboardInterrupt:
        print("\nstega: interrupted", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as e:
        print(f"\nstega: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
