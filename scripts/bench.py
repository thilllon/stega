"""Profile x capture-condition sweep: encode -> camsim -> decode, report frame statistics and goodput.

  uv run python scripts/bench.py --size 200000 --profiles balanced color4 --presets mild phone harsh
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from stega import camsim
from stega.decoder import decode_video
from stega.encoder import encode_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=200_000)
    ap.add_argument("--profiles", nargs="+", default=["robust", "balanced", "dense", "color4", "color8"])
    ap.add_argument("--presets", nargs="+", default=["mild", "phone", "harsh"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--overhead", type=float, default=0.25)
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--keep", default=None, help="directory to keep videos in")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    work = Path(a.keep or tempfile.mkdtemp(prefix="stega-bench-"))
    work.mkdir(parents=True, exist_ok=True)
    src = work / "payload.bin"
    src.write_bytes(os.urandom(a.size))
    rows = []
    print(f"{'profile':9s} {'preset':6s} {'ok':>3s} {'frames':>6s} {'nofind':>6s} {'noori':>5s} {'rsfail':>6s} {'good':>5s} {'sym':>9s} {'video s':>7s} {'need %':>6s} {'kbps':>6s} {'dec s':>6s}")
    for prof in a.profiles:
        video = work / f"{prof}.mp4"
        enc = encode_file(src, video, prof, overhead=a.overhead, loops=a.loops, lead_in=1.0)
        for preset in a.presets:
            rec = work / f"{prof}_{preset}.mp4"
            camsim.simulate(str(video), str(rec), preset=preset, seed=a.seed)
            t = time.time()
            rep = decode_video(rec, work / f"out_{prof}_{preset}.bin", profile="auto", keep_going=True)
            dec_s = time.time() - t
            ok = rep.ok and (work / f"out_{prof}_{preset}.bin").read_bytes() == src.read_bytes()
            st = rep.stats
            good = st.get("ok", 0) + st.get("dup", 0)
            code_s = enc.video_seconds
            need = 100.0 * rep.completed_at_frame / rep.frames_read if rep.completed_at_frame is not None else float("nan")
            row = dict(need_pct=round(need, 1), profile=prof, preset=preset, ok=ok, frames=rep.frames_read, stats=st, symbols=rep.symbols,
                       unique=rep.unique_frames, code_frames=enc.code_frames, video_s=round(enc.video_seconds, 1),
                       net_kbps=round(enc.net_kbps, 1) if ok else 0.0, decode_s=round(dec_s, 1))  # fmt: skip
            rows.append(row)
            print(
                f"{prof:9s} {preset:6s} {'Y' if ok else 'N':>3s} {rep.frames_read:>6d} {st.get('nofinder', 0):>6d} "
                f"{st.get('noorient', 0):>5d} {st.get('rsfail', 0):>6d} {good:>5d} "
                f"{rep.symbols[0]:>4d}/{rep.symbols[1]:<4d} {code_s:>7.1f} {need:>6.1f} {row['net_kbps']:>6.1f} {dec_s:>6.1f}",
                flush=True,
            )
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2))
    print(f"artifacts in {work}")


if __name__ == "__main__":
    main()
