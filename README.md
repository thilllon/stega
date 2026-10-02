# stega — send a file through a screen → phone-camera video channel

Encode any file into an MP4 of 2D cell-grid code frames, play it fullscreen on a monitor,
**re-record it with a phone**, and decode the recording back into the **byte-identical original file**.

This has been verified end to end on a real phone: a **1 MB file was recovered bit-for-bit** from a
handheld Galaxy S23 recording of a laptop screen — including the case where the recording was
interrupted by a phone call and finished in a second clip (the two clips were combined automatically).

```
  file ──encode──▶ code.mp4 ──play on monitor──▶ 📱 record ──▶ recording.mp4 ──decode──▶ file (identical)
```

> **Is this steganography?** Not yet — and that distinction matters. This project transmits data through
> *visible* code frames (think "a fast slideshow of QR-like codes"). Anyone watching sees that data is
> being sent. True *hidden* steganography that also survives a phone re-recording is, at the current
> state of the art, limited to tens–hundreds of bits per image (~1 kbps in the field), so 1 MB would take
> hours. This project first builds a channel that **provably works** through the brutal screen→camera path
> (marker detection, perspective/lens correction, error correction, resuming across clips), with the data
> layers kept **independent of how pixels are drawn**. Swapping the visible modulator for a semi-invisible
> or deep-learning one is future work; see [`docs/RESEARCH.md`](docs/RESEARCH.md) for the survey and roadmap.

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- [ffmpeg](https://ffmpeg.org/) on your `PATH` (macOS: `brew install ffmpeg`)

```bash
uv sync    # installs numpy, opencv-python-headless, reedsolo
```

## Quick start

```bash
uv run stega info                                # capacity/duration table for every profile

uv run stega encode secret.zip -o code.mp4       # file  -> MP4 (default profile: balanced)

# Play code.mp4 fullscreen and record the screen with a phone, then copy the recording back.
uv run stega check   recording.mp4               # rate the recording BEFORE trusting it
uv run stega decode  recording.mp4 -o out/       # recording -> out/secret.zip (auto-detects profile)

# No phone handy? A built-in, independent simulator fakes a phone re-recording:
uv run python -m stega.camsim code.mp4 rec.mp4 --preset phone   # mild | phone | harsh
uv run stega decode rec.mp4 -o out/
```

## How big a file, and how long is the video?

The transmission time is `file_size ÷ (bytes-per-frame × code-frames-per-second)`. Bigger files just
mean longer videos — there is no separate "max" until the container's hard cap of **256 MiB**, but in
practice you want to keep videos to a few minutes. Rough guide (playback length; you record ~10–20% more
than the point where it becomes recoverable):

| file size | `balanced` (safe, 720p/30 fps ok) | `color4` (1080p) | `dense` (fast, 1080p/60 fps) |
|----------:|----------------------------------:|-----------------:|-----------------------------:|
|   100 KB  | ~11 s   | ~6 s   | ~3 s   |
|   500 KB  | ~53 s   | ~28 s  | ~13 s  |
| **1 MB**  | **~1.8 min** | **~0.9 min** | **~0.4 min** |
|     2 MB  | ~3.6 min | ~1.9 min | ~0.9 min |
|     5 MB  | ~9 min  | ~4.7 min | ~2.2 min |
|    10 MB  | ~18 min | ~9.4 min | ~4.4 min |
|    50 MB  | ~91 min | ~47 min | ~22 min |

Practical advice: **≤ 1 MB is comfortable, up to ~10 MB is fine if you can keep the phone steady on a
stand. Beyond that the recording gets unwieldy** — split it into several files instead. Run
`uv run stega info --size <bytes>` for exact numbers for your file and profile.

## Recording with a real phone

1. **Phone camera:** turn **HDR video OFF**, prefer **1080p (or 4K) at 60 fps**, and **lock focus and
   exposure** (long-press the subject on the recording screen).
2. **Monitor:** play the video **fullscreen**, brightness high, and **hide the player controls and mouse
   cursor** — if the on-screen player bar covers the bottom two markers, that frame can't be read. Turn
   off Night Shift / True Tone / auto-brightness.
3. Frame the shot so the code (including its white border) fills **70–90%** of the phone frame, roughly
   head-on. Use the "STEGA …" intro banner (first few seconds) to line up the shot.
4. Landscape, portrait, and mirrored orientations are all handled automatically.
5. **If you get interrupted, don't delete the clip.** Just keep filming the rest (or record again), and
   pass every clip to `decode` together — received frames are de-duplicated and only the gaps are filled:
   ```bash
   uv run stega decode clip1.mp4 clip2.mp4 -o out/
   ```

### Getting the video off an Android phone

The cleanest path is a USB cable with `adb` (Android's platform tools; `brew install android-platform-tools`):

```bash
# One-time on the phone: Settings ▸ About phone ▸ tap "Build number" 7×, then
# Settings ▸ Developer options ▸ enable "USB debugging". Plug in and tap "Allow".
adb devices                                              # your phone shows as "device"
f=$(adb shell 'ls -t /sdcard/DCIM/Camera/*.mp4 | head -1' | tr -d '\r')
adb pull "$f" ~/Downloads/
```

Or upload the file to **Google Drive** (keeps the original quality) and download it on your computer.
**Do not send it through messaging apps** (KakaoTalk, Telegram, etc.) — they re-compress the video, which
smears the cells and hurts recovery.

## `stega check` — validate a recording before a long run

Before committing to a multi-minute transmission, record ~10 seconds of the video and rate it:

```bash
uv run stega check recording.mp4
```

```
resolution 1280x720 @ 29.98 fps, checked 80 frames (excluded: 21 banner, 19 before/after playback)
  markers found     40/40  (100%)
  fully decoded     32/40  (80%)   profile: balanced
  cell error rate median 0.0000  worst 0.0000  (known marker cells)

VERDICT: good. Record the real transmission the same way.
```

- **≥ 50% decoded → good.** Record the real file the same way.
- **25–50% → tight.** Record twice (or `encode --loops 2`) so gaps get filled.
- **markers found but cells misread →** lock focus/exposure, add light, fill more of the frame, or use a
  bigger-cell profile (`-p robust`) / longer exposure per frame (`--hold 5`).
- **markers rarely found →** get closer/steadier, avoid glare, keep the whole code (with its white border)
  in frame, and hide the player UI.

## Profiles (numbers for 1 MB, from `stega info`)

| profile  | grid    | cell px | bits/cell | RS parity/word | bytes/frame | code fps | video (1 MB) | recording needs |
|----------|---------|--------:|----------:|---------------:|------------:|---------:|-------------:|-----------------|
| robust   | 120×67  | 16 | 1 | 48/239 |  550 | 7.5 | ~318 s | anything, even a bad/far camera |
| balanced | 160×90  | 12 | 1 | 40/244 | 1202 | 10  | ~109 s | **720p/30 fps is enough** (verified on a real phone) |
| fast     | 192×108 | 10 | 1 | 40/247 | 1836 | 15  |  ~48 s | **1080p/60 fps** (holds each frame only 2 video frames) |
| dense    | 240×135 |  8 | 1 | 40/242 | 2998 | 10  |  ~44 s | steady, close, 1080p+ |
| color4   | 160×90  | 12 | 2 | 48/244 | 2330 | 10  |  ~56 s | 1080p+ (color needs resolution) |
| color8   | 137×77  | 14 | 3 | 56/253 | 2336 | 7.5 |  ~75 s | steady, in focus, 1080p+ |
| uhd      | 480×270 |  8 | 2 | 48/254 | 24401 | 30 |  ~1.7 s | **4K screen + 4K/60 fps recording** — see below |
| uhd3     | 480×270 |  8 | 3 | 56/254 | 35212 | 30 |  ~1.2 s | same, experimental |

`encode` also takes `--hold N` (video frames each code frame is shown; lower = faster but needs a
higher-fps camera) and `--overhead F` (extra repair frames per generation; lower = shorter but less
margin). **Start with `balanced`; once `check` reports "good", move up to `fast`/`color4`/`dense`.**
The fast profiles trade robustness for speed — they need a steady, close, in-focus 1080p/60 fps recording.
To push further, `dense --hold 2 --overhead 0.12` reaches ~26 s for 1 MB but needs excellent capture.

## High-density 4K mode (toward 270 MB in 5 minutes)

The `uhd` profiles draw a 3840×2160 code canvas, hold each code frame for 2 frames at 60 Hz
(30 code frames/s), and are decoded from a 4K/60 fps recording at native resolution.

| profile | net rate | per 5 min of video | time for 270 MB |
|---------|---------:|-------------------:|----------------:|
| `uhd`  (8 px, 2 bit colour, overhead 12%) | 5.1 Mbps | ~184 MB | ~7.3 min |
| `uhd3` (8 px, 3 bit colour, overhead 10%) | 7.4 Mbps | ~265 MB | ~5.1 min |

Status, measured with the 4K phone simulator (`camsim --size 3840x2160 --cam-fps 60`):

- **`uhd` decodes in 5 of 8 randomly sampled handheld setups.** When it works, every code frame is
  recovered; when it fails, almost every capture fails. Failures cluster at high moiré (the monitor's RGB
  subpixel stripes aliasing with the camera grid — a *colour* pattern that hurts colour cells) and
  depend on distance. Because the outcome is decided by the setup, **run `stega check` on a 10 s trial and
  move the phone slightly until it reports "good"**, then record the full transmission from there.
- **`uhd3` reaches ~91% of its code frames** — just short of what its fountain overhead covers. It is the
  configuration that meets 270 MB / 5 min on paper; making it reliable is the next workstream
  (sub-framing so that rolling-shutter-torn captures are not discarded whole).
- Even when decoding succeeds, ~50% of captures are torn across a code-frame change (hold=2 at 60 fps is
  the limit); those are currently discarded entirely.

Requirements and pitfalls:
- **A real 3840×2160 monitor showing the video 1:1.** A laptop panel (e.g. 2880/3024/3456 px wide)
  resamples a 4K video by a non-integer factor and smears 8 px cells before the camera sees them.
- A phone recording **4K at 60 fps**, HDR off, focus/exposure locked. Expect ~3.5 GB of 4K recording
  per 5 minutes; the encoded code video itself is ~2.4 GB (≈63 Mbps).
- Decode at native resolution (the default `--max-width 3840` does this; don't lower it).

Why colour and not gray levels? Luma is not chroma-subsampled, so storing 2–3 bits as 4–8 gray levels
looked attractive. Measured, it loses: a capture exposed across a code-frame change blends the old and
new frame, which keeps a *binary* per-channel colour decision on one side but turns a mid gray into a
different *valid* level. Gray palettes remain available (`Profile(..., palette="gray")`) for
experiments with longer holds.

## Verified results

- **Real phone (Galaxy S23, 720p/30 fps, handheld):** 1 MB recovered byte-identical, SHA-256 match,
  including a two-clip recording (interrupted by a call, resumed) combined into one decode.
- **Simulator sweep (`scripts/bench.py`):** all 5 profiles × 3 capture presets (`mild`/`phone`/`harsh`)
  recover 1 MB byte-identical. Independent reviewers also confirmed decoding through curved monitors,
  35–50° tilt, portrait/mirrored orientations, 4K, a finder occluded by a finger/glare, and HEVC `.mov` /
  10-bit HLG-PQ / rotated / 60 fps / VFR phone files.
- `uv run pytest -q` → all tests pass (unit, synthetic capture, end-to-end, decoder I/O robustness).

## How it works

```
 file ─► container (name, SHA-256, zlib) ─► fountain code (GF(2) random linear, generations of ≤256 symbols)
      ─► frame: [header 18B | symbol | CRC32] ─► Reed-Solomon(≤255) blocks ─► byte interleave ─► scrambler
      ─► cell symbols (1/2/3 bit) ─► 1920×1080 code frame (4 finders · alignment lattice · PN edge strips)
      ─► H.264 MP4 (each code frame held N video frames)

 recording(s) ─► ffmpeg decode ─► (process pool) find finders (4, or 3 if one is hidden) → orientation/mirror/
      profile via PN correlation → homography from finder corners → re-fit on the alignment lattice → local
      displacement field → sample cells → local black/white normalise → cell-domain equaliser → nearest palette
      → descramble/de-interleave → RS decode (retry low-confidence bytes as erasures) → CRC32
      ─► per-session fountain decode (stops as soon as one file is solvable) ─► SHA-256 verify ─► file
```

The **fountain code** is why interrupted or partial recordings still work: the file is spread across frames
so that *any* sufficiently large subset of distinct frames reconstructs it. Lost, blurred, or
rolling-shutter-torn frames are simply erasures.

| layer | file | role |
|---|---|---|
| profiles | `stega/profiles.py` | cell size, bits/cell, RS parity, hold |
| layout | `stega/layout.py` | quiet zone, QR-like finders, alignment lattice, PN edge strips |
| outer code | `stega/fountain.py` | generation-based random linear fountain code |
| inner code | `stega/framing.py` | header/CRC, Reed-Solomon, interleave, scrambler, bit↔cell mapping |
| modulator | `stega/render.py` | palettes (B/W, K·C·M·Y, RGB cube) → canvas ← *replace this for a stego modulator* |
| detector | `stega/detector.py` | finders, homography, refinement, normalisation, equaliser, classify ← *and this* |
| encoder/decoder | `stega/encoder.py`, `stega/decoder.py` | make MP4 / multi-process, multi-clip, multi-session decode |
| video I/O | `stega/video_io.py` | ffmpeg pipe (HEVC/HDR/rotation-metadata safe) |
| channel simulator | `stega/camsim.py` | independent phone-recapture model (for testing without a phone); H.264 bitrate scales with the recording's pixel rate like a phone's (override with `bitrate=`) |
| benchmark | `scripts/bench.py` | profile × capture-condition sweep |

## Command reference

```
stega info   [--size BYTES] [--overhead F]                 capacity/duration of every profile
stega encode INPUT -o OUT.mp4 [-p PROFILE] [--hold N]      file -> MP4
             [--overhead F] [--loops N] [--lead-in SEC] [--crf N] [--no-compress]
stega check  RECORDING [-p PROFILE] [-n FRAMES]            rate a trial recording/photo
stega decode REC [REC ...] [-o OUT|DIR/] [-p auto|PROFILE] recording(s) -> file
             [-j WORKERS] [--max-width PX (default 3840)] [--keep-going] [--timeline FILE.json]
stega frame  [-p PROFILE] [-o frame.png]                   write one sample code frame (to inspect/print)
```

Notes: the output filename comes from inside the video but is reduced to a safe basename (no path
traversal), and an existing file is never overwritten silently (a `name (1).ext` copy is written instead).
Pass `-o DIR/` (trailing slash) to save under the original name in a directory.

## Testing & benchmarking

```bash
uv run pytest -q
uv run python scripts/bench.py --size 1048576 --profiles balanced color4 --presets mild phone harsh
```

## License

MIT — see [LICENSE](LICENSE).
