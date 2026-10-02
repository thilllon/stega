# RESEARCH: hiding data in video, and recovering it after a screen → phone-camera re-recording

Goal: embed ~1 MB into an MP4, play it on a monitor, re-record it with a phone, and recover the original
data from the recording. This document covers (1) a capacity reality check, (2) a taxonomy of techniques
and how each survives re-recording, (3) how this repository's baseline maps onto that landscape, (4) a
staged research roadmap, (5) a real-phone experiment checklist, and (6) references.

Annotation convention
- `[paper]`: value verified directly in the paper's PDF/abstract.
- `[secondary]`: confirmed only through a secondary source (abstract summary, search snippet); needs
  cross-checking against the original.
- `[unverified]`: original not obtained, so no number is given, or the value is approximate.
- `[estimate]`: computed here under stated assumptions — not a number from any paper.
- 1 MB is treated as 1 MiB = 1,048,576 bytes = **8,388,608 bits (~8.39 Mbit)**. Time figures below do not
  include FEC/fountain overhead (in practice add 5–30%).

---

## 1. One-line summary and capacity reality check

**Summary.** A *visible* (overt) 2D-code stream can move 1 MB through a phone re-recording in 1–2 minutes
(hundreds of kbps). A method that is also *statistically hidden* (steganographic) **and** survives
re-recording is, at the published state of the art, limited to tens–hundreds of bits per image and ~1 kbps
in the field — so 1 MB takes hours to days, **2–5 orders of magnitude** short of the goal. "Invisible to the
eye" flicker-fusion methods reach ~100 kbps in the lab, but only under demanding conditions and with no
resistance to steganalysis.

### 1.1 Why overt works: the bit budget

Screen-camera channel capacity is roughly `cells/frame × bits/cell × code rate × usable code frames/s`.

- A 1920×1080 display split into 8–12 px cells gives 14,400–32,400 cells. Assume ~20% overhead for
  finder/alignment/edge-strip structure.
- In a 4K phone recording where the screen fills ~70% of the frame width, one display pixel maps to ~1.4
  camera pixels, so an 8 px cell is ~11 camera px — enough margin to survive blur and codec loss. At 1080p
  recording the same cell is only ~5.6 px, so 12–16 px cells are needed `[estimate]`.
- Recording at 60 fps and holding each code frame for ≥3 display frames (≤~20 code frames/s) means that
  even after discarding rolling-shutter captures that blend two code frames, each code frame still has at
  least one clean capture. LightSync reported that asynchronous communication works when the receiver
  frame rate is at least half the sender's `[secondary]`.

| setting `[estimate]` | cells | payload bits/frame (20% overhead, RS rate) | code fps | throughput | time for 1 MB (incl. 5% fountain) |
|---|---|---|---|---|---|
| conservative: 12 px, 1 bit (B/W), RS rate 0.8 | 14,400 | 9,216 | 5 | ~46 kbps | ~3.2 min |
| middle: 10 px, 2 bit (4-color), RS rate 0.75 | 20,736 | 24,883 | 10 | ~249 kbps | ~35 s |
| aggressive: 8 px, 2 bit, RS rate 0.75 | 32,400 | 38,880 | 15 | ~583 kbps | ~15 s |
| aggressive: 8 px, 3 bit (8-color), RS rate 0.6 | 32,400 | 46,656 | 15 | ~700 kbps | ~13 s |

"1 MB in 1–2 minutes" needs ~70–140 kbps. Even if the middle setting delivers only half of its theoretical
rate in practice, it clears that bar. Phone video codecs run at tens of Mbps, far above the data rate — the
bottleneck is not codec bitrate but the symbol errors caused by blur, color distortion, and quantization.

### 1.2 Reported capacity per system and time for 1 MB

A. Overt (visible-code) screen-camera

| system | reported figure | conditions / caveats | time for 1 MB |
|---|---|---|---|
| PixNet (MobiCom 2010) | up to 12 Mb/s @10 m, 8 Mb/s at 120° view `[paper]` | 30" 2560×1600 LCD, Casio EX-F1 / Nikon D3X. A **nominal** value ("per-frame throughput × 30 fps"), not live phone recording. Nokia N82 phone: nominal 4.301 Mb/s `[paper]` | 0.7 s (12 Mb/s), 2.0 s (N82 nominal) |
| TXQR (open source, 2018) | ~13 KB in 501 ms, 12 FPS, 1850 B/QR, low EC `[author blog]` | The author wrote "almost 25kbps" but 13 KB/0.501 s ≈ 26 kB/s (~208 kbit/s), so the unit is ambiguous. Best case, tripod + external monitor | ~40 s (26 kB/s reading) / ~5.6 min (25 kbit/s reading) |
| COBRA (MobiSys 2012) | figures `[unverified]` | phone↔phone color barcode stream, blur-tolerant | – |
| LightSync (MobiCom 2013) | "~2× prior throughput" `[secondary]`, absolute `[unverified]` | async frame rates, inter-frame erasure coding | – |
| RDCode (MobiCom 2014) | ~10% error rate, >2× COBRA `[secondary]` | packet-frame-block 3-level FEC | – |
| Strata (MobiCom 2014) | absolute `[unverified]` | hierarchical code; decodable layer depends on distance/resolution | – |
| RainBar (ICDCS 2015) | "higher average throughput" `[secondary]`, absolute `[unverified]` | color barcode, flexible frame sync | – |

B. Unobtrusive screen-camera (hidden from viewers, but *not* statistically steganographic)

| system | reported figure | conditions / caveats | time for 1 MB |
|---|---|---|---|
| InFrame (HotNets 2014) | ~12.8 kbps @120 FPS `[secondary]` | complementary frames | ~10.9 min |
| InFrame++ (MobiSys 2015) | up to 360 kbps (data:video 1:6), 150–240 kbps on a 120 FPS 24" LCD `[secondary]` | a later paper (TextureCode) describes it as "18 kbps, noticeable flicker" `[paper]` | 23 s (360) – 56 s (150) |
| HiLight (MobiSys 2015) | 1.1 kbps, ≥91% accuracy (images with mean brightness >150) `[secondary]` | encodes via alpha (translucency) changes | ~2.1 h |
| TextureCode (INFOCOM 2016) | ~22 kbps average goodput `[paper]` | embeds only in textured regions to suppress flicker | ~6.4 min |
| ChromaCode (MobiCom 2018) | raw 777 kbps, goodput 120 kbps, BER 0.05 `[paper]` | 120 Hz 27" 1920×1080, Nexus 6P, CIELAB lightness adaptation, RS+convolutional, 20-person user study | ~70 s (goodput) |
| DeepLight (IPSN 2021) | goodput ≥0.95 kbps (1–1.2), frame error rate <0.2, ~2 m, handheld `[paper]` | **60 Hz** commodity display, Blue channel only, DNN decoder | ~2–2.5 h |
| Revelio (ICASSP 2025) | 16-bit code (288 bit after RS) from a 3 s 120 FPS recording `[paper]` | 65" 4K TV, iPhone 15 Pro Max, OKLAB, flicker strength d=0.0425 | ~18 days (~5.3 bps) |
| ICASSP 2024 (Zhang et al.) | up to 80% accuracy at 3 m `[secondary]` | 60 Hz commodity display, color modulation | – |

C. Single-image DL watermark/stego (print/screen → camera resilient)

| method | bits/image | recapture performance | 1 MB assuming a new decodable image per frame @30 fps `[estimate]` |
|---|---|---|---|
| StegaStamp (CVPR 2020) | 100 raw, 56 after BCH on 400×400 `[paper]` | 6 media × 3 cameras, 98.7% mean bit accuracy over 1,890 captures `[paper]` | 56 bit × 30 fps: ~83 min / with 8 tiles on 1080p: ~10 min |
| LFM (CVPR 2019) | 1024 bit (32×32) `[paper]` | BER 7.37% (frontal), 14.08% (45°), no ECC `[paper]` | BSC-capacity upper bound ~635 bit/image → ~7.3 min (upper bound, before ECC overhead) |
| PIMoG (ACM MM 2022) | 30 bit on 128×128 (official code `nn.Linear(30,256)`) `[paper code]` | ≥97% under several screen-shooting conditions `[secondary]` | ~3.2 h |
| S2R (AAAI 2026) | 64 bit on 128×128 `[paper]` | BER 2.1/3.3/6.0% (0/20/40°), PSNR 42.27 dB `[paper]` | ~85 min |
| VideoSeal (Meta, 2024) | 96 bit per video, per-frame soft bits averaged `[paper]` | not trained on recapture; H.264+crop+brightness → SA-V bit acc 0.73 `[paper]` | 96 bit/video → effectively N/A |

Conclusions:
- Overt designs sit at 10²–10³ kbps and clear the ~100 kbps target easily.
- Results that are hidden from viewers *and* run on a 60 Hz commodity display with a handheld phone
  (DeepLight, HiLight) are ~1 kbps — >2 h for 1 MB, 2–3 orders slower than overt.
- Methods that are *statistically hidden* (steganalysis-resistant) and survive recapture carry tens of
  bits per image, so 1 MB is hours to days.
- The ~100 kbps ChromaCode/InFrame++ numbers assume a 120 Hz display, ≥120 fps capture, and a controlled
  distance. A single frame-difference exposes them, so they are "eye-evasion" rather than steganography;
  at 60 Hz flicker is reported visible (DeepLight).

---

## 2. Taxonomy and re-recording survivability

### 2.1 Classic video steganography (file level)

| family | method | after re-recording |
|---|---|---|
| Spatial LSB | LSB replacement, ±1 embedding | dead |
| Transform domain | DCT/DWT coefficients (JPEG/MPEG coeffs, QIM, spread spectrum) | mostly dead; some low-frequency spread-spectrum watermarks partly survive |
| Codec domain | motion-vector edits (Aly, TIFS 2011), H.264/HEVC syntax (MVD, MVP index, intra mode, quantized coeffs) | dead — the bitstream is regenerated from scratch |
| Adaptive | cost functions + Syndrome-Trellis Codes (Filler et al., TIFS 2011) to minimize distortion | dead |

These are excellent for file-level transfer but do not survive the pixel path of a re-recording.

### 2.2 Screen-camera communication: visible codes

| system | key idea | lesson for the baseline |
|---|---|---|
| PixNet (MobiCom 2010) | 2D OFDM, 81×81 px symbols, 5 px cyclic prefix; drops DC so it is insensitive to lighting | frequency-domain modulation resists perspective and blur; design in lighting invariance |
| COBRA (MobiSys 2012) | color barcode for phone screen↔phone camera, blur-tolerant layout | motion blur is the dominant error source |
| LightSync (MobiCom 2013) | works with mismatched send/receive frame rates; linear inter-frame erasure coding | treat lost/mixed frames as erasures — the basis for our fountain layer |
| RDCode (MobiCom 2014) | block/frame/packet three-level FEC | layer per-frame RS with an inter-frame code |
| Strata (MobiCom 2014) | layered code; a low-resolution receiver still decodes the coarse layer | absorb distance/device variance via capacity adaptation |
| RainBar (ICDCS 2015) | high-capacity color layout, flexible frame sync, locator-based color recognition | put reference cells/locators for color correction inside the code |
| TXQR (2018) | standard QR replayed as animation + LT fountain code | even standard symbols become order/loss-robust when combined with a fountain code |

Common design principles across visible codes, all five of which this repo's baseline follows:
1. Localize geometry with finders and alignment patterns.
2. Correct color every frame with reference cells.
3. Fix bit/symbol errors within a frame with RS or BCH.
4. Handle across-frame loss with an erasure code, so synchronization can be loose.
5. Keep code frame rate at ≤ half the capture rate.

### 2.3 Semi-invisible screen-camera and the flicker-fusion limit

| system | modulation | requirement | note |
|---|---|---|---|
| InFrame / InFrame++ | +Δ/−Δ complementary frames, hierarchical frame structure, CDMA-like modulation | 120 FPS display | high throughput but follow-ups report visible flicker |
| HiLight | alpha-channel translucency change (~1%), 24-block sequence per bit | real time, over any content | ~1 kbps |
| TextureCode | embed only in edge/texture regions (spatially adaptive) | flat regions unused | ~22 kbps; capacity collapses on flat video |
| ChromaCode | CIELAB lightness ±ΔL, adapts to pixel brightness/texture, RS+convolutional | 120 Hz monitor | 120 kbps goodput, BER 0.05 |
| DeepLight | Blue channel only, DNN decodes all bits of a frame at once (no exact cell boundaries needed) | 60 Hz display | ~1 kbps, robustness-first |
| Revelio | OKLAB, spatially adaptive flicker, shape (ellipse-orientation) symbols, 2-stage NN decoder | 60 Hz display, 120 FPS recording | small payload (16–48 bit) |

Physical limits of flicker fusion:
- The commonly cited critical flicker frequency is ~40–50 Hz (ChromaCode intro). Revelio cites ~60 Hz for
  lightness and ~25 Hz for color `[paper]`.
- With sharp spatial edges, humans perceive flicker artifacts **above 500 Hz** (Davis et al., Scientific
  Reports 2015); the median observer needs >500 Hz to stop seeing them, and some perceived >800 Hz. A
  high-contrast complementary pattern can therefore show a phantom-array effect even at 120 Hz.
- On a 60 Hz display, alternating +Δ/−Δ produces a 30 Hz component, i.e. visible flicker (DeepLight). You
  need a ≥120 Hz display, or a combination of spatial adaptation and small Δ.
- Camera-side constraints:
  - separating a complementary pair needs a capture rate ≥ the display modulation rate (ChromaCode, Revelio);
  - phone 120/240 fps modes drop resolution, exposure, and compression quality;
  - phone encoders tend to smooth out small-amplitude temporal changes — record with low compression
    (e.g. ProRes) when possible to isolate the cause;
  - rolling shutter mixes +Δ and −Δ frames as horizontal bands within one capture.
- Net: semi-invisible capacity depends heavily on cover content (brightness, texture) and hardware refresh
  rate. Expect ~1/10 of overt (120 Hz, controlled) down to ≤1/100 (60 Hz, in the field) `[estimate]`.

### 2.4 Screen-shooting resilient watermarking

| work | type | key idea | capacity/performance |
|---|---|---|---|
| Fang et al., TIFS 2019 | classic | intensity-based SIFT (I-SIFT) to find embedding regions; small DCT templates repeated across regions to resist lighting and moiré | tens of bits `[unverified]` |
| RIHOOP, TCYB 2022 (online 2020) | DL | a distortion network between encoder and decoder that models camera distortion | capacity `[unverified]` |
| PIMoG, ACM MM 2022 | DL | differentiable perspective, illumination, moiré, Gaussian noise; gradient-mask image loss, edge-mask loss | 30 bit on 128×128, ≥97% |
| DeNoL, ACM MM 2023 | DL | decouples the noise layer from a few real samples | `[unverified]` |
| SSDS, IEEE TMM 2024 | DL | grayscale-deviation simulation | `[unverified]` |
| RoPaSS, AAAI 2025 | DL | handles partial screen-shooting (only part of the screen captured) | `[unverified]` |
| S2R, AAAI 2026 | DL | learns the "sim → real" noise gap from unpaired real captures (MIMO-UNet + unpaired Pix2Pix); 3 phone×monitor pairs, 900 images each | 64 bit/128×128, BER 2.1% |

All of these solve "recover a few tens of ID bits from one photo." Our problem (millions of bits over
thousands of frames) has a different objective, but the **noise-layer design and real-capture
fine-tuning methodology** transfer directly.

### 2.5 Deep-learning image/video steganography and watermarking

| method | capacity | training noise layers | quality | pros for our goal | cons |
|---|---|---|---|---|---|
| **HiDDeN** (ECCV 2018) | stego: 52 bit on 16×16 gray (0.203 bpp); watermark: 30 bit on 128×128 YUV `[paper]` | Dropout, Cropout, Crop, Gaussian blur, JPEG-Mask, JPEG-Drop, Combined | Combined PSNR(Y) 33.55 dB `[paper]` | the canonical encoder-decoder-noise-adversary template; simple code | no geometric/color/capture distortion training → fails recapture (StegaStamp comparison) |
| **StegaStamp** (CVPR 2020) | 100 bit on 400×400, 56 after BCH | perspective (corners ±40 px), motion blur (3–7 px), Gaussian blur (σ 1–3), hue ±0.1, desaturation, brightness/contrast m∈[0.5,1.5]·b∈[−0.3,0.3], Gaussian noise σ∼U[0,0.2], differentiable JPEG (q 50–100) | 100 bit: PSNR 28.50, SSIM 0.905, LPIPS 0.101; 200 bit drops to PSNR 21.79 `[paper]` | real print/screen→camera validation; published loss-ramp and perturbation curriculum | few bits/image; quality collapses as bits rise; visible pattern in flat regions |
| **LFM** (CVPR 2019) | 1024 bit/image | a learned CDTF network T() trained on Camera-Display 1M (25 pairs, 1,000,000 images) | subtle patch-like changes | learns the real display-camera transfer function from data; evaluated on 1000-frame video playback | BER 7–20% before ECC; degrades on new device pairs; T takes 7 days to train `[paper]` |
| **RIHOOP** (TCYB) | `[unverified]` | 3D-render + camera-distortion network | – | offline/online photo hyperlinks | small capacity |
| **PIMoG** (MM 2022) | 30 bit on 128×128 | perspective, illumination, moiré, Gaussian | – | explicit moiré model | recommends the image fill ≥1/4 of the screen at capture (README) → limits tiling density |
| **HiNet** (ICCV 2021) / **DeepMIH** (TPAMI 2022) | full-size secret image(s) hidden in a cover; invertible NN, wavelet domain | none (assumes lossless) | secret-recovery PSNR +10 dB over prior (HiNet) `[paper]` | overwhelming digital capacity (several bpp) | extremely fragile to quantization, compression, geometry; cannot survive recapture; file-level only |
| **RivaGAN** (2019) | 32/64 bit per video (recovered from ≥1 frame) | scaling, cropping, real MJPEG compression (non-differentiable path), attention, critic, adversary | PSNR ~42 dB; ~0.98–0.998 accuracy after MJPEG/crop/scale `[paper]` | video-domain attention design | small capacity; no capture-distortion training |
| **VideoSeal** (Meta, 2024) | 96 bit; embedded at 256×256 then upscaled; embed every k=4 frames and propagate | geometric/brightness/color transforms, differentiable video-codec augmentation; image pretrain → video posttrain → extractor fine-tune | best PSNR/SSIM among the compared models, though the authors note it is visible up close in flat regions `[paper]` | open source (code, weights); good reference for a codec-augmented training pipeline | provenance-focused, so tiny capacity; no recapture training |
| **MBRS** (MM 2021) | 256 bit (as reported in the VideoSeal comparison) `[paper]` | alternates real JPEG, simulated JPEG, and no-noise per mini-batch | BER <0.01% at JPEG Q50, PSNR >36 `[secondary]` | its "mix a real non-differentiable op into the mini-batch" trick applies directly to an H.264 proxy | weak on geometric transforms (VideoSeal Table 4) |
| TrustMark (2023) / InvisMark (2024) / WAM (ICLR 2025) | 100 / 256 / 32 bit | editing-centric | InvisMark PSNR ~51, SSIM ~0.998 `[paper]` | high-resolution techniques | not recapture-targeted |
| S2R (AAAI 2026) | 64 bit/128×128 | PIMoG simulation + unpaired real-noise learning | PSNR 42.27 dB | uses real captures for label-free domain adaptation | small capacity, still-photo basis |

---

## 3. How this repository's baseline maps onto the landscape

### 3.1 Current baseline layers (mapped to files)

```
file bytes
  └─ [L6] Container (container.py): filename, original size, SHA-256, zlib (only when it helps)
     └─ [L5] Outer code (fountain.py): generation-based (256–2048 symbols, sized to the file) random linear fountain over GF(2)
           systematic symbols first, then repair; generations interleaved round-robin to spread bursts
           (any sufficient subset of captured frames reconstructs; lost/blurred/rolling-shutter = erasure)
        └─ [L4] Packet framing (framing.py): 18B header (magic, version, profile id, session, total_len,
              gen_size, gen, esi) + symbol + CRC32. Coefficient vectors are derived by SHAKE-128(session,gen,esi)
           └─ [L3] Inner code: shortened RS(≤255, n−nsym) blocks, byte round-robin interleaving
              └─ [L2] Whitening: XOR with a fixed SHAKE-128 mask
                 └─ [L1] Modulator (render.py): 1920×1080 cell grid, 1–3 bits/cell
                        (B/W, K·C·M·Y tetrahedron, 8-color RGB cube) — see profiles.py
                      + Sync (layout.py): 4 QR-like finders, 5×5 alignment lattice, per-profile PN edge strips
  ── play on monitor → record with phone ──
  [R1] Localizer (detector.py): nested-contour finder detection (4, or 3 if one is hidden) → homography from
       finder centers + corners → rank 8 orientation × profile hypotheses by PN correlation → re-fit the
       homography over the whole alignment lattice → per-node template matching for a local displacement field
       (lens distortion, rolling-shutter shear)
  [R2] Demodulator: sample cell centers → per-channel local black/white (min/max filter) normalization →
       4-neighbor cell-domain equalizer (removes blur ISI) → nearest-palette decision + per-cell margin
  [R3] De-whitening → RS decode (retry low-confidence bytes as erasures, retry equalizer strength) → CRC32
  [R4] Fountain decoder (decoder.py): per-generation online Gauss-Jordan, stops as soon as solvable → SHA-256
```

`stega.camsim` is a phone-recapture channel simulator written independently of these layers (perspective,
lens distortion, rolling shutter, exposure integration, moiré/Bayer demosaic, defocus, hand shake, glare,
noise, H.264 re-compression), and `scripts/bench.py` runs a profile × capture-condition sweep. When you
add a new modulator, compare it on the same harness.

### 3.2 What stays the same when the modulator is swapped for a stego modulator

- **L5 fountain**: modulation-independent. It only needs each frame abstracted as "an erasure channel
  carrying a few packets." For semi-invisible modes the frame success rate is lower and payload/frame is
  smaller, so you only adjust generation size and total frame count. A random GF(2) matrix reaches full
  rank with probability ≈ 1−2^(−m) given k+m equations (standard result); an overhead m of a few to a few
  tens of packets is enough.
- **L4 CRC32 + header**: reused. But a semi-invisible frame carries tens–hundreds of bits, so a 32-bit CRC
  and the header become large relative overhead — consider shrinking the header below 16 bit or moving to
  CRC-16, and replacing the generation id with an implicit time slot.
- **L3 RS/interleaving, L2 whitening**: keep the interface, change the parameters (see 3.3).

### 3.3 What changes

| aspect | visible modulator | stego/DL modulator |
|---|---|---|
| symbol unit | 1 cell = 1–3 bit, position explicit | 1 image (or tile) = N bit; bit↔pixel mapping implicit (the network spreads it) |
| demod output | hard symbol + per-cell confidence | **soft bits (logit/LLR)** vector; soft decoding required |
| raw BER regime | 10⁻³–10⁻² | 10⁻²–10⁻¹ (LFM 7–20%, StegaStamp raw ~1.3%, S2R 2–6%) |
| inner code | RS(255,k), byte-wise | bit errors are near-spatially-independent and BER is high → **LDPC/polar/convolutional (soft Viterbi)** or BCH; if keeping RS, mark low-confidence bytes as erasures (LLR-based) to nearly double the correction radius |
| sync/geometry | visible finders, alignment, PN strips | (a) detect off the screen bezel or the cover video itself (StegaStamp uses a separate detector), (b) low-amplitude PN template (Fang 2019 style), (c) hybrid: a thin visible border marker + invisible payload |
| fine registration | local displacement field to pin cell centers | rely on learned warp-robustness; train so a coarse homography suffices (perspective noise layer) |
| whitening | prevents color bias and long same-cell runs | network input bits should already be uniform to match training; keep it (zero cost) |
| frame design | hold each code frame for N frames | every cover frame differs; temporal consistency (no shimmering residual) becomes a quality metric; repeat a message over k frames and average soft bits (VideoSeal style) to lower BER |
| capacity knob | cell size, palette, RS rate, fps | embedding strength α, bits/tile, repeat frames k, tile count |

Design principle: fix the modulator interface as `modulate(frame_bits, cover_frame) -> frame` and
`demodulate(captured_frame) -> (llr[], frame_confidence)`. Then visible, semi-invisible, and DL modulators
compare on the same L2–L5 stack and the same evaluation harness. Even the visible path should emit cell
confidence as LLRs, so the later swap is seamless.

### 3.4 Capacity ↔ invisibility trade-off

Thanks to the fountain layer, "trade capacity for time" is linear. Time for 1 MB ≈ `8.39 Mbit ×
(1+fountain overhead) / (usable bits/tile × tiles × independent-frame fps × frame success rate)`.

Example: 8 tiles, 64 usable bits/tile, 10 independent messages/s, 0.8 success → ~3.3 kbps, ~45 min for 1 MB
`[estimate]`. To send 1 MB in 2 minutes a semi-invisible scheme must reach ~70 kbps, 70× the published 60 Hz
in-the-field result (~1 kbps).

Realistic targets:
- Semi-invisible: aim for a few KB to tens of KB (a key, URL, hash, small document) in a few minutes.
- 1 MB: send it overt, or as a "visible but non-annoying (aesthetic) code."
- 1 MB hidden transfer: a long-term research problem.

---

## 4. Research roadmap (staged)

### (a) Tune and measure the visible baseline

1. Simulator loop: sweep parameters over the synthetic channel (homography, blur, color matrix, gamma,
   Gaussian noise, real H.264/HEVC re-encode via ffmpeg, rolling-shutter row mixing). Cell size {8,10,12,16}
   px × palette {2,4,8} × RS k × code fps {5,10,15,20}.
2. Measurement matrix: 2–3 phones × recording {1080p30, 1080p60, 4K30, 4K60} × distance {40, 70, 100 cm} ×
   angle {0, 20, 40°} × lighting {bright, dim}. For each, record ≥10 MB of random data repeatedly.
3. Deliverables: per-condition raw symbol error rate, post-RS frame success rate, goodput, time-to-1MB
   curves. Pick operating parameters at each curve's peak.
4. Hypotheses to test:
   - the 8-color palette gains little over 4 colors because of phone white balance and tone mapping;
   - 4K60 helps shrink cells more than 1080p60;
   - setting code fps to 1/3 of capture fps maximizes goodput after treating rolling-shutter-mixed frames
     as erasures.
5. Optional adaptation: Strata-style layered codes, or per-generation profiles that change cell size with
   received quality.

Measured so far (4K / 270 MB-in-5-min workstream, simulator):
- **Colour beats gray levels at equal bits/cell.** 2-bit gray (4 luma levels) decoded only the captures
  that did not straddle a code-frame change (exactly 2 of 3 at hold=3, ~0 at hold=2), while 2-bit colour
  decoded ~99%. Exposure across a frame change blends two frames: a binary per-channel colour decision lands
  on one of them; an amplitude level lands on a third, valid-looking level. 4:2:0 chroma subsampling was not
  the binding limit at ≥8 px cells.
- **Multi-level cells need a blur-matched equaliser.** The fixed strong cell equaliser tuned for B/W raised
  4-level symbol errors from 0% to 32%; estimating the blur per frame from known cells
  (`detector.estimate_beta`, observed = a + p·own + q·neighbour-mean, β = k/(1−k)) restored ~0%.
- **4K at hold=2 is tear-limited:** even successful runs lose ~50% of captures to rolling-shutter tears,
  and success across sampled setups is ~5/8, driven by chromatic moiré at certain distances.
- **Big frames are fragile:** a 4K frame has ~119 RS codewords and is lost if any fails → **sub-framing**
  (8 horizontal bands, each its own fountain symbol): `uhd3` coverage on a good setup 73% → 98%, and
  rolling-shutter-torn captures now yield every band above/below the tear.
- **Band position must not alias with generation.** Once only a few generations remain in the round-robin
  schedule, band *b* always carried the same generation, so a region read badly (moiré) starved it. Shuffling
  band assignment per frame fixed it.
- **Generation count is a reliability parameter at scale.** All generations must complete, so P(success) =
  P(one)^n. Fixed 256-symbol generations make 270 MB ~255 independent trials (45% success at 86% coverage /
  25% overhead in an i.i.d. model); ~32 generations of up to 2048 symbols make it ~100%, for ~1 s of decode
  per generation. Bigger generations remove variance; they do not rescue a mean coverage below 1/(1+overhead).

### (b) Semi-invisible modulation experiments

| experiment | method | expected capacity (vs visible) `[estimate]` | main risk |
|---|---|---|---|
| b1. low-amplitude luma/chroma cells | add a ±Δ (Δ=2–8/255) cell pattern on top of a cover; use large cells (24–48 px); differential demod needs a cover estimate | 1/20–1/100 | the phone codec removes low-amplitude components; interference with cover texture |
| b2. complementary frame @60 Hz | +Δ at frame t, −Δ at t+1; demod differences consecutive captures | 1/10–1/50 | 30 Hz flicker is visible (DeepLight); needs 60 fps capture and correct phase |
| b3. complementary frame @120 Hz | 120 Hz monitor + 120/240 fps recording; CIELAB/OKLAB lightness adaptation (ChromaCode/Revelio) | 1/3–1/10 (if reproducing ChromaCode) | phone high-fps modes lose resolution/compression; rolling-shutter bands; display overdrive |
| b4. chroma-only (Blue or Cb/Cr) | exploit low human color acuity (DeepLight Blue channel) | 1/20–1/100 | phone 4:2:0 chroma subsampling halves spatial resolution |
| b5. texture-adaptive | scale Δ by per-cell local variance (TextureCode/ChromaCode); skip flat cells and treat as erasures | cover-dependent; collapses on flat video | receiver doesn't know the sender's adaptation → needs a deterministic rule or soft decoding |

Common experimental design:
- sweep Δ, cell size, and modulation frequency;
- measure BER together with flicker visibility per condition — use ≥5–10-person MOS or 2AFC comparisons,
  and include high-edge covers (because of the Davis 2015 500 Hz result);
- to isolate the phone codec, also record each condition in ProRes (supported devices) or at max bitrate.

### (c) DL end-to-end (StegaStamp/HiDDeN-style with a screen-camera noise layer)

Architecture sketch (tile-based, 1080p cover)

```
message m ∈ {0,1}^N  (N = 64–256, per tile)
  └ Linear → reshape to 16×16×C_m → upsample
cover tile x ∈ R^{3×256×256}  (split a 1080p frame into 6–8 tiles, or downscale then upscale the residual)
Encoder E: U-Net (4 down/4 up, ch 32–256), input concat(x, msg_feat)
  → residual r = α · tanh(E(x, m))       # α: strength knob (same role as VideoSeal's α_w)
  → x_w = clip(x + r)
Noise layer 𝒩 (random per batch during training, strength ramped by curriculum)
Decoder D: ResNet-18/34 or ViT-S (VideoSeal extractor family), input 256×256 aligned tile
  → logits ∈ R^N
Critic C (optional): PatchGAN/WGAN discriminator
```

- Sync: in stage 1, detect the four screen corners with a thin visible border marker and align tiles by
  homography — reuse the existing R1 localizer. In stage 2, remove the marker and replace it with a detector
  network or a low-amplitude PN template.
- Video: embed the same m for k frames but recompute r per frame to match the cover; average the k captures'
  logits at the receiver. Add a temporal-consistency loss `‖r_t − warp(r_{t−1})‖` to suppress shimmering.

Noise layers (prefer differentiable implementations, e.g. Kornia)

| distortion | starting parameters (StegaStamp ranges as a starting point) |
|---|---|
| Perspective | corners ±10% random homography; most sensitive, so ramp it slowest |
| Lens/scale | 0.6–1.4 scale, mild barrel distortion |
| Motion blur | straight kernel 3–7 px, random angle |
| Defocus | Gaussian σ 1–3 px |
| Display gamma/gamut | gamma 1.8–2.6, 3×3 color-matrix perturbation, desaturation |
| Camera color/exposure | hue ±0.1, brightness m∈[0.5,1.5], b∈[−0.3,0.3], white-balance gain |
| Moiré | add/multiply a PIMoG-style synthetic moiré pattern |
| Subpixel/resample | nearest-upsample to display resolution → bilinear-resample to camera resolution |
| Rolling shutter | linearly blend x_t and x_{t+1} by row index y (random boundary); add 60 Hz PWM brightness bands |
| Sensor noise | Gaussian σ∼U[0,0.2] (the paper value is large; consider shrinking after measurement), Poisson-Gaussian |
| JPEG proxy | DCT quantization + differentiable approximation (Shin & Song, used by StegaStamp), q 50–100 |
| H.264/HEVC proxy | MBRS-style: per mini-batch pick one of {real ffmpeg encode (straight-through: `x + (enc(x) − x).detach()`), differentiable DCT proxy, identity}. VideoSeal reports differentiable-codec augmentation greatly improves high-compression bit acc |
| Frame drop/mix | randomly blend with the adjacent message frame (add a confidence head for erasure training) |

Loss

```
L = λ_msg · BCE(D(𝒩(x_w)), m)
  + λ_L2 · ‖r‖²_w                  # edge-weighted L2 (StegaStamp: suppress edge patterns)
  + λ_lpips · LPIPS(x_w, x)
  + λ_adv · L_critic(x_w)
  + λ_temp · ‖r_t − warp(r_{t−1})‖₁   # for video
```

- StegaStamp tip: start with the image-loss weight at 0 until the decoder reaches high accuracy, then ramp
  it up linearly; also start the perturbation strength at 0.
- Use PIMoG official-code defaults (msg 3, image 1, GAN 0.001) as initial λ references.
- Choose the operating point as "minimum LPIPS at bit acc ≥ 95%." StegaStamp picked 100 bit this way.

Datasets
- Cover images: MIRFLICKR (StegaStamp), MS-COCO (HiDDeN, LFM, PIMoG, S2R), DIV2K (high-res, HiNet eval).
- Cover video: SA-V (used by VideoSeal), plus other public video datasets (check licenses).
- Real display-camera pairs: LFM's Camera-Display 1M (check availability/license at GitHub `mathski/LFM`).
  Self-collection (below) is ultimately required.

Real recapture fine-tuning
1. Collect: play x_w (from the trained E) fullscreen, record with a phone, align via the visible corner
   markers to build (x_w, capture) pairs. Vary device pairs, distance, angle, lighting (S2R used 3 pairs ×
   900 images).
2. Method A (paired, LFM-style): train a CDTF network T on the pairs, then insert it into the noise layer.
3. Method B (unpaired, S2R-style): learn only the distribution gap between simulation output and real
   captures — no label alignment needed.
4. Method C (MBRS-style mix): mix real-capture batches directly into decoder fine-tuning; freeze the
   encoder and fine-tune only the extractor (same idea as VideoSeal's extractor fine-tuning stage).
5. Iterate: re-collect with the new E and repeat 1–4. Two or three closed loops recommended.

Apple Silicon (PyTorch MPS) practical notes
- Use `torch.device("mps")`; for unsupported ops set `PYTORCH_ENABLE_MPS_FALLBACK=1` for CPU fallback, and
  profile to confirm the fallback op isn't the bottleneck.
- Start with 256×256 tiles, U-Net channels 32–256, batch 16–32. LPIPS (AlexNet backbone) is light — compute
  it every step.
- Precompute real ffmpeg-encode noise in CPU DataLoader workers and cache it (the STE `enc(x)` becomes
  invalid when the encoder changes, so keep only a few batches on-the-fly).
- Mixed precision (fp16/bf16 autocast) MPS support varies by PyTorch version — check it, and keep fp32 if
  numerically unstable.
- Scale up in steps: reproduce 128×128/30 bit (HiDDeN/PIMoG class) → 400×400/100 bit (StegaStamp class) →
  tile-based 1080p → video with temporal loss.

Coupling with the outer fountain code
- Map N bits/tile to one L4 packet (e.g. 48–64 bit payload + CRC-16); use LDPC/BCH with LLRs for the inner
  code.
- Send frame failures to erasures for the fountain to absorb. Lowering the embedding strength α reduces the
  frame success rate, which shows up only as longer transmission time, not as recovery failure. This is the
  "capacity ↔ invisibility" lever.

### (d) Evaluation protocol and metrics

| metric | definition | note |
|---|---|---|
| Raw BER | hard-bit error rate after demod (pre-FEC) | save a per-cell/tile heatmap |
| Symbol/cell error rate | visible-path cell decision errors | per-palette-color confusion matrix |
| Frame success rate (FSR) | frames passing RS + CRC32 / code frames shown | report both per-capture and per-displayed-frame |
| Fountain overhead | packets used / k − 1 | per-generation distribution |
| Goodput (kbps) | recovered file bits / wall-clock (recording start → recoverable) | includes sync-acquisition time |
| Time-to-1MB | time to fully recover a 1 MiB random file | median and p90 over ≥10 runs per condition |
| Success probability | fraction with SHA-256 match within a time limit (e.g. 3 min) | the end-user metric |
| Cover quality | PSNR, SSIM, LPIPS (frame), VMAF (video) | as VideoSeal notes, high PSNR can still miss flat-region artifacts and temporal consistency — pair with visual review |
| Flicker visibility | MOS (1–5) or 2AFC accuracy (50% = indistinguishable) | state viewing distance, screen brightness, cover type (flat/edge) |
| Steganalysis resistance (if applicable) | AUC or P_E of a learned detector such as SRNet (cover/stego trained) | most published semi-invisible results don't meet this; evaluate file level and displayed-screen level separately |

Every report must record display (model, size, refresh, nit brightness, mode), phone (model, app,
resolution/fps/codec/bitrate), distance/angle/lighting (lux), cover video, code parameters, and repeat count.

---

## 5. Real-phone recording checklist

Display (transmitter)
- [ ] Set the display to native resolution (integer scaling for a 1920×1080 or 3840×2160 panel) and turn off
      OS and player scaling; confirm 1:1 pixel mapping.
- [ ] Use a player that does not drop frames (e.g. mpv). Confirm the MP4 fps divides the display refresh
      (30/20/15 fps on 60 Hz).
- [ ] Disable auto-brightness, True Tone, Night Shift (macOS), HDR tone mapping, dynamic contrast, motion
      smoothing (TV), local dimming, and power saving. Revelio disabled TV Motion Smoothing,
      Super-Resolution, and color correction.
- [ ] Fix brightness (e.g. 70–80% of max). Avoid low-brightness ranges with strong PWM dimming.
- [ ] For semi-invisible experiments, state the refresh rate (60/120/144 Hz) and disable VRR/FreeSync/G-Sync.
- [ ] Check for light reflections on a glossy screen; prefer a matte screen or adjust the light angle.
- [ ] Show a test pattern (gray steps, color patches) for 3 s before recording, for receiver color correction.

Phone (receiver)
- [ ] Lock exposure (AE) and focus (AF). On the iPhone default camera, long-press the screen for AE/AF Lock.
      With a manual-control app, set shutter to an integer multiple of the display refresh period (1/60,
      1/120 on 60 Hz) to reduce banding.
- [ ] Turn off HDR Video (Settings ▸ Camera ▸ Record Video). HDR tone mapping distorts the color palette and
      low-amplitude signals.
- [ ] Turn off Auto FPS, Enhanced Stabilization, and Action mode (they add crop and warp). Enable Lock Camera
      to prevent lens switching. On Android disable the maker's options (scene optimizer, AI enhancement,
      night mode).
- [ ] Do not use night mode, Portrait/Cinematic mode, or beauty/filters.
- [ ] Sweep resolution and fps:
  - 1080p30: baseline.
  - 1080p60: higher code fps.
  - 4K30: small cells.
  - 4K60: both (recommended default). Watch for thermal throttling.
  - For semi-invisible b2/b3, also record 120/240 fps (check resolution loss).
- [ ] Codec: compare default HEVC and "most compatible" (H.264). Use ProRes on supported devices to isolate
      the codec effect.
- [ ] Zoom: use optical 1× (main camera) by default; avoid digital zoom. Avoid ultra-wide (distortion, small
      sensor).

Geometry and environment
- [ ] Measure tripod (static) and handheld (dynamic) separately.
- [ ] Fill 60–90% of the capture frame with the screen; leave margin so corner finders aren't clipped.
- [ ] Angles 0°, 20°, 40° (both horizontal and vertical); distance 0.5–1.5× the screen diagonal.
- [ ] If moiré appears, change distance slightly or defocus a touch — moiré period is distance-sensitive.
- [ ] Lighting: record indoor lux. Check whether 50/60 Hz light flicker (fluorescent, cheap LED) causes
      banding, and remove light sources reflecting directly off the screen.
- [ ] Heat: check for frame drops and bitrate drops after ≥3 min of continuous 4K60.

Data handling
- [ ] Record the transmitted file's SHA-256, code parameters, and seed in the recording filename or a sidecar
      JSON.
- [ ] Transfer originals via AirDrop "all data" or a cable. Do not use messaging apps — they re-compress.
- [ ] Repeat each condition ≥3 times, and do not delete failed recordings (keep them for error analysis).

---

## 6. References

Screen-camera communication (visible codes)
1. S. D. Perli, N. Ahmed, D. Katabi. *PixNet: Interference-Free Wireless Links Using LCD-Camera Pairs.* ACM MobiCom 2010. https://doi.org/10.1145/1859995.1860012 (PDF: https://people.csail.mit.edu/nabeel/pixnet-mobicom10.pdf)
2. T. Hao, R. Zhou, G. Xing. *COBRA: Color Barcode Streaming for Smartphone Systems.* ACM MobiSys 2012. https://doi.org/10.1145/2307636.2307645
3. W. Hu, H. Gu, Q. Pu. *LightSync: Unsynchronized Visual Communication over Screen-Camera Links.* ACM MobiCom 2013. https://doi.org/10.1145/2500423.2500437
4. A. Wang, S. Ma, C. Hu, J. Huai, C. Peng, G. Shen. *Enhancing Reliability to Boost the Throughput over Screen-Camera Links* (RDCode). ACM MobiCom 2014. https://doi.org/10.1145/2639108.2639135
5. W. Hu, J. Mao, Z. Huang, Y. Xue, J. She, K. Bian, G. Shen. *Strata: Layered Coding for Scalable Visual Communication.* ACM MobiCom 2014. https://doi.org/10.1145/2639108.2639132
6. *RainBar: Robust Application-Driven Visual Communication Using Color Barcodes.* IEEE ICDCS 2015, pp. 537–546. https://doi.org/10.1109/ICDCS.2015.61
7. I. Daniluk. *TXQR: Transfer data via animated QR codes* (GitHub) https://github.com/divan/txqr, blog *Fountain codes and animated QR* https://divan.dev/posts/fountaincodes/

Semi-invisible screen-camera
8. A. Wang, C. Peng, O. Zhang, G. Shen, B. Zeng. *InFrame: Multiflexing Full-Frame Visible Communication Channel for Humans and Devices.* ACM HotNets 2014. https://doi.org/10.1145/2670518.2673867
9. A. Wang et al. *InFrame++: Achieve Simultaneous Screen-Human Viewing and Hidden Screen-Camera Communication.* ACM MobiSys 2015. https://doi.org/10.1145/2742647.2742652
10. T. Li, C. An, A. T. Campbell, X. Zhou. *HiLight: Hiding Bits in Pixel Translucency Changes.* ACM VLCS (MobiCom workshop) 2014. https://www.cs.dartmouth.edu/~xia/papers/vlcs14-hilight.pdf
11. T. Li, C. An, X. Xiao, A. T. Campbell, X. Zhou. *Real-Time Screen-Camera Communication Behind Any Scene* (HiLight). ACM MobiSys 2015. https://doi.org/10.1145/2742647.2742667
12. V. Nguyen, Y. Tang, A. Ashok, M. Gruteser, K. Dana, W. Hu, E. Wengrowski, N. Mandayam. *High-Rate Flicker-Free Screen-Camera Communication with Spatially Adaptive Embedding* (TextureCode). IEEE INFOCOM 2016. https://ieeexplore.ieee.org/document/7524512/
13. K. Zhang, C. Wu, C. Yang, Y. Zhao, K. Huang, C. Peng, Y. Liu, Z. Yang. *ChromaCode: A Fully Imperceptible Screen-Camera Communication System.* ACM MobiCom 2018. https://doi.org/10.1145/3241539.3241543 (PDF: https://cswu.me/papers/mobicom18_chromacode_paper.pdf)
14. V. Tran, G. Jayatilaka, A. Ashok, A. Misra. *DeepLight: Robust & Unobtrusive Real-time Screen-Camera Communication for Real-World Displays.* ACM/IEEE IPSN 2021. https://doi.org/10.1145/3412382.3458269, arXiv:2105.05092
15. H. Zhang, X. Yu, Z. Zhang, B. Zhu. *Robust and Imperceptible Commercial Camera-Screen Communication with 60Hz Refresh Rate.* IEEE ICASSP 2024. https://ieeexplore.ieee.org/document/10446206/
16. A. A. Mohamed Nishar, S. Kudekar, B. Kintzing, A. Ashok. *Revelio: A Real-World Screen-Camera Communication System with Visually Imperceptible Data Embedding.* IEEE ICASSP 2025. arXiv:2501.02349
17. H. Yu, J. Zhang, H. Du, K. Guo, X.-Y. Li. *InvisiCode: Boosting Intra-Frame Screen-Camera Communication by Breaking Through Noise Limitations.* IEEE IWQoS 2025. https://ieeexplore.ieee.org/document/11143478/ (figures unverified)
18. J. Davis, Y.-H. Hsieh, H.-C. Lee. *Humans perceive flicker artifacts at 500 Hz.* Scientific Reports 5:7861, 2015. https://doi.org/10.1038/srep07861

Screen-shooting resilient watermarking
19. H. Fang, W. Zhang, H. Zhou, H. Cui, N. Yu. *Screen-Shooting Resilient Watermarking.* IEEE TIFS 14:1403–1418, 2019.
20. J. Jia, Z. Gao, K. Chen, M. Hu, X. Min, G. Zhai, X. Yang. *RIHOOP: Robust Invisible Hyperlinks in Offline and Online Photographs.* IEEE Trans. Cybernetics 52(7):7094–7106. https://ieeexplore.ieee.org/abstract/document/9293163
21. H. Fang, Z. Jia, Z. Ma, E.-C. Chang, W. Zhang. *PIMoG: An Effective Screen-shooting Noise-Layer Simulation for Deep-Learning-Based Watermarking Network.* ACM MM 2022. https://doi.org/10.1145/3503161.3548049, code: https://github.com/FangHanNUS/PIMoG-An-Effective-Screen-shooting-Noise-Layer-Simulation-for-Deep-Learning-Based-Watermarking-Netw
22. *DeNoL: A Few-Shot-Sample-Based Decoupling Noise Layer for Cross-channel Watermarking Robustness.* ACM MM 2023. https://doi.org/10.1145/3581783.3612068
23. *Screen-Shooting Resistant Watermarking With Grayscale Deviation Simulation* (SSDS). IEEE TMM 2024. https://doi.org/10.1109/TMM.2024.3415415
24. *RoPaSS: Robust Watermarking for Partial Screen-Shooting Scenarios.* AAAI 2025. https://ojs.aaai.org/index.php/AAAI/article/view/34128
25. Y. Wu, X. Liao, B. Wang, H. Fang, X. Wu, M. Chen, G. Wang. *Sim-to-Real: An Unsupervised Noise Layer for Screen-Camera Watermarking Robustness.* AAAI 2026. arXiv:2504.18906

Deep-learning steganography/watermarking
26. J. Zhu, R. Kaplan, J. Johnson, L. Fei-Fei. *HiDDeN: Hiding Data With Deep Networks.* ECCV 2018. arXiv:1807.09937
27. M. Tancik, B. Mildenhall, R. Ng. *StegaStamp: Invisible Hyperlinks in Physical Photographs.* CVPR 2020. arXiv:1904.05343
28. E. Wengrowski, K. Dana. *Light Field Messaging With Deep Photographic Steganography.* CVPR 2019. https://openaccess.thecvf.com/content_CVPR_2019/html/Wengrowski_Light_Field_Messaging_With_Deep_Photographic_Steganography_CVPR_2019_paper.html, code/data: https://github.com/mathski/LFM
29. J. Jing, X. Deng, M. Xu, J. Wang, Z. Guan. *HiNet: Deep Image Hiding by Invertible Network.* ICCV 2021.
30. Z. Guan, J. Jing, X. Deng, M. Xu, L. Jiang, Z. Zhang, Y. Li. *DeepMIH: Deep Invertible Network for Multiple Image Hiding.* IEEE TPAMI 45(1):372–390. https://doi.org/10.1109/TPAMI.2022.3141725
31. K. A. Zhang, L. Xu, A. Cuesta-Infante, K. Veeramachaneni. *Robust Invisible Video Watermarking with Attention* (RivaGAN). arXiv:1909.01285, 2019.
32. P. Fernandez, H. Elsahar, I. Z. Yalniz, A. Mourachko. *Video Seal: Open and Efficient Video Watermarking.* arXiv:2412.09492, 2024. code: https://github.com/facebookresearch/videoseal
33. Z. Jia, H. Fang, W. Zhang. *MBRS: Enhancing Robustness of DNN-based Watermarking by Mini-Batch of Real and Simulated JPEG Compression.* ACM MM 2021. arXiv:2108.08211
34. T. Bui et al. *TrustMark: Universal Watermarking for Arbitrary Resolution Images.* arXiv:2311.18297, 2023.
35. R. Xu et al. *InvisMark: Invisible and Robust Watermarking for AI-generated Image Provenance.* arXiv:2411.07795, 2024.
36. T. Sander, P. Fernandez, A. Durmus, T. Furon, M. Douze. *Watermark Anything with Localized Messages* (WAM). ICLR 2025. arXiv:2411.07231
37. R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric* (LPIPS). CVPR 2018. arXiv:1801.03924

Classic steganography/steganalysis
38. H. A. Aly. *Data Hiding in Motion Vectors of Compressed Video Based on Their Associated Prediction Error.* IEEE TIFS 6(1):14–18, 2011. https://doi.org/10.1109/TIFS.2010.2090520
39. T. Filler, J. Judas, J. Fridrich. *Minimizing Additive Distortion in Steganography Using Syndrome-Trellis Codes.* IEEE TIFS 6(3):920–935, 2011. https://doi.org/10.1109/TIFS.2011.2134094
40. M. Boroumand, M. Chen, J. Fridrich. *Deep Residual Network for Steganalysis of Digital Images* (SRNet). IEEE TIFS 14(5):1181–1193, 2019. https://doi.org/10.1109/TIFS.2018.2871749
41. *A survey on information hiding using video steganography.* Artificial Intelligence Review, 2021. https://link.springer.com/article/10.1007/s10462-021-09968-0

Tools / practical
42. Kornia (differentiable geometry/image ops): https://github.com/kornia/kornia
43. Apple Support. *Adjust HDR camera settings on iPhone*, *Change video recording settings on iPhone*, *Record ProRes video.* https://support.apple.com/guide/iphone/
