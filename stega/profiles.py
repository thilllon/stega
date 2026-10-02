"""Transmission profiles: the visual geometry + error-correction knobs shared by encoder and decoder.

A profile fully determines the frame layout, so the decoder only needs the profile name
(or it can auto-detect it: every profile has its own pseudo-noise edge strips).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    pid: int  # 1 byte id stored in every frame header
    cell_px: int  # size of one code cell on the 1920x1080 canvas (even -> aligns with 4:2:0 chroma)
    bits_per_cell: int  # 1: black/white, 2: K/C/M/Y tetrahedron, 3: RGB cube corners
    ecc_nsym: int  # Reed-Solomon parity bytes per <=255-byte codeword
    hold_frames: int  # how many video frames each code frame is shown (at `fps`)
    width: int = 1920
    height: int = 1080
    fps: int = 30
    quiet: int = 2  # white quiet-zone cells around the grid
    align_step_px: int = 220  # target spacing of the alignment-pattern lattice, in canvas pixels
    palette: str = "rgb"  # "rgb": colour cube corners | "gray": 2**bits_per_cell luma levels (immune to 4:2:0)
    bands: int = 1  # independently decodable horizontal bands per code frame (one fountain symbol each)

    @property
    def cols(self) -> int:
        return self.width // self.cell_px

    @property
    def rows(self) -> int:
        return self.height // self.cell_px

    @property
    def code_fps(self) -> float:
        return self.fps / self.hold_frames


PROFILES: dict[str, Profile] = {
    p.name: p
    for p in [
        # Big cells, slow: for bad cameras, long distances, or first experiments.
        Profile("robust", pid=1, cell_px=16, bits_per_cell=1, ecc_nsym=48, hold_frames=4),
        # Default: works with a handheld phone filling most of its frame with the screen.
        Profile("balanced", pid=2, cell_px=12, bits_per_cell=1, ecc_nsym=40, hold_frames=3),
        # Small B/W cells: needs a steady, close, in-focus capture (tripod, 4K recording helps).
        Profile("dense", pid=3, cell_px=8, bits_per_cell=1, ecc_nsym=40, hold_frames=3),
        # 2 bits/cell with black/cyan/magenta/yellow: colour survives 4:2:0 if cells are big enough.
        Profile("color4", pid=4, cell_px=12, bits_per_cell=2, ecc_nsym=48, hold_frames=3),
        # 3 bits/cell with the 8 RGB cube corners: highest raw rate, most sensitive to colour errors.
        Profile("color8", pid=5, cell_px=14, bits_per_cell=3, ecc_nsym=56, hold_frames=4),
        # Faster B/W: 10 px cells (smaller than balanced) shown for only 2 video frames, so it asks for a
        # 60 fps recording. ~1.7x faster than balanced; simulator-validated down to the "harsh" preset at
        # 1080p. Smaller cells (6-8 px) go faster still but need excellent, close, in-focus capture, so
        # they are reachable via `dense --hold 2` rather than shipped as their own profile.
        Profile("fast", pid=10, cell_px=10, bits_per_cell=1, ecc_nsym=40, hold_frames=2),
        # 4K screen + 4K/60 fps recording. Each code frame is held 2 frames at 60 Hz -> 30 code frames/s;
        # decode recordings at native resolution. Colour palettes beat gray levels here: a capture exposed
        # across a frame change blends old/new frames, which keeps a binary per-channel decision but turns
        # a mid gray into a different valid level. 8 horizontal bands per frame: a capture torn by the rolling
        # shutter still yields every band above/below the tear (symbol coverage 73% -> 98% in simulation).
        Profile("uhd", pid=30, cell_px=8, bits_per_cell=2, ecc_nsym=48, hold_frames=2,
                width=3840, height=2160, fps=60, bands=8),
        # 3 bits/cell: ~265 MB per 5 min, i.e. 270 MB in ~5.1 min.
        Profile("uhd3", pid=31, cell_px=8, bits_per_cell=3, ecc_nsym=56, hold_frames=2,
                width=3840, height=2160, fps=60, bands=8),
    ]
}

DEFAULT_PROFILE = "balanced"


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown profile {name!r}; choose from {', '.join(PROFILES)}") from None


def profile_by_pid(pid: int) -> Profile | None:
    return next((p for p in PROFILES.values() if p.pid == pid), None)
