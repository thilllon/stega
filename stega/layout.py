"""Frame geometry in *cell units*: quiet zone, finder patterns, alignment lattice, PN edge strips, data cells.

Coordinates: cell (row r, col c) covers [c, c+1) x [r, r+1) in cell units; its centre is (c+0.5, r+0.5).
Cell values in `fixed`: 0 = black, 1 = white, -1 = data cell.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .profiles import Profile, get_profile

FINDER = 7  # QR-style 7x7 finder: black ring, white ring, 3x3 black core
ALIGN = 5  # 5x5 alignment pattern: black ring, white ring, black centre

KIND_DATA, KIND_QUIET, KIND_FINDER, KIND_ALIGN, KIND_PN = 0, 1, 2, 3, 4


def prng_bytes(label: str, n: int) -> bytes:
    """Portable deterministic pseudo-random bytes (identical on every platform / numpy version)."""
    return hashlib.shake_128(label.encode()).digest(n)


def prng_bits(label: str, n: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(prng_bytes(label, (n + 7) // 8), dtype=np.uint8))[:n]


@lru_cache(maxsize=None)
def banner_bits(profile_name: str) -> np.ndarray:
    """Known B/W pattern filling the data area of lead-in/lead-out banner frames, so the decoder can
    recognise them as idle (the camera is aimed correctly) instead of reporting a read failure."""
    lay = build_layout(profile_name)
    return prng_bits(f"stega-banner-{profile_name}", lay.n_data_cells)


def _ring_pattern(size: int) -> np.ndarray:
    """Concentric rings: outer black, then alternating; value per cell (0 black / 1 white)."""
    idx = np.arange(size)
    ring = np.minimum.outer(np.minimum(idx, size - 1 - idx), np.minimum(idx, size - 1 - idx))
    pat = (ring % 2).astype(np.int8)  # ring 0 black, 1 white, 2 black ...
    if size == FINDER:
        pat[2:5, 2:5] = 0  # solid 3x3 core
    return pat


FINDER_PATTERN = _ring_pattern(FINDER)
ALIGN_PATTERN = _ring_pattern(ALIGN)


@dataclass(frozen=True)
class Layout:
    profile: Profile
    cols: int
    rows: int
    kind: np.ndarray  # (rows, cols) uint8
    fixed: np.ndarray  # (rows, cols) int8
    data_rc: tuple[np.ndarray, np.ndarray]  # row / col index arrays of data cells, transmission order
    finder_centers: np.ndarray  # (4, 2) float (x, y) cell units, order TL, TR, BR, BL (clockwise)
    lattice_x: np.ndarray  # centre-cell column of each lattice column
    lattice_y: np.ndarray  # centre-cell row of each lattice row
    pn_rc: tuple[np.ndarray, np.ndarray]  # PN strip cells
    pn_bits: np.ndarray  # expected 0/1 per PN cell
    pn_edge: np.ndarray  # edge id per PN cell: 0 top (TL-TR), 1 right (TR-BR), 2 bottom (BR-BL), 3 left (BL-TL)

    @property
    def n_data_cells(self) -> int:
        return len(self.data_rc[0])

    @property
    def capacity_bits(self) -> int:
        return self.n_data_cells * self.profile.bits_per_cell

    @property
    def capacity_bytes(self) -> int:
        return self.capacity_bits // 8

    def lattice_nodes(self):
        """Yield (iy, ix, cy, cx, is_corner) for every lattice node."""
        ny, nx = len(self.lattice_y), len(self.lattice_x)
        for iy, cy in enumerate(self.lattice_y):
            for ix, cx in enumerate(self.lattice_x):
                corner = iy in (0, ny - 1) and ix in (0, nx - 1)
                yield iy, ix, int(cy), int(cx), corner


def _lattice(lo: int, hi: int, step: int) -> np.ndarray:
    n = max(2, int(round((hi - lo) / step)) + 1)
    return np.round(np.linspace(lo, hi, n)).astype(int)


@lru_cache(maxsize=None)
def build_layout(profile_name: str) -> Layout:
    p = get_profile(profile_name)
    cols, rows, q = p.cols, p.rows, p.quiet
    kind = np.full((rows, cols), KIND_DATA, np.uint8)
    fixed = np.full((rows, cols), -1, np.int8)

    def put(r0, c0, pat, k):
        h, w = pat.shape
        kind[r0 : r0 + h, c0 : c0 + w] = k
        fixed[r0 : r0 + h, c0 : c0 + w] = pat

    # quiet zone
    kind[:q, :] = kind[-q:, :] = kind[:, :q] = kind[:, -q:] = KIND_QUIET
    fixed[kind == KIND_QUIET] = 1

    # finders + 1-cell white separator on the inner sides (reserved 8x8 corner squares)
    sep = np.ones((FINDER + 1, FINDER + 1), np.int8)
    corners = {  # top-left cell of the 8x8 reserved square, and where the 7x7 sits inside it
        "TL": (q, q, 0, 0),
        "TR": (q, cols - q - 8, 0, 1),
        "BR": (rows - q - 8, cols - q - 8, 1, 1),
        "BL": (rows - q - 8, q, 1, 0),
    }
    centers = []
    for r0, c0, dr, dc in corners.values():
        put(r0, c0, sep, KIND_FINDER)
        fr, fc = r0 + dr, c0 + dc
        put(fr, fc, FINDER_PATTERN, KIND_FINDER)
        centers.append((fc + FINDER / 2, fr + FINDER / 2))
    finder_centers = np.array(centers, dtype=np.float64)

    # alignment lattice: node centres span finder-core centre to finder-core centre
    step = max(10, round(p.align_step_px / p.cell_px))
    lat_x = _lattice(q + 3, cols - q - 4, step)
    lat_y = _lattice(q + 3, rows - q - 4, step)
    nx, ny = len(lat_x), len(lat_y)
    for iy, cy in enumerate(lat_y):
        for ix, cx in enumerate(lat_x):
            if iy in (0, ny - 1) and ix in (0, nx - 1):
                continue  # corner nodes are the finders
            put(cy - 2, cx - 2, ALIGN_PATTERN, KIND_ALIGN)

    # PN edge strips (row q / row rows-q-1 / col q / col cols-q-1, between the finder squares).
    # Distinct per edge and per profile -> resolves rotation/mirroring and identifies the profile.
    pn_r, pn_c, pn_b, pn_e = [], [], [], []
    strips = {
        "top": [(q, c) for c in range(q + 8, cols - q - 8)],
        "bottom": [(rows - q - 1, c) for c in range(q + 8, cols - q - 8)],
        "left": [(r, q) for r in range(q + 8, rows - q - 8)],
        "right": [(r, cols - q - 1) for r in range(q + 8, rows - q - 8)],
    }
    edge_id = {"top": 0, "right": 1, "bottom": 2, "left": 3}
    for edge, cells in strips.items():
        cells = [(r, c) for r, c in cells if kind[r, c] == KIND_DATA]
        bits = prng_bits(f"stega-pn-{p.name}-{edge}", len(cells))
        for (r, c), b in zip(cells, bits):
            kind[r, c] = KIND_PN
            fixed[r, c] = b
            pn_r.append(r)
            pn_c.append(c)
            pn_b.append(b)
            pn_e.append(edge_id[edge])

    data_r, data_c = np.nonzero(kind == KIND_DATA)  # row-major order
    return Layout(
        profile=p,
        cols=cols,
        rows=rows,
        kind=kind,
        fixed=fixed,
        data_rc=(data_r, data_c),
        finder_centers=finder_centers,
        lattice_x=lat_x,
        lattice_y=lat_y,
        pn_rc=(np.array(pn_r), np.array(pn_c)),
        pn_bits=np.array(pn_b, dtype=np.uint8),
        pn_edge=np.array(pn_e, dtype=np.uint8),
    )
