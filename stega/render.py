"""Cell grid -> RGB canvas. Palettes are points of the unit RGB cube so the decoder can classify
cells by nearest neighbour after per-channel local black/white normalisation."""

from __future__ import annotations

import cv2
import numpy as np

from .layout import Layout, banner_bits

PALETTES = {
    1: np.array([[0, 0, 0], [1, 1, 1]], np.float32),
    # tetrahedron inside the cube (max. min-distance for 4 colours): K, C, M, Y  as (R,G,B)
    2: np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], np.float32),
    # all 8 corners; symbol bits == (R,G,B) bits, so a single-channel error flips a single bit
    3: np.array([[(i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(8)], np.float32),
}

LEVEL_LO, LEVEL_HI = 16, 235  # keep off the clipping rails of video range


def cell_grid_rgb(layout: Layout, data_symbols: np.ndarray | None, fill_gray: int | None = None) -> np.ndarray:
    """(rows, cols, 3) uint8 grid of cell colours."""
    bpc = layout.profile.bits_per_cell
    grid = np.empty((layout.rows, layout.cols, 3), np.uint8)
    fixed = layout.fixed
    grid[fixed == 0] = LEVEL_LO
    grid[fixed == 1] = LEVEL_HI
    r, c = layout.data_rc
    if data_symbols is None:
        grid[r, c] = fill_gray if fill_gray is not None else 128
    else:
        pal = PALETTES[bpc]
        colors = (LEVEL_LO + pal * (LEVEL_HI - LEVEL_LO)).astype(np.uint8)
        grid[r, c] = colors[data_symbols]
    return grid


def grid_to_canvas(layout: Layout, grid: np.ndarray) -> np.ndarray:
    """Upscale the grid (nearest) and centre it on a white WxH canvas. Offsets are even."""
    p = layout.profile
    img = cv2.resize(grid, (layout.cols * p.cell_px, layout.rows * p.cell_px), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((p.height, p.width, 3), LEVEL_HI, np.uint8)
    oy = ((p.height - img.shape[0]) // 2) & ~1
    ox = ((p.width - img.shape[1]) // 2) & ~1
    canvas[oy : oy + img.shape[0], ox : ox + img.shape[1]] = img
    return canvas


def render_frame(layout: Layout, data_symbols: np.ndarray) -> np.ndarray:
    return grid_to_canvas(layout, cell_grid_rgb(layout, data_symbols))


def render_banner(layout: Layout, lines: list[str]) -> np.ndarray:
    """Lead-in / lead-out frame: all markers visible (so the user can frame the shot) + big text.
    The data area carries a known pattern, which makes these frames identifiable as idle."""
    bits = banner_bits(layout.profile.name)
    grid = np.empty((layout.rows, layout.cols, 3), np.uint8)
    fixed = layout.fixed
    grid[fixed == 0] = LEVEL_LO
    grid[fixed == 1] = LEVEL_HI
    r, c = layout.data_rc
    grid[r, c] = np.where(bits[:, None] == 1, LEVEL_HI, LEVEL_LO)
    canvas = grid_to_canvas(layout, grid)
    h, w = canvas.shape[:2]
    scale = w / 1920 * 2.2
    box_h = int(len(lines) * 55 * scale)
    y0 = h // 2 - box_h // 2
    cv2.rectangle(canvas, (int(w * 0.1), y0 - int(20 * scale)), (int(w * 0.9), y0 + box_h), (LEVEL_HI,) * 3, -1)
    y = y0 + int(30 * scale)
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 4)
        cv2.putText(canvas, line, ((w - tw) // 2, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (LEVEL_LO,) * 3, 4, cv2.LINE_AA)
        y += int(55 * scale)
    return canvas
