"""Camera frame -> sampled cell values -> bytes.

Pipeline per frame
  1. finder candidates: nested-contour test (black ring / white ring / black core) on a binarised image
  2. quad selection: best 4 candidates of similar size forming the largest convex quadrilateral
  3. orientation + profile: 8 corner assignments (4 rotations x mirror) x profiles; score by correlation
     of the known pseudo-noise edge strips sampled through each homography
  4. rectify to S px/cell, then measure local drift at every alignment pattern (template matching) and
     fit a smooth displacement field (handles lens distortion, screen bow, rolling-shutter shear)
  5. sample cell centres, normalise each channel with local black/white levels, nearest-palette decision
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations

import cv2
import numpy as np

from .framing import FrameCodec, FrameHeader, build_codec, symbols_to_bits
from .layout import ALIGN_PATTERN, Layout, banner_bits
from .render import palette

S = 6  # rectified pixels per cell
SEARCH = 3  # alignment search radius in cells


@dataclass
class FrameResult:
    status: str  # nofinder | noorient | banner | rsfail | dup | ok
    profile: str | None = None
    header: FrameHeader | None = None  # first decoded band (single-band profiles: the frame)
    symbol: bytes = b""
    payloads: list = field(default_factory=list)  # (header, symbol) of every decoded band; b"" = already known
    bands: int = 0
    bands_ok: int = 0
    pn_score: float = 0.0
    fixed_ber: float = 1.0
    align_ok: float = 0.0
    debug: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# 1-2. finders
# ---------------------------------------------------------------------------------------------
def _binarizations(gray: np.ndarray):
    h, w = gray.shape
    big = max(h, w)
    for div in (30, 60, 12):  # large blocks survive glare; small blocks keep thin rings of small finders open
        b = max(15, int(big / div) | 1)
        yield cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, b, 3)
    # motion blur / tiny far finders: the 1-cell rings wash out; an unsharp mask re-opens them
    sharp = cv2.addWeighted(gray, 2.0, cv2.GaussianBlur(gray, (0, 0), 2.0), -1.0, 0)
    yield cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, max(15, int(big / 60) | 1), 3)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    yield otsu


def _children(hier: np.ndarray, i: int):
    c = hier[i][2]
    while c >= 0:
        yield c
        c = hier[c][0]


@dataclass
class Finder:
    x: float
    y: float
    side: float
    corners: np.ndarray | None  # (4, 2) outer-square corners in image px (any order), if found


def finder_candidates(binimg: np.ndarray, min_side: float) -> list[Finder]:
    contours, hier = cv2.findContours(binimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    min_area = min_side * min_side
    max_area = (0.2 * max(binimg.shape)) ** 2  # a finder is 7 cells of a >=60-cell-wide grid
    out = []
    for i, cnt in enumerate(contours):
        if hier[i][2] < 0 or len(cnt) < 4:
            continue
        a0 = cv2.contourArea(cnt)
        if not min_area <= a0 <= max_area:
            continue
        for c in _children(hier, i):
            a1 = cv2.contourArea(contours[c])
            if a1 <= 0 or not 1.25 < a0 / a1 < 3.2:
                continue
            for g in _children(hier, c):
                a2 = cv2.contourArea(contours[g])
                if a2 <= 0 or not 1.6 < a1 / a2 < 7.0:
                    continue
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area <= 0 or a0 / hull_area < 0.85:
                    continue
                (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
                if min(rw, rh) <= 0 or max(rw, rh) / min(rw, rh) > 2.2:
                    continue
                m0, m2 = cv2.moments(cnt), cv2.moments(contours[g])
                if m0["m00"] <= 0 or m2["m00"] <= 0:
                    continue
                x0, y0 = m0["m10"] / m0["m00"], m0["m01"] / m0["m00"]
                x2, y2 = m2["m10"] / m2["m00"], m2["m01"] / m2["m00"]
                side = float(np.sqrt(a0))
                if np.hypot(x0 - x2, y0 - y2) > 0.2 * side:
                    continue
                poly = cv2.approxPolyDP(hull, 0.08 * cv2.arcLength(hull, True), True).reshape(-1, 2)
                corners = poly.astype(np.float64) if len(poly) == 4 else None
                # core centroid is the most precise centre; average with the ring centroid for noise
                out.append(Finder((x0 + 3 * x2) / 4, (y0 + 3 * y2) / 4, side, corners))
    return out


def _dedupe(cands: list[Finder]) -> list[Finder]:
    kept: list[Finder] = []
    for f in sorted(cands, key=lambda f: (f.corners is None, -f.side)):
        if all(np.hypot(f.x - k.x, f.y - k.y) > 0.5 * max(f.side, k.side) for k in kept):
            kept.append(f)
    return kept


def _cyclic(fs: list[Finder]) -> list[Finder]:
    c = np.mean([(f.x, f.y) for f in fs], axis=0)
    return sorted(fs, key=lambda f: np.arctan2(f.y - c[1], f.x - c[0]))  # clockwise on screen (y down)


def select_quad(cands: list[Finder]) -> list[Finder] | None:
    cands = _dedupe(cands)[:10]
    if len(cands) < 4:
        return None
    best, best_area = None, 0.0
    for combo in combinations(cands, 4):
        sides = [f.side for f in combo]
        if max(sides) / min(sides) > 2.5:
            continue
        hull = cv2.convexHull(np.float32([(f.x, f.y) for f in combo]))
        if len(hull) != 4:
            continue
        area = cv2.contourArea(hull)
        if area >= (4 * np.median(sides)) ** 2 and area > best_area:
            best, best_area = combo, area
    return _cyclic(list(best)) if best else None


def select_triple(cands: list[Finder]) -> list[Finder] | None:
    """Fallback when one finder is lost (glare, blur, occlusion): the 3 similar-sized candidates
    spanning the largest triangle."""
    cands = _dedupe(cands)[:10]
    best, best_area = None, 0.0
    for combo in combinations(cands, 3):
        sides = [f.side for f in combo]
        if max(sides) / min(sides) > 2.5:
            continue
        (x0, y0), (x1, y1), (x2, y2) = [(f.x, f.y) for f in combo]
        area = abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / 2
        if area >= 8 * np.median(sides) ** 2 and area > best_area:
            best, best_area = combo, area
    return _cyclic(list(best)) if best else None


def find_finders(gray: np.ndarray) -> list[Finder] | None:
    """4 (or, failing that, 3) finders in cyclic (clockwise on screen) order, or None."""
    min_side = max(10.0, 0.005 * max(gray.shape))
    pooled = []
    for b in _binarizations(gray):
        cands = finder_candidates(b, min_side)
        q = select_quad(cands)
        if q is not None and len(_dedupe(cands)) == 4:
            return q
        pooled.extend(cands)
    return select_quad(pooled) or select_triple(pooled)


# ---------------------------------------------------------------------------------------------
# 3. orientation / profile
# ---------------------------------------------------------------------------------------------
def _sample(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    mx = pts[:, 0].astype(np.float32).reshape(-1, 1)
    my = pts[:, 1].astype(np.float32).reshape(-1, 1)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(len(pts), -1)


@lru_cache(maxsize=None)
def _pn_points(profile_name: str):
    lay = build_codec(profile_name).layout
    r, c = lay.pn_rc
    pts = np.stack([c + 0.5, r + 0.5], axis=1).astype(np.float32).reshape(-1, 1, 2)
    expect = lay.pn_bits.astype(np.float32) * 2 - 1
    return pts, expect, lay.pn_edge


@lru_cache(maxsize=None)
def _pn_near_finder(profile_name: str, corners: tuple, radius: float = 24.0) -> np.ndarray:
    """PN cells within `radius` cells of one of the given (detected) finders."""
    lay = build_codec(profile_name).layout
    pts = _pn_points(profile_name)[0].reshape(-1, 2)
    fc = lay.finder_centers[list(corners)]
    return np.min(np.linalg.norm(pts[:, None, :] - fc[None, :, :], axis=2), axis=1) < radius


FINDER_CORNER_OFFS = np.array([[-3.5, -3.5], [3.5, -3.5], [3.5, 3.5], [-3.5, 3.5]])


def _hypotheses(fs: list[Finder]):
    """Yield (canonical corner indices, finders) assignments to TL,TR,BR,BL: 4 rotations x mirror for a
    quad. For a triple any vertex may be the middle corner (adjacent to both others): under strong pitch
    a near edge can be longer than the diagonal, so no geometric shortcut; 3 x 4 corners x mirror."""
    if len(fs) == 4:
        for seq in (fs, fs[::-1]):
            for r in range(4):
                yield (0, 1, 2, 3), seq[r:] + seq[:r]
        return
    for mid in range(3):
        for sign in (1, -1):
            ordered = [fs[(mid - sign) % 3], fs[mid], fs[(mid + sign) % 3]]
            for m in range(4):  # canonical corner of the middle vertex
                yield ((m - 1) % 4, m, (m + 1) % 4), ordered


def _fit_homography(lay: Layout, corners: tuple, fs: list[Finder]) -> np.ndarray | None:
    """Initial guess from finder centres (perspective for 4, affine for 3), then a least-squares
    homography over centres + the outer-square corners of every finder (12-20 points), which fixes
    the perspective an affine 3-point guess cannot represent."""
    src = lay.finder_centers[list(corners)].astype(np.float32)
    dst = np.float32([(f.x, f.y) for f in fs])
    if len(fs) == 4:
        H0 = cv2.getPerspectiveTransform(src, dst)
    else:
        H0 = np.vstack([cv2.getAffineTransform(src, dst), [0, 0, 1]])
    S_pts, D_pts = [src.astype(np.float64)], [dst.astype(np.float64)]
    for k, f in zip(corners, fs):
        if f.corners is None:
            continue
        canon = lay.finder_centers[k] + FINDER_CORNER_OFFS
        pred = cv2.perspectiveTransform(canon.reshape(-1, 1, 2), H0).reshape(-1, 2)
        d = np.linalg.norm(pred[:, None, :] - f.corners[None, :, :], axis=2)
        match = d.argmin(axis=1)
        if len(set(match)) == 4 and d[np.arange(4), match].max() < 0.45 * f.side:
            S_pts.append(canon)
            D_pts.append(f.corners[match])
    if len(S_pts) == 1:
        return H0 if len(fs) == 4 else None
    H, _ = cv2.findHomography(np.vstack(S_pts), np.vstack(D_pts), 0)
    return H if H is not None else H0


def orient(gray_blur: np.ndarray, fs: list[Finder], profiles: list[str]):
    """All (score, profile_name, H cell->image, known finders) hypotheses, best first. The PN strips sit
    on the grid edges where lens distortion bows most, so this score is a coarse ranking; the caller
    verifies the top hypotheses after refinement."""
    scored = []
    hyps = list(_hypotheses(fs))
    for name in profiles:
        lay = build_codec(name).layout
        pn_pts, expect_all, pn_edge = _pn_points(name)
        for corners, ordered in hyps:
            H = _fit_homography(lay, corners, ordered)
            if H is None:
                continue
            if len(corners) == 4:
                usable = np.ones(len(pn_edge), bool)
            else:  # skip the two edges running into the lost finder
                usable = np.isin(pn_edge, [e for e in range(4) if e in corners and (e + 1) % 4 in corners])
            proj = cv2.perspectiveTransform(pn_pts, H).reshape(-1, 2)
            v_all = _sample(gray_blur, proj)[:, 0]
            # Score the whole strip AND only its part near the finders, keep the better: H is pinned at
            # the finders, while lens distortion bows mid-edge cells off by more than a cell on dense grids.
            near = usable & _pn_near_finder(name, tuple(corners))
            best = None
            for sel in (usable, near):
                if sel.sum() < 24:
                    continue
                v = v_all[sel]
                sd = v.std()
                if sd < 1e-3:
                    continue
                sc = float(np.mean((v - v.mean()) / sd * expect_all[sel]))
                best = sc if best is None else max(best, sc)
            if best is None:
                continue
            known = {k: np.array([f.x, f.y]) for k, f in zip(corners, ordered)}
            scored.append((best, name, H, known))
    scored.sort(key=lambda t: -t[0])
    return scored


# ---------------------------------------------------------------------------------------------
# 4. rectification + local refinement
# ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _align_template() -> np.ndarray:
    n = 5 * S + 1
    ss = 8
    t = (np.arange(n * ss) / ss - 0.5 + 0.5 / ss) / S  # supersampled pixel positions -> cell coords
    inside = (t >= 0) & (t < 5)
    idx = np.clip(np.floor(t).astype(int), 0, 4)
    val = np.where(np.outer(inside, inside), ALIGN_PATTERN[np.ix_(idx, idx)].astype(np.float32), 0.5)
    return val.reshape(n, ss, n, ss).mean(axis=(1, 3)).astype(np.float32)


def _match(padded: np.ndarray, pad: int, cx: int, cy: int, radius_px: int, center_off=(0.0, 0.0)):
    tmpl = _align_template()
    tn = tmpl.shape[0]
    ox = int(round((cx - 2) * S + center_off[0])) - radius_px + pad
    oy = int(round((cy - 2) * S + center_off[1])) - radius_px + pad
    patch = padded[oy : oy + tn + 2 * radius_px, ox : ox + tn + 2 * radius_px]
    if patch.shape[0] < tn + 2 * radius_px or patch.shape[1] < tn + 2 * radius_px:
        return None, 0.0
    res = cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, (bx, by) = cv2.minMaxLoc(res)

    def parabola(a, b, c):
        d = a - 2 * b + c
        return 0.0 if abs(d) < 1e-6 else 0.5 * (a - c) / d

    sx = parabola(res[by, bx - 1], res[by, bx], res[by, bx + 1]) if 0 < bx < res.shape[1] - 1 else 0.0
    sy = parabola(res[by - 1, bx], res[by, bx], res[by + 1, bx]) if 0 < by < res.shape[0] - 1 else 0.0
    off = (ox - pad + bx + sx - (cx - 2) * S, oy - pad + by + sy - (cy - 2) * S)
    return off, float(score)


def _poly_basis(x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
    cols = [x**i * y**j for i in range(deg + 1) for j in range(deg + 1 - i)]
    return np.stack(cols, axis=-1)


def _warp(img: np.ndarray, lay: Layout, H: np.ndarray) -> np.ndarray:
    M = H @ np.diag([1.0 / S, 1.0 / S, 1.0])
    return cv2.warpPerspective(img, M, (lay.cols * S, lay.rows * S), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


CORNER_NODE = {0: (0, 0), 1: (0, -1), 2: (-1, -1), 3: (-1, 0)}  # canonical corner -> lattice (iy, ix)


def _corner_offsets(lay: Layout, H: np.ndarray, known: dict) -> dict:
    """Measured finder centres expressed as rect-px offsets from where H puts them."""
    Hinv = np.linalg.inv(H)
    out = {}
    for k, img_pt in known.items():
        cell = cv2.perspectiveTransform(np.array([[img_pt]], np.float64), Hinv)[0, 0]
        out[CORNER_NODE[k]] = tuple((cell - lay.finder_centers[k]) * S)
    return out


def bootstrap_homography(gray: np.ndarray, lay: Layout, H: np.ndarray, known: dict, iters: int) -> np.ndarray:
    """Re-estimate H from every alignment pattern that can be found (plus the detected finders).
    Starting from a rough (e.g. affine, 3-finder) guess, each pass locks onto more of the lattice."""
    for _ in range(iters):
        rect = _warp(gray, lay, H).astype(np.float32)
        pad = (SEARCH + 2) * S
        padded = cv2.copyMakeBorder(rect, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        src, rect_pts = [], []
        for _, _, cy, cx, corner in lay.lattice_nodes():
            if corner:
                continue
            off, score = _match(padded, pad, cx, cy, SEARCH * S)
            if off is not None and score > 0.55:
                src.append((cx + 0.5, cy + 0.5))
                rect_pts.append(((cx + 0.5) + off[0] / S, (cy + 0.5) + off[1] / S))
        if len(src) < 6:
            break
        dst = cv2.perspectiveTransform(np.array([rect_pts], np.float64), H)[0]
        src = np.vstack([np.array(src), lay.finder_centers[list(known)]])
        dst = np.vstack([dst, np.array(list(known.values()))])
        centre = np.array([[[lay.cols / 2, lay.rows / 2], [lay.cols / 2 + 1, lay.rows / 2]]])
        c = cv2.perspectiveTransform(centre, H)[0]
        cell_px = float(np.hypot(*(c[1] - c[0])))
        Hn, mask = cv2.findHomography(src, dst, cv2.RANSAC, max(1.5, 0.35 * cell_px))
        if Hn is None or mask.sum() < 6:
            break
        if len(known) == 4 and not mask[-4:].all():
            break  # a bowed (curved-screen) lattice: one plane can't fit it; keep the finder-based H
        H = Hn
    return H


def refine_field(gray_rect: np.ndarray, lay: Layout, corner_offsets: dict | None = None):
    """Displacement (dx, dy) in rect px for every lattice node. Corner nodes are the finders: they take
    the measured finder offsets (missing finder -> predicted from the fit)."""
    corner_offsets = corner_offsets or {}
    ny, nx = len(lay.lattice_y), len(lay.lattice_x)
    pad = (SEARCH + 2) * S
    padded = cv2.copyMakeBorder(gray_rect, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    DX = np.zeros((ny, nx), np.float64)
    DY = np.zeros((ny, nx), np.float64)
    good = np.zeros((ny, nx), bool)
    nodes = list(lay.lattice_nodes())
    measured_corner = np.zeros((ny, nx), bool)
    for (iy, ix), (dx, dy) in corner_offsets.items():
        DX[iy, ix], DY[iy, ix] = dx, dy
        good[iy, ix] = measured_corner[iy, ix] = True
    for iy, ix, cy, cx, corner in nodes:
        if corner:
            continue
        off, score = _match(padded, pad, cx, cy, SEARCH * S)
        if off is not None and score > 0.5:
            DX[iy, ix], DY[iy, ix] = off
            good[iy, ix] = True

    # robust smooth fit, then re-search suspicious nodes near the prediction
    wx = (lay.lattice_x - lay.lattice_x.mean()) / max(1, np.ptp(lay.lattice_x))
    wy = (lay.lattice_y - lay.lattice_y.mean()) / max(1, np.ptp(lay.lattice_y))
    GX, GY = np.meshgrid(wx, wy)
    inl = good.copy()
    pred_x = pred_y = None
    for _ in range(3):
        n = inl.sum()
        deg = 3 if n >= 16 else 2 if n >= 8 else 1
        A = _poly_basis(GX[inl], GY[inl], deg)
        cx_, *_ = np.linalg.lstsq(A, DX[inl], rcond=None)
        cy_, *_ = np.linalg.lstsq(A, DY[inl], rcond=None)
        Afull = _poly_basis(GX, GY, deg)
        pred_x, pred_y = Afull @ cx_, Afull @ cy_
        resid = np.hypot(DX - pred_x, DY - pred_y)
        new_inl = (good & (resid < 0.4 * S)) | measured_corner
        if (new_inl == inl).all():
            break
        inl = new_inl
    for iy, ix, cy, cx, corner in nodes:
        if inl[iy, ix]:
            continue
        if corner:  # lost finder: trust the smooth fit
            DX[iy, ix], DY[iy, ix] = pred_x[iy, ix], pred_y[iy, ix]
            continue
        off, score = _match(padded, pad, cx, cy, S, center_off=(pred_x[iy, ix], pred_y[iy, ix]))
        if off is not None and score > 0.45 and np.hypot(off[0] - pred_x[iy, ix], off[1] - pred_y[iy, ix]) < 0.4 * S:
            DX[iy, ix], DY[iy, ix] = off
            inl[iy, ix] = True
        else:
            DX[iy, ix], DY[iy, ix] = pred_x[iy, ix], pred_y[iy, ix]
    return DX, DY, float(inl.mean())


def _interp_weights(nodes: np.ndarray, n_cells: int):
    c = np.arange(n_cells, dtype=np.float64)
    i0 = np.clip(np.searchsorted(nodes, c, side="right") - 1, 0, len(nodes) - 2)
    t = (c - nodes[i0]) / (nodes[i0 + 1] - nodes[i0])
    return i0, np.clip(t, -0.35, 1.35)


def cell_maps(lay: Layout, DX: np.ndarray, DY: np.ndarray):
    ix0, tx = _interp_weights(lay.lattice_x.astype(np.float64), lay.cols)
    iy0, ty = _interp_weights(lay.lattice_y.astype(np.float64), lay.rows)

    def bilinear(D):
        a = D[np.ix_(iy0, ix0)]
        b = D[np.ix_(iy0, ix0 + 1)]
        c = D[np.ix_(iy0 + 1, ix0)]
        d = D[np.ix_(iy0 + 1, ix0 + 1)]
        tx_, ty_ = tx[None, :], ty[:, None]
        return (a * (1 - tx_) + b * tx_) * (1 - ty_) + (c * (1 - tx_) + d * tx_) * ty_

    xs = (np.arange(lay.cols) + 0.5) * S
    ys = (np.arange(lay.rows) + 0.5) * S
    mapx = (xs[None, :] + bilinear(DX)).astype(np.float32)
    mapy = (ys[:, None] + bilinear(DY)).astype(np.float32)
    return mapx, mapy


# ---------------------------------------------------------------------------------------------
# 5. classification
# ---------------------------------------------------------------------------------------------
def normalise(values: np.ndarray) -> np.ndarray:
    V = values.astype(np.float32)
    ker = np.ones((5, 5), np.uint8)
    hi = cv2.blur(cv2.dilate(V, ker), (9, 9))
    lo = cv2.blur(cv2.erode(V, ker), (9, 9))
    return (V - lo) / np.maximum(hi - lo, 12.0)


_CROSS = np.float32([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
EQ_BETAS = (1.2, 2.2)  # primary, retry-on-RS-failure


def equalise(N: np.ndarray, beta: float) -> np.ndarray:
    """Cell-domain zero-forcing-ish equaliser: camera blur leaks each cell into its 4 neighbours, so
    push every cell away from its neighbourhood mean. Measured on the harsh camsim preset it cuts
    fixed-cell errors ~3x and turns most RS failures into successes; neutral on sharp captures."""
    return N + beta * (N - cv2.filter2D(N, -1, _CROSS, borderType=cv2.BORDER_REFLECT))


@lru_cache(maxsize=None)
def _blur_fit_cells(profile_name: str):
    """Known cells (finders, alignment, PN, quiet zone) whose 4 neighbours are also known and differ
    from them: (rows, cols, own value, neighbour mean) — the samples that reveal how much blur leaks."""
    lay = build_codec(profile_name).layout
    F = lay.fixed.astype(np.float32)
    known = np.pad(lay.fixed >= 0, 1, constant_values=False)
    nb_known = known[:-2, 1:-1] & known[2:, 1:-1] & known[1:-1, :-2] & known[1:-1, 2:]
    Fp = np.pad(F, 1, mode="edge")
    m = (Fp[:-2, 1:-1] + Fp[2:, 1:-1] + Fp[1:-1, :-2] + Fp[1:-1, 2:]) / 4
    sel = known[1:-1, 1:-1] & nb_known & (np.abs(F - m) >= 0.25)
    r, c = np.nonzero(sel)
    return r, c, F[sel], m[sel]


def estimate_beta(N: np.ndarray, lay: Layout) -> float:
    """Equaliser strength that undoes THIS frame's blur. Model each known cell as
    observed = a + p*own + q*neighbour_mean; the leaked fraction is k = q/(p+q) and the first-order
    inverse is equalise(beta = k/(1-k)). Multi-level (gray) cells need this: an over-strong beta keeps
    the sign of a B/W cell but throws intermediate levels to the extremes."""
    r, c, x, m = _blur_fit_cells(lay.profile.name)
    if len(x) < 30:
        return EQ_BETAS[0]
    y = N[r, c].mean(axis=1)
    A = np.stack([np.ones_like(x), x, m], axis=1)
    (_, p, q), *_ = np.linalg.lstsq(A, y, rcond=None)
    if p <= 0.05 or p + q <= 0:
        return EQ_BETAS[0]
    k = float(np.clip(q / (p + q), 0.0, 0.7))
    return k / (1.0 - k)


def eq_betas(N: np.ndarray, lay: Layout) -> tuple[float, ...]:
    """Equaliser strengths to try, best guess first."""
    if lay.profile.palette != "gray":
        return EQ_BETAS  # B/W and colour: validated on real recordings, tolerant of overshoot
    b = estimate_beta(N, lay)
    return tuple(dict.fromkeys(round(x, 3) for x in (b, b * 0.6, b * 1.4, 0.0)))


def classify_cells(N: np.ndarray, lay: Layout) -> tuple[np.ndarray, np.ndarray]:
    """-> (palette index, decision margin) for every data cell, in transmission order."""
    bpc = lay.profile.bits_per_cell
    r, c = lay.data_rc
    X = N[r, c]  # (n, 3)
    if bpc == 1:
        g = X.mean(axis=1)
        sym = (g > 0.5).astype(np.uint8)
        margin = np.abs(g - 0.5) * 2
    else:
        P = palette(bpc, lay.profile.palette)
        if lay.profile.palette == "gray":
            # luma only: averaging R,G,B cuts noise and ignores the camera's colour cast
            d = np.abs(X.mean(axis=1)[:, None] - P[None, :, 0])
        else:
            d = np.sqrt(((X[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        part = np.partition(d, 1, axis=1)
        sym = d.argmin(axis=1).astype(np.uint8)
        margin = part[:, 1] - part[:, 0]
    return sym, margin


def classify(N: np.ndarray, lay: Layout):
    """Whole-frame view: -> (scrambled byte stream, per-byte confidence, symbols). Mostly for analysis;
    decoding goes band by band via UnitCodec.stream_from_cells."""
    bpc = lay.profile.bits_per_cell
    sym, margin = classify_cells(N, lay)
    cap = lay.capacity_bytes
    bits = symbols_to_bits(sym, bpc)[: cap * 8]
    conf = np.repeat(margin, bpc)[: cap * 8].reshape(cap, 8).min(axis=1)
    return np.packbits(bits), conf, sym


def fixed_cell_ber(N: np.ndarray, lay: Layout) -> float:
    m = (lay.fixed >= 0) & (lay.kind != 1)  # finder/align/PN, not the quiet zone
    g = N[m].mean(axis=1) > 0.5
    return float(np.mean(g != (lay.fixed[m] == 1)))


# ---------------------------------------------------------------------------------------------
# full frame
# ---------------------------------------------------------------------------------------------
def _sample_grid(img_rgb: np.ndarray, gray: np.ndarray, lay: Layout, H0: np.ndarray, known: dict, debug: bool):
    """Sample with the lattice-bootstrapped homography; if that looks poor, also try the finder-only
    homography and keep whichever gives the lower fixed-cell error."""
    Hb = bootstrap_homography(gray, lay, H0, known, iters=1 if len(known) == 4 else 3)
    best = _sample_with(img_rgb, lay, Hb, known, debug)
    if best[2] > 0.02 and Hb is not H0:
        alt = _sample_with(img_rgb, lay, H0, known, debug)
        if alt[2] < best[2]:
            best = alt
    return best


def _sample_with(img_rgb: np.ndarray, lay: Layout, H: np.ndarray, known: dict, debug: bool):
    rect = _warp(img_rgb, lay, H)
    rect_gray = cv2.cvtColor(rect, cv2.COLOR_RGB2GRAY).astype(np.float32)
    DX, DY, align_ok = refine_field(rect_gray, lay, _corner_offsets(lay, H, known))
    mapx, mapy = cell_maps(lay, DX, DY)
    k = max(1, S // 2)
    blurred = cv2.blur(rect, (k, k)).astype(np.float32)
    values = cv2.remap(blurred, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    N = normalise(values)
    dbg = dict(rect=rect, mapx=mapx, mapy=mapy, N=N, H=H) if debug else {}
    return N, align_ok, fixed_cell_ber(equalise(N, EQ_BETAS[0]), lay), dbg


def read_frame(
    img_rgb: np.ndarray, profiles: list[str], skip=None, debug: bool = False, min_pn: float = 0.15, max_fixed_ber: float = 0.2
) -> FrameResult:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    pts = find_finders(gray_blur)
    if pts is None:
        return FrameResult("nofinder")
    hyps = orient(gray_blur.astype(np.float32), pts, profiles)
    res = FrameResult("noorient", pn_score=hyps[0][0] if hyps else 0.0)
    if debug:
        res.debug["finders"] = pts
    best_score = hyps[0][0] if hyps else 0.0
    chosen = None
    for score, name, H, known in hyps[:3]:
        if score < max(min_pn, 0.5 * best_score):
            break
        lay = build_codec(name).layout
        N, align_ok, ber, dbg = _sample_grid(img_rgb, gray, lay, H, known, debug)
        if ber < res.fixed_ber:
            res.profile, res.pn_score, res.align_ok, res.fixed_ber = name, score, align_ok, ber
            res.debug.update(dbg)
            chosen = (name, N)
        if ber <= max_fixed_ber:
            break
    if chosen is None or res.fixed_ber > max_fixed_ber:
        return res
    name, N = chosen
    codec: FrameCodec = build_codec(name)
    decoded: dict[int, tuple] = {}
    for beta in eq_betas(N, codec.layout):
        sym, margin = classify_cells(equalise(N, beta), codec.layout)
        for b, unit in enumerate(codec.units):
            if b in decoded:
                continue
            stream, conf = unit.stream_from_cells(sym, margin)
            out = unit.decode(stream, conf, skip=skip)
            if out is not None:
                decoded[b] = out
        if len(decoded) == codec.bands:
            break
    res.bands, res.bands_ok = codec.bands, len(decoded)
    if not decoded:
        # a lead-in/lead-out banner reads cleanly but carries no payload: report it as idle, not as
        # a failure, so "how good is my recording?" statistics stay meaningful
        hard = N[codec.layout.data_rc].mean(axis=1) > 0.5
        if np.mean(hard == (banner_bits(name) == 1)) > 0.75:
            res.status = "banner"
            return res
        res.status = "rsfail"
        return res
    res.payloads = [decoded[b] for b in sorted(decoded)]
    fresh = [p for p in res.payloads if p[1]]
    res.header, res.symbol = (fresh or res.payloads)[0]
    res.status = "ok" if fresh else "dup"
    return res
