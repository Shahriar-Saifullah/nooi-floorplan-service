"""
Floor Plan Analysis Pipeline v3 — polygon-based reconstruction
---------------------------------------------------------------
Architecture (geometry-first, OCR names only):
  1. Binarize ink, OCR all text, then ERASE text from the ink mask
  2. Wall mask     = thick strokes only (auto-estimated wall thickness)
  3. Rooms         = watershed on sealed wall mask -> true room POLYGONS
  4. Names         = OCR labels assigned by point-in-polygon
  5. Walls         = vectorized H/V centerlines from the wall mask
  6. Openings      = classified along each wall band:
                       no wall + no ink    -> door (gap)
                       no wall + thin ink  -> window (glazing lines)
  7. Scale         = parsed from dimension strings like 14'-7" x 16'

Pure OpenCV + Tesseract. No external AI APIs. CPU-only.
"""

import logging
import re
import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── Room type colors ──────────────────────────────────────────────────────────

ROOM_COLORS = {
    "living": "#c3f4f0", "great": "#c3f4f0", "family": "#c3f4f0",
    "kitchen": "#b9eac5",
    "bedroom": "#87ddd7", "bed": "#87ddd7", "master": "#6dd0c4",
    "bath": "#f7dfad", "powder": "#f7dfad", "pwdr": "#f7dfad", "wc": "#f7dfad",
    "hall": "#d5dbda", "corridor": "#d5dbda", "foyer": "#d5dbda", "entry": "#d5dbda",
    "storage": "#ffc9c0", "closet": "#ffc9c0", "wic": "#ffc9c0", "w.i.c": "#ffc9c0",
    "laundry": "#ffc9c0", "utility": "#ffc9c0",
    "dining": "#c7d2fe",
    "study": "#fde68a", "office": "#fde68a", "den": "#fde68a",
    "porch": "#a7f3d0", "patio": "#a7f3d0", "balcony": "#a7f3d0", "deck": "#a7f3d0",
    "terrace": "#a7f3d0", "covered": "#a7f3d0",
    "stair": "#e0c3fc", "elev": "#e5e7eb", "garage": "#e5e7eb",
}
FALLBACK_COLORS = [
    "#c3f4f0", "#b9eac5", "#87ddd7", "#f7dfad", "#d5dbda", "#ffc9c0",
    "#c7d2fe", "#fde68a", "#a7f3d0", "#e0c3fc", "#6dd0c4", "#e5e7eb",
]


def room_color(name: str, idx: int) -> str:
    lower = name.lower()
    for key, color in ROOM_COLORS.items():
        if key in lower:
            return color
    return FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]


# ── 1. Pre-process ────────────────────────────────────────────────────────────

def preprocess(img_bytes: bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Cannot decode image")

    grey_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Upscale small images — everything downstream is more stable at >=1400px
    h0, w0 = bgr.shape[:2]
    work_scale = 1.0
    if max(h0, w0) < 1400:
        work_scale = 1400.0 / max(h0, w0)
        bgr = cv2.resize(bgr, None, fx=work_scale, fy=work_scale,
                         interpolation=cv2.INTER_CUBIC)

    h, w = bgr.shape[:2]
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Ink mask: dark strokes on light background (255 = ink)
    _, otsu = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = cv2.bitwise_not(otsu) if np.mean(otsu) > 127 else otsu.copy()

    return bgr, grey, grey_orig, ink, h, w, work_scale


# ── 2. OCR ────────────────────────────────────────────────────────────────────

_DIM_RE = re.compile(
    r"""(\d{1,3})\s*['’]\s*(?:-?\s*(\d{1,2})\s*["”])?"""   # 14'-7"  or  16'
    , re.VERBOSE)

def _is_dimension(text: str) -> bool:
    t = text.strip()
    if re.fullmatch(r"[\d\s'’\"”xX×./\-+()°]+", t):
        return True
    return False


def ocr_words(grey_orig, work_scale: float, h: int, w: int) -> list:
    """OCR on the RAW original grayscale (no threshold/denoise — Tesseract's
    own binarization beats ours on clean plans), upscaled ~4x. psm 6 reads the
    stacked room labels far better than sparse-text mode. Word boxes are
    returned in WORKING-image coordinates."""
    try:
        import pytesseract
    except ImportError:
        log.warning("pytesseract not available")
        return []

    oh, ow = grey_orig.shape[:2]
    ocr_scale = max(3.0, 2200.0 / max(oh, ow))
    up = cv2.resize(grey_orig, None, fx=ocr_scale, fy=ocr_scale,
                    interpolation=cv2.INTER_CUBIC)

    words = []
    try:
        data = pytesseract.image_to_data(
            up, config="--psm 6 --oem 3",
            output_type=pytesseract.Output.DICT)
    except Exception as e:
        log.error(f"Tesseract error: {e}")
        return []

    to_work = work_scale / ocr_scale
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if conf < 30 or not text:
            continue
        x = int(data["left"][i] * to_work)
        y = int(data["top"][i] * to_work)
        bw = max(1, int(data["width"][i] * to_work))
        bh = max(1, int(data["height"][i] * to_work))
        # sanity: word boxes bigger than ~1.5% of the image are misreads of
        # graphics; erasing them would punch holes in walls
        if bw * bh > 0.015 * h * w:
            continue
        words.append({
            "text": text, "conf": conf,
            "x": x, "y": y, "w": bw, "h": bh,
            "cx": x + bw // 2, "cy": y + bh // 2,
            "block": data["block_num"][i], "line": data["line_num"][i],
            "par": data["par_num"][i],
        })
    log.info(f"  OCR: {len(words)} words")
    return words


def ocr_dimensions(grey_orig, work_scale: float) -> list:
    """Second, digit-focused OCR pass for dimension strings (14'-7" x 16').
    They're tiny — needs a bigger upscale and sparse-text mode."""
    try:
        import pytesseract
    except ImportError:
        return []
    oh, ow = grey_orig.shape[:2]
    scale = max(4.0, 3400.0 / max(oh, ow))
    up = cv2.resize(grey_orig, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC)
    try:
        data = pytesseract.image_to_data(
            up, config="--psm 11 --oem 3", output_type=pytesseract.Output.DICT)
    except Exception:
        return []
    out = []
    to_work = work_scale / scale
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text or int(data["conf"][i]) < 20:
            continue
        if not re.search(r"\d", text):
            continue
        x = int(data["left"][i] * to_work); y = int(data["top"][i] * to_work)
        bw = max(1, int(data["width"][i] * to_work))
        bh = max(1, int(data["height"][i] * to_work))
        out.append({"text": text, "conf": int(data["conf"][i]),
                    "x": x, "y": y, "w": bw, "h": bh,
                    "cx": x + bw // 2, "cy": y + bh // 2})
    return out


def erase_text(ink, words, h: int, w: int):
    """Remove OCR'd text strokes from the ink mask so they don't pollute walls."""
    clean = ink.copy()
    for wd in words:
        pad = 2
        x0 = max(0, wd["x"] - pad); y0 = max(0, wd["y"] - pad)
        x1 = min(w, wd["x"] + wd["w"] + pad); y1 = min(h, wd["y"] + wd["h"] + pad)
        clean[y0:y1, x0:x1] = 0
    return clean


# ── 3. Wall mask ──────────────────────────────────────────────────────────────

def estimate_wall_thickness(ink_clean) -> int:
    """Walls are the thickest strokes. distanceTransform peak * 2 ≈ thickness."""
    dt = cv2.distanceTransform(ink_clean, cv2.DIST_L2, 5)
    vals = dt[dt > 1.0]
    if vals.size == 0:
        return 4
    t = int(round(2.0 * np.percentile(vals, 97)))
    return max(3, min(t, 40))


def wall_mask_from_ink(ink_clean, thickness: int):
    """Opening with a kernel just under wall thickness keeps walls,
    erases thin furniture / dimension / symbol strokes."""
    k = max(2, int(thickness * 0.55))
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    walls = cv2.morphologyEx(ink_clean, cv2.MORPH_OPEN, kern)
    # Re-connect small nicks along walls
    kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, kern2)
    return walls


# ── 4. Room segmentation (watershed) ─────────────────────────────────────────

def long_thin_lines(ink_clean, wall_mask, thickness: int):
    """Long straight strokes that are NOT walls: window glazing, patio/porch
    outlines. These act as room boundaries even though they're thin."""
    thin = cv2.subtract(ink_clean, wall_mask)
    L = max(7, int(thickness * 2.0)) | 1
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))
    lines = cv2.bitwise_or(cv2.morphologyEx(thin, cv2.MORPH_OPEN, kh),
                           cv2.morphologyEx(thin, cv2.MORPH_OPEN, kv))
    return lines


def build_seal(barrier, walls_px: list, thickness: int, h: int, w: int,
               diag_segments: list | None = None):
    """
    Seal raster for room segmentation:
      barrier  +  rasterized vector walls  +  endpoint extensions.
    Extending each wall endpoint along its own axis until it hits barrier ink
    (within a door-sized reach) closes door gaps regardless of where they sit
    (mid-wall or at corners) WITHOUT filling room interiors the way a
    morphological closing does.
    """
    seal = barrier.copy()
    tk = max(2, int(thickness * 0.8))
    reach = int(thickness * 14)

    wallras = np.zeros((h, w), np.uint8)
    for seg in walls_px:
        p1 = (int(seg["x1"]), int(seg["y1"]))
        p2 = (int(seg["x2"]), int(seg["y2"]))
        cv2.line(seal, p1, p2, 255, tk)
        cv2.line(wallras, p1, p2, 255, tk)
    target = cv2.bitwise_or(barrier, wallras)

    def _extend(px, py, dx, dy):
        # skip past our own wall body first
        step = 1
        while step < thickness * 2:
            x, y = px + dx * step, py + dy * step
            if x < 0 or y < 0 or x >= w or y >= h:
                return
            if target[y, x] == 0:
                break
            step += 1
        for s in range(step, reach):
            x, y = px + dx * s, py + dy * s
            if x < 0 or y < 0 or x >= w or y >= h:
                return
            if target[y, x] > 0:
                cv2.line(seal, (px, py), (x, y), 255, tk)
                return

    for seg in walls_px:
        horiz = abs(seg["x2"] - seg["x1"]) >= abs(seg["y2"] - seg["y1"])
        x1, y1 = int(seg["x1"]), int(seg["y1"])
        x2, y2 = int(seg["x2"]), int(seg["y2"])
        if horiz:
            lo, hi = (x1, x2) if x1 < x2 else (x2, x1)
            _extend(lo, y1, -1, 0)
            _extend(hi, y1, +1, 0)
        else:
            lo, hi = (y1, y2) if y1 < y2 else (y2, y1)
            _extend(x1, lo, 0, -1)
            _extend(x1, hi, 0, +1)

    # Bridge COLLINEAR wall pairs across wide glazed openings (sliding doors,
    # window bands). Collinearity makes long bridges safe: if two wall stubs
    # line up, the gap between them is an opening in that same wall.
    col_tol = thickness * 1.2
    max_span = thickness * 14
    hsegs = [s for s in walls_px
             if abs(s["x2"] - s["x1"]) >= abs(s["y2"] - s["y1"])]
    vsegs = [s for s in walls_px
             if abs(s["x2"] - s["x1"]) < abs(s["y2"] - s["y1"])]

    for group, horiz in ((hsegs, True), (vsegs, False)):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if horiz:
                    if abs(a["y1"] - b["y1"]) > col_tol:
                        continue
                    amin, amax = sorted((a["x1"], a["x2"]))
                    bmin, bmax = sorted((b["x1"], b["x2"]))
                    gap = max(amin, bmin) - min(amax, bmax)
                    if 0 < gap <= max_span:
                        y = int((a["y1"] + b["y1"]) / 2)
                        x1g = int(min(amax, bmax)); x2g = int(max(amin, bmin))
                        cv2.line(seal, (x1g, y), (x2g, y), 255, tk)
                else:
                    if abs(a["x1"] - b["x1"]) > col_tol:
                        continue
                    amin, amax = sorted((a["y1"], a["y2"]))
                    bmin, bmax = sorted((b["y1"], b["y2"]))
                    gap = max(amin, bmin) - min(amax, bmax)
                    if 0 < gap <= max_span:
                        x = int((a["x1"] + b["x1"]) / 2)
                        y1g = int(min(amax, bmax)); y2g = int(max(amin, bmin))
                        cv2.line(seal, (x, y1g), (x, y2g), 255, tk)

    if diag_segments:
        for (x1, y1, x2, y2) in diag_segments:
            cv2.line(seal, (int(x1), int(y1)), (int(x2), int(y2)), 255, tk)

    return seal


def diagonal_walls(barrier, thickness: int, h: int, w: int) -> list:
    """Hough for non-axis wall segments (angled walls) to add to the seal."""
    edges = cv2.Canny(barrier, 50, 150)
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180, 40,
                          minLineLength=int(thickness * 2.5),
                          maxLineGap=int(thickness * 1.5))
    out = []
    if raw is None:
        return out
    for line in raw:
        x1, y1, x2, y2 = line[0]
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        L = np.hypot(dx, dy)
        if L < thickness * 2.5:
            continue
        ang = np.degrees(np.arctan2(dy, max(1, dx)))
        if 20 < ang < 70:           # genuinely diagonal only
            out.append((x1, y1, x2, y2))
    return out


def segment_rooms(seal, wall_mask, ink_clean, label_points: list,
                  thickness: int, h: int, w: int):
    """
    1. free space = NOT seal (seal already has door gaps bridged by vectors)
    2. multi-scale EROSION of free space: doorways are narrow, rooms are wide,
       so eroding by > door-half-width splits rooms apart regardless of the
       doorway's orientation. Larger erosions find big rooms; smaller erosions
       add small rooms (WC, closets) in areas not yet claimed.
    3. watershed grows all seeds back to the true wall faces
    Returns int32 label map (0 = outside/walls, 1..N = rooms).
    """
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sealed = cv2.morphologyEx(seal, cv2.MORPH_CLOSE, k3)
    free = cv2.bitwise_not(sealed)

    # Building footprint: dilate walls until exterior openings close, fill all
    # holes (rooms included — that IS the footprint), erode back. Immune to
    # door/window gaps, no flood-fill leak fragility. Escalate the radius until
    # the footprint stops growing appreciably (wide porch openings need more).
    footprint = None
    prev_area = 0
    for mult in (3, 5, 7, 9):
        r = max(4, int(thickness * mult))
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        fat = cv2.dilate(seal, kern)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(fat, 8)
        if n > 1:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            fat = np.where(lbl == biggest, 255, 0).astype(np.uint8)
        ffill = fat.copy()
        m2 = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ffill, m2, (0, 0), 255)      # fill the outside
        holes = cv2.bitwise_not(ffill)              # interior holes
        cand = cv2.erode(cv2.bitwise_or(fat, holes), kern)
        area = int(np.count_nonzero(cand))
        if footprint is None or area > prev_area * 1.10:
            footprint, prev_area = cand, area
        else:
            break

    inside = cv2.bitwise_and(footprint, free)
    log.info(f"  Interior: {np.count_nonzero(inside)/(h*w)*100:.0f}% of image")

    min_area = h * w * 0.0012
    claimed = np.zeros((h, w), np.uint8)
    seeds = []

    for mult in (3.0, 2.0, 1.3, 0.8):
        r = max(2, int(thickness * mult))
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        eroded = cv2.erode(inside, kern)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(eroded, 8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < max(30, min_area * 0.25):
                continue
            comp = (lbl == i)
            if claimed[comp].any():
                continue
            # restore the seed to its full (un-eroded) free-space component
            grown = cv2.dilate(comp.astype(np.uint8) * 255, kern)
            grown = cv2.bitwise_and(grown, inside)
            if int(np.count_nonzero(grown)) < min_area:
                continue
            claimed |= comp.astype(np.uint8)
            seeds.append(grown)

    if not seeds:
        log.info("  Rooms segmented: 0")
        return np.zeros((h, w), np.int32), 0

    # Watershed grows all seeds out to the seal, then we clip to the interior
    # so nothing escapes the footprint. Same result as iterative competitive
    # growth, ~100x faster.
    labels = np.zeros((h, w), np.int32)
    markers = np.zeros((h, w), np.int32)
    markers[cv2.bitwise_and(free, cv2.bitwise_not(inside)) > 0] = 1  # outside
    for mi, s in enumerate(seeds):
        markers[s > 0] = mi + 2
    cv2.watershed(cv2.cvtColor(sealed, cv2.COLOR_GRAY2BGR), markers)
    inside_b = inside > 0
    labels = np.where((markers >= 2) & inside_b, markers - 1, 0).astype(np.int32)

    # Unclaimed interior components (patio / porch: sealed off, no seed
    # survived erosion or none placed) become rooms of their own.
    n_rooms = len(seeds)
    # Drop grown regions that touch the image border — those are strips
    # between dimension lines and the building, not rooms.
    for rid in range(1, n_rooms + 1):
        ys, xs = np.where(labels == rid)
        if ys.size and (ys.min() <= 2 or xs.min() <= 2 or
                        ys.max() >= h - 3 or xs.max() >= w - 3):
            labels[labels == rid] = 0
    leftover = (inside_b & (labels == 0)).astype(np.uint8) * 255
    nL, lblL, statsL, _ = cv2.connectedComponentsWithStats(leftover, 8)
    min_area = h * w * 0.004
    for i in range(1, nL):
        if statsL[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y = statsL[i, cv2.CC_STAT_LEFT], statsL[i, cv2.CC_STAT_TOP]
        ww, hh = statsL[i, cv2.CC_STAT_WIDTH], statsL[i, cv2.CC_STAT_HEIGHT]
        if x <= 2 or y <= 2 or x + ww >= w - 2 or y + hh >= h - 2:
            continue              # touches image border: not a room
        n_rooms += 1
        labels[lblL == i] = n_rooms

    labels[seal > 0] = 0
    labels = _merge_unwalled(labels, n_rooms, wall_mask, ink_clean, footprint,
                             label_points, thickness, h, w)
    n_rooms = int(labels.max())
    log.info(f"  Rooms segmented: {n_rooms}")
    return labels, n_rooms, footprint


def _merge_unwalled(labels, n: int, wall_mask, ink_clean, footprint,
                    label_points: list, thickness: int, h: int, w: int):
    """Merge adjacent regions whose shared boundary has almost no true wall
    pixels — split by furniture lines or seal bridges, not real walls.
    EXCEPTION: at the building footprint edge, thin ink (window glazing,
    slider dashes) is a real indoor/outdoor divider — those stay separate.
    Interior furniture ink never blocks a merge."""
    # Which regions own a room-name label? A labelled region is a room the
    # architect named — it must NEVER merge with another labelled region,
    # even across a pure open-plan pass-through (Kitchen / Dining / Great
    # Room stay three rooms; the 3D still builds no wall between them
    # because only real wall pixels become walls).
    def _region_at(cx: int, cy: int) -> int:
        rid = int(labels[cy, cx]) if 0 <= cy < h and 0 <= cx < w else 0
        if rid:
            return rid
        for r in range(3, 22, 3):
            for dx, dy in ((r, 0), (-r, 0), (0, r), (0, -r),
                           (r, r), (-r, -r), (r, -r), (-r, r)):
                x, y = cx + dx, cy + dy
                if 0 <= x < w and 0 <= y < h and labels[y, x]:
                    return int(labels[y, x])
        return 0

    labelled: dict = {}
    for (lx, ly) in label_points:
        rid = _region_at(int(lx), int(ly))
        if rid:
            labelled[rid] = labelled.get(rid, 0) + 1

    wall_band = cv2.dilate(wall_mask, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (thickness | 1, thickness | 1)))
    ink_band = cv2.dilate(ink_clean, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * thickness | 1, 2 * thickness | 1)))
    er = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(1.5 * thickness) | 1, int(1.5 * thickness) | 1))
    fp_edge = cv2.subtract(cv2.dilate(footprint, er), cv2.erode(footprint, er))

    parent = list(range(n + 1))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    root_labels = {rid: labelled.get(rid, 0) for rid in range(1, n + 1)}

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # transitive-safe: if BOTH groups already contain a named room,
        # keep them apart (an unlabelled fragment can join either side)
        if root_labels.get(ra, 0) and root_labels.get(rb, 0):
            return
        parent[rb] = ra
        root_labels[ra] = root_labels.get(ra, 0) + root_labels.get(rb, 0)

    # Examine the SEPARATOR pixels between each adjacent pair and classify
    # what the drawing actually put there:
    #   thick wall ink        -> real wall, keep separate
    #   nothing (pure bridge) -> open-plan pass-through, merge (if wide)
    #   thin ink, interior    -> furniture line, merge
    #   thin ink, on footprint edge -> glazing (indoor/outdoor), keep separate
    reach = max(3, int(thickness * 0.9))
    kR = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    wall_n = cv2.dilate(wall_mask, k3)
    ink_n = cv2.dilate(ink_clean, k3)
    dils = {}
    for rid in range(1, n + 1):
        m = (labels == rid).astype(np.uint8) * 255
        dils[rid] = cv2.dilate(m, kR) if m.max() else None

    for a in range(1, n + 1):
        if dils[a] is None:
            continue
        for b in range(a + 1, n + 1):
            if dils[b] is None:
                continue
            sep = np.logical_and(dils[a] > 0, dils[b] > 0)
            n_sep = int(np.count_nonzero(sep))
            if n_sep < thickness * 2:
                continue                       # not adjacent
            contact_len = n_sep / (2.0 * reach)
            if contact_len < thickness * 3.0:
                continue                       # doorway-sized: keep separate
            wall_frac = np.count_nonzero(np.logical_and(sep, wall_n > 0)) / n_sep
            if wall_frac >= 0.38:
                continue                       # real wall between them
            ink_frac = np.count_nonzero(np.logical_and(sep, ink_n > 0)) / n_sep
            edge_frac = np.count_nonzero(np.logical_and(sep, fp_edge > 0)) / n_sep
            if ink_frac >= 0.30 and edge_frac > 0.25:
                continue                       # glazed indoor/outdoor divider
            union(a, b)

    remap = {}
    out = np.zeros_like(labels)
    for rid in range(1, n + 1):
        root = find(rid)
        if root not in remap:
            remap[root] = len(remap) + 1
        out[labels == rid] = remap[root]
    if len(remap) != n:
        log.info(f"  Merged {n} -> {len(remap)} rooms (unwalled boundaries)")
    return out


# ── 5. Polygonize rooms ───────────────────────────────────────────────────────

def _snap_rectilinear(pts: np.ndarray) -> np.ndarray:
    """Snap near-axis edges to exact H/V; leave diagonal/curved edges alone."""
    out = pts.astype(np.float64).copy()
    n = len(out)
    for _ in range(2):
        for i in range(n):
            a, b = out[i], out[(i + 1) % n]
            dx, dy = b[0] - a[0], b[1] - a[1]
            L = max(1e-6, np.hypot(dx, dy))
            if abs(dy) / L < 0.20:            # nearly horizontal
                m = (a[1] + b[1]) / 2
                a[1] = b[1] = m
            elif abs(dx) / L < 0.20:          # nearly vertical
                m = (a[0] + b[0]) / 2
                a[0] = b[0] = m
    # collapse consecutive duplicates
    keep = [0]
    for i in range(1, n):
        if np.hypot(*(out[i] - out[keep[-1]])) > 2:
            keep.append(i)
    return out[keep]


def polygonize_rooms(labels, n_rooms: int, thickness: int, h: int, w: int) -> list:
    polys = []
    for rid in range(1, n_rooms + 1):
        mask = (labels == rid).astype(np.uint8) * 255
        if mask.max() == 0:
            continue
        # fuse multi-part merged regions across their (thin) separators and
        # smooth ragged edges before tracing
        fuse = max(5, int(1.4 * thickness)) | 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                          (fuse, fuse)))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < h * w * 0.0015:
            continue
        eps = 0.008 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        snapped = _snap_rectilinear(approx)
        polys.append({
            "region_id": rid,
            "pts": snapped,                       # px coords
            "area_px": float(area),
            "centroid": snapped.mean(axis=0),
        })
    polys.sort(key=lambda p: -p["area_px"])
    return polys


# ── 6. Label assignment ───────────────────────────────────────────────────────

_SKIP_LABEL = re.compile(
    r"floor\s*plan|copyright|scale|north|drawn|sheet|revision|clg$|ceil",
    re.IGNORECASE)

_LEXICON = [
    "LIVING ROOM", "GREAT ROOM", "FAMILY ROOM", "KITCHEN", "DINING",
    "DINING ROOM", "BEDROOM", "BED RM", "MASTER BEDROOM", "MASTER BED RM",
    "BATH", "M. BATH", "BATHROOM", "POWDER", "PWDR", "W.I.C.", "WIC",
    "CLOSET", "LAUNDRY", "PANTRY", "HALL", "HALLWAY", "FOYER", "ENTRY",
    "GARAGE", "PORCH", "COVERED PORCH", "PATIO", "BALCONY", "DECK",
    "TERRACE", "STUDY", "OFFICE", "DEN", "ELEV", "ELEVATOR", "STAIRS",
    "MUD ROOM", "UTILITY", "STORAGE", "STORE", "MAJLIS", "MAID ROOM",
    "GUEST ROOM", "PRAYER ROOM", "NOOK", "LIBRARY", "GYM", "SAUNA",
]

def _lev(a: str, b: str) -> int:
    """Levenshtein distance, small strings only."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _lexicon_match(text: str):
    """Fuzzy-match a phrase (or its words) to the room lexicon.
    Returns canonical name or None."""
    clean = re.sub(r"[^A-Za-z. ]", " ", text).upper()
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        return None
    best, best_score = None, 10 ** 9

    def try_cand(cand: str):
        nonlocal best, best_score
        for entry in _LEXICON:
            d = _lev(cand, entry)
            allowed = max(1, int(len(entry) * 0.34))
            if d <= allowed and (d, -len(entry)) < (best_score, 0):
                # prefer smaller distance, then longer lexicon entries
                if d < best_score or (d == best_score and
                                      (best is None or len(entry) > len(best))):
                    best, best_score = entry, d

    try_cand(clean)
    words = clean.split()
    for i in range(len(words)):
        for j in range(i + 1, min(len(words), i + 3) + 1):
            try_cand(" ".join(words[i:j]))
    return best


def build_phrases(words: list) -> list:
    """Group OCR words into label phrases (split on big horizontal gaps)."""
    # group words by (block, par, line)
    groups: dict = {}
    for wd in words:
        groups.setdefault((wd["block"], wd["par"], wd["line"]), []).append(wd)

    phrases = []
    for key, ws in groups.items():
        ws.sort(key=lambda x: x["x"])
        # psm 6 puts words from opposite ends of the sheet on the same "line";
        # split on horizontal gaps larger than ~2.5x the text height
        runs, cur = [], [ws[0]]
        for wd in ws[1:]:
            gap = wd["x"] - (cur[-1]["x"] + cur[-1]["w"])
            if gap > max(30, 2.5 * max(wd["h"], cur[-1]["h"])):
                runs.append(cur)
                cur = [wd]
            else:
                cur.append(wd)
        runs.append(cur)

        for run in runs:
            kept = [x for x in run if not _is_dimension(x["text"])]
            if not kept:
                continue
            text = " ".join(x["text"] for x in kept)
            letters = re.sub(r"[^A-Za-z]", "", text)
            if len(letters) < 3:
                continue
            if _SKIP_LABEL.search(text):
                continue
            cx = int(np.mean([x["cx"] for x in kept]))
            cy = int(np.mean([x["cy"] for x in kept]))
            conf = int(np.mean([x["conf"] for x in kept]))
            phrases.append({"text": text, "cx": cx, "cy": cy, "conf": conf})
    return phrases


def assign_names(polys: list, phrases: list) -> None:

    for p in polys:
        p["names"] = []

    for ph in phrases:
        placed = False
        for p in polys:
            if cv2.pointPolygonTest(
                    p["pts"].astype(np.float32).reshape(-1, 1, 2),
                    (float(ph["cx"]), float(ph["cy"])), False) >= 0:
                p["names"].append(ph)
                placed = True
                break
        if not placed:
            # label sits on an erased boundary: snap to the nearest polygon
            best, bd = None, 1e18
            for p in polys:
                d = -cv2.pointPolygonTest(
                    p["pts"].astype(np.float32).reshape(-1, 1, 2),
                    (float(ph["cx"]), float(ph["cy"])), True)
                if d < bd:
                    bd, best = d, p
            if best is not None and bd < 60:
                best["names"].append(ph)

    for p in polys:
        if p["names"]:
            p["names"].sort(key=lambda x: x["cy"])
            matched, seen = [], set()
            for n in p["names"]:
                mx = _lexicon_match(n["text"])
                if mx and mx not in seen:
                    matched.append(mx)
                    seen.add(mx)
            if matched:
                # open-plan region with several labels -> "Kitchen & Dining"
                raw = " & ".join(matched[:3])
            else:
                # fall back to the highest-confidence phrase, letters only
                bestp = max(p["names"], key=lambda x: x["conf"])
                raw = re.sub(r"[^A-Za-z. ]", " ", bestp["text"])
                raw = re.sub(r"\s+", " ", raw).strip()
            p["name"] = " ".join(wd.capitalize() for wd in raw.split())[:48] or None
            p["conf"] = int(np.mean([n["conf"] for n in p["names"]]))
        else:
            p["name"] = None
            p["conf"] = 50


# ── 7. Wall vectorization (H/V centerlines) ───────────────────────────────────

def vectorize_walls(wall_mask, thickness: int, h: int, w: int) -> list:
    walls = []
    min_len = max(int(thickness * 1.5), 8)

    for orient in ("h", "v"):
        if orient == "h":
            kern = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
        else:
            kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))
        directional = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, kern)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(directional, 8)
        for i in range(1, n):
            x = stats[i, cv2.CC_STAT_LEFT]; y = stats[i, cv2.CC_STAT_TOP]
            ww = stats[i, cv2.CC_STAT_WIDTH]; hh = stats[i, cv2.CC_STAT_HEIGHT]
            if orient == "h":
                if ww < min_len:
                    continue
                walls.append({"x1": float(x), "y1": y + hh / 2.0,
                              "x2": float(x + ww), "y2": y + hh / 2.0,
                              "thickness": float(min(hh, thickness * 2))})
            else:
                if hh < min_len:
                    continue
                walls.append({"x1": x + ww / 2.0, "y1": float(y),
                              "x2": x + ww / 2.0, "y2": float(y + hh),
                              "thickness": float(min(ww, thickness * 2))})

    walls = _merge_collinear(walls, tol=thickness * 1.2)
    log.info(f"  Walls: {len(walls)} segments")
    return walls


def _merge_collinear(walls: list, tol: float) -> list:
    merged, used = [], set()
    for i, a in enumerate(walls):
        if i in used:
            continue
        ah = abs(a["x2"] - a["x1"]) >= abs(a["y2"] - a["y1"])
        grp = [a]; used.add(i)
        changed = True
        while changed:
            changed = False
            for j, b in enumerate(walls):
                if j in used:
                    continue
                bh = abs(b["x2"] - b["x1"]) >= abs(b["y2"] - b["y1"])
                if ah != bh:
                    continue
                if ah:
                    if abs(a["y1"] - b["y1"]) > tol:
                        continue
                    amin, amax = min(a["x1"], a["x2"]), max(a["x1"], a["x2"])
                    bmin, bmax = min(b["x1"], b["x2"]), max(b["x1"], b["x2"])
                else:
                    if abs(a["x1"] - b["x1"]) > tol:
                        continue
                    amin, amax = min(a["y1"], a["y2"]), max(a["y1"], a["y2"])
                    bmin, bmax = min(b["y1"], b["y2"]), max(b["y1"], b["y2"])
                gmin = max(amin, bmin); gmax = min(amax, bmax)
                if gmin - gmax > tol * 1.5:      # gap too large to merge
                    continue
                grp.append(b); used.add(j)
                # extend a to cover b
                if ah:
                    a["x1"], a["x2"] = min(amin, bmin), max(amax, bmax)
                else:
                    a["y1"], a["y2"] = min(amin, bmin), max(amax, bmax)
                changed = True
        # average the fixed axis + max thickness
        if ah:
            fy = float(np.mean([g["y1"] for g in grp]))
            a["y1"] = a["y2"] = fy
        else:
            fx = float(np.mean([g["x1"] for g in grp]))
            a["x1"] = a["x2"] = fx
        a["thickness"] = float(max(g["thickness"] for g in grp))
        merged.append(a)
    return merged


# ── 8. Openings (door / window) along each wall band ─────────────────────────

def _refine_axis(wall_mask, c: int, a: int, b: int, horiz: bool,
                 thickness: int, h: int, w: int) -> int:
    """Snap a (possibly averaged) centerline to the max-coverage row/col."""
    best_c, best_cov = c, -1.0
    for dc in range(-thickness, thickness + 1):
        cc = c + dc
        if horiz:
            if cc < 0 or cc >= h:
                continue
            cov = float(np.mean(wall_mask[cc, a:b] > 128))
        else:
            if cc < 0 or cc >= w:
                continue
            cov = float(np.mean(wall_mask[a:b, cc] > 128))
        if cov > best_cov:
            best_cov, best_c = cov, cc
    return best_c


def detect_openings(walls: list, wall_mask, ink_clean, thickness: int,
                    h: int, w: int) -> list:
    """
    Openings live in the GAPS between collinear wall segments.
    Classify each gap by what the drawing put there:
        thin ink lines across the gap -> WINDOW (glazing)
        (mostly) nothing              -> DOOR  (swing arcs bulge into the
                                        room, outside the thin sampling band)
    """
    openings = []
    band = max(2, int(thickness * 0.5))
    min_gap = int(thickness * 1.1)
    max_gap = int(thickness * 14)
    col_tol = thickness * 1.2

    for horiz in (True, False):
        segs = []
        for wi, s in enumerate(walls):
            is_h = abs(s["x2"] - s["x1"]) >= abs(s["y2"] - s["y1"])
            if is_h != horiz:
                continue
            if horiz:
                a, b = sorted((s["x1"], s["x2"]))
                c = s["y1"]
            else:
                a, b = sorted((s["y1"], s["y2"]))
                c = s["x1"]
            segs.append({"a": a, "b": b, "c": c, "wi": wi})

        # cluster by axis position
        segs.sort(key=lambda s: s["c"])
        clusters = []
        for s in segs:
            if clusters and abs(s["c"] - clusters[-1][-1]["c"]) <= col_tol:
                clusters[-1].append(s)
            else:
                clusters.append([s])

        for cl in clusters:
            cl.sort(key=lambda s: s["a"])
            for i in range(len(cl) - 1):
                cur, nxt = cl[i], cl[i + 1]
                gap = nxt["a"] - cur["b"]
                if gap < min_gap or gap > max_gap:
                    continue
                g0, g1 = int(cur["b"]), int(nxt["a"])
                c = int(round((cur["c"] + nxt["c"]) / 2))
                c = _refine_axis(wall_mask, c, max(0, g0 - 40),
                                 min((w if horiz else h), g1 + 40),
                                 horiz, thickness, h, w)
                # sample the gap strip
                if horiz:
                    lo, hi = max(0, c - band), min(h, c + band + 1)
                    strip_w = wall_mask[lo:hi, g0:g1]
                    strip_i = ink_clean[lo:hi, g0:g1]
                else:
                    lo, hi = max(0, c - band), min(w, c + band + 1)
                    strip_w = wall_mask[g0:g1, lo:hi]
                    strip_i = ink_clean[g0:g1, lo:hi]
                if strip_w.size == 0:
                    continue
                if float(np.mean(strip_w > 128)) > 0.45:
                    continue        # not actually open (vector artefact)
                ink_frac = float(np.mean(strip_i > 128))
                kind = "window" if ink_frac > 0.12 else "door"
                mid = (g0 + g1) // 2
                px, py = (mid, c) if horiz else (c, mid)
                openings.append({
                    "type": kind,
                    "wall_index": cur["wi"],
                    "wall": "horizontal" if horiz else "vertical",
                    "px": px, "py": py, "len_px": gap,
                })

    openings = _dedup_openings_px(openings, thickness)
    log.info(f"  Openings: {len(openings)} "
             f"({sum(1 for o in openings if o['type']=='door')} doors, "
             f"{sum(1 for o in openings if o['type']=='window')} windows)")
    return openings


def _dedup_openings_px(openings: list, thickness: int) -> list:
    unique = []
    for op in openings:
        if not any(np.hypot(op["px"] - u["px"], op["py"] - u["py"])
                   < thickness * 2 for u in unique):
            unique.append(op)
    return unique


# ── 9. Scale calibration from dimension text ──────────────────────────────────

def calibrate_scale(words: list, polys: list) -> float | None:
    """
    Find 'A x B' dimension pairs (feet-inches), match to the room polygon that
    contains them, derive metres-per-pixel. Median across matches.
    """
    FT = 0.3048; IN = 0.0254
    ests = []
    # collect dimension strings with positions
    for wd in words:
        m = re.findall(r"(\d{1,2})['’](?:\s*-?\s*(\d{1,2})[\"”])?", wd["text"])
        if not m:
            continue
        # find which polygon this sits in
        for p in polys:
            if cv2.pointPolygonTest(
                    p["pts"].astype(np.float32).reshape(-1, 1, 2),
                    (float(wd["cx"]), float(wd["cy"])), False) >= 0:
                xs = p["pts"][:, 0]; ys = p["pts"][:, 1]
                bw = float(xs.max() - xs.min()); bh = float(ys.max() - ys.min())
                for ft, inch in m:
                    metres = int(ft) * FT + (int(inch) * IN if inch else 0)
                    if metres < 1 or metres > 30:
                        continue
                    # match against the closer of width/height
                    for dim in (bw, bh):
                        if dim > 10:
                            ests.append(metres / dim)
                break
    if not ests:
        return None
    scale = float(np.median(ests))
    if not (0.001 < scale < 0.2):
        return None
    return scale




# ── 9b. Per-room crop OCR: recover names + parse dimension strings ───────────

_CROP_DIM = re.compile(
    r"(\d{1,2})\s*[\'`\u2019]?\s*(?:[-\u2013\u2014]\s*(\d{1,2})\s*[\"\u201d]?)?"
    r"\s*[xX\u00d7]\s*"
    r"(\d{1,2})\s*[\'`\u2019]?\s*(?:[-\u2013\u2014]\s*(\d{1,2})\s*[\"\u201d]?)?")

def recover_room_details(polys: list, grey_orig, work_scale: float,
                         h: int, w: int) -> float | None:
    """For every room, OCR a padded high-res crop of the ORIGINAL image:
       - unnamed rooms get a second naming chance (lexicon-matched)
       - dimension strings like 14'-7" x 16' become length/width in metres,
         but ONLY if they pass validation (plausible size + aspect ratio
         matching the room's pixel aspect) — a wrong number is worse than none
       Returns the median metres-per-WORKING-pixel scale, or None."""
    try:
        import pytesseract
    except ImportError:
        return None
    FT, IN = 0.3048, 0.0254
    oh, ow = grey_orig.shape[:2]
    scale_votes: list[float] = []

    for p in polys:
        xs = p["pts"][:, 0] / work_scale
        ys = p["pts"][:, 1] / work_scale
        px_w = float(xs.max() - xs.min())
        px_h = float(ys.max() - ys.min())
        padx, pady = int(px_w * 0.35), int(px_h * 0.35)
        x0 = max(0, int(xs.min()) - padx); x1 = min(ow, int(xs.max()) + padx)
        y0 = max(0, int(ys.min()) - pady); y1 = min(oh, int(ys.max()) + pady)
        crop = grey_orig[y0:y1, x0:x1]
        if crop.size == 0 or min(crop.shape) < 8:
            continue
        s = max(3.0, 750.0 / max(crop.shape))
        up = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        try:
            data = pytesseract.image_to_data(
                up, config="--psm 6 --oem 3",
                output_type=pytesseract.Output.DICT)
        except Exception:
            continue
        toks = [t.strip() for t, cf in zip(data["text"], data["conf"])
                if t.strip() and int(cf) >= 25]
        if not toks:
            continue
        joined = " ".join(toks).replace("\u2019", "'")

        # second-chance naming for "Room N" placeholders
        if not p.get("name"):
            for span in range(min(3, len(toks)), 0, -1):
                hit = None
                for i in range(len(toks) - span + 1):
                    cand = " ".join(toks[i:i + span])
                    if len(re.sub(r"[^A-Za-z]", "", cand)) < 3:
                        continue
                    hit = _lexicon_match(cand)
                    if hit:
                        break
                if hit:
                    p["name"] = " ".join(wd.capitalize() for wd in hit.split())
                    p["conf"] = max(p.get("conf", 50), 55)
                    break

        # validated dimension parse
        m = _CROP_DIM.search(joined)
        if not m:
            continue
        a = int(m.group(1)) * FT + (int(m.group(2)) * IN if m.group(2) else 0)
        b = int(m.group(3)) * FT + (int(m.group(4)) * IN if m.group(4) else 0)
        if not (1.0 < a < 25 and 1.0 < b < 25):
            continue
        ar_txt = max(a, b) / max(0.1, min(a, b))
        ar_px = max(px_w, px_h) / max(1.0, min(px_w, px_h))
        if abs(ar_txt - ar_px) / ar_px > 0.35:
            continue                    # digits misread — reject
        # width follows the horizontally-larger extent
        if px_w >= px_h:
            p["dim_w"], p["dim_l"] = max(a, b), min(a, b)
        else:
            p["dim_w"], p["dim_l"] = min(a, b), max(a, b)
        scale_votes.append(max(a, b) / (max(px_w, px_h) * work_scale))
        scale_votes.append(min(a, b) / (min(px_w, px_h) * work_scale))

    if len(scale_votes) < 2:
        return None
    med = float(np.median(scale_votes))
    good = [v for v in scale_votes if abs(v - med) / med < 0.30]
    if len(good) < 2:
        return None
    return float(np.median(good))


# ── 9c. Outdoor areas (patio / porch): thin-line-enclosed regions ─────────────

_OUTDOOR_NAMES = {"PATIO", "PORCH", "COVERED PORCH", "BALCONY", "DECK",
                  "TERRACE", "GARDEN"}

def detect_outdoor_areas(ink_clean, seal, footprint, labels, phrases,
                         polys, thickness: int, h: int, w: int) -> list:
    """Patio/porch/covered porch are open-sided, so enclosure tests fail.
    Instead: flood-fill from each outdoor-named OCR label within the thin-ink
    barrier, clipped to a window around the label. Slight overreach is fine —
    these render as floor patches, not walled rooms."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (int(thickness * 1.2) | 1,) * 2)
    barrier = cv2.bitwise_or(cv2.morphologyEx(ink_clean, cv2.MORPH_CLOSE, k),
                             seal)
    # NOTE: footprint is deliberately NOT a barrier — attached porches sit
    # inside it. Labeled rooms + walls already stop floods entering the house.
    barrier[labels > 0] = 255
    free = (barrier == 0).astype(np.uint8) * 255

    out = []
    # Pass 1: collect usable outdoor label seeds
    seeds = []
    for ph in phrases:
        name = _lexicon_match(ph["text"])
        if not name or name not in _OUTDOOR_NAMES:
            continue
        cx, cy = int(ph["cx"]), int(ph["cy"])
        if any(cv2.pointPolygonTest(
                p["pts"].astype(np.float32).reshape(-1, 1, 2),
                (float(cx), float(cy)), False) >= 0 for p in polys):
            continue
        seed = None
        for r in range(0, 25, 3):
            for dx, dy in ((0, 0), (r, 0), (-r, 0), (0, r), (0, -r)):
                x, y = cx + dx, cy + dy
                if 0 <= x < w and 0 <= y < h and free[y, x] > 0:
                    seed = (x, y)
                    break
            if seed:
                break
        if seed:
            seeds.append({"name": name, "cx": cx, "cy": cy, "seed": seed,
                          "conf": ph["conf"]})

    if not seeds:
        log.info("  Outdoor areas: 0 (none)")
        return out

    # Pass 2: flood the outdoor space once per seed, then split shared
    # ground by NEAREST label (several labels usually share one connected
    # outdoor area — patio, porch, covered porch wrap around the house)
    flooded = np.zeros((h, w), np.uint8)
    for s in seeds:
        ff = free.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ff, mask, s["seed"], 128)
        flooded |= (ff == 128).astype(np.uint8)

    ys_f, xs_f = np.nonzero(flooded)
    if ys_f.size == 0:
        log.info("  Outdoor areas: 0 (none)")
        return out
    d2 = np.stack([(xs_f - s["cx"]) ** 2 + (ys_f - s["cy"]) ** 2
                   for s in seeds])           # (n_seeds, n_pixels)
    owner = np.argmin(d2, axis=0)

    for si, s in enumerate(seeds):
        region = np.zeros((h, w), np.uint8)
        sel = owner == si
        region[ys_f[sel], xs_f[sel]] = 255
        # clip to a window around the label to stop border creep
        win = np.zeros((h, w), np.uint8)
        hw, hh = int(w * 0.34), int(h * 0.34)
        win[max(0, s["cy"] - hh):min(h, s["cy"] + hh),
            max(0, s["cx"] - hw):min(w, s["cx"] + hw)] = 255
        region = cv2.bitwise_and(region, win)
        region = cv2.morphologyEx(region, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(
                                      cv2.MORPH_ELLIPSE, (11, 11)))
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(region, 8)
        if n <= 1:
            continue
        sx, sy = s["seed"]
        sid = lbl[sy, sx]
        if sid == 0:
            sid = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        comp = (lbl == sid).astype(np.uint8) * 255
        if int(np.count_nonzero(comp)) < h * w * 0.006:
            continue
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        eps = 0.012 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        pts = _snap_rectilinear(approx)
        pretty = " ".join(wd.capitalize() for wd in s["name"].split())
        out.append({"pts": pts,
                    "area_px": float(cv2.contourArea(cnt)),
                    "centroid": pts.mean(axis=0),
                    "names": [], "name": pretty,
                    "conf": min(90, s["conf"]),
                    "outdoor": True})

    log.info(f"  Outdoor areas: {len(out)} "
             f"({', '.join(o['name'] for o in out) or 'none'})")
    return out




# ── 9d. Gemini dimension fallback (tiny/blurry dimension text) ────────────────
# OCR engines fail below ~8px glyph height; vision LLMs read such text via
# context priors. Used ONLY when Tesseract found no scale, and every value it
# returns must still pass the same aspect-ratio validation — a hallucinated
# digit fails the geometry check and is discarded.

async def gemini_dimensions(image_bytes: bytes, polys: list,
                            api_key: str) -> float | None:
    if not api_key:
        return None
    try:
        import httpx, base64, json as _json
    except ImportError:
        return None

    room_names = [p["name"] for p in polys if p.get("name")]
    prompt = (
        "This is an architectural floor plan. Under each room label there is "
        "usually a printed size like 14'-7\" X 16' (feet and inches). "
        "Read them carefully. Return ONLY a JSON array, one item per room "
        "you can read a size for: "
        '[{"name":"<room label as printed>","feet_a":14,"inches_a":7,'
        '"feet_b":16,"inches_b":0}]. '
        "Use 0 for missing inches. Skip rooms whose size you cannot read. "
        f"Known room labels: {', '.join(room_names[:20])}."
    )
    b64 = base64.b64encode(image_bytes).decode()
    body = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/png", "data": b64}},
        ]}],
        "generationConfig": {"responseMimeType": "application/json",
                             "temperature": 0},
    }
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           "gemini-2.5-flash:generateContent")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, json=body,
                                  headers={"x-goog-api-key": api_key})
            if r.status_code != 200:
                log.warning(f"  Gemini dims: HTTP {r.status_code}")
                return None
            raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            items = _json.loads(raw[raw.index("["):raw.rindex("]") + 1])
    except Exception as e:
        log.warning(f"  Gemini dims failed: {e}")
        return None

    FT, IN = 0.3048, 0.0254
    scale_votes: list[float] = []
    accepted = 0
    for it in items:
        try:
            a = float(it["feet_a"]) * FT + float(it.get("inches_a") or 0) * IN
            b = float(it["feet_b"]) * FT + float(it.get("inches_b") or 0) * IN
            gname = str(it["name"]).upper()
        except Exception:
            continue
        if not (1.0 < a < 30 and 1.0 < b < 30):
            continue
        # fuzzy-match to a detected room
        best_p, best_d = None, 10 ** 9
        for p in polys:
            if not p.get("name"):
                continue
            d = _lev(gname[:24], p["name"].upper()[:24])
            if d < best_d:
                best_d, best_p = d, p
        if best_p is None or best_d > max(3, len(gname) // 3):
            continue
        xs = best_p["pts"][:, 0]; ys = best_p["pts"][:, 1]
        px_w = float(xs.max() - xs.min()); px_h = float(ys.max() - ys.min())
        ar_txt = max(a, b) / max(0.1, min(a, b))
        ar_px = max(px_w, px_h) / max(1.0, min(px_w, px_h))
        if abs(ar_txt - ar_px) / ar_px > 0.40:
            continue                          # geometry disagrees: reject
        if px_w >= px_h:
            best_p["dim_w"], best_p["dim_l"] = max(a, b), min(a, b)
        else:
            best_p["dim_w"], best_p["dim_l"] = min(a, b), max(a, b)
        accepted += 1
        scale_votes.append(max(a, b) / max(px_w, px_h))
        scale_votes.append(min(a, b) / min(px_w, px_h))

    log.info(f"  Gemini dims: {len(items)} read, {accepted} passed validation")
    if len(scale_votes) < 2:
        return None
    med = float(np.median(scale_votes))
    good = [v for v in scale_votes if abs(v - med) / med < 0.30]
    return float(np.median(good)) if len(good) >= 2 else None


# ── Main ──────────────────────────────────────────────────────────────────────

async def analyse_floor_plan(image_bytes: bytes, image_url: str = "",
                             project_id: str = "p",
                             gemini_api_key: str = "", **_) -> dict:
    bgr, grey, grey_orig, ink, h, w, ws = preprocess(image_bytes)
    log.info(f"Image: {w}x{h}px (work scale {ws:.2f})  project={project_id}")

    words = ocr_words(grey_orig, ws, h, w)
    ink_clean = erase_text(ink, words, h, w)

    thickness = estimate_wall_thickness(ink)
    log.info(f"  Wall thickness ~ {thickness}px")
    # Wall mask from RAW ink: text glyph strokes are far thinner than walls,
    # so the thickness opening removes them anyway — erasing OCR boxes first
    # only punches holes in walls where labels touch them.
    wmask = wall_mask_from_ink(ink, thickness)
    barrier = cv2.bitwise_or(wmask, long_thin_lines(ink_clean, wmask, thickness))

    walls_px = vectorize_walls(wmask, thickness, h, w)
    diag = diagonal_walls(barrier, thickness, h, w)
    seal = build_seal(barrier, walls_px, thickness, h, w, diag)

    phrases = build_phrases(words)
    label_points = [(ph["cx"], ph["cy"]) for ph in phrases
                    if _lexicon_match(ph["text"])]

    labels, n_rooms, footprint = segment_rooms(seal, wmask, ink_clean,
                                                label_points, thickness, h, w)
    polys = polygonize_rooms(labels, n_rooms, thickness, h, w)

    assign_names(polys, phrases)

    # outdoor areas (patio / porch): label-seeded, added AFTER naming so
    # their pre-set names aren't clobbered
    outdoor = detect_outdoor_areas(ink_clean, seal, footprint, labels,
                                   phrases, polys, thickness, h, w)
    polys.extend(outdoor)
    openings_px = detect_openings(walls_px, wmask, ink_clean, thickness, h, w)
    # names + validated dimensions from high-res per-room crops
    scale_m = recover_room_details(polys, grey_orig, ws, h, w)
    if not scale_m:
        dim_words = ocr_dimensions(grey_orig, ws)
        scale_m = calibrate_scale(words + dim_words, polys)
    if not scale_m:
        # OCR floor reached (tiny/blurry print) -> one Gemini read, validated
        scale_m = await gemini_dimensions(image_bytes, polys, gemini_api_key)
    if scale_m:
        log.info(f"  Scale: {scale_m*1000:.2f} mm/px")

    # ── serialize (rooms/walls in 0-100 %, openings keep 0-1000 for compat) ──
    rooms_out = []
    unnamed = 0
    for i, p in enumerate(polys):
        if p["name"]:
            name = p["name"]
        else:
            unnamed += 1
            name = f"Room {unnamed}"
        xs = p["pts"][:, 0]; ys = p["pts"][:, 1]
        room = {
            "id": f"{project_id}-r{i+1}",
            "name": name,
            "confidence": min(95, p["conf"]),
            "color": room_color(name, i),
            "polygon": [[round(float(x) / w * 100, 3),
                         round(float(y) / h * 100, 3)] for x, y in p["pts"]],
            "box": {
                "top":    round(float(ys.min()) / h * 100, 3),
                "left":   round(float(xs.min()) / w * 100, 3),
                "width":  round(float(xs.max() - xs.min()) / w * 100, 3),
                "height": round(float(ys.max() - ys.min()) / h * 100, 3),
            },
        }
        # pixel extents always included: lets the app derive ALL room
        # measurements from a single user-entered dimension when the plan's
        # text is too low-res for auto-scale
        room["px_size"] = {"w": round(float(xs.max() - xs.min()), 1),
                           "h": round(float(ys.max() - ys.min()), 1)}
        # dimensions read directly off the plan beat scale-derived ones
        if p.get("dim_l") and p.get("dim_w"):
            room["length"] = p["dim_l"]
            room["width"] = p["dim_w"]
        elif scale_m:
            room["length"] = round((ys.max() - ys.min()) * scale_m, 2)
            room["width"] = round((xs.max() - xs.min()) * scale_m, 2)
        if scale_m:
            # true polygon area (shoelace), not bbox area
            px_area = 0.5 * abs(float(
                np.dot(p["pts"][:, 0], np.roll(p["pts"][:, 1], 1)) -
                np.dot(p["pts"][:, 1], np.roll(p["pts"][:, 0], 1))))
            room["area_m2"] = round(px_area * scale_m * scale_m, 1)
        rooms_out.append(room)

    walls_out = [{
        "id": f"w{i}",
        "x1": round(s["x1"] / w * 100, 3), "y1": round(s["y1"] / h * 100, 3),
        "x2": round(s["x2"] / w * 100, 3), "y2": round(s["y2"] / h * 100, 3),
        "thickness": round(s["thickness"] / max(w, h) * 100, 3),
    } for i, s in enumerate(walls_px)]

    openings_out = [{
        "type": o["type"],
        "wall_id": f"w{o['wall_index']}",
        "wall": o["wall"],
        "x": round(o["px"] / w * 1000, 2),
        "y": round(o["py"] / h * 1000, 2),
        "width": round(o["len_px"] / max(w, h) * 1000, 2),
    } for o in openings_px]

    result = {
        "rooms": rooms_out,
        "walls": walls_out,
        "openings": openings_out,
        "image_size": {"width": w, "height": h},
    }
    if scale_m:
        result["scale_m_per_px"] = round(scale_m, 6)
    return result