import functools
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


def thinnest_wall(ink_clean, work_scale: float, h: int, w: int) -> int | None:
    """Width of the THINNEST walls, in px.

    estimate_wall_thickness() returns the thickest strokes. When exterior
    walls are drawn much heavier than interior ones (a grey-filled shell,
    a fireplace), a wall mask cut at half of that width erases the interior
    walls entirely and whole rooms merge into one. Stroke widths form
    clusters — thin lines (furniture, dimensions, glazing) and walls; this
    finds where the wall cluster starts. Text is assumed already erased."""
    dt = cv2.distanceTransform(ink_clean, cv2.DIST_L2, 5)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-3) & (dt > 0.9)
    if not ridge.any():
        return None
    hist = np.bincount(np.round(2 * dt[ridge]).astype(int))
    # anything this thin is a drawn line, not a wall
    floor = max(int(round(3 * work_scale)), int(round(max(h, w) * 0.0045)))
    if floor >= len(hist):
        return None
    top = int(hist[floor:].max())
    if top <= 0:
        return None
    for width in range(floor, len(hist)):
        if hist[width] >= 0.45 * top:
            return width
    return None


def wall_mask_from_ink(ink_clean, thickness: int, thin_wall: int | None = None):
    """Opening with a kernel just under wall thickness keeps walls,
    erases thin furniture / dimension / symbol strokes. The kernel is sized
    to the thinnest walls (thinnest_wall), not the thickest."""
    k = max(2, int(thickness * 0.55))
    if thin_wall:
        k = max(2, min(k, int(thin_wall * 0.75)))
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
    max_span = thickness * 22
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
        # OpenCV 4 returns (N, 1, 4), OpenCV 5 (N, 4)
        x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4]
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        L = np.hypot(dx, dy)
        if L < thickness * 2.5:
            continue
        ang = np.degrees(np.arctan2(dy, max(1, dx)))
        if 20 < ang < 70:           # genuinely diagonal only
            out.append((x1, y1, x2, y2))
    return out


def _grow_footprint(seal, thickness: int, mults: tuple, min_r: int, h: int, w: int,
                    scale: float = 0.5):
    """Dilate until the shell closes, fill the holes (the rooms), erode back;
    escalate the radius until the area stops growing appreciably.

    Run at half resolution: dilations with radii of 9 wall thicknesses were
    the slowest step of the whole analysis, and the footprint only needs to
    be right to a pixel or two."""
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    small = (cv2.resize(seal, (sw, sh), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8) * 255
    fp, prev = None, 0
    for mult in mults:
        r = max(2, int(max(min_r, thickness * mult) * scale))
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        fat = cv2.dilate(small, kern)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(fat, 8)
        if n > 1:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            fat = np.where(lbl == biggest, 255, 0).astype(np.uint8)
        ff = fat.copy()
        cv2.floodFill(ff, np.zeros((sh + 2, sw + 2), np.uint8), (0, 0), 255)
        cand = cv2.erode(cv2.bitwise_or(fat, cv2.bitwise_not(ff)), kern)
        area = int(np.count_nonzero(cand))
        if fp is None or area > prev * 1.10:
            fp, prev = cand, area
        else:
            break
    return cv2.resize(fp, (w, h), interpolation=cv2.INTER_NEAREST)


def wall_footprint(wall_seal, thickness: int, h: int, w: int):
    """The building as its WALLS draw it: walls (with door and window gaps
    bridged) dilated until the shell closes, holes filled, eroded back.

    The room footprint below is grown from every long line on the sheet,
    dimension lines included, so it also takes in the strips between the
    dimension lines and the house. This one stops at the outer wall face."""
    return _grow_footprint(wall_seal, thickness, (2, 4, 6, 9), 3, h, w)


def _clip_to_walls(labels, n: int, wall_fp, label_points: list, h: int, w: int):
    """Rooms end at the outer walls.
      mostly inside the walls  -> clipped to them (a bedroom no longer runs
                                   out through its window to the dimension line)
      mostly outside, unnamed   -> not a room (margin strips between the
                                   dimension lines and the house)
      mostly outside, named     -> kept: a carport, porch or patio someone labelled
    """
    inside = wall_fp > 0
    named = set()
    for (lx, ly) in label_points:
        x0, x1 = max(0, int(lx) - 12), min(w, int(lx) + 13)
        y0, y1 = max(0, int(ly) - 12), min(h, int(ly) + 13)
        for rid in np.unique(labels[y0:y1, x0:x1]):
            if rid:
                named.add(int(rid))
    dropped = 0
    for rid in range(1, n + 1):
        m = labels == rid
        area = int(np.count_nonzero(m))
        if not area:
            continue
        frac = np.count_nonzero(m & inside) / area
        if frac >= 0.5:
            labels[m & ~inside] = 0
        elif rid not in named:
            labels[m] = 0
            dropped += 1
    if dropped:
        log.info(f"  Dropped {dropped} unnamed region(s) outside the walls")
    return labels


def fill_label_boxes(labels, words: list, wall_mask, h: int, w: int):
    """Text strokes are ink, so a room's floor stops at its own label and
    the floor outline gets a bite (or a hole) where the name is printed.
    Give the unclaimed pixels of each recognised label box to the room that
    surrounds it — never across a wall."""
    out = labels.copy()
    for wd in words:
        t = wd["text"]
        if _word_match(t) is None and not re.search(r"\d", t):
            continue
        x0, y0 = max(0, wd["x"] - 2), max(0, wd["y"] - 2)
        x1, y1 = min(w, wd["x"] + wd["w"] + 2), min(h, wd["y"] + wd["h"] + 2)
        if x1 <= x0 or y1 <= y0:
            continue
        pad = max(3, wd["h"] // 2)
        ring = labels[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
        vals, counts = np.unique(ring[ring > 0], return_counts=True)
        if vals.size == 0:
            continue
        rid = int(vals[np.argmax(counts)])
        if counts.max() < 0.7 * counts.sum():
            continue                     # the label sits between two rooms
        box = out[y0:y1, x0:x1]
        free = (box == 0) & (wall_mask[y0:y1, x0:x1] == 0)
        box[free] = rid
    return out


def segment_rooms(seal, wall_mask, ink_clean, label_points: list,
                  thickness: int, h: int, w: int, wall_seal=None):
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
    footprint = _grow_footprint(seal, thickness, (3, 5, 7, 9), 4, h, w)

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
        # (returned two values here; the caller unpacks three and crashed
        # on any plan where no room was found)
        return np.zeros((h, w), np.int32), 0, footprint

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
    if wall_seal is not None:
        labels = _clip_to_walls(labels, n_rooms,
                                wall_footprint(wall_seal, thickness, h, w),
                                label_points, h, w)
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

    # Furniture drawn as a closed outline (a kitchen island, a rug, a bed)
    # encloses a little "room" of its own, and a label printed on it then
    # names that island instead of the room. A region entirely surrounded by
    # ONE other region, with no wall on its boundary, is part of that room —
    # whatever labels either carries.
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (int(1.5 * thickness) | 1,) * 2)
    for a in range(1, n + 1):
        m = labels == a
        if not m.any():
            continue
        ring = cv2.dilate(m.astype(np.uint8), ring_k).astype(bool) & ~m
        neigh = labels[ring]
        neigh = neigh[neigh > 0]
        if neigh.size == 0:
            continue
        # count neighbours by the room they already merged into: the room
        # around an island may still be two pieces at this point
        roots = np.array([find(int(v)) for v in neigh])
        vals, counts = np.unique(roots, return_counts=True)
        b = int(vals[np.argmax(counts)])
        if counts.max() < 0.8 * neigh.size or b == find(a):
            continue
        # most of the ring must be that room, not outside
        if np.count_nonzero(labels[ring] == 0) > 0.6 * ring.sum():
            continue
        if np.count_nonzero(wall_n[ring] > 0) / max(1, ring.sum()) >= 0.3:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            root_labels[rb] = root_labels.get(rb, 0) + root_labels.get(ra, 0)

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
    "LIVING ROOM", "LIVING", "GREAT ROOM", "FAMILY ROOM", "FAMILY", "KITCHEN", "DINING",
    "DINING ROOM", "DINING AREA", "BEDROOM", "BED RM", "MASTER BEDROOM",
    "MASTER BED RM", "MASTER BATH",
    "BATH", "M. BATH", "BATHROOM", "POWDER", "PWDR", "W.I.C.", "WIC",
    "CLOSET", "LAUNDRY", "PANTRY", "HALL", "HALLWAY", "FOYER", "ENTRY",
    "GARAGE", "CARPORT", "PORCH", "COVERED PORCH", "PATIO", "COVERED PATIO",
    "BALCONY", "DECK", "TERRACE", "STUDY", "OFFICE", "DEN", "ELEV",
    "ELEVATOR", "STAIRS", "MUD", "MUD ROOM", "UTILITY", "STORAGE", "STORE",
    "WASHROOM", "TOILET", "W.C.", "LINEN", "DRESSING", "LOBBY", "LOUNGE",
    "RECEPTION", "SITTING", "SITTING ROOM",
    # Gulf-region plans
    "MAJLIS", "MEN MAJLIS", "LADIES MAJLIS", "DIWANIYA", "MAID ROOM",
    "MAID", "DRIVER ROOM", "GUEST ROOM", "PRAYER ROOM",
    "NOOK", "LIBRARY", "GYM", "SAUNA",
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


def _allowed_edits(entry: str) -> int:
    """Edits a misread may have: 1 for short words (6 letters or fewer:
    "MALS" is not MAJLIS), about 3 in 10 for longer ones ("ITCHE" is
    KITCHEN)."""
    n = len(entry.replace(" ", ""))
    return 1 if n <= 6 else max(1, int(n * 0.3))


@functools.lru_cache(maxsize=4096)
def _lexicon_exact(text: str):
    """The lexicon entry the WHOLE text reads as (fuzzy), or None — no
    sub-phrases, unlike _lexicon_match."""
    clean = re.sub(r"[^A-Za-z. ]", " ", text).upper()
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        return None
    best, best_d = None, 10 ** 9
    for entry in _LEXICON:
        d = _lev(clean, entry)
        if d <= _allowed_edits(entry) and (d < best_d or (d == best_d and len(entry) > len(best))):
            best, best_d = entry, d
    return best


@functools.lru_cache(maxsize=4096)
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
            allowed = _allowed_edits(entry)
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


# ── 6b. Room names: read labels the way people print them ───────────────────
#
# One full-page Tesseract pass (psm 6) reads a plan as paragraphs, and on
# plans with thin or light lettering it returns almost nothing — or garbage
# ("Ine", "Oreo", "Toot") that was then used as the room's name because no
# real label was found. Room labels are short, scattered words, so:
#   - two extra passes in SPARSE-TEXT mode (psm 11), on the plan and on the
#     plan with the walls whitened out (letters touching walls read better);
#   - a word only becomes a name through the room lexicon (fuzzy, so
#     "ITCHE" is KITCHEN) — never as raw OCR text;
#   - each label lands in the room that contains it, word by word, so a
#     line of labels across the sheet cannot name the wrong room.

_DISPLAY = {
    "M. BATH": "Master Bath", "M BATH": "Master Bath", "MASTER BATH": "Master Bath",
    "PWDR": "Powder Room", "POWDER": "Powder Room", "PWDR.": "Powder Room",
    "W.I.C.": "Walk-in Closet", "WIC": "Walk-in Closet",
    "BED RM": "Bedroom", "MASTER BED RM": "Master Bedroom",
    "ELEV": "Elevator", "DINING AREA": "Dining", "DINNING AREA": "Dining",
    # first word alone (the rest was misread or hidden by a furniture line)
    "FAMILY": "Family Room",
    "WC": "WC", "W.C.": "WC",
}


def display_name(canon: str) -> str:
    if canon in _DISPLAY:
        return _DISPLAY[canon]
    return " ".join(wd.capitalize() for wd in canon.split())


def ocr_label_words(grey_orig, work_scale: float, h: int, w: int) -> list:
    """Sparse-text OCR passes for room labels. Same word format as ocr_words
    (working-image coordinates), tagged with the pass that found them."""
    try:
        import pytesseract
    except ImportError:
        return []
    oh, ow = grey_orig.shape[:2]
    sc = max(3.0, 2200.0 / max(oh, ow))
    up = cv2.resize(grey_orig, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    # whiten the walls: heavy strokes beside a label merge into its letters
    _, bw = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = max(3, int(sc * 3))
    heavy = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    nowall = up.copy()
    nowall[cv2.dilate(heavy, np.ones((3, 3), np.uint8)) > 0] = 255

    to_work = work_scale / sc
    out = []

    def _run(item):
        tag, img = item
        try:
            return tag, pytesseract.image_to_data(img, config="--psm 11 --oem 3",
                                                  output_type=pytesseract.Output.DICT)
        except Exception as e:
            log.warning(f"  label OCR ({tag}) failed: {e}")
            return tag, None
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(_run, (("sparse", up), ("sparse-nowall", nowall))))
    for tag, d in results:
        if d is None:
            continue
        for i in range(len(d["text"])):
            text = d["text"][i].strip()
            conf = int(float(d["conf"][i]))
            if not text or conf < 30:
                continue
            x = int(d["left"][i] * to_work); y = int(d["top"][i] * to_work)
            bw_, bh_ = max(1, int(d["width"][i] * to_work)), max(1, int(d["height"][i] * to_work))
            if bw_ * bh_ > 0.015 * h * w:
                continue
            out.append({"text": text, "conf": conf, "x": x, "y": y, "w": bw_, "h": bh_,
                        "cx": x + bw_ // 2, "cy": y + bh_ // 2, "pass": tag})
    return out


def _word_match(text: str):
    """Lexicon entry for ONE word, or None. Short or letter-poor tokens never
    match (OCR noise is mostly 1-3 characters)."""
    letters = re.sub(r"[^A-Za-z]", "", text)
    if len(letters) < 3:
        return None
    m = _lexicon_match(text)
    # a 3-letter label must be read exactly: "WIZ" is not WIC
    if m and len(re.sub(r"[^A-Z]", "", m)) <= 3 and letters.upper() != re.sub(r"[^A-Z]", "", m):
        return None
    return m


def label_phrases(words: list) -> list:
    """Room labels from OCR words of several passes: duplicates of one
    printed word are merged, neighbours on a line are joined, and each run is
    matched to the lexicon (longest match first). A room number printed after
    the label ("BEDROOM #2") is kept: "Bedroom 2"."""
    # 1. one entry per printed word: overlapping boxes from different passes
    #    keep the reading that matches the lexicon, then the more confident
    ws_ = sorted(words, key=lambda wd: (_word_match(wd["text"]) is None, -wd["conf"]))
    kept = []
    for wd in ws_:
        clash = False
        for k in kept:
            ix = min(wd["x"] + wd["w"], k["x"] + k["w"]) - max(wd["x"], k["x"])
            iy = min(wd["y"] + wd["h"], k["y"] + k["h"]) - max(wd["y"], k["y"])
            if ix > 0 and iy > 0 and ix * iy > 0.3 * min(wd["w"] * wd["h"], k["w"] * k["h"]):
                clash = True
                break
        if not clash:
            kept.append(wd)

    # 2. lines: words at the same height, close together
    kept.sort(key=lambda wd: (wd["cy"], wd["x"]))
    lines = []
    for wd in kept:
        for ln in lines:
            hgt = max(wd["h"], max(o["h"] for o in ln))
            cy = np.mean([o["cy"] for o in ln])
            x0 = min(o["x"] for o in ln); x1 = max(o["x"] + o["w"] for o in ln)
            # beside the line on either end (word order in the sort is by
            # height, not by x)
            gap = max(wd["x"] - x1, x0 - (wd["x"] + wd["w"]))
            if abs(wd["cy"] - cy) <= 0.6 * hgt and -0.2 * hgt <= gap <= 1.6 * hgt:
                ln.append(wd)
                break
        else:
            lines.append([wd])
    # merging can leave lines unsorted by x
    for ln in lines:
        ln.sort(key=lambda wd: wd["x"])

    # 3a. two-line labels: MASTER / BEDROOM printed one above the other
    phrases = []
    used = set()
    def _box(ln):
        return (min(o["x"] for o in ln), min(o["y"] for o in ln),
                max(o["x"] + o["w"] for o in ln), max(o["y"] + o["h"] for o in ln))
    for i, top in enumerate(lines):
        for j, bot in enumerate(lines):
            if i == j or i in used or j in used:
                continue
            ax0, ay0, ax1, ay1 = _box(top)
            bx0, by0, bx1, by1 = _box(bot)
            hgt = max(o["h"] for o in top + bot)
            if not (0 <= by0 - ay1 <= 1.0 * hgt):
                continue
            if abs((ax0 + ax1) / 2 - (bx0 + bx1) / 2) > 0.5 * max(ax1 - ax0, bx1 - bx0):
                continue
            t1 = " ".join(o["text"] for o in top)
            t2 = " ".join(o["text"] for o in bot)
            whole = _lexicon_exact(f"{t1} {t2}")
            if (whole and len(whole.split()) >= 2
                    and whole != _lexicon_match(t1) and whole != _lexicon_match(t2)):
                seg = top + bot
                phrases.append({
                    "text": whole, "name": display_name(whole),
                    "cx": int(np.mean([o["cx"] for o in seg])),
                    "cy": int(np.mean([o["cy"] for o in seg])),
                    "h": float(np.mean([o["h"] for o in seg])),
                    "conf": int(np.mean([o["conf"] for o in seg])),
                })
                used.update((i, j))
    lines = [ln for k, ln in enumerate(lines) if k not in used]

    # 3b. longest lexicon match first, along each line
    for ln in lines:
        i = 0
        while i < len(ln):
            hit = None
            for span in (3, 2, 1):
                if i + span > len(ln):
                    continue
                seg = ln[i:i + span]
                text = " ".join(wd["text"] for wd in seg)
                if len(re.sub(r"[^A-Za-z]", "", text)) < 3:
                    continue
                # a multi-word span must read, as a whole, as a multi-word
                # entry: "I DINING" is DINING plus noise, and taking both
                # words moved the label's position onto the next room
                m = _lexicon_exact(text) if span > 1 else _word_match(text)
                if m and (span == 1 or len(m.split()) >= 2):
                    hit = (span, m)
                    break
            if not hit:
                i += 1
                continue
            span, canon = hit
            seg = ln[i:i + span]
            name = display_name(canon)
            nxt = ln[i + span] if i + span < len(ln) else None
            if nxt is not None:
                num = re.fullmatch(r"#?\s*(\d{1,2})[.,]?", nxt["text"].strip())
                if num and "BED" in canon:
                    name = f"{name} {int(num.group(1))}"
                    seg = seg + [nxt]
                    span += 1
            phrases.append({
                "text": canon, "name": name,
                "cx": int(np.mean([wd["cx"] for wd in seg])),
                "cy": int(np.mean([wd["cy"] for wd in seg])),
                "h": float(np.mean([wd["h"] for wd in seg])),
                "conf": int(np.mean([wd["conf"] for wd in seg])),
            })
            i += span
    return phrases


def name_rooms(polys: list, phrases: list) -> None:
    """Each label names the room it sits in (or the nearest one, if it sits
    on a wall). A room with several labels is an open-plan space:
    "Kitchen & Dining". A room no label reached stays unnamed ("Room N")."""
    for p in polys:
        p["names"] = []
    for ph in phrases:
        best, bd = None, 1e18
        for p in polys:
            d = -cv2.pointPolygonTest(p["pts"].astype(np.float32).reshape(-1, 1, 2),
                                      (float(ph["cx"]), float(ph["cy"])), True)
            if d < bd:
                bd, best = d, p
        if best is not None and bd <= max(8.0, 2.5 * ph["h"]):
            best["names"].append(ph)
    for p in polys:
        if p.get("outdoor") and p.get("name"):
            continue
        uniq, seen = [], set()
        for ph in sorted(p["names"], key=lambda ph: (-ph["h"], ph["cy"])):
            if ph["name"] not in seen:
                seen.add(ph["name"])
                uniq.append(ph)
        if uniq:
            # biggest lettering first, then top to bottom
            p["name"] = " & ".join(ph["name"] for ph in uniq[:3])[:48]
            p["conf"] = int(np.mean([ph["conf"] for ph in uniq]))
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


# ── 8. Openings: classify every gap in a wall line by what is DRAWN there ────
#
# A gap between two collinear wall pieces is one of four things, and only the
# drawing can say which:
#   door    a swing arc (or an open leaf) beside it, or a short cased gap
#   window  glazing: thin line(s) running along the wall across the gap
#           (a WIDE glazed gap with arrow marks is a sliding door)
#   open    nothing at all and wider than a door: open plan. NOT a wall.
#   noise   a sliver between two pieces of one wall
# The old rule called every empty gap a door, and the shell then bridged it
# with a wall: open-plan kitchens, great rooms and carports got walls that
# are not in the drawing. Now a gap is bridged only when it is a door, a
# window or noise.

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


def _refine_axis_ends(wall_mask, c: int, cur: dict, nxt: dict, horiz: bool,
                      thickness: int, h: int, w: int) -> int:
    """Best-covered line over the wall pieces on both sides of a gap."""
    reach = int(thickness * 6)
    spans = [(int(max(cur["a"], cur["b"] - reach)), int(cur["b"])),
             (int(nxt["a"]), int(min(nxt["b"], nxt["a"] + reach)))]
    best_c, best_cov = c, -1.0
    for dc in range(-thickness, thickness + 1):
        cc = c + dc
        if cc < 0 or cc >= (h if horiz else w):
            continue
        tot = n = 0.0
        for a, b in spans:
            if b <= a:
                continue
            seg = wall_mask[cc, a:b] if horiz else wall_mask[a:b, cc]
            tot += float(np.sum(seg > 128)); n += seg.size
        cov = tot / n if n else 0.0
        # prefer the centre of a thick band over its edge
        if cov > best_cov + 1e-6 or (abs(cov - best_cov) <= 1e-6 and abs(dc) < abs(best_c - c)):
            best_cov, best_c = cov, cc
    return best_c


def _sample_ink(mask, pts) -> float:
    h, w = mask.shape[:2]
    hits = n = 0
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            n += 1
            hits += mask[yi, xi] > 0
    return hits / n if n else 0.0


def _swing_score(ink_d, horiz: bool, c: int, g0: int, g1: int,
                 keep_off: float = 0.0, single: bool = True) -> float:
    """How strongly a door swing is drawn beside the gap: a quarter arc
    hinged at either jamb, on either side, radius ≈ the gap (or two half-size
    arcs for a pair of doors).

    Scored as CONTRAST: ink on the arc minus ink on rings a little inside and
    outside it. Furniture, beds and hatching put ink everywhere and score on
    every ring; a real swing is ink on exactly one radius.
    ink_d has the walls removed, and samples near the wall line are skipped —
    they would hit the glazing or the jambs, not a swing."""
    L = g1 - g0
    if L < 4:
        return 0.0
    angles = np.radians(np.linspace(20, 80, 16))

    def pt(along, perp):
        return (along, perp) if horiz else (perp, along)

    def ring(hinge, toward, side, r):
        pts = [pt(hinge + toward * r * np.cos(a), c + side * r * np.sin(a))
               for a in angles]
        pts = [p for p in pts if abs((p[1] if horiz else p[0]) - c) >= keep_off]
        return _sample_ink(ink_d, pts) if len(pts) >= 6 else 0.0

    best = 0.0
    for side in (1, -1):
        for hinge, toward in (((g0, 1), (g1, -1)) if single else ()):
            on = max(ring(hinge, toward, side, L * f) for f in (0.88, 0.96, 1.04))
            off = max(ring(hinge, toward, side, L * f) for f in (0.68, 1.3))
            best = max(best, on - off)
        # a pair of doors: two half-width arcs meeting in the middle
        r = L / 2
        on = min(max(ring(g0, 1, side, r * f) for f in (0.9, 1.0)),
                 max(ring(g1, -1, side, r * f) for f in (0.9, 1.0)))
        off = max(ring(g0, 1, side, r * 0.6), ring(g0, 1, side, r * 1.4))
        best = max(best, on - off)
        # the open leaf: a line straight out from a jamb, against a control
        # line out from the middle of the gap
        # A leaf is a line exactly one door-width long that ENDS there (a wall
        # or a cabinet edge carries on past it), and some of its arc is drawn.
        for hinge, toward in (((g0, 1), (g1, -1)) if single else ()):
            def line(along, f0=0.3, f1=0.9):
                pts = [pt(along, c + side * L * f) for f in np.linspace(f0, f1, 10)]
                pts = [p for p in pts if abs((p[1] if horiz else p[0]) - c) >= keep_off]
                return _sample_ink(ink_d, pts) if len(pts) >= 5 else 0.0
            d = max((0, L * 0.04, L * 0.08), key=lambda d: line(hinge + toward * d))
            leaf = line(hinge + toward * d)
            beyond = line(hinge + toward * d, 1.15, 1.45)
            arc_seen = max(ring(hinge, toward, side, L * f) for f in (0.9, 1.0))
            if beyond < 0.3 and arc_seen >= 0.25:
                best = max(best, leaf - line(hinge + toward * L * 0.5))
    return best


def _glazing(light, horiz: bool, c: int, g0: int, g1: int, thickness: int) -> bool:
    """Thin lines along the wall spanning most of the gap."""
    band = max(2, int(thickness * 0.6))
    if horiz:
        strip = light[max(0, c - band):c + band + 1, g0:g1]
    else:
        strip = light[g0:g1, max(0, c - band):c + band + 1].T
    if strip.size == 0 or strip.shape[1] == 0:
        return False
    rows = (strip > 0).mean(axis=1) >= 0.7     # each row runs along the wall
    # Glazing is usually two or more parallel lines (frame + glass + sill);
    # some plans draw a slider or window as one line. But ONE line across a
    # very wide gap is something else — a roof edge, a property line, the
    # edge of a drive (the carport's open side) — not a window.
    groups, prev = 0, False
    for r in rows:
        if r and not prev:
            groups += 1
        prev = bool(r)
    return groups >= 2 or (groups == 1 and (g1 - g0) <= 16 * thickness)


def _sliding_marks(ink_clean, horiz: bool, c: int, g0: int, g1: int,
                   thickness: int) -> bool:
    """Sliding doors are drawn with arrow marks (← →) just inside the glazing.
    Look for at least two short strokes parallel to the wall, off its line."""
    t = thickness
    for side in (1, -1):
        lo, hi = (c + int(0.4 * t), c + int(4 * t)) if side > 0 else \
                 (c - int(4 * t), c - int(0.4 * t))
        lo, hi = max(0, lo), max(0, hi)
        if horiz:
            win = ink_clean[lo:hi, g0:g1]
        else:
            win = ink_clean[g0:g1, lo:hi].T
        if win.size == 0:
            continue
        n, _, stats, _ = cv2.connectedComponentsWithStats((win > 0).astype(np.uint8), 8)
        marks = 0
        for i in range(1, n):
            ww, hh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if 0.7 * t <= ww <= 5 * t and hh <= 0.9 * t and ww >= 2 * hh:
                marks += 1
        if 2 <= marks <= 6:
            return True
    return False


import os as _os
_DEBUG_GAPS = bool(_os.getenv("NOOI_DEBUG_GAPS"))


def light_ink(grey):
    """Every mark darker than the paper, however faint: glazing lines and door
    swings are often light grey and anti-aliased, below the Otsu threshold.
    Not text-erased — OCR boxes often cover window lines and punch holes in
    them.

    Two thresholds (hysteresis): a faint pixel counts when it belongs to a
    stroke that is somewhere clearly darker, or to a long faint stroke.
    A single cut-off made a faint door swing flicker in and out with the
    JPEG decoder; JPEG noise forms short specks and is dropped."""
    paper = float(np.percentile(grey, 75))
    weak = (grey < paper - 10).astype(np.uint8)
    strong = grey < paper - 25
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(weak, 8)
    has_strong = np.zeros(n, bool)
    has_strong[np.unique(lbl[strong])] = True
    keep = has_strong & (stats[:, cv2.CC_STAT_AREA] >= 6)
    keep |= stats[:, cv2.CC_STAT_AREA] >= 40
    keep[0] = False
    return (keep[lbl] * 255).astype(np.uint8)


def swing_ink(ink_clean, light, wall_mask, thickness: int):
    """Thin ink only (walls removed), slightly dilated, for swing sampling."""
    t = thickness
    thin = cv2.bitwise_or(ink_clean, light)
    walls_d = cv2.dilate(wall_mask, cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(3, int(t * 0.5)) | 1,) * 2))
    thin = cv2.bitwise_and(thin, cv2.bitwise_not(walls_d))
    return cv2.dilate(thin, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(3, int(t * 0.35)) | 1,) * 2))


def classify_gap(ink_clean, light, ink_d, horiz: bool, c: int, g0: int, g1: int,
                 thickness: int, ink_raw=None, door_max=None) -> tuple:
    """-> (kind, style): kind in door | window | open; style for doors.
    door_max: the widest gap that can be a plain doorway with nothing drawn
    in it (no swing, no glazing)."""
    L = g1 - g0
    # a door leaf is wider than ~2 wall thicknesses; narrower gaps cannot swing
    # no single leaf is wider than ~6 wall thicknesses (≈ 1 m): a wider
    # swing can only be a pair of doors
    score = (_swing_score(ink_d, horiz, c, g0, g1, keep_off=0.9 * thickness,
                          single=L < 6 * thickness)
             if L >= 2 * thickness else 0.0)
    glazed = _glazing(light, horiz, c, g0, g1, thickness)
    # glazing across the gap is the stronger evidence: furniture or the
    # window's own frame can look like part of an arc, but a door gap has no
    # lines running along the wall
    swing = score >= (0.5 if glazed else 0.4)
    if _DEBUG_GAPS:
        log.info(f"    gap {'h' if horiz else 'v'} c={c} {g0}-{g1} L={L} "
                 f"glazed={glazed} swing={score:.2f}")
    if glazed and not swing:
        # arrow marks are often read as text and erased from ink_clean
        marks_src = ink_raw if ink_raw is not None else ink_clean
        if L >= 6 * thickness and _sliding_marks(marks_src, horiz, c, g0, g1, thickness):
            return "door", "sliding"
        return "window", None
    if swing:
        return "door", "hinged"
    if L <= (door_max or 5 * thickness):
        return "door", "opening"        # a cased doorway / bifold closet
    return "open", None


def _split_at_posts(pieces: list, wall_mask, horiz: bool, c: int, t: int) -> list:
    """Short solid posts inside a gap (too short to vectorize as walls)
    separate what is drawn on either side — a door beside a window."""
    band = max(2, int(t * 0.4))
    out = []
    for p0, p1 in pieces:
        if p1 - p0 < 3:
            continue
        if horiz:
            strip = wall_mask[max(0, c - band):c + band + 1, p0:p1]
            cov = (strip > 128).mean(axis=0) if strip.size else np.zeros(0)
        else:
            strip = wall_mask[p0:p1, max(0, c - band):c + band + 1]
            cov = (strip > 128).mean(axis=1) if strip.size else np.zeros(0)
        solid = cov > 0.6
        k, start = 0, p0
        while k < len(solid):
            if solid[k]:
                k0 = k
                while k < len(solid) and solid[k]:
                    k += 1
                if k - k0 >= max(2, int(t * 0.4)):
                    if p0 + k0 > start:
                        out.append((start, p0 + k0))
                    start = p0 + k
            else:
                k += 1
        if p1 > start:
            out.append((start, p1))
    return out


def _wall_lines(walls: list, horiz: bool, col_tol: float) -> list:
    segs = []
    for wi, s in enumerate(walls):
        if (abs(s["x2"] - s["x1"]) >= abs(s["y2"] - s["y1"])) != horiz:
            continue
        a, b = (sorted((s["x1"], s["x2"])) if horiz else sorted((s["y1"], s["y2"])))
        segs.append({"a": a, "b": b, "c": s["y1"] if horiz else s["x1"], "wi": wi})
    segs.sort(key=lambda s: s["c"])
    clusters = []
    for s in segs:
        if clusters and abs(s["c"] - clusters[-1][-1]["c"]) <= col_tol:
            clusters[-1].append(s)
        else:
            clusters.append([s])
    for cl in clusters:
        cl.sort(key=lambda s: s["a"])
    return clusters


def _junctions(walls: list, horiz: bool, c: float, g0: float, g1: float,
               thickness: int) -> list:
    """Perpendicular walls meeting this line inside (g0, g1): a T-junction or
    a crossing. A gap is never one opening across another wall."""
    out = []
    tol = thickness * 1.2
    for s in walls:
        if (abs(s["x2"] - s["x1"]) >= abs(s["y2"] - s["y1"])) == horiz:
            continue
        p = s["x1"] if horiz else s["y1"]
        lo, hi = (sorted((s["y1"], s["y2"])) if horiz else sorted((s["x1"], s["x2"])))
        if g0 < p < g1 and lo - tol <= c <= hi + tol:
            out.append(p)
    return sorted(out)


def detect_openings(walls: list, wall_mask, ink_clean, thickness: int,
                    h: int, w: int, light=None, ink_raw=None,
                    thin_wall=None) -> tuple:
    """Classify every gap between collinear wall pieces.
    Returns (openings, bridges): bridges are the gaps the shell may close —
    doors, windows and slivers — as (horiz, c, g0, g1)."""
    if light is None:
        light = ink_clean
    t = thickness
    ink_d = swing_ink(ink_clean, light, wall_mask, t)
    min_gap = int(t * 1.1)
    max_gap = int(t * 40)
    # pieces of one wall line agree to within about a wall's width; the
    # thickest-wall measure let parallel walls a hall apart share a line
    col_tol = 1.2 * min(t, thin_wall or t)
    # a doorway is ~6-8 interior-wall thicknesses wide (0.8-0.9 m)
    door_max = min(5 * t, 7 * thin_wall) if thin_wall else None
    openings, bridges = [], []

    for horiz in (True, False):
        for cl in _wall_lines(walls, horiz, col_tol):
            for i in range(len(cl) - 1):
                cur, nxt = cl[i], cl[i + 1]
                g0, g1 = int(cur["b"]), int(nxt["a"])
                gap = g1 - g0
                c = int(round((cur["c"] + nxt["c"]) / 2))
                if gap <= min_gap:
                    bridges.append((horiz, c, g0, g1))
                    continue
                if gap > max_gap:
                    continue
                # Centre the test band on the WALL either side of the gap, not
                # on the gap itself: measured over the gap, the best-covered
                # row is whatever furniture or glazing happens to be drawn
                # there, which moved the band a whole wall-thickness away.
                c = _refine_axis_ends(wall_mask, c, cur, nxt, horiz, t, h, w)
                # split at walls meeting this line inside the gap
                cuts = _junctions(walls, horiz, c, g0, g1, t)
                pieces, start = [], g0
                for p in cuts:
                    pieces.append((start, int(p - t / 2)))
                    start = int(p + t / 2)
                pieces.append((start, g1))
                pieces = _split_at_posts(pieces, wall_mask, horiz, c, t)
                found, is_open = [], False
                for p0, p1 in pieces:
                    if p1 - p0 < min_gap:
                        continue                    # junction remnant
                    if horiz:
                        sw = wall_mask[max(0, c - 2):c + 3, p0:p1]
                    else:
                        sw = wall_mask[p0:p1, max(0, c - 2):c + 3]
                    if sw.size and float(np.mean(sw > 128)) > 0.45:
                        continue                    # actually solid
                    kind, style = classify_gap(ink_clean, light, ink_d, horiz, c,
                                               p0, p1, t, ink_raw=ink_raw,
                                               door_max=door_max)
                    if kind == "open":
                        is_open = True
                        continue
                    mid = (p0 + p1) // 2
                    found.append({
                        "type": kind, "style": style,
                        "wall_index": cur["wi"],
                        "wall": "horizontal" if horiz else "vertical",
                        "px": mid if horiz else c, "py": c if horiz else mid,
                        "len_px": p1 - p0,
                    })
                if is_open:
                    continue        # open plan: no wall, so no opening either
                openings.extend(found)
                bridges.append((horiz, c, g0, g1))

    openings = _dedup_openings_px(openings, t)
    log.info(f"  Openings: {len(openings)} "
             f"({sum(1 for o in openings if o['type']=='door')} doors, "
             f"{sum(1 for o in openings if o['type']=='window')} windows); "
             f"{len(bridges)} bridgeable gaps")
    return openings, bridges


def _dedup_openings_px(openings: list, thickness: int) -> list:
    unique = []
    for op in openings:
        if not any(np.hypot(op["px"] - u["px"], op["py"] - u["py"])
                   < thickness * 2 for u in unique):
            unique.append(op)
    return unique




# ── 8b. Continuous render walls + within-wall opening scan ───────────────────
# The raw vectorized fragments have gaps at every door/window. For 2D/3D
# rendering we fuse collinear fragments across those gaps into CONTINUOUS
# walls (Coohom-style), then scan along each continuous wall and classify
# every low-coverage run as a door or window — so openings become cuts in a
# solid wall instead of holes between floating panels.

def build_render_walls(walls_px: list, thickness: int, h: int, w: int) -> list:
    segs = [dict(s) for s in walls_px]
    merged = _merge_collinear(segs, tol=thickness * 1.2)
    # aggressive second pass: also merge across opening-sized gaps
    out, used = [], set()
    for horiz in (True, False):
        group = []
        for i, s in enumerate(merged):
            is_h = abs(s["x2"] - s["x1"]) >= abs(s["y2"] - s["y1"])
            if is_h == horiz:
                group.append(s)
        # cluster by axis
        key = (lambda s: s["y1"]) if horiz else (lambda s: s["x1"])
        group.sort(key=key)
        clusters = []
        for s in group:
            if clusters and abs(key(s) - key(clusters[-1][-1])) <= thickness * 1.2:
                clusters[-1].append(s)
            else:
                clusters.append([s])
        for cl in clusters:
            lo_k = (lambda s: min(s["x1"], s["x2"])) if horiz                 else (lambda s: min(s["y1"], s["y2"]))
            hi_k = (lambda s: max(s["x1"], s["x2"])) if horiz                 else (lambda s: max(s["y1"], s["y2"]))
            cl.sort(key=lo_k)
            cur = dict(cl[0])
            for s in cl[1:]:
                gap = lo_k(s) - hi_k(cur)
                if gap <= thickness * 16:      # fuse across door/slider gaps
                    if horiz:
                        cur["x1"] = min(cur["x1"], cur["x2"], s["x1"], s["x2"])
                        cur["x2"] = max(cur["x1"], cur["x2"], s["x1"], s["x2"])
                        cur["y1"] = cur["y2"] = float(np.mean(
                            [cur["y1"], s["y1"]]))
                    else:
                        cur["y1"] = min(cur["y1"], cur["y2"], s["y1"], s["y2"])
                        cur["y2"] = max(cur["y1"], cur["y2"], s["y1"], s["y2"])
                        cur["x1"] = cur["x2"] = float(np.mean(
                            [cur["x1"], s["x1"]]))
                    cur["thickness"] = max(cur["thickness"], s["thickness"])
                else:
                    out.append(cur)
                    cur = dict(s)
            out.append(cur)
    log.info(f"  Render walls: {len(out)} continuous "
             f"(from {len(walls_px)} fragments)")
    return out


def scan_wall_openings(render_walls: list, wall_mask, ink_clean,
                       thickness: int, h: int, w: int) -> list:
    """Walk each CONTINUOUS wall; classify low-wall-coverage runs:
         thin ink present -> window,   empty -> door."""
    openings = []
    band = max(2, int(thickness * 0.5))
    min_open = int(thickness * 1.1)
    max_open = int(thickness * 16)

    for wi, seg in enumerate(render_walls):
        horiz = abs(seg["x2"] - seg["x1"]) >= abs(seg["y2"] - seg["y1"])
        if horiz:
            a = int(min(seg["x1"], seg["x2"])); b = int(max(seg["x1"], seg["x2"]))
            c = int(seg["y1"])
        else:
            a = int(min(seg["y1"], seg["y2"])); b = int(max(seg["y1"], seg["y2"]))
            c = int(seg["x1"])
        L = b - a
        if L < min_open * 2:
            continue
        c = _refine_axis(wall_mask, c, a, b, horiz, thickness, h, w)

        cov = np.zeros(L, np.float32)
        inkl = np.zeros(L, np.float32)
        lo = max(0, c - band)
        if horiz:
            hi = min(h, c + band + 1)
            strip_w = wall_mask[lo:hi, a:b]
            strip_i = ink_clean[lo:hi, a:b]
            cov = (strip_w > 128).mean(axis=0)
            inkl = (strip_i > 128).mean(axis=0)
        else:
            hi = min(w, c + band + 1)
            strip_w = wall_mask[a:b, lo:hi]
            strip_i = ink_clean[a:b, lo:hi]
            cov = (strip_w > 128).mean(axis=1)
            inkl = (strip_i > 128).mean(axis=1)

        k = max(3, thickness // 2) | 1
        cov = cv2.blur(cov.reshape(1, -1).astype(np.float32), (k, 1)).ravel()
        absent = cov < 0.30
        t = 0
        while t < L:
            if not absent[t]:
                t += 1
                continue
            t0 = t
            while t < L and absent[t]:
                t += 1
            run = t - t0
            if run < min_open or run > max_open:
                continue
            mean_ink = float(inkl[t0:t].mean()) if t > t0 else 0.0
            kind = "window" if mean_ink > 0.12 else "door"
            mid = a + (t0 + t) // 2
            px, py = (mid, c) if horiz else (c, mid)
            openings.append({
                "type": kind, "wall_index": wi,
                "wall": "horizontal" if horiz else "vertical",
                "px": px, "py": py, "len_px": run,
            })
    return openings


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
    CFGS = ("--psm 6 --oem 3", "--psm 11 --oem 3")

    def crop_of(p):
        xs = p["pts"][:, 0] / work_scale
        ys = p["pts"][:, 1] / work_scale
        px_w = float(xs.max() - xs.min())
        px_h = float(ys.max() - ys.min())
        padx, pady = int(px_w * 0.35), int(px_h * 0.35)
        x0 = max(0, int(xs.min()) - padx); x1 = min(ow, int(xs.max()) + padx)
        y0 = max(0, int(ys.min()) - pady); y1 = min(oh, int(ys.max()) + pady)
        crop = grey_orig[y0:y1, x0:x1]
        if crop.size == 0 or min(crop.shape) < 8:
            return None
        s = max(3.0, 750.0 / max(crop.shape))
        up = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        return px_w, px_h, x0, y0, s, up

    # Every crop is OCR'd twice; Tesseract runs as a separate process, so
    # the passes run side by side (several cores) instead of one by one.
    crops = {id(p): crop_of(p) for p in polys}
    jobs = [(pid, cfg) for pid, c in crops.items() if c is not None for cfg in CFGS]

    def _ocr(job):
        pid, cfg = job
        try:
            return job, pytesseract.image_to_data(
                crops[pid][5], config=cfg, output_type=pytesseract.Output.DICT)
        except Exception:
            return job, None
    from concurrent.futures import ThreadPoolExecutor
    import os as _os2
    with ThreadPoolExecutor(max_workers=max(1, min(6, (_os2.cpu_count() or 2)))) as ex:
        ocr_out = dict(ex.map(_ocr, jobs))

    for p in polys:
        c = crops[id(p)]
        if c is None:
            continue
        px_w, px_h, x0, y0, s, up = c
        crop_words, toks = [], []
        for cfg in CFGS:
            data = ocr_out.get((id(p), cfg))
            if data is None:
                continue
            for i in range(len(data["text"])):
                t = data["text"][i].strip()
                cf = int(float(data["conf"][i]))
                if not t or cf < 25:
                    continue
                if cfg.startswith("--psm 6"):
                    toks.append(t)
                # crop px -> original px -> working px
                k = work_scale / s
                x = int(x0 * work_scale + data["left"][i] * k)
                y = int(y0 * work_scale + data["top"][i] * k)
                bw_ = max(1, int(data["width"][i] * k)); bh_ = max(1, int(data["height"][i] * k))
                crop_words.append({"text": t, "conf": cf, "x": x, "y": y, "w": bw_, "h": bh_,
                                   "cx": x + bw_ // 2, "cy": y + bh_ // 2})
        if not toks and not crop_words:
            continue
        joined = " ".join(toks).replace("\u2019", "'")

        # labels read in this crop that sit INSIDE this room (the crop is
        # padded, so neighbours' labels are in it too and must not count)
        poly = p["pts"].astype(np.float32).reshape(-1, 1, 2)
        p["crop_phrases"] = [
            ph for ph in label_phrases(crop_words)
            if cv2.pointPolygonTest(poly, (float(ph["cx"]), float(ph["cy"])), True)
            >= -max(4.0, ph["h"])]

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

def detect_outdoor_areas(grey, ink_clean, seal, footprint, labels, phrases,
                         polys, words, thickness: int, h: int, w: int) -> list:
    """Patio/porch/covered porch are open-sided, so enclosure tests fail.
    Instead: flood-fill from each outdoor-named OCR label within the thin-ink
    barrier, clipped to a window around the label. Slight overreach is fine —
    these render as floor patches, not walled rooms."""
    # Outdoor outlines are pale double lines + dashes that global Otsu
    # binarization DISCARDS — so build a dedicated LIGHT-ink layer for them:
    # near-white-sensitive threshold, speck removal, then directional
    # closings to reconnect the dashes into continuous boundaries.
    light = (cv2.GaussianBlur(grey, (3, 3), 0) < 245).astype(np.uint8) * 255
    light = cv2.morphologyEx(light, cv2.MORPH_OPEN,
                             np.ones((2, 2), np.uint8))
    # text glyphs are light too — erase OCR word boxes so a label's own
    # letters can't wall in the flood seed placed right next to it
    light = erase_text(light, words, h, w)
    # Bridge dashed outlines with a SMALL closing only: outdoor interiors are
    # often textured with light deck-board lines ~1-2 wall-thicknesses apart,
    # and a large closing fuses that texture into a solid block that entombs
    # the flood seed (the porch became a 12-pixel pocket).
    blen = max(5, int(thickness * 0.9)) | 1
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (blen, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, blen))
    bridged = cv2.bitwise_or(
        cv2.morphologyEx(light, cv2.MORPH_CLOSE, kh),
        cv2.morphologyEx(light, cv2.MORPH_CLOSE, kv))
    barrier = cv2.bitwise_or(bridged, seal)
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
                          "conf": ph["conf"], "th": float(ph.get("h", 12))})

    if not seeds:
        log.info("  Outdoor areas: 0 (none)")
        return out

    # Pass 2: flood ALL labels first (no claiming). Seeds sharing one
    # connected area (patio+porch behind a faint divider) get that area's
    # TRUE outer boundary, split internally by nearest label. Only floods
    # that reach the image border fall back to window-clip + spur shave.
    for s in seeds:
        ff = free.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ff, mask, s["seed"], 128)
        s["flood"] = (ff == 128)
        ys_r, xs_r = np.nonzero(s["flood"])
        s["touches"] = bool(ys_r.min() <= 2 or xs_r.min() <= 2 or
                            ys_r.max() >= h - 3 or xs_r.max() >= w - 3)
        s["frac"] = ys_r.size / (h * w)

    union = np.zeros((h, w), bool)
    for s in seeds:
        union |= s["flood"]
    ys_f, xs_f = np.nonzero(union)
    if ys_f.size:
        # nearest label among the seeds whose flood actually covers the pixel
        dists = []
        for s in seeds:
            d = (xs_f - s["cx"]) ** 2.0 + (ys_f - s["cy"]) ** 2.0
            d[~s["flood"][ys_f, xs_f]] = np.inf
            dists.append(d)
        owner = np.argmin(np.stack(dists), axis=0)
        for si, s in enumerate(seeds):
            region = np.zeros((h, w), np.uint8)
            sel = owner == si
            region[ys_f[sel], xs_f[sel]] = 255
            if s["touches"] or not (0.005 < s["frac"] < 0.45):
                # leaky: window-clip + shave
                win = np.zeros((h, w), np.uint8)
                hw, hh = int(w * 0.34), int(h * 0.34)
                win[max(0, s["cy"] - hh):min(h, s["cy"] + hh),
                    max(0, s["cx"] - hw):min(w, s["cx"] + hw)] = 255
                region = cv2.bitwise_and(region, win)
                shave = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (int(thickness * 3) | 1,) * 2)
                region = cv2.morphologyEx(region, cv2.MORPH_OPEN, shave)
            else:
                log.info(f"    {s['name']}: true outer boundary "
                         f"({s['frac']*100:.1f}%)")
            s["region"] = region

    for s in seeds:
        region = s.get("region")
        if region is None:
            continue
        # the label and its size line are ink inside the area: give their
        # box back to it, or the outline has a bite where the name is
        th = s["th"]
        bx0, bx1 = max(0, int(s["cx"] - 4 * th)), min(w, int(s["cx"] + 4 * th))
        by0, by1 = max(0, int(s["cy"] - 1.5 * th)), min(h, int(s["cy"] + 3 * th))
        box = np.zeros((h, w), np.uint8)
        box[by0:by1, bx0:bx1] = 255
        box[seal > 0] = 0
        region = cv2.bitwise_or(region, box)
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




# ── 10. Output wall shell: chain fragments into continuous walls ─────────────
# Raw vectorized fragments have holes wherever windows/doors interrupt them,
# which renders as a broken 3D shell. For OUTPUT we chain collinear stubs
# across their gaps into one continuous wall (the gaps stay recorded as
# openings, which the 3D cuts back out as doors/windows), and extend wall
# ends to meet perpendicular walls so corners close.

def chain_output_walls(walls_px: list, openings_px: list, thickness: int,
                       h: int, w: int, wall_mask=None, bridges=None,
                       ink_clean=None, light=None, thin_wall=None):
    col_tol = 1.2 * min(thickness, thin_wall or thickness)
    min_gap = thickness * 1.1
    chains = []

    def bridgeable(horiz, c, g0, g1):
        """Only close a gap the drawing explains: a sliver, a door or a
        window (detect_openings). An unexplained gap is open plan."""
        if g1 - g0 <= min_gap:
            return True
        if bridges is None:
            return g1 - g0 <= thickness * 22          # legacy behaviour
        for bh, bc, b0, b1 in bridges:
            if (bh == horiz and abs(bc - c) <= col_tol * 1.5
                    and b0 <= g0 + thickness and b1 >= g1 - thickness):
                return True
        return False

    for horiz in (True, False):
        segs = []
        for s in walls_px:
            is_h = abs(s["x2"] - s["x1"]) >= abs(s["y2"] - s["y1"])
            if is_h != horiz:
                continue
            if horiz:
                a, b = sorted((s["x1"], s["x2"])); c = s["y1"]
            else:
                a, b = sorted((s["y1"], s["y2"])); c = s["x1"]
            segs.append({"a": a, "b": b, "c": c, "t": s["thickness"]})
        segs.sort(key=lambda s: s["c"])
        clusters = []
        for s in segs:
            if clusters and abs(s["c"] - clusters[-1][-1]["c"]) <= col_tol:
                clusters[-1].append(s)
            else:
                clusters.append([s])
        for cl in clusters:
            cl.sort(key=lambda s: s["a"])
            cur = dict(cl[0])
            members = [cl[0]]
            for s in cl[1:]:
                if s["a"] <= cur["b"] or bridgeable(horiz, s["c"], cur["b"], s["a"]):
                    cur["b"] = max(cur["b"], s["b"])
                    cur["t"] = max(cur["t"], s["t"])
                    members.append(s)
                else:
                    cur["c"] = float(np.average(
                        [m["c"] for m in members],
                        weights=[m["b"] - m["a"] + 1 for m in members]))
                    chains.append({"horiz": horiz, **cur})
                    cur = dict(s); members = [s]
            cur["c"] = float(np.average(
                [m["c"] for m in members],
                weights=[m["b"] - m["a"] + 1 for m in members]))
            chains.append({"horiz": horiz, **cur})

    # One wall drawn as two parallel lines (or a wall face plus a thick
    # outline) vectorizes as two chains a few px apart. Merge them: one wall,
    # as thick as the pair.
    merged = True
    while merged:
        merged = False
        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                a, b = chains[i], chains[j]
                if a["horiz"] != b["horiz"]:
                    continue
                off = abs(a["c"] - b["c"])
                # two faces of ONE wall are about a wall-width apart; measured
                # against the thickest walls this also swallowed real walls a
                # narrow hall apart
                if off > (thin_wall or thickness) * 2.0:
                    continue
                ov = min(a["b"], b["b"]) - max(a["a"], b["a"])
                if ov <= 0.5 * min(a["b"] - a["a"], b["b"] - b["a"]):
                    continue
                la, lb = a["b"] - a["a"], b["b"] - b["a"]
                a["c"] = (a["c"] * la + b["c"] * lb) / (la + lb)
                a["a"], a["b"] = min(a["a"], b["a"]), max(a["b"], b["b"])
                a["t"] = min(thickness * 2, max(a["t"], b["t"], off + min(a["t"], b["t"])))
                chains.pop(j)
                merged = True
                break
            if merged:
                break

    # corner closure: extend each chain end to the nearest perpendicular
    # chain line beyond it (within 4 thicknesses), if that chain's span
    # covers this axis position — the corner piece was too short to
    # vectorize. A window or door drawn in that stretch is recorded.
    # (This step used to compute the gap with the wrong sign and never
    # extended anything, which left corners open wherever a window sat
    # next to them.)
    # far_reach: tried 16 thicknesses; stair treads and door swings then
    # "explained" walls across stairwells and closets. Corners only.
    near_reach = far_reach = thickness * 4.0
    ink_d = (swing_ink(ink_clean, light if light is not None else ink_clean,
                       wall_mask, thickness)
             if (ink_clean is not None and wall_mask is not None) else None)
    for ch in chains:
        for end in ("a", "b"):
            cands = []
            for other in chains:
                if other["horiz"] == ch["horiz"] or other is ch:
                    continue
                line = other["c"]
                covers = other["a"] - col_tol <= ch["c"] <= other["b"] + col_tol
                # an L corner where BOTH walls stop short of it (a window
                # next to the corner on each side): neither covers the other
                mutual = (not covers and
                          other["a"] - near_reach <= ch["c"] <= other["b"] + near_reach)
                if not (covers or mutual):
                    continue
                gap = (ch["a"] - line) if end == "a" else (line - ch["b"])
                if 0 < gap <= far_reach:
                    cands.append((gap, line, other, mutual))
            if not cands:
                continue
            gap, line, other, mutual = min(cands, key=lambda c: c[0])
            # a mutual corner needs evidence (wall, window, door) — two walls
            # that merely end near each other may be an open corner
            ok = gap <= near_reach and not mutual
            if ink_d is not None:
                p0, p1 = (int(line + other["t"] / 2), int(ch["a"])) if end == "a" \
                    else (int(ch["b"]), int(line - other["t"] / 2))
                cc = int(ch["c"])
                if p1 - p0 > 2:
                    if ch["horiz"]:
                        sw = wall_mask[max(0, cc - 2):cc + 3, p0:p1]
                    else:
                        sw = wall_mask[p0:p1, max(0, cc - 2):cc + 3]
                    if sw.size and float(np.mean(sw > 128)) >= 0.6:
                        ok = True
                    elif p1 - p0 >= thickness * 1.1:
                        kind, style = classify_gap(
                            ink_clean, light if light is not None else ink_clean,
                            ink_d, ch["horiz"], cc, p0, p1, thickness)
                        # a bare gap at a corner is just the corner piece
                        # that was too short to vectorize — not a doorway
                        if kind != "open" and style != "opening":
                            ok = True
                            mid = (p0 + p1) // 2
                            openings_px.append({
                                "type": kind, "style": style, "wall_index": 0,
                                "wall": "horizontal" if ch["horiz"] else "vertical",
                                "px": mid if ch["horiz"] else cc,
                                "py": cc if ch["horiz"] else mid,
                                "len_px": p1 - p0,
                            })
            if ok:
                if end == "a":
                    ch["a"] = min(ch["a"], line)
                else:
                    ch["b"] = max(ch["b"], line)

    # synthesize RETURN WALLS: two parallel chains whose ends sit close in
    # the along-axis direction but offset across it (a bay/bump-out) need a
    # short perpendicular connector, or the 3D shell shows a gap between them
    connectors = []
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            a, b = chains[i], chains[j]
            if a["horiz"] != b["horiz"]:
                continue
            off = abs(a["c"] - b["c"])
            if not (col_tol < off <= thickness * 10):
                continue
            # connect an ENDPOINT of one chain to the other chain wherever
            # the other's span covers it (bay walls meet the middle of the
            # long facade chain, not its ends)
            for src_ch, dst_ch in ((a, b), (b, a)):
                for e in ("a", "b"):
                    pos = src_ch[e]
                    if dst_ch["a"] - col_tol <= pos <= dst_ch["b"] + col_tol:
                        lo, hi = sorted((a["c"], b["c"]))
                        connectors.append({
                            "horiz": not a["horiz"],
                            "a": lo, "b": hi, "c": pos,
                            "t": min(a["t"], b["t"]),
                        })
    # keep only connectors backed by ACTUAL drawn wall ink — a real return
    # wall exists in the drawing; a fictional shortcut between two unrelated
    # parallel walls does not
    if wall_mask is not None:
        band = max(2, int(thickness * 0.8))
        backed = []
        for c0 in connectors:
            lo, hi = int(c0["a"]), int(c0["b"])
            if hi - lo < 2:
                continue
            # the drawn return wall may sit a few px off the chain endpoint:
            # search cc +/- thickness for the best-covered line and refine
            best_cov, best_cc = 0.0, int(c0["c"])
            for dc in range(-thickness, thickness + 1, 2):
                cc = int(c0["c"]) + dc
                if c0["horiz"]:
                    y0, y1 = max(0, cc - band), min(h, cc + band + 1)
                    strip = wall_mask[y0:y1, max(0, lo):min(w, hi)]
                else:
                    x0, x1 = max(0, cc - band), min(w, cc + band + 1)
                    strip = wall_mask[max(0, lo):min(h, hi), x0:x1]
                cov = float(np.mean(strip > 128)) if strip.size else 0.0
                if cov > best_cov:
                    best_cov, best_cc = cov, cc
            if best_cov >= 0.40:
                c0["c"] = float(best_cc)
                backed.append(c0)
        connectors = backed
    # a connector where a wall already runs is that wall again — drop it
    connectors = [
        c0 for c0 in connectors
        if not any(ch["horiz"] == c0["horiz"]
                   and abs(ch["c"] - c0["c"]) <= thickness * 1.5
                   and min(ch["b"], c0["b"]) - max(ch["a"], c0["a"])
                       >= 0.5 * (c0["b"] - c0["a"])
                   for ch in chains)]
    # avoid duplicate connectors at the same spot
    dedup = []
    for c0 in connectors:
        if not any(c1["horiz"] == c0["horiz"] and
                   abs(c1["c"] - c0["c"]) < thickness * 1.5 and
                   abs(c1["a"] - c0["a"]) < thickness * 1.5
                   for c1 in dedup):
            dedup.append(c0)
    if dedup:
        log.info(f"  Shell: +{len(dedup)} return-wall connectors")
    # A connector is a short wall found from the drawing's ink; the part of it
    # that is NOT solid wall is an opening (bay sides are usually windows).
    if wall_mask is not None and ink_clean is not None:
        ink_d = swing_ink(ink_clean, light if light is not None else ink_clean,
                          wall_mask, thickness)
        band = max(2, int(thickness * 0.5))
        for c0 in dedup:
            a0, b0, cc = int(c0["a"]), int(c0["b"]), int(c0["c"])
            if c0["horiz"]:
                strip = wall_mask[max(0, cc - band):cc + band + 1, a0:b0]
                cov = (strip > 128).mean(axis=0) if strip.size else np.zeros(0)
            else:
                strip = wall_mask[a0:b0, max(0, cc - band):cc + band + 1]
                cov = (strip > 128).mean(axis=1) if strip.size else np.zeros(0)
            absent = cov < 0.3
            k = 0
            while k < len(absent):
                if not absent[k]:
                    k += 1
                    continue
                k0 = k
                while k < len(absent) and absent[k]:
                    k += 1
                p0, p1 = a0 + k0, a0 + k
                if p1 - p0 < min_gap:
                    continue
                kind, style = classify_gap(ink_clean, light if light is not None else ink_clean,
                                           ink_d, c0["horiz"], cc, p0, p1, thickness)
                if kind == "open":
                    continue
                mid = (p0 + p1) // 2
                openings_px.append({
                    "type": kind, "style": style, "wall_index": 0,
                    "wall": "horizontal" if c0["horiz"] else "vertical",
                    "px": mid if c0["horiz"] else cc, "py": cc if c0["horiz"] else mid,
                    "len_px": p1 - p0,
                })
    chains.extend(dedup)

    out_walls = []
    for ch in chains:
        if ch["b"] - ch["a"] < thickness * 1.2:
            continue
        if ch["horiz"]:
            out_walls.append({"x1": ch["a"], "y1": ch["c"],
                              "x2": ch["b"], "y2": ch["c"],
                              "thickness": ch["t"]})
        else:
            out_walls.append({"x1": ch["c"], "y1": ch["a"],
                              "x2": ch["c"], "y2": ch["b"],
                              "thickness": ch["t"]})

    # re-home each opening onto the chained wall that now contains it; one
    # that no wall contains is dropped (it used to land on wall w0)
    homed = []
    for op in openings_px:
        horiz = op["wall"] == "horizontal"
        pos = op["px"] if horiz else op["py"]
        axis = op["py"] if horiz else op["px"]
        best_i, best_d = 0, 1e18
        for i, wl in enumerate(out_walls):
            is_h = abs(wl["x2"] - wl["x1"]) >= abs(wl["y2"] - wl["y1"])
            if is_h != horiz:
                continue
            c = wl["y1"] if is_h else wl["x1"]
            lo = min(wl["x1"], wl["x2"]) if is_h else min(wl["y1"], wl["y2"])
            hi = max(wl["x1"], wl["x2"]) if is_h else max(wl["y1"], wl["y2"])
            if not (lo - col_tol <= pos <= hi + col_tol):
                continue
            d = abs(axis - c)
            if d < best_d:
                best_d, best_i = d, i
        if best_d <= col_tol * 1.5:
            op["wall_index"] = best_i
            homed.append(op)
    openings_px[:] = homed

    log.info(f"  Shell: {len(walls_px)} fragments -> {len(out_walls)} walls")
    return out_walls




def _ray_clear(walls_px: list, skip: int, horiz: bool, c: float, along: float,
               side: int, start: float) -> bool:
    """From the wall line c at position `along`, does a ray going `side`
    (across the wall) reach the edge of the sheet without crossing a wall?"""
    for j, o in enumerate(walls_px):
        if j == skip:
            continue
        oh = abs(o["x2"] - o["x1"]) >= abs(o["y2"] - o["y1"])
        half = o["thickness"] / 2
        if horiz:          # ray runs along y at x = along
            lo, hi = min(o["x1"], o["x2"]) - (0 if oh else half), max(o["x1"], o["x2"]) + (0 if oh else half)
            n0, n1 = min(o["y1"], o["y2"]) - (half if oh else 0), max(o["y1"], o["y2"]) + (half if oh else 0)
        else:
            lo, hi = min(o["y1"], o["y2"]) - (half if oh else 0), max(o["y1"], o["y2"]) + (half if oh else 0)
            n0, n1 = min(o["x1"], o["x2"]) - (0 if oh else half), max(o["x1"], o["x2"]) + (0 if oh else half)
        if not (lo <= along <= hi):
            continue
        frm = c + side * start
        if (side > 0 and n1 > frm) or (side < 0 and n0 < frm):
            return False
    return True


def windows_inside_walls(walls_px: list, openings_px: list, grey,
                         thickness: int, h: int, w: int) -> list:
    """Windows drawn as a FILLED box in the wall line (frame lines with a
    tinted fill between them) are as dark and as thick as the wall itself,
    so they vectorize as solid wall and leave no gap to classify.

    Look inside every output wall for stretches where, across the wall, the
    profile is three or more dark lines with light between them (a solid wall
    is dark all the way through, a hollow wall is two lines), and where those
    lines run unbroken along the stretch (hatching changes along the wall;
    glazing does not)."""
    added = []
    t = thickness
    taken = []
    for op in openings_px:
        horiz = op["wall"] == "horizontal"
        mid = op["px"] if horiz else op["py"]
        taken.append((horiz, op["py"] if horiz else op["px"],
                      mid - op["len_px"] / 2, mid + op["len_px"] / 2))
    for wi, wl in enumerate(walls_px):
        horiz = abs(wl["x2"] - wl["x1"]) >= abs(wl["y2"] - wl["y1"])
        c = int(round(wl["y1"] if horiz else wl["x1"]))
        a = int(min(wl["x1"], wl["x2"]) if horiz else min(wl["y1"], wl["y2"]))
        b = int(max(wl["x1"], wl["x2"]) if horiz else max(wl["y1"], wl["y2"]))
        half = max(3, int(max(wl["thickness"], t * 0.8) * 0.75))
        if horiz:
            band = grey[max(0, c - half):c + half + 1, max(0, a):min(w, b)]
        else:
            band = grey[max(0, a):min(h, b), max(0, c - half):c + half + 1].T
        if band.size == 0 or band.shape[0] < 4:
            continue
        dark, lite = band < 170, band > 205
        # per position along the wall: dark, then light, then dark again
        striped = np.zeros(band.shape[1], bool)
        for k in range(band.shape[1]):
            d = np.nonzero(dark[:, k])[0]
            if len(d) >= 2 and lite[d[0]:d[-1] + 1, k].any():
                striped[k] = True
        sm = cv2.blur(striped.astype(np.float32).reshape(1, -1),
                      (max(3, int(t * 0.5)) | 1, 1)).ravel() > 0.6
        k = 0
        while k < len(sm):
            if not sm[k]:
                k += 1
                continue
            k0 = k
            while k < len(sm) and sm[k]:
                k += 1
            if k - k0 < 1.5 * t:
                continue
            seg = band[:, k0:k]
            rows_dark = (seg < 170).mean(axis=1) >= 0.85
            rows_lite = (seg > 205).mean(axis=1) >= 0.7
            idx = np.nonzero(rows_dark)[0]
            if len(idx) < 2 or not rows_lite[idx[0]:idx[-1] + 1].any():
                continue           # not continuous lines along the wall
            # a hollow wall is TWO lines with paper between them; a window
            # box has three or more (frame, glass, frame)
            groups = int(np.count_nonzero(np.diff(np.r_[0, rows_dark.astype(int)]) == 1))
            if groups < 3:
                continue
            p0, p1 = a + k0, a + k
            # outer walls only (closet shelving and stair treads inside the
            # house draw the same stripes): looking straight out from one
            # side of the wall, no other wall is in the way
            mid = (p0 + p1) / 2
            if not any(_ray_clear(walls_px, wi, horiz, c, mid, side, half)
                       for side in (-1, 1)):
                continue
            if any(th == horiz and abs(tc - c) <= t and min(p1, t1) - max(p0, t0) > 0
                   for th, tc, t0, t1 in taken):
                continue
            mid = (p0 + p1) // 2
            added.append({
                "type": "window", "style": None, "wall_index": wi,
                "wall": "horizontal" if horiz else "vertical",
                "px": mid if horiz else c, "py": c if horiz else mid,
                "len_px": p1 - p0,
            })
    if added:
        log.info(f"  Windows inside wall lines: {len(added)}")
    return openings_px + added


def _regularize_doors(openings: list, thickness: int) -> list:
    """Two fixes observed in 3D side views:
    1. MERGE doors on the same wall whose edges nearly touch — a single real
       door sometimes splits into two detected openings across an artifact.
    2. NORMALIZE interior door widths: standard doors on a plan are all the
       same size; per-gap measurement noise makes them render differently.
       Snap door widths within +/-35%% of the median to the median."""
    # 1) merge near-duplicates per wall
    doors = [o for o in openings if o["type"] == "door"]
    others = [o for o in openings if o["type"] != "door"]
    by_wall: dict = {}
    for o in doors:
        by_wall.setdefault((o["wall_index"], o["wall"]), []).append(o)
    merged = []
    for key, group in by_wall.items():
        horiz = key[1] == "horizontal"
        group.sort(key=lambda o: o["px"] if horiz else o["py"])
        cur = dict(group[0])
        for o in group[1:]:
            c_cur = cur["px"] if horiz else cur["py"]
            c_o = o["px"] if horiz else o["py"]
            gap = (c_o - o["len_px"] / 2) - (c_cur + cur["len_px"] / 2)
            if gap < thickness * 1.5:
                lo = min(c_cur - cur["len_px"] / 2, c_o - o["len_px"] / 2)
                hi = max(c_cur + cur["len_px"] / 2, c_o + o["len_px"] / 2)
                mid = (lo + hi) / 2
                cur["len_px"] = hi - lo
                if horiz:
                    cur["px"] = int(mid)
                else:
                    cur["py"] = int(mid)
            else:
                merged.append(cur)
                cur = dict(o)
        merged.append(cur)

    # 2) normalize standard door widths to the median
    std = [o for o in merged if o["len_px"] <= thickness * 4.5]
    if len(std) >= 3:
        med = float(np.median([o["len_px"] for o in std]))
        for o in std:
            if abs(o["len_px"] - med) / med <= 0.35:
                o["len_px"] = med
    return merged + others


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
    thin_wall = thinnest_wall(ink_clean, ws, h, w)
    log.info(f"  Thinnest walls ~ {thin_wall}px")
    wmask = wall_mask_from_ink(ink, thickness, thin_wall)
    barrier = cv2.bitwise_or(wmask, long_thin_lines(ink_clean, wmask, thickness))

    walls_px = vectorize_walls(wmask, thickness, h, w)
    diag = diagonal_walls(barrier, thickness, h, w)
    seal = build_seal(barrier, walls_px, thickness, h, w, diag)

    # room labels: the page pass plus sparse-text passes, lexicon-matched
    label_words = words + ocr_label_words(grey_orig, ws, h, w)
    phrases = label_phrases(label_words)
    label_points = [(ph["cx"], ph["cy"]) for ph in phrases
                    if _lexicon_match(ph["text"])]

    # the same seal from walls alone (no dimension or other thin lines)
    wall_seal = build_seal(wmask, walls_px, thickness, h, w, diag)
    labels, n_rooms, footprint = segment_rooms(seal, wmask, ink_clean,
                                                label_points, thickness, h, w,
                                                wall_seal=wall_seal)
    # a room's own printed name (and its dimension line) must not leave a
    # bite out of its floor
    labels = fill_label_boxes(labels, label_words, wmask, h, w)
    polys = polygonize_rooms(labels, n_rooms, thickness, h, w)

    name_rooms(polys, phrases)

    # outdoor areas (patio / porch): label-seeded, added AFTER naming so
    # their pre-set names aren't clobbered
    outdoor = detect_outdoor_areas(grey, ink_clean, seal, footprint, labels,
                                   phrases, polys, label_words, thickness, h, w)
    polys.extend(outdoor)
    # light ink: glazing and door swings are often drawn grey, which the
    # Otsu ink mask drops
    light = light_ink(grey)
    openings_px, bridges = detect_openings(walls_px, wmask, ink_clean,
                                           thickness, h, w, light=light,
                                           ink_raw=ink, thin_wall=thin_wall)
    openings_px = _regularize_doors(openings_px, thickness)
    walls_px = chain_output_walls(walls_px, openings_px, thickness, h, w,
                                  wall_mask=wmask, bridges=bridges,
                                  ink_clean=ink_clean, light=light,
                                  thin_wall=thin_wall)
    openings_px = windows_inside_walls(walls_px, openings_px, grey,
                                       thickness, h, w)

    # names + validated dimensions from high-res per-room crops
    scale_m = recover_room_details(polys, grey_orig, ws, h, w)
    # name again with the labels the per-room crops read (thin or small
    # lettering the page passes missed)
    name_rooms(polys, phrases + [ph for p in polys for ph in p.get("crop_phrases", [])])
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
        # hinged | sliding | opening (doors only): the 3D already renders a
        # door wider than ~1.3 m as sliding glass; this makes it explicit
        **({"style": o["style"]} if o.get("style") else {}),
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