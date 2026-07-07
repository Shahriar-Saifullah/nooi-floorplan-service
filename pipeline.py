"""
Floor Plan Analysis Pipeline
-----------------------------
Architecture:
  1. Tesseract OCR  — finds room label text positions in the image
  2. Wall network   — OpenCV Hough lines for wall segments  
  3. Room boxes     — built from OCR text positions + wall intersections
  4. Door detection — arc symbols via Hough circles
  5. Window detect  — parallel line patterns on walls

No external AI APIs. Pure OpenCV + Tesseract.
"""

import logging
import re
import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── Color palette ─────────────────────────────────────────────────────────────

ROOM_COLORS = {
    "living":     "#c3f4f0",
    "great":      "#c3f4f0",
    "kitchen":    "#b9eac5",
    "bedroom":    "#87ddd7",
    "master":     "#6dd0c4",
    "bath":       "#f7dfad",
    "hall":       "#d5dbda",
    "corridor":   "#d5dbda",
    "storage":    "#ffc9c0",
    "closet":     "#ffc9c0",
    "wic":        "#ffc9c0",
    "dining":     "#c7d2fe",
    "study":      "#fde68a",
    "office":     "#fde68a",
    "porch":      "#a7f3d0",
    "patio":      "#a7f3d0",
    "balcony":    "#a7f3d0",
    "stair":      "#e0c3fc",
    "laundry":    "#ffc9c0",
    "garage":     "#e5e7eb",
    "elevator":   "#e5e7eb",
    "powder":     "#f7dfad",
    "pwdr":       "#f7dfad",
    "covered":    "#a7f3d0",
}
FALLBACK_COLORS = [
    "#c3f4f0","#b9eac5","#87ddd7","#f7dfad","#d5dbda","#ffc9c0",
    "#c7d2fe","#fde68a","#a7f3d0","#e0c3fc","#6dd0c4","#e5e7eb",
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
    h, w = bgr.shape[:2]
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    wall_mask = cv2.bitwise_not(binary) if np.mean(binary) > 127 else binary.copy()
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, k)
    return bgr, wall_mask, grey, h, w


# ── 2. OCR — find room labels and their positions ─────────────────────────────

def ocr_find_rooms(grey, h: int, w: int) -> list:
    """
    Use Tesseract to find all text blocks in the floor plan.
    Returns list of {text, cx, cy, bx, by, bw, bh} for each text label
    that looks like a room name (not a dimension like "14.7 x 16").
    """
    try:
        import pytesseract
    except ImportError:
        log.warning("pytesseract not available")
        return []

    # Upscale for better OCR accuracy
    scale = max(1.5, 1800 / max(w, h))
    upscaled = cv2.resize(grey, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    # High contrast for OCR
    _, binary = cv2.threshold(upscaled, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Floor plans have dark text on white — ensure that
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    try:
        data = pytesseract.image_to_data(
            binary,
            config='--psm 11 --oem 3',
            output_type=pytesseract.Output.DICT,
        )
    except Exception as e:
        log.error(f"Tesseract error: {e}")
        return []

    rooms = []
    n = len(data['text'])
    i = 0
    while i < n:
        text = data['text'][i].strip()
        conf = int(data['conf'][i])

        if conf < 30 or len(text) < 2:
            i += 1
            continue

        # Skip dimension strings: "14.7", "x", "16", "9 CLG", etc.
        if re.match(r'^[\d\s\'.\"]+$', text):
            i += 1
            continue
        if re.match(r'^[xX×]$', text):
            i += 1
            continue

        # Try to merge adjacent words on the same line into a phrase
        # (room names like "MASTER BED RM" span multiple words)
        phrase_words = [text]
        phrase_conf  = [conf]
        bx = data['left'][i]
        by = data['top'][i]
        bw = data['width'][i]
        bh = data['height'][i]
        block = data['block_num'][i]
        line  = data['line_num'][i]

        j = i + 1
        while j < n:
            next_text = data['text'][j].strip()
            next_conf = int(data['conf'][j])
            if (data['block_num'][j] == block and
                    data['line_num'][j] == line and
                    next_conf >= 30 and len(next_text) >= 1):
                if not re.match(r'^[\d\s\'.\"xX×]+$', next_text):
                    phrase_words.append(next_text)
                    phrase_conf.append(next_conf)
                    bw = (data['left'][j] + data['width'][j]) - bx
            else:
                break
            j += 1

        # Also check next line if it's part of the same block (e.g. "MASTER BED RM\n14.7x16")
        full_text = " ".join(phrase_words).strip()

        # Filter: must be a meaningful room name (letters, not just numbers)
        if not re.search(r'[A-Za-z]{2,}', full_text):
            i = j
            continue

        # Filter common non-room text
        skip_words = ['clg', 'vault', 'plan', 'floor', 'scale', 'north',
                      'copyright', 'note', 'drawn', 'date', 'sheet']
        if any(sw in full_text.lower() for sw in skip_words):
            i = j
            continue

        # Convert bbox back to original image coordinates
        orig_bx = int(bx / scale)
        orig_by = int(by / scale)
        orig_bw = max(1, int(bw / scale))
        orig_bh = max(1, int(bh / scale))
        cx = orig_bx + orig_bw // 2
        cy = orig_by + orig_bh // 2

        rooms.append({
            "text": full_text,
            "cx": cx, "cy": cy,
            "bx": orig_bx, "by": orig_by,
            "bw": orig_bw, "bh": orig_bh,
            "conf": int(np.mean(phrase_conf)),
        })
        i = j

    # Deduplicate overlapping text detections
    rooms = _dedup_text_boxes(rooms)
    log.info(f"  OCR found {len(rooms)} room labels")
    for r in rooms:
        log.info(f"    '{r['text']}' at ({r['cx']}, {r['cy']})")
    return rooms


def _dedup_text_boxes(rooms: list) -> list:
    """Remove text detections whose centers are very close together."""
    unique = []
    for r in rooms:
        if not any(abs(r['cx']-u['cx']) < 30 and abs(r['cy']-u['cy']) < 20
                   for u in unique):
            unique.append(r)
    return unique


# ── 3. Build room boxes from OCR positions + wall network ─────────────────────

def build_room_boxes(ocr_rooms: list, wall_segs: list,
                     h: int, w: int, project_id: str) -> list:
    """
    For each OCR-detected room label:
    1. Start from the text center point
    2. Expand outward in all 4 directions until we hit a wall segment
    3. That gives us the room's bounding box

    Falls back to a box around the text label if wall expansion fails.
    """
    if not ocr_rooms:
        return []

    # Build a wall raster for boundary checking
    wall_raster = _rasterise_walls(wall_segs, h, w)

    result = []
    used_names: dict = {}

    for idx, room in enumerate(ocr_rooms):
        cx, cy = room['cx'], room['cy']

        # Expand from center outward until hitting a wall
        # Search in each direction with step size 1px
        left   = _expand(wall_raster, cx, cy, h, w, "left")
        right  = _expand(wall_raster, cx, cy, h, w, "right")
        top    = _expand(wall_raster, cx, cy, h, w, "up")
        bottom = _expand(wall_raster, cx, cy, h, w, "down")

        # Clamp to image bounds with margin
        margin = max(5, min(w, h) // 40)
        left   = max(margin, left)
        right  = min(w - margin, right)
        top    = max(margin, top)
        bottom = min(h - margin, bottom)

        bw = right - left
        bh = bottom - top

        # Sanity check — must be a reasonable room size
        min_dim = min(w, h) * 0.03
        if bw < min_dim or bh < min_dim:
            # Fall back to a box around the text label
            pad = max(20, min(w, h) // 15)
            left   = max(margin, cx - pad)
            right  = min(w - margin, cx + pad)
            top    = max(margin, cy - pad)
            bottom = min(h - margin, cy + pad)
            bw = right - left
            bh = bottom - top

        # Normalize name
        name = " ".join(word.capitalize() for word in room['text'].split())
        base = name
        if base in used_names:
            used_names[base] += 1
            name = f"{base} {used_names[base]}"
        else:
            used_names[base] = 1

        result.append({
            "id":         f"{project_id}-r{idx+1}",
            "name":       name,
            "confidence": min(95, room['conf']),
            "color":      room_color(name, idx),
            "box": {
                "top":    round(top  / h * 100, 3),
                "left":   round(left / w * 100, 3),
                "width":  round(bw   / w * 100, 3),
                "height": round(bh   / h * 100, 3),
            },
        })

    log.info(f"  Rooms built: {len(result)}")
    return result


def _rasterise_walls(wall_segs: list, h: int, w: int) -> np.ndarray:
    """Draw wall segments onto a blank canvas for boundary checking."""
    canvas = np.zeros((h, w), np.uint8)
    for seg in wall_segs:
        x1 = int(seg["x1"] / 100 * w)
        y1 = int(seg["y1"] / 100 * h)
        x2 = int(seg["x2"] / 100 * w)
        y2 = int(seg["y2"] / 100 * h)
        # Draw wall with thickness proportional to its detected thickness
        thick = max(2, int(seg.get("thickness", 0.5) / 100 * max(w, h)))
        cv2.line(canvas, (x1, y1), (x2, y2), 255, thick)
    # Also dilate slightly to close tiny gaps
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    canvas = cv2.dilate(canvas, k)
    return canvas


def _expand(raster: np.ndarray, cx: int, cy: int,
            h: int, w: int, direction: str) -> int:
    """Walk from (cx,cy) in given direction until hitting a wall pixel."""
    MAX_STEPS = max(w, h)  # safety limit
    x, y = cx, cy
    for _ in range(MAX_STEPS):
        if direction == "left":
            x -= 1
            if x < 0 or raster[max(0,min(h-1,y)), x] > 128:
                return x + 1
        elif direction == "right":
            x += 1
            if x >= w or raster[max(0,min(h-1,y)), x] > 128:
                return x - 1
        elif direction == "up":
            y -= 1
            if y < 0 or raster[y, max(0,min(w-1,x))] > 128:
                return y + 1
        elif direction == "down":
            y += 1
            if y >= h or raster[y, max(0,min(w-1,x))] > 128:
                return y - 1
    return x if direction in ("left", "right") else y


# ── 4. Wall detection ─────────────────────────────────────────────────────────

def detect_walls(wall_mask, h: int, w: int) -> list:
    edges = cv2.Canny(wall_mask, 50, 150, apertureSize=3)
    min_len = max(w, h) * 0.025
    raw = cv2.HoughLinesP(edges, 1, np.pi/180, 30,
                          minLineLength=int(min_len),
                          maxLineGap=int(max(w,h)*0.025))
    if raw is None:
        return []
    walls = []
    for line in raw:
        x1, y1, x2, y2 = line[0]
        if np.hypot(x2-x1, y2-y1) < min_len:
            continue
        thick = _wall_thickness(wall_mask, x1, y1, x2, y2, h, w)
        walls.append({
            "x1": round(x1/w*100, 3), "y1": round(y1/h*100, 3),
            "x2": round(x2/w*100, 3), "y2": round(y2/h*100, 3),
            "thickness": round(thick/max(w,h)*100, 3),
        })
    walls = _dedup_walls(walls)
    log.info(f"  Walls: {len(walls)} segments")
    return walls

def _wall_thickness(mask, x1, y1, x2, y2, h, w) -> float:
    mx, my = (x1+x2)//2, (y1+y2)//2
    dx, dy = x2-x1, y2-y1
    ln = max(1, np.hypot(dx, dy))
    px, py = -dy/ln, dx/ln
    count = 0
    for t in range(-20, 21):
        sx, sy = int(mx+px*t), int(my+py*t)
        if 0 <= sx < w and 0 <= sy < h and mask[sy, sx] > 128:
            count += 1
    return float(count)

def _dedup_walls(walls: list) -> list:
    DIST = 2.5
    merged, used = [], set()
    for i, a in enumerate(walls):
        if i in used: continue
        ah = abs(a["x2"]-a["x1"]) > abs(a["y2"]-a["y1"])
        grp = [a]; used.add(i)
        for j, b in enumerate(walls):
            if j in used: continue
            bh = abs(b["x2"]-b["x1"]) > abs(b["y2"]-b["y1"])
            if ah != bh: continue
            ap = (a["y1"]+a["y2"])/2 if ah else (a["x1"]+a["x2"])/2
            bp = (b["y1"]+b["y2"])/2 if bh else (b["x1"]+b["x2"])/2
            if abs(ap-bp) > DIST: continue
            amin = min(a["x1"],a["x2"]) if ah else min(a["y1"],a["y2"])
            amax = max(a["x1"],a["x2"]) if ah else max(a["y1"],a["y2"])
            bmin = min(b["x1"],b["x2"]) if bh else min(b["y1"],b["y2"])
            bmax = max(b["x1"],b["x2"]) if bh else max(b["y1"],b["y2"])
            ovlp = min(amax,bmax)-max(amin,bmin)
            if ovlp < min(amax-amin,bmax-bmin)*0.3: continue
            grp.append(b); used.add(j)
        if len(grp) == 1:
            merged.append(a)
        else:
            avg = sum((g["y1"]+g["y2"])/2 if ah else (g["x1"]+g["x2"])/2 for g in grp)/len(grp)
            mn = min(min(g["x1"],g["x2"]) if ah else min(g["y1"],g["y2"]) for g in grp)
            mx = max(max(g["x1"],g["x2"]) if ah else max(g["y1"],g["y2"]) for g in grp)
            tk = max(g["thickness"] for g in grp)
            if ah: merged.append({"x1":mn,"y1":avg,"x2":mx,"y2":avg,"thickness":tk})
            else:  merged.append({"x1":avg,"y1":mn,"x2":avg,"y2":mx,"thickness":tk})
    return merged


# ── 5. Opening detection ──────────────────────────────────────────────────────

def detect_openings(wall_mask, h: int, w: int) -> list:
    openings = []

    # Doors — Hough circles (arc symbols)
    blurred = cv2.GaussianBlur(wall_mask, (5,5), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(w,h)*0.05,
        param1=60, param2=25,
        minRadius=int(max(w,h)*0.025),
        maxRadius=int(max(w,h)*0.10),
    )
    if circles is not None:
        for cx, cy, r in np.round(circles[0]).astype(int):
            x0,y0 = max(0,cx-r-4), max(0,cy-r-4)
            x1b,y1b = min(w,cx+r+4), min(h,cy+r+4)
            if float(np.mean(wall_mask[y0:y1b,x0:x1b]>128)) < 0.10:
                continue
            hd = float(np.mean(wall_mask[max(0,cy-3):min(h,cy+3),max(0,cx-r):min(w,cx+r)]>128))
            vd = float(np.mean(wall_mask[max(0,cy-r):min(h,cy+r),max(0,cx-3):min(w,cx+3)]>128))
            openings.append({
                "type":  "door",
                "wall":  "horizontal" if hd >= vd else "vertical",
                "x":     round(cx/w*1000, 2),
                "y":     round(cy/h*1000, 2),
                "width": max(80.0, round(r*2/max(w,h)*1000, 2)),
            })

    # Windows — short parallel line segments directly on walls
    edges = cv2.Canny(wall_mask, 50, 150, apertureSize=3)
    min_win = max(10, min(w,h)//40)
    max_win = max(w,h)//10

    for kern, orient in [
        (cv2.getStructuringElement(cv2.MORPH_RECT, (min_win,1)), "horizontal"),
        (cv2.getStructuringElement(cv2.MORPH_RECT, (1,min_win)), "vertical"),
    ]:
        m = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kern)
        n, _, st, ct = cv2.connectedComponentsWithStats(m, 8)
        for lbl in range(1, n):
            seg_len = st[lbl,cv2.CC_STAT_WIDTH] if orient=="horizontal" else st[lbl,cv2.CC_STAT_HEIGHT]
            if seg_len < min_win or seg_len > max_win: continue
            if st[lbl,cv2.CC_STAT_AREA] < 15: continue
            icx,icy = int(ct[lbl][0]), int(ct[lbl][1])
            roi = wall_mask[max(0,icy-3):min(h,icy+3), max(0,icx-3):min(w,icx+3)]
            if roi.max() < 128: continue
            openings.append({
                "type":  "window",
                "wall":  orient,
                "x":     round(float(ct[lbl][0])/w*1000, 2),
                "y":     round(float(ct[lbl][1])/h*1000, 2),
                "width": max(60.0, round(float(seg_len)/max(w,h)*1000, 2)),
            })

    openings = _dedup_openings(openings)
    doors   = sum(1 for o in openings if o["type"]=="door")
    windows = sum(1 for o in openings if o["type"]=="window")
    log.info(f"  Openings: {len(openings)} ({doors} doors, {windows} windows)")
    return openings

def _dedup_openings(openings: list) -> list:
    unique = []
    for op in openings:
        if not any(np.hypot(op["x"]-u["x"],op["y"]-u["y"]) < 25.0 for u in unique):
            unique.append(op)
    return unique


# ── Main ──────────────────────────────────────────────────────────────────────

async def analyse_floor_plan(
    image_bytes: bytes,
    image_url: str,
    project_id: str,
    gemini_api_key: str = "",
) -> dict:
    bgr, wall_mask, grey, h, w = preprocess(image_bytes)
    log.info(f"Image: {w}×{h}px  project={project_id}")

    # Walls first — needed for room box expansion
    walls     = detect_walls(wall_mask, h, w)

    # OCR to find room labels and their positions
    ocr_rooms = ocr_find_rooms(grey, h, w)

    # Build room boxes by expanding from OCR text positions until hitting walls
    rooms     = build_room_boxes(ocr_rooms, walls, h, w, project_id)

    # Openings
    openings  = detect_openings(wall_mask, h, w)

    return {
        "rooms":      rooms,
        "walls":      walls,
        "openings":   openings,
        "image_size": {"width": w, "height": h},
    }