"""
Floor Plan Analysis Pipeline — OpenCV + Tesseract, no external AI
"""
import logging, re, cv2, numpy as np

log = logging.getLogger(__name__)

ROOM_COLORS = {
    "living":"#c3f4f0","great room":"#c3f4f0","kitchen":"#b9eac5",
    "bedroom":"#87ddd7","master":"#6dd0c4","bath":"#f7dfad",
    "hall":"#d5dbda","corridor":"#d5dbda","storage":"#ffc9c0",
    "closet":"#ffc9c0","wic":"#ffc9c0","w.i.c":"#ffc9c0",
    "dining":"#c7d2fe","study":"#fde68a","office":"#fde68a",
    "porch":"#a7f3d0","patio":"#a7f3d0","balcony":"#a7f3d0",
    "stair":"#e0c3fc","laundry":"#ffc9c0","garage":"#e5e7eb",
    "elevator":"#e5e7eb","powder":"#f7dfad","pwdr":"#f7dfad",
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

    # Otsu threshold
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Walls = white on black
    wall_mask = cv2.bitwise_not(binary) if np.mean(binary) > 127 else binary.copy()

    # Remove salt-and-pepper noise
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, k2)

    return bgr, wall_mask, grey, h, w

# ── 2. Wall detection ─────────────────────────────────────────────────────────

def detect_walls(wall_mask, h: int, w: int) -> list:
    edges = cv2.Canny(wall_mask, 50, 150, apertureSize=3)
    min_len = max(w, h) * 0.04
    max_gap = max(w, h) * 0.02

    raw = cv2.HoughLinesP(edges, 1, np.pi/180, 40,
                          minLineLength=int(min_len),
                          maxLineGap=int(max_gap))
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

# ── 3. Room region detection (wall-drawing approach) ──────────────────────────

def detect_room_regions(wall_mask, h: int, w: int) -> list:
    """
    Draw detected Hough walls onto a blank canvas and flood-fill enclosed areas.
    This is far more robust than using the raw binary mask directly, because
    floor plan images contain text, dimensions, and symbols that break connectivity.
    """
    # Get Hough lines
    edges = cv2.Canny(wall_mask, 50, 150, apertureSize=3)
    min_len = max(w, h) * 0.035
    max_gap = max(w, h) * 0.025

    raw = cv2.HoughLinesP(edges, 1, np.pi/180, 35,
                          minLineLength=int(min_len),
                          maxLineGap=int(max_gap))

    # Draw walls onto blank canvas with thick strokes to close gaps
    wall_canvas = np.zeros((h, w), dtype=np.uint8)
    if raw is not None:
        for line in raw:
            x1, y1, x2, y2 = line[0]
            # Line thickness proportional to wall width in image
            length = np.hypot(x2-x1, y2-y1)
            thick = max(3, int(length * 0.03))
            cv2.line(wall_canvas, (x1, y1), (x2, y2), 255, thick)

    # Also include the original wall mask (scaled down) for thin walls Hough missed
    orig_thin = cv2.dilate(wall_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)))
    combined = cv2.bitwise_or(wall_canvas, orig_thin)

    # Aggressive morphological closing to seal small gaps between wall segments
    for ksize in [5, 9, 13]:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)

    # Floor = everything NOT covered by walls
    floor = cv2.bitwise_not(combined)

    # Remove image border — not a room
    margin = max(8, min(w,h)//35)
    floor[:margin,:]=0; floor[-margin:,:]=0
    floor[:,:margin]=0; floor[:,-margin:]=0

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(floor, 8)
    min_area = w * h * 0.004   # at least 0.4% of image
    max_area = w * h * 0.55    # at most 55% (prevents background)

    regions = []
    for lbl in range(1, n):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        x  = stats[lbl, cv2.CC_STAT_LEFT]
        y  = stats[lbl, cv2.CC_STAT_TOP]
        bw = stats[lbl, cv2.CC_STAT_WIDTH]
        bh = stats[lbl, cv2.CC_STAT_HEIGHT]
        # Skip very elongated thin regions (corridors of dimension lines)
        aspect = max(bw,bh) / max(min(bw,bh), 1)
        if aspect > 15:
            continue
        regions.append({
            "box": {
                "top":    round(y/h*100, 3),
                "left":   round(x/w*100, 3),
                "width":  round(bw/w*100, 3),
                "height": round(bh/h*100, 3),
            },
            "area_px":    int(area),
            "centroid_x": float(centroids[lbl][0]),
            "centroid_y": float(centroids[lbl][1]),
            "pixel_x": int(x), "pixel_y": int(y),
            "pixel_w": int(bw), "pixel_h": int(bh),
        })

    regions.sort(key=lambda r: r["area_px"], reverse=True)
    log.info(f"  Room regions: {len(regions)}")
    return regions

# ── 4. OCR room naming ────────────────────────────────────────────────────────

def ocr_room_names(grey, regions: list, h: int, w: int) -> list:
    try:
        import pytesseract
        tess_ok = True
    except ImportError:
        log.warning("pytesseract not available — using heuristic names")
        tess_ok = False

    # Prepare OCR image — high contrast, upscaled
    ocr_img = cv2.bitwise_not(grey)
    scale = max(1.0, 1500/max(w, h))
    if scale > 1.2:
        ocr_img = cv2.resize(ocr_img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)

    named = []
    used_names: dict = {}

    for idx, region in enumerate(regions):
        raw_name = ""

        if tess_ok:
            pad = 8
            px = max(0, int(region["pixel_x"]*scale) - pad)
            py = max(0, int(region["pixel_y"]*scale) - pad)
            pw = min(ocr_img.shape[1], int((region["pixel_x"]+region["pixel_w"])*scale) + pad)
            ph = min(ocr_img.shape[0], int((region["pixel_y"]+region["pixel_h"])*scale) + pad)
            crop = ocr_img[py:ph, px:pw]
            if crop.size > 0:
                try:
                    import pytesseract
                    text = pytesseract.image_to_string(
                        crop,
                        config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ./-'
                    )
                    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 2]
                    # Filter dimension strings like "14.7 x 16" or "9 CLG"
                    lines = [l for l in lines if not re.match(r'^[\d\s\'.\"xX×\/]+$', l)]
                    lines = [l for l in lines if not re.search(r'\d+\s*[xX×]\s*\d+', l)]
                    if lines:
                        raw_name = " ".join(lines[:2]).strip()
                        raw_name = " ".join(word.capitalize() for word in raw_name.split())
                except Exception as e:
                    log.debug(f"OCR error region {idx}: {e}")

        if not raw_name or len(raw_name) < 2:
            raw_name = _guess_room_type(region, idx)

        # Ensure unique names
        base = raw_name
        if base in used_names:
            used_names[base] += 1
            raw_name = f"{base} {used_names[base]}"
        else:
            used_names[base] = 1

        named.append({
            **region,
            "name":       raw_name,
            "confidence": 85 if tess_ok else 60,
            "color":      room_color(raw_name, idx),
        })

    log.info(f"  OCR naming: {len(named)} rooms")
    return named

def _guess_room_type(region: dict, idx: int) -> str:
    area = region["area_px"]
    if area > 50000: return "Living Room"
    if area > 25000: return ["Bedroom","Dining Room","Great Room"][idx%3]
    if area > 10000: return ["Kitchen","Study","Bedroom"][idx%3]
    if area > 4000:  return ["Bathroom","Hallway","Closet"][idx%3]
    return f"Room {idx+1}"

# ── 5. Opening detection ──────────────────────────────────────────────────────

def detect_openings(wall_mask, h: int, w: int) -> list:
    openings = []

    # ── Doors: arc (quarter-circle) detection ──
    # Floor plan doors show as arc symbols — detect circles then filter by
    # proximity to a wall and by the arc being only partial (not full circle)
    blurred = cv2.GaussianBlur(wall_mask, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(w,h)*0.04,
        param1=50, param2=20,
        minRadius=int(max(w,h)*0.025),
        maxRadius=int(max(w,h)*0.12),
    )
    if circles is not None:
        for cx, cy, r in np.round(circles[0]).astype(int):
            # Must be near a wall
            x0,y0 = max(0,cx-r-4), max(0,cy-r-4)
            x1b,y1b = min(w,cx+r+4), min(h,cy+r+4)
            region = wall_mask[y0:y1b, x0:x1b]
            wall_density = float(np.mean(region>128))
            if wall_density < 0.08:
                continue
            # Determine wall orientation
            hd = float(np.mean(wall_mask[max(0,cy-3):min(h,cy+3), max(0,cx-r):min(w,cx+r)] > 128))
            vd = float(np.mean(wall_mask[max(0,cy-r):min(h,cy+r), max(0,cx-3):min(w,cx+3)] > 128))
            orient = "horizontal" if hd >= vd else "vertical"
            door_w = max(80.0, round(r*2/max(w,h)*1000, 2))
            openings.append({
                "type":  "door",
                "wall":  orient,
                "x":     round(cx/w*1000, 2),
                "y":     round(cy/h*1000, 2),
                "width": door_w,
            })

    # ── Windows: gaps in exterior walls ──
    # Windows appear as short parallel lines crossing a wall.
    # Use morphology to find small horizontal/vertical segments on wall edges.
    edges = cv2.Canny(wall_mask, 50, 150, apertureSize=3)

    # Only look for segments significantly shorter than room-spanning walls
    min_win = max(8, min(w,h)//50)
    max_win = max(w,h)//8

    for kern, orient in [
        (cv2.getStructuringElement(cv2.MORPH_RECT, (min_win, 1)), "horizontal"),
        (cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_win)), "vertical"),
    ]:
        m = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kern)
        n, _, st, ct = cv2.connectedComponentsWithStats(m, 8)
        for lbl in range(1, n):
            seg_w = st[lbl, cv2.CC_STAT_WIDTH]
            seg_h = st[lbl, cv2.CC_STAT_HEIGHT]
            area  = st[lbl, cv2.CC_STAT_AREA]
            # Must be short segment (window symbol, not a wall)
            seg_len = seg_w if orient=="horizontal" else seg_h
            if seg_len < min_win or seg_len > max_win:
                continue
            if area < 10:
                continue
            cx, cy = ct[lbl]
            # Must sit on an actual wall pixel
            icx, icy = int(cx), int(cy)
            if wall_mask[max(0,icy-2):min(h,icy+2), max(0,icx-2):min(w,icx+2)].max() < 128:
                continue
            openings.append({
                "type":  "window",
                "wall":  orient,
                "x":     round(float(cx)/w*1000, 2),
                "y":     round(float(cy)/h*1000, 2),
                "width": max(60.0, round(float(seg_len)/max(w,h)*1000, 2)),
            })

    openings = _dedup_openings(openings)
    log.info(f"  Openings: {len(openings)} ({sum(1 for o in openings if o['type']=='door')} doors, {sum(1 for o in openings if o['type']=='window')} windows)")
    return openings

def _dedup_openings(openings: list) -> list:
    THRESH = 25.0
    unique = []
    for op in openings:
        if not any(np.hypot(op["x"]-u["x"], op["y"]-u["y"]) < THRESH for u in unique):
            unique.append(op)
    return unique

# ── 6. Build rooms ────────────────────────────────────────────────────────────

def build_rooms(named_regions: list, project_id: str) -> list:
    return [{
        "id":         f"{project_id}-r{i+1}",
        "name":       r["name"],
        "confidence": r["confidence"],
        "color":      r["color"],
        "box":        r["box"],
    } for i, r in enumerate(named_regions)]

# ── Main ──────────────────────────────────────────────────────────────────────

async def analyse_floor_plan(
    image_bytes: bytes,
    image_url: str,
    project_id: str,
    gemini_api_key: str = "",
) -> dict:
    bgr, wall_mask, grey, h, w = preprocess(image_bytes)
    log.info(f"Image: {w}×{h}px  project={project_id}")

    walls    = detect_walls(wall_mask, h, w)
    regions  = detect_room_regions(wall_mask, h, w)
    named    = ocr_room_names(grey, regions, h, w)
    rooms    = build_rooms(named, project_id)
    openings = detect_openings(wall_mask, h, w)

    return {
        "rooms":      rooms,
        "walls":      walls,
        "openings":   openings,
        "image_size": {"width": w, "height": h},
    }