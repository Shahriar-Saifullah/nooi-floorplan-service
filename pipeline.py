"""
Floor Plan Analysis Pipeline — OpenCV + Tesseract
Uses adaptive thresholds and multiple detection strategies.
"""
import logging, re, cv2, numpy as np

log = logging.getLogger(__name__)

ROOM_COLORS = {
    "living":"#c3f4f0","great":"#c3f4f0","kitchen":"#b9eac5",
    "bedroom":"#87ddd7","master":"#6dd0c4","bath":"#f7dfad",
    "hall":"#d5dbda","corridor":"#d5dbda","storage":"#ffc9c0",
    "closet":"#ffc9c0","wic":"#ffc9c0","dining":"#c7d2fe",
    "study":"#fde68a","office":"#fde68a","porch":"#a7f3d0",
    "patio":"#a7f3d0","balcony":"#a7f3d0","stair":"#e0c3fc",
    "laundry":"#ffc9c0","garage":"#e5e7eb","elevator":"#e5e7eb",
    "powder":"#f7dfad","pwdr":"#f7dfad","covered":"#a7f3d0",
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
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, k2)
    return bgr, wall_mask, grey, h, w

# ── 2. Wall detection ─────────────────────────────────────────────────────────

def detect_walls(wall_mask, h: int, w: int) -> list:
    edges = cv2.Canny(wall_mask, 50, 150, apertureSize=3)
    # Use lower threshold to catch more wall segments
    min_len = max(w, h) * 0.025
    max_gap = max(w, h) * 0.025

    raw = cv2.HoughLinesP(edges, 1, np.pi/180, 30,
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

# ── 3. Room region detection ──────────────────────────────────────────────────

def detect_room_regions(wall_mask, h: int, w: int) -> list:
    """
    Multi-strategy approach:
    1. Try standard flood-fill with progressively larger closing kernels
    2. If that fails, draw Hough walls onto canvas and flood-fill
    3. If that still fails, use contour-based region detection
    Always returns at least the largest candidate regions found.
    """
    # Strategy 1: Progressive closing on raw wall mask
    regions = _flood_fill_strategy(wall_mask, h, w, source="raw")
    if len(regions) >= 2:
        log.info(f"  Room regions: {len(regions)} (strategy: raw mask)")
        return regions

    # Strategy 2: Draw Hough lines onto blank canvas then flood-fill
    edges = cv2.Canny(wall_mask, 30, 120, apertureSize=3)
    canvas = np.zeros((h, w), np.uint8)
    raw = cv2.HoughLinesP(edges, 1, np.pi/180, 25,
                          minLineLength=int(max(w,h)*0.02),
                          maxLineGap=int(max(w,h)*0.03))
    if raw is not None:
        for line in raw:
            x1,y1,x2,y2 = line[0]
            cv2.line(canvas, (x1,y1), (x2,y2), 255, 4)
    combined = cv2.bitwise_or(canvas, wall_mask)
    regions = _flood_fill_strategy(combined, h, w, source="hough_canvas")
    if len(regions) >= 2:
        log.info(f"  Room regions: {len(regions)} (strategy: Hough canvas)")
        return regions

    # Strategy 3: Contour-based — find large closed contours in the wall image
    regions = _contour_strategy(wall_mask, h, w)
    if regions:
        log.info(f"  Room regions: {len(regions)} (strategy: contours)")
        return regions

    # Strategy 4: Last resort — divide image into grid based on wall line intersections
    regions = _grid_strategy(wall_mask, h, w)
    log.info(f"  Room regions: {len(regions)} (strategy: grid fallback)")
    return regions


def _flood_fill_strategy(mask, h, w, source="") -> list:
    """Try progressively larger closing kernels until rooms appear."""
    min_area = w * h * 0.003
    max_area = w * h * 0.55
    best_regions = []

    for ksize in [5, 9, 13, 19, 27]:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        # Also dilate to connect nearby walls
        closed = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        floor = cv2.bitwise_not(closed)
        margin = max(5, min(w,h)//40)
        floor[:margin,:]=0; floor[-margin:,:]=0
        floor[:,:margin]=0; floor[:,-margin:]=0

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(floor, 8)
        regions = []
        for lbl in range(1, n):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < min_area or area > max_area:
                continue
            x  = stats[lbl, cv2.CC_STAT_LEFT]
            y  = stats[lbl, cv2.CC_STAT_TOP]
            bw = stats[lbl, cv2.CC_STAT_WIDTH]
            bh = stats[lbl, cv2.CC_STAT_HEIGHT]
            aspect = max(bw,bh) / max(min(bw,bh), 1)
            if aspect > 12: continue
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
        if len(regions) > len(best_regions):
            best_regions = regions
        if len(regions) >= 2:
            return regions

    return best_regions


def _contour_strategy(wall_mask, h, w) -> list:
    """Find closed contours as room boundaries."""
    min_area = w * h * 0.005
    max_area = w * h * 0.55
    # Close gaps in walls
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx = M["m10"]/M["m00"]
        cy = M["m01"]/M["m00"]
        regions.append({
            "box": {
                "top":    round(y/h*100, 3),
                "left":   round(x/w*100, 3),
                "width":  round(bw/w*100, 3),
                "height": round(bh/h*100, 3),
            },
            "area_px": int(area),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "pixel_x": x, "pixel_y": y,
            "pixel_w": bw, "pixel_h": bh,
        })
    regions.sort(key=lambda r: r["area_px"], reverse=True)
    # Remove regions that contain other regions (keep children not parents)
    filtered = []
    for i, r in enumerate(regions):
        is_parent = any(
            j != i and
            regions[j]["box"]["left"]  >= r["box"]["left"] and
            regions[j]["box"]["top"]   >= r["box"]["top"] and
            regions[j]["box"]["left"] + regions[j]["box"]["width"]  <= r["box"]["left"] + r["box"]["width"] and
            regions[j]["box"]["top"]  + regions[j]["box"]["height"] <= r["box"]["top"]  + r["box"]["height"]
            for j in range(len(regions))
        )
        if not is_parent:
            filtered.append(r)
    return filtered[:15]


def _grid_strategy(wall_mask, h, w) -> list:
    """
    Last resort: find horizontal and vertical wall lines, use their
    intersections to infer room grid cells.
    """
    edges = cv2.Canny(wall_mask, 50, 150, apertureSize=3)
    raw = cv2.HoughLinesP(edges, 1, np.pi/180, 25,
                          minLineLength=int(max(w,h)*0.08),
                          maxLineGap=int(max(w,h)*0.04))
    if raw is None:
        return []

    h_lines, v_lines = [], []
    for line in raw:
        x1,y1,x2,y2 = line[0]
        if abs(x2-x1) > abs(y2-y1):
            h_lines.append(sorted([y1,y2]))
        else:
            v_lines.append(sorted([x1,x2]))

    if not h_lines or not v_lines:
        return []

    # Cluster line positions
    def cluster(vals, gap=h*0.04):
        if not vals: return []
        vals = sorted(set(v[0] for v in vals))
        clusters, cur = [[vals[0]]], vals[0]
        for v in vals[1:]:
            if v - cur < gap: clusters[-1].append(v)
            else: clusters.append([v]); cur = v
        return [sum(c)//len(c) for c in clusters]

    hy = cluster(h_lines, h*0.04)
    vx = cluster(v_lines, w*0.04)

    if len(hy) < 2 or len(vx) < 2:
        return []

    # Each cell between consecutive lines is a potential room
    min_area = w * h * 0.003
    regions = []
    for i in range(len(hy)-1):
        for j in range(len(vx)-1):
            x, y = vx[j], hy[i]
            bw = vx[j+1] - vx[j]
            bh = hy[i+1]  - hy[i]
            area = bw * bh
            if area < min_area: continue
            regions.append({
                "box": {
                    "top":    round(y/h*100, 3),
                    "left":   round(x/w*100, 3),
                    "width":  round(bw/w*100, 3),
                    "height": round(bh/h*100, 3),
                },
                "area_px":    area,
                "centroid_x": float(x + bw/2),
                "centroid_y": float(y + bh/2),
                "pixel_x": x, "pixel_y": y,
                "pixel_w": bw, "pixel_h": bh,
            })

    regions.sort(key=lambda r: r["area_px"], reverse=True)
    return regions[:12]

# ── 4. OCR room naming ────────────────────────────────────────────────────────

def ocr_room_names(grey, regions: list, h: int, w: int) -> list:
    try:
        import pytesseract
        tess_ok = True
    except ImportError:
        tess_ok = False
        log.warning("pytesseract not available")

    scale = max(1.0, 1500/max(w,h))
    if tess_ok and scale > 1.2:
        ocr_img = cv2.resize(cv2.bitwise_not(grey), None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
    elif tess_ok:
        ocr_img = cv2.bitwise_not(grey)
    else:
        ocr_img = None

    named, used_names = [], {}
    for idx, region in enumerate(regions):
        raw_name = ""
        if tess_ok and ocr_img is not None:
            pad = 8
            px = max(0, int(region["pixel_x"]*scale)-pad)
            py = max(0, int(region["pixel_y"]*scale)-pad)
            pw = min(ocr_img.shape[1], int((region["pixel_x"]+region["pixel_w"])*scale)+pad)
            ph = min(ocr_img.shape[0], int((region["pixel_y"]+region["pixel_h"])*scale)+pad)
            crop = ocr_img[py:ph, px:pw]
            if crop.size > 0:
                try:
                    import pytesseract
                    text = pytesseract.image_to_string(
                        crop,
                        config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ./-'
                    )
                    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 1]
                    lines = [l for l in lines if not re.match(r'^[\d\s\'.\"xX×\/\-]+$', l)]
                    lines = [l for l in lines if not re.search(r'\d+[\s]*[xX×][\s]*\d+', l)]
                    lines = [l for l in lines if len(l) > 1]
                    if lines:
                        raw_name = " ".join(lines[:2]).strip()
                        raw_name = " ".join(word.capitalize() for word in raw_name.split())
                except Exception as e:
                    log.debug(f"OCR error {idx}: {e}")

        if not raw_name or len(raw_name) < 2:
            raw_name = _guess_room_type(region, idx)

        base = raw_name
        if base in used_names:
            used_names[base] += 1
            raw_name = f"{base} {used_names[base]}"
        else:
            used_names[base] = 1

        named.append({**region, "name": raw_name,
                      "confidence": 85 if tess_ok else 60,
                      "color": room_color(raw_name, idx)})

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

    # Doors — arc detection
    blurred = cv2.GaussianBlur(wall_mask, (5,5), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(w,h)*0.04, param1=50, param2=20,
        minRadius=int(max(w,h)*0.025), maxRadius=int(max(w,h)*0.12),
    )
    if circles is not None:
        for cx, cy, r in np.round(circles[0]).astype(int):
            x0,y0 = max(0,cx-r-4), max(0,cy-r-4)
            x1b,y1b = min(w,cx+r+4), min(h,cy+r+4)
            if float(np.mean(wall_mask[y0:y1b,x0:x1b]>128)) < 0.08:
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

    # Windows — short parallel line segments on walls
    edges = cv2.Canny(wall_mask, 50, 150, apertureSize=3)
    min_win = max(8, min(w,h)//50)
    max_win = max(w,h)//8

    for kern, orient in [
        (cv2.getStructuringElement(cv2.MORPH_RECT, (min_win,1)), "horizontal"),
        (cv2.getStructuringElement(cv2.MORPH_RECT, (1,min_win)), "vertical"),
    ]:
        m = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kern)
        n, _, st, ct = cv2.connectedComponentsWithStats(m, 8)
        for lbl in range(1, n):
            seg_len = st[lbl, cv2.CC_STAT_WIDTH] if orient=="horizontal" else st[lbl, cv2.CC_STAT_HEIGHT]
            if seg_len < min_win or seg_len > max_win: continue
            if st[lbl, cv2.CC_STAT_AREA] < 10: continue
            cx, cy = int(ct[lbl][0]), int(ct[lbl][1])
            # Must be on a wall pixel
            if wall_mask[max(0,cy-2):min(h,cy+2), max(0,cx-2):min(w,cx+2)].max() < 128:
                continue
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
        if not any(np.hypot(op["x"]-u["x"], op["y"]-u["y"]) < 25.0 for u in unique):
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