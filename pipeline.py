"""
Floor Plan Analysis Pipeline — OpenCV + Tesseract
Improved room detection using morphological reconstruction.
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
    Key insight: floor plan walls are thin lines. Standard morphological 
    closing merges adjacent rooms because it erases thin walls.
    
    Solution: use SMALL closing kernel (just to seal tiny gaps at wall 
    endpoints) then use WATERSHED or distance-transform to properly 
    separate touching regions.
    """
    
    # Step 1: Seal only tiny endpoint gaps (kernel=3, not 13)
    k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sealed = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, k_small)
    
    # Step 2: Floor = not wall, remove border
    floor = cv2.bitwise_not(sealed)
    margin = max(5, min(w,h)//40)
    floor[:margin,:]=0; floor[-margin:,:]=0
    floor[:,:margin]=0; floor[:,-margin:]=0
    
    # Step 3: Distance transform — finds room centers (peaks = far from walls)
    dist = cv2.distanceTransform(floor, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    
    # Step 4: Threshold distance to find "sure foreground" (room centers)
    # Use a lower threshold to find more room seeds
    _, sure_fg = cv2.threshold(dist, 0.15, 1.0, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg * 255)
    
    # Step 5: Find connected room seeds
    n, markers, stats, centroids = cv2.connectedComponentsWithStats(sure_fg, 8)
    
    min_area = w * h * 0.002  # lower threshold to catch small rooms
    max_area = w * h * 0.55
    
    regions = []
    for lbl in range(1, n):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        x  = stats[lbl, cv2.CC_STAT_LEFT]
        y  = stats[lbl, cv2.CC_STAT_TOP]
        bw = stats[lbl, cv2.CC_STAT_WIDTH]
        bh = stats[lbl, cv2.CC_STAT_HEIGHT]
        
        # Skip very thin/elongated regions (not rooms)
        if bw < 5 or bh < 5:
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
    
    # If too many (>20), watershed is over-segmenting — merge nearby small ones
    if len(regions) > 20:
        regions = _merge_nearby_regions(regions, w, h)
    
    # If still 0, fall back to contour approach
    if len(regions) == 0:
        regions = _contour_fallback(wall_mask, h, w)
    
    log.info(f"  Room regions: {len(regions)}")
    return regions[:15]  # cap at 15


def _merge_nearby_regions(regions: list, w: int, h: int) -> list:
    """Merge regions whose centroids are very close together."""
    DIST_THRESH = max(w, h) * 0.05  # 5% of image
    merged_flags = set()
    result = []
    for i, a in enumerate(regions):
        if i in merged_flags:
            continue
        group = [a]
        for j, b in enumerate(regions):
            if j <= i or j in merged_flags:
                continue
            dist = np.hypot(a["centroid_x"]-b["centroid_x"],
                           a["centroid_y"]-b["centroid_y"])
            if dist < DIST_THRESH:
                group.append(b)
                merged_flags.add(j)
        # Use the largest region in the group
        group.sort(key=lambda r: r["area_px"], reverse=True)
        result.append(group[0])
        merged_flags.add(i)
    return result


def _contour_fallback(wall_mask, h, w) -> list:
    """Contour-based room detection as final fallback."""
    min_area = w * h * 0.003
    max_area = w * h * 0.55
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, k)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    if hierarchy is None:
        return regions
    for i, cnt in enumerate(contours):
        # Only use inner contours (holes in walls = rooms)
        if hierarchy[0][i][3] == -1:  # no parent = outer contour, skip
            continue
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        regions.append({
            "box": {
                "top":    round(y/h*100, 3),
                "left":   round(x/w*100, 3),
                "width":  round(bw/w*100, 3),
                "height": round(bh/h*100, 3),
            },
            "area_px":    int(area),
            "centroid_x": float(M["m10"]/M["m00"]),
            "centroid_y": float(M["m01"]/M["m00"]),
            "pixel_x": x, "pixel_y": y,
            "pixel_w": bw, "pixel_h": bh,
        })
    regions.sort(key=lambda r: r["area_px"], reverse=True)
    return regions

# ── 4. OCR room naming ────────────────────────────────────────────────────────

def ocr_room_names(grey, regions: list, h: int, w: int) -> list:
    try:
        import pytesseract
        tess_ok = True
    except ImportError:
        tess_ok = False
        log.warning("pytesseract not available")

    scale = max(1.0, 1500/max(w,h))
    if tess_ok:
        ocr_img = cv2.resize(cv2.bitwise_not(grey), None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC) if scale > 1.2 else cv2.bitwise_not(grey)
    else:
        ocr_img = None

    named, used_names = [], {}
    for idx, region in enumerate(regions):
        raw_name = ""
        if tess_ok and ocr_img is not None:
            pad = 10
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

    # Doors — arc detection with strict wall proximity
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

    # Windows — strict: must be on wall, correct size range
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
            seg_len = st[lbl, cv2.CC_STAT_WIDTH] if orient=="horizontal" else st[lbl, cv2.CC_STAT_HEIGHT]
            if seg_len < min_win or seg_len > max_win: continue
            if st[lbl, cv2.CC_STAT_AREA] < 15: continue
            icx, icy = int(ct[lbl][0]), int(ct[lbl][1])
            # Must sit directly on a wall pixel
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