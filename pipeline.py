"""
Floor Plan Analysis Pipeline
-----------------------------
Architecture (final):
  Step 1: Roboflow    — precise wall segments, doors, windows (pixel coords)
  Step 2: Gemini      — room names + dimensions ONLY (not geometry)
  Step 3: OpenCV      — room regions from wall network
  Step 4: Match       — pair Gemini names to OpenCV regions by centroid
  Step 5: Normalise   — all coords as 0-100% of image dimensions

Gemini is used ONLY for what it's good at: reading text and dimensions.
OpenCV is used ONLY for what it's good at: pixel geometry.
Roboflow is used ONLY for what it's good at: trained wall/door/window detection.
"""

import asyncio
import base64
import json
import logging
import re
import cv2
import httpx
import numpy as np

log = logging.getLogger(__name__)

# ── Color palette ─────────────────────────────────────────────────────────────

ROOM_COLORS = {
    "living":    "#c3f4f0", "great":    "#c3f4f0",
    "kitchen":   "#b9eac5", "bedroom":  "#87ddd7",
    "master":    "#6dd0c4", "bath":     "#f7dfad",
    "hall":      "#d5dbda", "corridor": "#d5dbda",
    "storage":   "#ffc9c0", "closet":   "#ffc9c0",
    "wic":       "#ffc9c0", "dining":   "#c7d2fe",
    "study":     "#fde68a", "office":   "#fde68a",
    "porch":     "#a7f3d0", "patio":    "#a7f3d0",
    "balcony":   "#a7f3d0", "stair":    "#e0c3fc",
    "laundry":   "#ffc9c0", "garage":   "#e5e7eb",
    "elevator":  "#e5e7eb", "powder":   "#f7dfad",
    "covered":   "#a7f3d0", "pwdr":     "#f7dfad",
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
    return bgr, wall_mask, h, w

# ── 2. Gemini — room names + dimensions ONLY ──────────────────────────────────

async def gemini_name_rooms(image_url: str, gemini_key: str) -> list:
    """
    Ask Gemini to identify room names and read printed dimensions.
    NOT used for geometry — only for text/label reading.
    Returns list of {name, confidence, box_2d, dimensions}.
    """
    if not gemini_key:
        return []

    prompt = """You are reading a floor plan image.

List every distinct room or space you can identify.
Return ONLY a valid JSON array, no markdown, no explanation.

Each object must have:
- "name": unique human-readable name (e.g. "Master Bedroom", "Kitchen")
  Number duplicates: "Bedroom 1", "Bedroom 2"
- "confidence": 0-100
- "box_2d": [ymin, xmin, ymax, xmax] approximate position on 0-1000 scale
- "dimensions": {"length": number, "width": number, "unit": "ft"|"m"} or null
  Only include if you can clearly see printed measurements in the floor plan.
  Convert feet-inches like 14'-7" to decimal feet (14.58).

Return ONLY the JSON array."""

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                img_resp = await c.get(image_url)
                img_b64 = base64.b64encode(img_resp.content).decode()

            payload = {
                "contents": [{"role": "user", "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                ]}],
                "generationConfig": {"responseMimeType": "application/json"},
            }
            async with httpx.AsyncClient(timeout=45) as c:
                r = await c.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"gemini-2.5-flash:generateContent?key={gemini_key}",
                    json=payload,
                )
            if r.status_code == 503:
                wait = [3, 6, 10][attempt]
                log.warning(f"Gemini 503 attempt {attempt+1}, retry in {wait}s")
                await asyncio.sleep(wait)
                continue
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            arr = _parse_json_array(text)
            log.info(f"  Gemini: {len(arr)} rooms named")
            return arr
        except Exception as e:
            log.error(f"Gemini attempt {attempt+1}: {e}")
            await asyncio.sleep(3)
    return []

def _parse_json_array(text: str) -> list:
    text = re.sub(r"^```json\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"^```\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    s, e = text.find("["), text.rfind("]")
    if s == -1 or e == -1:
        return []
    return json.loads(text[s:e+1])

# ── 3. OpenCV wall detection ──────────────────────────────────────────────────

def detect_walls_cv(wall_mask, h: int, w: int) -> list:
    """Detect wall line segments using Hough transform."""
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

# ── 4. OpenCV room region detection ──────────────────────────────────────────

def detect_room_regions(wall_mask, h: int, w: int) -> list:
    """
    Find enclosed room regions using distance transform.
    This correctly separates adjacent rooms even with thin shared walls.
    """
    min_area = w * h * 0.002
    max_area = w * h * 0.55
    best_regions = []

    # Try distance transform first (best for separating adjacent rooms)
    k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sealed = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, k_small)
    floor = cv2.bitwise_not(sealed)
    margin = max(5, min(w,h)//40)
    floor[:margin,:]=0; floor[-margin:,:]=0
    floor[:,:margin]=0; floor[:,-margin:]=0

    dist = cv2.distanceTransform(floor, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    _, sure_fg = cv2.threshold(dist, 0.05, 1.0, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg * 255)

    n, _, stats, centroids = cv2.connectedComponentsWithStats(sure_fg, 8)
    log.info(f"  Distance transform: {n-1} components, min_area={int(min_area)}, max_area={int(max_area)}")
    for lbl in range(1, min(n, 20)):
        area = stats[lbl, cv2.CC_STAT_AREA]
        bw = stats[lbl, cv2.CC_STAT_WIDTH]
        bh = stats[lbl, cv2.CC_STAT_HEIGHT]
        log.info(f"    component {lbl}: area={area} bw={bw} bh={bh} {'OK' if min_area<area<max_area else 'SKIP'}")
    for lbl in range(1, n):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area: continue
        x  = stats[lbl, cv2.CC_STAT_LEFT]
        y  = stats[lbl, cv2.CC_STAT_TOP]
        bw = stats[lbl, cv2.CC_STAT_WIDTH]
        bh = stats[lbl, cv2.CC_STAT_HEIGHT]
        if bw < 5 or bh < 5: continue
        best_regions.append({
            "box": {
                "top":    round(y/h*100, 3),
                "left":   round(x/w*100, 3),
                "width":  round(bw/w*100, 3),
                "height": round(bh/h*100, 3),
            },
            "area_px":    int(area),
            "centroid_x": float(centroids[lbl][0]),
            "centroid_y": float(centroids[lbl][1]),
        })

    # If distance transform gave too few, try progressive flood-fill
    if len(best_regions) < 3:
        for ksize in [5, 9, 13, 19]:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
            closed = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, k)
            fl = cv2.bitwise_not(closed)
            fl[:margin,:]=0; fl[-margin:,:]=0; fl[:,:margin]=0; fl[:,-margin:]=0
            n2, _, stats2, cent2 = cv2.connectedComponentsWithStats(fl, 8)
            regions = []
            for lbl in range(1, n2):
                area = stats2[lbl, cv2.CC_STAT_AREA]
                if area < min_area or area > max_area: continue
                x  = stats2[lbl, cv2.CC_STAT_LEFT]
                y  = stats2[lbl, cv2.CC_STAT_TOP]
                bw = stats2[lbl, cv2.CC_STAT_WIDTH]
                bh = stats2[lbl, cv2.CC_STAT_HEIGHT]
                if max(bw,bh)/max(min(bw,bh),1) > 12: continue
                regions.append({
                    "box": {
                        "top":    round(y/h*100, 3),
                        "left":   round(x/w*100, 3),
                        "width":  round(bw/w*100, 3),
                        "height": round(bh/h*100, 3),
                    },
                    "area_px":    int(area),
                    "centroid_x": float(cent2[lbl][0]),
                    "centroid_y": float(cent2[lbl][1]),
                })
            if len(regions) > len(best_regions):
                best_regions = regions
            if len(regions) >= 4:
                break

    best_regions.sort(key=lambda r: r["area_px"], reverse=True)
    log.info(f"  Room regions: {len(best_regions)}")
    return best_regions[:15]

# ── 5. Match Gemini names to OpenCV regions ───────────────────────────────────

def match_names_to_regions(
    gemini_rooms: list,
    cv_regions: list,
    image_w: int,
    image_h: int,
    project_id: str,
) -> list:
    """
    Gemini gives room names + approximate box_2d positions.
    OpenCV gives exact region bounding boxes.
    Match by centroid distance. Use OpenCV box for precision.
    If no OpenCV regions, fall back to Gemini box directly.
    """
    FT_TO_M = 0.3048
    used: set = set()
    result = []

    for idx, g in enumerate(gemini_rooms):
        # Gemini centroid in pixel coords
        if g.get("box_2d") and len(g["box_2d"]) == 4:
            ymin, xmin, ymax, xmax = g["box_2d"]
            gcx = ((xmin+xmax)/2/1000) * image_w
            gcy = ((ymin+ymax)/2/1000) * image_h
        else:
            gcx, gcy = image_w/2, image_h/2

        # Find nearest unused OpenCV region
        best_i, best_d = -1, float("inf")
        for ri, reg in enumerate(cv_regions):
            if ri in used: continue
            d = np.hypot(reg["centroid_x"]-gcx, reg["centroid_y"]-gcy)
            if d < best_d:
                best_d, best_i = d, ri

        if best_i >= 0:
            box = cv_regions[best_i]["box"]
            used.add(best_i)
        elif g.get("box_2d") and len(g["box_2d"]) == 4:
            ymin, xmin, ymax, xmax = g["box_2d"]
            box = {
                "top":    max(0, min(100, ymin/10)),
                "left":   max(0, min(100, xmin/10)),
                "width":  max(0, min(100, (xmax-xmin)/10)),
                "height": max(0, min(100, (ymax-ymin)/10)),
            }
        else:
            continue

        # Extract dimensions
        dims = g.get("dimensions")
        length = width = None
        if dims and dims.get("length") and dims.get("width"):
            factor = FT_TO_M if dims.get("unit") == "ft" else 1
            length = round(dims["length"] * factor, 2)
            width  = round(dims["width"]  * factor, 2)

        name = g.get("name", f"Room {idx+1}")
        result.append({
            "id":         f"{project_id}-r{idx+1}",
            "name":       name,
            "confidence": int(g.get("confidence", 70)),
            "color":      room_color(name, idx),
            "box":        box,
            "length":     length,
            "width":      width,
        })

    # Add any unmatched OpenCV regions as unnamed rooms
    for ri, reg in enumerate(cv_regions):
        if ri in used: continue
        idx = len(result)
        name = f"Room {idx+1}"
        result.append({
            "id":         f"{project_id}-r{idx+1}",
            "name":       name,
            "confidence": 50,
            "color":      room_color(name, idx),
            "box":        reg["box"],
        })

    return result

# ── 6. Opening detection ──────────────────────────────────────────────────────

def detect_openings(wall_mask, h: int, w: int) -> list:
    openings = []

    # Doors — arc detection (door swing symbols are quarter-circles)
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
            if float(np.mean(wall_mask[y0:y1b,x0:x1b]>128)) < 0.10: continue
            hd = float(np.mean(wall_mask[max(0,cy-3):min(h,cy+3),max(0,cx-r):min(w,cx+r)]>128))
            vd = float(np.mean(wall_mask[max(0,cy-r):min(h,cy+r),max(0,cx-3):min(w,cx+3)]>128))
            openings.append({
                "type":  "door",
                "wall":  "horizontal" if hd >= vd else "vertical",
                "x":     round(cx/w*1000, 2),
                "y":     round(cy/h*1000, 2),
                "width": max(80.0, round(r*2/max(w,h)*1000, 2)),
            })

    # Windows — thin parallel segments crossing walls
    edges = cv2.Canny(wall_mask, 50, 150, apertureSize=3)
    min_win = max(12, min(w,h)//35)
    max_win = max(w,h)//12

    for kern, orient in [
        (cv2.getStructuringElement(cv2.MORPH_RECT, (min_win,1)), "horizontal"),
        (cv2.getStructuringElement(cv2.MORPH_RECT, (1,min_win)), "vertical"),
    ]:
        m = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kern)
        n, _, st, ct = cv2.connectedComponentsWithStats(m, 8)
        for lbl in range(1, n):
            seg_w = st[lbl,cv2.CC_STAT_WIDTH]
            seg_h = st[lbl,cv2.CC_STAT_HEIGHT]
            seg_len  = seg_w if orient=="horizontal" else seg_h
            seg_perp = seg_h if orient=="horizontal" else seg_w
            if seg_len < min_win or seg_len > max_win: continue
            if seg_perp > min_win * 2: continue  # must be thin symbol
            if st[lbl,cv2.CC_STAT_AREA] < 20: continue
            icx,icy = int(ct[lbl][0]), int(ct[lbl][1])
            roi = wall_mask[max(0,icy-4):min(h,icy+4), max(0,icx-4):min(w,icx+4)]
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
    bgr, wall_mask, h, w = preprocess(image_bytes)
    log.info(f"Image: {w}×{h}px  project={project_id}")

    # Step 1: Walls (OpenCV Hough)
    walls = detect_walls_cv(wall_mask, h, w)

    # Step 2: Room names + dimensions (Gemini — always called)
    gemini_rooms = await gemini_name_rooms(image_url, gemini_api_key)

    # Step 3: Room regions (OpenCV geometry)
    cv_regions = detect_room_regions(wall_mask, h, w)

    # Step 4: Match — if OpenCV found regions, use them for precise boxes
    # If OpenCV found nothing, use Gemini's box_2d directly
    rooms = match_names_to_regions(gemini_rooms, cv_regions, w, h, project_id)

    # Step 5: Openings (OpenCV)
    openings = detect_openings(wall_mask, h, w)

    return {
        "rooms":      rooms,
        "walls":      walls,
        "openings":   openings,
        "image_size": {"width": w, "height": h},
    }