"""
Plan regression checks — run from the service folder:

    python tests/check_plans.py            # all plans, prints a scorecard
    python tests/check_plans.py great-room # one plan
    python tests/check_plans.py --overlay  # also write tests/out/<plan>.png
    python tests/check_plans.py --noise    # also re-run each plan with slight
                                           # pixel noise: results that flip
                                           # are fragile, not just wrong

Each plan in tests/plans has a checks file in tests/checks with what a
person sees on the drawing, in the ORIGINAL image's pixels:

  walls     segments that must be built as walls (openings inside a wall
            still count as wall — the 3D cuts them out)
  open      segments where NO wall may be built (open-plan spaces, carports)
  openings  doors / windows that must be found, with the right type
  max_duplicate_walls   two output walls on top of each other
  rooms_inside          no room may be detected outside this box
  names     the room at a point must be named right: its name contains one
            of "accept" (an open-plan room may carry several names)

A check marked "known": "<why>" is a known gap in the pipeline: it is
reported as KNOWN and does not fail the run — but if it starts passing the
script says so, so the mark can be removed.

The script exits with status 1 if any check fails, so it can run in CI.
Every change to pipeline.py should keep (or raise) the number of passes.
"""

import asyncio
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pipeline  # noqa: E402

import logging  # noqa: E402
logging.disable(logging.CRITICAL)


# ── geometry helpers (original-image pixels) ────────────────────────────────

def _horiz(seg):
    x1, y1, x2, y2 = seg
    return abs(x2 - x1) >= abs(y2 - y1)


def _span(seg):
    x1, y1, x2, y2 = seg
    return (min(x1, x2), max(x1, x2)) if _horiz(seg) else (min(y1, y2), max(y1, y2))


def _axis(seg):
    x1, y1, x2, y2 = seg
    return (y1 + y2) / 2 if _horiz(seg) else (x1 + x2) / 2


def _covered(seg, others, tol):
    """Fraction of `seg` lying along `others` (same orientation, same line)."""
    lo, hi = _span(seg)
    if hi - lo < 1e-6:
        return 0.0
    ivs = []
    for o in others:
        if _horiz(o) != _horiz(seg) or abs(_axis(o) - _axis(seg)) > tol:
            continue
        a, b = _span(o)
        a, b = max(a, lo), min(b, hi)
        if b > a:
            ivs.append((a, b))
    ivs.sort()
    total, cur_a, cur_b = 0.0, None, None
    for a, b in ivs:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return total / (hi - lo)


# ── run one plan ─────────────────────────────────────────────────────────────

def analyse(path):
    data = open(path, "rb").read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    H0, W0 = img.shape[:2]
    r = asyncio.run(pipeline.analyse_floor_plan(data, project_id="t"))
    walls = [[w["x1"] / 100 * W0, w["y1"] / 100 * H0,
              w["x2"] / 100 * W0, w["y2"] / 100 * H0] for w in r["walls"]]
    ops = []
    for o in r["openings"]:
        cx, cy = o["x"] / 1000 * W0, o["y"] / 1000 * H0
        half = o["width"] / 1000 * max(W0, H0) / 2
        seg = ([cx - half, cy, cx + half, cy] if o["wall"] == "horizontal"
               else [cx, cy - half, cx, cy + half])
        ops.append({"type": o["type"], "seg": seg})
    rooms = []
    for rm in r["rooms"]:
        pts = np.array(rm["polygon"], float)
        rooms.append({"name": rm["name"],
                      "cx": pts[:, 0].mean() / 100 * W0,
                      "cy": pts[:, 1].mean() / 100 * H0,
                      "poly": np.stack([pts[:, 0] / 100 * W0, pts[:, 1] / 100 * H0], 1)
                                .astype(np.float32)})
    return img, r, walls, ops, rooms


def check(spec, walls, ops, rooms):
    tol = spec.get("tol", 8)
    results = []
    known = {k["name"]: k["known"] for grp in ("walls", "open", "openings")
             for k in spec.get(grp, []) if k.get("known")}
    known.update(spec.get("known", {}))

    for c in spec.get("walls", []):
        cov = _covered(c["seg"], walls, tol)
        results.append(("wall", c["name"], cov >= 0.8, f"{cov:.0%} built"))

    for c in spec.get("open", []):
        cov = _covered(c["seg"], walls, tol)
        results.append(("open", c["name"], cov <= 0.15,
                        "clear" if cov <= 0.15 else f"{cov:.0%} walled"))

    for c in spec.get("openings", []):
        lo, hi = _span(c["seg"])
        best = None
        for o in ops:
            s = o["seg"]
            if _horiz(s) != _horiz(c["seg"]) or abs(_axis(s) - _axis(c["seg"])) > tol:
                continue
            a, b = _span(s)
            ov = min(b, hi) - max(a, lo)
            if ov <= 0.5 * min(b - a, hi - lo):
                continue
            best = o if (best is None or o["type"] == c["type"]) else best
        if best is None:
            results.append(("opening", c["name"], False, f"missing ({c['type']})"))
            continue
        a, b = _span(best["seg"])
        ok = best["type"] == c["type"] and (b - a) >= c.get("min_width", 0)
        detail = best["type"] if best["type"] != c["type"] else f"{c['type']} {b - a:.0f}px"
        results.append(("opening", c["name"], ok, detail))

    if "max_duplicate_walls" in spec:
        dup = 0
        for i in range(len(walls)):
            for j in range(i + 1, len(walls)):
                a, b = walls[i], walls[j]
                if _horiz(a) != _horiz(b) or abs(_axis(a) - _axis(b)) > 1.5 * tol:
                    continue
                if abs(_axis(a) - _axis(b)) < 0.5:
                    continue          # collinear pieces of one wall line, not a duplicate
                (a0, a1), (b0, b1) = _span(a), _span(b)
                ov = min(a1, b1) - max(a0, b0)
                if ov > 0.5 * min(a1 - a0, b1 - b0):
                    dup += 1
        results.append(("dup", "duplicate parallel walls", dup <= spec["max_duplicate_walls"],
                        f"{dup} found"))

    for c in spec.get("names", []):
        x, y = map(float, c["at"])
        inside = [rm for rm in rooms
                  if cv2.pointPolygonTest(rm["poly"].reshape(-1, 1, 2), (x, y), False) >= 0]
        if not inside:
            results.append(("name", c["name"], False, "no room there"))
            continue
        rm = min(inside, key=lambda r: cv2.contourArea(r["poly"]))
        got = rm["name"]
        ok = any(a in got.lower() for a in c["accept"])
        results.append(("name", c["name"], ok, f'"{got}"'))

    if "rooms_inside" in spec:
        x0, y0, x1, y1 = spec["rooms_inside"]
        out = [rm["name"] for rm in rooms
               if not (x0 - tol <= rm["cx"] <= x1 + tol and y0 - tol <= rm["cy"] <= y1 + tol)]
        results.append(("rooms", "no rooms outside the house", not out,
                        "none" if not out else ", ".join(out)))
    return [(k, n, ok, d, known.get(n)) for k, n, ok, d in results]


def overlay(img, walls, ops, path, result=None):
    s = 2.0
    im = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    H0, W0 = img.shape[:2]
    if result is not None:
        rng = np.random.RandomState(7)
        for rm in result["rooms"]:
            pts = np.array([[x / 100 * W0 * s, y / 100 * H0 * s] for x, y in rm["polygon"]], np.int32)
            ov = im.copy()
            cv2.fillPoly(ov, [pts], tuple(int(v) for v in rng.randint(120, 230, 3)))
            im = cv2.addWeighted(ov, 0.35, im, 0.65, 0)
        for rm in result["rooms"]:
            pts = np.array([[x / 100 * W0 * s, y / 100 * H0 * s] for x, y in rm["polygon"]], np.int32)
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(im, rm["name"], (int(cx) - 30, int(cy) + 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (120, 0, 120), 2)
    for w in walls:
        cv2.line(im, (int(w[0] * s), int(w[1] * s)), (int(w[2] * s), int(w[3] * s)),
                 (200, 60, 30), 4)
    for o in ops:
        col = (40, 170, 40) if o["type"] == "window" else (30, 30, 220)
        x1, y1, x2, y2 = o["seg"]
        cv2.line(im, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), col, 6)
    cv2.imwrite(path, im)


def main(argv):
    want = [a for a in argv if not a.startswith("--")]
    do_overlay = "--overlay" in argv
    specs = sorted(f for f in os.listdir(os.path.join(HERE, "checks")) if f.endswith(".json"))
    total = passed = failed = 0
    for f in specs:
        name = f[:-5]
        if want and name not in want:
            continue
        spec = json.load(open(os.path.join(HERE, "checks", f)))
        img, result, walls, ops, rooms = analyse(os.path.join(HERE, "plans", spec["plan"]))
        res = check(spec, walls, ops, rooms)
        ok = sum(1 for r in res if r[2])
        total += len(res); passed += ok
        failed += sum(1 for r in res if not r[2] and not r[4])
        print(f"\n{name}: {ok}/{len(res)}  ({len(walls)} walls, {len(ops)} openings)")
        for kind, label, good, detail, why in res:
            tag = "PASS" if good else ("KNOWN" if why else "FAIL")
            if good and why:
                tag = "PASS*"
                detail += "  (marked known — remove the mark)"
            print(f"  {tag:<5} {kind:<8} {label:<48} {detail}"
                  + (f"  [{why}]" if why and not good else ""))
        if do_overlay:
            os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
            overlay(img, walls, ops, os.path.join(HERE, "out", f"{name}.png"), result)
        if "--noise" in argv:
            import tempfile
            for seed in (1, 2):
                noisy = np.clip(img.astype(np.int16) + np.random.RandomState(seed)
                                .randint(-4, 5, img.shape), 0, 255).astype(np.uint8)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    f.write(cv2.imencode(".png", noisy)[1].tobytes())
                _, _, nw, no, nr = analyse(f.name)
                os.unlink(f.name)
                nres = check(spec, nw, no, nr)
                flips = [r[1] for r, q in zip(res, nres) if r[2] != q[2]]
                failed += sum(1 for r, q in zip(res, nres) if r[2] and not q[2] and not q[4])
                print(f"  noise {seed}: " + ("stable" if not flips else "FLIPPED " + "; ".join(flips)))
    print(f"\nTOTAL {passed}/{total} passing, {failed} unexpected failure(s)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
