"""utils/court_roi.py — Step 2: auto-detect editable court polygon + foot filter.

Three jobs:
  1. auto_detect_polygon  — propose 4 court corners from a (player-free) frame
  2. edit_polygon         — interactive click-drag corner editor (returns corners)
  3. foot_point / inside_polygon — keep a player only if their FEET are on court

The proposal is just a starting guess; the user corrects it in the editor and the
result is saved to cameras[].court_polygon in config.json. Foot point = midpoint
of the two ankle keypoints, fallback bbox bottom-center (hard rule).
"""
from __future__ import annotations

import cv2
import numpy as np

# COCO-17 ankle indices.
L_ANKLE, R_ANKLE = 15, 16


def order_corners(pts):
    """Order 4 points as TL, TR, BR, BL. Returns (4,2) float32."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    d = (pts[:, 0] - pts[:, 1])
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def median_background(source, n=25):
    """Median-blend ~n frames sampled across the clip to remove moving players,
    giving a clean static court image for detection. Returns BGR frame or None."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    if total > 0:
        for idx in np.linspace(0, total - 1, n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if ok:
                frames.append(f)
    else:  # unknown length — read sequentially
        for _ in range(n):
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
    cap.release()
    if not frames:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def auto_detect_polygon(frame):
    """Propose 4 court corners from a frame (ideally the median background).

    The padel playing surface is a low-saturation gray region (asphalt/concrete +
    white lines) — distinct from the saturated green turf borders and the dark
    fence/glass surroundings. We mask "low saturation AND not dark", take the
    largest contour, and approximate it to a quad. Falls back to a sensible inset
    rectangle if nothing court-shaped is found. Returns (4,2) float32.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (7, 7), 0), cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # gray court + white lines: desaturated but not in shadow.
    mask = ((sat < 70) & (val > 45)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))

    frame_area = float(w * h)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Accept only a court-sized blob: big enough to be the court, but not the whole
    # frame (which means the mask just swallowed the desaturated night scene).
    cnts = [c for c in cnts
            if 0.12 * frame_area < cv2.contourArea(c) < 0.90 * frame_area]
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        for eps in (0.02, 0.03, 0.05, 0.08):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                return order_corners(approx.reshape(-1, 2))
        return order_corners(cv2.boxPoints(cv2.minAreaRect(c)))

    # Fallback: a perspective trapezoid (narrower at the far baseline) — a better
    # starting guess for a high-corner court view than a plain rectangle.
    return order_corners([
        [0.17 * w, 0.20 * h], [0.83 * w, 0.20 * h],     # far baseline (top)
        [0.99 * w, 0.93 * h], [0.01 * w, 0.93 * h],     # near baseline (bottom)
    ])


def edit_polygon(frame, corners, win="setup-court"):
    """Interactive corner-drag editor on `frame` (drawn in the frame's own coords).

    Keys: S = save, R = reset to the auto proposal, Q/ESC = cancel.
    Returns a list of 4 [x, y] int pairs, or None if cancelled.
    """
    auto = [list(map(float, p)) for p in np.asarray(corners).reshape(-1, 2)]
    pts = [p[:] for p in auto]
    state = {"drag": -1}
    grab = 14
    labels = ["TL", "TR", "BR", "BL"]

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (px, py) in enumerate(pts):
                if (px - x) ** 2 + (py - y) ** 2 <= grab ** 2:
                    state["drag"] = i
                    break
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"] >= 0:
            pts[state["drag"]] = [float(x), float(y)]
        elif event == cv2.EVENT_LBUTTONUP:
            state["drag"] = -1

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    try:
        while True:
            canvas = frame.copy()
            arr = np.array(pts, np.int32)
            cv2.polylines(canvas, [arr], True, (0, 255, 255), 2, cv2.LINE_AA)
            for i, (px, py) in enumerate(pts):
                cv2.circle(canvas, (int(px), int(py)), 7, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(canvas, labels[i], (int(px) + 10, int(py) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "drag corners   S=save   R=reset   Q/ESC=cancel",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, "drag corners   S=save   R=reset   Q/ESC=cancel",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, canvas)
            k = cv2.waitKey(20) & 0xFF
            if k in (ord("s"), ord("S")):
                return [[int(round(px)), int(round(py))] for px, py in pts]
            if k in (ord("q"), 27):
                return None
            if k in (ord("r"), ord("R")):
                pts = [p[:] for p in auto]
    finally:
        cv2.destroyWindow(win)


def foot_point(bbox, keypoints, kpt_thr=0.3):
    """On-court test point: ankle midpoint if confident, else bbox bottom-center."""
    la, ra = keypoints[L_ANKLE], keypoints[R_ANKLE]
    if la[2] > kpt_thr and ra[2] > kpt_thr:
        return ((la[0] + ra[0]) / 2.0, (la[1] + ra[1]) / 2.0)
    if la[2] > kpt_thr:
        return (float(la[0]), float(la[1]))
    if ra[2] > kpt_thr:
        return (float(ra[0]), float(ra[1]))
    x1, _, x2, y2 = bbox
    return ((float(x1) + float(x2)) / 2.0, float(y2))


def inside_polygon(point, polygon):
    """True if point is inside the polygon. With no polygon set (empty), keep all
    detections so the pipeline still runs before court setup."""
    if polygon is None or len(polygon) < 3:
        return True
    poly = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
    return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0
