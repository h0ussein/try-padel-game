"""utils/jersey.py — jersey color (HSV) as a shared identity anchor (Step 4).

Extracts the dominant torso color of a player (sampled from the shoulders-to-hips
region via keypoints, fallback to the upper-center of the bbox), classifies it to a
named color, and matches a player's jersey against the enrolled registry to recover
their NAME. Because identity is anchored to the jersey, the name follows the player
even when the tracker ID swaps — this is the ID-swap re-check the spec calls for.
"""
from __future__ import annotations

import cv2
import numpy as np

# COCO-17: shoulders 5/6, hips 11/12.
_TORSO_KPTS = (5, 6, 11, 12)


def _clamp_box(region, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = region
    x1 = int(max(0, min(w - 1, round(x1))))
    x2 = int(max(0, min(w, round(x2))))
    y1 = int(max(0, min(h - 1, round(y1))))
    y2 = int(max(0, min(h, round(y2))))
    return x1, y1, x2, y2


def torso_region(bbox, keypoints, kpt_thr=0.3):
    """Return an (x1,y1,x2,y2) ROI over the player's chest."""
    pts = []
    if keypoints is not None:
        pts = [(keypoints[i][0], keypoints[i][1]) for i in _TORSO_KPTS
               if keypoints[i][2] > kpt_thr]
    if len(pts) >= 3:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        w, h = (x2 - x1), (y2 - y1)
        return (x1 + 0.20 * w, y1 + 0.10 * h, x2 - 0.20 * w, y1 + 0.70 * h)
    # fallback: upper-center of the bbox (chest band)
    bx1, by1, bx2, by2 = bbox
    w, h = (bx2 - bx1), (by2 - by1)
    cx = (bx1 + bx2) / 2.0
    return (cx - 0.18 * w, by1 + 0.18 * h, cx + 0.18 * w, by1 + 0.48 * h)


def classify(h, s, v):
    """Classify an HSV (OpenCV ranges H:0-179, S/V:0-255) into a named color.

    Saturation bar is deliberately high: under dim/warm court lighting a white or
    pale shirt picks up a low-saturation warm tint that would otherwise be called
    "orange"/"red". Treat anything under-saturated as white/gray/black.
    """
    if v < 55:
        return "black"
    if s < 60:
        return "white" if v > 160 else "gray"
    if h < 10 or h >= 170:
        return "red"
    if h < 20:
        return "orange"
    if h < 33:
        return "yellow"
    if h < 85:
        return "green"
    if h < 100:
        return "cyan"
    if h < 130:
        return "blue"
    if h < 150:
        return "purple"
    return "pink"


def jersey_color(frame, bbox, keypoints, kpt_thr=0.3):
    """Return (color_name, (h, s, v)) for the player's jersey."""
    x1, y1, x2, y2 = _clamp_box(torso_region(bbox, keypoints, kpt_thr), frame.shape)
    if x2 <= x1 or y2 <= y1:
        return ("unknown", (0.0, 0.0, 0.0))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return ("unknown", (0.0, 0.0, 0.0))

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].ravel()
    S = hsv[:, :, 1].ravel()
    V = hsv[:, :, 2].ravel()
    colored = (S > 85) & (V > 50) & (V < 245)
    if colored.sum() > 0.25 * H.size:
        h = float(np.median(H[colored]))
        s = float(np.median(S[colored]))
        v = float(np.median(V[colored]))
    else:  # achromatic shirt (black/white/gray)
        h = float(np.median(H))
        s = float(np.median(S))
        v = float(np.median(V))
    return (classify(h, s, v), (h, s, v))


def color_distance(hsv_a, hsv_b):
    """Perceptual-ish distance between two HSV colors (hue is circular, weighted)."""
    dh = abs(hsv_a[0] - hsv_b[0])
    dh = min(dh, 180 - dh) / 90.0
    ds = abs(hsv_a[1] - hsv_b[1]) / 255.0
    dv = abs(hsv_a[2] - hsv_b[2]) / 255.0
    return 2.0 * dh + 0.5 * ds + 0.5 * dv


def assign_names(track_jerseys, registry, max_cost=1.0):
    """One-to-one match current tracks' jerseys to enrolled players.

    track_jerseys: list of (color_name, (h,s,v)) per track.
    registry: list of {"name", "jersey_color", "jersey_hsv"}.
    Returns a list of names (or None) aligned with track_jerseys.
    """
    n = len(track_jerseys)
    if not registry or n == 0:
        return [None] * n

    from scipy.optimize import linear_sum_assignment

    p = len(registry)
    cost = np.full((n, p), 5.0, dtype=np.float64)
    for i, (cname, hsv) in enumerate(track_jerseys):
        for j, player in enumerate(registry):
            d = color_distance(hsv, player.get("jersey_hsv", (0, 0, 0)))
            if cname != player.get("jersey_color"):
                d += 0.6  # mild penalty when the named color disagrees
            cost[i, j] = d

    names = [None] * n
    rows, cols = linear_sum_assignment(cost)
    for r, c in zip(rows, cols):
        if cost[r, c] <= max_cost:
            names[r] = registry[c]["name"]
    return names
