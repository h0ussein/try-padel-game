"""utils/skeleton.py — COCO-17 skeleton drawing + HUD/FPS overlay (Step 1).

COCO-17 keypoint order:
 0 nose        1 left_eye     2 right_eye   3 left_ear     4 right_ear
 5 left_sho    6 right_sho    7 left_elb    8 right_elb    9 left_wri   10 right_wri
11 left_hip   12 right_hip   13 left_knee  14 right_knee  15 left_ank   16 right_ank
"""
from __future__ import annotations

import cv2

# Bones connecting COCO-17 keypoints.
SKELETON = [
    (5, 7), (7, 9),            # left arm
    (6, 8), (8, 10),           # right arm
    (5, 6),                    # shoulders
    (11, 12),                  # hips
    (5, 11), (6, 12),          # torso sides
    (11, 13), (13, 15),        # left leg
    (12, 14), (14, 16),        # right leg
    (0, 1), (0, 2), (1, 3), (2, 4),  # face
    (0, 5), (0, 6),            # neck-ish (nose -> shoulders)
]

KPT_CONF_THRESH = 0.3       # only draw a joint/bone above this keypoint confidence
JOINT_COLOR = (0, 0, 255)   # red dots (BGR)


def draw_skeleton(frame, keypoints, color=(0, 255, 0), kpt_radius=4, thickness=3):
    """Draw bones + joints for one person's (17,3) keypoint array."""
    for a, b in SKELETON:
        xa, ya, ca = keypoints[a]
        xb, yb, cb = keypoints[b]
        if ca > KPT_CONF_THRESH and cb > KPT_CONF_THRESH:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)),
                     color, thickness, cv2.LINE_AA)
    for x, y, c in keypoints:
        if c > KPT_CONF_THRESH:
            cv2.circle(frame, (int(x), int(y)), kpt_radius, JOINT_COLOR, -1, cv2.LINE_AA)


def draw_bbox(frame, bbox, score, color=(0, 255, 0), label=None):
    """Draw a person bounding box with a bold, readable label (survives downscaling)."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    text = label if label is not None else f"{score:.2f}"
    fs, ft = 0.75, 2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, text, (x1 + 4, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), ft, cv2.LINE_AA)


def draw_hud(frame, cam_name, fps, n_players, n_total=None):
    """Top-left heads-up display: camera name, FPS, player count.

    n_total: when a court polygon is active, the total detected before filtering,
    rendered as "on-court: k/n"; otherwise just "players: k".
    """
    if n_total is None:
        count = f"players: {n_players}"
    else:
        count = f"on-court: {n_players}/{n_total}"
    text = f"{cam_name} | {fps:5.1f} FPS | {count}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (0, 0), (tw + 16, th + 16), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, th + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
