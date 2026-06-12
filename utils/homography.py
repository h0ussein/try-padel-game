"""utils/homography.py — dual-camera homography → one shared real-court frame (Step 5).

Court coordinates (meters): x along the 20 m length, y across the 10 m width. Net at
net_x; the two service lines sit service_from_back metres in front of each back wall
(real padel ~3 m, i.e. ~7 m from the net); center service line at y = width/2.

For EACH camera we findHomography from the SAME named real-court reference points
(clicked once in `--setup-homography`) into this shared frame, so a given court
position maps to the same minimap spot from either camera — which is what makes
cross-camera fusion (Step 6) possible. Both cameras MUST use the same reference
points / origin or fusion cannot match players.
"""
from __future__ import annotations

import os

import cv2
import numpy as np


def court_reference_points(net_x=10.0, width=10.0, baseline_x=0.0):
    """The 4 unmistakable points to click for a camera mounted behind ONE baseline:
    the two NET posts (shared by both cameras) then the camera's own two BACK corners.
    `baseline_x` is that camera's back wall in court coords (0 for one end, L for the
    other). LEFT/RIGHT MUST mean the same physical side in both cameras, so the two
    halves register into one shared court frame."""
    W = width
    return [
        ("NET post LEFT  (shared with other cam)",   (net_x, 0.0)),
        ("NET post RIGHT (shared with other cam)",   (net_x, W)),
        ("BACK corner LEFT  (your back wall)",       (baseline_x, 0.0)),
        ("BACK corner RIGHT (your back wall)",       (baseline_x, W)),
    ]


def draw_reprojection(frame, H, court_size=(20.0, 10.0), net_x=10.0, service_from_back=3.0):
    """Draw the real court template back onto the image via H^-1 (green lines, red net)
    so the user can visually confirm a calibration is correct."""
    L, W = court_size
    try:
        Hinv = np.linalg.inv(np.asarray(H, dtype=np.float64))
    except Exception:  # noqa: BLE001
        return

    def pl(a, b, col, th=2):
        ip = cv2.perspectiveTransform(np.array([[a], [b]], np.float32), Hinv).reshape(-1, 2)
        if np.all(np.isfinite(ip)):
            cv2.line(frame, tuple(ip[0].astype(int)), tuple(ip[1].astype(int)), col, th, cv2.LINE_AA)

    sb, sf = service_from_back, L - service_from_back
    for a, b in [((0, 0), (L, 0)), ((L, 0), (L, W)), ((L, W), (0, W)), ((0, W), (0, 0)),
                 ((sb, 0), (sb, W)), ((sf, 0), (sf, W)), ((sb, W / 2), (sf, W / 2))]:
        pl(a, b, (0, 255, 0))
    pl((net_x, 0), (net_x, W), (0, 0, 255), 3)  # net


def compute_homography(img_pts, court_pts):
    """H maps image pixels -> court meters. RANSAC when more than 4 correspondences."""
    img = np.asarray(img_pts, dtype=np.float32)
    crt = np.asarray(court_pts, dtype=np.float32)
    if len(img) < 4:
        return None
    if len(img) > 4:
        H, _ = cv2.findHomography(img, crt, cv2.RANSAC, 5.0)
    else:
        H, _ = cv2.findHomography(img, crt)
    return H


def pixel_to_court(H, pts):
    """Project image points (N,2) through H into court coords (N,2)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    if H is None or pts.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def load_homography(path):
    if path and os.path.exists(path):
        try:
            return np.load(path)
        except Exception:  # noqa: BLE001
            return None
    return None


def save_homography(path, H):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, H)


def _collect_points(frame, ref_points, win):
    """Click each named point in order. Returns list of (img_xy, court_xy) once all
    are set and the user presses ENTER, or None if cancelled. U = undo last."""
    names = list(ref_points)
    collected = {}
    state = {"idx": 0, "click": None}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (float(x), float(y))

    cv2.setMouseCallback(win, on_mouse)
    while True:
        i = state["idx"]
        if state["click"] is not None and i < len(names):
            collected[names[i][0]] = (state["click"], names[i][1])
            state["click"] = None
            state["idx"] += 1
            i = state["idx"]

        disp = frame.copy()
        for (ix, iy), _c in collected.values():
            cv2.circle(disp, (int(ix), int(iy)), 8, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(disp, (int(ix), int(iy)), 8, (0, 0, 0), 1, cv2.LINE_AA)
        if i < len(names):
            msg = f"Click: {names[i][0]}    ({i + 1}/{len(names)})    U=undo  Q=cancel"
        else:
            msg = f"All {len(names)} set.  ENTER=preview the court fit   U=undo  Q=cancel"
        for col, th in (((0, 0, 0), 5), ((0, 255, 255), 1)):
            cv2.putText(disp, msg, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, th, cv2.LINE_AA)
        cv2.imshow(win, disp)

        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            return None
        if k in (ord("u"), ord("U")) and state["idx"] > 0:
            state["idx"] -= 1
            collected.pop(names[state["idx"]][0], None)
        elif k in (13, 10) and len(collected) == len(names):  # ENTER
            return list(collected.values())


def calibrate_camera(frame, ref_points, court_size=(20.0, 10.0), net_x=10.0,
                     service_from_back=3.0, win="calibrate"):
    """Collect the reference points, compute H, then show an INSTANT reprojection
    preview (court drawn back onto the frame). Y = accept, R = redo, Q = cancel.
    Returns H or None."""
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    try:
        while True:
            pairs = _collect_points(frame, ref_points, win)
            if pairs is None:
                return None
            H = compute_homography([p[0] for p in pairs], [p[1] for p in pairs])
            if H is None:
                print("[calibrate] could not compute homography — try again.")
                continue

            redo = False
            while not redo:
                prev = frame.copy()
                draw_reprojection(prev, H, court_size, net_x, service_from_back)
                msg = "Court fit drawn (GREEN lines, RED net). Y=accept  R=redo  Q=cancel"
                for col, th in (((0, 0, 0), 5), ((0, 255, 255), 1)):
                    cv2.putText(prev, msg, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, th, cv2.LINE_AA)
                cv2.imshow(win, prev)
                k = cv2.waitKey(20) & 0xFF
                if k in (ord("y"), ord("Y"), 13, 10):
                    return H
                if k in (ord("r"), ord("R")):
                    redo = True
                if k in (ord("q"), 27):
                    return None
    finally:
        cv2.destroyWindow(win)
