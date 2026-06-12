"""core/ball.py — ball detection + shot-speed (Step 9).

Accuracy ladder (spec Section 7). A padel ball is tiny, fast and low-contrast, so a
pretrained/general model is NOT reliable (COCO "sports ball" maxes ~0.2 conf and
mostly fires on lights/fence). The accuracy comes from a TRAINED detector:

  Level 1 (here, no training): MOTION + SHAPE. Frame-difference motion, then REMOVE
    the players' boxes (we already track them — their limbs move too) and keep only
    small, round, in-court moving blobs. Track the candidate with smooth, fast motion.
    A useful bring-up + speed, but NOT high accuracy on its own.

  Level 2/3 (for "very accurate"): a TRAINED detector — a custom YOLOv11 ball model
    or TrackNet. Set `ball.model` in config to a .pt and it is used instead of motion.
    Build the dataset with `--collect-ball`, label in Roboflow/CVAT, train, then point
    `ball.model` at the weights.

Speed: project the ball's court position (via the camera homography) frame-to-frame
-> m/s -> km/h.
"""
from __future__ import annotations

import os
from collections import deque

import cv2
import numpy as np


class BallDetector:
    """Per-camera ball detector. Uses a trained YOLO ball model if `model_path` exists,
    else the motion+shape fallback."""

    def __init__(self, model_path=None, device="cpu", conf=0.25, imgsz=1280,
                 diff_thresh=18, min_area=10, max_area=900, min_circularity=0.45,
                 player_pad=14):
        self.yolo = None
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        if model_path and os.path.exists(model_path):
            from ultralytics import YOLO
            self.yolo = YOLO(model_path)          # trained custom ball detector
        self.diff_thresh = diff_thresh
        self.min_area = min_area
        self.max_area = max_area
        self.min_circ = min_circularity
        self.player_pad = player_pad
        self._prev_gray = None

    @property
    def trained(self):
        return self.yolo is not None

    def detect(self, frame, player_boxes=None, polygon=None):
        """Return (x, y, score) of the best ball candidate in pixels, or None."""
        if self.yolo is not None:
            return self._detect_yolo(frame, polygon)
        return self._detect_motion(frame, player_boxes, polygon)

    # --- Level 2/3: trained model ---
    def _detect_yolo(self, frame, polygon):
        r = self.yolo.predict(frame, conf=self.conf, imgsz=self.imgsz,
                              device=self.device, verbose=False)[0]
        best = None
        if r.boxes is not None:
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                if polygon and not _inside(polygon, (cx, cy)):
                    continue
                if best is None or c > best[2]:
                    best = (float(cx), float(cy), float(c))
        return best

    # --- Level 1: motion + shape ---
    def _detect_motion(self, frame, player_boxes, polygon):
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return None
        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray
        _, mask = cv2.threshold(diff, self.diff_thresh, 255, cv2.THRESH_BINARY)

        if player_boxes:                          # remove moving player limbs
            for x1, y1, x2, y2 in player_boxes:
                p = self.player_pad
                cv2.rectangle(mask, (int(x1 - p), int(y1 - p)),
                              (int(x2 + p), int(y2 + p)), 0, -1)
        if polygon and len(polygon) >= 3:         # keep only the court area
            pm = np.zeros_like(mask)
            cv2.fillPoly(pm, [np.asarray(polygon, np.int32)], 255)
            mask = cv2.bitwise_and(mask, pm)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            a = cv2.contourArea(c)
            if a < self.min_area or a > self.max_area:
                continue
            (x, y), rad = cv2.minEnclosingCircle(c)
            circ = a / (np.pi * rad * rad + 1e-6)     # 1.0 = perfect circle
            if circ < self.min_circ:
                continue
            if best is None or circ > best[2]:
                best = (float(x), float(y), float(circ))
        return best


class BallTrack:
    """Trajectory smoothing + speed. Feed the ball's COURT position (meters) each
    frame; get a smoothed position and a km/h estimate. Drops outliers (jumps too far
    to be one ball in one frame) and coasts briefly through misses."""

    def __init__(self, fps=30.0, max_jump_m=4.0, max_miss=8, smooth=3):
        self.fps = max(fps, 1.0)
        self.max_jump = max_jump_m
        self.max_miss = max_miss
        self.hist = deque(maxlen=max(smooth, 2))   # (frame_idx, (x,y))
        self.miss = 0
        self.frame = 0
        self.kmh = 0.0

    def update(self, court_xy):
        self.frame += 1
        if court_xy is None:
            self.miss += 1
            if self.miss > self.max_miss:
                self.hist.clear()
                self.kmh = 0.0
            return None, self.kmh
        xy = np.asarray(court_xy, dtype=float)
        if self.hist:
            prev = self.hist[-1][1]
            if np.hypot(*(xy - prev)) > self.max_jump:   # implausible jump -> reject
                return (float(prev[0]), float(prev[1])), self.kmh
        self.miss = 0
        self.hist.append((self.frame, xy))
        if len(self.hist) >= 2:
            (f0, p0), (f1, p1) = self.hist[0], self.hist[-1]
            dt = max(f1 - f0, 1) / self.fps
            self.kmh = float(np.hypot(*(p1 - p0)) / dt * 3.6)
        sm = np.mean([p for _, p in self.hist], axis=0)
        return (float(sm[0]), float(sm[1])), self.kmh


def _inside(polygon, point):
    poly = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
    return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0
