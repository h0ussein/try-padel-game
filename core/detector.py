"""core/detector.py — YOLOv11-Pose wrapper (Step 1).

Loads a YOLOv11-Pose model once and runs detection + 17-keypoint pose on a
single BGR frame. The model path comes from config.json ("model" key) — never
hardcoded. ultralytics auto-downloads the weights on first use if missing.
"""
from __future__ import annotations

import numpy as np
from ultralytics import YOLO


class Detection:
    """One detected person.

    bbox      : np.ndarray (4,)    -> x1, y1, x2, y2  (pixels)
    score     : float              -> detection confidence
    keypoints : np.ndarray (17,3)  -> COCO-17 (x, y, conf) per joint
    """

    __slots__ = ("bbox", "score", "keypoints")

    def __init__(self, bbox, score, keypoints):
        self.bbox = bbox
        self.score = float(score)
        self.keypoints = keypoints


class PoseDetector:
    def __init__(self, model_path, conf=0.25, iou=0.45, device="cpu", imgsz=640,
                 min_keypoints=0, min_aspect=0.0, kpt_conf=0.3):
        # YOLO() downloads the weights to the working dir on first use if absent.
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        # person-plausibility filter (rejects chairs/objects mis-detected as people):
        # keep a detection only if it has >= min_keypoints confident joints AND is at
        # least as tall as it is wide (min_aspect = height/width). 0 disables a check.
        self.min_keypoints = min_keypoints
        self.min_aspect = min_aspect
        self.kpt_conf = kpt_conf

    def _is_person(self, det):
        if self.min_keypoints and int((det.keypoints[:, 2] > self.kpt_conf).sum()) < self.min_keypoints:
            return False
        if self.min_aspect:
            x1, y1, x2, y2 = det.bbox
            w = float(x2) - float(x1)
            if w > 0 and (float(y2) - float(y1)) / w < self.min_aspect:
                return False
        return True

    def detect(self, frame):
        """Run pose detection on one BGR frame. Returns list[Detection]."""
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            imgsz=self.imgsz,
            classes=[0],      # person class only
            verbose=False,
        )
        out = []
        if not results:
            return out

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        if r.keypoints is not None and r.keypoints.data is not None:
            kpts = r.keypoints.data.cpu().numpy()          # (N, 17, 3)
        else:
            kpts = np.zeros((len(boxes), 17, 3), dtype=np.float32)

        for i in range(len(boxes)):
            det = Detection(boxes[i], scores[i], kpts[i])
            if self._is_person(det):
                out.append(det)
        return out

    @staticmethod
    def ensure_weights(model_path):
        """Trigger a one-time download of the weights in the main thread so the
        two camera workers don't race to download the same file."""
        YOLO(model_path)
