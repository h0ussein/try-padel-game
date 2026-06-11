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
    def __init__(self, model_path, conf=0.25, iou=0.45, device="cpu", imgsz=640):
        # YOLO() downloads the weights to the working dir on first use if absent.
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz

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
            out.append(Detection(boxes[i], scores[i], kpts[i]))
        return out

    @staticmethod
    def ensure_weights(model_path):
        """Trigger a one-time download of the weights in the main thread so the
        two camera workers don't race to download the same file."""
        YOLO(model_path)
