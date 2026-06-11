"""core/tracker.py — Deep-EIoU tracker wrapper (Step 3).

One Tracker per camera. Feeds YOLOv11-Pose detections (boxes + scores + keypoints)
and, when available, OSNet ReID embeddings into the vendored Deep-EIoU tracker
(core/trackers). Returns the active tracks (each exposes .track_id, .tlbr, .kpts,
.score). Hard rule: this is Deep-EIoU (+ OSNet ReID) — never ByteTrack.
"""
from __future__ import annotations

import numpy as np

from core.trackers import DeepEIoU


class Tracker:
    def __init__(self, cfg, reid_extractor=None):
        params = cfg.get("tracker_params", {}) or {}
        self.reid = reid_extractor
        # Defaults tuned for this dim padel footage (YOLOv11n-pose scores run
        # ~0.6-0.85 for real players); override per-deployment via tracker_params.
        self.tracker = DeepEIoU(
            track_high_thresh=params.get("track_high_thresh", 0.4),
            track_low_thresh=params.get("track_low_thresh", 0.1),
            new_track_thresh=params.get("new_track_thresh", 0.5),
            match_thresh=params.get("match_thresh", 0.8),
            track_buffer=params.get("track_buffer", 30),
            proximity_thresh=params.get("proximity_thresh", 0.5),
            appearance_thresh=params.get("appearance_thresh", 0.25),
            with_reid=reid_extractor is not None,
            frame_rate=params.get("frame_rate", 30),
        )

    def update(self, detections, frame):
        """detections: list of core.detector.Detection (already court-filtered).
        Returns the list of active STrack objects."""
        if not detections:
            return self.tracker.update(np.zeros((0, 4)), np.zeros((0,)))

        boxes = np.array([d.bbox for d in detections], dtype=np.float64)
        scores = np.array([d.score for d in detections], dtype=np.float64)
        kpts = [d.keypoints for d in detections]
        feats = self.reid.extract(frame, boxes) if self.reid is not None else None
        if feats is not None and (feats.ndim != 2 or feats.shape[0] != len(boxes)):
            feats = None  # shape mismatch -> fall back to motion-only association
        return self.tracker.update(boxes, scores, feats, kpts)
