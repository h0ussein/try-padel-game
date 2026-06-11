"""utils/reid.py — OSNet ReID embedding extractor (Step 3).

Wraps boxmot's ReID runtime, which defaults to an OSNet backbone
(osnet_x0_25_msmt17.pt, auto-downloaded on first use). Returns one L2-normalized
appearance embedding per detection box, fed into Deep-EIoU's appearance branch.
If a padel fine-tuned OSNet exists at cfg["reid_weights"], it is used instead
(Section 9 / Step 3 optional fine-tuning).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class OSNetReID:
    def __init__(self, weights=None, device="cpu", half=False):
        from boxmot.reid.core.reid import ReID  # lazy: heavy import
        kwargs = {"device": device, "half": half}
        if weights and os.path.exists(weights):
            kwargs["weights"] = Path(weights)     # fine-tuned padel OSNet, if present
        # else ReID falls back to its default OSNet and auto-downloads the weights.
        self.model = ReID(**kwargs)

    def extract(self, frame, boxes):
        """boxes: (N,4) xyxy in frame pixels. Returns (N,D) L2-normalized features."""
        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.size == 0:
            return np.empty((0, 0), dtype=np.float32)
        feats = self.model(frame, boxes=boxes)
        return np.asarray(feats, dtype=np.float32)


def build_reid(cfg):
    """Construct the OSNet extractor if ReID is enabled in config, else None."""
    if not cfg.get("use_reid", True):
        return None
    device = cfg.get("detection", {}).get("device", "cpu")
    return OSNetReID(weights=cfg.get("reid_weights"), device=device)
