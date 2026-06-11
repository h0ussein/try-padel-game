"""core/trackers/matching.py — ExpansionIoU (EIoU) + ReID association costs (Step 3).

The defining piece of Deep-EIoU: boxes are EXPANDED by a scale factor before IoU,
so fast/irregular sports motion still overlaps frame-to-frame. The tracker calls
eiou_distance() with growing scale factors (iterative scale-up). Appearance costs
come from embedding_distance(); the tracker fuses them as min(iou, emb).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

try:  # lapx (installed via boxmot) provides the fast Jonker-Volgenant solver.
    import lap

    def _solve(cost_matrix, thresh):
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        return x, y
except Exception:  # pragma: no cover - fallback if lap is unavailable
    from scipy.optimize import linear_sum_assignment

    def _solve(cost_matrix, thresh):
        n, m = cost_matrix.shape
        x = np.full(n, -1, dtype=int)
        y = np.full(m, -1, dtype=int)
        cm = cost_matrix.copy()
        cm[cm > thresh] = thresh + 1e-4
        rows, cols = linear_sum_assignment(cm)
        for r, c in zip(rows, cols):
            if cost_matrix[r, c] <= thresh:
                x[r] = c
                y[c] = r
        return x, y


def linear_assignment(cost_matrix, thresh):
    """Return matches (K,2), unmatched_a indices, unmatched_b indices."""
    if cost_matrix.size == 0:
        return (np.empty((0, 2), dtype=int),
                tuple(range(cost_matrix.shape[0])),
                tuple(range(cost_matrix.shape[1])))
    x, y = _solve(cost_matrix, thresh)
    matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches) if matches else np.empty((0, 2), dtype=int)
    return matches, unmatched_a, unmatched_b


def _expand(tlbrs, e):
    """Expand each box by factor e on every side: new size = old * (1 + 2e)."""
    out = np.asarray(tlbrs, dtype=np.float64).copy()
    if out.size == 0:
        return out
    w = out[:, 2] - out[:, 0]
    h = out[:, 3] - out[:, 1]
    ew, eh = w * e, h * e
    out[:, 0] -= ew
    out[:, 1] -= eh
    out[:, 2] += ew
    out[:, 3] += eh
    return out


def _ious(atlbrs, btlbrs):
    """Pairwise IoU between two sets of [x1,y1,x2,y2] boxes -> (A,B)."""
    A, B = len(atlbrs), len(btlbrs)
    ious = np.zeros((A, B), dtype=np.float64)
    if A == 0 or B == 0:
        return ious
    a = np.asarray(atlbrs, dtype=np.float64)
    b = np.asarray(btlbrs, dtype=np.float64)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    for i in range(A):
        xx1 = np.maximum(a[i, 0], b[:, 0])
        yy1 = np.maximum(a[i, 1], b[:, 1])
        xx2 = np.minimum(a[i, 2], b[:, 2])
        yy2 = np.minimum(a[i, 3], b[:, 3])
        iw = np.clip(xx2 - xx1, 0, None)
        ih = np.clip(yy2 - yy1, 0, None)
        inter = iw * ih
        ious[i] = inter / (area_a[i] + area_b - inter + 1e-12)
    return ious


def eiou_distance(tracks, detections, e):
    """1 - EIoU between track and detection boxes after expanding both by e."""
    atlbrs = _expand([t.tlbr for t in tracks], e)
    btlbrs = _expand([d.tlbr for d in detections], e)
    return 1.0 - _ious(atlbrs, btlbrs)


def iou_distance(tracks, detections):
    return 1.0 - _ious([t.tlbr for t in tracks], [d.tlbr for d in detections])


def embedding_distance(tracks, detections, metric="cosine"):
    """Cosine distance between track (smoothed) features and detection features."""
    cost = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost.size == 0:
        return cost
    det_feats = np.asarray([d.curr_feat for d in detections], dtype=np.float64)
    trk_feats = np.asarray([t.smooth_feat for t in tracks], dtype=np.float64)
    cost = np.maximum(0.0, cdist(trk_feats, det_feats, metric))
    return cost


def fuse_score(cost_matrix, detections):
    """Blend an IoU cost with detection confidence (BoT-SORT fuse)."""
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1.0 - cost_matrix
    det_scores = np.array([d.score for d in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fused = iou_sim * det_scores
    return 1.0 - fused
