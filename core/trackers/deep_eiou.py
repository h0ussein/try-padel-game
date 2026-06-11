"""core/trackers/deep_eiou.py — STrack + Deep-EIoU tracker (Step 3).

Association cascade (faithful to hsiangwei0903/Deep-EIoU):
  Stage 1  high-confidence detections, iterative scale-up EIoU at e=0.7 then 0.8,
           fused with OSNet ReID via dists = min(iou_dist, emb_dist).
  Stage 2  low-confidence detections, EIoU at e=0.5, IoU only (BYTE recovery).
  Stage 3  unconfirmed (just-born) tracks vs leftover high detections, EIoU.
New tracks are born from leftover high detections above new_track_thresh; tracks
unseen for > max_time_lost frames are removed. IDs come from a per-instance counter
so each camera numbers independently and the two tracker threads never collide.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from . import matching
from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None, kpts=None, feat_history=50):
        super().__init__()
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False
        self.score = float(score)
        self.tracklet_len = 0
        self.kpts = kpts

        self.smooth_feat = None
        self.curr_feat = None
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat):
        feat = np.asarray(feat, dtype=np.float64)
        feat /= (np.linalg.norm(feat) + 1e-12)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= (np.linalg.norm(self.smooth_feat) + 1e-12)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0  # zero w,h velocity when not actively tracked
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if not stracks:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id, track_id):
        self.kalman_filter = kalman_filter
        self.track_id = track_id
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = frame_id == 1
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, track_id=None):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if track_id is not None:
            self.track_id = track_id
        self.score = new_track.score
        self.kpts = new_track.kpts

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.kpts = new_track.kpts

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()       # x,y,w,h (center)
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        if exists.get(t.track_id, 0) == 0:
            exists[t.track_id] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {t.track_id: t for t in tlista}
    for t in tlistb:
        stracks.pop(t.track_id, None)
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb


class DeepEIoU:
    """One instance per camera. Feed YOLOv11 detections + optional OSNet features."""

    def __init__(self, track_high_thresh=0.6, track_low_thresh=0.1,
                 new_track_thresh=0.7, match_thresh=0.8, track_buffer=30,
                 proximity_thresh=0.5, appearance_thresh=0.25, with_reid=True,
                 frame_rate=30, expansion=(0.7, 0.8)):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self._id_count = 0

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.with_reid = with_reid
        self.expansion = expansion
        self.kalman_filter = KalmanFilter()
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)

    def _next_id(self):
        self._id_count += 1
        return self._id_count

    def _fuse_reid(self, ious_dists, tracks, dets):
        if not self.with_reid or len(tracks) == 0 or len(dets) == 0 \
                or dets[0].curr_feat is None:
            return ious_dists
        emb = matching.embedding_distance(tracks, dets) / 2.0
        emb[emb > self.appearance_thresh] = 1.0
        emb[ious_dists > self.proximity_thresh] = 1.0
        return np.minimum(ious_dists, emb)

    def update(self, dets, scores, feats=None, kpts=None):
        """dets: (N,4) tlbr; scores: (N,); feats: (N,D) or None; kpts: (N,17,3) or None.
        Returns the list of active STrack objects (each exposes .track_id/.tlbr/.kpts)."""
        self.frame_id += 1
        activated, refind, lost, removed = [], [], [], []

        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        n = len(dets)
        if feats is None:
            feats = [None] * n
        if kpts is None:
            kpts = [None] * n

        remain = scores > self.track_high_thresh
        low = (scores > self.track_low_thresh) & (scores <= self.track_high_thresh)

        def make(mask):
            idx = np.where(mask)[0]
            return [STrack(STrack_tlbr_to_tlwh(dets[i]), scores[i],
                           feats[i] if feats[i] is not None else None, kpts[i])
                    for i in idx]

        detections = make(remain)
        detections_low = make(low)

        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked = [t for t in self.tracked_stracks if t.is_activated]

        # --- Stage 1: high dets, iterative scale-up EIoU + ReID ---
        strack_pool = joint_stracks(tracked, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        for e in self.expansion:
            if not detections or not strack_pool:
                break
            ious_dists = matching.eiou_distance(strack_pool, detections, e)
            dists = self._fuse_reid(ious_dists, strack_pool, detections)
            matches, u_track, u_det = matching.linear_assignment(dists, thresh=self.match_thresh)
            for itracked, idet in matches:
                track, det = strack_pool[itracked], detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated.append(track)
                else:
                    track.re_activate(det, self.frame_id)
                    refind.append(track)
            strack_pool = [strack_pool[i] for i in u_track]
            detections = [detections[i] for i in u_det]

        # --- Stage 2: low dets vs still-tracked leftovers, EIoU e=0.5 ---
        r_tracked = [t for t in strack_pool if t.state == TrackState.Tracked]
        dists = matching.eiou_distance(r_tracked, detections_low, 0.5)
        matches, u_track, u_det_low = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track, det = r_tracked[itracked], detections_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind.append(track)
        for it in u_track:
            track = r_tracked[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

        # --- Stage 3: unconfirmed vs leftover high dets, EIoU e=0.7 ---
        ious_dists = matching.eiou_distance(unconfirmed, detections, 0.7)
        dists = self._fuse_reid(ious_dists, unconfirmed, detections)
        matches, u_unconfirmed, u_det = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed.append(track)

        # --- init new tracks from leftover high dets ---
        for inew in u_det:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id, self._next_id())
            activated.append(track)

        # --- age out lost tracks ---
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)
        # keep removed list bounded
        self.removed_stracks = self.removed_stracks[-1000:]

        return [t for t in self.tracked_stracks if t.is_activated]


def STrack_tlbr_to_tlwh(tlbr):
    ret = np.asarray(tlbr, dtype=np.float64).copy()
    ret[2:] -= ret[:2]
    return ret
