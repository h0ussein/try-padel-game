"""core/pipeline.py — per-camera worker loop (Steps 1–3).

One CameraWorker thread per camera: capture -> YOLOv11-Pose -> court foot-filter
-> Deep-EIoU tracking (+ OSNet ReID) -> draw IDs/skeletons/HUD. The thread does
ALL capture + inference + drawing and publishes the latest annotated frame; the
MAIN thread does cv2.imshow. Two cameras always run in parallel — there is no
single-camera path, and each camera owns ONE Deep-EIoU tracker (IDs are per-camera
at this stage; they get unified by cross-camera fusion in Step 6).

Later steps extend this loop: jersey/enrollment (Step 4), homography + minimap
(Step 5), fusion (Step 6), actions (Step 8), ball (Step 9).
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from core.ball import BallDetector, BallTrack
from core.detector import PoseDetector
from core.fusion import match_tracks_to_roster
from core.tracker import Tracker
from utils import colors, court_roi, homography, jersey, skeleton

_OFFCOURT = (130, 130, 130)


class CameraWorker(threading.Thread):
    def __init__(self, cam_cfg, global_cfg, reid_extractor=None, players=None,
                 sync_barrier=None):
        super().__init__(daemon=True)
        self.cam_name = cam_cfg.get("name", "cam")
        self.source = cam_cfg["source"]
        self.cam_cfg = cam_cfg
        self.global_cfg = global_cfg
        self.reid = reid_extractor
        self.players = players or []
        self.sync_barrier = sync_barrier   # keeps the two dev clips frame-aligned
        self.polygon = cam_cfg.get("court_polygon") or []
        self.H = homography.load_homography(cam_cfg.get("homography"))
        self.ball_cfg = global_cfg.get("ball", {})

        self._lock = threading.Lock()
        self._latest = None          # latest annotated BGR frame
        self._tracks_info = []        # latest per-track records (for cross-camera fusion)
        self._ball = None             # latest {"xy": court_xy, "kmh": speed} or None
        self._frame_no = 0            # processed-frame counter (for headless recording)
        self._fps = 0.0
        self._running = threading.Event()
        self._running.set()

        self.error = None            # set if init/open fails; read by main thread
        self.opened = False

    # --- public API (main thread) ---
    def stop(self):
        self._running.clear()

    @property
    def fps(self):
        return self._fps

    def latest(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def tracks_info(self):
        with self._lock:
            return list(self._tracks_info)

    @property
    def frame_no(self):
        with self._lock:
            return self._frame_no

    def ball(self):
        with self._lock:
            return self._ball

    # --- worker thread ---
    def run(self):
        det_cfg = self.global_cfg.get("detection", {})
        try:
            detector = PoseDetector(
                self.global_cfg["model"],
                conf=det_cfg.get("conf", 0.25),
                iou=det_cfg.get("iou", 0.45),
                device=det_cfg.get("device", "cpu"),
                imgsz=det_cfg.get("imgsz", 640),
                min_keypoints=det_cfg.get("min_keypoints", 0),
                min_aspect=det_cfg.get("min_aspect", 0.0),
                kpt_conf=det_cfg.get("kpt_conf", 0.3),
            )
        except Exception as exc:  # noqa: BLE001 - report any init failure to main
            self.error = f"detector init failed: {exc}"
            return

        tracker = Tracker(self.global_cfg, reid_extractor=self.reid)

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error = f"could not open source: {self.source}"
            return
        self.opened = True

        loop_clips = self.global_cfg.get("display", {}).get("loop_clips", True)
        poly_np = (np.array(self.polygon, np.int32).reshape(-1, 1, 2)
                   if len(self.polygon) >= 3 else None)

        ball_det = ball_track = None
        if self.ball_cfg.get("enabled", False):
            ball_det = BallDetector(
                model_path=self.ball_cfg.get("model"), device=det_cfg.get("device", "cpu"),
                conf=self.ball_cfg.get("conf", 0.25), imgsz=det_cfg.get("imgsz", 1280),
                diff_thresh=self.ball_cfg.get("diff_thresh", 18),
                min_area=self.ball_cfg.get("min_area", 10),
                max_area=self.ball_cfg.get("max_area", 900),
                min_circularity=self.ball_cfg.get("min_circularity", 0.45))
            _bfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            ball_track = BallTrack(fps=_bfps if _bfps > 1 else 30.0)

        while self._running.is_set():
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if loop_clips:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                if not ok:
                    break

            detections = detector.detect(frame)

            # Foot-point court filter: only on-court players go to the tracker.
            on_court, off_court = [], []
            for det in detections:
                foot = court_roi.foot_point(det.bbox, det.keypoints)
                if court_roi.inside_polygon(foot, self.polygon):
                    on_court.append(det)
                else:
                    off_court.append((det, foot))

            tracks = tracker.update(on_court, frame)

            # Hard cap: a padel match has at most 4 players. Per camera, keep the
            # most-established tracks (the authoritative global 4-player cap is at
            # fusion, Step 6). Suppresses transient ghost IDs.
            max_players = self.global_cfg.get("max_players", 4)
            if len(tracks) > max_players:
                tracks = sorted(tracks, key=lambda t: (t.tracklet_len, t.score),
                                reverse=True)[:max_players]

            # excluded (off-court) people: thin gray box + red foot dot, no ID
            for det, foot in off_court:
                x1, y1, x2, y2 = (int(v) for v in det.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), _OFFCOURT, 1)
                cv2.circle(frame, (int(foot[0]), int(foot[1])), 5, (0, 0, 255), -1, cv2.LINE_AA)

            # Accurate per-camera names: match each track to the enrolled roster by
            # OSNet ReID — the SAME identity source the fused minimap uses, so the
            # feed labels and the minimap agree. Falls back to jersey-only naming if
            # the roster has no appearance signatures (i.e. not enrolled with ReID).
            track_jerseys = [jersey.jersey_color(frame, st.tlbr, st.kpts) for st in tracks]
            names = match_tracks_to_roster(
                [st.smooth_feat for st in tracks], track_jerseys, self.players)
            if self.players and all(n is None for n in names):
                names = jersey.assign_names(track_jerseys, self.players)

            # tracked players: stable color + name/ID + skeleton; build fusion records
            feet, records = [], []
            for idx, st in enumerate(tracks):
                color = colors.color_for_id(st.track_id)
                nm = names[idx] if names[idx] else f"ID {st.track_id}"
                label = f"{nm}  {int(round(st.score * 100))}%"   # name + detection accuracy
                skeleton.draw_bbox(frame, st.tlbr, st.score, color=color, label=label)
                if st.kpts is not None:
                    skeleton.draw_skeleton(frame, st.kpts, color=color)
                    foot = court_roi.foot_point(st.tlbr, st.kpts)
                else:
                    x1, y1, x2, y2 = st.tlbr
                    foot = ((x1 + x2) / 2.0, y2)
                cv2.circle(frame, (int(foot[0]), int(foot[1])), 6, (0, 255, 0), -1, cv2.LINE_AA)
                feet.append(foot)
                cname, hsv = track_jerseys[idx]
                records.append({"local_id": int(st.track_id), "name": names[idx],
                                "jersey": cname, "hsv": list(hsv), "xy": None,
                                "reid": None if st.smooth_feat is None else st.smooth_feat.copy()})

            # Project feet through this camera's homography into the shared court frame.
            if self.H is not None and feet:
                for rec, (cx, cy) in zip(records, homography.pixel_to_court(self.H, feet)):
                    rec["xy"] = (float(cx), float(cy))

            # Ball: detect (motion+shape, or a trained model), draw it + km/h, project
            # to court coords for the minimap. Players' boxes are excluded inside detect().
            ball_pub = None
            if ball_det is not None:
                bp = ball_det.detect(frame, player_boxes=[d.bbox for d in detections],
                                     polygon=self.polygon)
                bxy = None
                if bp is not None:
                    cv2.circle(frame, (int(bp[0]), int(bp[1])), 12, (0, 255, 255), 2, cv2.LINE_AA)
                    if self.H is not None:
                        cc = homography.pixel_to_court(self.H, [(bp[0], bp[1])])
                        if len(cc):
                            bxy = (float(cc[0][0]), float(cc[0][1]))
                smoothed, kmh = ball_track.update(bxy)
                if bp is not None and kmh > 0:
                    cv2.putText(frame, f"{kmh:.0f} km/h", (int(bp[0]) + 14, int(bp[1]) - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                if smoothed is not None:
                    ball_pub = {"xy": smoothed, "kmh": kmh}

            if poly_np is not None:
                cv2.polylines(frame, [poly_np], True, (0, 255, 255), 2, cv2.LINE_AA)

            dt = time.perf_counter() - t0
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            self._fps = inst_fps if self._fps == 0 else 0.9 * self._fps + 0.1 * inst_fps
            total = None if poly_np is None else len(detections)
            skeleton.draw_hud(frame, self.cam_name, self._fps, len(tracks), n_total=total)

            with self._lock:
                self._latest = frame
                self._tracks_info = records
                self._ball = ball_pub
                self._frame_no += 1

            # Keep both camera clips on the same frame index (so a player's position
            # matches across cameras for fusion). The faster worker waits for the
            # slower; if a worker dies the barrier breaks and both run solo.
            if self.sync_barrier is not None:
                try:
                    self.sync_barrier.wait(timeout=5.0)
                except threading.BrokenBarrierError:
                    self.sync_barrier = None

        if self.sync_barrier is not None:
            try:
                self.sync_barrier.abort()
            except Exception:  # noqa: BLE001
                pass
        cap.release()
