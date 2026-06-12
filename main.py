"""main.py — entry point (Step 1: Dual-Camera Capture & Detection).

Reads config.json, launches ONE capture/inference thread per camera (always two,
in parallel), and displays both annotated feeds. OpenCV GUI runs on the main
thread; the workers only produce annotated frames.

Run:
    python main.py --show

Quit:
    press 'q' or ESC in any window.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time

import cv2
import numpy as np

from core.detector import PoseDetector
from core.fusion import Fusion
from core.pipeline import CameraWorker
from utils import colors, court_roi, enrollment, homography
from utils.court_view import Minimap
from utils.reid import build_reid


def compose_canvas(frames, names, pane_w, pane_h):
    """Tile the per-camera annotated frames side by side into ONE image so both
    feeds share a single window. Missing frames render as a 'loading' placeholder.
    The two capture/inference threads are unchanged — only the display is merged.
    """
    panes = []
    for frame, name in zip(frames, names):
        if frame is None:
            pane = np.zeros((pane_h, pane_w, 3), np.uint8)
            cv2.putText(pane, f"{name}: loading...", (16, pane_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 200), 2, cv2.LINE_AA)
        else:
            pane = cv2.resize(frame, (pane_w, pane_h))
        panes.append(pane)
    sep = np.full((pane_h, 4, 3), 60, np.uint8)     # thin divider between feeds
    out = panes[0]
    for pane in panes[1:]:
        out = np.hstack([out, sep, pane])
    return out


def build_canvas(frames, workers, names, dw, dh, minimap, fusion, flips, court_L,
                 court_W, mm_h, now, view_mode=0):
    """Compose the full output image: the two annotated feeds (or one full-size when
    view_mode>0) with the fused minimap strip beneath. Shared by the live window and
    the headless recorder so both look identical."""
    if view_mode == 0:
        canvas = compose_canvas(frames, names, dw, dh)
    else:
        sel = frames[view_mode - 1]
        big = (int(dw * 1.6), int(dh * 1.6))
        canvas = (cv2.resize(sel, big) if sel is not None
                  else np.zeros((big[1], big[0], 3), np.uint8))
    if minimap is not None:
        cam_tracks = []
        for ci, w in enumerate(workers):
            recs = []
            for r in w.tracks_info():
                rr = dict(r, cam=ci)
                if rr.get("xy") is not None:
                    x, y = rr["xy"]
                    if flips[ci]["x"]:
                        x = court_L - x
                    if flips[ci]["y"]:
                        y = court_W - y
                    rr["xy"] = (x, y)
                recs.append(rr)
            cam_tracks.append(recs)
        fused = fusion.update(cam_tracks, now)
        dots = [(g["xy"][0], g["xy"][1], colors.color_for_id(g["gid"]),
                 g["name"] or f"P{g['gid']}") for g in fused]
        mm = minimap.render(dots)
        mw = min(int(mm.shape[1] * mm_h / mm.shape[0]), canvas.shape[1])
        mm = cv2.resize(mm, (mw, mm_h))
        strip = np.zeros((mm_h, canvas.shape[1], 3), np.uint8)
        x0 = (canvas.shape[1] - mw) // 2
        strip[:, x0:x0 + mw] = mm
        canvas = np.vstack([canvas, strip])
    return canvas


def record(workers, names, dw, dh, minimap, fusion, flips, court_L, court_W, mm_h,
           out_path, fps, max_frames=None):
    """Headless recording: write ONE output frame per processed frame to an .mp4 (no
    GUI — for servers / virtual GPUs). Stops when the source clips end, or after
    max_frames (use that for never-ending live RTSP streams)."""
    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    last, written = -1, 0
    print(f"[record] writing annotated video to {out_path} at {fps:.0f} fps (headless, no window)...")
    try:
        while True:
            cur = min((w.frame_no for w in workers), default=-1)
            frames = [w.latest() for w in workers]
            ready = cur > last and all(f is not None for f in frames)
            if ready:
                canvas = build_canvas(frames, workers, names, dw, dh, minimap, fusion,
                                      flips, court_L, court_W, mm_h, now=cur / fps, view_mode=0)
                if writer is None:
                    h, wd = canvas.shape[:2]
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (wd, h))
                writer.write(canvas)
                written += 1
                last = cur
                if written % 50 == 0:
                    print(f"[record] {written} frames written...")
                if max_frames and written >= max_frames:
                    break
            if not any(w.is_alive() for w in workers) and not ready:
                break
            time.sleep(0.003)
    finally:
        if writer is not None:
            writer.release()
    print(f"[record] DONE — wrote {written} frames -> {out_path}")


def load_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
        fh.write("\n")


def setup_court(cfg, cfg_path):
    """Step 2: per camera, auto-detect the court polygon, let the user drag the
    corners, and save the result to cameras[].court_polygon in config.json."""
    for cam in cfg["cameras"]:
        name = cam.get("name", "cam")
        print(f"[setup-court] {name}: building player-free background...")
        bg = court_roi.median_background(cam["source"])
        if bg is None:
            print(f"  ! could not read frames from {cam['source']} — skipped.")
            continue

        h, w = bg.shape[:2]
        scale = min(1.0, 1280.0 / w)          # edit on a screen-friendly size
        disp = cv2.resize(bg, (int(w * scale), int(h * scale)))
        auto_native = court_roi.auto_detect_polygon(bg)
        auto_disp = auto_native * scale

        print(f"  drag the 4 corners to match the court, then press S to save "
              f"(R=reset, Q=cancel).")
        edited = court_roi.edit_polygon(disp, auto_disp, win=f"setup-court [{name}]")
        if edited is None:
            print(f"  cancelled — keeping existing polygon for {name}.")
            continue

        native = [[int(round(x / scale)), int(round(y / scale))] for x, y in edited]
        cam["court_polygon"] = native
        print(f"  saved {name} polygon: {native}")

    save_config(cfg, cfg_path)
    print(f"[setup-court] written to {cfg_path}. Now run: python main.py --show")


def setup_homography(cfg):
    """Step 5: per camera, click 4 clear points (2 NET posts + 2 BACK corners), see
    an instant court-fit preview, and save the homography to calibration/camN_H.npy.
    The first camera's back wall is x=0, the second's is x=L; the net (x=net_x) is the
    shared anchor between them."""
    court = tuple(cfg.get("court_real_size", [20.0, 10.0]))
    L, W = court
    net_x = cfg.get("net_x", 10.0)
    sfb = cfg.get("service_line_from_back", 3.0)
    print("[setup-homography] For EACH camera click 4 points: the 2 NET posts, then your")
    print("  2 BACK corners. Press ENTER to preview the court fit, then Y to accept / R to redo.")
    print("  IMPORTANT: click the SAME physical net post as 'LEFT' in BOTH cameras so the two")
    print("  halves line up into one court frame.")
    for i, cam in enumerate(cfg["cameras"]):
        name = cam.get("name", "cam")
        baseline_x = 0.0 if i == 0 else float(L)
        print(f"[setup-homography] {name}: back wall is x={baseline_x:.0f} m. "
              f"Building player-free background...")
        bg = court_roi.median_background(cam["source"])
        if bg is None:
            print(f"  ! could not read frames from {cam['source']} — skipped.")
            continue
        refs = homography.court_reference_points(net_x, W, baseline_x)
        H = homography.calibrate_camera(bg, refs, court, net_x, sfb, win=f"calibrate [{name}]")
        if H is None:
            print(f"  cancelled — keeping existing homography for {name}.")
            continue
        out_path = cam.get("homography", f"calibration/{name}_H.npy")
        homography.save_homography(out_path, H)
        print(f"  saved {name} homography -> {out_path}")
    print("[setup-homography] done. Now run: python main.py --show --minimap")


def parse_args():
    p = argparse.ArgumentParser(description="Padel CV System — dual-camera pipeline")
    p.add_argument("--config", default="config.json", help="path to config.json")
    # Documented flags across the roadmap; only --show is active in Step 1.
    p.add_argument("--show", action="store_true", help="display both camera windows")
    p.add_argument("--minimap", action="store_true", help="(Step 5) live court minimap")
    p.add_argument("--setup-court", action="store_true", help="(Step 2) edit court polygon")
    p.add_argument("--setup-homography", action="store_true",
                   help="(Step 5) calibrate each camera's court homography")
    p.add_argument("--enroll", action="store_true", help="(Step 4) enroll players")
    p.add_argument("--collect", action="store_true", help="(Step 7) save keypoint windows")
    p.add_argument("--save-report", action="store_true", help="(Step 10) write session report")
    p.add_argument("--save-video", action="store_true",
                   help="HEADLESS: write the annotated output to an .mp4 (no window — for servers/GPU)")
    p.add_argument("--output", default="output/output.mp4",
                   help="output .mp4 path for --save-video")
    p.add_argument("--max-frames", type=int, default=None,
                   help="stop --save-video after N frames (use for never-ending live RTSP streams)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    cameras = cfg.get("cameras", [])
    if len(cameras) != 2:
        print(f"ERROR: config must define exactly 2 cameras, found {len(cameras)}. "
              "This system has no single-camera path.", file=sys.stderr)
        return 2

    if args.setup_court:
        setup_court(cfg, args.config)
        return 0

    if args.setup_homography:
        setup_homography(cfg)
        return 0

    players_path = cfg.get("players_registry", "players.json")
    if args.enroll:
        enrollment.enroll(cfg, players_path)
        return 0

    for stub in ("collect", "save_report"):
        if getattr(args, stub):
            print(f"[note] --{stub.replace('_', '-')} is not implemented yet "
                  f"(later step). Running detection display.")

    # Headless recording processes each clip ONCE (no looping) so the .mp4 is finite.
    if args.save_video:
        cfg.setdefault("display", {})["loop_clips"] = False

    # Pre-download the weights once on the main thread so the two workers don't
    # race the first-run download.
    print(f"[init] loading model '{cfg['model']}' (auto-downloads on first run)...")
    PoseDetector.ensure_weights(cfg["model"])

    # One shared OSNet ReID extractor (stateless inference) for both Deep-EIoU
    # trackers; building it here also pre-triggers the weight download.
    reid = None
    if cfg.get("use_reid", True):
        print("[init] loading OSNet ReID (auto-downloads on first run)...")
        reid = build_reid(cfg)

    players = enrollment.load_players(players_path)
    if players:
        print(f"[init] loaded {len(players)} enrolled player(s) for jersey-color naming")
    else:
        print("[init] no enrolled players (run --enroll); labels fall back to ID n")

    # Keep the two dev clips frame-aligned (the live RTSP streams are time-synced
    # by the cameras; recorded clips otherwise drift apart at different CPU rates).
    sync = (threading.Barrier(len(cameras))
            if cfg.get("display", {}).get("sync_feeds", True) else None)
    workers = [CameraWorker(cam, cfg, reid_extractor=reid, players=players, sync_barrier=sync)
               for cam in cameras]
    for w in workers:
        w.start()

    disp = cfg.get("display", {})
    dw, dh = int(disp.get("width", 960)), int(disp.get("height", 540))
    names = [w.cam_name for w in workers]

    minimap = fusion = None
    if args.minimap:
        minimap = Minimap(tuple(cfg.get("court_real_size", [20.0, 10.0])),
                          cfg.get("net_x", 10.0), cfg.get("service_line_from_back", 3.0))
        fusion = Fusion(roster=players,
                        court_size=tuple(cfg.get("court_real_size", [20.0, 10.0])),
                        overlap_zone=tuple(cfg.get("overlap_zone", [7.0, 13.0])),
                        max_players=cfg.get("max_players", 4))
        if fusion.roster_active:
            print(f"[fusion] ROSTER mode: {len(fusion.players)} enrolled players matched by "
                  "OSNet ReID (one-to-one — no shared/swapped IDs).")
        else:
            print("[fusion] TRANSIENT mode (no ReID roster). Run --enroll to register the 4 "
                  "players (name + team) for stable, named, one-per-player IDs.")
        if not any(w.H is not None for w in workers):
            print("[minimap] no homography calibrated yet — run --setup-homography. "
                  "The court template shows; fused dots appear once both cams are calibrated.")

    # Per-camera court-axis flips (to align the two halves into one shared frame).
    # Cameras face each other, so "screen-left" is a different physical side in each;
    # toggle these live (keys 1-4) until the minimap matches reality, then persist to
    # config (cameras[].flip_x / flip_y).
    court_L, court_W = cfg.get("court_real_size", [20.0, 10.0])
    flips = [{"x": bool(c.get("flip_x", False)), "y": bool(c.get("flip_y", False))}
             for c in cameras]

    mm_h = int(disp.get("minimap_height", 300)) if minimap is not None else 0
    feeds_w = len(workers) * dw + 4 * (len(workers) - 1)

    # --- headless: write annotated .mp4 and exit (no GUI) ---
    if args.save_video:
        probe = cv2.VideoCapture(cameras[0]["source"])
        out_fps = probe.get(cv2.CAP_PROP_FPS) or 25.0
        probe.release()
        if out_fps < 1:
            out_fps = 25.0
        try:
            record(workers, names, dw, dh, minimap, fusion, flips, court_L, court_W,
                   mm_h, args.output, out_fps, args.max_frames)
        finally:
            for w in workers:
                w.stop()
            for w in workers:
                w.join(timeout=2.0)
        return 0

    win = "Padel CV - dual camera"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    ww = min(feeds_w, 1760)
    cv2.resizeWindow(win, ww, int((dh + mm_h) * ww / feeds_w))
    cv2.moveWindow(win, 20, 20)

    print("[run] streaming both cameras in one window — press 'q' or ESC to quit. "
          "Press 'f' to view ONE camera full-size (cycles both / cam1 / cam2).")
    if minimap is not None:
        print("      minimap orientation: 1/2 = flip cam1/cam2 LEFT-RIGHT, 3/4 = flip near-far. "
              "Toggle until the two halves line up, then save to config.")
    reported = set()
    view_mode = 0   # 0 = both side by side; 1..N = that camera shown full-size
    try:
        while True:
            for w in workers:
                if w.error and w.cam_name not in reported:
                    print(f"ERROR [{w.cam_name}]: {w.error}", file=sys.stderr)
                    reported.add(w.cam_name)
            if all(w.error for w in workers):
                return 1

            frames = [w.latest() for w in workers]
            canvas = build_canvas(frames, workers, names, dw, dh, minimap, fusion,
                                  flips, court_L, court_W, mm_h, time.perf_counter(), view_mode)
            cv2.imshow(win, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("f"):                     # cycle: both -> cam1 -> cam2 -> both
                view_mode = (view_mode + 1) % (len(workers) + 1)
                print(f"[view] {'both cameras' if view_mode == 0 else names[view_mode-1] + ' (full-size)'}")
            # live minimap-orientation flips: 1/2 = cam1/cam2 left-right (y), 3/4 = near-far (x)
            if minimap is not None and key in (ord("1"), ord("2"), ord("3"), ord("4")):
                ci = 0 if key in (ord("1"), ord("3")) else 1
                ax = "y" if key in (ord("1"), ord("2")) else "x"
                flips[ci][ax] = not flips[ci][ax]
                print(f"[flip] {names[ci]} flip_{ax} = {flips[ci][ax]}   "
                      f"(persist in config.json as cameras[{ci}].flip_{ax})")
            if not any(w.is_alive() for w in workers):
                break
            time.sleep(0.001)
    finally:
        for w in workers:
            w.stop()
        for w in workers:
            w.join(timeout=2.0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
