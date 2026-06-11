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
import time

import cv2
import numpy as np

from core.detector import PoseDetector
from core.pipeline import CameraWorker
from utils import court_roi, enrollment
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


def parse_args():
    p = argparse.ArgumentParser(description="Padel CV System — dual-camera pipeline")
    p.add_argument("--config", default="config.json", help="path to config.json")
    # Documented flags across the roadmap; only --show is active in Step 1.
    p.add_argument("--show", action="store_true", help="display both camera windows")
    p.add_argument("--minimap", action="store_true", help="(Step 5+) live court minimap")
    p.add_argument("--setup-court", action="store_true", help="(Step 2) edit court polygon")
    p.add_argument("--enroll", action="store_true", help="(Step 4) enroll players")
    p.add_argument("--collect", action="store_true", help="(Step 7) save keypoint windows")
    p.add_argument("--save-report", action="store_true", help="(Step 10) write session report")
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

    players_path = cfg.get("players_registry", "players.json")
    if args.enroll:
        enrollment.enroll(cfg, players_path)
        return 0

    for stub in ("minimap", "collect", "save_report"):
        if getattr(args, stub):
            print(f"[note] --{stub.replace('_', '-')} is not implemented yet "
                  f"(later step). Running detection display.")

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

    workers = [CameraWorker(cam, cfg, reid_extractor=reid, players=players)
               for cam in cameras]
    for w in workers:
        w.start()

    disp = cfg.get("display", {})
    dw, dh = int(disp.get("width", 960)), int(disp.get("height", 540))
    names = [w.cam_name for w in workers]

    win = "Padel CV - dual camera"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(len(workers) * dw + 4 * (len(workers) - 1), 1840), dh)
    cv2.moveWindow(win, 20, 30)

    print("[run] streaming both cameras in one window — press 'q' or ESC to quit.")
    reported = set()
    try:
        while True:
            for w in workers:
                if w.error and w.cam_name not in reported:
                    print(f"ERROR [{w.cam_name}]: {w.error}", file=sys.stderr)
                    reported.add(w.cam_name)
            if all(w.error for w in workers):
                return 1

            frames = [w.latest() for w in workers]
            cv2.imshow(win, compose_canvas(frames, names, dw, dh))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
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
