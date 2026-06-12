"""utils/enrollment.py — player enrollment (Step 4).

`python main.py --enroll` plays one camera, detects on-court players, shows each
one's auto-detected jersey color, and lets you assign a real name. The name +
jersey color (+ HSV reference) are written to players.json — one registry shared by
the whole system (both cameras). Enrollment accumulates across runs.
"""
from __future__ import annotations

import json

import cv2

from core.detector import PoseDetector
from utils import court_roi, jersey
from utils.reid import build_reid


def load_players(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh).get("players", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_players(players, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"players": players}, fh, indent=2)
        fh.write("\n")


def enroll(cfg, players_path):
    cam = cfg["cameras"][cfg.get("enroll_camera_index", 0)]
    det_cfg = cfg.get("detection", {})
    detector = PoseDetector(
        cfg["model"],
        conf=det_cfg.get("conf", 0.25), iou=det_cfg.get("iou", 0.45),
        device=det_cfg.get("device", "cpu"), imgsz=det_cfg.get("imgsz", 640),
        min_keypoints=det_cfg.get("min_keypoints", 0),
        min_aspect=det_cfg.get("min_aspect", 0.0), kpt_conf=det_cfg.get("kpt_conf", 0.3),
    )
    reid = build_reid(cfg)
    if reid is None:
        print("[enroll] WARNING: use_reid is false — players will have no appearance "
              "signature; roster identity needs ReID. Set use_reid:true and re-enroll.")
    polygon = cam.get("court_polygon") or []
    by_name = {p["name"]: p for p in load_players(players_path)}

    cap = cv2.VideoCapture(cam["source"])
    if not cap.isOpened():
        print(f"[enroll] could not open {cam['source']}")
        return
    win = f"enroll [{cam.get('name', 'cam')}]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    print("[enroll] press a DIGIT (0-9) over a player, then type their NAME and TEAM (1/2) "
          "in the console. Enroll all 4 players. Q/ESC = finish.")

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        on_court = [d for d in detector.detect(frame)
                    if court_roi.inside_polygon(
                        court_roi.foot_point(d.bbox, d.keypoints), polygon)]
        info = []
        disp = frame.copy()
        for i, d in enumerate(on_court):
            cname, hsv = jersey.jersey_color(frame, d.bbox, d.keypoints)
            info.append((cname, hsv, d))
            x1, y1, x2, y2 = (int(v) for v in d.bbox)
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(disp, f"{i}: {cname}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, f"DIGIT=name player  Q=finish   enrolled: {len(by_name)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(disp, f"DIGIT=name player  Q=finish   enrolled: {len(by_name)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, disp)

        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        if ord("0") <= k <= ord("9"):
            i = k - ord("0")
            if i < len(info):
                cname, hsv, d = info[i]
                print(f"[enroll] player #{i}: jersey '{cname}' "
                      f"HSV~{tuple(round(x) for x in hsv)}")
                name = input("  name (blank to cancel): ").strip()
                if name:
                    team = input("  team (1 or 2): ").strip()
                    rec = {
                        "name": name,
                        "team": int(team) if team in ("1", "2") else 0,
                        "jersey_color": cname,
                        "jersey_hsv": [round(x, 1) for x in hsv],
                        "reid": None,
                    }
                    if reid is not None:
                        emb = reid.extract(frame, [d.bbox])
                        if emb is not None and len(emb) > 0:
                            rec["reid"] = [round(float(x), 5) for x in emb[0]]
                    by_name[name] = rec
                    save_players(list(by_name.values()), players_path)
                    print(f"  saved {name} (team {rec['team']}) -> {cname}"
                          f"{' +ReID' if rec['reid'] else ' (no ReID!)'}")

    cap.release()
    cv2.destroyWindow(win)
    save_players(list(by_name.values()), players_path)
    print(f"[enroll] {len(by_name)} player(s) written to {players_path}")
