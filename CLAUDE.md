# Padel CV System — Build Guide (for Claude / future sessions)

Real-time computer-vision system for a padel court at **InMobiles Holding** (Mkalles,
Lebanon). Two fixed Hikvision cameras, each covering 50–60% of a 20 m × 10 m court
with overlap around the net; that overlap is what resolves the occlusion a single
camera cannot. Built **accuracy-first**, one ordered step at a time.

Authoritative spec: `PadelCV_Roadmap_StepByStep (2).pdf` / `.docx` in the project
root. Read it before changing architecture.

## HARD RULES (never violate)
- **Two cameras everywhere.** No single-camera path. `config.json` must define
  exactly 2 cameras; `main.py` runs one capture/inference thread per camera in
  parallel + a coordinator (from Step 6).
- **Tracker is Deep-EIoU (+ OSNet ReID).** Never ByteTrack. One tracker instance
  per camera.
- **Action recognition is a TRAINED neural network** (LSTM → ST-GCN) over
  normalized 30-frame × 17-keypoint windows. **No rule-based action logic.**
- **Court ROI is an auto-detected, EDITABLE polygon.** Players are kept only if
  their FEET (ankle midpoint; fallback bbox bottom-center) are inside it.
- **Homography maps to the REAL padel-court layout** (sidelines, service lines,
  center service line, net). Both cameras use the **same shared reference points**.
- **Config-driven only.** No hardcoded paths/IPs in code — everything in
  `config.json`.
- **One step at a time.** Build a step, stop, let the user run + verify against the
  step's checklist, then continue. Do not build ahead.

## Tech stack (Section 2 of the spec)
- Detection + skeleton: **YOLOv11-Pose** (`ultralytics`, `yolo11n-pose.pt`) — bbox
  + 17 COCO keypoints.
- Tracking: **Deep-EIoU** (`boxmot` or the `hsiangwei0903/Deep-EIoU` repo) + **OSNet**
  ReID (`torchreid`).
- Identity: OSNet embeddings + jersey color (HSV) + court homography.
- Actions: LSTM → ST-GCN (MMAction2). Ball: MOG2 → custom YOLO → TrackNet.
- Video/geometry: `opencv-python`, `numpy`. Reports: `fpdf2` + JSON. Optional web:
  `flask`. Edge: NVIDIA Jetson + TensorRT (`.engine`).

## File layout (Section 3.2)
```
main.py              entry point — load config, launch 2 camera threads (+ coordinator)
config.json          2 camera sources, court polygons, homography paths, settings
players.json         enrolled names + jersey colors
core/pipeline.py     per-camera worker loop                    [Step 1 ✓]
core/detector.py     YOLOv11-Pose wrapper                      [Step 1 ✓]
core/tracker.py      Deep-EIoU wrapper (motion+OSNet+keypoints)[Step 3 ✓]
core/trackers/       vendored Deep-EIoU (STrack/KF/EIoU match)  [Step 3 ✓]
core/fusion.py       cross-camera merge → global IDs           [Step 6 ✓ — accuracy gate]
core/actions.py      action detection (trained NN)             [Step 8]
core/ball.py         ball detection + speed                    [Step 9]
core/report.py       session logging + JSON/PDF                [Step 10]
utils/skeleton.py    COCO-17 drawing + HUD/FPS                 [Step 1 ✓]
utils/court_roi.py   auto-detect/edit court polygon + foot filter [Step 2 ✓]
utils/jersey.py      jersey color (HSV) + name match           [Step 4 ✓]
utils/enrollment.py  player enrollment → players.json          [Step 4 ✓]
utils/homography.py  calibration + pixel→court mapping         [Step 5 ✓]
utils/court_view.py  live top-down minimap                     [Step 5 ✓]
utils/reid.py        OSNet embeddings (boxmot ReID)            [Step 3 ✓]
utils/clips.py       auto-clip export                          [Step 10]
utils/heatmap.py     position heatmap                          [Step 10]
utils/colors.py      per-ID color palette                      [Step 3 ✓]
models/ data/ output/ calibration/   weights, training data, outputs, calibration
```

## Config keys (Section 3.3)
`cameras` (exactly 2; each has `source`, `coverage` 'left'/'right', `court_polygon`,
`homography`, `calibration`), `court_real_size` [20,10], `net_x`, `overlap_zone`,
`tracker` ('deep_eiou'), `reid_weights`, `model` (`yolo11n-pose.pt` or `.engine`),
`detection` (conf/iou/device/imgsz), `display`. During dev, camera `source` points
at local clips; in production it is the RTSP URL.

## Build progress
- **Step 1 — Dual-Camera Capture & Detection — DONE.** Two threads, YOLOv11-Pose,
  COCO-17 skeleton + FPS HUD, dev sources = `side1-1m.mp4` / `side2-1m.mp4`.
- **Step 2 — Court Polygon + Foot Filter — DONE.** `utils/court_roi.py`: auto-detect
  a court-polygon guess per camera (gray-surface contour, trapezoid fallback — single
  -frame color seg is unreliable on this night court, so the editable step is the real
  fit), `--setup-court` click-drag editor saves to `cameras[].court_polygon`, and the
  pipeline keeps a player only if their FEET (ankle midpoint / bbox bottom-center) are
  inside. With no polygon set, all detections are kept (no regression).
- **Step 3 — Deep-EIoU Tracking (+ OSNet ReID) — DONE.** Vendored Deep-EIoU in
  `core/trackers/` (STrack + BoT-SORT Kalman + ExpansionIoU iterative scale-up
  e=0.7/0.8 + BYTE high/low split + ReID fusion `min(iou,emb)`), one tracker per
  camera via `core/tracker.py`. OSNet ReID via boxmot (`utils/reid.py`, default
  `osnet_x0_25_msmt17.pt`, auto-downloaded). Stable per-ID color (`utils/colors.py`).
  Thresholds tuned for this dim footage and exposed in `config.json.tracker_params`
  (`new_track_thresh` lowered to 0.5 — players run ~0.6-0.85 conf; `track_buffer` 60
  so a briefly-lost player recovers their ID instead of spawning a new one). Per-camera
  `max_players` cap (config, default 4) keeps the most-established tracks. Single tracker
  display window (both feeds tiled — user preference). IDs are per-camera here; the same
  player gets one ID PER camera — that's unified into one global ID, and the global
  4-player cap enforced, by fusion in Step 6 (needs Step 5 shared-court homography first).
- **Step 4 — Jersey Color + Enrollment — DONE.** `utils/jersey.py`: dominant torso
  color (HSV, sampled shoulders→hips via keypoints) → named color; `assign_names`
  one-to-one matches tracks to enrolled players (scipy) so the NAME follows the jersey
  even when the tracker ID swaps. `utils/enrollment.py`: `--enroll` plays one camera,
  shows each player's jersey color, press a DIGIT + type a name → `players.json`.
  Pipeline labels tracks by matched name (fallback `ID n`). NOTE: on this dim night
  court jersey color is COARSE (reliably separates light/white/"gray" vs navy/"blue");
  two same-team same-shirt players can't be split by jersey alone — that's what ReID +
  court position + fusion are for. Saturation bar raised so warm-lit white isn't called
  orange. Verified headless: per-player color is consistent across frames and names
  re-match; could not test the interactive enroll keypress/console flow.
- **Step 5 — Dual Homography + Minimap — DONE.** `utils/homography.py`: `--setup-homography`
  is a **4-click per-camera** calibration (2 NET posts + that camera's 2 BACK corners) with an
  INSTANT reprojection preview (court drawn back on the frame; Y=accept/R=redo). Camera-aware:
  cam0 back wall x=0, cam1 back wall x=L, net (x=net_x) shared → links the two halves.
  `calibration/camN_H.npy`, `pixel_to_court` projection. CAMERA GEOMETRY (confirmed by user):
  cameras are BEHIND the baselines, each sees ~50% (split at the net); the net is the shared
  anchor. NOTE: fully-AUTOMATIC court-line detection was attempted (multi-approach) but is NOT
  reliable on this dim/cluttered night footage (fence/glass/2 courts/raised net) — deferred as a
  later goal; the 4-click + preview is the reliable path for these fixed cameras. The previous
  13-named-point clicker + the service-line-3m-from-net assumption are GONE (service lines are
  3 m from the BACK WALLS; key `service_line_from_back`).
  `utils/court_view.py` `Minimap`: renders the real padel court (boundary, net, 2 service lines,
  center service line) + projected player dots. Pipeline loads each cam's H, projects foot points,
  publishes court positions; `main.py --show --minimap` tiles the shared minimap beneath the two
  feeds (cam1 dots yellow, cam2 magenta). Geometry config-driven: `court_real_size`, `net_x`,
  `service_line_gap` (spec says 3 m from net; real padel is ~6.95 m — adjust in config if needed).
  Verified headless: template renders correctly, homography round-trips at 0.0 m error, feet
  project inside the court. NOT yet calibrated (no .npy) — user must run `--setup-homography`
  clicking the SAME physical points on BOTH cameras, else they don't share a frame and Step 6
  fusion can't match. Could not test the interactive click calibration or live minimap window.
- **Step 6 — Cross-Camera Fusion — DONE (ACCURACY GATE).** `core/fusion.py` `Fusion`:
  (1) MERGES both cameras' tracks of the same physical player by shared court position
  (primary) + jersey (soft cue) within `merge_dist`; (2) ASSIGNS stable GLOBAL ids by
  matching merged observations to a persistent registry (court pos + jersey), behind a
  `threading.Lock`; (3) enforces the global 4-player cap, time-based aging (`max_age_sec`,
  rate-independent). Workers now publish `tracks_info()` (court xy + jersey + name + local
  id); `main.py --show --minimap` runs fusion each frame and the minimap shows ONE dot per
  GLOBAL player (per-gid color, enrolled name). Tuned `pos_alpha=0.6`, `match_dist=3.0` for
  low-FPS movement. Verified headless: same player on both cams→1 id; a player crossing
  cam0→overlap→cam1 with a local id swap 1→7 holds ONE global id across the handoff (the
  user's exact bug); 2 players→2 ids; 6→capped at 4. REQUIRES `--setup-homography` done on
  BOTH cams (shared frame) — else positions don't align and fusion mis-merges. The live
  net-exchange test on the fused minimap is the user's hardware verification.
- Steps 7–11: not started (stubs in place). Next: Step 7 (action data collection & labeling).

## How to run
```
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt      # includes boxmot (OSNet ReID + lapx)
python main.py --setup-court      # Step 2: drag court corners per camera, S=save
python main.py --setup-homography # Step 5: click SAME court points on BOTH cams -> H
python main.py --enroll           # Step 4: digit-key a player, type name -> players.json
python main.py --show --minimap   # 'q' or ESC to quit; 'f' = one camera full-size; 1-4 = flip minimap axes
python main.py --save-video --minimap --output output/out.mp4   # HEADLESS: write annotated mp4 (servers/GPU, no window)
```
`yolo11n-pose.pt` + the OSNet ReID weights auto-download on first run. CPU is the
default device (`detection.device` in config) — set to `cuda` if a GPU is available.
On CPU the dual stream + ReID is slow (~1-3 FPS); that is expected (Jetson/TensorRT
is Step 11). Set `"use_reid": false` to run Deep-EIoU motion-only (still not ByteTrack).

## Verify checklists
Step 1: two feeds play in parallel; every player has a bbox + 17-joint skeleton;
an FPS counter shows per stream (top-left HUD).
Step 2: `--setup-court` shows an auto polygon as a starting guess, dragging a corner
saves to config, and on `--show` only players whose feet are inside the polygon are
kept (off-court people draw as a thin gray box); HUD shows `on-court: k/n`.
Step 3: each feed shows stable per-player IDs (colored box + `ID n` + skeleton) that
don't flicker; IDs are per-camera (both feeds number from 1 independently); note any
net-exchange ID swaps — those are fixed by fusion in Step 6. Verified headless: cam1
2 players → IDs [1,2] stable / 40 frames, cam2 3 players → [1,2,3] stable / 30 frames.
Step 4: after `--enroll`, players are labeled by their real NAME in both feeds; the
jersey color holds frame-to-frame, so when a tracker ID swaps the name re-attaches via
the jersey re-check (vs flipping to `ID n`). Best with distinct, saturated shirts.
Step 5: `--setup-homography` = 4 clicks/camera (2 NET posts + 2 BACK corners) with an instant
court-fit preview (GREEN court / RED net drawn on the frame) — accept only if the drawn court
sits on the real lines. Then `--show --minimap` shows the court template + dots. On-the-line
test: a player on a known line lands on that line from EITHER camera. If cam1 & cam2 dots for the
same player are MIRRORED, the user clicked opposite physical net posts as "LEFT" — redo one camera
swapping left/right. (The per-camera preview can look perfect yet still be cross-mirrored; the
fused minimap is what reveals it.)
Step 6 (ACCURACY GATE): the fused minimap shows exactly ONE dot per player (correct name)
through occlusion; a net-crossing that swapped IDs in one feed now HOLDS one global id
because the other camera separates the players. Do not proceed to Step 7 until a real net
exchange holds global IDs on the fused minimap; if it fails, fix homography accuracy /
overlap matching / jersey gate here first.
