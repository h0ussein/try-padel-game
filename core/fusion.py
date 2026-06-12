"""core/fusion.py — cross-camera fusion → global player IDs (Step 6, ACCURACY GATE).

Two modes:

ROSTER mode (when players.json is enrolled with ReID signatures): identity is anchored
to the fixed roster of enrolled players (2 teams x 2). Every frame, each camera's
tracks are matched ONE-TO-ONE to the roster by OSNet ReID appearance (primary, tells
teammates in the same jersey apart) + jersey + position. Because the assignment is a
one-to-one Hungarian against a fixed roster, two players can never share an id and a
player can never hold two ids. Identity is appearance-based, so crossing sides — or
teams switching ends between games — does not cause a swap.

TRANSIENT mode (no roster): falls back to position+jersey clustering with stable ids
(the original behaviour), so the minimap still works before enrollment.
"""
from __future__ import annotations

import threading

import numpy as np
from scipy.optimize import linear_sum_assignment


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _reid_dist(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _jersey_penalty(j1, j2, w=0.8):
    if not j1 or not j2 or j1 == "unknown" or j2 == "unknown":
        return 0.0
    return 0.0 if j1 == j2 else w


def match_tracks_to_roster(reids, jerseys, roster, reid_gate=0.35):
    """Per-camera one-to-one match of tracks to the enrolled roster by OSNet ReID
    (+ jersey tiebreak). Returns a list of names (or None) aligned with the tracks —
    the SAME identity the fused minimap uses, so feed labels and minimap agree.

    reids:   list of per-track embeddings (or None).
    jerseys: list of per-track (color_name, hsv).
    roster:  enrolled players, each {name, jersey_color, reid:[...]}.
    """
    n = len(reids)
    ros = [p for p in (roster or []) if p.get("reid") is not None]
    if n == 0 or not ros:
        return [None] * n
    cost = np.full((n, len(ros)), 1e3)
    rid = np.full((n, len(ros)), 1.0)
    for i, feat in enumerate(reids):
        if feat is None:
            continue
        for j, p in enumerate(ros):
            rd = _reid_dist(feat, p["reid"])
            rid[i, j] = rd
            jc = jerseys[i][0] if (jerseys and i < len(jerseys) and jerseys[i]) else None
            jm = 1.0 if (jc and jc != "unknown" and p.get("jersey_color")
                         and jc != p["jersey_color"]) else 0.0
            cost[i, j] = rd + 0.15 * jm
    names = [None] * n
    rows, cols = linear_sum_assignment(cost)
    for r, c in zip(rows, cols):
        if rid[r, c] < reid_gate:
            names[r] = ros[c]["name"]
    return names


class Fusion:
    def __init__(self, roster=None, court_size=(20.0, 10.0), overlap_zone=(7.0, 13.0),
                 max_players=4, merge_dist=2.0, match_dist=3.0, max_age_sec=2.5,
                 pos_alpha=0.6, reid_gate=0.3, reid_alpha=0.9):
        self.court_size = tuple(court_size)
        self.overlap = tuple(overlap_zone)
        self.max_players = max_players
        self.merge_dist = merge_dist
        self.match_dist = match_dist
        self.max_age = max_age_sec
        self.pos_alpha = pos_alpha
        self.reid_gate = reid_gate          # accept a roster match only if ReID dist < this
        self.reid_alpha = reid_alpha        # EMA weight for the player's running appearance
        self.lock = threading.Lock()

        # roster: list of {name, team, jersey_color, jersey_hsv, reid:[...]}
        self.players = {}
        for i, p in enumerate(roster or []):
            if p.get("reid") is None:
                continue
            self.players[p["name"]] = {
                "gid": i + 1,
                "name": p["name"],
                "team": p.get("team", 0),
                "jersey": p.get("jersey_color"),
                "reid": _normalize(p["reid"]),
                "xy": None,
                "last_seen": -1e9,
                "cams": [],
            }
        self.roster_active = len(self.players) > 0

        # transient-mode state
        self._tr = {}
        self._id = 0

    def update(self, cam_tracks, now):
        with self.lock:
            if self.roster_active:
                self._assign_roster(cam_tracks, now)
                return self._snapshot_roster(now)
            self._transient(cam_tracks, now)
            return self._snapshot_transient(now)

    # --- ROSTER mode ---
    def _assign_roster(self, cam_tracks, now):
        names = list(self.players)
        seen = {nm: [] for nm in names}
        for cam in cam_tracks:
            tracks = [t for t in cam if t.get("xy") is not None and t.get("reid") is not None]
            if not tracks:
                continue
            cost = np.zeros((len(tracks), len(names)))
            rid = np.zeros((len(tracks), len(names)))
            for i, t in enumerate(tracks):
                for j, nm in enumerate(names):
                    p = self.players[nm]
                    rd = _reid_dist(t["reid"], p["reid"])
                    rid[i, j] = rd
                    jm = 0.0 if (not t.get("jersey") or t["jersey"] == p["jersey"]) else 1.0
                    pd = 0.0 if p["xy"] is None else min(_dist(t["xy"], p["xy"]) / 10.0, 1.0)
                    cost[i, j] = rd + 0.15 * jm + 0.1 * pd
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if rid[r, c] < self.reid_gate:        # ReID must agree to accept
                    seen[names[c]].append(tracks[r])

        for nm, obs in seen.items():
            if not obs:
                continue
            p = self.players[nm]
            xy = np.mean([np.asarray(o["xy"], float) for o in obs], axis=0)
            p["xy"] = xy if p["xy"] is None else (1 - self.pos_alpha) * p["xy"] + self.pos_alpha * xy
            feat = np.mean([np.asarray(o["reid"], float) for o in obs], axis=0)
            p["reid"] = _normalize(self.reid_alpha * p["reid"] + (1 - self.reid_alpha) * feat)
            p["cams"] = sorted({o["cam"] for o in obs if "cam" in o})
            p["last_seen"] = now

    def _snapshot_roster(self, now):
        return [{"gid": p["gid"], "name": nm, "team": p["team"],
                 "xy": (float(p["xy"][0]), float(p["xy"][1])), "cams": p["cams"]}
                for nm, p in self.players.items()
                if p["xy"] is not None and now - p["last_seen"] < self.max_age]

    # --- TRANSIENT mode (no roster yet) ---
    def _next_id(self):
        self._id += 1
        return self._id

    def _transient(self, cam_tracks, now):
        obs = []
        a = [t for t in (cam_tracks[0] if cam_tracks else []) if t.get("xy")]
        b = [t for t in (cam_tracks[1] if len(cam_tracks) > 1 else []) if t.get("xy")]
        used_b = set()
        if a and b:
            cost = np.array([[_dist(ta["xy"], tb["xy"]) +
                              _jersey_penalty(ta.get("jersey"), tb.get("jersey"))
                              for tb in b] for ta in a])
            rows, cols = linear_sum_assignment(cost)
            matched_a = set()
            for r, c in zip(rows, cols):
                if cost[r, c] <= self.merge_dist:
                    obs.append(self._obs(a[r], b[c]))
                    matched_a.add(r)
                    used_b.add(c)
            obs += [self._obs(t) for i, t in enumerate(a) if i not in matched_a]
            obs += [self._obs(t) for j, t in enumerate(b) if j not in used_b]
        else:
            obs += [self._obs(t) for t in a + b]

        gids = list(self._tr)
        matched = set()
        if obs and gids:
            cost = np.array([[_dist(o["xy"], self._tr[g]["xy"]) +
                              _jersey_penalty(o["jersey"], self._tr[g]["jersey"])
                              for g in gids] for o in obs])
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= self.match_dist:
                    p = self._tr[gids[c]]
                    p["xy"] = (1 - self.pos_alpha) * p["xy"] + self.pos_alpha * np.asarray(obs[r]["xy"], float)
                    p["jersey"] = obs[r]["jersey"] or p["jersey"]
                    p["last_seen"] = now
                    matched.add(r)
        for i, o in enumerate(obs):
            if i in matched or len(self._tr) >= self.max_players:
                continue
            gid = self._next_id()
            self._tr[gid] = {"gid": gid, "xy": np.asarray(o["xy"], float),
                             "jersey": o["jersey"], "last_seen": now}
        for g in [g for g, p in self._tr.items() if now - p["last_seen"] > self.max_age]:
            del self._tr[g]

    @staticmethod
    def _obs(*tracks):
        xy = np.mean([np.asarray(t["xy"], float) for t in tracks], axis=0)
        jersey = next((t["jersey"] for t in tracks
                       if t.get("jersey") and t["jersey"] != "unknown"), tracks[0].get("jersey"))
        return {"xy": xy, "jersey": jersey, "cams": {t["cam"] for t in tracks if "cam" in t}}

    def _snapshot_transient(self, now):
        return [{"gid": p["gid"], "name": None, "team": 0,
                 "xy": (float(p["xy"][0]), float(p["xy"][1])), "cams": []}
                for p in self._tr.values() if now - p["last_seen"] < self.max_age]
