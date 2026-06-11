"""core/trackers/basetrack.py — track state enum + base track (Step 3)."""
from __future__ import annotations

import numpy as np


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """Base bookkeeping for a track. Track IDs are assigned by the owning tracker
    (per-camera counter) — NOT a global, so the two cameras number independently
    and the two tracker threads never race a shared counter."""

    def __init__(self):
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.score = 0.0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        self.location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed
