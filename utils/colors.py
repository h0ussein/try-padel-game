"""utils/colors.py — stable per-track-ID color palette (Step 3).

Each track ID maps to a fixed, visually distinct BGR color (golden-ratio hue
spacing) so a player keeps the same color for as long as their ID holds.
"""
from __future__ import annotations

import colorsys

_CACHE = {}


def color_for_id(track_id):
    """Return a stable BGR tuple for an integer track ID."""
    if track_id not in _CACHE:
        h = (int(track_id) * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.98)
        _CACHE[track_id] = (int(b * 255), int(g * 255), int(r * 255))
    return _CACHE[track_id]
