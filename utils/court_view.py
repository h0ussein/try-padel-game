"""utils/court_view.py — live top-down minimap of the real padel court (Step 5).

Draws the actual padel-court template (outer boundary, net, the two service lines,
and the center service line between them) and overlays player dots projected from
each camera's homography into the shared court frame.
"""
from __future__ import annotations

import cv2
import numpy as np

_TURF = (70, 115, 70)      # green-ish background (BGR)
_WHITE = (245, 245, 245)
_NET = (40, 40, 235)       # red net line


class Minimap:
    def __init__(self, court_size=(20.0, 10.0), net_x=10.0, service_from_back=3.0,
                 ppm=40, margin=26):
        self.L, self.W = court_size
        self.net_x = net_x
        self.sfb = service_from_back
        self.ppm = ppm
        self.margin = margin
        self.w = int(self.L * ppm) + 2 * margin
        self.h = int(self.W * ppm) + 2 * margin
        self.base = self._draw_template()

    def to_px(self, x, y):
        return (int(self.margin + x * self.ppm), int(self.margin + y * self.ppm))

    def _draw_template(self):
        img = np.full((self.h, self.w, 3), _TURF, np.uint8)
        # outer court boundary
        cv2.rectangle(img, self.to_px(0, 0), self.to_px(self.L, self.W), _WHITE, 2)
        # net (across the full width at net_x)
        cv2.line(img, self.to_px(self.net_x, 0), self.to_px(self.net_x, self.W), _NET, 2)
        # two service lines, service_from_back metres in front of each back wall
        sb, sf = self.sfb, self.L - self.sfb
        for sx in (sb, sf):
            cv2.line(img, self.to_px(sx, 0), self.to_px(sx, self.W), _WHITE, 1)
        # center service line (y = W/2), spanning between the two service lines
        cv2.line(img, self.to_px(sb, self.W / 2), self.to_px(sf, self.W / 2), _WHITE, 1)
        return img

    def render(self, players):
        """players: iterable of (court_x, court_y, color_bgr, label)."""
        img = self.base.copy()
        for x, y, color, label in players:
            px = self.to_px(x, y)
            cv2.circle(img, px, 7, color, -1, cv2.LINE_AA)
            cv2.circle(img, px, 7, (0, 0, 0), 1, cv2.LINE_AA)
            if label:
                cv2.putText(img, str(label), (px[0] + 9, px[1] - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        return img
