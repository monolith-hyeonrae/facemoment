"""Layout manager for panel-based debug visualization.

Manages a canvas split into three regions:
- Video panel (left ~75%): original video + minimal annotations
- Stats panel (right ~25%): FPS, timing bars, gate status, thumbnails
- Timeline panel (bottom): emotion graph + trigger markers
"""

from typing import Tuple

import cv2
import numpy as np

from facemoment.moment_detector.visualize.components import (
    COLOR_PANEL_BG,
    COLOR_PANEL_BORDER,
)


class LayoutManager:
    """Manages panel-based canvas layout.

    The canvas is divided into:

        +-------------------+----------+
        |                   |  Stats   |
        |  Video Panel      |  Panel   |
        |                   |          |
        +-------------------+----------+
        |         Timeline Panel       |
        +------------------------------+

    Args:
        video_width: Width of the video frame.
        video_height: Height of the video frame.
        side_width: Width of the stats panel (right side).
        bottom_height: Height of the timeline panel (bottom).
    """

    def __init__(
        self,
        video_width: int,
        video_height: int,
        side_width: int = 220,
        bottom_height: int = 120,
    ):
        self.video_width = video_width
        self.video_height = video_height
        self.side_width = side_width
        self.bottom_height = bottom_height

        # Total canvas size
        self.canvas_width = video_width + side_width
        self.canvas_height = video_height + bottom_height

    def create_canvas(self) -> np.ndarray:
        """Create the full canvas with panel backgrounds."""
        canvas = np.zeros(
            (self.canvas_height, self.canvas_width, 3), dtype=np.uint8
        )

        # Stats panel background
        sx1, sy1, sx2, sy2 = self.stats_region()
        cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), COLOR_PANEL_BG, -1)
        cv2.line(canvas, (sx1, sy1), (sx1, sy2), COLOR_PANEL_BORDER, 1)

        # Timeline panel background
        tx1, ty1, tx2, ty2 = self.timeline_region()
        cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), COLOR_PANEL_BG, -1)
        cv2.line(canvas, (tx1, ty1), (tx2, ty1), COLOR_PANEL_BORDER, 1)

        return canvas

    def place_video(self, canvas: np.ndarray, video_frame: np.ndarray) -> None:
        """Place the video frame onto the canvas (video region)."""
        vx1, vy1, vx2, vy2 = self.video_region()
        h, w = video_frame.shape[:2]
        # Resize if needed (video dimensions may differ)
        if w != (vx2 - vx1) or h != (vy2 - vy1):
            video_frame = cv2.resize(video_frame, (vx2 - vx1, vy2 - vy1))
        canvas[vy1:vy2, vx1:vx2] = video_frame

    def video_region(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) for the video panel."""
        return (0, 0, self.video_width, self.video_height)

    def stats_region(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) for the stats panel."""
        return (
            self.video_width,
            0,
            self.canvas_width,
            self.video_height,
        )

    def timeline_region(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) for the timeline panel."""
        return (
            0,
            self.video_height,
            self.canvas_width,
            self.canvas_height,
        )
