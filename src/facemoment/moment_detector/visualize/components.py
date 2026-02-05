"""Shared drawing utilities and color constants for visualization.

Provides common drawing primitives used across panels.
"""

from enum import IntEnum
from typing import Dict, Tuple

import cv2
import numpy as np


# --- Debug layers ---

class DebugLayer(IntEnum):
    """Visualization layers that can be toggled at runtime with keys 1-9."""

    FACE = 1        # Face bbox, ID labels, classified roles, emotion bars
    POSE = 2        # Upper body pose skeleton
    GESTURE = 3     # Hand landmarks and gesture labels
    ROI = 4         # ROI boundary rectangle
    STATS = 5       # Stats panel (right side: timing, gate, summary)
    TIMELINE = 6    # Timeline panel (bottom: emotion graph, triggers)
    TRIGGER = 7     # Trigger flash on video + thumbnails in stats
    FUSION = 8      # Fusion info (spike bar, baseline in stats)
    FRAME_INFO = 9  # Frame number / timestamp overlay on video


# Short labels for status display
_LAYER_LABELS: Dict[int, str] = {
    DebugLayer.FACE: "FACE",
    DebugLayer.POSE: "POSE",
    DebugLayer.GESTURE: "GEST",
    DebugLayer.ROI: "ROI",
    DebugLayer.STATS: "STAT",
    DebugLayer.TIMELINE: "TIME",
    DebugLayer.TRIGGER: "TRIG",
    DebugLayer.FUSION: "FUSE",
    DebugLayer.FRAME_INFO: "INFO",
}


class LayerState:
    """Tracks which debug visualization layers are enabled.

    All layers start enabled. Toggle with ``toggle(layer)``.
    Check with ``self[layer]`` or ``is_enabled(layer)``.
    """

    def __init__(self):
        self._enabled: Dict[DebugLayer, bool] = {
            layer: True for layer in DebugLayer
        }

    def toggle(self, layer: DebugLayer) -> bool:
        """Toggle a layer. Returns the new state (True=on)."""
        self._enabled[layer] = not self._enabled[layer]
        return self._enabled[layer]

    def is_enabled(self, layer: DebugLayer) -> bool:
        return self._enabled[layer]

    def __getitem__(self, layer: DebugLayer) -> bool:
        return self._enabled[layer]


# --- BGR color constants ---

# General
COLOR_DARK_BGR = (40, 40, 40)
COLOR_WHITE_BGR = (255, 255, 255)
COLOR_GRAY_BGR = (128, 128, 128)
COLOR_RED_BGR = (0, 0, 255)
COLOR_GREEN_BGR = (0, 255, 0)

# Emotion colors
COLOR_HAPPY_BGR = (0, 255, 255)  # Yellow
COLOR_ANGRY_BGR = (0, 0, 255)
COLOR_NEUTRAL_BGR = (200, 200, 200)

# Role colors for face classification
COLOR_MAIN_BGR = (0, 255, 0)         # Green
COLOR_PASSENGER_BGR = (0, 165, 255)  # Orange
COLOR_TRANSIENT_BGR = (0, 255, 255)  # Yellow
COLOR_NOISE_BGR = (128, 128, 128)    # Gray

# Pose skeleton colors
COLOR_SKELETON_BGR = (255, 200, 100)  # Light blue
COLOR_KEYPOINT_BGR = (0, 255, 255)    # Yellow

# Panel background
COLOR_PANEL_BG = (30, 30, 30)
COLOR_PANEL_BORDER = (60, 60, 60)

# Role-to-color mapping
ROLE_COLORS = {
    "main": COLOR_MAIN_BGR,
    "passenger": COLOR_PASSENGER_BGR,
    "transient": COLOR_TRANSIENT_BGR,
    "noise": COLOR_NOISE_BGR,
}

# Font settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.33
FONT_MEDIUM = 0.4
FONT_LARGE = 0.5


def draw_horizontal_bar(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    value: float,
    color: Tuple[int, int, int],
    bg_color: Tuple[int, int, int] = (50, 50, 50),
) -> None:
    """Draw a horizontal progress bar.

    Args:
        image: Target image.
        x, y: Top-left corner.
        width, height: Bar dimensions.
        value: Fill ratio [0, 1].
        color: Fill color (BGR).
        bg_color: Background color (BGR).
    """
    cv2.rectangle(image, (x, y), (x + width, y + height), bg_color, -1)
    fill_w = int(width * min(1.0, max(0.0, value)))
    if fill_w > 0:
        cv2.rectangle(image, (x, y), (x + fill_w, y + height), color, -1)


def draw_label_with_bg(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_scale: float = FONT_MEDIUM,
    text_color: Tuple[int, int, int] = COLOR_DARK_BGR,
    bg_color: Tuple[int, int, int] = COLOR_GREEN_BGR,
    thickness: int = 1,
    padding: int = 2,
) -> None:
    """Draw text with a filled background rectangle."""
    text_size = cv2.getTextSize(text, FONT, font_scale, thickness)[0]
    cv2.rectangle(
        image,
        (x, y - text_size[1] - padding),
        (x + text_size[0] + padding * 2, y + padding),
        bg_color,
        -1,
    )
    cv2.putText(image, text, (x + padding, y), FONT, font_scale, text_color, thickness)


def timing_color(ms: float, threshold: float = 50.0) -> Tuple[int, int, int]:
    """Return color based on timing value.

    Green for fast, yellow for moderate, red for slow.
    """
    if ms < threshold * 0.6:
        return COLOR_GREEN_BGR
    elif ms < threshold:
        return COLOR_HAPPY_BGR
    else:
        return COLOR_RED_BGR
