"""Timeline panel — draws emotion graph, baseline, and trigger markers.

Renders a time-series graph at the bottom of the canvas showing:
- Raw happy value (yellow line)
- Baseline (gray dots)
- Spike area (green/red fill between baseline and current)
- Gate background (green/red tint)
- Trigger markers (red vertical lines)
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from facemoment.moment_detector.extractors.base import Observation
from facemoment.moment_detector.visualize.components import (
    COLOR_DARK_BGR,
    COLOR_WHITE_BGR,
    COLOR_GRAY_BGR,
    COLOR_RED_BGR,
    COLOR_GREEN_BGR,
    COLOR_HAPPY_BGR,
    FONT,
)


class TimelinePanel:
    """Time-series graph for the bottom panel.

    Tracks history of:
    - Raw happy values
    - Baseline (slow EWMA)
    - Spike (recent - baseline)
    - Gate open/closed state
    - Trigger timestamps
    """

    def __init__(self, max_history: int = 300):
        self._max_history = max_history
        self._happy_history: List[float] = []
        self._baseline_history: List[float] = []
        self._spike_history: List[float] = []
        self._gate_history: List[bool] = []
        self._trigger_times: List[int] = []

    def reset(self) -> None:
        self._happy_history.clear()
        self._baseline_history.clear()
        self._spike_history.clear()
        self._gate_history.clear()
        self._trigger_times.clear()

    @property
    def trigger_count(self) -> int:
        return len(self._trigger_times)

    def update(
        self,
        face_obs: Optional[Observation],
        fusion_result: Optional[Observation],
        is_gate_open: bool,
    ) -> None:
        """Record one frame's data into the history."""
        happy = 0.0
        if face_obs is not None:
            happy = face_obs.signals.get("expression_happy", 0)

        self._happy_history.append(happy)
        self._gate_history.append(is_gate_open)

        # Adaptive summary from fusion result
        adaptive_summary = None
        if fusion_result and fusion_result.metadata:
            adaptive_summary = fusion_result.metadata.get("adaptive_summary")

        if adaptive_summary and adaptive_summary.get("states"):
            first_state = next(iter(adaptive_summary["states"].values()), None)
            if first_state:
                self._baseline_history.append(first_state["baseline"])
                self._spike_history.append(first_state["spike"])
            else:
                self._baseline_history.append(happy)
                self._spike_history.append(0)
        else:
            self._baseline_history.append(happy)
            self._spike_history.append(0)

        # Trigger marker
        if fusion_result and fusion_result.should_trigger:
            self._trigger_times.append(len(self._happy_history))

        # Trim
        if len(self._happy_history) > self._max_history:
            self._happy_history.pop(0)
            self._baseline_history.pop(0)
            self._spike_history.pop(0)
            self._gate_history.pop(0)
            self._trigger_times = [t - 1 for t in self._trigger_times if t > 1]

    def draw(
        self,
        canvas: np.ndarray,
        region: Tuple[int, int, int, int],
        threshold: float = 0.12,
    ) -> None:
        """Draw the timeline graph onto the canvas.

        Args:
            canvas: Full canvas to draw on.
            region: (x1, y1, x2, y2) for this panel.
            threshold: Spike threshold for coloring.
        """
        rx1, ry1, rx2, ry2 = region
        margin = 8
        x = rx1 + margin
        y = ry1 + margin
        width = rx2 - rx1 - margin * 2
        height = ry2 - ry1 - margin * 2

        if len(self._happy_history) < 2:
            cv2.putText(
                canvas, "Waiting for data...",
                (x + 10, y + height // 2),
                FONT, 0.35, COLOR_GRAY_BGR, 1,
            )
            return

        # Gate background
        for i, gate in enumerate(self._gate_history):
            px = x + int(i * width / self._max_history)
            cv2.line(
                canvas, (px, y), (px, y + height),
                (40, 60, 40) if gate else (60, 40, 40), 1,
            )

        # Threshold line
        th_y = y + height - int(threshold * height * 2)
        cv2.line(canvas, (x, th_y), (x + width, th_y), (100, 100, 100), 1)

        # Baseline (gray dots)
        if self._baseline_history:
            for i in range(0, len(self._baseline_history), 3):
                px = x + int(i * width / self._max_history)
                py = y + height - int(self._baseline_history[i] * height)
                cv2.circle(canvas, (px, py), 1, COLOR_GRAY_BGR, -1)

        # Raw happy line (yellow)
        pts = [
            (
                x + int(i * width / self._max_history),
                y + height - int(v * height),
            )
            for i, v in enumerate(self._happy_history)
        ]
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], COLOR_HAPPY_BGR, 1)

        # Spike area
        if self._spike_history and self._baseline_history:
            for i in range(len(self._spike_history)):
                if i >= len(self._baseline_history):
                    break
                px = x + int(i * width / self._max_history)
                base_y = y + height - int(self._baseline_history[i] * height)
                spike = self._spike_history[i]
                if spike > 0:
                    spike_h = int(spike * height)
                    color = COLOR_RED_BGR if spike > threshold else (0, 180, 0)
                    cv2.line(canvas, (px, base_y), (px, base_y - spike_h), color, 1)

        # Trigger markers
        for t in self._trigger_times:
            if t < len(self._happy_history):
                px = x + int(t * width / self._max_history)
                cv2.line(canvas, (px, y), (px, y + height), COLOR_RED_BGR, 2)

        # Legend
        cv2.putText(canvas, "Happy", (x + 5, y + 12), FONT, 0.3, COLOR_HAPPY_BGR, 1)
        cv2.putText(canvas, "base", (x + 45, y + 12), FONT, 0.3, COLOR_GRAY_BGR, 1)
        cv2.putText(
            canvas, f"th={threshold:.2f}", (x + width - 50, y + 12),
            FONT, 0.25, COLOR_GRAY_BGR, 1,
        )

        # Frame info on right side
        cv2.putText(
            canvas, f"{len(self._happy_history)} frames",
            (x + width - 60, y + height - 4),
            FONT, 0.25, COLOR_GRAY_BGR, 1,
        )
