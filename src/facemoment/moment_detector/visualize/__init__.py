"""Panel-based visualization for debug views.

Provides a modular visualization system with separate panels:
- VideoPanel: spatial annotations on the video frame
- StatsPanel: timing, FPS, gate status (side panel)
- TimelinePanel: emotion graph and trigger markers (bottom panel)
- LayoutManager: canvas creation and panel placement

Backward compatibility:
- DebugVisualizer is re-exported with the same public API
- VisualizationConfig is available for legacy code
- ExtractorVisualizer and FusionVisualizer are available for direct use
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from visualbase import Frame

from facemoment.moment_detector.extractors.base import Observation
from facemoment.moment_detector.visualize.layout import LayoutManager
from facemoment.moment_detector.visualize.video_panel import VideoPanel
from facemoment.moment_detector.visualize.stats_panel import StatsPanel
from facemoment.moment_detector.visualize.timeline_panel import TimelinePanel
from facemoment.moment_detector.visualize.components import (  # noqa: F401
    COLOR_DARK_BGR,
    COLOR_WHITE_BGR,
    COLOR_GRAY_BGR,
    COLOR_RED_BGR,
    COLOR_GREEN_BGR,
    COLOR_HAPPY_BGR,
    COLOR_ANGRY_BGR,
    COLOR_NEUTRAL_BGR,
    COLOR_MAIN_BGR,
    COLOR_PASSENGER_BGR,
    COLOR_TRANSIENT_BGR,
    COLOR_NOISE_BGR,
    COLOR_SKELETON_BGR,
    COLOR_KEYPOINT_BGR,
    DebugLayer,
    LayerState,
    _LAYER_LABELS,
)


@dataclass
class RenderContext:
    """Bundles all data needed for a single debug frame render.

    Replaces the 11+ individual parameters of create_debug_view().
    """

    frame: Frame
    observations: Dict[str, Observation] = field(default_factory=dict)
    classifier_obs: Optional[Observation] = None
    fusion_result: Optional[Observation] = None
    monitor_stats: Optional[Dict] = None
    is_gate_open: bool = False
    in_cooldown: bool = False
    roi: Optional[Tuple[float, float, float, float]] = None
    profile_timing: Optional[Dict[str, float]] = None
    backend_label: str = ""


@dataclass
class VisualizationConfig:
    """Configuration for visualization (legacy compatibility)."""

    graph_height: int = 100


class ExtractorVisualizer:
    """Legacy wrapper around VideoPanel for backward compatibility.

    Provides the same draw_* methods that test_visualize.py expects.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self._video_panel = VideoPanel()

    def draw_face_observation(self, image: np.ndarray, observation: Observation) -> np.ndarray:
        """Draw face observations on image (legacy API)."""
        output = image.copy()
        self._video_panel._draw_faces(output, observation)
        # Also draw summary text on image for backward compat
        h, w = output.shape[:2]
        face_count = len(observation.faces)
        happy = observation.signals.get("expression_happy", 0)
        angry = observation.signals.get("expression_angry", 0)
        neutral = observation.signals.get("expression_neutral", 1)
        cv2.putText(
            output, f"Faces: {face_count} | H:{happy:.2f} A:{angry:.2f} N:{neutral:.2f}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE_BGR, 1,
        )
        return output

    def draw_face_classifier_observation(
        self,
        image: np.ndarray,
        classifier_obs: Observation,
        face_obs: Optional[Observation] = None,
    ) -> np.ndarray:
        """Draw face classification results (legacy API)."""
        output = image.copy()
        self._video_panel._draw_classified_faces(output, classifier_obs)

        # Summary text
        if classifier_obs.data is not None:
            data = classifier_obs.data
            main_detected = 1 if hasattr(data, "main_face") and data.main_face else 0
            passenger_count = len(data.passenger_faces) if hasattr(data, "passenger_faces") else 0
            transient_count = data.transient_count if hasattr(data, "transient_count") else 0
            noise_count = data.noise_count if hasattr(data, "noise_count") else 0
            summary = f"Main: {main_detected} | Pass: {passenger_count} | Trans: {transient_count} | Noise: {noise_count}"
            cv2.putText(output, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE_BGR, 1)

        return output

    def draw_pose_observation(self, image: np.ndarray, observation: Observation) -> np.ndarray:
        """Draw pose observations (legacy API)."""
        output = image.copy()
        self._video_panel._draw_pose(output, observation)
        # Summary text
        person_count = int(observation.signals.get("person_count", 0))
        hands_raised = int(observation.signals.get("hands_raised_count", 0))
        wave = observation.signals.get("hand_wave_detected", 0) > 0.5
        color = COLOR_HAPPY_BGR if hands_raised > 0 else COLOR_GREEN_BGR
        cv2.putText(
            output, f"Persons: {person_count} | Hands Up: {hands_raised}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )
        if wave:
            cv2.putText(output, "WAVE DETECTED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED_BGR, 2)
        return output

    def draw_quality_observation(self, image: np.ndarray, observation: Observation) -> np.ndarray:
        """Draw quality metrics (legacy API)."""
        output = image.copy()
        h, w = output.shape[:2]

        blur = observation.signals.get("blur_quality", 0)
        bright = observation.signals.get("brightness_quality", 0)
        contrast = observation.signals.get("contrast_quality", 0)
        gate = observation.signals.get("quality_gate", 0)

        bar_x, bar_w, bar_h = w - 120, 80, 12
        for i, (name, val) in enumerate([("Blur", blur), ("Bright", bright), ("Contrast", contrast)]):
            y = 30 + i * 25
            color = COLOR_GREEN_BGR if val >= 1.0 else COLOR_RED_BGR
            cv2.putText(output, name, (bar_x - 55, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WHITE_BGR, 1)
            cv2.rectangle(output, (bar_x, y), (bar_x + bar_w, y + bar_h), COLOR_DARK_BGR, -1)
            cv2.rectangle(output, (bar_x, y), (bar_x + int(bar_w * min(1.0, val)), y + bar_h), color, -1)

        gate_y = 30 + 3 * 25 + 10
        gate_color = COLOR_GREEN_BGR if gate > 0.5 else COLOR_GRAY_BGR
        cv2.putText(
            output, "GATE: OPEN" if gate > 0.5 else "GATE: CLOSED",
            (bar_x - 55, gate_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gate_color, 2,
        )
        return output


class DebugVisualizer:
    """Combined visualizer for pipeline debugging.

    Panel-based layout: video (left) + stats (right) + timeline (bottom).

    Supports both the new RenderContext API and the legacy
    create_debug_view() with individual parameters.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.extractor_viz = ExtractorVisualizer(config)
        self.layers = LayerState()

        self._video_panel = VideoPanel()
        self._stats_panel = StatsPanel()
        self._timeline_panel = TimelinePanel()
        self._layout: Optional[LayoutManager] = None

    def reset(self) -> None:
        self._stats_panel.reset()
        self._timeline_panel.reset()

    # --- New API ---

    def render(self, ctx: RenderContext) -> np.ndarray:
        """Render a debug view from a RenderContext.

        Args:
            ctx: All data needed for one frame's visualization.

        Returns:
            Complete canvas with all panels.
        """
        return self.create_debug_view(
            frame=ctx.frame,
            face_obs=ctx.observations.get("face") or ctx.observations.get("dummy"),
            pose_obs=ctx.observations.get("pose"),
            gesture_obs=ctx.observations.get("gesture"),
            quality_obs=ctx.observations.get("quality"),
            classifier_obs=ctx.classifier_obs,
            fusion_result=ctx.fusion_result,
            is_gate_open=ctx.is_gate_open,
            in_cooldown=ctx.in_cooldown,
            timing=ctx.profile_timing,
            roi=ctx.roi,
            monitor_stats=ctx.monitor_stats,
            backend_label=ctx.backend_label,
        )

    # --- Legacy API (backward compatible) ---

    def create_debug_view(
        self,
        frame: Frame,
        face_obs: Optional[Observation] = None,
        pose_obs: Optional[Observation] = None,
        gesture_obs: Optional[Observation] = None,
        quality_obs: Optional[Observation] = None,
        classifier_obs: Optional[Observation] = None,
        fusion_result: Optional[Observation] = None,
        is_gate_open: bool = False,
        in_cooldown: bool = False,
        timing: Optional[Dict[str, float]] = None,
        roi: Optional[Tuple[float, float, float, float]] = None,
        monitor_stats: Optional[Dict] = None,
        backend_label: str = "",
    ) -> np.ndarray:
        """Create combined debug visualization with panel layout.

        Args:
            frame: Input frame.
            face_obs: Face observation.
            pose_obs: Pose observation.
            gesture_obs: Gesture observation (hand landmarks).
            quality_obs: Quality observation.
            classifier_obs: Face classifier observation.
            fusion_result: Fusion result.
            is_gate_open: Whether gate is open.
            in_cooldown: Whether in cooldown.
            timing: Timing info for profile mode.
            roi: ROI boundary (x1, y1, x2, y2) in normalized coords.
            monitor_stats: PathwayMonitor frame stats.
            backend_label: Backend indicator label.
        """
        h, w = frame.data.shape[:2]

        # Lazy-init layout on first frame (video dimensions known)
        if self._layout is None or self._layout.video_width != w or self._layout.video_height != h:
            self._layout = LayoutManager(w, h)

        # 1. Create canvas
        canvas = self._layout.create_canvas()

        # 2. Video panel: annotate the frame (respects layers)
        video_frame = self._video_panel.draw(
            frame.data,
            face_obs=face_obs,
            pose_obs=pose_obs,
            gesture_obs=gesture_obs,
            classifier_obs=classifier_obs,
            fusion_result=fusion_result,
            roi=roi,
            layers=self.layers,
        )

        # Frame info overlay on video
        if self.layers[DebugLayer.FRAME_INFO]:
            cv2.putText(
                video_frame,
                f"Frame: {frame.frame_id} | t: {frame.t_src_ns / 1e9:.3f}s",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE_BGR, 1,
            )

        self._layout.place_video(canvas, video_frame)

        # 3. Stats panel (skip drawing if STATS layer is off)
        if self.layers[DebugLayer.STATS]:
            self._stats_panel.draw(
                canvas,
                self._layout.stats_region(),
                face_obs=face_obs,
                classifier_obs=classifier_obs,
                fusion_result=fusion_result,
                is_gate_open=is_gate_open,
                in_cooldown=in_cooldown,
                profile_timing=timing,
                monitor_stats=monitor_stats,
                backend_label=backend_label,
                source_image=frame.data,
                layers=self.layers,
            )

        # 4. Timeline panel: always update history, draw only if enabled
        spike_threshold = 0.12
        if fusion_result and fusion_result.metadata:
            adaptive_summary = fusion_result.metadata.get("adaptive_summary")
            if adaptive_summary:
                spike_threshold = adaptive_summary.get("threshold", 0.12)

        self._timeline_panel.update(face_obs, fusion_result, is_gate_open)
        if self.layers[DebugLayer.TIMELINE]:
            self._timeline_panel.draw(canvas, self._layout.timeline_region(), threshold=spike_threshold)

        # 5. Layer status overlay
        self._draw_layer_status(canvas)

        return canvas

    def _draw_layer_status(self, canvas: np.ndarray) -> None:
        """Draw compact layer toggle status at the bottom of the canvas."""
        ch = canvas.shape[0]
        y = ch - 5
        x = 8

        for layer in DebugLayer:
            enabled = self.layers[layer]
            label = f"{layer.value}:{_LAYER_LABELS[layer]}"
            color = (0, 200, 100) if enabled else (50, 50, 50)
            cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)
            text_w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.28, 1)[0][0]
            x += text_w + 6


# Backward-compatible re-exports
# FusionVisualizer is no longer needed (split into StatsPanel + TimelinePanel)
# but we keep it importable for any external code that might reference it.
class FusionVisualizer:
    """Deprecated: use StatsPanel + TimelinePanel instead.

    Kept for backward compatibility only.
    """

    THUMB_SIZE = 64
    MAX_THUMBS = 5

    def __init__(self, config=None):
        self._timeline = TimelinePanel()
        self._stats = StatsPanel()

    def reset(self):
        self._timeline.reset()
        self._stats.reset()


__all__ = [
    "RenderContext",
    "DebugVisualizer",
    "VisualizationConfig",
    "ExtractorVisualizer",
    "FusionVisualizer",
    "LayoutManager",
    "VideoPanel",
    "StatsPanel",
    "TimelinePanel",
    "DebugLayer",
    "LayerState",
]
