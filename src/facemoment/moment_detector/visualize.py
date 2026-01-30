"""Visualization utilities for debugging extractors and fusion."""

from typing import Optional, List, Tuple, Iterator, Dict
from dataclasses import dataclass
from pathlib import Path
from collections import deque
import logging
import time

import cv2
import numpy as np

from visualbase import Frame

from facemoment.moment_detector.extractors.base import Observation, FaceObservation
from facemoment.moment_detector.fusion.base import FusionResult

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""

    # Colors (BGR format)
    face_box_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    face_box_color_bad: Tuple[int, int, int] = (0, 0, 255)  # Red
    pose_color: Tuple[int, int, int] = (255, 255, 0)  # Cyan
    hand_raised_color: Tuple[int, int, int] = (0, 255, 255)  # Yellow
    quality_good_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    quality_bad_color: Tuple[int, int, int] = (0, 0, 255)  # Red
    trigger_color: Tuple[int, int, int] = (0, 0, 255)  # Red
    gate_open_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    gate_closed_color: Tuple[int, int, int] = (128, 128, 128)  # Gray

    # Drawing params
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.5
    thickness: int = 2
    text_color: Tuple[int, int, int] = (255, 255, 255)

    # Panel sizes
    info_panel_width: int = 300
    graph_height: int = 100


class ExtractorVisualizer:
    """Visualizer for individual extractor outputs."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()

    def draw_face_observation(
        self,
        image: np.ndarray,
        observation: Observation,
        show_details: bool = True,
    ) -> np.ndarray:
        """Draw face observations on image.

        Args:
            image: Input BGR image.
            observation: Observation from FaceExtractor.
            show_details: Whether to show detailed info per face.

        Returns:
            Annotated image.
        """
        output = image.copy()
        h, w = output.shape[:2]
        cfg = self.config

        for face in observation.faces:
            # Convert normalized bbox to pixels
            x = int(face.bbox[0] * w)
            y = int(face.bbox[1] * h)
            bw = int(face.bbox[2] * w)
            bh = int(face.bbox[3] * h)

            # Choose color based on gate conditions
            is_good = (
                face.confidence >= 0.7
                and abs(face.yaw) <= 25
                and abs(face.pitch) <= 20
                and face.inside_frame
            )
            color = cfg.face_box_color if is_good else cfg.face_box_color_bad

            # Draw bounding box
            cv2.rectangle(output, (x, y), (x + bw, y + bh), color, cfg.thickness)

            if show_details:
                # Draw face ID
                cv2.putText(
                    output,
                    f"ID:{face.face_id}",
                    (x, y - 10),
                    cfg.font,
                    cfg.font_scale,
                    color,
                    1,
                )

                # Draw info box
                info_lines = [
                    f"conf: {face.confidence:.2f}",
                    f"yaw: {face.yaw:.1f}",
                    f"pitch: {face.pitch:.1f}",
                    f"expr: {face.expression:.2f}",
                ]

                for i, line in enumerate(info_lines):
                    cv2.putText(
                        output,
                        line,
                        (x + bw + 5, y + 15 + i * 18),
                        cfg.font,
                        cfg.font_scale * 0.8,
                        cfg.text_color,
                        1,
                    )

                # Draw expression bar
                bar_x = x
                bar_y = y + bh + 5
                bar_w = bw
                bar_h = 10
                expr_w = int(bar_w * face.expression)

                cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                cv2.rectangle(output, (bar_x, bar_y), (bar_x + expr_w, bar_y + bar_h), (0, 255, 255), -1)

        # Draw summary
        face_count = len(observation.faces)
        max_expr = observation.signals.get("max_expression", 0)
        cv2.putText(
            output,
            f"Faces: {face_count} | Max Expr: {max_expr:.2f}",
            (10, 25),
            cfg.font,
            cfg.font_scale,
            cfg.text_color,
            1,
        )

        return output

    def draw_pose_observation(
        self,
        image: np.ndarray,
        observation: Observation,
        show_skeleton: bool = True,
    ) -> np.ndarray:
        """Draw pose observations on image.

        Args:
            image: Input BGR image.
            observation: Observation from PoseExtractor.
            show_skeleton: Whether to draw skeleton connections.

        Returns:
            Annotated image.
        """
        output = image.copy()
        cfg = self.config

        # COCO skeleton connections for upper body
        skeleton = [
            (5, 6),  # shoulders
            (5, 7),  # left shoulder -> elbow
            (7, 9),  # left elbow -> wrist
            (6, 8),  # right shoulder -> elbow
            (8, 10),  # right elbow -> wrist
            (5, 11),  # left shoulder -> hip
            (6, 12),  # right shoulder -> hip
            (11, 12),  # hips
        ]

        # Check if we have keypoints in metadata
        keypoints_list = observation.metadata.get("keypoints", [])

        # Draw signals-based info
        person_count = int(observation.signals.get("person_count", 0))
        hands_raised = int(observation.signals.get("hands_raised_count", 0))
        wave_detected = observation.signals.get("hand_wave_detected", 0) > 0.5
        wave_conf = observation.signals.get("hand_wave_confidence", 0)

        # Draw summary
        status_color = cfg.hand_raised_color if hands_raised > 0 else cfg.pose_color
        cv2.putText(
            output,
            f"Persons: {person_count} | Hands Up: {hands_raised}",
            (10, 25),
            cfg.font,
            cfg.font_scale,
            status_color,
            1,
        )

        if wave_detected:
            cv2.putText(
                output,
                f"WAVE DETECTED ({wave_conf:.2f})",
                (10, 50),
                cfg.font,
                cfg.font_scale * 1.2,
                (0, 0, 255),
                2,
            )

        # Draw per-person hand status
        y_offset = 75
        for i in range(person_count):
            left_raised = observation.signals.get(f"person_{i}_left_raised", 0) > 0
            right_raised = observation.signals.get(f"person_{i}_right_raised", 0) > 0

            status = ""
            if left_raised and right_raised:
                status = "Both hands up"
            elif left_raised:
                status = "Left hand up"
            elif right_raised:
                status = "Right hand up"

            if status:
                cv2.putText(
                    output,
                    f"Person {i}: {status}",
                    (10, y_offset),
                    cfg.font,
                    cfg.font_scale * 0.8,
                    cfg.hand_raised_color,
                    1,
                )
                y_offset += 20

        return output

    def draw_quality_observation(
        self,
        image: np.ndarray,
        observation: Observation,
    ) -> np.ndarray:
        """Draw quality metrics overlay.

        Args:
            image: Input BGR image.
            observation: Observation from QualityExtractor.

        Returns:
            Annotated image with quality overlay.
        """
        output = image.copy()
        h, w = output.shape[:2]
        cfg = self.config

        # Get metrics
        blur_score = observation.signals.get("blur_score", 0)
        blur_quality = observation.signals.get("blur_quality", 0)
        brightness = observation.signals.get("brightness", 0)
        brightness_quality = observation.signals.get("brightness_quality", 0)
        contrast = observation.signals.get("contrast", 0)
        contrast_quality = observation.signals.get("contrast_quality", 0)
        quality_gate = observation.signals.get("quality_gate", 0)

        # Draw quality bars on the right side
        bar_x = w - 150
        bar_w = 100
        bar_h = 15
        y_start = 30

        metrics = [
            ("Blur", blur_quality, f"{blur_score:.0f}"),
            ("Bright", brightness_quality, f"{brightness:.0f}"),
            ("Contrast", contrast_quality, f"{contrast:.0f}"),
        ]

        for i, (name, quality, raw_val) in enumerate(metrics):
            y = y_start + i * 30

            # Label
            cv2.putText(output, name, (bar_x - 60, y + 12), cfg.font, cfg.font_scale * 0.8, cfg.text_color, 1)

            # Background bar
            cv2.rectangle(output, (bar_x, y), (bar_x + bar_w, y + bar_h), (50, 50, 50), -1)

            # Quality fill
            fill_w = int(bar_w * min(1.0, quality))
            color = cfg.quality_good_color if quality >= 1.0 else cfg.quality_bad_color
            cv2.rectangle(output, (bar_x, y), (bar_x + fill_w, y + bar_h), color, -1)

            # Raw value
            cv2.putText(output, raw_val, (bar_x + bar_w + 5, y + 12), cfg.font, cfg.font_scale * 0.7, cfg.text_color, 1)

        # Gate status
        gate_y = y_start + len(metrics) * 30 + 10
        gate_color = cfg.gate_open_color if quality_gate > 0.5 else cfg.gate_closed_color
        gate_text = "GATE: OPEN" if quality_gate > 0.5 else "GATE: CLOSED"
        cv2.putText(output, gate_text, (bar_x - 60, gate_y + 12), cfg.font, cfg.font_scale, gate_color, 2)

        return output


class FusionVisualizer:
    """Visualizer for fusion state and decisions."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self._expression_history: List[float] = []
        self._gate_history: List[bool] = []
        self._trigger_times: List[int] = []
        self._max_history = 300  # ~10 seconds at 30fps

    def reset(self) -> None:
        """Reset history."""
        self._expression_history.clear()
        self._gate_history.clear()
        self._trigger_times.clear()

    def draw_fusion_state(
        self,
        image: np.ndarray,
        observation: Observation,
        result: FusionResult,
        is_gate_open: bool,
        in_cooldown: bool,
    ) -> np.ndarray:
        """Draw fusion state overlay.

        Args:
            image: Input BGR image.
            observation: Current observation.
            result: Fusion result.
            is_gate_open: Whether gate is open.
            in_cooldown: Whether in cooldown.

        Returns:
            Annotated image.
        """
        output = image.copy()
        h, w = output.shape[:2]
        cfg = self.config

        # Update history
        max_expr = observation.signals.get("max_expression", 0)
        self._expression_history.append(max_expr)
        self._gate_history.append(is_gate_open)
        if len(self._expression_history) > self._max_history:
            self._expression_history.pop(0)
            self._gate_history.pop(0)

        if result.should_trigger:
            self._trigger_times.append(len(self._expression_history))

        # Draw status panel at top
        panel_h = 80
        cv2.rectangle(output, (0, 0), (w, panel_h), (30, 30, 30), -1)

        # Gate status
        gate_color = cfg.gate_open_color if is_gate_open else cfg.gate_closed_color
        gate_text = "GATE: OPEN" if is_gate_open else "GATE: CLOSED"
        cv2.putText(output, gate_text, (10, 25), cfg.font, cfg.font_scale, gate_color, 2)

        # Cooldown status
        if in_cooldown:
            cv2.putText(output, "COOLDOWN", (150, 25), cfg.font, cfg.font_scale, (0, 165, 255), 2)

        # Current expression
        cv2.putText(output, f"Expression: {max_expr:.2f}", (10, 50), cfg.font, cfg.font_scale, cfg.text_color, 1)

        # Metadata from result
        state = result.metadata.get("state", "monitoring")
        consecutive = result.metadata.get("consecutive_high", 0)
        cv2.putText(output, f"State: {state} | Consecutive: {consecutive}", (10, 70), cfg.font, cfg.font_scale * 0.8, cfg.text_color, 1)

        # Trigger flash
        if result.should_trigger:
            # Flash border
            cv2.rectangle(output, (0, 0), (w - 1, h - 1), cfg.trigger_color, 10)
            cv2.putText(
                output,
                f"TRIGGER: {result.reason}",
                (w // 2 - 100, h // 2),
                cfg.font,
                1.0,
                cfg.trigger_color,
                3,
            )

        # Draw expression graph at bottom
        graph_h = cfg.graph_height
        graph_y = h - graph_h - 10
        self._draw_expression_graph(output, 10, graph_y, w - 20, graph_h)

        return output

    def _draw_expression_graph(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        """Draw expression history graph."""
        cfg = self.config

        # Background
        cv2.rectangle(image, (x, y), (x + width, y + height), (40, 40, 40), -1)

        if len(self._expression_history) < 2:
            return

        # Draw gate status as background color bands
        for i, gate_open in enumerate(self._gate_history):
            px = x + int(i * width / self._max_history)
            color = (40, 60, 40) if gate_open else (60, 40, 40)
            cv2.line(image, (px, y), (px, y + height), color, 1)

        # Draw threshold line
        threshold_y = y + int(height * (1 - 0.7))  # Assuming 0.7 threshold
        cv2.line(image, (x, threshold_y), (x + width, threshold_y), (100, 100, 100), 1)

        # Draw expression line
        points = []
        for i, expr in enumerate(self._expression_history):
            px = x + int(i * width / self._max_history)
            py = y + int(height * (1 - expr))
            points.append((px, py))

        for i in range(1, len(points)):
            cv2.line(image, points[i - 1], points[i], (0, 255, 255), 1)

        # Draw trigger markers
        for t in self._trigger_times:
            if t < len(self._expression_history):
                px = x + int(t * width / self._max_history)
                cv2.line(image, (px, y), (px, y + height), cfg.trigger_color, 2)

        # Labels
        cv2.putText(image, "1.0", (x - 25, y + 10), cfg.font, cfg.font_scale * 0.6, cfg.text_color, 1)
        cv2.putText(image, "0.0", (x - 25, y + height), cfg.font, cfg.font_scale * 0.6, cfg.text_color, 1)


class TraceVisualizer:
    """Visualizer for observability trace data.

    Renders timing panels, component performance graphs, and
    diagnostic information at VERBOSE trace level.
    """

    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        history_size: int = 100,
    ):
        self.config = config or VisualizationConfig()
        self._history_size = history_size

        # Timing history per component
        self._timing_history: Dict[str, deque] = {}
        self._frame_count = 0
        self._start_time = time.monotonic()
        self._fps_history: deque = deque(maxlen=30)
        self._last_frame_time = time.monotonic()

        # Stats
        self._dropped_count = 0
        self._slow_count = 0

    def reset(self) -> None:
        """Reset all state."""
        self._timing_history.clear()
        self._frame_count = 0
        self._start_time = time.monotonic()
        self._fps_history.clear()
        self._last_frame_time = time.monotonic()
        self._dropped_count = 0
        self._slow_count = 0

    def record_timing(self, component: str, processing_ms: float, is_slow: bool = False) -> None:
        """Record component timing data.

        Args:
            component: Component name (e.g., "face", "pose", "gesture").
            processing_ms: Processing time in milliseconds.
            is_slow: Whether this was a slow frame.
        """
        if component not in self._timing_history:
            self._timing_history[component] = deque(maxlen=self._history_size)
        self._timing_history[component].append(processing_ms)
        if is_slow:
            self._slow_count += 1

    def record_frame(self) -> None:
        """Record a frame for FPS calculation."""
        self._frame_count += 1
        now = time.monotonic()
        dt = now - self._last_frame_time
        if dt > 0:
            self._fps_history.append(1.0 / dt)
        self._last_frame_time = now

    def record_drop(self, count: int = 1) -> None:
        """Record dropped frames."""
        self._dropped_count += count

    def draw_timing_panel(
        self,
        image: np.ndarray,
        x: int = 10,
        y: int = 10,
        width: int = 280,
        target_fps: float = 10.0,
    ) -> np.ndarray:
        """Draw timing panel overlay on image.

        Args:
            image: Input image.
            x: Panel x position.
            y: Panel y position.
            width: Panel width.
            target_fps: Target FPS for comparison.

        Returns:
            Image with timing panel overlay.
        """
        cfg = self.config
        output = image.copy()

        # Calculate stats
        actual_fps = sum(self._fps_history) / len(self._fps_history) if self._fps_history else 0
        elapsed = time.monotonic() - self._start_time
        avg_latency = 0

        # Panel background
        panel_height = 30 + 25 * (len(self._timing_history) + 1)
        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + panel_height),
            (40, 40, 40),
            -1,
        )
        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + panel_height),
            (100, 100, 100),
            1,
        )

        # Title
        cv2.putText(
            output,
            "Timing Panel (VERBOSE)",
            (x + 5, y + 15),
            cfg.font,
            cfg.font_scale * 0.7,
            (200, 200, 200),
            1,
        )

        # FPS line
        fps_color = (0, 255, 0) if actual_fps >= target_fps * 0.9 else (0, 255, 255) if actual_fps >= target_fps * 0.7 else (0, 0, 255)
        cv2.putText(
            output,
            f"FPS: {actual_fps:.1f}/{target_fps:.0f}",
            (x + 5, y + 35),
            cfg.font,
            cfg.font_scale * 0.6,
            fps_color,
            1,
        )

        # Dropped frames
        drop_text = f"Dropped: {self._dropped_count}"
        if self._dropped_count > 0:
            drop_color = (0, 0, 255)  # Red
        else:
            drop_color = (0, 255, 0)  # Green
        cv2.putText(
            output,
            drop_text,
            (x + 120, y + 35),
            cfg.font,
            cfg.font_scale * 0.6,
            drop_color,
            1,
        )

        # Component timing bars
        row_y = y + 50
        bar_max_width = width - 100

        for component, history in self._timing_history.items():
            if not history:
                continue

            avg_ms = sum(history) / len(history)
            max_ms = max(history)

            # Bar width based on time (scale: 0-100ms)
            bar_width = min(int(avg_ms * bar_max_width / 100), bar_max_width)
            bar_color = (0, 255, 0) if avg_ms < 50 else (0, 255, 255) if avg_ms < 80 else (0, 0, 255)

            # Component label
            cv2.putText(
                output,
                f"{component[:8]:>8}:",
                (x + 5, row_y + 5),
                cfg.font,
                cfg.font_scale * 0.5,
                cfg.text_color,
                1,
            )

            # Bar
            cv2.rectangle(
                output,
                (x + 70, row_y - 8),
                (x + 70 + bar_width, row_y + 4),
                bar_color,
                -1,
            )

            # Value
            cv2.putText(
                output,
                f"{avg_ms:.0f}ms",
                (x + 75 + bar_max_width, row_y + 5),
                cfg.font,
                cfg.font_scale * 0.5,
                cfg.text_color,
                1,
            )

            row_y += 20

        return output

    def draw_gate_checklist(
        self,
        image: np.ndarray,
        conditions: Dict[str, bool],
        x: int = 10,
        y: int = 200,
    ) -> np.ndarray:
        """Draw gate condition checklist.

        Args:
            image: Input image.
            conditions: Dict of condition name -> pass/fail.
            x: Panel x position.
            y: Panel y position.

        Returns:
            Image with gate checklist overlay.
        """
        cfg = self.config
        output = image.copy()

        # Panel background
        panel_height = 20 + 15 * len(conditions)
        cv2.rectangle(
            output,
            (x, y),
            (x + 180, y + panel_height),
            (40, 40, 40),
            -1,
        )

        # Title
        cv2.putText(
            output,
            "Gate Conditions:",
            (x + 5, y + 15),
            cfg.font,
            cfg.font_scale * 0.6,
            cfg.text_color,
            1,
        )

        row_y = y + 30
        for name, passed in conditions.items():
            mark = "v" if passed else "x"
            color = (0, 255, 0) if passed else (0, 0, 255)
            cv2.putText(
                output,
                f"[{mark}] {name}",
                (x + 5, row_y),
                cfg.font,
                cfg.font_scale * 0.5,
                color,
                1,
            )
            row_y += 15

        return output


class DebugVisualizer:
    """Combined visualizer for full pipeline debugging."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.extractor_viz = ExtractorVisualizer(config)
        self.fusion_viz = FusionVisualizer(config)

    def reset(self) -> None:
        """Reset all state."""
        self.fusion_viz.reset()

    def create_debug_view(
        self,
        frame: Frame,
        face_obs: Optional[Observation] = None,
        pose_obs: Optional[Observation] = None,
        quality_obs: Optional[Observation] = None,
        fusion_result: Optional[FusionResult] = None,
        is_gate_open: bool = False,
        in_cooldown: bool = False,
        timing: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Create combined debug visualization.

        Args:
            frame: Input frame.
            face_obs: Face extractor observation.
            pose_obs: Pose extractor observation.
            quality_obs: Quality extractor observation.
            fusion_result: Fusion result.
            is_gate_open: Gate state.
            in_cooldown: Cooldown state.
            timing: Optional timing dict for profile mode.

        Returns:
            Debug visualization image.
        """
        image = frame.data.copy()
        h, w = image.shape[:2]

        # Draw face observations
        if face_obs is not None:
            image = self.extractor_viz.draw_face_observation(image, face_obs, show_details=True)

        # Draw pose observations
        if pose_obs is not None:
            image = self.extractor_viz.draw_pose_observation(image, pose_obs)

        # Draw quality overlay
        if quality_obs is not None:
            image = self.extractor_viz.draw_quality_observation(image, quality_obs)

        # Draw fusion state if we have both observation and result
        if fusion_result is not None and face_obs is not None:
            image = self.fusion_viz.draw_fusion_state(
                image, face_obs, fusion_result, is_gate_open, in_cooldown
            )

        # Draw timing overlay in profile mode
        if timing is not None:
            image = self.draw_timing_overlay(image, timing)

        # Frame info
        cv2.putText(
            image,
            f"Frame: {frame.frame_id} | t: {frame.t_src_ns / 1e9:.3f}s",
            (10, h - 10),
            self.config.font,
            self.config.font_scale * 0.8,
            self.config.text_color,
            1,
        )

        return image

    def draw_timing_overlay(
        self,
        image: np.ndarray,
        timing: Dict[str, float],
    ) -> np.ndarray:
        """Draw timing overlay for profile mode.

        Args:
            image: Input BGR image.
            timing: Dict with timing values (detect_ms, expression_ms, total_ms).

        Returns:
            Image with timing overlay.
        """
        output = image.copy()
        h, w = output.shape[:2]
        cfg = self.config

        # Panel position (right side, below quality metrics)
        panel_w = 180
        panel_h = 70
        panel_x = w - panel_w - 10
        panel_y = 140  # Below quality overlay area

        # Semi-transparent background
        overlay = output.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)

        # Border
        cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (100, 100, 100), 1)

        # Title
        cv2.putText(
            output,
            "Profile",
            (panel_x + 5, panel_y + 15),
            cfg.font,
            cfg.font_scale * 0.7,
            (200, 200, 200),
            1,
        )

        # Timing values
        detect_ms = timing.get('detect_ms', 0)
        expr_ms = timing.get('expression_ms', 0)
        total_ms = timing.get('total_ms', 0)

        # Calculate FPS from total
        fps = 1000.0 / total_ms if total_ms > 0 else 0

        # Color code based on performance
        def get_color(ms: float, threshold: float = 50.0) -> Tuple[int, int, int]:
            if ms < threshold * 0.6:
                return (0, 255, 0)  # Green - fast
            elif ms < threshold:
                return (0, 255, 255)  # Yellow - acceptable
            else:
                return (0, 0, 255)  # Red - slow

        y_offset = panel_y + 30
        line_height = 14

        cv2.putText(output, f"Detect:  {detect_ms:6.1f}ms", (panel_x + 5, y_offset), cfg.font, cfg.font_scale * 0.6, get_color(detect_ms, 50), 1)
        y_offset += line_height
        cv2.putText(output, f"Express: {expr_ms:6.1f}ms", (panel_x + 5, y_offset), cfg.font, cfg.font_scale * 0.6, get_color(expr_ms, 50), 1)
        y_offset += line_height
        cv2.putText(output, f"Total:   {total_ms:6.1f}ms ({fps:.1f} FPS)", (panel_x + 5, y_offset), cfg.font, cfg.font_scale * 0.6, get_color(total_ms, 100), 1)

        return output


def run_debug_session(
    video_path: str,
    output_path: Optional[str] = None,
    fps: float = 10.0,
    use_ml_backends: Optional[bool] = None,
    show_window: bool = True,
) -> None:
    """Run interactive debug session on a video.

    Args:
        video_path: Path to input video.
        output_path: Optional path to save debug video.
        fps: Processing FPS.
        use_ml_backends: Whether to use real ML backends.
            None (default): Auto-detect, use ML if available.
            True: Force ML backends, error if not available.
            False: Force dummy backends.
        show_window: Whether to show interactive window.
    """
    from facemoment.moment_detector.extractors import (
        DummyExtractor,
        QualityExtractor,
    )
    from facemoment.moment_detector.fusion import DummyFusion, HighlightFusion

    # Set up extractors - auto-detect ML backends if not explicitly disabled
    extractors = []
    face_available = False
    pose_available = False

    if use_ml_backends or use_ml_backends is None:
        # Try to load and initialize FaceExtractor
        try:
            from facemoment.moment_detector.extractors import FaceExtractor
            face_ext = FaceExtractor()
            face_ext.initialize()  # Initialize now to catch errors early
            extractors.append(face_ext)
            face_available = True
            logger.info("FaceExtractor loaded and initialized")
        except Exception as e:
            if use_ml_backends:
                raise RuntimeError(f"FaceExtractor failed: {e}\nInstall with: uv sync --extra ml")
            logger.warning(f"FaceExtractor not available: {e}")

        # Try to load and initialize PoseExtractor
        try:
            from facemoment.moment_detector.extractors import PoseExtractor
            pose_ext = PoseExtractor()
            pose_ext.initialize()  # Initialize now to catch errors early
            extractors.append(pose_ext)
            pose_available = True
            logger.info("PoseExtractor loaded and initialized")
        except Exception as e:
            if use_ml_backends:
                raise RuntimeError(f"PoseExtractor failed: {e}\nInstall with: uv sync --extra ml")
            logger.warning(f"PoseExtractor not available: {e}")

    # Add QualityExtractor (always works, no ML deps)
    extractors.append(QualityExtractor())

    # Fall back to dummy if no ML extractors loaded
    if not face_available and not pose_available:
        extractors.insert(0, DummyExtractor(num_faces=1, spike_probability=0.1, seed=42))
        logger.info("Using DummyExtractor (no ML backends available)")

    # Choose fusion based on available extractors
    if face_available:
        fusion = HighlightFusion()
    else:
        fusion = DummyFusion()

    # Initialize extractors that weren't initialized during setup
    for ext in extractors:
        # FaceExtractor and PoseExtractor are already initialized
        if ext.name not in ("face", "pose"):
            ext.initialize()

    visualizer = DebugVisualizer()

    # Open video - try visualbase first, fallback to cv2
    vb = None
    source_info = None
    stream = None
    use_legacy = False

    try:
        from visualbase import VisualBase, FileSource

        source = FileSource(video_path)
        source.open()

        # Check required API compatibility
        for attr in ['fps', 'frame_count', 'width', 'height']:
            if not hasattr(source, attr):
                raise AttributeError(f"FileSource missing '{attr}'")

        vb = VisualBase()
        vb.connect(source)
        stream = vb.get_stream(fps=int(fps))
        source_info = source

    except (ImportError, AttributeError, TypeError) as e:
        logger.warning(f"visualbase API incompatible, using cv2 fallback: {e}")
        use_legacy = True

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(video_fps / fps)) if fps > 0 else 1

        class _SourceInfo:
            def __init__(self):
                self.fps = video_fps
                self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        source_info = _SourceInfo()

        def _legacy_stream():
            frame_id = 0
            while True:
                ret, image = cap.read()
                if not ret:
                    break
                if frame_id % frame_skip == 0:
                    t_ns = int(frame_id / video_fps * 1e9)
                    yield Frame.from_array(image, frame_id=frame_id, t_src_ns=t_ns)
                frame_id += 1

        stream = _legacy_stream()

    # Output writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (source_info.width, source_info.height))

    frame_count = 0
    try:
        for frame in stream:
            # Run extractors
            observations = {}
            for ext in extractors:
                obs = ext.extract(frame)
                if obs:
                    observations[ext.name] = obs

            # Run fusion (use face observation if available)
            fusion_obs = observations.get("face") or observations.get("dummy")
            fusion_result = None
            if fusion_obs:
                fusion_result = fusion.update(fusion_obs)

            # Create debug view
            debug_image = visualizer.create_debug_view(
                frame,
                face_obs=observations.get("face") or observations.get("dummy"),
                pose_obs=observations.get("pose"),
                quality_obs=observations.get("quality"),
                fusion_result=fusion_result,
                is_gate_open=fusion.is_gate_open,
                in_cooldown=fusion.in_cooldown,
            )

            # Write output
            if writer:
                writer.write(debug_image)

            # Show window
            if show_window:
                cv2.imshow("Debug View", debug_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    fusion.reset()
                    visualizer.reset()
                elif key == ord(" "):
                    cv2.waitKey(0)  # Pause

            frame_count += 1

            # Progress
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count}/{source_info.frame_count} frames")

    finally:
        if vb:
            vb.disconnect()
        elif use_legacy:
            cap.release()
        if writer:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()

        for ext in extractors:
            ext.cleanup()

    logger.info(f"Debug session complete. Processed {frame_count} frames.")
