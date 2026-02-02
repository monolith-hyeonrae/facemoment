"""Visualization utilities for debugging extractors and fusion.

Uses the supervision library for annotations.
"""

from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import logging

import cv2
import numpy as np
import supervision as sv

from visualbase import Frame

from facemoment.moment_detector.extractors.base import Observation, FaceObservation
from facemoment.moment_detector.extractors.types import KeypointIndex
from facemoment.moment_detector.extractors.outputs import PoseOutput
from facemoment.moment_detector.fusion.base import FusionResult

logger = logging.getLogger(__name__)


# Colors
COLOR_GREEN = sv.Color.from_hex("#00FF00")
COLOR_RED = sv.Color.from_hex("#FF0000")
COLOR_YELLOW = sv.Color.from_hex("#FFFF00")
COLOR_GRAY = sv.Color.from_hex("#808080")
COLOR_WHITE = sv.Color.from_hex("#FFFFFF")
COLOR_DARK = sv.Color.from_hex("#282828")

# Emotion colors (BGR for cv2 drawing)
COLOR_HAPPY_BGR = (0, 255, 255)
COLOR_ANGRY_BGR = (0, 0, 255)
COLOR_NEUTRAL_BGR = (200, 200, 200)
COLOR_DARK_BGR = (40, 40, 40)
COLOR_WHITE_BGR = (255, 255, 255)
COLOR_RED_BGR = (0, 0, 255)
COLOR_GREEN_BGR = (0, 255, 0)
COLOR_GRAY_BGR = (128, 128, 128)

# Role colors for face classification (BGR format for OpenCV)
COLOR_MAIN_BGR = (0, 255, 0)         # Green - main subject
COLOR_PASSENGER_BGR = (0, 165, 255)  # Orange - passenger (BGR: blue=0, green=165, red=255)
COLOR_TRANSIENT_BGR = (0, 255, 255)  # Yellow - transient (BGR: blue=0, green=255, red=255)
COLOR_NOISE_BGR = (128, 128, 128)    # Gray - noise

# Pose skeleton colors (BGR)
COLOR_SKELETON_BGR = (255, 200, 100)  # Light blue - skeleton lines
COLOR_KEYPOINT_BGR = (0, 255, 255)    # Yellow - keypoints


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    graph_height: int = 100


class ExtractorVisualizer:
    """Visualizer for extractor outputs using supervision."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()

        # Color palette: 0=green (good), 1=red (bad)
        palette = sv.ColorPalette.from_hex(["#00FF00", "#FF0000"])
        self._box_annotator = sv.BoxAnnotator(thickness=2, color=palette, color_lookup=sv.ColorLookup.CLASS)
        self._label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1, text_padding=3,
                                                   color=palette, color_lookup=sv.ColorLookup.CLASS)

    def draw_face_observation(self, image: np.ndarray, observation: Observation) -> np.ndarray:
        """Draw face observations on image."""
        output = image.copy()
        h, w = output.shape[:2]

        if not observation.faces:
            self._draw_summary(output, observation)
            return output

        # Build detections
        xyxy, confidences, class_ids, labels = [], [], [], []

        for face in observation.faces:
            x1, y1 = int(face.bbox[0] * w), int(face.bbox[1] * h)
            x2, y2 = int((face.bbox[0] + face.bbox[2]) * w), int((face.bbox[1] + face.bbox[3]) * h)
            xyxy.append([x1, y1, x2, y2])
            confidences.append(face.confidence)

            # Gate condition check for color
            is_good = (face.confidence >= 0.7 and abs(face.yaw) <= 25 and
                       abs(face.pitch) <= 20 and face.inside_frame)
            class_ids.append(0 if is_good else 1)

            # Label with emotions
            happy = face.signals.get("em_happy", 0.0)
            angry = face.signals.get("em_angry", 0.0)
            labels.append(f"ID:{face.face_id} H:{happy:.2f} A:{angry:.2f}")

        detections = sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidences),
            class_id=np.array(class_ids),
        )

        output = self._box_annotator.annotate(output, detections)
        output = self._label_annotator.annotate(output, detections, labels=labels)

        # Emotion bars below faces
        for i, face in enumerate(observation.faces):
            self._draw_emotion_bars(output, face, xyxy[i][0], xyxy[i][3] + 5, xyxy[i][2] - xyxy[i][0])

        self._draw_summary(output, observation)
        return output

    def _draw_emotion_bars(self, image: np.ndarray, face: FaceObservation, x: int, y: int, width: int) -> None:
        """Draw 3 emotion bars."""
        bar_h, gap = 6, 2
        emotions = [
            (face.signals.get("em_happy", 0.0), COLOR_HAPPY_BGR),
            (face.signals.get("em_angry", 0.0), COLOR_ANGRY_BGR),
            (face.signals.get("em_neutral", 0.0), COLOR_NEUTRAL_BGR),
        ]
        for i, (value, color) in enumerate(emotions):
            ey = y + i * (bar_h + gap)
            cv2.rectangle(image, (x, ey), (x + width, ey + bar_h), COLOR_DARK_BGR, -1)
            cv2.rectangle(image, (x, ey), (x + int(width * min(1.0, value)), ey + bar_h), color, -1)

    def _draw_summary(self, image: np.ndarray, observation: Observation) -> None:
        """Draw summary at top."""
        face_count = len(observation.faces)
        h = observation.signals.get("expression_happy", 0)
        a = observation.signals.get("expression_angry", 0)
        n = observation.signals.get("expression_neutral", 1)
        cv2.putText(image, f"Faces: {face_count} | H:{h:.2f} A:{a:.2f} N:{n:.2f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE_BGR, 1)

    def draw_face_classifier_observation(
        self,
        image: np.ndarray,
        classifier_obs: Observation,
        face_obs: Optional[Observation] = None,
    ) -> np.ndarray:
        """Draw face classification results with role-specific colors.

        Args:
            image: Input image.
            classifier_obs: Observation from FaceClassifierExtractor.
            face_obs: Optional face observation for bbox data.

        Supports FaceClassifierOutput structure from face_classifier.py:
        - faces: List[ClassifiedFace] with .face, .role, .confidence, .track_length
        - main_face: Optional[ClassifiedFace]
        - passenger_faces: List[ClassifiedFace]
        - transient_count, noise_count: int
        """
        output = image.copy()
        h, w = output.shape[:2]

        if classifier_obs.data is None:
            return output

        data = classifier_obs.data

        # Check for faces attribute (from FaceClassifierOutput)
        if not hasattr(data, 'faces') or not data.faces:
            return output

        # Role to color mapping
        role_colors = {
            "main": COLOR_MAIN_BGR,
            "passenger": COLOR_PASSENGER_BGR,
            "transient": COLOR_TRANSIENT_BGR,
            "noise": COLOR_NOISE_BGR,
        }

        # Draw each classified face
        for cf in data.faces:
            # ClassifiedFace has: face, role, confidence, track_length, avg_area
            face = cf.face
            role = cf.role
            track_length = cf.track_length

            # Get bbox in pixel coordinates
            x1 = int(face.bbox[0] * w)
            y1 = int(face.bbox[1] * h)
            x2 = int((face.bbox[0] + face.bbox[2]) * w)
            y2 = int((face.bbox[1] + face.bbox[3]) * h)

            color = role_colors.get(role, COLOR_GRAY_BGR)

            # Draw bbox with role color
            thickness = 3 if role == "main" else 2
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            # Draw role label with background
            label = f"{role.upper()} ({track_length}f)"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            label_y = y1 - 5 if y1 > 25 else y2 + 15

            # Background rectangle for label
            cv2.rectangle(
                output,
                (x1, label_y - label_size[1] - 4),
                (x1 + label_size[0] + 4, label_y + 2),
                color,
                -1,
            )
            cv2.putText(
                output, label, (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_DARK_BGR, 1,
            )

            # Draw face_id small
            cv2.putText(
                output, f"ID:{face.face_id}", (x1 + 2, y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
            )

            # Draw emotion bars for main/passenger (not noise/transient)
            if role in ("main", "passenger"):
                self._draw_emotion_bars(output, face, x1, y2 + 5, x2 - x1)

        # Draw summary
        main_detected = 1 if hasattr(data, 'main_face') and data.main_face else 0
        passenger_count = len(data.passenger_faces) if hasattr(data, 'passenger_faces') else 0
        transient_count = data.transient_count if hasattr(data, 'transient_count') else 0
        noise_count = data.noise_count if hasattr(data, 'noise_count') else 0
        summary = f"Main: {main_detected} | Pass: {passenger_count} | Trans: {transient_count} | Noise: {noise_count}"
        cv2.putText(output, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE_BGR, 1)

        return output

    def draw_pose_observation(self, image: np.ndarray, observation: Observation) -> np.ndarray:
        """Draw pose observations with upper body landmarks."""
        output = image.copy()
        h, w = output.shape[:2]
        person_count = int(observation.signals.get("person_count", 0))
        hands_raised = int(observation.signals.get("hands_raised_count", 0))
        wave = observation.signals.get("hand_wave_detected", 0) > 0.5

        # Draw keypoints if available
        if observation.data is not None and hasattr(observation.data, 'keypoints'):
            pose_data: PoseOutput = observation.data
            for person in pose_data.keypoints:
                self._draw_upper_body_skeleton(output, person, w, h)

        # Draw summary text
        color = COLOR_HAPPY_BGR if hands_raised > 0 else COLOR_GREEN_BGR
        cv2.putText(output, f"Persons: {person_count} | Hands Up: {hands_raised}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if wave:
            cv2.putText(output, "WAVE DETECTED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED_BGR, 2)
        return output

    def _draw_upper_body_skeleton(self, image: np.ndarray, person: Dict, w: int, h: int) -> None:
        """Draw upper body skeleton for a person.

        Draws: head, shoulders, elbows, wrists with connecting lines.
        """
        keypoints = person.get("keypoints", [])
        if not keypoints or len(keypoints) < 11:  # Need at least up to wrists
            return

        # Upper body skeleton connections (indices)
        # Head: nose, eyes, ears
        # Body: shoulders, elbows, wrists
        connections = [
            # Face connections
            (KeypointIndex.NOSE, KeypointIndex.LEFT_EYE),
            (KeypointIndex.NOSE, KeypointIndex.RIGHT_EYE),
            (KeypointIndex.LEFT_EYE, KeypointIndex.LEFT_EAR),
            (KeypointIndex.RIGHT_EYE, KeypointIndex.RIGHT_EAR),
            # Shoulder to shoulder
            (KeypointIndex.LEFT_SHOULDER, KeypointIndex.RIGHT_SHOULDER),
            # Left arm
            (KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_ELBOW),
            (KeypointIndex.LEFT_ELBOW, KeypointIndex.LEFT_WRIST),
            # Right arm
            (KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_ELBOW),
            (KeypointIndex.RIGHT_ELBOW, KeypointIndex.RIGHT_WRIST),
            # Nose to shoulder center (neck approximation)
        ]

        min_conf = 0.3

        # Draw skeleton lines
        for idx1, idx2 in connections:
            if idx1 >= len(keypoints) or idx2 >= len(keypoints):
                continue
            kpt1, kpt2 = keypoints[idx1], keypoints[idx2]
            if kpt1[2] < min_conf or kpt2[2] < min_conf:
                continue
            pt1 = (int(kpt1[0]), int(kpt1[1]))
            pt2 = (int(kpt2[0]), int(kpt2[1]))
            cv2.line(image, pt1, pt2, COLOR_SKELETON_BGR, 2)

        # Draw keypoints (upper body only: 0-10)
        upper_body_indices = [
            KeypointIndex.NOSE,
            KeypointIndex.LEFT_EYE, KeypointIndex.RIGHT_EYE,
            KeypointIndex.LEFT_EAR, KeypointIndex.RIGHT_EAR,
            KeypointIndex.LEFT_SHOULDER, KeypointIndex.RIGHT_SHOULDER,
            KeypointIndex.LEFT_ELBOW, KeypointIndex.RIGHT_ELBOW,
            KeypointIndex.LEFT_WRIST, KeypointIndex.RIGHT_WRIST,
        ]
        for idx in upper_body_indices:
            if idx >= len(keypoints):
                continue
            kpt = keypoints[idx]
            if kpt[2] < min_conf:
                continue
            pt = (int(kpt[0]), int(kpt[1]))
            # Different colors for different parts
            if idx == KeypointIndex.NOSE:
                color = COLOR_WHITE_BGR
                radius = 4
            elif idx in (KeypointIndex.LEFT_WRIST, KeypointIndex.RIGHT_WRIST):
                color = COLOR_HAPPY_BGR  # Yellow for wrists
                radius = 6
            elif idx in (KeypointIndex.LEFT_SHOULDER, KeypointIndex.RIGHT_SHOULDER):
                color = COLOR_GREEN_BGR
                radius = 5
            else:
                color = COLOR_KEYPOINT_BGR
                radius = 3
            cv2.circle(image, pt, radius, color, -1)
            cv2.circle(image, pt, radius, COLOR_DARK_BGR, 1)

    def draw_quality_observation(self, image: np.ndarray, observation: Observation) -> np.ndarray:
        """Draw quality metrics."""
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
        cv2.putText(output, "GATE: OPEN" if gate > 0.5 else "GATE: CLOSED",
                    (bar_x - 55, gate_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gate_color, 2)
        return output


class FusionVisualizer:
    """Visualizer for fusion state with adaptive happy tracking."""

    THUMB_SIZE = 64
    MAX_THUMBS = 5

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        # Raw happy history
        self._happy_history: List[float] = []
        # Baseline history (slow EWMA)
        self._baseline_history: List[float] = []
        # Spike history (recent - baseline)
        self._spike_history: List[float] = []
        # Other
        self._gate_history: List[bool] = []
        self._trigger_times: List[int] = []
        self._trigger_thumbs: List[Tuple[int, np.ndarray, str]] = []
        self._max_history = 300

    def reset(self) -> None:
        """Reset history."""
        self._happy_history.clear()
        self._baseline_history.clear()
        self._spike_history.clear()
        self._gate_history.clear()
        self._trigger_times.clear()
        self._trigger_thumbs.clear()

    def draw_fusion_state(
        self,
        image: np.ndarray,
        observation: Observation,
        result: FusionResult,
        is_gate_open: bool,
        in_cooldown: bool,
        source_image: Optional[np.ndarray] = None,
        adaptive_summary: Optional[Dict] = None,
    ) -> np.ndarray:
        """Draw fusion state overlay with adaptive happy tracking.

        Args:
            image: Input image.
            observation: Current observation.
            result: Fusion result.
            is_gate_open: Gate state.
            in_cooldown: Cooldown state.
            source_image: Original image for thumbnail capture.
            adaptive_summary: Adaptive happy tracking summary from fusion.
        """
        output = image.copy()
        h, w = output.shape[:2]

        # Update raw happy history
        happy = observation.signals.get("expression_happy", 0)
        self._happy_history.append(happy)
        self._gate_history.append(is_gate_open)

        # Update adaptive tracking history
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

        # Trim history
        if len(self._happy_history) > self._max_history:
            self._happy_history.pop(0)
            self._baseline_history.pop(0)
            self._spike_history.pop(0)
            self._gate_history.pop(0)
            self._trigger_times = [t - 1 for t in self._trigger_times if t > 1]
            self._trigger_thumbs = [(t - 1, th, r) for t, th, r in self._trigger_thumbs if t > 1]

        if result.should_trigger:
            time_idx = len(self._happy_history)
            self._trigger_times.append(time_idx)
            thumb = self._capture_thumbnail(source_image if source_image is not None else image, observation)
            if thumb is not None:
                self._trigger_thumbs.append((time_idx, thumb, result.reason))
                if len(self._trigger_thumbs) > self.MAX_THUMBS:
                    self._trigger_thumbs.pop(0)

        # Top panel
        panel_height = 70
        cv2.rectangle(output, (0, 0), (w, panel_height), COLOR_DARK_BGR, -1)

        # Gate and cooldown status
        gate_color = COLOR_GREEN_BGR if is_gate_open else COLOR_GRAY_BGR
        cv2.putText(output, "GATE: OPEN" if is_gate_open else "GATE: CLOSED",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, gate_color, 1)
        if in_cooldown:
            cv2.putText(output, "COOLDOWN", (130, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

        # Adaptive tracking info
        if adaptive_summary:
            max_spike = adaptive_summary.get("max_spike", 0)
            threshold = adaptive_summary.get("threshold", 0.12)

            first_state = next(iter(adaptive_summary.get("states", {}).values()), None)
            if first_state:
                baseline = first_state["baseline"]
                recent = first_state["recent"] if "recent" in first_state else happy
                spike = first_state["spike"]

                # Show: Happy (current) | Baseline | Spike
                cv2.putText(output, f"Happy: {happy:.2f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HAPPY_BGR, 1)
                cv2.putText(output, f"Base: {baseline:.2f}", (130, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY_BGR, 1)

                # Spike with color coding
                spike_color = COLOR_RED_BGR if spike > threshold else (COLOR_HAPPY_BGR if spike > threshold * 0.5 else COLOR_GRAY_BGR)
                cv2.putText(output, f"Spike: {spike:+.2f}", (240, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, spike_color, 1)

            # Spike bar (right side)
            spike_pct = min(1.0, max(0, max_spike) / (threshold * 2))
            bar_x, bar_w, bar_h = w - 150, 100, 20
            cv2.rectangle(output, (bar_x, 25), (bar_x + bar_w, 25 + bar_h), (60, 60, 60), -1)
            bar_color = COLOR_RED_BGR if max_spike > threshold else COLOR_GREEN_BGR
            cv2.rectangle(output, (bar_x, 25), (bar_x + int(bar_w * spike_pct), 25 + bar_h), bar_color, -1)
            # Threshold marker at 50%
            th_x = bar_x + bar_w // 2
            cv2.line(output, (th_x, 23), (th_x, 25 + bar_h + 2), COLOR_WHITE_BGR, 2)
            cv2.putText(output, f"{max_spike:.2f}", (bar_x + bar_w + 5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE_BGR, 1)
        else:
            cv2.putText(output, f"Happy: {happy:.2f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HAPPY_BGR, 1)

        # Trigger flash
        if result.should_trigger:
            cv2.rectangle(output, (0, 0), (w - 1, h - 1), COLOR_RED_BGR, 10)
            cv2.putText(output, f"TRIGGER: {result.reason}", (w // 2 - 100, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_RED_BGR, 3)

        # Draw graph
        spike_threshold = adaptive_summary.get("threshold", 0.12) if adaptive_summary else 0.12
        self._draw_adaptive_graph(output, 10, h - self.config.graph_height - 10, w - 20,
                                   self.config.graph_height, spike_threshold)
        self._draw_thumbnails(output)

        return output

    def _draw_graph(self, image: np.ndarray, x: int, y: int, width: int, height: int) -> None:
        """Draw raw emotion history graph (legacy)."""
        cv2.rectangle(image, (x, y), (x + width, y + height), COLOR_DARK_BGR, -1)

        if len(self._happy_history) < 2:
            return

        # Gate background
        for i, gate in enumerate(self._gate_history):
            px = x + int(i * width / self._max_history)
            cv2.line(image, (px, y), (px, y + height), (40, 60, 40) if gate else (60, 40, 40), 1)

        # Threshold line
        cv2.line(image, (x, y + int(height * 0.3)), (x + width, y + int(height * 0.3)), (100, 100, 100), 1)

        # Emotion lines
        for history, color in [(self._happy_history, COLOR_HAPPY_BGR),
                               (self._angry_history, COLOR_ANGRY_BGR),
                               (self._neutral_history, COLOR_NEUTRAL_BGR)]:
            pts = [(x + int(i * width / self._max_history), y + int(height * (1 - v)))
                   for i, v in enumerate(history)]
            for i in range(1, len(pts)):
                cv2.line(image, pts[i - 1], pts[i], color, 1)

        # Trigger markers
        for t in self._trigger_times:
            if t < len(self._happy_history):
                px = x + int(t * width / self._max_history)
                cv2.line(image, (px, y), (px, y + height), COLOR_RED_BGR, 2)

        # Legend
        cv2.putText(image, "H", (x + width + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_HAPPY_BGR, 1)
        cv2.putText(image, "A", (x + width + 5, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_ANGRY_BGR, 1)
        cv2.putText(image, "N", (x + width + 5, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_NEUTRAL_BGR, 1)

    def _draw_adaptive_graph(self, image: np.ndarray, x: int, y: int, width: int, height: int,
                              threshold: float = 0.12) -> None:
        """Draw adaptive happy tracking graph.

        Shows:
        - Yellow line: Raw happy value (0-1)
        - Gray dashed: Baseline (person's typical level)
        - Green/Red area: Spike (recent - baseline)
        """
        cv2.rectangle(image, (x, y), (x + width, y + height), COLOR_DARK_BGR, -1)

        if len(self._happy_history) < 2:
            return

        # Gate background
        for i, gate in enumerate(self._gate_history):
            px = x + int(i * width / self._max_history)
            cv2.line(image, (px, y), (px, y + height), (40, 60, 40) if gate else (60, 40, 40), 1)

        # Threshold line (from bottom, at threshold height)
        th_y = y + height - int(threshold * height * 2)  # Scale threshold
        cv2.line(image, (x, th_y), (x + width, th_y), (100, 100, 100), 1)

        # Draw baseline (gray, dashed effect with dots)
        if self._baseline_history:
            for i in range(0, len(self._baseline_history), 3):
                px = x + int(i * width / self._max_history)
                py = y + height - int(self._baseline_history[i] * height)
                cv2.circle(image, (px, py), 1, COLOR_GRAY_BGR, -1)

        # Draw raw happy line (yellow)
        pts_happy = [(x + int(i * width / self._max_history), y + height - int(v * height))
                     for i, v in enumerate(self._happy_history)]
        for i in range(1, len(pts_happy)):
            cv2.line(image, pts_happy[i - 1], pts_happy[i], COLOR_HAPPY_BGR, 1)

        # Draw spike area (filled between baseline and current)
        if self._spike_history and self._baseline_history:
            for i in range(len(self._spike_history)):
                if i >= len(self._baseline_history):
                    break
                px = x + int(i * width / self._max_history)
                base_y = y + height - int(self._baseline_history[i] * height)
                spike = self._spike_history[i]
                if spike > 0:
                    # Positive spike - green to red based on threshold
                    spike_h = int(spike * height)
                    color = COLOR_RED_BGR if spike > threshold else (0, 180, 0)
                    cv2.line(image, (px, base_y), (px, base_y - spike_h), color, 1)

        # Trigger markers
        for t in self._trigger_times:
            if t < len(self._happy_history):
                px = x + int(t * width / self._max_history)
                cv2.line(image, (px, y), (px, y + height), COLOR_RED_BGR, 2)

        # Legend
        cv2.putText(image, "Happy", (x + 5, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_HAPPY_BGR, 1)
        cv2.putText(image, "base", (x + 45, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_GRAY_BGR, 1)
        cv2.putText(image, f"th={threshold:.2f}", (x + width - 50, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, COLOR_GRAY_BGR, 1)

    def _capture_thumbnail(self, image: np.ndarray, observation: Observation) -> Optional[np.ndarray]:
        """Capture largest face as thumbnail."""
        if not observation.faces:
            return None

        h, w = image.shape[:2]
        best = max(observation.faces, key=lambda f: f.bbox[2] * f.bbox[3])

        bx, by, bw, bh = best.bbox
        x1 = int(max(0, (bx - bw * 0.2) * w))
        y1 = int(max(0, (by - bh * 0.2) * h))
        x2 = int(min(w, (bx + bw * 1.2) * w))
        y2 = int(min(h, (by + bh * 1.2) * h))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]
        return cv2.resize(crop, (self.THUMB_SIZE, self.THUMB_SIZE)) if crop.size > 0 else None

    def _draw_thumbnails(self, image: np.ndarray) -> None:
        """Draw trigger thumbnails on right side."""
        if not self._trigger_thumbs:
            return

        h, w = image.shape[:2]
        thumb_x = w - self.THUMB_SIZE - 10

        for i, (_, thumb, reason) in enumerate(self._trigger_thumbs):
            y_pos = 90 + i * (self.THUMB_SIZE + 25)
            if y_pos + self.THUMB_SIZE > h - self.config.graph_height - 20:
                break

            cv2.rectangle(image, (thumb_x - 2, y_pos - 2),
                          (thumb_x + self.THUMB_SIZE + 2, y_pos + self.THUMB_SIZE + 2), COLOR_RED_BGR, 2)
            image[y_pos:y_pos + self.THUMB_SIZE, thumb_x:thumb_x + self.THUMB_SIZE] = thumb
            cv2.putText(image, reason[:12], (thumb_x, y_pos + self.THUMB_SIZE + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_WHITE_BGR, 1)


class DebugVisualizer:
    """Combined visualizer for pipeline debugging."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.extractor_viz = ExtractorVisualizer(config)
        self.fusion_viz = FusionVisualizer(config)

    def reset(self) -> None:
        self.fusion_viz.reset()

    def create_debug_view(
        self,
        frame: Frame,
        face_obs: Optional[Observation] = None,
        pose_obs: Optional[Observation] = None,
        quality_obs: Optional[Observation] = None,
        classifier_obs: Optional[Observation] = None,
        fusion_result: Optional[FusionResult] = None,
        is_gate_open: bool = False,
        in_cooldown: bool = False,
        timing: Optional[Dict[str, float]] = None,
        roi: Optional[Tuple[float, float, float, float]] = None,
    ) -> np.ndarray:
        """Create combined debug visualization.

        Args:
            frame: Input frame.
            face_obs: Face observation.
            pose_obs: Pose observation.
            quality_obs: Quality observation.
            classifier_obs: Face classifier observation (role classification).
            fusion_result: Fusion result.
            is_gate_open: Whether gate is open.
            in_cooldown: Whether in cooldown.
            timing: Timing info for profile mode.
            roi: ROI boundary (x1, y1, x2, y2) in normalized coords [0-1].
        """
        image = frame.data.copy()
        h, w = image.shape[:2]

        # Draw ROI boundary first (under annotations)
        if roi is not None:
            self._draw_roi(image, roi)

        # Draw pose first (skeleton under other annotations)
        if pose_obs is not None:
            image = self.extractor_viz.draw_pose_observation(image, pose_obs)

        # Draw face classifier if available (role-based colors)
        if classifier_obs is not None:
            image = self.extractor_viz.draw_face_classifier_observation(image, classifier_obs, face_obs)
        elif face_obs is not None:
            # Fallback to basic face observation
            image = self.extractor_viz.draw_face_observation(image, face_obs)

        if quality_obs is not None:
            image = self.extractor_viz.draw_quality_observation(image, quality_obs)
        if fusion_result is not None and face_obs is not None:
            image = self.fusion_viz.draw_fusion_state(image, face_obs, fusion_result,
                                                       is_gate_open, in_cooldown, source_image=frame.data,
                                                       adaptive_summary=fusion_result.metadata.get("adaptive_summary"))
        if timing is not None:
            image = self._draw_timing(image, timing)

        cv2.putText(image, f"Frame: {frame.frame_id} | t: {frame.t_src_ns / 1e9:.3f}s",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE_BGR, 1)
        return image

    def _draw_roi(self, image: np.ndarray, roi: Tuple[float, float, float, float]) -> None:
        """Draw subtle ROI boundary."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = roi
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)

        # Subtle gray color for ROI boundary
        roi_color = (80, 80, 80)

        # Thin solid rectangle
        cv2.rectangle(image, (px1, py1), (px2, py2), roi_color, 1)

    def _draw_timing(self, image: np.ndarray, timing: Dict[str, float]) -> np.ndarray:
        """Draw timing overlay."""
        output = image.copy()
        h, w = output.shape[:2]

        panel_x, panel_y = w - 190, 140
        overlay = output.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + 180, panel_y + 70), COLOR_DARK_BGR, -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)

        detect_ms = timing.get('detect_ms', 0)
        expr_ms = timing.get('expression_ms', 0)
        total_ms = timing.get('total_ms', 0)
        fps = 1000.0 / total_ms if total_ms > 0 else 0

        def color(ms, th=50):
            return COLOR_GREEN_BGR if ms < th * 0.6 else COLOR_HAPPY_BGR if ms < th else COLOR_RED_BGR

        y = panel_y + 20
        cv2.putText(output, f"Detect:  {detect_ms:5.1f}ms", (panel_x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color(detect_ms), 1)
        cv2.putText(output, f"Express: {expr_ms:5.1f}ms", (panel_x + 5, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color(expr_ms), 1)
        cv2.putText(output, f"Total:   {total_ms:5.1f}ms ({fps:.1f}fps)", (panel_x + 5, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color(total_ms, 100), 1)

        return output
