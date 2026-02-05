"""Stats panel — draws timing, FPS, gate status, and thumbnails on the side panel.

All textual/numeric information that was previously drawn on the video
is now rendered in this dedicated panel.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from facemoment.moment_detector.extractors.base import Observation, FaceObservation
from facemoment.moment_detector.visualize.components import (
    COLOR_DARK_BGR,
    COLOR_WHITE_BGR,
    COLOR_GRAY_BGR,
    COLOR_RED_BGR,
    COLOR_GREEN_BGR,
    COLOR_HAPPY_BGR,
    COLOR_MAIN_BGR,
    FONT,
    FONT_SMALL,
    FONT_MEDIUM,
    draw_horizontal_bar,
    timing_color,
    DebugLayer,
    LayerState,
)


class StatsPanel:
    """Draws statistics, timing bars, and thumbnails in the side panel.

    Content (top to bottom):
    1. Backend indicator (PATHWAY / SIMPLE / DISTRIBUTED)
    2. FPS and frame time
    3. Per-extractor timing bars
    4. Fusion timing
    5. Bottleneck indicator
    6. Gate / cooldown status
    7. Main face info
    8. Trigger thumbnails
    """

    THUMB_SIZE = 56
    MAX_THUMBS = 4

    def __init__(self):
        self._trigger_thumbs: List[Tuple[int, np.ndarray, str]] = []
        self._frame_count = 0

    def reset(self) -> None:
        self._trigger_thumbs.clear()
        self._frame_count = 0

    def draw(
        self,
        canvas: np.ndarray,
        region: Tuple[int, int, int, int],
        face_obs: Optional[Observation] = None,
        classifier_obs: Optional[Observation] = None,
        fusion_result: Optional[Observation] = None,
        is_gate_open: bool = False,
        in_cooldown: bool = False,
        profile_timing: Optional[Dict[str, float]] = None,
        monitor_stats: Optional[Dict] = None,
        backend_label: str = "",
        source_image: Optional[np.ndarray] = None,
        layers: Optional[LayerState] = None,
    ) -> None:
        """Draw stats panel content onto canvas.

        Args:
            canvas: Full canvas to draw on.
            region: (x1, y1, x2, y2) for this panel.
            face_obs: Face observation (for summary text).
            classifier_obs: Classifier observation (for main face info).
            fusion_result: Fusion result.
            is_gate_open: Gate state.
            in_cooldown: Cooldown state.
            profile_timing: Per-component timing dict.
            monitor_stats: PathwayMonitor frame stats.
            backend_label: Label like "PATHWAY", "SIMPLE", "DISTRIBUTED".
            source_image: Source image for trigger thumbnail capture.
            layers: Layer visibility state. None means all layers enabled.
        """
        self._frame_count += 1
        rx1, ry1, rx2, ry2 = region
        x = rx1 + 8
        y = ry1 + 18

        # Backend label with color coding:
        # Green = confirmed pathway, Orange = fallback to simple, Cyan = explicit simple
        if backend_label:
            label_upper = backend_label.upper()
            if "PATHWAY" in label_upper and "unavailable" not in label_upper:
                label_color = (0, 255, 100)  # green
            elif "SIMPLE" in label_upper or "unavailable" in label_upper or "subprocess" in label_upper:
                label_color = (0, 165, 255)  # orange — fallback
            elif "DISTRIBUTED" in label_upper:
                label_color = (255, 200, 0)  # cyan
            else:
                label_color = (0, 200, 255)  # yellow-ish default
            cv2.putText(canvas, f"[{backend_label}]", (x, y), FONT, 0.40, label_color, 1)
            y += 20

        # Monitor stats (Pathway mode)
        if monitor_stats:
            y = self._draw_monitor_stats(canvas, x, y, rx2 - rx1 - 16, monitor_stats)
        elif profile_timing:
            y = self._draw_profile_timing(canvas, x, y, rx2 - rx1 - 16, profile_timing)

        # Gate / cooldown
        y = self._draw_gate_status(canvas, x, y, is_gate_open, in_cooldown)

        # Face summary (skip if FACE layer is off)
        if layers is None or layers[DebugLayer.FACE]:
            if classifier_obs is not None:
                y = self._draw_classifier_summary(canvas, x, y, classifier_obs)
            elif face_obs is not None:
                y = self._draw_face_summary(canvas, x, y, face_obs)

        # Fusion info (skip if FUSION layer is off)
        if fusion_result is not None:
            if layers is None or layers[DebugLayer.FUSION]:
                y = self._draw_fusion_info(canvas, x, y, fusion_result, face_obs)

        # Capture and draw trigger thumbnails (skip if TRIGGER layer is off)
        if layers is None or layers[DebugLayer.TRIGGER]:
            if fusion_result and fusion_result.should_trigger:
                self._capture_thumbnail(source_image, face_obs, fusion_result.trigger_reason)
            self._draw_thumbnails(canvas, rx1, y + 5, rx2)

    def _draw_monitor_stats(
        self, canvas: np.ndarray, x: int, y: int, panel_w: int, stats: Dict
    ) -> int:
        """Draw PathwayMonitor stats: FPS, frame time, extractor timing bars."""
        line_h = 16

        # FPS
        fps = stats.get("effective_fps", 0)
        target = stats.get("target_fps", 10)
        ratio = stats.get("fps_ratio", 0)
        fps_color = COLOR_GREEN_BGR if ratio >= 0.9 else (COLOR_HAPPY_BGR if ratio >= 0.7 else COLOR_RED_BGR)
        cv2.putText(canvas, f"FPS: {fps:.1f}/{target:.0f} ({ratio:.0%})", (x, y), FONT, FONT_SMALL, fps_color, 1)
        y += line_h

        # Frame time
        total_ms = stats.get("total_frame_ms", 0)
        frame_color = timing_color(total_ms, 100)
        cv2.putText(canvas, f"Frame: {total_ms:.1f}ms", (x, y), FONT, FONT_SMALL, frame_color, 1)
        y += line_h

        # Separator
        cv2.line(canvas, (x, y - 4), (x + panel_w, y - 4), COLOR_GRAY_BGR, 1)

        # Extractor timing bars
        ext_timings = stats.get("extractor_timings_ms", {})
        bar_max_w = min(70, panel_w - 120)
        bar_scale = max(100.0, max(ext_timings.values()) if ext_timings else 1)

        for name, ms in ext_timings.items():
            bar_color = timing_color(ms, 50)
            short_name = name[:10]
            cv2.putText(canvas, short_name, (x, y), FONT, FONT_SMALL, COLOR_WHITE_BGR, 1)
            cv2.putText(canvas, f"{ms:.0f}ms", (x + 65, y), FONT, FONT_SMALL, bar_color, 1)

            bar_x = x + 105
            bar_y = y - 8
            bar_w = int(min(1.0, ms / bar_scale) * bar_max_w)
            draw_horizontal_bar(canvas, bar_x, bar_y, bar_max_w, 10, ms / bar_scale, bar_color)
            y += line_h

        # Fusion timing
        fusion_ms = stats.get("fusion_ms", 0)
        if fusion_ms > 0:
            fusion_color = timing_color(fusion_ms, 50)
            cv2.putText(canvas, "fusion", (x, y), FONT, FONT_SMALL, COLOR_WHITE_BGR, 1)
            cv2.putText(canvas, f"{fusion_ms:.0f}ms", (x + 65, y), FONT, FONT_SMALL, fusion_color, 1)
            y += line_h

        # Separator
        cv2.line(canvas, (x, y - 4), (x + panel_w, y - 4), COLOR_GRAY_BGR, 1)

        # Bottleneck
        slowest = stats.get("slowest_extractor", "")
        bneck_pct = stats.get("bottleneck_pct", 0)
        if slowest:
            cv2.putText(canvas, f"Bottleneck: {slowest}", (x, y), FONT, FONT_SMALL, COLOR_WHITE_BGR, 1)
            y += line_h
            cv2.putText(canvas, f"  ({bneck_pct:.0f}% of frame)", (x, y), FONT, FONT_SMALL, COLOR_GRAY_BGR, 1)
            y += line_h

        return y

    def _draw_profile_timing(
        self, canvas: np.ndarray, x: int, y: int, panel_w: int, timing: Dict[str, float]
    ) -> int:
        """Draw profile timing information."""
        line_h = 16

        detect_ms = timing.get("detect_ms", 0)
        expr_ms = timing.get("expression_ms", 0)
        total_ms = timing.get("total_ms", 0)
        fps = 1000.0 / total_ms if total_ms > 0 else 0

        for label, ms, threshold in [
            ("Detect", detect_ms, 50),
            ("Express", expr_ms, 50),
            ("Total", total_ms, 100),
        ]:
            color = timing_color(ms, threshold)
            cv2.putText(canvas, f"{label}: {ms:.1f}ms", (x, y), FONT, FONT_SMALL, color, 1)
            y += line_h

        cv2.putText(canvas, f"FPS: {fps:.1f}", (x, y), FONT, FONT_SMALL, COLOR_WHITE_BGR, 1)
        y += line_h

        cv2.line(canvas, (x, y - 4), (x + panel_w, y - 4), COLOR_GRAY_BGR, 1)
        return y

    def _draw_gate_status(
        self, canvas: np.ndarray, x: int, y: int, is_gate_open: bool, in_cooldown: bool
    ) -> int:
        """Draw gate and cooldown status."""
        gate_color = COLOR_GREEN_BGR if is_gate_open else COLOR_GRAY_BGR
        gate_text = "OPEN" if is_gate_open else "CLOSED"
        cv2.putText(canvas, f"Gate: {gate_text}", (x, y), FONT, FONT_MEDIUM, gate_color, 1)

        if in_cooldown:
            cv2.putText(canvas, "  CD", (x + 95, y), FONT, FONT_MEDIUM, (0, 165, 255), 1)

        y += 18
        return y

    def _draw_face_summary(
        self, canvas: np.ndarray, x: int, y: int, obs: Observation
    ) -> int:
        """Draw basic face count and emotion summary."""
        face_count = len(obs.faces)
        happy = obs.signals.get("expression_happy", 0)
        cv2.putText(canvas, f"Faces: {face_count}", (x, y), FONT, FONT_SMALL, COLOR_WHITE_BGR, 1)
        y += 14
        cv2.putText(canvas, f"Happy: {happy:.2f}", (x, y), FONT, FONT_SMALL, COLOR_HAPPY_BGR, 1)
        y += 16
        return y

    def _draw_classifier_summary(
        self, canvas: np.ndarray, x: int, y: int, obs: Observation
    ) -> int:
        """Draw classifier summary (main/passenger/transient/noise counts)."""
        if obs.data is None:
            return y

        data = obs.data
        main_detected = 1 if hasattr(data, "main_face") and data.main_face else 0
        passenger_count = len(data.passenger_faces) if hasattr(data, "passenger_faces") else 0
        transient_count = data.transient_count if hasattr(data, "transient_count") else 0
        noise_count = data.noise_count if hasattr(data, "noise_count") else 0

        cv2.putText(canvas, f"Main: {main_detected}  Pass: {passenger_count}", (x, y), FONT, FONT_SMALL, COLOR_WHITE_BGR, 1)
        y += 14
        cv2.putText(canvas, f"Trans: {transient_count}  Noise: {noise_count}", (x, y), FONT, FONT_SMALL, COLOR_GRAY_BGR, 1)
        y += 16

        # Show all faces with details
        if hasattr(data, "faces") and data.faces:
            cv2.putText(canvas, "Faces:", (x, y), FONT, FONT_SMALL, COLOR_GRAY_BGR, 1)
            y += 12
            for cf in data.faces[:4]:  # Max 4 faces
                face = cf.face
                role = cf.role
                track = cf.track_length
                # Color by role
                role_colors = {
                    "main": COLOR_MAIN_BGR,
                    "passenger": (0, 165, 255),  # orange
                    "transient": (0, 255, 255),  # yellow
                    "noise": COLOR_GRAY_BGR,
                }
                color = role_colors.get(role, COLOR_WHITE_BGR)
                # Show: ID, role, confidence, area, track length
                area = cf.avg_area
                info = f"  {face.face_id}: {role[:4]} c={face.confidence:.2f} a={area:.3f} t={track}"
                cv2.putText(canvas, info, (x, y), FONT, 0.32, color, 1)
                y += 11

        return y

    def _draw_fusion_info(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        result: Observation,
        face_obs: Optional[Observation],
    ) -> int:
        """Draw adaptive fusion info (spike bar, baseline)."""
        adaptive_summary = result.metadata.get("adaptive_summary") if result.metadata else None
        if not adaptive_summary:
            return y

        threshold = adaptive_summary.get("threshold", 0.12)
        max_spike = adaptive_summary.get("max_spike", 0)

        first_state = next(iter(adaptive_summary.get("states", {}).values()), None)
        if first_state:
            baseline = first_state["baseline"]
            spike = first_state["spike"]

            cv2.putText(canvas, f"Base: {baseline:.2f}", (x, y), FONT, FONT_SMALL, COLOR_GRAY_BGR, 1)
            y += 14

            spike_color = COLOR_RED_BGR if spike > threshold else (COLOR_HAPPY_BGR if spike > threshold * 0.5 else COLOR_GRAY_BGR)
            cv2.putText(canvas, f"Spike: {spike:+.2f}", (x, y), FONT, FONT_SMALL, spike_color, 1)
            y += 14

        # Spike bar
        panel_w = 160
        spike_pct = min(1.0, max(0, max_spike) / (threshold * 2))
        bar_color = COLOR_RED_BGR if max_spike > threshold else COLOR_GREEN_BGR
        draw_horizontal_bar(canvas, x, y, panel_w, 12, spike_pct, bar_color)
        # Threshold marker at 50%
        th_x = x + panel_w // 2
        cv2.line(canvas, (th_x, y - 1), (th_x, y + 13), COLOR_WHITE_BGR, 2)
        cv2.putText(canvas, f"{max_spike:.2f}", (x + panel_w + 4, y + 10), FONT, 0.25, COLOR_WHITE_BGR, 1)
        y += 20

        return y

    def _capture_thumbnail(
        self,
        source_image: Optional[np.ndarray],
        face_obs: Optional[Observation],
        reason: str,
    ) -> None:
        """Capture largest face as thumbnail on trigger."""
        if source_image is None or face_obs is None or not face_obs.faces:
            return

        h, w = source_image.shape[:2]
        best = max(face_obs.faces, key=lambda f: f.bbox[2] * f.bbox[3])

        bx, by, bw, bh = best.bbox
        x1 = int(max(0, (bx - bw * 0.2) * w))
        y1 = int(max(0, (by - bh * 0.2) * h))
        x2 = int(min(w, (bx + bw * 1.2) * w))
        y2 = int(min(h, (by + bh * 1.2) * h))

        if x2 <= x1 or y2 <= y1:
            return

        crop = source_image[y1:y2, x1:x2]
        if crop.size > 0:
            thumb = cv2.resize(crop, (self.THUMB_SIZE, self.THUMB_SIZE))
            self._trigger_thumbs.append((self._frame_count, thumb, reason))
            if len(self._trigger_thumbs) > self.MAX_THUMBS:
                self._trigger_thumbs.pop(0)

    def _draw_thumbnails(
        self, canvas: np.ndarray, rx1: int, y_start: int, rx2: int
    ) -> None:
        """Draw trigger thumbnails."""
        if not self._trigger_thumbs:
            return

        thumb_x = rx1 + 8
        available_w = rx2 - rx1 - 16

        for i, (_, thumb, reason) in enumerate(self._trigger_thumbs):
            y_pos = y_start + i * (self.THUMB_SIZE + 18)
            if y_pos + self.THUMB_SIZE > canvas.shape[0] - 10:
                break

            cv2.rectangle(
                canvas,
                (thumb_x - 2, y_pos - 2),
                (thumb_x + self.THUMB_SIZE + 2, y_pos + self.THUMB_SIZE + 2),
                COLOR_RED_BGR, 2,
            )
            canvas[
                y_pos : y_pos + self.THUMB_SIZE,
                thumb_x : thumb_x + self.THUMB_SIZE,
            ] = thumb
            cv2.putText(
                canvas, reason[:18], (thumb_x, y_pos + self.THUMB_SIZE + 12),
                FONT, 0.25, COLOR_WHITE_BGR, 1,
            )
