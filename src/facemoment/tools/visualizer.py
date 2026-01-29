"""Visualization tools for MomentDetector debugging."""

from typing import Optional
from pathlib import Path

import cv2
import numpy as np

from visualbase import Frame, FrameViewer
from facemoment.moment_detector import MomentDetector
from facemoment.moment_detector.extractors.base import Observation, FaceObservation
from facemoment.moment_detector.fusion.base import FusionResult


class DetectorVisualizer:
    """Visualizes MomentDetector processing in real-time.

    Displays:
    - Video frames with face bounding boxes
    - Expression levels as bars
    - Gate status (open/closed)
    - Trigger events

    Args:
        window_name: OpenCV window name.

    Example:
        >>> viz = DetectorVisualizer()
        >>> for frame, result in detector.process_stream("video.mp4"):
        ...     if not viz.show(frame, result):
        ...         break
        >>> viz.close()
    """

    def __init__(self, window_name: str = "MomentDetector"):
        self._window_name = window_name
        self._viewer = FrameViewer(window_name=window_name, show_info=False)
        self._last_observation: Optional[Observation] = None
        self._trigger_flash_frames: int = 0

    def show(
        self,
        frame: Frame,
        fusion_result: Optional[FusionResult] = None,
        observation: Optional[Observation] = None,
        wait_ms: int = 1,
    ) -> bool:
        """Display frame with overlays.

        Args:
            frame: Video frame to display.
            fusion_result: Result from fusion module.
            observation: Observation from extractor (if available).
            wait_ms: Wait time for key press.

        Returns:
            True to continue, False if user wants to quit.
        """
        if observation is not None:
            self._last_observation = observation

        # Trigger flash effect
        if fusion_result and fusion_result.should_trigger:
            self._trigger_flash_frames = 10

        return self._viewer.show(
            frame,
            wait_ms=wait_ms,
            overlay_fn=lambda img, f: self._draw_overlay(img, f, fusion_result),
        )

    def _draw_overlay(
        self,
        image: np.ndarray,
        frame: Frame,
        fusion_result: Optional[FusionResult],
    ) -> np.ndarray:
        """Draw all visualization overlays."""
        h, w = image.shape[:2]

        # Trigger flash effect (red border)
        if self._trigger_flash_frames > 0:
            cv2.rectangle(image, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
            self._trigger_flash_frames -= 1

        # Draw frame info
        self._draw_frame_info(image, frame)

        # Draw face boxes and info
        if self._last_observation:
            self._draw_faces(image, self._last_observation)
            self._draw_signals(image, self._last_observation)

        # Draw fusion status
        if fusion_result:
            self._draw_fusion_status(image, fusion_result)

        return image

    def _draw_frame_info(self, image: np.ndarray, frame: Frame) -> None:
        """Draw frame ID and timestamp."""
        h, w = image.shape[:2]
        t_sec = frame.t_src_ns / 1_000_000_000

        text = f"Frame {frame.frame_id} | {t_sec:.2f}s"
        cv2.putText(
            image, text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        cv2.putText(
            image, text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
        )

    def _draw_faces(self, image: np.ndarray, obs: Observation) -> None:
        """Draw face bounding boxes and labels."""
        h, w = image.shape[:2]

        for face in obs.faces:
            x, y, fw, fh = face.bbox
            x1, y1 = int(x * w), int(y * h)
            x2, y2 = int((x + fw) * w), int((y + fh) * h)

            # Color based on expression level
            expr = face.expression
            if expr > 0.7:
                color = (0, 255, 0)  # Green for high expression
            elif expr > 0.4:
                color = (0, 255, 255)  # Yellow for medium
            else:
                color = (128, 128, 128)  # Gray for low

            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw expression bar
            bar_width = x2 - x1
            bar_height = 8
            bar_y = y1 - 12
            if bar_y > 10:
                # Background
                cv2.rectangle(
                    image,
                    (x1, bar_y),
                    (x2, bar_y + bar_height),
                    (50, 50, 50),
                    -1,
                )
                # Fill
                fill_width = int(bar_width * expr)
                cv2.rectangle(
                    image,
                    (x1, bar_y),
                    (x1 + fill_width, bar_y + bar_height),
                    color,
                    -1,
                )

            # Face label
            label = f"ID:{face.face_id} E:{expr:.2f}"
            cv2.putText(
                image, label, (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

    def _draw_signals(self, image: np.ndarray, obs: Observation) -> None:
        """Draw signal values panel."""
        h, w = image.shape[:2]

        # Panel background
        panel_x = w - 150
        panel_y = 10
        panel_w = 140
        panel_h = 60

        cv2.rectangle(
            image,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (0, 0, 0),
            -1,
        )
        cv2.rectangle(
            image,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (100, 100, 100),
            1,
        )

        # Signals
        y_offset = panel_y + 18
        for key, value in list(obs.signals.items())[:3]:
            text = f"{key}: {value:.2f}"
            cv2.putText(
                image, text, (panel_x + 5, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
            )
            y_offset += 18

    def _draw_fusion_status(self, image: np.ndarray, result: FusionResult) -> None:
        """Draw fusion module status."""
        h, w = image.shape[:2]

        # Status panel at bottom
        panel_y = h - 40
        cv2.rectangle(image, (0, panel_y), (w, h), (30, 30, 30), -1)

        # Gate status
        state = result.metadata.get("state", "unknown")
        if state == "cooldown":
            status_text = "COOLDOWN"
            status_color = (0, 165, 255)  # Orange
        elif state == "gate_closed":
            status_text = "GATE CLOSED"
            status_color = (128, 128, 128)
        elif result.should_trigger:
            status_text = f"TRIGGER! Score: {result.score:.2f}"
            status_color = (0, 255, 0)
        else:
            consec = result.metadata.get("consecutive_high", 0)
            status_text = f"Monitoring (consec: {consec})"
            status_color = (255, 255, 255)

        cv2.putText(
            image, status_text, (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2,
        )

        # Expression value
        max_expr = result.metadata.get("max_expression", 0)
        if max_expr > 0:
            expr_text = f"Expr: {max_expr:.2f}"
            cv2.putText(
                image, expr_text, (w - 100, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

    def close(self) -> None:
        """Close the visualizer window."""
        self._viewer.close()

    def __enter__(self) -> "DetectorVisualizer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def visualize(
    video_path: str,
    fps: int = 10,
    resolution: Optional[tuple[int, int]] = None,
    num_faces: int = 2,
    expression_threshold: float = 0.7,
    spike_probability: float = 0.1,
    clip_output_dir: str = "./clips",
) -> None:
    """Quick visualization function for testing MomentDetector.

    Args:
        video_path: Path to video file.
        fps: Analysis frame rate.
        resolution: Analysis resolution.
        num_faces: Number of simulated faces.
        expression_threshold: Threshold for triggering.
        spike_probability: Probability of high expression.
        clip_output_dir: Directory for extracted clips.

    Example:
        >>> from facemoment.tools import visualize
        >>> visualize("video.mp4", fps=10, spike_probability=0.2)
    """
    from facemoment.moment_detector.extractors import DummyExtractor
    from facemoment.moment_detector.fusion import DummyFusion

    detector = MomentDetector(
        extractors=[
            DummyExtractor(
                num_faces=num_faces,
                spike_probability=spike_probability,
            )
        ],
        fusion=DummyFusion(
            expression_threshold=expression_threshold,
            consecutive_frames=3,
            cooldown_sec=2.0,
        ),
        clip_output_dir=Path(clip_output_dir),
    )

    print(f"Processing: {video_path}")
    print(f"FPS: {fps}, Resolution: {resolution or 'original'}")
    print(f"Expression threshold: {expression_threshold}")
    print("Press 'q' or ESC to quit")
    print("-" * 40)

    last_obs = None

    def on_obs(obs):
        nonlocal last_obs
        last_obs = obs

    detector.set_on_observation(on_obs)

    with DetectorVisualizer() as viz:
        for frame, result in detector.process_stream(video_path, fps=fps, resolution=resolution):
            if not viz.show(frame, result, last_obs):
                break

    print("-" * 40)
    print(f"Frames processed: {detector.frames_processed}")
    print(f"Triggers fired: {detector.triggers_fired}")
