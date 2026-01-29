"""Pose extractor for gesture and body analysis."""

from typing import Optional, Dict, List
from collections import deque
import logging

import numpy as np

from visualbase import Frame

from facemoment.moment_detector.extractors.base import (
    BaseExtractor,
    Observation,
)
from facemoment.moment_detector.extractors.backends.base import (
    PoseBackend,
    PoseKeypoints,
)

logger = logging.getLogger(__name__)


# Keypoint indices for COCO 17 format
class KeypointIndex:
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


class PoseExtractor(BaseExtractor):
    """Extractor for pose estimation and gesture detection.

    Uses YOLOv8-Pose or similar backends to extract body keypoints
    and detect gestures like hand waving.

    Features:
    - Upper body keypoint extraction (shoulders, elbows, wrists)
    - Hand position analysis (above/below shoulders)
    - Hand waving detection (x-axis oscillation pattern)
    - Multi-person tracking

    Args:
        pose_backend: Pose estimation backend (default: YOLOPoseBackend).
        device: Device for inference (default: "cuda:0").
        wave_window_frames: Number of frames for wave detection (default: 15).
        wave_frequency_min: Minimum oscillation frequency for wave (default: 1.5 Hz).
        wave_frequency_max: Maximum oscillation frequency for wave (default: 5.0 Hz).
        wave_amplitude_threshold: Minimum x-movement for wave (default: 0.05 normalized).

    Example:
        >>> extractor = PoseExtractor()
        >>> with extractor:
        ...     obs = extractor.extract(frame)
        ...     if obs.signals.get("hand_wave_detected", 0) > 0:
        ...         print("Hand wave detected!")
    """

    def __init__(
        self,
        pose_backend: Optional[PoseBackend] = None,
        device: str = "cuda:0",
        wave_window_frames: int = 15,
        wave_frequency_min: float = 1.5,
        wave_frequency_max: float = 5.0,
        wave_amplitude_threshold: float = 0.05,
    ):
        self._device = device
        self._pose_backend = pose_backend
        self._wave_window = wave_window_frames
        self._wave_freq_min = wave_frequency_min
        self._wave_freq_max = wave_frequency_max
        self._wave_amp_threshold = wave_amplitude_threshold
        self._initialized = False

        # History for wave detection (per person)
        # Key: person_id, Value: deque of (t_ns, left_wrist_x, right_wrist_x)
        self._wrist_history: Dict[int, deque] = {}
        self._last_fps_estimate = 30.0

    @property
    def name(self) -> str:
        return "pose"

    def initialize(self) -> None:
        """Initialize pose estimation backend."""
        if self._initialized:
            return  # Already initialized

        if self._pose_backend is None:
            from facemoment.moment_detector.extractors.backends.pose_backends import (
                YOLOPoseBackend,
            )

            self._pose_backend = YOLOPoseBackend()

        self._pose_backend.initialize(self._device)
        self._initialized = True
        logger.info("PoseExtractor initialized")

    def cleanup(self) -> None:
        """Release backend resources."""
        if self._pose_backend is not None:
            self._pose_backend.cleanup()

        self._wrist_history.clear()
        logger.info("PoseExtractor cleaned up")

    def extract(self, frame: Frame) -> Optional[Observation]:
        """Extract pose observations from a frame.

        Args:
            frame: Input frame to analyze.

        Returns:
            Observation with pose signals and gesture detection.
        """
        if self._pose_backend is None:
            raise RuntimeError("Extractor not initialized. Call initialize() first.")

        image = frame.data
        h, w = image.shape[:2]
        t_ns = frame.t_src_ns

        # Detect poses
        poses = self._pose_backend.detect(image)

        if not poses:
            return Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=t_ns,
                signals={
                    "person_count": 0,
                    "hands_raised_count": 0,
                    "hand_wave_detected": 0.0,
                },
            )

        # Analyze each pose
        hands_raised_count = 0
        wave_detected = False
        wave_confidence = 0.0
        pose_signals: Dict[str, float] = {}

        for pose in poses:
            person_id = pose.person_id or 0
            kpts = pose.keypoints

            # Check hand positions relative to shoulders
            left_raised = self._is_hand_raised(kpts, KeypointIndex.LEFT_WRIST, KeypointIndex.LEFT_SHOULDER)
            right_raised = self._is_hand_raised(kpts, KeypointIndex.RIGHT_WRIST, KeypointIndex.RIGHT_SHOULDER)

            if left_raised or right_raised:
                hands_raised_count += 1

            # Update wrist history for wave detection
            left_wrist_x = self._get_normalized_x(kpts, KeypointIndex.LEFT_WRIST, w)
            right_wrist_x = self._get_normalized_x(kpts, KeypointIndex.RIGHT_WRIST, w)

            if left_wrist_x is not None or right_wrist_x is not None:
                if person_id not in self._wrist_history:
                    self._wrist_history[person_id] = deque(maxlen=self._wave_window)

                self._wrist_history[person_id].append(
                    (t_ns, left_wrist_x, right_wrist_x)
                )

                # Check for wave pattern
                wave_score = self._detect_wave_pattern(person_id, t_ns)
                if wave_score > wave_confidence:
                    wave_confidence = wave_score
                    if wave_score > 0.5:
                        wave_detected = True

            # Store per-person signals
            pose_signals[f"person_{person_id}_left_raised"] = 1.0 if left_raised else 0.0
            pose_signals[f"person_{person_id}_right_raised"] = 1.0 if right_raised else 0.0

        # Clean up old history entries
        self._cleanup_old_history(t_ns)

        signals = {
            "person_count": float(len(poses)),
            "hands_raised_count": float(hands_raised_count),
            "hand_wave_detected": 1.0 if wave_detected else 0.0,
            "hand_wave_confidence": wave_confidence,
            **pose_signals,
        }

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=t_ns,
            signals=signals,
            metadata={
                "wave_detected": wave_detected,
                "poses_detected": len(poses),
            },
        )

    def _is_hand_raised(
        self, keypoints: np.ndarray, wrist_idx: int, shoulder_idx: int
    ) -> bool:
        """Check if hand is raised above shoulder.

        Args:
            keypoints: Array of shape (17, 3) with keypoints.
            wrist_idx: Index of wrist keypoint.
            shoulder_idx: Index of shoulder keypoint.

        Returns:
            True if hand is raised above shoulder.
        """
        wrist = keypoints[wrist_idx]
        shoulder = keypoints[shoulder_idx]

        # Check confidence
        if wrist[2] < 0.5 or shoulder[2] < 0.5:
            return False

        # Y-axis is inverted (0 at top)
        # Hand is raised if wrist.y < shoulder.y (above)
        return wrist[1] < shoulder[1]

    def _get_normalized_x(
        self, keypoints: np.ndarray, idx: int, width: int
    ) -> Optional[float]:
        """Get normalized x-coordinate of a keypoint.

        Args:
            keypoints: Array of shape (17, 3) with keypoints.
            idx: Keypoint index.
            width: Image width.

        Returns:
            Normalized x-coordinate [0, 1], or None if low confidence.
        """
        kpt = keypoints[idx]
        if kpt[2] < 0.5:
            return None
        return float(kpt[0]) / width

    def _detect_wave_pattern(self, person_id: int, current_t_ns: int) -> float:
        """Detect hand waving pattern from wrist history.

        Uses x-axis oscillation analysis to detect waving motion.

        Args:
            person_id: Person ID for history lookup.
            current_t_ns: Current timestamp.

        Returns:
            Wave confidence score [0, 1].
        """
        if person_id not in self._wrist_history:
            return 0.0

        history = self._wrist_history[person_id]
        if len(history) < 5:
            return 0.0

        # Extract wrist x positions over time
        times = []
        left_xs = []
        right_xs = []

        for t_ns, lx, rx in history:
            times.append(t_ns)
            if lx is not None:
                left_xs.append((t_ns, lx))
            if rx is not None:
                right_xs.append((t_ns, rx))

        # Analyze each wrist for oscillation
        left_score = self._compute_oscillation_score(left_xs)
        right_score = self._compute_oscillation_score(right_xs)

        return max(left_score, right_score)

    def _compute_oscillation_score(
        self, positions: List[tuple[int, float]]
    ) -> float:
        """Compute oscillation score from position history.

        Args:
            positions: List of (timestamp_ns, x_position) tuples.

        Returns:
            Oscillation score [0, 1].
        """
        if len(positions) < 5:
            return 0.0

        # Extract positions and compute time span
        times = np.array([p[0] for p in positions])
        xs = np.array([p[1] for p in positions])

        time_span_sec = (times[-1] - times[0]) / 1e9
        if time_span_sec < 0.2:
            return 0.0

        # Compute amplitude (peak-to-peak)
        amplitude = np.max(xs) - np.min(xs)
        if amplitude < self._wave_amp_threshold:
            return 0.0

        # Count direction changes (zero crossings of derivative)
        dx = np.diff(xs)
        sign_changes = np.sum(np.diff(np.sign(dx)) != 0)

        # Estimate frequency (direction changes / 2 / time)
        estimated_freq = sign_changes / 2 / time_span_sec

        # Check if frequency is in valid range
        if self._wave_freq_min <= estimated_freq <= self._wave_freq_max:
            # Score based on amplitude and frequency quality
            amplitude_score = min(1.0, amplitude / (self._wave_amp_threshold * 3))
            freq_center = (self._wave_freq_min + self._wave_freq_max) / 2
            freq_range = (self._wave_freq_max - self._wave_freq_min) / 2
            freq_score = 1.0 - abs(estimated_freq - freq_center) / freq_range

            return amplitude_score * freq_score

        return 0.0

    def _cleanup_old_history(self, current_t_ns: int) -> None:
        """Remove old entries from wrist history.

        Args:
            current_t_ns: Current timestamp.
        """
        max_age_ns = int(1.0 * 1e9)  # 1 second max age
        to_remove = []

        for person_id, history in self._wrist_history.items():
            if history and (current_t_ns - history[-1][0]) > max_age_ns:
                to_remove.append(person_id)

        for person_id in to_remove:
            del self._wrist_history[person_id]
