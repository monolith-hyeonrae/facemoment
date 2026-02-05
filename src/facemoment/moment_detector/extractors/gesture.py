"""Gesture extractor for hand gesture recognition."""

from typing import Optional, List, Dict
import logging
import time

import numpy as np

from visualbase import Frame

from facemoment.moment_detector.extractors.base import (
    Module,
    Observation,
)
from facemoment.moment_detector.extractors.types import GestureType, HandLandmarkIndex
from facemoment.moment_detector.extractors.outputs import GestureOutput
from facemoment.moment_detector.extractors.backends.base import (
    HandLandmarkBackend,
    HandLandmarks,
)
from facemoment.observability import ObservabilityHub, TraceLevel
from facemoment.observability.records import FrameExtractRecord, TimingRecord

logger = logging.getLogger(__name__)

# Get the global observability hub
_hub = ObservabilityHub.get_instance()


class GestureExtractor(Module):
    """Extractor for hand gesture recognition using MediaPipe Hands.

    Detects specific hand gestures useful for gokart scenario:
    - V-sign (peace sign)
    - Thumbs up
    - OK sign
    - Open palm
    - Fist
    - Pointing

    Features:
    - Uses MediaPipe Hands for 21-landmark hand detection
    - Rule-based gesture classification
    - Per-hand gesture detection
    - Confidence scoring

    Args:
        hand_backend: Hand landmark backend (default: MediaPipeHandsBackend).
        device: Device for inference (default: "cpu").
        min_gesture_confidence: Minimum confidence for gesture (default: 0.6).

    Example:
        >>> extractor = GestureExtractor()
        >>> with extractor:
        ...     obs = extractor.process(frame)
        ...     if obs.signals.get("gesture_detected", 0) > 0:
        ...         print(f"Gesture: {obs.metadata.get('gesture_type')}")
    """

    def __init__(
        self,
        hand_backend: Optional[HandLandmarkBackend] = None,
        device: str = "cpu",
        min_gesture_confidence: float = 0.6,
    ):
        self._device = device
        self._hand_backend = hand_backend
        self._min_gesture_confidence = min_gesture_confidence
        self._initialized = False

    @property
    def name(self) -> str:
        return "gesture"

    def initialize(self) -> None:
        """Initialize hand landmark backend."""
        if self._initialized:
            return

        if self._hand_backend is None:
            from facemoment.moment_detector.extractors.backends.hand_backends import (
                MediaPipeHandsBackend,
            )

            self._hand_backend = MediaPipeHandsBackend()

        self._hand_backend.initialize(self._device)
        self._initialized = True
        logger.info("GestureExtractor initialized")

    def cleanup(self) -> None:
        """Release backend resources."""
        if self._hand_backend is not None:
            self._hand_backend.cleanup()

        self._initialized = False
        logger.info("GestureExtractor cleaned up")

    def process(self, frame: Frame, deps=None) -> Optional[Observation]:
        """Extract gesture observations from a frame.

        Args:
            frame: Input frame to analyze.
            deps: Not used (no dependencies).

        Returns:
            Observation with gesture signals.
        """
        if not self._initialized or self._hand_backend is None:
            raise RuntimeError("Extractor not initialized. Call initialize() first.")

        # Start timing for observability
        start_ns = time.perf_counter_ns() if _hub.enabled else 0

        image = frame.data
        t_ns = frame.t_src_ns

        # Detect hands
        hands = self._hand_backend.detect(image)

        if not hands:
            # Emit timing record
            if _hub.enabled:
                processing_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
                self._emit_extract_record(frame, 0, False, "", processing_ms, {})
            return Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=t_ns,
                signals={
                    "hand_count": 0,
                    "gesture_detected": 0.0,
                    "gesture_confidence": 0.0,
                },
                data=GestureOutput(gestures=[], hand_landmarks=[]),
                metadata={
                    "gesture_type": "",
                    "hands_detected": 0,
                },
            )

        # Classify gesture for each hand
        best_gesture = GestureType.NONE
        best_confidence = 0.0
        all_gestures = []

        for hand in hands:
            gesture, confidence = self._classify_gesture(hand)
            all_gestures.append(
                {
                    "handedness": hand.handedness,
                    "gesture": gesture.value,
                    "confidence": confidence,
                }
            )

            # Keep track of best gesture
            if confidence > best_confidence:
                best_gesture = gesture
                best_confidence = confidence

        # Build signals
        gesture_detected = (
            1.0
            if best_gesture != GestureType.NONE
            and best_confidence >= self._min_gesture_confidence
            else 0.0
        )

        signals = {
            "hand_count": float(len(hands)),
            "gesture_detected": gesture_detected,
            "gesture_confidence": best_confidence if gesture_detected else 0.0,
        }

        # Add per-gesture type signals
        for gesture_type in GestureType:
            if gesture_type != GestureType.NONE:
                signals[f"gesture_{gesture_type.value}"] = (
                    1.0 if best_gesture == gesture_type else 0.0
                )

        # Emit observability records
        if _hub.enabled:
            processing_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            self._emit_extract_record(
                frame,
                len(hands),
                gesture_detected > 0,
                best_gesture.value if gesture_detected else "",
                processing_ms,
                signals,
            )

        # Build hand landmarks data for visualization
        h, w = image.shape[:2]
        hand_landmarks_data = []
        for i, hand in enumerate(hands):
            hand_landmarks_data.append({
                "handedness": hand.handedness,
                "landmarks": hand.landmarks.tolist(),  # (21, 3) array
                "confidence": hand.confidence,
                "gesture": all_gestures[i]["gesture"] if i < len(all_gestures) else "",
                "gesture_confidence": all_gestures[i]["confidence"] if i < len(all_gestures) else 0.0,
                "image_size": (w, h),
            })

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=t_ns,
            signals=signals,
            data=GestureOutput(
                gestures=[g["gesture"] for g in all_gestures],
                hand_landmarks=hand_landmarks_data,
            ),
            metadata={
                "gesture_type": best_gesture.value if gesture_detected else "",
                "hands_detected": len(hands),
                "all_gestures": all_gestures,
            },
        )

    def _classify_gesture(self, hand: HandLandmarks) -> tuple[GestureType, float]:
        """Classify gesture from hand landmarks.

        Args:
            hand: Hand landmarks from detection.

        Returns:
            Tuple of (gesture_type, confidence).
        """
        landmarks = hand.landmarks  # Shape: (21, 3)

        # Get finger states (up/down)
        fingers_up = self._get_fingers_up(landmarks, hand.handedness)

        # Classify based on finger states
        gesture, confidence = self._classify_from_fingers(fingers_up, landmarks)

        # Combine with hand detection confidence
        final_confidence = confidence * hand.confidence

        return gesture, final_confidence

    def _get_fingers_up(
        self, landmarks: np.ndarray, handedness: str
    ) -> List[bool]:
        """Determine which fingers are extended.

        Args:
            landmarks: Hand landmarks array (21, 3).
            handedness: "Left" or "Right".

        Returns:
            List of 5 booleans [thumb, index, middle, ring, pinky].
        """
        fingers_up = []

        # Thumb: Compare tip x to IP joint x
        # For right hand: thumb up if tip.x > ip.x
        # For left hand: thumb up if tip.x < ip.x
        thumb_tip = landmarks[HandLandmarkIndex.THUMB_TIP]
        thumb_ip = landmarks[HandLandmarkIndex.THUMB_IP]

        if handedness == "Right":
            thumb_up = thumb_tip[0] > thumb_ip[0]
        else:
            thumb_up = thumb_tip[0] < thumb_ip[0]
        fingers_up.append(thumb_up)

        # Other fingers: Compare tip y to PIP joint y
        # Finger is up if tip.y < pip.y (y increases downward)
        finger_tips = [
            HandLandmarkIndex.INDEX_FINGER_TIP,
            HandLandmarkIndex.MIDDLE_FINGER_TIP,
            HandLandmarkIndex.RING_FINGER_TIP,
            HandLandmarkIndex.PINKY_TIP,
        ]
        finger_pips = [
            HandLandmarkIndex.INDEX_FINGER_PIP,
            HandLandmarkIndex.MIDDLE_FINGER_PIP,
            HandLandmarkIndex.RING_FINGER_PIP,
            HandLandmarkIndex.PINKY_PIP,
        ]

        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            finger_up = tip[1] < pip[1]
            fingers_up.append(finger_up)

        return fingers_up

    def _classify_from_fingers(
        self, fingers_up: List[bool], landmarks: np.ndarray
    ) -> tuple[GestureType, float]:
        """Classify gesture from finger states.

        Args:
            fingers_up: List of finger up states [thumb, index, middle, ring, pinky].
            landmarks: Hand landmarks for additional checks.

        Returns:
            Tuple of (gesture_type, confidence).
        """
        thumb, index, middle, ring, pinky = fingers_up

        # V-sign: Index and middle up, others down
        if not thumb and index and middle and not ring and not pinky:
            return GestureType.V_SIGN, 0.9

        # Thumbs up: Only thumb up
        if thumb and not index and not middle and not ring and not pinky:
            return GestureType.THUMBS_UP, 0.9

        # Open palm: All fingers up
        if all(fingers_up):
            return GestureType.OPEN_PALM, 0.85

        # Fist: All fingers down
        if not any(fingers_up):
            return GestureType.FIST, 0.85

        # Pointing: Only index up
        if not thumb and index and not middle and not ring and not pinky:
            return GestureType.POINTING, 0.85

        # OK sign: Thumb and index touching, others up
        # Check distance between thumb tip and index tip
        if self._is_ok_sign(landmarks, fingers_up):
            return GestureType.OK_SIGN, 0.8

        return GestureType.NONE, 0.0

    def _is_ok_sign(self, landmarks: np.ndarray, fingers_up: List[bool]) -> bool:
        """Check if hand is making OK sign.

        OK sign: Thumb and index tips close together, middle/ring/pinky up.

        Args:
            landmarks: Hand landmarks.
            fingers_up: Finger up states.

        Returns:
            True if OK sign detected.
        """
        thumb, index, middle, ring, pinky = fingers_up

        # Middle, ring, pinky should be up
        if not (middle and ring and pinky):
            return False

        # Thumb and index tips should be close
        thumb_tip = landmarks[HandLandmarkIndex.THUMB_TIP]
        index_tip = landmarks[HandLandmarkIndex.INDEX_FINGER_TIP]

        # Calculate distance (normalized coordinates)
        distance = np.linalg.norm(thumb_tip[:2] - index_tip[:2])

        # Threshold for "touching" (normalized units)
        return distance < 0.05

    def _emit_extract_record(
        self,
        frame: Frame,
        hand_count: int,
        gesture_detected: bool,
        gesture_type: str,
        processing_ms: float,
        signals: Dict[str, float],
    ) -> None:
        """Emit extraction observability records.

        Args:
            frame: The processed frame.
            hand_count: Number of hands detected.
            gesture_detected: Whether a gesture was detected.
            gesture_type: Type of gesture detected.
            processing_ms: Processing time in milliseconds.
            signals: Signal dictionary.
        """
        threshold_ms = 60.0  # Gesture detection can be slower
        _hub.emit(FrameExtractRecord(
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            source=self.name,
            gesture_detected=gesture_detected,
            processing_ms=processing_ms,
            signals=signals if _hub.is_level_enabled(TraceLevel.VERBOSE) else {},
        ))
        _hub.emit(TimingRecord(
            frame_id=frame.frame_id,
            component=self.name,
            processing_ms=processing_ms,
            threshold_ms=threshold_ms,
            is_slow=processing_ms > threshold_ms,
        ))
