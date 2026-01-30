"""Tests for GestureExtractor (Phase 9)."""

import pytest
import numpy as np

from facemoment.moment_detector.extractors.gesture import (
    GestureExtractor,
    GestureType,
    HandLandmarkIndex,
)
from facemoment.moment_detector.extractors.backends.base import HandLandmarks


class MockHandBackend:
    """Mock backend for testing without MediaPipe dependency."""

    def __init__(self):
        self._hands = []
        self._initialized = False

    def set_hands(self, hands: list[HandLandmarks]):
        """Set hands to return from detect()."""
        self._hands = hands

    def initialize(self, device: str = "cpu") -> None:
        self._initialized = True

    def detect(self, image: np.ndarray) -> list[HandLandmarks]:
        return self._hands

    def cleanup(self) -> None:
        self._initialized = False


class MockFrame:
    """Mock Frame for testing."""

    def __init__(
        self,
        frame_id: int = 0,
        t_src_ns: int = 0,
        width: int = 640,
        height: int = 480,
    ):
        self.frame_id = frame_id
        self.t_src_ns = t_src_ns
        self.data = np.zeros((height, width, 3), dtype=np.uint8)


def create_hand_landmarks(
    thumb_up: bool = False,
    index_up: bool = False,
    middle_up: bool = False,
    ring_up: bool = False,
    pinky_up: bool = False,
    handedness: str = "Right",
    confidence: float = 0.9,
) -> HandLandmarks:
    """Create hand landmarks with specified finger states.

    Simplified model: fingers up means tip.y < pip.y
    """
    # Base wrist position
    landmarks = np.zeros((21, 3), dtype=np.float32)

    # Set wrist at center
    landmarks[HandLandmarkIndex.WRIST] = [0.5, 0.6, 0.0]

    # Thumb landmarks (x-based check for up)
    # Right hand: thumb up if tip.x > ip.x
    if handedness == "Right":
        if thumb_up:
            landmarks[HandLandmarkIndex.THUMB_CMC] = [0.45, 0.55, 0.0]
            landmarks[HandLandmarkIndex.THUMB_MCP] = [0.42, 0.50, 0.0]
            landmarks[HandLandmarkIndex.THUMB_IP] = [0.40, 0.45, 0.0]
            landmarks[HandLandmarkIndex.THUMB_TIP] = [0.45, 0.40, 0.0]  # tip.x > ip.x
        else:
            landmarks[HandLandmarkIndex.THUMB_CMC] = [0.45, 0.55, 0.0]
            landmarks[HandLandmarkIndex.THUMB_MCP] = [0.42, 0.50, 0.0]
            landmarks[HandLandmarkIndex.THUMB_IP] = [0.40, 0.48, 0.0]
            landmarks[HandLandmarkIndex.THUMB_TIP] = [0.38, 0.50, 0.0]  # tip.x < ip.x
    else:
        # Left hand: thumb up if tip.x < ip.x
        if thumb_up:
            landmarks[HandLandmarkIndex.THUMB_CMC] = [0.55, 0.55, 0.0]
            landmarks[HandLandmarkIndex.THUMB_MCP] = [0.58, 0.50, 0.0]
            landmarks[HandLandmarkIndex.THUMB_IP] = [0.60, 0.45, 0.0]
            landmarks[HandLandmarkIndex.THUMB_TIP] = [0.55, 0.40, 0.0]  # tip.x < ip.x
        else:
            landmarks[HandLandmarkIndex.THUMB_CMC] = [0.55, 0.55, 0.0]
            landmarks[HandLandmarkIndex.THUMB_MCP] = [0.58, 0.50, 0.0]
            landmarks[HandLandmarkIndex.THUMB_IP] = [0.60, 0.48, 0.0]
            landmarks[HandLandmarkIndex.THUMB_TIP] = [0.62, 0.50, 0.0]  # tip.x > ip.x

    # Index finger (y-based: up means tip.y < pip.y)
    base_x = 0.48
    if index_up:
        landmarks[HandLandmarkIndex.INDEX_FINGER_MCP] = [base_x, 0.50, 0.0]
        landmarks[HandLandmarkIndex.INDEX_FINGER_PIP] = [base_x, 0.40, 0.0]
        landmarks[HandLandmarkIndex.INDEX_FINGER_DIP] = [base_x, 0.30, 0.0]
        landmarks[HandLandmarkIndex.INDEX_FINGER_TIP] = [base_x, 0.20, 0.0]  # Above PIP
    else:
        landmarks[HandLandmarkIndex.INDEX_FINGER_MCP] = [base_x, 0.50, 0.0]
        landmarks[HandLandmarkIndex.INDEX_FINGER_PIP] = [base_x, 0.52, 0.0]
        landmarks[HandLandmarkIndex.INDEX_FINGER_DIP] = [base_x, 0.54, 0.0]
        landmarks[HandLandmarkIndex.INDEX_FINGER_TIP] = [base_x, 0.56, 0.0]  # Below PIP

    # Middle finger
    base_x = 0.50
    if middle_up:
        landmarks[HandLandmarkIndex.MIDDLE_FINGER_MCP] = [base_x, 0.48, 0.0]
        landmarks[HandLandmarkIndex.MIDDLE_FINGER_PIP] = [base_x, 0.38, 0.0]
        landmarks[HandLandmarkIndex.MIDDLE_FINGER_DIP] = [base_x, 0.28, 0.0]
        landmarks[HandLandmarkIndex.MIDDLE_FINGER_TIP] = [base_x, 0.18, 0.0]
    else:
        landmarks[HandLandmarkIndex.MIDDLE_FINGER_MCP] = [base_x, 0.48, 0.0]
        landmarks[HandLandmarkIndex.MIDDLE_FINGER_PIP] = [base_x, 0.50, 0.0]
        landmarks[HandLandmarkIndex.MIDDLE_FINGER_DIP] = [base_x, 0.52, 0.0]
        landmarks[HandLandmarkIndex.MIDDLE_FINGER_TIP] = [base_x, 0.54, 0.0]

    # Ring finger
    base_x = 0.52
    if ring_up:
        landmarks[HandLandmarkIndex.RING_FINGER_MCP] = [base_x, 0.50, 0.0]
        landmarks[HandLandmarkIndex.RING_FINGER_PIP] = [base_x, 0.40, 0.0]
        landmarks[HandLandmarkIndex.RING_FINGER_DIP] = [base_x, 0.30, 0.0]
        landmarks[HandLandmarkIndex.RING_FINGER_TIP] = [base_x, 0.20, 0.0]
    else:
        landmarks[HandLandmarkIndex.RING_FINGER_MCP] = [base_x, 0.50, 0.0]
        landmarks[HandLandmarkIndex.RING_FINGER_PIP] = [base_x, 0.52, 0.0]
        landmarks[HandLandmarkIndex.RING_FINGER_DIP] = [base_x, 0.54, 0.0]
        landmarks[HandLandmarkIndex.RING_FINGER_TIP] = [base_x, 0.56, 0.0]

    # Pinky finger
    base_x = 0.54
    if pinky_up:
        landmarks[HandLandmarkIndex.PINKY_MCP] = [base_x, 0.52, 0.0]
        landmarks[HandLandmarkIndex.PINKY_PIP] = [base_x, 0.42, 0.0]
        landmarks[HandLandmarkIndex.PINKY_DIP] = [base_x, 0.32, 0.0]
        landmarks[HandLandmarkIndex.PINKY_TIP] = [base_x, 0.22, 0.0]
    else:
        landmarks[HandLandmarkIndex.PINKY_MCP] = [base_x, 0.52, 0.0]
        landmarks[HandLandmarkIndex.PINKY_PIP] = [base_x, 0.54, 0.0]
        landmarks[HandLandmarkIndex.PINKY_DIP] = [base_x, 0.56, 0.0]
        landmarks[HandLandmarkIndex.PINKY_TIP] = [base_x, 0.58, 0.0]

    return HandLandmarks(
        landmarks=landmarks,
        handedness=handedness,
        confidence=confidence,
    )


class TestGestureExtractor:
    def test_name(self):
        """Test extractor name."""
        backend = MockHandBackend()
        extractor = GestureExtractor(hand_backend=backend)
        assert extractor.name == "gesture"

    def test_initialize_and_cleanup(self):
        """Test initialize and cleanup lifecycle."""
        backend = MockHandBackend()
        extractor = GestureExtractor(hand_backend=backend)

        extractor.initialize()
        assert backend._initialized

        extractor.cleanup()
        assert not backend._initialized

    def test_context_manager(self):
        """Test context manager protocol."""
        backend = MockHandBackend()
        extractor = GestureExtractor(hand_backend=backend)

        with extractor:
            assert backend._initialized

        assert not backend._initialized

    def test_no_hands_detected(self):
        """Test observation when no hands detected."""
        backend = MockHandBackend()
        backend.set_hands([])

        extractor = GestureExtractor(hand_backend=backend)
        extractor.initialize()

        frame = MockFrame(frame_id=0, t_src_ns=1000)
        obs = extractor.extract(frame)

        assert obs is not None
        assert obs.source == "gesture"
        assert obs.signals["hand_count"] == 0
        assert obs.signals["gesture_detected"] == 0.0
        assert obs.metadata["gesture_type"] == ""

    def test_vsign_gesture(self):
        """Test V-sign gesture detection (index + middle up)."""
        backend = MockHandBackend()
        hand = create_hand_landmarks(
            thumb_up=False,
            index_up=True,
            middle_up=True,
            ring_up=False,
            pinky_up=False,
        )
        backend.set_hands([hand])

        extractor = GestureExtractor(hand_backend=backend)
        extractor.initialize()

        frame = MockFrame(frame_id=0, t_src_ns=1000)
        obs = extractor.extract(frame)

        assert obs.signals["hand_count"] == 1
        assert obs.signals["gesture_detected"] == 1.0
        assert obs.metadata["gesture_type"] == "v_sign"
        assert obs.signals["gesture_v_sign"] == 1.0

    def test_thumbsup_gesture(self):
        """Test thumbs up gesture detection (only thumb up)."""
        backend = MockHandBackend()
        hand = create_hand_landmarks(
            thumb_up=True,
            index_up=False,
            middle_up=False,
            ring_up=False,
            pinky_up=False,
        )
        backend.set_hands([hand])

        extractor = GestureExtractor(hand_backend=backend)
        extractor.initialize()

        frame = MockFrame(frame_id=0, t_src_ns=1000)
        obs = extractor.extract(frame)

        assert obs.signals["gesture_detected"] == 1.0
        assert obs.metadata["gesture_type"] == "thumbs_up"
        assert obs.signals["gesture_thumbs_up"] == 1.0

    def test_open_palm_gesture(self):
        """Test open palm gesture detection (all fingers up)."""
        backend = MockHandBackend()
        hand = create_hand_landmarks(
            thumb_up=True,
            index_up=True,
            middle_up=True,
            ring_up=True,
            pinky_up=True,
        )
        backend.set_hands([hand])

        extractor = GestureExtractor(hand_backend=backend)
        extractor.initialize()

        frame = MockFrame(frame_id=0, t_src_ns=1000)
        obs = extractor.extract(frame)

        assert obs.signals["gesture_detected"] == 1.0
        assert obs.metadata["gesture_type"] == "open_palm"
        assert obs.signals["gesture_open_palm"] == 1.0

    def test_fist_gesture(self):
        """Test fist gesture detection (all fingers down)."""
        backend = MockHandBackend()
        hand = create_hand_landmarks(
            thumb_up=False,
            index_up=False,
            middle_up=False,
            ring_up=False,
            pinky_up=False,
        )
        backend.set_hands([hand])

        extractor = GestureExtractor(hand_backend=backend)
        extractor.initialize()

        frame = MockFrame(frame_id=0, t_src_ns=1000)
        obs = extractor.extract(frame)

        assert obs.signals["gesture_detected"] == 1.0
        assert obs.metadata["gesture_type"] == "fist"
        assert obs.signals["gesture_fist"] == 1.0

    def test_pointing_gesture(self):
        """Test pointing gesture detection (only index up)."""
        backend = MockHandBackend()
        hand = create_hand_landmarks(
            thumb_up=False,
            index_up=True,
            middle_up=False,
            ring_up=False,
            pinky_up=False,
        )
        backend.set_hands([hand])

        extractor = GestureExtractor(hand_backend=backend)
        extractor.initialize()

        frame = MockFrame(frame_id=0, t_src_ns=1000)
        obs = extractor.extract(frame)

        assert obs.signals["gesture_detected"] == 1.0
        assert obs.metadata["gesture_type"] == "pointing"
        assert obs.signals["gesture_pointing"] == 1.0

    def test_multiple_hands(self):
        """Test detection with multiple hands."""
        backend = MockHandBackend()
        hand1 = create_hand_landmarks(
            thumb_up=True,
            index_up=False,
            middle_up=False,
            ring_up=False,
            pinky_up=False,
            handedness="Right",
            confidence=0.8,
        )
        hand2 = create_hand_landmarks(
            thumb_up=False,
            index_up=True,
            middle_up=True,
            ring_up=False,
            pinky_up=False,
            handedness="Left",
            confidence=0.9,  # Higher confidence
        )
        backend.set_hands([hand1, hand2])

        extractor = GestureExtractor(hand_backend=backend)
        extractor.initialize()

        frame = MockFrame(frame_id=0, t_src_ns=1000)
        obs = extractor.extract(frame)

        assert obs.signals["hand_count"] == 2
        assert obs.signals["gesture_detected"] == 1.0
        # Best gesture should be V-sign (higher confidence)
        assert obs.metadata["gesture_type"] == "v_sign"
        assert len(obs.metadata["all_gestures"]) == 2

    def test_low_confidence_hand(self):
        """Test that low confidence gestures are not reported."""
        backend = MockHandBackend()
        hand = create_hand_landmarks(
            thumb_up=True,
            confidence=0.3,  # Low confidence
        )
        backend.set_hands([hand])

        extractor = GestureExtractor(
            hand_backend=backend,
            min_gesture_confidence=0.6,  # Require higher confidence
        )
        extractor.initialize()

        frame = MockFrame(frame_id=0, t_src_ns=1000)
        obs = extractor.extract(frame)

        # Hand is detected but gesture confidence is too low
        assert obs.signals["hand_count"] == 1
        assert obs.signals["gesture_detected"] == 0.0
        assert obs.metadata["gesture_type"] == ""

    def test_left_hand_thumbsup(self):
        """Test thumbs up detection for left hand."""
        backend = MockHandBackend()
        hand = create_hand_landmarks(
            thumb_up=True,
            index_up=False,
            middle_up=False,
            ring_up=False,
            pinky_up=False,
            handedness="Left",
        )
        backend.set_hands([hand])

        extractor = GestureExtractor(hand_backend=backend)
        extractor.initialize()

        frame = MockFrame(frame_id=0, t_src_ns=1000)
        obs = extractor.extract(frame)

        assert obs.signals["gesture_detected"] == 1.0
        assert obs.metadata["gesture_type"] == "thumbs_up"

    def test_observation_timestamps(self):
        """Test that observation preserves frame metadata."""
        backend = MockHandBackend()
        backend.set_hands([])

        extractor = GestureExtractor(hand_backend=backend)
        extractor.initialize()

        frame = MockFrame(frame_id=42, t_src_ns=123456789)
        obs = extractor.extract(frame)

        assert obs.frame_id == 42
        assert obs.t_ns == 123456789

    def test_extract_without_initialization_raises(self):
        """Test that extract raises error without initialization."""
        backend = MockHandBackend()
        extractor = GestureExtractor(hand_backend=backend)

        frame = MockFrame()

        with pytest.raises(RuntimeError, match="not initialized"):
            extractor.extract(frame)


class TestGestureType:
    def test_gesture_type_values(self):
        """Test GestureType enum values."""
        assert GestureType.NONE.value == "none"
        assert GestureType.V_SIGN.value == "v_sign"
        assert GestureType.THUMBS_UP.value == "thumbs_up"
        assert GestureType.OK_SIGN.value == "ok_sign"
        assert GestureType.OPEN_PALM.value == "open_palm"
        assert GestureType.FIST.value == "fist"
        assert GestureType.POINTING.value == "pointing"


class TestHandLandmarkIndex:
    def test_landmark_indices(self):
        """Test hand landmark indices match MediaPipe convention."""
        assert HandLandmarkIndex.WRIST == 0
        assert HandLandmarkIndex.THUMB_TIP == 4
        assert HandLandmarkIndex.INDEX_FINGER_TIP == 8
        assert HandLandmarkIndex.MIDDLE_FINGER_TIP == 12
        assert HandLandmarkIndex.RING_FINGER_TIP == 16
        assert HandLandmarkIndex.PINKY_TIP == 20
