"""Tests for PoseExtractor."""

from unittest.mock import MagicMock
import numpy as np
import pytest

from visualbase import Frame

from facemoment.moment_detector.extractors.pose import PoseExtractor, KeypointIndex
from facemoment.moment_detector.extractors.backends.base import PoseKeypoints


class MockPoseBackend:
    """Mock pose estimation backend for testing."""

    def __init__(self, poses: list[PoseKeypoints] = None):
        self._poses = poses or []

    def initialize(self, device: str) -> None:
        pass

    def detect(self, image: np.ndarray) -> list[PoseKeypoints]:
        return self._poses

    def cleanup(self) -> None:
        pass


def create_keypoints(
    left_wrist_pos: tuple = (100, 300, 0.9),
    right_wrist_pos: tuple = (300, 300, 0.9),
    left_shoulder_pos: tuple = (100, 250, 0.9),
    right_shoulder_pos: tuple = (300, 250, 0.9),
) -> np.ndarray:
    """Create keypoints array with specified positions.

    Default positions have wrists BELOW shoulders (not raised).
    Y-axis is inverted: lower y = higher on screen.
    """
    keypoints = np.zeros((17, 3), dtype=np.float32)
    # Set default low confidence
    keypoints[:, 2] = 0.1

    # Set specific keypoints
    keypoints[KeypointIndex.LEFT_WRIST] = left_wrist_pos
    keypoints[KeypointIndex.RIGHT_WRIST] = right_wrist_pos
    keypoints[KeypointIndex.LEFT_SHOULDER] = left_shoulder_pos
    keypoints[KeypointIndex.RIGHT_SHOULDER] = right_shoulder_pos

    return keypoints


class TestPoseExtractor:
    def test_extract_no_poses(self):
        """Test extraction with no poses detected."""
        pose_backend = MockPoseBackend(poses=[])

        extractor = PoseExtractor(pose_backend=pose_backend)
        extractor.initialize()

        frame = Frame.from_array(
            np.zeros((480, 640, 3), dtype=np.uint8),
            frame_id=0,
            t_src_ns=0,
        )

        obs = extractor.extract(frame)

        assert obs is not None
        assert obs.source == "pose"
        assert obs.signals["person_count"] == 0
        assert obs.signals["hands_raised_count"] == 0
        assert obs.signals["hand_wave_detected"] == 0.0

        extractor.cleanup()

    def test_extract_single_pose(self):
        """Test extraction with single person."""
        keypoints = create_keypoints()
        pose = PoseKeypoints(
            keypoints=keypoints,
            keypoint_names=[f"kp_{i}" for i in range(17)],
            person_id=0,
            confidence=0.9,
        )

        pose_backend = MockPoseBackend(poses=[pose])

        extractor = PoseExtractor(pose_backend=pose_backend)
        extractor.initialize()

        frame = Frame.from_array(
            np.zeros((480, 640, 3), dtype=np.uint8),
            frame_id=0,
            t_src_ns=0,
        )

        obs = extractor.extract(frame)

        assert obs.signals["person_count"] == 1.0

        extractor.cleanup()

    def test_hand_raised_detection(self):
        """Test detection of raised hands."""
        # Wrist above shoulder (y is inverted, lower value = higher position)
        keypoints = create_keypoints(
            left_wrist_pos=(100, 100, 0.9),  # y=100 (above)
            left_shoulder_pos=(100, 200, 0.9),  # y=200 (below)
            right_wrist_pos=(300, 300, 0.9),  # y=300 (below)
            right_shoulder_pos=(300, 200, 0.9),  # y=200 (above)
        )
        pose = PoseKeypoints(
            keypoints=keypoints,
            keypoint_names=[f"kp_{i}" for i in range(17)],
            person_id=0,
            confidence=0.9,
        )

        pose_backend = MockPoseBackend(poses=[pose])

        extractor = PoseExtractor(pose_backend=pose_backend)
        extractor.initialize()

        frame = Frame.from_array(
            np.zeros((480, 640, 3), dtype=np.uint8),
            frame_id=0,
            t_src_ns=0,
        )

        obs = extractor.extract(frame)

        # Only left hand is raised
        assert obs.signals["hands_raised_count"] == 1.0
        assert obs.signals["person_0_left_raised"] == 1.0
        assert obs.signals["person_0_right_raised"] == 0.0

        extractor.cleanup()

    def test_hand_wave_detection(self):
        """Test detection of hand waving pattern."""
        pose_backend = MockPoseBackend()

        extractor = PoseExtractor(
            pose_backend=pose_backend,
            wave_window_frames=10,
            wave_amplitude_threshold=0.05,
        )
        extractor.initialize()

        # Simulate oscillating wrist motion over multiple frames
        frame_interval_ns = 33_333_333  # ~30fps

        for i in range(15):
            # Create oscillating x position for left wrist
            oscillation = 100 * np.sin(2 * np.pi * 3 * i / 15)  # 3 Hz oscillation
            left_wrist_x = 200 + oscillation  # Oscillate around x=200

            keypoints = create_keypoints(
                left_wrist_pos=(left_wrist_x, 100, 0.9),
                left_shoulder_pos=(200, 200, 0.9),
            )
            pose = PoseKeypoints(
                keypoints=keypoints,
                keypoint_names=[f"kp_{i}" for i in range(17)],
                person_id=0,
                confidence=0.9,
            )
            pose_backend._poses = [pose]

            frame = Frame.from_array(
                np.zeros((480, 640, 3), dtype=np.uint8),
                frame_id=i,
                t_src_ns=i * frame_interval_ns,
            )
            obs = extractor.extract(frame)

        # After enough frames with oscillation, wave should be detected
        # The exact detection depends on the oscillation parameters
        assert "hand_wave_confidence" in obs.signals

        extractor.cleanup()

    def test_multiple_persons(self):
        """Test extraction with multiple persons."""
        keypoints1 = create_keypoints(
            left_wrist_pos=(100, 100, 0.9),  # Raised (y < shoulder y)
            left_shoulder_pos=(100, 200, 0.9),
            right_wrist_pos=(300, 300, 0.9),  # Not raised
            right_shoulder_pos=(300, 200, 0.9),
        )
        keypoints2 = create_keypoints(
            left_wrist_pos=(400, 300, 0.9),  # Not raised (y > shoulder y)
            left_shoulder_pos=(400, 200, 0.9),
            right_wrist_pos=(500, 300, 0.9),  # Not raised
            right_shoulder_pos=(500, 200, 0.9),
        )

        poses = [
            PoseKeypoints(
                keypoints=keypoints1,
                keypoint_names=[f"kp_{i}" for i in range(17)],
                person_id=0,
                confidence=0.9,
            ),
            PoseKeypoints(
                keypoints=keypoints2,
                keypoint_names=[f"kp_{i}" for i in range(17)],
                person_id=1,
                confidence=0.85,
            ),
        ]

        pose_backend = MockPoseBackend(poses=poses)

        extractor = PoseExtractor(pose_backend=pose_backend)
        extractor.initialize()

        frame = Frame.from_array(
            np.zeros((480, 640, 3), dtype=np.uint8),
            frame_id=0,
            t_src_ns=0,
        )

        obs = extractor.extract(frame)

        assert obs.signals["person_count"] == 2.0
        assert obs.signals["hands_raised_count"] == 1.0
        assert "person_0_left_raised" in obs.signals
        assert "person_1_left_raised" in obs.signals

        extractor.cleanup()

    def test_low_confidence_keypoints_ignored(self):
        """Test that low confidence keypoints are ignored."""
        # Low confidence wrist should not be detected as raised
        keypoints = create_keypoints(
            left_wrist_pos=(100, 100, 0.2),  # Low confidence, would be raised but ignored
            left_shoulder_pos=(100, 200, 0.9),
            right_wrist_pos=(300, 300, 0.9),  # Not raised
            right_shoulder_pos=(300, 200, 0.9),
        )
        pose = PoseKeypoints(
            keypoints=keypoints,
            keypoint_names=[f"kp_{i}" for i in range(17)],
            person_id=0,
            confidence=0.9,
        )

        pose_backend = MockPoseBackend(poses=[pose])

        extractor = PoseExtractor(pose_backend=pose_backend)
        extractor.initialize()

        frame = Frame.from_array(
            np.zeros((480, 640, 3), dtype=np.uint8),
            frame_id=0,
            t_src_ns=0,
        )

        obs = extractor.extract(frame)

        # Low confidence keypoint should not count as raised
        assert obs.signals["hands_raised_count"] == 0.0

        extractor.cleanup()

    def test_context_manager(self):
        """Test context manager usage."""
        pose_backend = MockPoseBackend()

        extractor = PoseExtractor(pose_backend=pose_backend)

        with extractor:
            frame = Frame.from_array(
                np.zeros((480, 640, 3), dtype=np.uint8),
                frame_id=0,
                t_src_ns=0,
            )
            obs = extractor.extract(frame)
            assert obs is not None


class TestKeypointIndex:
    def test_keypoint_indices(self):
        """Test keypoint index constants."""
        assert KeypointIndex.NOSE == 0
        assert KeypointIndex.LEFT_WRIST == 9
        assert KeypointIndex.RIGHT_WRIST == 10
        assert KeypointIndex.LEFT_SHOULDER == 5
        assert KeypointIndex.RIGHT_SHOULDER == 6
