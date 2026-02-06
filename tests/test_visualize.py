"""Tests for visualization utilities."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from facemoment.moment_detector.visualize import (
    ExtractorVisualizer,
    DebugVisualizer,
    VisualizationConfig,
)
from facemoment.moment_detector.extractors.base import Observation, FaceObservation
from facemoment.moment_detector.extractors.outputs import PoseOutput
from facemoment.moment_detector.extractors.types import KeypointIndex


class TestExtractorVisualizer:
    """Tests for ExtractorVisualizer."""

    @pytest.fixture
    def visualizer(self):
        return ExtractorVisualizer()

    @pytest.fixture
    def sample_image(self):
        """Create a sample 640x480 BGR image."""
        return np.zeros((480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_face_observation(self):
        """Create a sample face observation."""
        return Observation(
            source="face",
            frame_id=1,
            t_ns=0,
            signals={"expression_happy": 0.8, "expression_angry": 0.1, "expression_neutral": 0.1},
            faces=[
                FaceObservation(
                    face_id=1,
                    confidence=0.95,
                    bbox=(0.3, 0.2, 0.2, 0.3),  # x, y, w, h normalized
                    inside_frame=True,
                    yaw=5.0,
                    pitch=3.0,
                    roll=0.0,
                    area_ratio=0.06,
                    signals={"em_happy": 0.8, "em_angry": 0.1, "em_neutral": 0.1},
                ),
            ],
        )

    @pytest.fixture
    def sample_pose_observation(self):
        """Create a sample pose observation with keypoints."""
        # Create 17 keypoints (COCO format) with (x, y, confidence)
        keypoints = np.zeros((17, 3))
        # Set upper body keypoints with high confidence
        keypoints[KeypointIndex.NOSE] = [320, 100, 0.9]
        keypoints[KeypointIndex.LEFT_EYE] = [310, 90, 0.85]
        keypoints[KeypointIndex.RIGHT_EYE] = [330, 90, 0.85]
        keypoints[KeypointIndex.LEFT_EAR] = [300, 95, 0.7]
        keypoints[KeypointIndex.RIGHT_EAR] = [340, 95, 0.7]
        keypoints[KeypointIndex.LEFT_SHOULDER] = [280, 180, 0.9]
        keypoints[KeypointIndex.RIGHT_SHOULDER] = [360, 180, 0.9]
        keypoints[KeypointIndex.LEFT_ELBOW] = [250, 250, 0.8]
        keypoints[KeypointIndex.RIGHT_ELBOW] = [390, 250, 0.8]
        keypoints[KeypointIndex.LEFT_WRIST] = [230, 320, 0.75]
        keypoints[KeypointIndex.RIGHT_WRIST] = [410, 320, 0.75]

        return Observation(
            source="pose",
            frame_id=1,
            t_ns=0,
            signals={
                "person_count": 1.0,
                "hands_raised_count": 0.0,
                "hand_wave_detected": 0.0,
            },
            data=PoseOutput(
                keypoints=[{
                    "person_id": 0,
                    "keypoints": keypoints.tolist(),
                    "image_size": (640, 480),
                }],
                person_count=1,
            ),
        )

    def test_draw_face_observation(self, visualizer, sample_image, sample_face_observation):
        """Test drawing face observation."""
        result = visualizer.draw_face_observation(sample_image, sample_face_observation)

        assert result is not None
        assert result.shape == sample_image.shape
        # Result should be different from input (annotations drawn)
        assert not np.array_equal(result, sample_image)

    def test_draw_face_observation_no_faces(self, visualizer, sample_image):
        """Test drawing with no faces."""
        obs = Observation(
            source="face",
            frame_id=1,
            t_ns=0,
            signals={},
            faces=[],
        )
        result = visualizer.draw_face_observation(sample_image, obs)

        assert result is not None
        assert result.shape == sample_image.shape

    def test_draw_pose_observation(self, visualizer, sample_image, sample_pose_observation):
        """Test drawing pose observation with skeleton."""
        result = visualizer.draw_pose_observation(sample_image, sample_pose_observation)

        assert result is not None
        assert result.shape == sample_image.shape
        # Result should be different from input (skeleton drawn)
        assert not np.array_equal(result, sample_image)

    def test_draw_pose_observation_no_keypoints(self, visualizer, sample_image):
        """Test drawing pose with no keypoints."""
        obs = Observation(
            source="pose",
            frame_id=1,
            t_ns=0,
            signals={"person_count": 0.0},
            data=PoseOutput(keypoints=[], person_count=0),
        )
        result = visualizer.draw_pose_observation(sample_image, obs)

        assert result is not None
        assert result.shape == sample_image.shape

    def test_draw_quality_observation(self, visualizer, sample_image):
        """Test drawing quality observation."""
        obs = Observation(
            source="quality",
            frame_id=1,
            t_ns=0,
            signals={
                "blur_quality": 1.0,
                "brightness_quality": 0.8,
                "contrast_quality": 0.9,
                "quality_gate": 1.0,
            },
        )
        result = visualizer.draw_quality_observation(sample_image, obs)

        assert result is not None
        assert result.shape == sample_image.shape


class TestDebugVisualizer:
    """Tests for DebugVisualizer."""

    @pytest.fixture
    def visualizer(self):
        return DebugVisualizer()

    @pytest.fixture
    def sample_frame(self):
        """Create a sample Frame."""
        frame = MagicMock()
        frame.data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame.frame_id = 1
        frame.t_src_ns = 0
        return frame

    def test_create_debug_view_basic(self, visualizer, sample_frame):
        """Test creating basic debug view."""
        result = visualizer.create_debug_view(sample_frame)

        assert result is not None
        # Canvas is larger than video: video + side panel + bottom panel
        vh, vw = sample_frame.data.shape[:2]
        assert result.shape[0] > vh  # includes bottom panel
        assert result.shape[1] > vw  # includes side panel
        assert result.shape[2] == 3

    def test_create_debug_view_with_roi(self, visualizer, sample_frame):
        """Test creating debug view with ROI."""
        roi = (0.1, 0.1, 0.9, 0.9)
        result = visualizer.create_debug_view(sample_frame, roi=roi)

        assert result is not None
        vh, vw = sample_frame.data.shape[:2]
        assert result.shape[0] > vh
        assert result.shape[1] > vw
        assert result.shape[2] == 3

    def test_reset(self, visualizer):
        """Test reset method."""
        visualizer.reset()
        # Should not raise any errors


class TestFaceClassifierVisualization:
    """Tests for face classifier visualization."""

    @pytest.fixture
    def visualizer(self):
        return ExtractorVisualizer()

    @pytest.fixture
    def sample_image(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_draw_face_classifier_observation(self, visualizer, sample_image):
        """Test drawing face classifier results."""
        from dataclasses import dataclass, field
        from typing import List, Optional

        # Create mock ClassifiedFace structure
        @dataclass
        class MockClassifiedFace:
            face: FaceObservation
            role: str
            confidence: float
            track_length: int
            avg_area: float

        @dataclass
        class MockFaceClassifierOutput:
            faces: List[MockClassifiedFace] = field(default_factory=list)
            main_face: Optional[MockClassifiedFace] = None
            passenger_faces: List[MockClassifiedFace] = field(default_factory=list)
            transient_count: int = 0
            noise_count: int = 0

        # Create test data
        main_face = FaceObservation(
            face_id=1,
            confidence=0.95,
            bbox=(0.3, 0.2, 0.2, 0.3),
            signals={"em_happy": 0.7},
        )
        passenger_face = FaceObservation(
            face_id=2,
            confidence=0.85,
            bbox=(0.6, 0.3, 0.15, 0.25),
            signals={"em_happy": 0.3},
        )

        cf_main = MockClassifiedFace(
            face=main_face,
            role="main",
            confidence=0.9,
            track_length=50,
            avg_area=0.06,
        )
        cf_passenger = MockClassifiedFace(
            face=passenger_face,
            role="passenger",
            confidence=0.7,
            track_length=30,
            avg_area=0.04,
        )

        output = MockFaceClassifierOutput(
            faces=[cf_main, cf_passenger],
            main_face=cf_main,
            passenger_faces=[cf_passenger],
            transient_count=1,
            noise_count=2,
        )

        obs = Observation(
            source="face_classifier",
            frame_id=1,
            t_ns=0,
            signals={},
            data=output,
        )

        result = visualizer.draw_face_classifier_observation(sample_image, obs)

        assert result is not None
        assert result.shape == sample_image.shape
        # Result should show annotations
        assert not np.array_equal(result, sample_image)

    def test_draw_face_classifier_observation_no_data(self, visualizer, sample_image):
        """Test drawing with no classifier data."""
        obs = Observation(
            source="face_classifier",
            frame_id=1,
            t_ns=0,
            signals={},
            data=None,
        )
        result = visualizer.draw_face_classifier_observation(sample_image, obs)

        assert result is not None
        assert result.shape == sample_image.shape


class TestScoreVisualization:
    """Tests for frame score visualization in debug view."""

    @pytest.fixture
    def visualizer(self):
        return DebugVisualizer()

    @pytest.fixture
    def sample_frame(self):
        """Create a sample Frame."""
        frame = MagicMock()
        frame.data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame.frame_id = 1
        frame.t_src_ns = 0
        return frame

    def test_create_debug_view_with_score_result(self, visualizer, sample_frame):
        """Test creating debug view with score result."""
        from facemoment.moment_detector.scoring import ScoreResult

        score_result = ScoreResult(
            total_score=0.75,
            technical_score=0.80,
            action_score=0.70,
            identity_score=0.85,
            is_filtered=False,
            filter_reason=None,
        )

        result = visualizer.create_debug_view(sample_frame, score_result=score_result)

        assert result is not None
        vh, vw = sample_frame.data.shape[:2]
        assert result.shape[0] > vh
        assert result.shape[1] > vw

    def test_create_debug_view_with_filtered_score(self, visualizer, sample_frame):
        """Test creating debug view with filtered score result."""
        from facemoment.moment_detector.scoring import ScoreResult

        score_result = ScoreResult(
            total_score=0.0,
            technical_score=0.0,
            action_score=0.0,
            identity_score=0.0,
            is_filtered=True,
            filter_reason="no_face",
        )

        result = visualizer.create_debug_view(sample_frame, score_result=score_result)

        assert result is not None
        vh, vw = sample_frame.data.shape[:2]
        assert result.shape[0] > vh
        assert result.shape[1] > vw
