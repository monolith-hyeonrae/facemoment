"""Tests for FaceExtractor."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from visualbase import Frame

from facemoment.moment_detector.extractors.face import FaceExtractor
from facemoment.moment_detector.extractors.backends.base import (
    DetectedFace,
    FaceExpression,
)


class MockFaceBackend:
    """Mock face detection backend for testing."""

    def __init__(self, faces: list[DetectedFace] = None):
        self._faces = faces or []

    def initialize(self, device: str) -> None:
        pass

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        return self._faces

    def cleanup(self) -> None:
        pass


class MockExpressionBackend:
    """Mock expression analysis backend for testing."""

    def __init__(self, expressions: list[FaceExpression] = None):
        self._expressions = expressions or []

    def initialize(self, device: str) -> None:
        pass

    def analyze(
        self, image: np.ndarray, faces: list[DetectedFace]
    ) -> list[FaceExpression]:
        # Return expressions matching number of faces
        return self._expressions[: len(faces)]

    def cleanup(self) -> None:
        pass


class TestFaceExtractor:
    def test_extract_no_faces(self):
        """Test extraction with no faces detected."""
        face_backend = MockFaceBackend(faces=[])
        expr_backend = MockExpressionBackend()

        extractor = FaceExtractor(
            face_backend=face_backend,
            expression_backend=expr_backend,
        )
        extractor.initialize()

        frame = Frame.from_array(
            np.zeros((480, 640, 3), dtype=np.uint8),
            frame_id=0,
            t_src_ns=0,
        )

        obs = extractor.extract(frame)

        assert obs is not None
        assert obs.source == "face"
        assert obs.signals["face_count"] == 0
        assert obs.signals["max_expression"] == 0.0
        assert len(obs.faces) == 0

        extractor.cleanup()

    def test_extract_single_face(self):
        """Test extraction with single face."""
        face = DetectedFace(
            bbox=(100, 100, 200, 200),  # x, y, w, h
            confidence=0.95,
            yaw=5.0,
            pitch=-3.0,
            roll=1.0,
        )
        expression = FaceExpression(
            action_units={"AU12": 0.7, "AU06": 0.5},
            emotions={"happy": 0.8, "neutral": 0.2},
            expression_intensity=0.75,
            dominant_emotion="happy",
        )

        face_backend = MockFaceBackend(faces=[face])
        expr_backend = MockExpressionBackend(expressions=[expression])

        extractor = FaceExtractor(
            face_backend=face_backend,
            expression_backend=expr_backend,
        )
        extractor.initialize()

        frame = Frame.from_array(
            np.zeros((480, 640, 3), dtype=np.uint8),
            frame_id=1,
            t_src_ns=33_333_333,  # ~30fps
        )

        obs = extractor.extract(frame)

        assert obs is not None
        assert obs.signals["face_count"] == 1
        assert obs.signals["max_expression"] == 0.75
        assert len(obs.faces) == 1

        face_obs = obs.faces[0]
        assert face_obs.confidence == 0.95
        assert face_obs.yaw == 5.0
        assert face_obs.pitch == -3.0
        assert face_obs.expression == 0.75
        # Normalized bbox
        assert pytest.approx(face_obs.bbox[0], rel=0.01) == 100 / 640
        assert pytest.approx(face_obs.bbox[1], rel=0.01) == 100 / 480

        extractor.cleanup()

    def test_extract_multiple_faces(self):
        """Test extraction with multiple faces."""
        faces = [
            DetectedFace(bbox=(50, 100, 150, 180), confidence=0.9, yaw=-10.0),
            DetectedFace(bbox=(350, 100, 150, 180), confidence=0.85, yaw=15.0),
        ]
        expressions = [
            FaceExpression(expression_intensity=0.3),
            FaceExpression(expression_intensity=0.9),
        ]

        face_backend = MockFaceBackend(faces=faces)
        expr_backend = MockExpressionBackend(expressions=expressions)

        extractor = FaceExtractor(
            face_backend=face_backend,
            expression_backend=expr_backend,
            roi=(0, 0, 1, 1),  # Full frame - no ROI filtering for this test
        )
        extractor.initialize()

        frame = Frame.from_array(
            np.zeros((480, 640, 3), dtype=np.uint8),
            frame_id=0,
            t_src_ns=0,
        )

        obs = extractor.extract(frame)

        assert obs.signals["face_count"] == 2
        assert obs.signals["max_expression"] == 0.9
        assert len(obs.faces) == 2

        extractor.cleanup()

    def test_face_tracking(self):
        """Test simple IoU-based face tracking."""
        # Frame 1: Face at position A
        face1 = DetectedFace(bbox=(100, 100, 200, 200), confidence=0.9)

        face_backend = MockFaceBackend(faces=[face1])
        expr_backend = MockExpressionBackend(expressions=[FaceExpression()])

        extractor = FaceExtractor(
            face_backend=face_backend,
            expression_backend=expr_backend,
            track_faces=True,
            roi=(0, 0, 1, 1),  # Full frame - no ROI filtering for tracking test
        )
        extractor.initialize()

        frame1 = Frame.from_array(
            np.zeros((480, 640, 3), dtype=np.uint8),
            frame_id=0,
            t_src_ns=0,
        )
        obs1 = extractor.extract(frame1)
        first_face_id = obs1.faces[0].face_id

        # Frame 2: Face at slightly moved position (should keep same ID)
        face2 = DetectedFace(bbox=(110, 105, 200, 200), confidence=0.9)
        face_backend._faces = [face2]

        frame2 = Frame.from_array(
            np.zeros((480, 640, 3), dtype=np.uint8),
            frame_id=1,
            t_src_ns=33_333_333,
        )
        obs2 = extractor.extract(frame2)

        assert obs2.faces[0].face_id == first_face_id  # Same ID due to IoU overlap

        # Frame 3: Face at completely different position (should get new ID)
        face3 = DetectedFace(bbox=(400, 100, 100, 100), confidence=0.9)
        face_backend._faces = [face3]

        frame3 = Frame.from_array(
            np.zeros((480, 640, 3), dtype=np.uint8),
            frame_id=2,
            t_src_ns=66_666_666,
        )
        obs3 = extractor.extract(frame3)

        assert obs3.faces[0].face_id != first_face_id  # New ID, no overlap

        extractor.cleanup()

    def test_inside_frame_detection(self):
        """Test detection of faces at frame edges."""
        # Face partially outside frame
        face_edge = DetectedFace(bbox=(0, 0, 100, 100), confidence=0.9)

        face_backend = MockFaceBackend(faces=[face_edge])
        expr_backend = MockExpressionBackend(expressions=[FaceExpression()])

        # Use full-frame ROI to test inside_frame detection without ROI filtering
        extractor = FaceExtractor(
            face_backend=face_backend,
            expression_backend=expr_backend,
            roi=(0, 0, 1, 1),  # Full frame - no ROI filtering
        )
        extractor.initialize()

        frame = Frame.from_array(
            np.zeros((480, 640, 3), dtype=np.uint8),
            frame_id=0,
            t_src_ns=0,
        )
        obs = extractor.extract(frame)

        # Face at (0, 0) is not inside frame (needs margin)
        assert obs.faces[0].inside_frame is False

        extractor.cleanup()

    def test_iou_computation(self):
        """Test IoU computation helper."""
        # Identical boxes
        box1 = (100, 100, 200, 200)
        iou = FaceExtractor._compute_iou(box1, box1)
        assert iou == 1.0

        # Non-overlapping boxes
        box2 = (500, 500, 100, 100)
        iou = FaceExtractor._compute_iou(box1, box2)
        assert iou == 0.0

        # Partially overlapping boxes
        box3 = (200, 100, 200, 200)  # 50% horizontal overlap
        iou = FaceExtractor._compute_iou(box1, box3)
        assert 0.3 < iou < 0.4  # ~1/3 IoU

    def test_context_manager(self):
        """Test context manager usage."""
        face_backend = MockFaceBackend()
        expr_backend = MockExpressionBackend()

        extractor = FaceExtractor(
            face_backend=face_backend,
            expression_backend=expr_backend,
        )

        with extractor:
            frame = Frame.from_array(
                np.zeros((480, 640, 3), dtype=np.uint8),
                frame_id=0,
                t_src_ns=0,
            )
            obs = extractor.extract(frame)
            assert obs is not None

    def test_roi_filtering(self):
        """Test ROI filtering excludes faces outside region."""
        # Face in center of frame (inside ROI) - center at (320, 240) = (0.5, 0.5)
        face_center = DetectedFace(bbox=(270, 190, 100, 100), confidence=0.9)
        # Face in top-left corner (outside default ROI) - center at (25, 25) = (0.04, 0.05)
        face_corner = DetectedFace(bbox=(0, 0, 50, 50), confidence=0.9)

        face_backend = MockFaceBackend(faces=[face_center, face_corner])
        expr_backend = MockExpressionBackend(expressions=[FaceExpression(), FaceExpression()])

        # Default ROI (0.3, 0.1, 0.7, 0.6) should filter out corner face
        extractor = FaceExtractor(
            face_backend=face_backend,
            expression_backend=expr_backend,
        )
        extractor.initialize()

        frame = Frame.from_array(
            np.zeros((480, 640, 3), dtype=np.uint8),
            frame_id=0,
            t_src_ns=0,
        )
        obs = extractor.extract(frame)

        # Only center face should be included (corner face filtered out)
        assert obs.signals["face_count"] == 1
        assert len(obs.faces) == 1

        extractor.cleanup()

    def test_roi_property(self):
        """Test ROI property getter and setter."""
        extractor = FaceExtractor()

        # Default ROI (center 40% width, top 50% height)
        assert extractor.roi == (0.3, 0.1, 0.7, 0.6)

        # Valid ROI update
        extractor.roi = (0.2, 0.2, 0.8, 0.8)
        assert extractor.roi == (0.2, 0.2, 0.8, 0.8)

        # Invalid ROI should raise
        with pytest.raises(ValueError):
            extractor.roi = (0.8, 0.2, 0.2, 0.8)  # x1 > x2

        with pytest.raises(ValueError):
            extractor.roi = (-0.1, 0.2, 0.8, 0.8)  # x1 < 0

        with pytest.raises(ValueError):
            extractor.roi = (0.2, 0.2, 1.1, 0.8)  # x2 > 1
