"""Backend implementations for feature extraction."""

from facemoment.moment_detector.extractors.backends.base import (
    FaceDetectionBackend,
    ExpressionBackend,
    PoseBackend,
    DetectedFace,
    FaceExpression,
    PoseKeypoints,
)
from facemoment.moment_detector.extractors.backends.face_backends import (
    InsightFaceSCRFD,
    PyFeatBackend,
)
from facemoment.moment_detector.extractors.backends.pose_backends import (
    YOLOPoseBackend,
)

__all__ = [
    # Protocols
    "FaceDetectionBackend",
    "ExpressionBackend",
    "PoseBackend",
    # Data classes
    "DetectedFace",
    "FaceExpression",
    "PoseKeypoints",
    # Implementations
    "InsightFaceSCRFD",
    "PyFeatBackend",
    "YOLOPoseBackend",
]
