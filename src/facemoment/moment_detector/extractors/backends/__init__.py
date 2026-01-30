"""Backend implementations for feature extraction.

Each backend can be imported independently to avoid dependency conflicts.

Usage:
    # Import base types (always available)
    from facemoment.moment_detector.extractors.backends import DetectedFace, FaceExpression

    # Import specific backends (requires corresponding dependencies)
    from facemoment.moment_detector.extractors.backends.face_backends import InsightFaceSCRFD
    from facemoment.moment_detector.extractors.backends.pose_backends import YOLOPoseBackend
    from facemoment.moment_detector.extractors.backends.hand_backends import MediaPipeHandsBackend
"""

# Base types - always available (no ML dependencies)
from facemoment.moment_detector.extractors.backends.base import (
    FaceDetectionBackend,
    ExpressionBackend,
    PoseBackend,
    HandLandmarkBackend,
    DetectedFace,
    FaceExpression,
    PoseKeypoints,
    HandLandmarks,
)

__all__ = [
    # Protocols
    "FaceDetectionBackend",
    "ExpressionBackend",
    "PoseBackend",
    "HandLandmarkBackend",
    # Data classes
    "DetectedFace",
    "FaceExpression",
    "PoseKeypoints",
    "HandLandmarks",
    # Lazy imports (import directly from submodule)
    # "InsightFaceSCRFD",     # from .face_backends import InsightFaceSCRFD
    # "PyFeatBackend",        # from .face_backends import PyFeatBackend
    # "HSEmotionBackend",     # from .face_backends import HSEmotionBackend
    # "YOLOPoseBackend",      # from .pose_backends import YOLOPoseBackend
    # "MediaPipeHandsBackend",# from .hand_backends import MediaPipeHandsBackend
]


def __getattr__(name: str):
    """Lazy import for ML-dependent backends."""
    # Face backends
    if name in ("InsightFaceSCRFD", "PyFeatBackend", "HSEmotionBackend"):
        from facemoment.moment_detector.extractors.backends import face_backends
        return getattr(face_backends, name)
    # Pose backends
    elif name == "YOLOPoseBackend":
        from facemoment.moment_detector.extractors.backends.pose_backends import YOLOPoseBackend
        return YOLOPoseBackend
    # Hand backends
    elif name == "MediaPipeHandsBackend":
        from facemoment.moment_detector.extractors.backends.hand_backends import MediaPipeHandsBackend
        return MediaPipeHandsBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
