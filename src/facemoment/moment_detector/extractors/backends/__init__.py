"""Backend implementations for feature extraction.

Each backend can be imported independently to avoid dependency conflicts.

Usage:
    # Import base types (always available)
    from facemoment.moment_detector.extractors.backends import DetectedFace, FaceExpression

    # Import specific backends (requires corresponding dependencies)
    from facemoment.moment_detector.extractors.backends.insightface import InsightFaceSCRFD
    from facemoment.moment_detector.extractors.backends.hsemotion import HSEmotionBackend
    from facemoment.moment_detector.extractors.backends.pyfeat import PyFeatBackend
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
    # "InsightFaceSCRFD",     # from .insightface import InsightFaceSCRFD
    # "PyFeatBackend",        # from .pyfeat import PyFeatBackend
    # "HSEmotionBackend",     # from .hsemotion import HSEmotionBackend
    # "YOLOPoseBackend",      # from .pose_backends import YOLOPoseBackend
    # "MediaPipeHandsBackend",# from .hand_backends import MediaPipeHandsBackend
]


def __getattr__(name: str):
    """Lazy import for ML-dependent backends."""
    # Face detection backends
    if name == "InsightFaceSCRFD":
        from facemoment.moment_detector.extractors.backends.insightface import InsightFaceSCRFD
        return InsightFaceSCRFD
    # Expression backends
    elif name == "PyFeatBackend":
        from facemoment.moment_detector.extractors.backends.pyfeat import PyFeatBackend
        return PyFeatBackend
    elif name == "HSEmotionBackend":
        from facemoment.moment_detector.extractors.backends.hsemotion import HSEmotionBackend
        return HSEmotionBackend
    # Pose backends
    elif name == "YOLOPoseBackend":
        from facemoment.moment_detector.extractors.backends.pose_backends import YOLOPoseBackend
        return YOLOPoseBackend
    # Hand backends
    elif name == "MediaPipeHandsBackend":
        from facemoment.moment_detector.extractors.backends.hand_backends import MediaPipeHandsBackend
        return MediaPipeHandsBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
