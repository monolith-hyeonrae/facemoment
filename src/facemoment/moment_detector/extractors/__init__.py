"""Extractors for moment detection.

Each extractor can be imported independently to avoid dependency conflicts
when running in isolated worker environments.

Usage:
    # Import only what you need (lazy loading)
    from facemoment.moment_detector.extractors.face import FaceExtractor
    from facemoment.moment_detector.extractors.pose import PoseExtractor
    from facemoment.moment_detector.extractors.gesture import GestureExtractor

    # Or import base types (always available)
    from facemoment.moment_detector.extractors import BaseExtractor, Observation
"""

# Base types - always available (no ML dependencies)
from facemoment.moment_detector.extractors.base import (
    BaseExtractor,
    Observation,
    FaceObservation,
)
from facemoment.moment_detector.extractors.dummy import DummyExtractor
from facemoment.moment_detector.extractors.quality import QualityExtractor

__all__ = [
    # Base types (always available)
    "BaseExtractor",
    "Observation",
    "FaceObservation",
    "DummyExtractor",
    "QualityExtractor",
    # Lazy imports (import directly from submodule)
    # "FaceExtractor",      # from .face import FaceExtractor
    # "PoseExtractor",      # from .pose import PoseExtractor
    # "GestureExtractor",   # from .gesture import GestureExtractor
]


def __getattr__(name: str):
    """Lazy import for ML-dependent extractors."""
    if name == "FaceExtractor":
        from facemoment.moment_detector.extractors.face import FaceExtractor
        return FaceExtractor
    elif name == "PoseExtractor":
        from facemoment.moment_detector.extractors.pose import PoseExtractor
        return PoseExtractor
    elif name == "GestureExtractor":
        from facemoment.moment_detector.extractors.gesture import GestureExtractor
        return GestureExtractor
    elif name == "GestureType":
        from facemoment.moment_detector.extractors.gesture import GestureType
        return GestureType
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
