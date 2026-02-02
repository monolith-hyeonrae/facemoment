"""Base extractor interface for feature extraction (B modules).

This module re-exports the base classes from visualpath and provides
facemoment-specific observation types.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

# Re-export BaseExtractor from visualpath
from visualpath.core.extractor import BaseExtractor
from visualpath.core.isolation import IsolationLevel

# Re-export for backward compatibility
from visualbase import Frame  # noqa: F401


@dataclass
class FaceObservation:
    """Observation for a single detected face.

    Attributes:
        face_id: Unique identifier for tracking.
        confidence: Detection confidence [0, 1].
        bbox: Bounding box (x, y, width, height) normalized [0, 1].
        inside_frame: Whether face is fully inside frame.
        yaw: Head yaw angle in degrees.
        pitch: Head pitch angle in degrees.
        roll: Head roll angle in degrees.
        area_ratio: Face area as ratio of frame area.
        center_distance: Normalized distance from frame center.
        expression: Expression intensity [0, 1].
        signals: Additional per-face signals.
    """

    face_id: int
    confidence: float
    bbox: tuple[float, float, float, float]  # x, y, w, h normalized
    inside_frame: bool = True
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    area_ratio: float = 0.0
    center_distance: float = 0.0
    expression: float = 0.0
    signals: Dict[str, float] = field(default_factory=dict)


@dataclass
class Observation:
    """Observation output from an extractor.

    Observations are timestamped feature extractions that flow from
    B modules (extractors) to C module (fusion).

    This is a facemoment-specific Observation that includes face-specific
    fields. It is compatible with visualpath's generic Observation interface.

    Attributes:
        source: Name of the extractor that produced this observation.
        frame_id: Frame identifier from the source video.
        t_ns: Timestamp in nanoseconds (source timeline).
        signals: Dictionary of extracted signals/features.
        data: Type-safe output data (e.g., PoseOutput, FaceClassifierOutput).
        faces: Optional list of face observations.
        metadata: Additional metadata about the observation.
        timing: Optional per-component timing in milliseconds.
    """

    source: str
    frame_id: int
    t_ns: int
    signals: Dict[str, float] = field(default_factory=dict)
    data: Optional[Any] = None  # Type-safe output (PoseOutput, FaceDetectOutput, etc.)
    faces: List[FaceObservation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timing: Optional[Dict[str, float]] = None  # {"detect_ms": 42.3, "expression_ms": 28.1}


__all__ = [
    "BaseExtractor",
    "Observation",
    "FaceObservation",
    "IsolationLevel",
]
