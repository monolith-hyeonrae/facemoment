"""Type-safe output definitions for extractors.

Each extractor defines its output structure as a dataclass,
enabling type checking and IDE autocomplete for deps access.

Example:
    >>> obs: Observation[FaceDetectOutput] = face_detect.extract(frame)
    >>> faces = obs.data.faces  # IDE knows this is List[FaceObservation]
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from facemoment.moment_detector.extractors.base import FaceObservation
from facemoment.moment_detector.extractors.backends.base import DetectedFace


@dataclass
class FaceDetectOutput:
    """Output from FaceDetectionExtractor.

    Attributes:
        faces: List of detected faces with normalized bboxes and head pose.
        detected_faces: Raw DetectedFace objects (for expression analysis).
        image_size: Original image size as (width, height).
    """
    faces: List[FaceObservation] = field(default_factory=list)
    detected_faces: List[DetectedFace] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)


@dataclass
class ExpressionOutput:
    """Output from ExpressionExtractor.

    Attributes:
        faces: List of faces with expression data added.
    """
    faces: List[FaceObservation] = field(default_factory=list)


@dataclass
class PoseOutput:
    """Output from PoseExtractor.

    Attributes:
        keypoints: List of keypoint dicts per person.
        person_count: Number of detected persons.
    """
    keypoints: List[Dict[str, Any]] = field(default_factory=list)
    person_count: int = 0


@dataclass
class GestureOutput:
    """Output from GestureExtractor.

    Attributes:
        gestures: List of detected gestures.
        hand_landmarks: Raw hand landmark data.
    """
    gestures: List[str] = field(default_factory=list)
    hand_landmarks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class QualityOutput:
    """Output from QualityExtractor.

    Attributes:
        blur_score: Laplacian variance (higher = sharper).
        brightness: Mean brightness [0-255].
        contrast: Standard deviation of brightness.
    """
    blur_score: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0


@dataclass
class ClassifiedFaceInfo:
    """Lightweight face classification info for output typing."""
    face_id: int
    role: str  # "main", "passenger", "transient", "noise"
    confidence: float
    track_length: int


@dataclass
class FaceClassifierOutput:
    """Output from FaceClassifierExtractor.

    Attributes:
        main_face_id: ID of the main subject face (or None).
        passenger_face_ids: IDs of passenger faces.
        classifications: Dict mapping face_id to classification info.
        transient_count: Number of transient detections.
        noise_count: Number of noise detections.
    """
    main_face_id: Optional[int] = None
    passenger_face_ids: List[int] = field(default_factory=list)
    classifications: Dict[int, ClassifiedFaceInfo] = field(default_factory=dict)
    transient_count: int = 0
    noise_count: int = 0


# Type alias for convenient access
__all__ = [
    "FaceDetectOutput",
    "ExpressionOutput",
    "PoseOutput",
    "GestureOutput",
    "QualityOutput",
    "ClassifiedFaceInfo",
    "FaceClassifierOutput",
]
