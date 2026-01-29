"""Base extractor interface for feature extraction (B modules)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from visualbase import Frame


@dataclass
class Observation:
    """Observation output from an extractor.

    Observations are timestamped feature extractions that flow from
    B modules (extractors) to C module (fusion).

    Attributes:
        source: Name of the extractor that produced this observation.
        frame_id: Frame identifier from the source video.
        t_ns: Timestamp in nanoseconds (source timeline).
        signals: Dictionary of extracted signals/features.
        faces: Optional list of face observations.
    """

    source: str
    frame_id: int
    t_ns: int
    signals: Dict[str, float] = field(default_factory=dict)
    faces: List["FaceObservation"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


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


class BaseExtractor(ABC):
    """Abstract base class for feature extractors (B modules).

    Extractors analyze frames and produce observations containing
    extracted features. Multiple extractors can run in parallel,
    each focusing on different aspects (face, gesture, quality, etc.).

    Example:
        >>> class FaceExtractor(BaseExtractor):
        ...     @property
        ...     def name(self) -> str:
        ...         return "face"
        ...
        ...     def extract(self, frame: Frame) -> Optional[Observation]:
        ...         # Detect faces and extract features
        ...         faces = self._detect_faces(frame.data)
        ...         return Observation(
        ...             source=self.name,
        ...             frame_id=frame.frame_id,
        ...             t_ns=frame.t_src_ns,
        ...             faces=faces,
        ...         )
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this extractor."""
        ...

    @abstractmethod
    def extract(self, frame: Frame) -> Optional[Observation]:
        """Extract features from a frame.

        Args:
            frame: Input frame to analyze.

        Returns:
            Observation containing extracted features, or None if
            no meaningful observation could be made.
        """
        ...

    def initialize(self) -> None:
        """Initialize extractor resources (models, etc.).

        Override this method to load models or initialize resources.
        Called once before processing begins.
        """
        pass

    def cleanup(self) -> None:
        """Clean up extractor resources.

        Override this method to release resources.
        Called when processing ends.
        """
        pass

    def __enter__(self) -> "BaseExtractor":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()
