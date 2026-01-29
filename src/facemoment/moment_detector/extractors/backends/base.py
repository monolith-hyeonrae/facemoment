"""Backend protocol definitions for swappable ML backends."""

from dataclasses import dataclass, field
from typing import Protocol, List, Dict, Any, Optional
import numpy as np


@dataclass
class DetectedFace:
    """Result from face detection backend.

    Attributes:
        bbox: Bounding box (x, y, width, height) in pixels.
        confidence: Detection confidence [0, 1].
        landmarks: Optional facial landmarks (5-point or 68-point).
        yaw: Head yaw angle in degrees (if available).
        pitch: Head pitch angle in degrees (if available).
        roll: Head roll angle in degrees (if available).
    """

    bbox: tuple[int, int, int, int]  # x, y, w, h in pixels
    confidence: float
    landmarks: Optional[np.ndarray] = None
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass
class FaceExpression:
    """Result from expression analysis backend.

    Attributes:
        action_units: Dictionary of Action Unit activations {AU_name: intensity}.
        emotions: Dictionary of emotion probabilities {emotion: probability}.
        expression_intensity: Overall expression intensity [0, 1].
        dominant_emotion: Most likely emotion label.
    """

    action_units: Dict[str, float] = field(default_factory=dict)
    emotions: Dict[str, float] = field(default_factory=dict)
    expression_intensity: float = 0.0
    dominant_emotion: str = "neutral"


@dataclass
class PoseKeypoints:
    """Result from pose estimation backend.

    Attributes:
        keypoints: Array of shape (N, 3) with (x, y, confidence) per keypoint.
        keypoint_names: Names of keypoints in order.
        person_id: Optional person/track ID.
        bbox: Optional person bounding box.
        confidence: Overall detection confidence.
    """

    keypoints: np.ndarray  # Shape: (N, 3) - x, y, conf
    keypoint_names: List[str]
    person_id: Optional[int] = None
    bbox: Optional[tuple[int, int, int, int]] = None
    confidence: float = 1.0


class FaceDetectionBackend(Protocol):
    """Protocol for face detection backends.

    Implementations should be swappable without changing extractor logic.
    Examples: InsightFace SCRFD, YOLOv11-Face, RetinaFace.
    """

    def initialize(self, device: str = "cuda:0") -> None:
        """Initialize the backend and load models.

        Args:
            device: Device to use (e.g., "cuda:0", "cpu").
        """
        ...

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces in an image.

        Args:
            image: BGR image as numpy array (H, W, 3).

        Returns:
            List of detected faces with bounding boxes and optional landmarks.
        """
        ...

    def cleanup(self) -> None:
        """Release resources and unload models."""
        ...


class ExpressionBackend(Protocol):
    """Protocol for expression/AU analysis backends.

    Implementations analyze facial expressions from detected face regions.
    Examples: Py-Feat, FER, EmotionNet.
    """

    def initialize(self, device: str = "cuda:0") -> None:
        """Initialize the backend and load models.

        Args:
            device: Device to use (e.g., "cuda:0", "cpu").
        """
        ...

    def analyze(
        self, image: np.ndarray, faces: List[DetectedFace]
    ) -> List[FaceExpression]:
        """Analyze expressions for detected faces.

        Args:
            image: BGR image as numpy array (H, W, 3).
            faces: List of detected faces to analyze.

        Returns:
            List of expression results corresponding to input faces.
        """
        ...

    def cleanup(self) -> None:
        """Release resources and unload models."""
        ...


class PoseBackend(Protocol):
    """Protocol for pose estimation backends.

    Implementations extract body keypoints for gesture analysis.
    Examples: YOLOv8-Pose, MediaPipe, OpenPose.
    """

    def initialize(self, device: str = "cuda:0") -> None:
        """Initialize the backend and load models.

        Args:
            device: Device to use (e.g., "cuda:0", "cpu").
        """
        ...

    def detect(self, image: np.ndarray) -> List[PoseKeypoints]:
        """Detect poses in an image.

        Args:
            image: BGR image as numpy array (H, W, 3).

        Returns:
            List of detected poses with keypoints.
        """
        ...

    def cleanup(self) -> None:
        """Release resources and unload models."""
        ...
