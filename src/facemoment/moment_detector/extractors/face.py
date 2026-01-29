"""Face extractor using pluggable backends."""

from typing import Optional, Dict, List
import logging

import numpy as np

from visualbase import Frame

from facemoment.moment_detector.extractors.base import (
    BaseExtractor,
    Observation,
    FaceObservation,
)
from facemoment.moment_detector.extractors.backends.base import (
    FaceDetectionBackend,
    ExpressionBackend,
    DetectedFace,
)

logger = logging.getLogger(__name__)


class FaceExtractor(BaseExtractor):
    """Extractor for face detection and expression analysis.

    Uses pluggable backends for face detection and expression analysis,
    allowing different ML models to be swapped without changing the
    extraction logic.

    Features:
    - Face detection with bounding boxes, landmarks, and head pose
    - Expression analysis with Action Units and emotion classification
    - Simple face tracking using IoU-based ID assignment
    - Normalized coordinates for resolution-independent processing

    Args:
        face_backend: Face detection backend (default: InsightFaceSCRFD).
        expression_backend: Expression analysis backend (default: PyFeatBackend).
        device: Device for inference (default: "cuda:0").
        track_faces: Enable simple IoU-based face tracking (default: True).
        iou_threshold: IoU threshold for track matching (default: 0.5).

    Example:
        >>> extractor = FaceExtractor()
        >>> with extractor:
        ...     obs = extractor.extract(frame)
        ...     for face in obs.faces:
        ...         print(f"Face {face.face_id}: expression={face.expression:.2f}")
    """

    def __init__(
        self,
        face_backend: Optional[FaceDetectionBackend] = None,
        expression_backend: Optional[ExpressionBackend] = None,
        device: str = "cuda:0",
        track_faces: bool = True,
        iou_threshold: float = 0.5,
    ):
        self._device = device
        self._track_faces = track_faces
        self._iou_threshold = iou_threshold
        self._initialized = False

        # Lazy import backends to avoid import errors when dependencies missing
        self._face_backend = face_backend
        self._expression_backend = expression_backend

        # Tracking state
        self._next_face_id = 0
        self._prev_faces: List[tuple[int, tuple[int, int, int, int]]] = []  # (id, bbox)

    @property
    def name(self) -> str:
        return "face"

    def initialize(self) -> None:
        """Initialize face detection and expression backends."""
        if self._initialized:
            return  # Already initialized

        # Initialize expression backend if not provided
        # Priority: 1. HSEmotion (fast, ~30ms), 2. PyFeat (accurate, ~2000ms)
        if self._expression_backend is None:
            # Try HSEmotion first (fast)
            try:
                from facemoment.moment_detector.extractors.backends.face_backends import (
                    HSEmotionBackend,
                )

                self._expression_backend = HSEmotionBackend()
                self._expression_backend.initialize(self._device)
                logger.info("Using HSEmotionBackend for expression analysis (fast)")
            except ImportError:
                logger.debug("hsemotion-onnx not available, trying PyFeat")
            except Exception as e:
                logger.warning(f"Failed to initialize HSEmotion: {e}")

            # Fall back to PyFeat if HSEmotion failed
            # IMPORTANT: Initialize PyFeat BEFORE InsightFace - InsightFace modifies
            # ONNX runtime state that can break py-feat imports
            if self._expression_backend is None:
                try:
                    from facemoment.moment_detector.extractors.backends.face_backends import (
                        PyFeatBackend,
                    )

                    self._expression_backend = PyFeatBackend()
                    self._expression_backend.initialize(self._device)
                    logger.info("Using PyFeatBackend for expression analysis (accurate)")
                except ImportError:
                    logger.warning(
                        "No expression backend available. Expression analysis disabled. "
                        "Install with: uv sync --extra ml"
                    )
                    self._expression_backend = None
                except Exception as e:
                    logger.warning(f"Failed to initialize expression backend: {e}")
                    self._expression_backend = None

        # Now initialize face backend (InsightFace)
        if self._face_backend is None:
            from facemoment.moment_detector.extractors.backends.face_backends import (
                InsightFaceSCRFD,
            )

            self._face_backend = InsightFaceSCRFD()

        self._face_backend.initialize(self._device)

        self._initialized = True
        backend_name = type(self._expression_backend).__name__ if self._expression_backend else "disabled"
        logger.info(
            f"FaceExtractor initialized (expression={backend_name})"
        )

    def cleanup(self) -> None:
        """Release backend resources."""
        if self._face_backend is not None:
            self._face_backend.cleanup()
        if self._expression_backend is not None:
            self._expression_backend.cleanup()

        # Reset tracking state
        self._next_face_id = 0
        self._prev_faces = []

        logger.info("FaceExtractor cleaned up")

    def extract(self, frame: Frame) -> Optional[Observation]:
        """Extract face observations from a frame.

        Args:
            frame: Input frame to analyze.

        Returns:
            Observation with detected faces and their features.
        """
        if self._face_backend is None:
            raise RuntimeError("Extractor not initialized. Call initialize() first.")

        image = frame.data
        h, w = image.shape[:2]

        # Detect faces
        detected_faces = self._face_backend.detect(image)

        if not detected_faces:
            return Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={"face_count": 0, "max_expression": 0.0},
                faces=[],
            )

        # Analyze expressions
        expressions = []
        if self._expression_backend is not None:
            expressions = self._expression_backend.analyze(image, detected_faces)

        # Assign face IDs (simple IoU-based tracking)
        face_ids = self._assign_face_ids(detected_faces)

        # Convert to FaceObservations
        face_observations = []
        max_expression = 0.0

        for i, (face, face_id) in enumerate(zip(detected_faces, face_ids)):
            # Get expression if available
            expression = expressions[i] if i < len(expressions) else None

            # Calculate normalized bbox
            x, y, bw, bh = face.bbox
            norm_x = x / w
            norm_y = y / h
            norm_w = bw / w
            norm_h = bh / h

            # Calculate derived metrics
            area_ratio = norm_w * norm_h
            center_x = norm_x + norm_w / 2
            center_y = norm_y + norm_h / 2
            center_distance = ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5

            # Check if face is fully inside frame (with small margin)
            margin = 0.02
            inside_frame = (
                norm_x > margin
                and norm_y > margin
                and (norm_x + norm_w) < (1 - margin)
                and (norm_y + norm_h) < (1 - margin)
            )

            # Expression intensity
            expr_intensity = 0.0
            signals: Dict[str, float] = {}

            if expression is not None:
                expr_intensity = expression.expression_intensity
                signals["dominant_emotion"] = hash(expression.dominant_emotion) % 100 / 100
                for au_name, au_val in expression.action_units.items():
                    signals[au_name.lower()] = au_val
                for em_name, em_val in expression.emotions.items():
                    signals[f"em_{em_name}"] = em_val

            max_expression = max(max_expression, expr_intensity)

            face_obs = FaceObservation(
                face_id=face_id,
                confidence=face.confidence,
                bbox=(norm_x, norm_y, norm_w, norm_h),
                inside_frame=inside_frame,
                yaw=face.yaw,
                pitch=face.pitch,
                roll=face.roll,
                area_ratio=area_ratio,
                center_distance=center_distance,
                expression=expr_intensity,
                signals=signals,
            )
            face_observations.append(face_obs)

        # Update tracking state
        self._prev_faces = [(f.face_id, detected_faces[i].bbox) for i, f in enumerate(face_observations)]

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={
                "face_count": len(face_observations),
                "max_expression": max_expression,
            },
            faces=face_observations,
        )

    def _assign_face_ids(self, faces: List[DetectedFace]) -> List[int]:
        """Assign face IDs using simple IoU-based tracking.

        Args:
            faces: List of detected faces.

        Returns:
            List of face IDs corresponding to input faces.
        """
        if not self._track_faces or not self._prev_faces:
            # No tracking or first frame - assign new IDs
            ids = list(range(self._next_face_id, self._next_face_id + len(faces)))
            self._next_face_id += len(faces)
            return ids

        # Match current faces to previous faces using IoU
        assigned_ids = []
        used_prev_ids = set()

        for face in faces:
            best_id = None
            best_iou = self._iou_threshold

            for prev_id, prev_bbox in self._prev_faces:
                if prev_id in used_prev_ids:
                    continue

                iou = self._compute_iou(face.bbox, prev_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_id = prev_id

            if best_id is not None:
                assigned_ids.append(best_id)
                used_prev_ids.add(best_id)
            else:
                # New face - assign new ID
                assigned_ids.append(self._next_face_id)
                self._next_face_id += 1

        return assigned_ids

    @staticmethod
    def _compute_iou(
        box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]
    ) -> float:
        """Compute Intersection over Union between two boxes.

        Args:
            box1: First box (x, y, w, h).
            box2: Second box (x, y, w, h).

        Returns:
            IoU value [0, 1].
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Convert to x1, y1, x2, y2 format
        xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
        xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2

        # Compute intersection
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area
