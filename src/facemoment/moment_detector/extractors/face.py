"""Face extractor using pluggable backends."""

from typing import Optional, Dict, List
import logging
import time

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
from facemoment.observability import ObservabilityHub, TraceLevel
from facemoment.observability.records import FrameExtractRecord, FaceExtractDetail, TimingRecord

logger = logging.getLogger(__name__)

# Get the global observability hub
_hub = ObservabilityHub.get_instance()


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
        roi: Optional[tuple[float, float, float, float]] = None,
    ):
        """Initialize FaceExtractor.

        Args:
            face_backend: Face detection backend.
            expression_backend: Expression analysis backend.
            device: Device for inference.
            track_faces: Enable face tracking.
            iou_threshold: IoU threshold for tracking.
            roi: Region of interest as (x1, y1, x2, y2) in normalized coords [0-1].
                 Default (0.3, 0.1, 0.7, 0.6) = center 40% width, top 50% height.
                 Set to (0, 0, 1, 1) for full frame.
        """
        self._device = device
        self._track_faces = track_faces
        self._iou_threshold = iou_threshold
        # ROI default: wider to include passengers on both sides
        # Old: (0.3, 0.1, 0.7, 0.6) - too narrow for two-person scenarios
        self._roi = roi if roi is not None else (0.1, 0.05, 0.9, 0.75)
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

    @property
    def roi(self) -> tuple[float, float, float, float]:
        """Get current ROI as (x1, y1, x2, y2) in normalized coords [0-1]."""
        return self._roi

    @roi.setter
    def roi(self, value: tuple[float, float, float, float]) -> None:
        """Set ROI as (x1, y1, x2, y2) in normalized coords [0-1]."""
        x1, y1, x2, y2 = value
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"Invalid ROI: {value}. Must be (x1, y1, x2, y2) with 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
        self._roi = value
        logger.info(f"FaceExtractor ROI set to {value}")

    def initialize(self) -> None:
        """Initialize face detection and expression backends."""
        if self._initialized:
            return  # Already initialized

        # Initialize expression backend if not provided
        # Priority: 1. HSEmotion (fast, ~30ms), 2. PyFeat (accurate, ~2000ms)
        if self._expression_backend is None:
            # Try HSEmotion first (fast)
            try:
                from facemoment.moment_detector.extractors.backends.hsemotion import (
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
                    from facemoment.moment_detector.extractors.backends.pyfeat import (
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
            from facemoment.moment_detector.extractors.backends.insightface import (
                InsightFaceSCRFD,
            )

            self._face_backend = InsightFaceSCRFD()

        self._face_backend.initialize(self._device)

        self._initialized = True
        backend_name = type(self._expression_backend).__name__ if self._expression_backend else "disabled"
        roi_pct = (
            f"{int(self._roi[0]*100)}%-{int(self._roi[2]*100)}% x "
            f"{int(self._roi[1]*100)}%-{int(self._roi[3]*100)}%"
        )
        logger.info(
            f"FaceExtractor initialized (expression={backend_name}, ROI={roi_pct})"
        )

    def get_backend_info(self) -> Dict[str, str]:
        """Get information about current backends for profiling.

        Returns:
            Dict with backend names and device info.
        """
        info = {}
        if self._face_backend is not None:
            backend_name = type(self._face_backend).__name__
            # Get actual provider info if available
            if hasattr(self._face_backend, 'get_provider_info'):
                provider = self._face_backend.get_provider_info()
                info["detection"] = f"{backend_name} [{provider}]"
            else:
                info["detection"] = f"{backend_name} ({self._device})"
        else:
            info["detection"] = "not initialized"

        if self._expression_backend is not None:
            backend_name = type(self._expression_backend).__name__
            info["expression"] = backend_name
        else:
            info["expression"] = "disabled"

        return info

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

    def extract(
        self,
        frame: Frame,
        deps: Optional[Dict[str, "Observation"]] = None,
    ) -> Optional[Observation]:
        """Extract face observations from a frame.

        Args:
            frame: Input frame to analyze.
            deps: Optional dependencies (not used by this composite extractor).

        Returns:
            Observation with detected faces and their features.
        """
        if self._face_backend is None:
            raise RuntimeError("Extractor not initialized. Call initialize() first.")

        # Start timing (always measure for profile mode)
        start_ns = time.perf_counter_ns()

        image = frame.data
        h, w = image.shape[:2]

        # Component timing
        timing = {}

        # Detect faces (with timing)
        detect_start = time.perf_counter_ns()
        detected_faces = self._face_backend.detect(image)
        timing["detect_ms"] = (time.perf_counter_ns() - detect_start) / 1_000_000

        if not detected_faces:
            # Emit timing record
            if _hub.enabled:
                processing_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
                self._emit_extract_record(frame, 0, 0.0, processing_ms, {})
            return Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={
                    "face_count": 0,
                    "max_expression": 0.0,
                    "expression_happy": 0.0,
                    "expression_angry": 0.0,
                    "expression_neutral": 1.0,
                },
                faces=[],
                timing=timing,
            )

        # Analyze expressions (with timing)
        expressions = []
        if self._expression_backend is not None:
            expr_start = time.perf_counter_ns()
            expressions = self._expression_backend.analyze(image, detected_faces)
            timing["expression_ms"] = (time.perf_counter_ns() - expr_start) / 1_000_000

        # Assign face IDs (simple IoU-based tracking)
        face_ids = self._assign_face_ids(detected_faces)

        # Convert to FaceObservations
        face_observations = []
        prev_faces_update = []  # Track (face_id, original_bbox) for IOU tracking
        max_expression = 0.0
        max_happy = 0.0
        max_angry = 0.0
        min_neutral = 1.0  # Track minimum neutral (more interesting = less neutral)

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

            # ROI filter: skip faces with center outside ROI
            roi_x1, roi_y1, roi_x2, roi_y2 = self._roi
            if not (roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2):
                logger.debug(
                    f"Face {i} filtered by ROI: center=({center_x:.2f}, {center_y:.2f}), "
                    f"ROI=({roi_x1:.2f}-{roi_x2:.2f}, {roi_y1:.2f}-{roi_y2:.2f})"
                )
                continue

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
            face_happy = 0.0
            face_angry = 0.0
            face_neutral = 1.0
            signals: Dict[str, float] = {}

            if expression is not None:
                expr_intensity = expression.expression_intensity
                signals["dominant_emotion"] = hash(expression.dominant_emotion) % 100 / 100
                for au_name, au_val in expression.action_units.items():
                    signals[au_name.lower()] = au_val
                for em_name, em_val in expression.emotions.items():
                    signals[f"em_{em_name}"] = em_val

                # Track individual emotions
                face_happy = expression.emotions.get("happy", 0.0)
                face_angry = expression.emotions.get("angry", 0.0)
                face_neutral = expression.emotions.get("neutral", 1.0)

            max_expression = max(max_expression, expr_intensity)
            max_happy = max(max_happy, face_happy)
            max_angry = max(max_angry, face_angry)
            min_neutral = min(min_neutral, face_neutral)

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
            # Store original bbox for IOU tracking (not filtered index)
            prev_faces_update.append((face_id, face.bbox))

        # Update tracking state with correct face_id -> bbox mapping
        self._prev_faces = prev_faces_update

        # Emit observability records
        if _hub.enabled:
            processing_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            self._emit_extract_record(
                frame,
                len(face_observations),
                max_expression,
                processing_ms,
                {"face_count": len(face_observations), "max_expression": max_expression},
            )
            # Emit detailed per-face records at VERBOSE level
            if _hub.is_level_enabled(TraceLevel.VERBOSE):
                for face_obs in face_observations:
                    _hub.emit(FaceExtractDetail(
                        frame_id=frame.frame_id,
                        face_id=face_obs.face_id,
                        confidence=face_obs.confidence,
                        bbox=face_obs.bbox,
                        yaw=face_obs.yaw,
                        pitch=face_obs.pitch,
                        roll=face_obs.roll,
                        expression=face_obs.expression,
                        inside_frame=face_obs.inside_frame,
                        area_ratio=face_obs.area_ratio,
                        center_distance=face_obs.center_distance,
                    ))

        # Record total timing
        timing["total_ms"] = (time.perf_counter_ns() - start_ns) / 1_000_000

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={
                "face_count": len(face_observations),
                "max_expression": max_expression,
                "expression_happy": max_happy,
                "expression_angry": max_angry,
                "expression_neutral": min_neutral,
            },
            faces=face_observations,
            timing=timing,
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

    def _emit_extract_record(
        self,
        frame: Frame,
        face_count: int,
        max_expression: float,
        processing_ms: float,
        signals: Dict[str, float],
    ) -> None:
        """Emit extraction observability records.

        Args:
            frame: The processed frame.
            face_count: Number of faces detected.
            max_expression: Maximum expression value.
            processing_ms: Processing time in milliseconds.
            signals: Signal dictionary.
        """
        threshold_ms = 50.0
        _hub.emit(FrameExtractRecord(
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            source=self.name,
            face_count=face_count,
            processing_ms=processing_ms,
            signals=signals if _hub.is_level_enabled(TraceLevel.VERBOSE) else {},
        ))
        _hub.emit(TimingRecord(
            frame_id=frame.frame_id,
            component=self.name,
            processing_ms=processing_ms,
            threshold_ms=threshold_ms,
            is_slow=processing_ms > threshold_ms,
        ))
