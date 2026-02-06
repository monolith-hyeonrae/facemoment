"""Face detection extractor - detection only, no expression analysis."""

from typing import Optional, Dict, List
import logging
import time

from visualbase import Frame

from facemoment.moment_detector.extractors.base import (
    Module,
    Observation,
    FaceObservation,
    ProcessingStep,
    processing_step,
    get_processing_steps,
)
from facemoment.moment_detector.extractors.backends.base import (
    FaceDetectionBackend,
    DetectedFace,
)
from facemoment.moment_detector.extractors.outputs import FaceDetectOutput

logger = logging.getLogger(__name__)


class FaceDetectionExtractor(Module):
    """Extractor for face detection only.

    Detects faces and outputs bounding boxes, landmarks, and head pose.
    Does NOT analyze expressions - use ExpressionExtractor for that.

    Outputs:
        - signals: face_count
        - data: {"faces": List[FaceObservation]}

    Example:
        >>> extractor = FaceDetectionExtractor()
        >>> with extractor:
        ...     obs = extractor.process(frame)
        ...     print(f"Detected {obs.signals['face_count']} faces")
    """

    def __init__(
        self,
        face_backend: Optional[FaceDetectionBackend] = None,
        device: str = "cuda:0",
        track_faces: bool = True,
        iou_threshold: float = 0.5,
        roi: Optional[tuple[float, float, float, float]] = None,
    ):
        self._device = device
        self._track_faces = track_faces
        self._iou_threshold = iou_threshold
        self._roi = roi if roi is not None else (0.3, 0.1, 0.7, 0.6)
        self._initialized = False
        self._face_backend = face_backend

        # Tracking state
        self._next_face_id = 0
        self._prev_faces: List[tuple[int, tuple[int, int, int, int]]] = []

        # Step timing tracking (auto-populated by @processing_step decorator)
        self._step_timings: Optional[Dict[str, float]] = None

    @property
    def name(self) -> str:
        return "face_detect"

    @property
    def processing_steps(self) -> List[ProcessingStep]:
        """Get the list of internal processing steps (auto-extracted from decorators)."""
        return get_processing_steps(self)

    def initialize(self) -> None:
        if self._initialized:
            return

        if self._face_backend is None:
            from facemoment.moment_detector.extractors.backends.insightface import (
                InsightFaceSCRFD,
            )
            self._face_backend = InsightFaceSCRFD()

        self._face_backend.initialize(self._device)
        self._initialized = True
        logger.info("FaceDetectionExtractor initialized")

    def cleanup(self) -> None:
        if self._face_backend is not None:
            self._face_backend.cleanup()
        self._next_face_id = 0
        self._prev_faces = []
        logger.info("FaceDetectionExtractor cleaned up")

    # ========== Processing Steps (decorated methods) ==========

    @processing_step(
        name="detect",
        description="Face detection with landmarks and head pose",
        backend="InsightFace SCRFD",
        input_type="Frame (BGR image)",
        output_type="List[DetectedFace]",
    )
    def _detect_faces(self, image) -> List[DetectedFace]:
        """Detect faces using backend."""
        return self._face_backend.detect(image)

    @processing_step(
        name="tracking",
        description="Face ID assignment using IOU matching",
        backend="IOU-based",
        input_type="List[DetectedFace]",
        output_type="List[int] (face IDs)",
        depends_on=["detect"],
    )
    def _assign_face_ids(self, faces: List[DetectedFace]) -> List[int]:
        """Assign face IDs using simple IoU-based tracking."""
        if not self._track_faces or not self._prev_faces:
            ids = list(range(self._next_face_id, self._next_face_id + len(faces)))
            self._next_face_id += len(faces)
            return ids

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
                assigned_ids.append(self._next_face_id)
                self._next_face_id += 1

        return assigned_ids

    @processing_step(
        name="roi_filter",
        description="Filter faces outside region of interest",
        input_type="List[DetectedFace] + IDs",
        output_type="List[FaceObservation] (filtered)",
        depends_on=["tracking"],
    )
    def _filter_and_convert(
        self,
        detected_faces: List[DetectedFace],
        face_ids: List[int],
        image_size: tuple[int, int],
    ) -> tuple[List[FaceObservation], List[tuple[int, tuple]]]:
        """Filter by ROI and convert to FaceObservation."""
        w, h = image_size
        face_observations = []
        prev_faces_update = []

        for i, (face, face_id) in enumerate(zip(detected_faces, face_ids)):
            x, y, bw, bh = face.bbox
            norm_x = x / w
            norm_y = y / h
            norm_w = bw / w
            norm_h = bh / h

            # Derived metrics
            area_ratio = norm_w * norm_h
            center_x = norm_x + norm_w / 2
            center_y = norm_y + norm_h / 2
            center_distance = ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5

            # ROI filter
            roi_x1, roi_y1, roi_x2, roi_y2 = self._roi
            if not (roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2):
                continue

            # Inside frame check
            margin = 0.02
            inside_frame = (
                norm_x > margin
                and norm_y > margin
                and (norm_x + norm_w) < (1 - margin)
                and (norm_y + norm_h) < (1 - margin)
            )

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
                expression=0.0,
                signals={},
            )
            face_observations.append(face_obs)
            prev_faces_update.append((face_id, face.bbox))

        return face_observations, prev_faces_update

    # ========== Main process method ==========

    def process(
        self,
        frame: Frame,
        deps: Optional[Dict[str, Observation]] = None,
    ) -> Optional[Observation]:
        if self._face_backend is None:
            raise RuntimeError("Extractor not initialized")

        # Enable step timing collection
        self._step_timings = {}

        image = frame.data
        h, w = image.shape[:2]

        # Execute processing steps (timing auto-tracked by decorators)
        detected_faces = self._detect_faces(image)

        if not detected_faces:
            timing = self._step_timings.copy() if self._step_timings else None
            self._step_timings = None
            return Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={"face_count": 0},
                data=FaceDetectOutput(faces=[], detected_faces=[], image_size=(w, h)),
                timing=timing,
            )

        face_ids = self._assign_face_ids(detected_faces)
        face_observations, prev_faces_update = self._filter_and_convert(
            detected_faces, face_ids, (w, h)
        )

        # Update tracking state
        self._prev_faces = prev_faces_update

        # Collect timing data
        timing = self._step_timings.copy() if self._step_timings else None
        self._step_timings = None

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"face_count": len(face_observations)},
            data=FaceDetectOutput(
                faces=face_observations,
                detected_faces=detected_faces,
                image_size=(w, h),
            ),
            timing=timing,
        )

    @staticmethod
    def _compute_iou(
        box1: tuple[int, int, int, int],
        box2: tuple[int, int, int, int],
    ) -> float:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
        xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2

        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area
