"""Expression analysis extractor - depends on face_detect."""

from typing import Optional, Dict, List
import logging
import time

from visualbase import Frame

from facemoment.moment_detector.extractors.base import (
    Module,
    Observation,
    FaceObservation,
)
from facemoment.moment_detector.extractors.backends.base import ExpressionBackend
from facemoment.moment_detector.extractors.outputs import FaceDetectOutput, ExpressionOutput

logger = logging.getLogger(__name__)


class ExpressionExtractor(Module):
    """Extractor for expression analysis.

    Depends on face_detect extractor for face bounding boxes.
    Analyzes emotions and action units for detected faces.

    depends: ["face_detect"]

    Outputs:
        - signals: max_expression, expression_happy, expression_angry, expression_neutral
        - data: {"faces": List[FaceObservation with expression data]}

    Example:
        >>> # Use with FlowGraph for automatic dependency resolution
        >>> graph = (FlowGraphBuilder()
        ...     .source()
        ...     .path("detect", modules=[FaceDetectionExtractor()])
        ...     .path("expr", modules=[ExpressionExtractor()])
        ...     .build())
    """

    # Dependency declaration
    depends = ["face_detect"]

    def __init__(
        self,
        expression_backend: Optional[ExpressionBackend] = None,
        device: str = "cuda:0",
    ):
        self._device = device
        self._expression_backend = expression_backend
        self._initialized = False

    @property
    def name(self) -> str:
        return "expression"

    def initialize(self) -> None:
        if self._initialized:
            return

        if self._expression_backend is None:
            # Try HSEmotion first (fast)
            try:
                from facemoment.moment_detector.extractors.backends.hsemotion import (
                    HSEmotionBackend,
                )
                self._expression_backend = HSEmotionBackend()
                self._expression_backend.initialize(self._device)
                logger.info("ExpressionExtractor using HSEmotionBackend")
            except ImportError:
                logger.debug("hsemotion-onnx not available, trying PyFeat")
            except Exception as e:
                logger.warning(f"Failed to initialize HSEmotion: {e}")

            # Fall back to PyFeat
            if self._expression_backend is None:
                try:
                    from facemoment.moment_detector.extractors.backends.pyfeat import (
                        PyFeatBackend,
                    )
                    self._expression_backend = PyFeatBackend()
                    self._expression_backend.initialize(self._device)
                    logger.info("ExpressionExtractor using PyFeatBackend")
                except ImportError:
                    logger.warning("No expression backend available")
                except Exception as e:
                    logger.warning(f"Failed to initialize PyFeat: {e}")

        self._initialized = True

    def cleanup(self) -> None:
        if self._expression_backend is not None:
            self._expression_backend.cleanup()
        logger.info("ExpressionExtractor cleaned up")

    def process(
        self,
        frame: Frame,
        deps: Optional[Dict[str, Observation]] = None,
    ) -> Optional[Observation]:
        # Get face_detect dependency (type-safe access)
        face_obs = deps.get("face_detect") if deps else None
        if face_obs is None:
            logger.warning("ExpressionExtractor: no face_detect dependency")
            return None

        face_data: FaceDetectOutput = face_obs.data
        if not face_data:
            return None

        # Type-safe attribute access
        detected_faces = face_data.detected_faces
        face_observations = face_data.faces

        if not detected_faces or not face_observations:
            return Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={
                    "max_expression": 0.0,
                    "expression_happy": 0.0,
                    "expression_angry": 0.0,
                    "expression_neutral": 1.0,
                },
                data=ExpressionOutput(faces=[]),
            )

        start_ns = time.perf_counter_ns()
        timing = {}

        # Analyze expressions
        expressions = []
        if self._expression_backend is not None:
            expr_start = time.perf_counter_ns()
            expressions = self._expression_backend.analyze(frame.data, detected_faces)
            timing["expression_ms"] = (time.perf_counter_ns() - expr_start) / 1_000_000

        # Update face observations with expression data
        updated_faces = []
        max_expression = 0.0
        max_happy = 0.0
        max_angry = 0.0
        min_neutral = 1.0

        for i, face_obs in enumerate(face_observations):
            expression = expressions[i] if i < len(expressions) else None

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

                face_happy = expression.emotions.get("happy", 0.0)
                face_angry = expression.emotions.get("angry", 0.0)
                face_neutral = expression.emotions.get("neutral", 1.0)

            max_expression = max(max_expression, expr_intensity)
            max_happy = max(max_happy, face_happy)
            max_angry = max(max_angry, face_angry)
            min_neutral = min(min_neutral, face_neutral)

            # Create updated FaceObservation with expression
            updated_face = FaceObservation(
                face_id=face_obs.face_id,
                confidence=face_obs.confidence,
                bbox=face_obs.bbox,
                inside_frame=face_obs.inside_frame,
                yaw=face_obs.yaw,
                pitch=face_obs.pitch,
                roll=face_obs.roll,
                area_ratio=face_obs.area_ratio,
                center_distance=face_obs.center_distance,
                expression=expr_intensity,
                signals=signals,
            )
            updated_faces.append(updated_face)

        timing["total_ms"] = (time.perf_counter_ns() - start_ns) / 1_000_000

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={
                "max_expression": max_expression,
                "expression_happy": max_happy,
                "expression_angry": max_angry,
                "expression_neutral": min_neutral,
            },
            data=ExpressionOutput(faces=updated_faces),
            timing=timing,
        )
