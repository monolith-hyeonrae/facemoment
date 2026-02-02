"""InsightFace SCRFD backend for face detection."""

from typing import List, Optional
import logging

import numpy as np

from facemoment.moment_detector.extractors.backends.base import DetectedFace

logger = logging.getLogger(__name__)


class InsightFaceSCRFD:
    """Face detection backend using InsightFace SCRFD.

    SCRFD (Sample and Computation Redistribution for Efficient Face Detection)
    provides fast and accurate face detection with optional landmark detection.

    Args:
        model_name: Model variant (default: "buffalo_l" for best accuracy).
        det_size: Detection input size (width, height).
        det_thresh: Detection confidence threshold.

    Example:
        >>> backend = InsightFaceSCRFD()
        >>> backend.initialize("cuda:0")
        >>> faces = backend.detect(image)
        >>> backend.cleanup()
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
    ):
        self._model_name = model_name
        self._det_size = det_size
        self._det_thresh = det_thresh
        self._app: Optional[object] = None
        self._initialized = False
        self._device = "cpu"
        self._actual_provider = "unknown"

    def initialize(self, device: str = "cuda:0") -> None:
        """Initialize InsightFace app with SCRFD detector."""
        if self._initialized:
            return  # Already initialized

        try:
            from insightface.app import FaceAnalysis
            import onnxruntime as ort

            # Check available ONNX providers
            available_providers = ort.get_available_providers()
            logger.info(f"Available ONNX providers: {available_providers}")

            # Parse device (insightface uses ctx_id: 0 for GPU, -1 for CPU)
            if device.startswith("cuda"):
                ctx_id = int(device.split(":")[-1]) if ":" in device else 0
                if "CUDAExecutionProvider" in available_providers:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    self._actual_provider = "CUDA"
                else:
                    providers = ["CPUExecutionProvider"]
                    self._actual_provider = "CPU (CUDA unavailable)"
                    logger.warning("CUDAExecutionProvider not available, falling back to CPU")
            else:
                ctx_id = -1
                providers = ["CPUExecutionProvider"]
                self._actual_provider = "CPU"

            self._device = device

            self._app = FaceAnalysis(
                name=self._model_name,
                providers=providers,
            )
            self._app.prepare(ctx_id=ctx_id, det_size=self._det_size)
            self._initialized = True
            logger.info(f"InsightFace SCRFD initialized (provider={self._actual_provider})")

        except ImportError:
            raise ImportError(
                "insightface is required for InsightFaceSCRFD backend. "
                "Install with: pip install insightface"
            )
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {e}")
            raise

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces using SCRFD.

        Args:
            image: BGR image as numpy array (H, W, 3).

        Returns:
            List of detected faces with bounding boxes, landmarks, and pose.
        """
        if not self._initialized or self._app is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        faces = self._app.get(image)
        results = []

        for face in faces:
            if face.det_score < self._det_thresh:
                continue

            # Extract bounding box (x1, y1, x2, y2 -> x, y, w, h)
            bbox = face.bbox.astype(int)
            x, y = int(bbox[0]), int(bbox[1])
            w, h = int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])

            # Extract pose angles if available
            yaw, pitch, roll = 0.0, 0.0, 0.0
            if hasattr(face, "pose") and face.pose is not None:
                pose = face.pose
                yaw = float(pose[1]) if len(pose) > 1 else 0.0
                pitch = float(pose[0]) if len(pose) > 0 else 0.0
                roll = float(pose[2]) if len(pose) > 2 else 0.0

            # Extract landmarks (5-point kps)
            landmarks = None
            if hasattr(face, "kps") and face.kps is not None:
                landmarks = face.kps.astype(np.float32)

            results.append(
                DetectedFace(
                    bbox=(x, y, w, h),
                    confidence=float(face.det_score),
                    landmarks=landmarks,
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                )
            )

        return results

    def cleanup(self) -> None:
        """Release InsightFace resources."""
        self._app = None
        self._initialized = False
        logger.info("InsightFace SCRFD cleaned up")

    def get_provider_info(self) -> str:
        """Get actual provider being used."""
        return self._actual_provider
