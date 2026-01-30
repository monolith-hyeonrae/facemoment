"""Face detection and expression analysis backend implementations."""

from typing import List, Optional
import logging

import numpy as np

from facemoment.moment_detector.extractors.backends.base import (
    DetectedFace,
    FaceExpression,
)

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


class PyFeatBackend:
    """Expression analysis backend using Py-Feat.

    Py-Feat provides Action Unit detection and emotion classification.
    Uses a pre-trained model for facial expression analysis.

    Args:
        au_model: AU detection model (default: "xgb" for XGBoost).
        emotion_model: Emotion classification model (default: "resmasknet").

    Example:
        >>> backend = PyFeatBackend()
        >>> backend.initialize("cuda:0")
        >>> expressions = backend.analyze(image, faces)
        >>> backend.cleanup()
    """

    # Mapping from Py-Feat emotion names to our standard names
    EMOTION_MAP = {
        "anger": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "happiness": "happy",
        "sadness": "sad",
        "surprise": "surprise",
        "neutral": "neutral",
    }

    def __init__(
        self,
        au_model: str = "xgb",
        emotion_model: str = "resmasknet",
    ):
        self._au_model = au_model
        self._emotion_model = emotion_model
        self._detector: Optional[object] = None
        self._initialized = False

    def initialize(self, device: str = "cuda:0") -> None:
        """Initialize Py-Feat detector."""
        if self._initialized:
            return  # Already initialized

        try:
            from feat import Detector

            # Py-Feat uses device string directly
            self._detector = Detector(
                au_model=self._au_model,
                emotion_model=self._emotion_model,
                device="cuda" if device.startswith("cuda") else "cpu",
            )
            self._initialized = True
            logger.info(f"Py-Feat backend initialized on device {device}")

        except ImportError:
            raise ImportError(
                "py-feat is required for PyFeatBackend. "
                "Install with: pip install py-feat"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Py-Feat: {e}")
            raise

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
        if not self._initialized or self._detector is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        if not faces:
            return []

        import cv2

        results = []

        for face in faces:
            x, y, w, h = face.bbox

            # Expand bbox slightly for better AU detection
            pad = int(max(w, h) * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(image.shape[1], x + w + pad)
            y2 = min(image.shape[0], y + h + pad)

            # Crop face region
            face_img = image[y1:y2, x1:x2]

            if face_img.size == 0:
                results.append(FaceExpression())
                continue

            # Convert BGR to RGB for Py-Feat
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            try:
                # Use individual detection methods for better performance
                # detect_faces returns bounding boxes
                detected_faces = self._detector.detect_faces(face_rgb)

                if detected_faces is None or len(detected_faces) == 0:
                    results.append(FaceExpression())
                    continue

                # detect_landmarks returns facial landmarks
                detected_landmarks = self._detector.detect_landmarks(
                    face_rgb, detected_faces
                )

                # detect_aus returns Action Units
                aus_result = self._detector.detect_aus(face_rgb, detected_landmarks)

                # detect_emotions returns emotion predictions
                # Returns: [anger, disgust, fear, happiness, sadness, surprise, neutral]
                emotions_result = self._detector.detect_emotions(
                    face_rgb, detected_faces, detected_landmarks
                )

                # Extract Action Units
                aus = {}
                if aus_result is not None and len(aus_result) > 0:
                    au_names = [
                        "AU01",
                        "AU02",
                        "AU04",
                        "AU05",
                        "AU06",
                        "AU07",
                        "AU09",
                        "AU10",
                        "AU11",
                        "AU12",
                        "AU14",
                        "AU15",
                        "AU17",
                        "AU20",
                        "AU23",
                        "AU24",
                        "AU25",
                        "AU26",
                        "AU28",
                        "AU43",
                    ]
                    for i, au_val in enumerate(aus_result[0][0]):
                        if i < len(au_names):
                            aus[au_names[i]] = float(au_val)

                # Extract emotions
                emotions = {}
                emotion_names = [
                    "angry",
                    "disgust",
                    "fear",
                    "happy",
                    "sad",
                    "surprise",
                    "neutral",
                ]
                max_emotion = "neutral"
                max_prob = 0.0

                if emotions_result is not None and len(emotions_result) > 0:
                    for i, em_val in enumerate(emotions_result[0][0]):
                        if i < len(emotion_names):
                            em_name = emotion_names[i]
                            emotions[em_name] = float(em_val)
                            if float(em_val) > max_prob:
                                max_prob = float(em_val)
                                max_emotion = em_name

                # Calculate overall expression intensity
                smile_au = aus.get("AU12", 0.0)
                cheek_au = aus.get("AU06", 0.0)
                happy_prob = emotions.get("happy", 0.0)
                surprise_prob = emotions.get("surprise", 0.0)

                expression_intensity = max(
                    (smile_au + cheek_au) / 2,
                    happy_prob,
                    surprise_prob * 0.8,
                )

                results.append(
                    FaceExpression(
                        action_units=aus,
                        emotions=emotions,
                        expression_intensity=min(1.0, expression_intensity),
                        dominant_emotion=max_emotion,
                    )
                )

            except Exception as e:
                logger.warning(f"Failed to analyze face expression: {e}")
                results.append(FaceExpression())

        return results

    def cleanup(self) -> None:
        """Release Py-Feat resources."""
        self._detector = None
        self._initialized = False
        logger.info("Py-Feat backend cleaned up")


class HSEmotionBackend:
    """Expression analysis backend using HSEmotion-ONNX.

    HSEmotion provides fast emotion classification using EfficientNet-B0.
    This is significantly faster than Py-Feat (~30ms vs ~2000ms per frame).

    Supports 8 emotions: Anger, Contempt, Disgust, Fear, Happiness,
    Neutral, Sadness, Surprise.

    Note: This backend does NOT support Action Units (AU) - only emotion
    classification and expression intensity.

    Args:
        model_name: HSEmotion model variant (default: "enet_b0_8_best_vgaf").

    Example:
        >>> backend = HSEmotionBackend()
        >>> backend.initialize("cuda:0")
        >>> expressions = backend.analyze(image, faces)
        >>> backend.cleanup()
    """

    # HSEmotion emotion labels (8 classes)
    EMOTIONS = [
        "Anger",
        "Contempt",
        "Disgust",
        "Fear",
        "Happiness",
        "Neutral",
        "Sadness",
        "Surprise",
    ]

    # Mapping from HSEmotion names to our standard names
    EMOTION_MAP = {
        "Anger": "angry",
        "Contempt": "contempt",
        "Disgust": "disgust",
        "Fear": "fear",
        "Happiness": "happy",
        "Neutral": "neutral",
        "Sadness": "sad",
        "Surprise": "surprise",
    }

    def __init__(self, model_name: str = "enet_b0_8_best_vgaf"):
        self._model_name = model_name
        self._model = None
        self._initialized = False

    def initialize(self, device: str = "cuda:0") -> None:
        """Initialize HSEmotion recognizer."""
        if self._initialized:
            return  # Already initialized

        try:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

            self._model = HSEmotionRecognizer(model_name=self._model_name)
            self._initialized = True
            logger.info(f"HSEmotion backend initialized (model={self._model_name})")

        except ImportError:
            raise ImportError(
                "hsemotion-onnx is required for HSEmotionBackend. "
                "Install with: pip install hsemotion-onnx"
            )
        except Exception as e:
            logger.error(f"Failed to initialize HSEmotion: {e}")
            raise

    def analyze(
        self, image: np.ndarray, faces: List[DetectedFace]
    ) -> List[FaceExpression]:
        """Analyze expressions for detected faces using HSEmotion.

        Args:
            image: BGR image as numpy array (H, W, 3).
            faces: List of detected faces to analyze.

        Returns:
            List of expression results corresponding to input faces.
        """
        if not self._initialized or self._model is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        if not faces:
            return []

        import cv2

        results = []

        for face in faces:
            x, y, w, h = face.bbox

            # Expand bbox slightly for better recognition
            pad = int(max(w, h) * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(image.shape[1], x + w + pad)
            y2 = min(image.shape[0], y + h + pad)

            # Crop face region
            face_img = image[y1:y2, x1:x2]

            if face_img.size == 0:
                results.append(FaceExpression())
                continue

            # Convert BGR to RGB for HSEmotion
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            try:
                # HSEmotion predict returns (emotion_label, emotion_scores)
                emotion_label, emotion_scores = self._model.predict_emotions(
                    face_rgb, logits=False
                )

                # Build emotions dictionary
                emotions = {}
                max_emotion = "neutral"
                max_prob = 0.0

                for i, score in enumerate(emotion_scores):
                    if i < len(self.EMOTIONS):
                        hs_name = self.EMOTIONS[i]
                        std_name = self.EMOTION_MAP.get(hs_name, hs_name.lower())
                        prob = float(score)
                        emotions[std_name] = prob

                        if prob > max_prob:
                            max_prob = prob
                            max_emotion = std_name

                # Calculate expression intensity from emotion probabilities
                # High intensity = strong non-neutral emotion
                neutral_prob = emotions.get("neutral", 0.0)
                happy_prob = emotions.get("happy", 0.0)
                surprise_prob = emotions.get("surprise", 0.0)

                # Expression intensity based on non-neutral emotions
                expression_intensity = max(
                    happy_prob,
                    surprise_prob * 0.8,
                    (1.0 - neutral_prob) * 0.7,  # Any non-neutral counts
                )

                results.append(
                    FaceExpression(
                        action_units={},  # HSEmotion doesn't support AUs
                        emotions=emotions,
                        expression_intensity=min(1.0, expression_intensity),
                        dominant_emotion=max_emotion,
                    )
                )

            except Exception as e:
                logger.warning(f"Failed to analyze face expression: {e}")
                results.append(FaceExpression())

        return results

    def cleanup(self) -> None:
        """Release HSEmotion resources."""
        self._model = None
        self._initialized = False
        logger.info("HSEmotion backend cleaned up")
