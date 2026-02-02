"""Py-Feat backend for expression analysis with Action Unit support."""

from typing import List, Optional
import logging

import numpy as np

from facemoment.moment_detector.extractors.backends.base import (
    DetectedFace,
    FaceExpression,
)

logger = logging.getLogger(__name__)


class PyFeatBackend:
    """Expression analysis backend using Py-Feat.

    Py-Feat provides Action Unit detection and emotion classification.
    Uses a pre-trained model for facial expression analysis.

    Note: This backend is slower than HSEmotionBackend (~2000ms vs ~30ms per frame)
    but provides Action Unit (AU) detection which HSEmotion does not support.

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
