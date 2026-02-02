"""HSEmotion backend for fast expression analysis."""

from typing import List, Optional
import logging

import numpy as np

from facemoment.moment_detector.extractors.backends.base import (
    DetectedFace,
    FaceExpression,
)

logger = logging.getLogger(__name__)


class HSEmotionBackend:
    """Expression analysis backend using HSEmotion-ONNX.

    HSEmotion provides fast emotion classification using EfficientNet-B0.
    This is significantly faster than Py-Feat (~30ms vs ~2000ms per frame).

    Supports 8 emotions: Anger, Contempt, Disgust, Fear, Happiness,
    Neutral, Sadness, Surprise.

    Note: This backend does NOT support Action Units (AU) - only emotion
    classification and expression intensity. Use PyFeatBackend if you need AU.

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
