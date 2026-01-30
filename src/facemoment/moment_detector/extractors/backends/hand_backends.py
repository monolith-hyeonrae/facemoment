"""Hand landmark detection backends for gesture recognition."""

from typing import List, Optional
from pathlib import Path
import logging
import urllib.request

import numpy as np

from facemoment.moment_detector.extractors.backends.base import HandLandmarks

logger = logging.getLogger(__name__)

# Model download URL
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def _get_model_path() -> Path:
    """Get path to hand landmarker model, downloading if necessary."""
    # Store in user's cache directory
    cache_dir = Path.home() / ".cache" / "facemoment" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = cache_dir / "hand_landmarker.task"

    if not model_path.exists():
        logger.info(f"Downloading hand landmarker model to {model_path}...")
        try:
            urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, model_path)
            logger.info("Download complete.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download hand landmarker model: {e}\n"
                f"You can manually download from: {HAND_LANDMARKER_MODEL_URL}\n"
                f"And save to: {model_path}"
            ) from e

    return model_path


class MediaPipeHandsBackend:
    """MediaPipe Hands backend for hand landmark detection.

    Uses MediaPipe Tasks API (0.10.x+) with HandLandmarker.
    Detects up to 2 hands and provides 21 landmarks per hand.

    Landmark indices:
    - WRIST = 0
    - THUMB_CMC = 1, THUMB_MCP = 2, THUMB_IP = 3, THUMB_TIP = 4
    - INDEX_FINGER_MCP = 5, INDEX_FINGER_PIP = 6, INDEX_FINGER_DIP = 7, INDEX_FINGER_TIP = 8
    - MIDDLE_FINGER_MCP = 9, MIDDLE_FINGER_PIP = 10, MIDDLE_FINGER_DIP = 11, MIDDLE_FINGER_TIP = 12
    - RING_FINGER_MCP = 13, RING_FINGER_PIP = 14, RING_FINGER_DIP = 15, RING_FINGER_TIP = 16
    - PINKY_MCP = 17, PINKY_PIP = 18, PINKY_DIP = 19, PINKY_TIP = 20

    Args:
        max_num_hands: Maximum number of hands to detect (default: 2).
        min_detection_confidence: Minimum confidence for detection (default: 0.5).
        min_tracking_confidence: Minimum confidence for tracking (default: 0.5).
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._max_num_hands = max_num_hands
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._landmarker: Optional[object] = None
        self._initialized = False

    def initialize(self, device: str = "cpu") -> None:
        """Initialize MediaPipe HandLandmarker.

        Args:
            device: Device to use (MediaPipe uses CPU by default).
        """
        if self._initialized:
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            # Get model path (downloads if necessary)
            model_path = _get_model_path()

            # Create options
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self._max_num_hands,
                min_hand_detection_confidence=self._min_detection_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
            )

            self._landmarker = vision.HandLandmarker.create_from_options(options)
            self._initialized = True
            logger.info("MediaPipe Hands backend initialized (Tasks API)")

        except ImportError as e:
            raise ImportError(
                "MediaPipe is required for gesture detection. "
                "Install it with: pip install mediapipe"
            ) from e

    def detect(self, image: np.ndarray) -> List[HandLandmarks]:
        """Detect hands and landmarks in an image.

        Args:
            image: BGR image as numpy array (H, W, 3).

        Returns:
            List of detected hands with landmarks.
        """
        if not self._initialized or self._landmarker is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        import mediapipe as mp
        import cv2

        # MediaPipe expects RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Detect hands
        result = self._landmarker.detect(mp_image)

        hands = []
        if result.hand_landmarks:
            for idx, hand_lms in enumerate(result.hand_landmarks):
                # Get handedness
                handedness = "Right"  # Default
                confidence = 0.8

                if result.handedness and idx < len(result.handedness):
                    # handedness is a list of categories
                    hand_info = result.handedness[idx]
                    if hand_info:
                        handedness = hand_info[0].category_name
                        confidence = hand_info[0].score

                # Extract landmarks as numpy array (normalized coordinates)
                landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_lms],
                    dtype=np.float32,
                )

                hands.append(
                    HandLandmarks(
                        landmarks=landmarks,
                        handedness=handedness,
                        confidence=confidence,
                    )
                )

        return hands

    def cleanup(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        self._initialized = False
        logger.info("MediaPipe Hands backend cleaned up")
