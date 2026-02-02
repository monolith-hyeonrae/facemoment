"""Pose estimation backend implementations."""

from typing import List, Optional
import logging

import numpy as np

from facemoment.moment_detector.extractors.backends.base import PoseKeypoints
from facemoment.moment_detector.extractors.types import COCO_KEYPOINT_NAMES

logger = logging.getLogger(__name__)


class YOLOPoseBackend:
    """Pose estimation backend using YOLOv8-Pose.

    YOLOv8-Pose provides fast and accurate pose estimation with
    17 COCO keypoints for body tracking and gesture detection.

    Args:
        model_name: Model variant (default: "yolov8m-pose.pt" for balanced speed/accuracy).
        conf_threshold: Detection confidence threshold.
        iou_threshold: NMS IoU threshold.

    Example:
        >>> backend = YOLOPoseBackend()
        >>> backend.initialize("cuda:0")
        >>> poses = backend.detect(image)
        >>> backend.cleanup()
    """

    def __init__(
        self,
        model_name: str = "yolov8m-pose.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.7,
    ):
        self._model_name = model_name
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._model: Optional[object] = None
        self._initialized = False

    def initialize(self, device: str = "cuda:0") -> None:
        """Initialize YOLOv8-Pose model."""
        if self._initialized:
            return  # Already initialized

        try:
            from ultralytics import YOLO

            self._model = YOLO(self._model_name)

            # Set device (YOLO accepts device string or int)
            if device.startswith("cuda"):
                device_id = device.split(":")[-1] if ":" in device else "0"
                self._device = int(device_id)
            else:
                self._device = "cpu"

            self._initialized = True
            logger.info(f"YOLOv8-Pose initialized with model {self._model_name}")

        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLOPoseBackend. "
                "Install with: pip install ultralytics"
            )
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8-Pose: {e}")
            raise

    def detect(self, image: np.ndarray) -> List[PoseKeypoints]:
        """Detect poses using YOLOv8-Pose.

        Args:
            image: BGR image as numpy array (H, W, 3).

        Returns:
            List of detected poses with COCO 17 keypoints.
        """
        if not self._initialized or self._model is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        # Run inference
        results = self._model(
            image,
            device=self._device,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            verbose=False,
        )

        poses = []

        for result in results:
            if result.keypoints is None:
                continue

            keypoints_data = result.keypoints.data.cpu().numpy()
            boxes = result.boxes

            for i, kpts in enumerate(keypoints_data):
                # kpts shape: (17, 3) - x, y, confidence
                if kpts.shape[0] != 17:
                    continue

                # Get bounding box if available
                bbox = None
                conf = 1.0
                if boxes is not None and i < len(boxes):
                    box = boxes[i]
                    xyxy = box.xyxy[0].cpu().numpy()
                    bbox = (
                        int(xyxy[0]),
                        int(xyxy[1]),
                        int(xyxy[2] - xyxy[0]),
                        int(xyxy[3] - xyxy[1]),
                    )
                    conf = float(box.conf[0])

                poses.append(
                    PoseKeypoints(
                        keypoints=kpts.astype(np.float32),
                        keypoint_names=COCO_KEYPOINT_NAMES,
                        person_id=i,
                        bbox=bbox,
                        confidence=conf,
                    )
                )

        return poses

    def cleanup(self) -> None:
        """Release YOLOv8-Pose resources."""
        self._model = None
        self._initialized = False
        logger.info("YOLOv8-Pose cleaned up")

    @staticmethod
    def get_keypoint_index(name: str) -> int:
        """Get index of a keypoint by name.

        Args:
            name: Keypoint name (e.g., "left_wrist").

        Returns:
            Index in the keypoints array.

        Raises:
            ValueError: If keypoint name is not found.
        """
        try:
            return COCO_KEYPOINT_NAMES.index(name)
        except ValueError:
            raise ValueError(
                f"Unknown keypoint: {name}. "
                f"Valid names: {COCO_KEYPOINT_NAMES}"
            )
