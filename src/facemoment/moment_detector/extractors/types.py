"""Type definitions for extractors.

Contains keypoint indices, gesture types, and other constants used
by multiple extractors.
"""

from enum import Enum


class KeypointIndex:
    """COCO 17 keypoint indices for pose estimation.

    Used with YOLOv8-Pose and similar backends.

    Example:
        >>> kpts = pose.keypoints
        >>> left_wrist = kpts[KeypointIndex.LEFT_WRIST]
        >>> if left_wrist[2] > 0.5:  # confidence threshold
        ...     print(f"Left wrist at ({left_wrist[0]}, {left_wrist[1]})")
    """

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


class HandLandmarkIndex:
    """MediaPipe hand landmark indices.

    21 landmarks per hand as defined by MediaPipe Hands.

    Example:
        >>> lms = hand.landmarks
        >>> thumb_tip = lms[HandLandmarkIndex.THUMB_TIP]
        >>> index_tip = lms[HandLandmarkIndex.INDEX_FINGER_TIP]
    """

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


class GestureType(Enum):
    """Recognized gesture types.

    Used by GestureExtractor to classify hand gestures.

    Example:
        >>> gesture, confidence = extractor._classify_gesture(hand)
        >>> if gesture == GestureType.V_SIGN:
        ...     print("Peace sign detected!")
    """

    NONE = "none"
    V_SIGN = "v_sign"
    THUMBS_UP = "thumbs_up"
    OK_SIGN = "ok_sign"
    OPEN_PALM = "open_palm"
    FIST = "fist"
    POINTING = "pointing"


# COCO keypoint names in order (for reference)
COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


__all__ = [
    "KeypointIndex",
    "HandLandmarkIndex",
    "GestureType",
    "COCO_KEYPOINT_NAMES",
]
