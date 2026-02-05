"""Video panel — draws annotations directly on the video frame.

Only draws spatial annotations that belong on the video:
- Face bounding boxes with role labels
- Pose skeletons
- ROI boundary
- Trigger flash
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from facemoment.moment_detector.extractors.base import Observation, FaceObservation
from facemoment.moment_detector.extractors.types import KeypointIndex, HandLandmarkIndex
from facemoment.moment_detector.extractors.outputs import PoseOutput, GestureOutput
from facemoment.moment_detector.visualize.components import (
    COLOR_DARK_BGR,
    COLOR_WHITE_BGR,
    COLOR_RED_BGR,
    COLOR_GREEN_BGR,
    COLOR_GRAY_BGR,
    COLOR_HAPPY_BGR,
    COLOR_SKELETON_BGR,
    COLOR_KEYPOINT_BGR,
    ROLE_COLORS,
    FONT,
    DebugLayer,
    LayerState,
)


class VideoPanel:
    """Draws spatial annotations on the video frame.

    Only annotations that refer to specific image locations belong here:
    bboxes, skeletons, ROI, and trigger flash.
    """

    def draw(
        self,
        image: np.ndarray,
        face_obs: Optional[Observation] = None,
        pose_obs: Optional[Observation] = None,
        gesture_obs: Optional[Observation] = None,
        classifier_obs: Optional[Observation] = None,
        fusion_result: Optional[Observation] = None,
        roi: Optional[Tuple[float, float, float, float]] = None,
        layers: Optional[LayerState] = None,
    ) -> np.ndarray:
        """Draw all video annotations.

        Args:
            image: Video frame (will be copied).
            face_obs: Face detection observation.
            pose_obs: Pose observation.
            gesture_obs: Gesture observation (hand landmarks).
            classifier_obs: Face classifier observation (role-based colors).
            fusion_result: Fusion result (for trigger flash).
            roi: ROI boundary in normalized coordinates (x1, y1, x2, y2).
            layers: Layer visibility state. None means all layers enabled.

        Returns:
            Annotated image.
        """
        output = image.copy()

        if roi is not None and (layers is None or layers[DebugLayer.ROI]):
            self._draw_roi(output, roi)

        if pose_obs is not None and (layers is None or layers[DebugLayer.POSE]):
            self._draw_pose(output, pose_obs)

        if gesture_obs is not None and (layers is None or layers[DebugLayer.GESTURE]):
            self._draw_gesture(output, gesture_obs)

        if layers is None or layers[DebugLayer.FACE]:
            if classifier_obs is not None:
                self._draw_classified_faces(output, classifier_obs)
            elif face_obs is not None:
                self._draw_faces(output, face_obs)

        if fusion_result is not None and fusion_result.should_trigger:
            if layers is None or layers[DebugLayer.TRIGGER]:
                self._draw_trigger_flash(output, fusion_result)

        return output

    # --- Face annotations ---

    def _draw_faces(self, image: np.ndarray, obs: Observation) -> None:
        """Draw basic face bboxes (no classifier data)."""
        h, w = image.shape[:2]
        for face in obs.faces:
            x1 = int(face.bbox[0] * w)
            y1 = int(face.bbox[1] * h)
            x2 = int((face.bbox[0] + face.bbox[2]) * w)
            y2 = int((face.bbox[1] + face.bbox[3]) * h)

            is_good = (
                face.confidence >= 0.7
                and abs(face.yaw) <= 25
                and abs(face.pitch) <= 20
                and face.inside_frame
            )
            color = COLOR_GREEN_BGR if is_good else COLOR_RED_BGR

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{face.face_id}"
            cv2.putText(
                image, label, (x1 + 2, y1 - 5),
                FONT, 0.35, color, 1,
            )

    def _draw_classified_faces(self, image: np.ndarray, obs: Observation) -> None:
        """Draw face bboxes with role-based colors from classifier."""
        h, w = image.shape[:2]

        if obs.data is None:
            return
        data = obs.data
        if not hasattr(data, "faces") or not data.faces:
            return

        for cf in data.faces:
            face = cf.face
            role = cf.role
            track_length = cf.track_length

            x1 = int(face.bbox[0] * w)
            y1 = int(face.bbox[1] * h)
            x2 = int((face.bbox[0] + face.bbox[2]) * w)
            y2 = int((face.bbox[1] + face.bbox[3]) * h)

            color = ROLE_COLORS.get(role, COLOR_GRAY_BGR)
            thickness = 3 if role == "main" else 2
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # Role label with background
            label = f"{role.upper()} ({track_length}f)"
            label_size = cv2.getTextSize(label, FONT, 0.45, 1)[0]
            label_y = y1 - 5 if y1 > 25 else y2 + 15

            cv2.rectangle(
                image,
                (x1, label_y - label_size[1] - 4),
                (x1 + label_size[0] + 4, label_y + 2),
                color, -1,
            )
            cv2.putText(
                image, label, (x1 + 2, label_y - 2),
                FONT, 0.45, COLOR_DARK_BGR, 1,
            )

            # Face ID
            cv2.putText(
                image, f"ID:{face.face_id}", (x1 + 2, y2 - 5),
                FONT, 0.35, color, 1,
            )

            # Emotion bars for main/passenger
            if role in ("main", "passenger"):
                self._draw_emotion_bars(image, face, x1, y2 + 5, x2 - x1)

    def _draw_emotion_bars(
        self, image: np.ndarray, face: FaceObservation, x: int, y: int, width: int
    ) -> None:
        """Draw 3 emotion bars below face bbox."""
        bar_h, gap = 6, 2
        emotions = [
            (face.signals.get("em_happy", 0.0), (0, 255, 255)),
            (face.signals.get("em_angry", 0.0), (0, 0, 255)),
            (face.signals.get("em_neutral", 0.0), (200, 200, 200)),
        ]
        for i, (value, color) in enumerate(emotions):
            ey = y + i * (bar_h + gap)
            cv2.rectangle(image, (x, ey), (x + width, ey + bar_h), COLOR_DARK_BGR, -1)
            cv2.rectangle(
                image, (x, ey),
                (x + int(width * min(1.0, value)), ey + bar_h),
                color, -1,
            )

    # --- Pose annotations ---

    def _draw_pose(self, image: np.ndarray, obs: Observation) -> None:
        """Draw upper body pose skeleton."""
        h, w = image.shape[:2]
        if obs.data is None:
            return

        # Support both PoseOutput (attribute access) and dict (serialized from subprocess)
        if hasattr(obs.data, "keypoints"):
            keypoints = obs.data.keypoints
        elif isinstance(obs.data, dict) and "keypoints" in obs.data:
            keypoints = obs.data["keypoints"]
        else:
            return

        for person in keypoints:
            self._draw_upper_body_skeleton(image, person, w, h)

    def _draw_upper_body_skeleton(
        self, image: np.ndarray, person: Dict, w: int, h: int
    ) -> None:
        """Draw upper body skeleton for a person."""
        keypoints = person.get("keypoints", [])
        if not keypoints or len(keypoints) < 11:
            return

        connections = [
            (KeypointIndex.NOSE, KeypointIndex.LEFT_EYE),
            (KeypointIndex.NOSE, KeypointIndex.RIGHT_EYE),
            (KeypointIndex.LEFT_EYE, KeypointIndex.LEFT_EAR),
            (KeypointIndex.RIGHT_EYE, KeypointIndex.RIGHT_EAR),
            (KeypointIndex.LEFT_SHOULDER, KeypointIndex.RIGHT_SHOULDER),
            (KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_ELBOW),
            (KeypointIndex.LEFT_ELBOW, KeypointIndex.LEFT_WRIST),
            (KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_ELBOW),
            (KeypointIndex.RIGHT_ELBOW, KeypointIndex.RIGHT_WRIST),
        ]

        min_conf = 0.3

        # Skeleton lines
        for idx1, idx2 in connections:
            if idx1 >= len(keypoints) or idx2 >= len(keypoints):
                continue
            kpt1, kpt2 = keypoints[idx1], keypoints[idx2]
            if kpt1[2] < min_conf or kpt2[2] < min_conf:
                continue
            pt1 = (int(kpt1[0]), int(kpt1[1]))
            pt2 = (int(kpt2[0]), int(kpt2[1]))
            cv2.line(image, pt1, pt2, COLOR_SKELETON_BGR, 2)

        # Keypoints
        upper_body_indices = [
            KeypointIndex.NOSE,
            KeypointIndex.LEFT_EYE, KeypointIndex.RIGHT_EYE,
            KeypointIndex.LEFT_EAR, KeypointIndex.RIGHT_EAR,
            KeypointIndex.LEFT_SHOULDER, KeypointIndex.RIGHT_SHOULDER,
            KeypointIndex.LEFT_ELBOW, KeypointIndex.RIGHT_ELBOW,
            KeypointIndex.LEFT_WRIST, KeypointIndex.RIGHT_WRIST,
        ]
        for idx in upper_body_indices:
            if idx >= len(keypoints):
                continue
            kpt = keypoints[idx]
            if kpt[2] < min_conf:
                continue
            pt = (int(kpt[0]), int(kpt[1]))
            if idx == KeypointIndex.NOSE:
                color, radius = COLOR_WHITE_BGR, 4
            elif idx in (KeypointIndex.LEFT_WRIST, KeypointIndex.RIGHT_WRIST):
                color, radius = COLOR_HAPPY_BGR, 6
            elif idx in (KeypointIndex.LEFT_SHOULDER, KeypointIndex.RIGHT_SHOULDER):
                color, radius = COLOR_GREEN_BGR, 5
            else:
                color, radius = COLOR_KEYPOINT_BGR, 3
            cv2.circle(image, pt, radius, color, -1)
            cv2.circle(image, pt, radius, COLOR_DARK_BGR, 1)

    # --- Gesture annotations ---

    # Gesture colors
    COLOR_GESTURE_BGR = (255, 128, 0)  # Orange for hand landmarks
    COLOR_GESTURE_LINE_BGR = (200, 100, 0)  # Darker orange for connections

    def _draw_gesture(self, image: np.ndarray, obs: Observation) -> None:
        """Draw hand landmarks and gesture labels."""
        h, w = image.shape[:2]
        if obs.data is None:
            return

        # Support both GestureOutput (attribute access) and dict (serialized)
        if hasattr(obs.data, "hand_landmarks"):
            hand_landmarks = obs.data.hand_landmarks
        elif isinstance(obs.data, dict) and "hand_landmarks" in obs.data:
            hand_landmarks = obs.data["hand_landmarks"]
        else:
            return

        for hand in hand_landmarks:
            self._draw_hand_landmarks(image, hand, w, h)

    def _draw_hand_landmarks(
        self, image: np.ndarray, hand: Dict, w: int, h: int
    ) -> None:
        """Draw 21 hand landmarks and connections for a single hand."""
        landmarks = hand.get("landmarks", [])
        if not landmarks or len(landmarks) < 21:
            return

        handedness = hand.get("handedness", "")
        gesture = hand.get("gesture", "")
        gesture_conf = hand.get("gesture_confidence", 0.0)

        # MediaPipe hand connections
        connections = [
            # Thumb
            (HandLandmarkIndex.WRIST, HandLandmarkIndex.THUMB_CMC),
            (HandLandmarkIndex.THUMB_CMC, HandLandmarkIndex.THUMB_MCP),
            (HandLandmarkIndex.THUMB_MCP, HandLandmarkIndex.THUMB_IP),
            (HandLandmarkIndex.THUMB_IP, HandLandmarkIndex.THUMB_TIP),
            # Index finger
            (HandLandmarkIndex.WRIST, HandLandmarkIndex.INDEX_FINGER_MCP),
            (HandLandmarkIndex.INDEX_FINGER_MCP, HandLandmarkIndex.INDEX_FINGER_PIP),
            (HandLandmarkIndex.INDEX_FINGER_PIP, HandLandmarkIndex.INDEX_FINGER_DIP),
            (HandLandmarkIndex.INDEX_FINGER_DIP, HandLandmarkIndex.INDEX_FINGER_TIP),
            # Middle finger
            (HandLandmarkIndex.WRIST, HandLandmarkIndex.MIDDLE_FINGER_MCP),
            (HandLandmarkIndex.MIDDLE_FINGER_MCP, HandLandmarkIndex.MIDDLE_FINGER_PIP),
            (HandLandmarkIndex.MIDDLE_FINGER_PIP, HandLandmarkIndex.MIDDLE_FINGER_DIP),
            (HandLandmarkIndex.MIDDLE_FINGER_DIP, HandLandmarkIndex.MIDDLE_FINGER_TIP),
            # Ring finger
            (HandLandmarkIndex.WRIST, HandLandmarkIndex.RING_FINGER_MCP),
            (HandLandmarkIndex.RING_FINGER_MCP, HandLandmarkIndex.RING_FINGER_PIP),
            (HandLandmarkIndex.RING_FINGER_PIP, HandLandmarkIndex.RING_FINGER_DIP),
            (HandLandmarkIndex.RING_FINGER_DIP, HandLandmarkIndex.RING_FINGER_TIP),
            # Pinky
            (HandLandmarkIndex.WRIST, HandLandmarkIndex.PINKY_MCP),
            (HandLandmarkIndex.PINKY_MCP, HandLandmarkIndex.PINKY_PIP),
            (HandLandmarkIndex.PINKY_PIP, HandLandmarkIndex.PINKY_DIP),
            (HandLandmarkIndex.PINKY_DIP, HandLandmarkIndex.PINKY_TIP),
            # Palm connections
            (HandLandmarkIndex.INDEX_FINGER_MCP, HandLandmarkIndex.MIDDLE_FINGER_MCP),
            (HandLandmarkIndex.MIDDLE_FINGER_MCP, HandLandmarkIndex.RING_FINGER_MCP),
            (HandLandmarkIndex.RING_FINGER_MCP, HandLandmarkIndex.PINKY_MCP),
        ]

        # Draw connections
        for idx1, idx2 in connections:
            if idx1 >= len(landmarks) or idx2 >= len(landmarks):
                continue
            lm1, lm2 = landmarks[idx1], landmarks[idx2]
            pt1 = (int(lm1[0] * w), int(lm1[1] * h))
            pt2 = (int(lm2[0] * w), int(lm2[1] * h))
            cv2.line(image, pt1, pt2, self.COLOR_GESTURE_LINE_BGR, 2)

        # Draw landmarks
        for i, lm in enumerate(landmarks):
            pt = (int(lm[0] * w), int(lm[1] * h))
            # Fingertips are larger
            if i in (
                HandLandmarkIndex.THUMB_TIP,
                HandLandmarkIndex.INDEX_FINGER_TIP,
                HandLandmarkIndex.MIDDLE_FINGER_TIP,
                HandLandmarkIndex.RING_FINGER_TIP,
                HandLandmarkIndex.PINKY_TIP,
            ):
                radius = 6
            elif i == HandLandmarkIndex.WRIST:
                radius = 5
            else:
                radius = 3
            cv2.circle(image, pt, radius, self.COLOR_GESTURE_BGR, -1)
            cv2.circle(image, pt, radius, COLOR_DARK_BGR, 1)

        # Draw gesture label near wrist
        if landmarks:
            wrist = landmarks[HandLandmarkIndex.WRIST]
            label_x = int(wrist[0] * w)
            label_y = int(wrist[1] * h) + 20

            # Handedness + gesture
            if gesture and gesture != "none":
                label = f"{handedness[0]}:{gesture} ({gesture_conf:.2f})"
                color = COLOR_GREEN_BGR
            else:
                label = f"{handedness[0]}: --"
                color = self.COLOR_GESTURE_BGR

            # Background for readability
            label_size = cv2.getTextSize(label, FONT, 0.5, 1)[0]
            cv2.rectangle(
                image,
                (label_x - 2, label_y - label_size[1] - 4),
                (label_x + label_size[0] + 4, label_y + 4),
                COLOR_DARK_BGR, -1,
            )
            cv2.putText(image, label, (label_x, label_y), FONT, 0.5, color, 1)

    # --- ROI ---

    def _draw_roi(self, image: np.ndarray, roi: Tuple[float, float, float, float]) -> None:
        """Draw subtle ROI boundary."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = roi
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        cv2.rectangle(image, (px1, py1), (px2, py2), (80, 80, 80), 1)

    # --- Trigger flash ---

    def _draw_trigger_flash(self, image: np.ndarray, result: Observation) -> None:
        """Draw trigger flash border and text."""
        h, w = image.shape[:2]
        cv2.rectangle(image, (0, 0), (w - 1, h - 1), COLOR_RED_BGR, 10)
        cv2.putText(
            image, f"TRIGGER: {result.trigger_reason}",
            (w // 2 - 100, h // 2),
            FONT, 1.0, COLOR_RED_BGR, 3,
        )
