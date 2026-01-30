"""Highlight fusion for detecting photo-worthy moments."""

from typing import Optional, Dict, List
from collections import deque
from dataclasses import dataclass
import logging

from visualbase import Trigger

from facemoment.moment_detector.extractors.base import Observation, FaceObservation
from facemoment.moment_detector.fusion.base import BaseFusion, FusionResult
from facemoment.observability import ObservabilityHub, TraceLevel
from facemoment.observability.records import (
    GateChangeRecord,
    GateConditionRecord,
    TriggerDecisionRecord,
    TriggerFireRecord,
)

logger = logging.getLogger(__name__)

# Get the global observability hub
_hub = ObservabilityHub.get_instance()


@dataclass
class ExpressionState:
    """State for expression spike detection using EWMA."""

    ewma: float = 0.0
    ewma_var: float = 0.01  # Variance for z-score
    count: int = 0


class HighlightFusion(BaseFusion):
    """Fusion module for detecting highlight-worthy moments.

    Combines signals from face, pose, and quality extractors to
    identify moments worth capturing. Uses a quality gate with
    hysteresis and detects various trigger events.

    Trigger Types:
    - expression_spike: Sudden increase in facial expression
    - head_turn: Quick head rotation (looking at camera)
    - hand_wave: Hand waving gesture
    - camera_gaze: Looking directly at camera (Phase 9)
    - passenger_interaction: Two people looking at each other (Phase 9)
    - gesture_vsign: V-sign gesture (Phase 9)
    - gesture_thumbsup: Thumbs up gesture (Phase 9)

    Gate Conditions (composition quality):
    - 1-2 faces detected
    - Face confidence above threshold
    - Face angles within limits
    - Face position centered
    - Quality metrics acceptable

    Args:
        face_conf_threshold: Minimum face confidence (default: 0.7).
        yaw_max: Maximum head yaw angle (default: 25.0).
        pitch_max: Maximum head pitch angle (default: 20.0).
        gate_open_duration_sec: Hysteresis duration to open gate (default: 0.7).
        gate_close_duration_sec: Hysteresis duration to close gate (default: 0.3).
        expression_z_threshold: Z-score threshold for expression spike (default: 2.0).
        ewma_alpha: EWMA smoothing factor (default: 0.1).
        head_turn_velocity_threshold: Angular velocity for head turn (default: 30.0 deg/sec).
        cooldown_sec: Cooldown between triggers (default: 2.0).
        consecutive_frames: Frames to confirm trigger (default: 2).
        pre_sec: Seconds before event in clip (default: 2.0).
        post_sec: Seconds after event in clip (default: 2.0).
        gaze_yaw_threshold: Max yaw for camera gaze detection (default: 10.0).
        gaze_pitch_threshold: Max pitch for camera gaze detection (default: 15.0).
        gaze_score_threshold: Min score to trigger camera gaze (default: 0.5).
        interaction_yaw_threshold: Min yaw for passenger interaction (default: 15.0).

    Example:
        >>> fusion = HighlightFusion()
        >>> for obs in observations:
        ...     result = fusion.update(obs)
        ...     if result.should_trigger:
        ...         print(f"Trigger: {result.reason}, score={result.score:.2f}")
    """

    def __init__(
        self,
        # Gate thresholds
        face_conf_threshold: float = 0.7,
        yaw_max: float = 25.0,
        pitch_max: float = 20.0,
        min_face_area_ratio: float = 0.01,
        max_center_distance: float = 0.4,
        # Hysteresis
        gate_open_duration_sec: float = 0.7,
        gate_close_duration_sec: float = 0.3,
        # Expression detection
        expression_z_threshold: float = 2.0,
        ewma_alpha: float = 0.1,
        # Head turn detection
        head_turn_velocity_threshold: float = 30.0,
        # Timing
        cooldown_sec: float = 2.0,
        consecutive_frames: int = 2,
        pre_sec: float = 2.0,
        post_sec: float = 2.0,
        # Camera gaze detection (Phase 9)
        gaze_yaw_threshold: float = 10.0,
        gaze_pitch_threshold: float = 15.0,
        gaze_score_threshold: float = 0.5,
        # Passenger interaction detection (Phase 9)
        interaction_yaw_threshold: float = 15.0,
    ):
        # Gate parameters
        self._face_conf_threshold = face_conf_threshold
        self._yaw_max = yaw_max
        self._pitch_max = pitch_max
        self._min_face_area = min_face_area_ratio
        self._max_center_dist = max_center_distance

        # Hysteresis (in nanoseconds)
        self._gate_open_duration_ns = int(gate_open_duration_sec * 1e9)
        self._gate_close_duration_ns = int(gate_close_duration_sec * 1e9)

        # Detection parameters
        self._expr_z_threshold = expression_z_threshold
        self._ewma_alpha = ewma_alpha
        self._head_turn_vel_threshold = head_turn_velocity_threshold

        # Camera gaze detection parameters (Phase 9)
        self._gaze_yaw_threshold = gaze_yaw_threshold
        self._gaze_pitch_threshold = gaze_pitch_threshold
        self._gaze_score_threshold = gaze_score_threshold

        # Passenger interaction parameters (Phase 9)
        self._interaction_yaw_threshold = interaction_yaw_threshold

        # Timing
        self._cooldown_ns = int(cooldown_sec * 1e9)
        self._consecutive_required = consecutive_frames
        self._pre_sec = pre_sec
        self._post_sec = post_sec

        # State initialization
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset all internal state."""
        # Gate state
        self._gate_open = False
        self._gate_condition_first_met_ns: Optional[int] = None
        self._gate_condition_first_failed_ns: Optional[int] = None

        # Cooldown
        self._last_trigger_ns: Optional[int] = None

        # Expression tracking (per face ID)
        self._expression_states: Dict[int, ExpressionState] = {}

        # Head pose tracking (per face ID)
        self._prev_yaw: Dict[int, tuple[int, float]] = {}  # face_id -> (t_ns, yaw)

        # Consecutive frame counting
        self._consecutive_high: int = 0
        self._pending_trigger_reason: Optional[str] = None
        self._pending_trigger_score: float = 0.0

        # History
        self._recent_observations: deque[Observation] = deque(maxlen=30)
        self._observation_count = 0

    def update(self, observation: Observation) -> FusionResult:
        """Process observation and decide on trigger.

        Args:
            observation: New observation from extractors.

        Returns:
            FusionResult indicating trigger decision.
        """
        self._recent_observations.append(observation)
        self._observation_count += 1
        t_ns = observation.t_ns

        # 1. Check cooldown
        if self._in_cooldown(t_ns):
            # Emit decision record at NORMAL level
            if _hub.enabled and _hub.is_level_enabled(TraceLevel.NORMAL):
                _hub.emit(TriggerDecisionRecord(
                    frame_id=observation.frame_id,
                    t_ns=t_ns,
                    gate_open=self._gate_open,
                    in_cooldown=True,
                    candidates=[],
                    consecutive_count=0,
                    consecutive_required=self._consecutive_required,
                    decision="blocked_cooldown",
                ))
            return FusionResult(
                should_trigger=False,
                observations_used=self._observation_count,
                metadata={"state": "cooldown"},
            )

        # 2. Update gate with hysteresis
        gate_conditions_met = self._check_gate_conditions(observation)
        self._update_gate_hysteresis(t_ns, gate_conditions_met, observation.frame_id)

        if not self._gate_open:
            self._consecutive_high = 0
            # Emit decision record at NORMAL level
            if _hub.enabled and _hub.is_level_enabled(TraceLevel.NORMAL):
                _hub.emit(TriggerDecisionRecord(
                    frame_id=observation.frame_id,
                    t_ns=t_ns,
                    gate_open=False,
                    in_cooldown=False,
                    candidates=[],
                    consecutive_count=0,
                    consecutive_required=self._consecutive_required,
                    decision="blocked_gate",
                ))
            return FusionResult(
                should_trigger=False,
                observations_used=self._observation_count,
                metadata={"state": "gate_closed", "conditions_met": gate_conditions_met},
            )

        # 3. Detect trigger events and collect candidates
        trigger_reason = None
        trigger_score = 0.0
        candidates: List[Dict[str, any]] = []

        # Check expression spikes
        expr_spike = self._detect_expression_spike(observation)
        if expr_spike is not None:
            candidates.append({"reason": "expression_spike", "score": expr_spike, "source": "face"})
            trigger_reason = "expression_spike"
            trigger_score = expr_spike

        # Check head turns
        head_turn = self._detect_head_turn(observation)
        if head_turn is not None:
            candidates.append({"reason": "head_turn", "score": head_turn, "source": "face"})
            if trigger_reason is None or head_turn > trigger_score:
                trigger_reason = "head_turn"
                trigger_score = head_turn

        # Check hand waves (from pose extractor)
        hand_wave = observation.signals.get("hand_wave_detected", 0.0)
        if hand_wave > 0.5:
            wave_score = observation.signals.get("hand_wave_confidence", 0.8)
            candidates.append({"reason": "hand_wave", "score": wave_score, "source": "pose"})
            if trigger_reason is None or wave_score > trigger_score:
                trigger_reason = "hand_wave"
                trigger_score = wave_score

        # Check camera gaze (Phase 9)
        gaze_detected, gaze_score = self._detect_camera_gaze(observation)
        if gaze_detected:
            candidates.append({"reason": "camera_gaze", "score": gaze_score, "source": "face"})
            if trigger_reason is None or gaze_score > trigger_score:
                trigger_reason = "camera_gaze"
                trigger_score = gaze_score

        # Check passenger interaction (Phase 9)
        interact_detected, interact_score = self._detect_passenger_interaction(observation)
        if interact_detected:
            candidates.append({"reason": "passenger_interaction", "score": interact_score, "source": "face"})
            if trigger_reason is None or interact_score > trigger_score:
                trigger_reason = "passenger_interaction"
                trigger_score = interact_score

        # Check gestures (Phase 9)
        gesture_reason, gesture_score = self._detect_gestures(observation)
        if gesture_reason:
            candidates.append({"reason": gesture_reason, "score": gesture_score, "source": "gesture"})
            if trigger_reason is None or gesture_score > trigger_score:
                trigger_reason = gesture_reason
                trigger_score = gesture_score

        # 4. Consecutive frame counting
        if trigger_reason is not None:
            if self._pending_trigger_reason == trigger_reason:
                self._consecutive_high += 1
                self._pending_trigger_score = max(self._pending_trigger_score, trigger_score)
            else:
                self._consecutive_high = 1
                self._pending_trigger_reason = trigger_reason
                self._pending_trigger_score = trigger_score
        else:
            self._consecutive_high = 0
            self._pending_trigger_reason = None
            self._pending_trigger_score = 0.0

        # 5. Fire trigger if consecutive threshold met
        if self._consecutive_high >= self._consecutive_required:
            self._last_trigger_ns = t_ns
            reason = self._pending_trigger_reason
            score = self._pending_trigger_score
            face_count = int(observation.signals.get("face_count", 0))

            # Reset consecutive counter
            self._consecutive_high = 0
            self._pending_trigger_reason = None
            self._pending_trigger_score = 0.0

            # Find event start time
            event_t_ns = self._find_event_start(reason, t_ns)

            trigger = Trigger.point(
                event_time_ns=event_t_ns,
                pre_sec=self._pre_sec,
                post_sec=self._post_sec,
                label="highlight",
                score=score,
                metadata={
                    "reason": reason,
                    "face_count": face_count,
                },
            )

            # Emit trigger decision and fire records
            if _hub.enabled:
                _hub.emit(TriggerDecisionRecord(
                    frame_id=observation.frame_id,
                    t_ns=t_ns,
                    gate_open=True,
                    in_cooldown=False,
                    candidates=candidates,
                    consecutive_count=self._consecutive_required,
                    consecutive_required=self._consecutive_required,
                    decision="triggered",
                    ewma_values={fid: s.ewma for fid, s in self._expression_states.items()},
                    ewma_vars={fid: s.ewma_var for fid, s in self._expression_states.items()},
                ))
                _hub.emit(TriggerFireRecord(
                    frame_id=observation.frame_id,
                    t_ns=t_ns,
                    event_t_ns=event_t_ns,
                    reason=reason,
                    score=score,
                    pre_sec=self._pre_sec,
                    post_sec=self._post_sec,
                    face_count=face_count,
                    consecutive_frames=self._consecutive_required,
                ))

            return FusionResult(
                should_trigger=True,
                trigger=trigger,
                score=score,
                reason=reason,
                observations_used=self._observation_count,
                metadata={"consecutive_frames": self._consecutive_required},
            )

        # Emit trigger decision record (no trigger)
        if _hub.enabled and _hub.is_level_enabled(TraceLevel.NORMAL):
            decision = "no_trigger"
            if candidates:
                decision = "consecutive_pending"
            _hub.emit(TriggerDecisionRecord(
                frame_id=observation.frame_id,
                t_ns=t_ns,
                gate_open=True,
                in_cooldown=False,
                candidates=candidates,
                consecutive_count=self._consecutive_high,
                consecutive_required=self._consecutive_required,
                decision=decision,
                ewma_values={fid: s.ewma for fid, s in self._expression_states.items()}
                    if _hub.is_level_enabled(TraceLevel.VERBOSE) else {},
                ewma_vars={fid: s.ewma_var for fid, s in self._expression_states.items()}
                    if _hub.is_level_enabled(TraceLevel.VERBOSE) else {},
            ))

        return FusionResult(
            should_trigger=False,
            observations_used=self._observation_count,
            metadata={
                "state": "monitoring",
                "gate_open": self._gate_open,
                "consecutive_high": self._consecutive_high,
                "pending_reason": self._pending_trigger_reason,
            },
        )

    def _check_gate_conditions(self, observation: Observation) -> bool:
        """Check if gate conditions are met.

        Args:
            observation: Current observation.

        Returns:
            True if all gate conditions are satisfied.
        """
        faces = observation.faces
        face_count = int(observation.signals.get("face_count", len(faces)))

        # Track individual conditions for observability
        face_count_ok = 1 <= face_count <= 2
        quality_gate = observation.signals.get("quality_gate", 1.0)
        quality_ok = quality_gate >= 0.5

        # Initialize per-face checks
        confidence_ok = True
        inside_frame_ok = True
        yaw_ok = True
        pitch_ok = True
        area_ok = True
        center_ok = True

        max_confidence = 0.0
        max_yaw = 0.0
        max_pitch = 0.0

        # Check each face
        for face in faces:
            max_confidence = max(max_confidence, face.confidence)
            max_yaw = max(max_yaw, abs(face.yaw))
            max_pitch = max(max_pitch, abs(face.pitch))

            if face.confidence < self._face_conf_threshold:
                confidence_ok = False
            if not face.inside_frame:
                inside_frame_ok = False
            if abs(face.yaw) > self._yaw_max:
                yaw_ok = False
            if abs(face.pitch) > self._pitch_max:
                pitch_ok = False
            if face.area_ratio < self._min_face_area:
                area_ok = False
            if face.center_distance > self._max_center_dist:
                center_ok = False

        all_met = (
            face_count_ok and quality_ok and confidence_ok and
            inside_frame_ok and yaw_ok and pitch_ok and area_ok and center_ok
        )

        # Emit VERBOSE condition record
        if _hub.enabled and _hub.is_level_enabled(TraceLevel.VERBOSE):
            _hub.emit(GateConditionRecord(
                frame_id=observation.frame_id,
                gate_open=self._gate_open,
                face_count_ok=face_count_ok,
                confidence_ok=confidence_ok,
                yaw_ok=yaw_ok,
                pitch_ok=pitch_ok,
                inside_frame_ok=inside_frame_ok,
                area_ok=area_ok,
                center_ok=center_ok,
                quality_ok=quality_ok,
                face_count=face_count,
                max_confidence=max_confidence,
                max_yaw=max_yaw,
                max_pitch=max_pitch,
            ))

        return all_met

    def _update_gate_hysteresis(
        self, t_ns: int, conditions_met: bool, frame_id: int = 0
    ) -> None:
        """Update gate state with hysteresis.

        Args:
            t_ns: Current timestamp.
            conditions_met: Whether gate conditions are currently met.
            frame_id: Current frame ID for observability.
        """
        old_gate_open = self._gate_open

        if conditions_met:
            self._gate_condition_first_failed_ns = None

            if self._gate_open:
                # Already open, stay open
                pass
            else:
                # Track when conditions first met
                if self._gate_condition_first_met_ns is None:
                    self._gate_condition_first_met_ns = t_ns
                elif t_ns - self._gate_condition_first_met_ns >= self._gate_open_duration_ns:
                    # Conditions met long enough, open gate
                    self._gate_open = True
                    duration_ns = t_ns - self._gate_condition_first_met_ns
                    logger.debug(f"Gate opened at t={t_ns / 1e9:.3f}s")

                    # Emit gate change record
                    if _hub.enabled:
                        _hub.emit(GateChangeRecord(
                            frame_id=frame_id,
                            t_ns=t_ns,
                            old_state="closed",
                            new_state="open",
                            duration_ns=duration_ns,
                        ))
        else:
            self._gate_condition_first_met_ns = None

            if self._gate_open:
                # Track when conditions first failed
                if self._gate_condition_first_failed_ns is None:
                    self._gate_condition_first_failed_ns = t_ns
                elif t_ns - self._gate_condition_first_failed_ns >= self._gate_close_duration_ns:
                    # Conditions failed long enough, close gate
                    self._gate_open = False
                    duration_ns = t_ns - self._gate_condition_first_failed_ns
                    logger.debug(f"Gate closed at t={t_ns / 1e9:.3f}s")

                    # Emit gate change record
                    if _hub.enabled:
                        _hub.emit(GateChangeRecord(
                            frame_id=frame_id,
                            t_ns=t_ns,
                            old_state="open",
                            new_state="closed",
                            duration_ns=duration_ns,
                        ))

    def _detect_expression_spike(self, observation: Observation) -> Optional[float]:
        """Detect expression spikes using EWMA and z-score.

        Args:
            observation: Current observation.

        Returns:
            Spike score if detected, None otherwise.
        """
        max_spike = None

        for face in observation.faces:
            face_id = face.face_id
            expression = face.expression

            # Get or create expression state
            if face_id not in self._expression_states:
                self._expression_states[face_id] = ExpressionState()

            state = self._expression_states[face_id]

            # Update EWMA
            if state.count == 0:
                state.ewma = expression
                state.ewma_var = 0.01
            else:
                # Update mean
                delta = expression - state.ewma
                state.ewma = state.ewma + self._ewma_alpha * delta

                # Update variance (exponentially weighted)
                state.ewma_var = (1 - self._ewma_alpha) * (
                    state.ewma_var + self._ewma_alpha * delta * delta
                )

            state.count += 1

            # Compute z-score
            std = max(0.05, state.ewma_var ** 0.5)  # Min std to avoid division issues
            z_score = (expression - state.ewma) / std

            # Check for spike
            if z_score > self._expr_z_threshold and expression > 0.5:
                spike_score = min(1.0, z_score / (self._expr_z_threshold * 2))
                if max_spike is None or spike_score > max_spike:
                    max_spike = spike_score

        return max_spike

    def _detect_head_turn(self, observation: Observation) -> Optional[float]:
        """Detect head turn events.

        Args:
            observation: Current observation.

        Returns:
            Turn score if detected, None otherwise.
        """
        t_ns = observation.t_ns
        max_turn = None

        for face in observation.faces:
            face_id = face.face_id
            yaw = face.yaw

            if face_id in self._prev_yaw:
                prev_t_ns, prev_yaw = self._prev_yaw[face_id]
                dt_sec = (t_ns - prev_t_ns) / 1e9

                if dt_sec > 0 and dt_sec < 0.5:  # Reasonable time gap
                    # Compute angular velocity
                    angular_velocity = abs(yaw - prev_yaw) / dt_sec

                    if angular_velocity > self._head_turn_vel_threshold:
                        # Score based on velocity magnitude
                        turn_score = min(
                            1.0,
                            angular_velocity / (self._head_turn_vel_threshold * 2)
                        )
                        if max_turn is None or turn_score > max_turn:
                            max_turn = turn_score

            # Update previous yaw
            self._prev_yaw[face_id] = (t_ns, yaw)

        return max_turn

    def _detect_camera_gaze(self, observation: Observation) -> tuple[bool, float]:
        """Detect when subject is looking directly at camera.

        For gokart scenario, camera is mounted in front facing the driver.
        Looking at camera means yaw and pitch are close to 0.

        Args:
            observation: Current observation.

        Returns:
            Tuple of (detected, score).
        """
        if not observation.faces:
            return False, 0.0

        max_score = 0.0
        detected = False

        for face in observation.faces:
            # Calculate gaze score based on how centered the head pose is
            # yaw close to 0 = looking straight ahead at camera
            # pitch close to 0 = not looking up or down
            yaw_deviation = abs(face.yaw)
            pitch_deviation = abs(face.pitch)

            # Score decreases linearly as deviation increases
            yaw_score = max(0.0, 1.0 - yaw_deviation / self._gaze_yaw_threshold)
            pitch_score = max(0.0, 1.0 - pitch_deviation / self._gaze_pitch_threshold)

            # Combined score
            gaze_score = yaw_score * pitch_score

            if gaze_score > self._gaze_score_threshold:
                detected = True
                max_score = max(max_score, gaze_score)

        return detected, max_score

    def _detect_passenger_interaction(
        self, observation: Observation
    ) -> tuple[bool, float]:
        """Detect when two passengers are looking at each other.

        For gokart scenario with 2 passengers, detect when they turn
        to look at each other.

        Args:
            observation: Current observation.

        Returns:
            Tuple of (detected, score).
        """
        faces = observation.faces
        if len(faces) != 2:
            return False, 0.0

        f1, f2 = faces[0], faces[1]

        # Determine which face is on the left vs right
        # bbox[0] is normalized x coordinate (0=left, 1=right)
        if f1.bbox[0] < f2.bbox[0]:
            left_face, right_face = f1, f2
        else:
            left_face, right_face = f2, f1

        # Check if they're looking at each other:
        # - Left person should have positive yaw (looking right)
        # - Right person should have negative yaw (looking left)
        left_looking_right = left_face.yaw > self._interaction_yaw_threshold
        right_looking_left = right_face.yaw < -self._interaction_yaw_threshold

        if left_looking_right and right_looking_left:
            # Calculate interaction score based on yaw angles
            left_score = min(abs(left_face.yaw) / 45.0, 1.0)
            right_score = min(abs(right_face.yaw) / 45.0, 1.0)
            score = (left_score + right_score) / 2.0

            return True, score

        return False, 0.0

    def _detect_gestures(self, observation: Observation) -> tuple[Optional[str], float]:
        """Detect hand gestures from gesture extractor signals.

        Looks for gesture signals in the observation (from GestureExtractor).

        Args:
            observation: Current observation.

        Returns:
            Tuple of (trigger_reason, score) or (None, 0.0).
        """
        # Check for gesture signals from GestureExtractor
        gesture_detected = observation.signals.get("gesture_detected", 0.0)
        if gesture_detected < 0.5:
            return None, 0.0

        gesture_type = observation.metadata.get("gesture_type", "")
        gesture_confidence = observation.signals.get("gesture_confidence", 0.0)

        # Map gesture types to trigger reasons
        gesture_triggers = {
            "v_sign": "gesture_vsign",
            "thumbs_up": "gesture_thumbsup",
            "ok_sign": "gesture_ok",
            "open_palm": "gesture_openpalm",
        }

        if gesture_type in gesture_triggers and gesture_confidence > 0.5:
            return gesture_triggers[gesture_type], gesture_confidence

        return None, 0.0

    def _find_event_start(self, reason: str, current_t_ns: int) -> int:
        """Find the start time of the trigger event.

        Args:
            reason: Trigger reason.
            current_t_ns: Current timestamp.

        Returns:
            Event start timestamp in nanoseconds.
        """
        # Look back through recent observations to find event start
        observations = list(self._recent_observations)

        if not observations:
            return current_t_ns

        for obs in reversed(observations[:-self._consecutive_required]):
            if reason == "expression_spike":
                max_expr = obs.signals.get("max_expression", 0)
                if max_expr < 0.5:
                    return obs.t_ns
            elif reason == "head_turn":
                # Find where angular velocity was low
                break
            elif reason == "hand_wave":
                wave = obs.signals.get("hand_wave_detected", 0)
                if wave < 0.5:
                    return obs.t_ns
            elif reason == "camera_gaze":
                # Find where gaze started
                detected, _ = self._detect_camera_gaze(obs)
                if not detected:
                    return obs.t_ns
            elif reason == "passenger_interaction":
                # Find where interaction started
                detected, _ = self._detect_passenger_interaction(obs)
                if not detected:
                    return obs.t_ns
            elif reason.startswith("gesture_"):
                # Find where gesture started
                gesture = obs.signals.get("gesture_detected", 0)
                if gesture < 0.5:
                    return obs.t_ns

        # Default to a bit before current time
        return observations[-min(3, len(observations))].t_ns

    def _in_cooldown(self, t_ns: int) -> bool:
        """Check if currently in cooldown.

        Args:
            t_ns: Current timestamp.

        Returns:
            True if in cooldown period.
        """
        if self._last_trigger_ns is None:
            return False
        return (t_ns - self._last_trigger_ns) < self._cooldown_ns

    def reset(self) -> None:
        """Reset fusion state."""
        self._reset_state()

    @property
    def is_gate_open(self) -> bool:
        return self._gate_open

    @property
    def in_cooldown(self) -> bool:
        return self._last_trigger_ns is not None
