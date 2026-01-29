"""Highlight fusion for detecting photo-worthy moments."""

from typing import Optional, Dict, List
from collections import deque
from dataclasses import dataclass
import logging

from visualbase import Trigger

from facemoment.moment_detector.extractors.base import Observation, FaceObservation
from facemoment.moment_detector.fusion.base import BaseFusion, FusionResult

logger = logging.getLogger(__name__)


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
            return FusionResult(
                should_trigger=False,
                observations_used=self._observation_count,
                metadata={"state": "cooldown"},
            )

        # 2. Update gate with hysteresis
        gate_conditions_met = self._check_gate_conditions(observation)
        self._update_gate_hysteresis(t_ns, gate_conditions_met)

        if not self._gate_open:
            self._consecutive_high = 0
            return FusionResult(
                should_trigger=False,
                observations_used=self._observation_count,
                metadata={"state": "gate_closed", "conditions_met": gate_conditions_met},
            )

        # 3. Detect trigger events
        trigger_reason = None
        trigger_score = 0.0

        # Check expression spikes
        expr_spike = self._detect_expression_spike(observation)
        if expr_spike is not None:
            trigger_reason = "expression_spike"
            trigger_score = expr_spike

        # Check head turns
        head_turn = self._detect_head_turn(observation)
        if head_turn is not None and (trigger_reason is None or head_turn > trigger_score):
            trigger_reason = "head_turn"
            trigger_score = head_turn

        # Check hand waves (from pose extractor)
        hand_wave = observation.signals.get("hand_wave_detected", 0.0)
        if hand_wave > 0.5 and (trigger_reason is None or hand_wave > trigger_score):
            trigger_reason = "hand_wave"
            trigger_score = observation.signals.get("hand_wave_confidence", 0.8)

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
                    "face_count": int(observation.signals.get("face_count", 0)),
                },
            )

            return FusionResult(
                should_trigger=True,
                trigger=trigger,
                score=score,
                reason=reason,
                observations_used=self._observation_count,
                metadata={"consecutive_frames": self._consecutive_required},
            )

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

        # Must have 1-2 faces
        if face_count < 1 or face_count > 2:
            return False

        # Check quality gate if available
        quality_gate = observation.signals.get("quality_gate", 1.0)
        if quality_gate < 0.5:
            return False

        # Check each face
        for face in faces:
            # Confidence check
            if face.confidence < self._face_conf_threshold:
                return False

            # Must be inside frame
            if not face.inside_frame:
                return False

            # Angle checks
            if abs(face.yaw) > self._yaw_max:
                return False
            if abs(face.pitch) > self._pitch_max:
                return False

            # Size check
            if face.area_ratio < self._min_face_area:
                return False

            # Position check
            if face.center_distance > self._max_center_dist:
                return False

        return True

    def _update_gate_hysteresis(self, t_ns: int, conditions_met: bool) -> None:
        """Update gate state with hysteresis.

        Args:
            t_ns: Current timestamp.
            conditions_met: Whether gate conditions are currently met.
        """
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
                    logger.debug(f"Gate opened at t={t_ns / 1e9:.3f}s")
        else:
            self._gate_condition_first_met_ns = None

            if self._gate_open:
                # Track when conditions first failed
                if self._gate_condition_first_failed_ns is None:
                    self._gate_condition_first_failed_ns = t_ns
                elif t_ns - self._gate_condition_first_failed_ns >= self._gate_close_duration_ns:
                    # Conditions failed long enough, close gate
                    self._gate_open = False
                    logger.debug(f"Gate closed at t={t_ns / 1e9:.3f}s")

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
