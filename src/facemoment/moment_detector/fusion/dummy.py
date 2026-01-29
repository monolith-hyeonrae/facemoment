"""Dummy fusion for testing."""

from typing import Optional
from collections import deque

from visualbase import Trigger

from facemoment.moment_detector.extractors.base import Observation
from facemoment.moment_detector.fusion.base import BaseFusion, FusionResult


class DummyFusion(BaseFusion):
    """Dummy fusion that triggers on high expression values.

    Simple rule-based fusion for testing:
    - Gate opens when face count is 1-2 and faces are well-positioned
    - Triggers when max_expression exceeds threshold for N consecutive frames
    - Cooldown prevents rapid re-triggering

    Args:
        expression_threshold: Expression level to trigger (default: 0.7).
        consecutive_frames: Number of consecutive high frames needed (default: 3).
        cooldown_sec: Seconds to wait after trigger (default: 2.0).
        trigger_duration_sec: Duration of generated clips (default: 5.0).
        pre_sec: Seconds before event in clip (default: 2.0).
        post_sec: Seconds after event in clip (default: 2.0).

    Example:
        >>> fusion = DummyFusion(expression_threshold=0.8)
        >>> result = fusion.update(observation)
        >>> if result.should_trigger:
        ...     print(f"Trigger! Score: {result.score}")
    """

    def __init__(
        self,
        expression_threshold: float = 0.7,
        consecutive_frames: int = 3,
        cooldown_sec: float = 2.0,
        trigger_duration_sec: float = 5.0,
        pre_sec: float = 2.0,
        post_sec: float = 2.0,
    ):
        self._threshold = expression_threshold
        self._consecutive_required = consecutive_frames
        self._cooldown_ns = int(cooldown_sec * 1_000_000_000)
        self._trigger_duration_sec = trigger_duration_sec
        self._pre_sec = pre_sec
        self._post_sec = post_sec

        # State
        self._recent_observations: deque[Observation] = deque(maxlen=10)
        self._consecutive_high: int = 0
        self._last_trigger_ns: Optional[int] = None
        self._gate_open: bool = False
        self._observation_count: int = 0

    def update(self, observation: Observation) -> FusionResult:
        """Process observation and decide on trigger."""
        self._recent_observations.append(observation)
        self._observation_count += 1

        current_t_ns = observation.t_ns

        # Check cooldown
        if self._last_trigger_ns is not None:
            if current_t_ns - self._last_trigger_ns < self._cooldown_ns:
                return FusionResult(
                    should_trigger=False,
                    observations_used=self._observation_count,
                    metadata={"state": "cooldown"},
                )

        # Check gate (quality conditions)
        self._gate_open = self._check_gate(observation)
        if not self._gate_open:
            self._consecutive_high = 0
            return FusionResult(
                should_trigger=False,
                observations_used=self._observation_count,
                metadata={"state": "gate_closed"},
            )

        # Check expression threshold
        max_expr = observation.signals.get("max_expression", 0.0)

        if max_expr >= self._threshold:
            self._consecutive_high += 1
        else:
            self._consecutive_high = 0

        # Trigger if consecutive threshold met
        if self._consecutive_high >= self._consecutive_required:
            self._last_trigger_ns = current_t_ns
            self._consecutive_high = 0

            # Find the frame that first crossed threshold
            first_high_t_ns = current_t_ns
            for obs in reversed(list(self._recent_observations)):
                if obs.signals.get("max_expression", 0) >= self._threshold:
                    first_high_t_ns = obs.t_ns
                else:
                    break

            # Create trigger
            trigger = Trigger.point(
                event_time_ns=first_high_t_ns,
                pre_sec=self._pre_sec,
                post_sec=self._post_sec + self._trigger_duration_sec,
                label="highlight",
                score=max_expr,
                metadata={
                    "reason": "expression_spike",
                    "face_count": int(observation.signals.get("face_count", 0)),
                },
            )

            return FusionResult(
                should_trigger=True,
                trigger=trigger,
                score=max_expr,
                reason="expression_spike",
                observations_used=self._observation_count,
                metadata={"consecutive_frames": self._consecutive_required},
            )

        return FusionResult(
            should_trigger=False,
            observations_used=self._observation_count,
            metadata={
                "state": "monitoring",
                "consecutive_high": self._consecutive_high,
                "max_expression": max_expr,
            },
        )

    def _check_gate(self, observation: Observation) -> bool:
        """Check if quality gate conditions are met."""
        face_count = int(observation.signals.get("face_count", 0))

        # Must have 1 or 2 faces
        if face_count < 1 or face_count > 2:
            return False

        # Check each face is well-positioned
        for face in observation.faces:
            # Must be inside frame
            if not face.inside_frame:
                return False

            # Confidence check
            if face.confidence < 0.7:
                return False

            # Angle checks (simplified)
            if abs(face.yaw) > 25:
                return False
            if abs(face.pitch) > 20:
                return False

        return True

    def reset(self) -> None:
        """Reset fusion state."""
        self._recent_observations.clear()
        self._consecutive_high = 0
        self._last_trigger_ns = None
        self._gate_open = False
        self._observation_count = 0

    @property
    def is_gate_open(self) -> bool:
        return self._gate_open

    @property
    def in_cooldown(self) -> bool:
        # Can't determine without current time; return False if no trigger yet
        return self._last_trigger_ns is not None
