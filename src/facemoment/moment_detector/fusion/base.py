"""Base fusion interface for combining observations (C module)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from visualbase import Trigger

from facemoment.moment_detector.extractors.base import Observation


@dataclass
class FusionResult:
    """Result from fusion module decision.

    Attributes:
        should_trigger: Whether a highlight trigger should fire.
        trigger: The trigger to send if should_trigger is True.
        score: Confidence/quality score [0, 1].
        reason: Primary reason for the trigger (e.g., "expression_spike").
        observations_used: Number of observations used in this decision.
        metadata: Additional metadata about the decision.
    """

    should_trigger: bool
    trigger: Optional[Trigger] = None
    score: float = 0.0
    reason: str = ""
    observations_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseFusion(ABC):
    """Abstract base class for fusion modules (C module).

    Fusion modules receive observations from multiple extractors,
    align them by timestamp, and decide when to fire triggers.

    The fusion module maintains state across frames to implement
    hysteresis, cooldown, and temporal smoothing.

    Example:
        >>> fusion = HighlightFusion(cooldown_sec=2.0)
        >>> for obs in observations:
        ...     result = fusion.update(obs)
        ...     if result.should_trigger:
        ...         handle_trigger(result.trigger)
    """

    @abstractmethod
    def update(self, observation: Observation) -> FusionResult:
        """Process a new observation and decide on trigger.

        Args:
            observation: New observation from an extractor.

        Returns:
            FusionResult indicating whether to trigger and with what parameters.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset fusion state.

        Call this when starting a new video or after a significant
        discontinuity in the stream.
        """
        ...

    @property
    @abstractmethod
    def is_gate_open(self) -> bool:
        """Whether the quality gate is currently open.

        The gate controls whether triggers can fire based on
        composition quality (face framing, angles, etc.).
        """
        ...

    @property
    @abstractmethod
    def in_cooldown(self) -> bool:
        """Whether the fusion is in cooldown after a recent trigger."""
        ...
