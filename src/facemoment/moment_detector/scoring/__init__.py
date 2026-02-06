"""Frame scoring module for selecting best frames.

Provides scoring and ranking of frames based on:
- Technical quality (blur, exposure, sharpness)
- Action/aesthetics (pose, expression, composition)
- Identity safety (face consistency)

Example:
    >>> from facemoment.moment_detector.scoring import FrameScorer, ScoringConfig
    >>>
    >>> scorer = FrameScorer(ScoringConfig(
    ...     weight_technical=0.45,
    ...     weight_action=0.35,
    ...     weight_identity=0.20,
    ... ))
    >>>
    >>> result = scorer.score(
    ...     face_obs=face_obs,
    ...     pose_obs=pose_obs,
    ...     quality_obs=quality_obs,
    ... )
    >>> print(f"Score: {result.total_score:.2f}")
    >>> if result.is_filtered:
    ...     print(f"Filtered: {result.filter_reason}")
"""

from facemoment.moment_detector.scoring.frame_scorer import (
    FrameScorer,
    ScoringConfig,
    ScoreResult,
    ScoreBreakdown,
)
from facemoment.moment_detector.scoring.frame_selector import (
    FrameSelector,
    SelectionConfig,
    ScoredFrame,
)

__all__ = [
    # Scoring
    "FrameScorer",
    "ScoringConfig",
    "ScoreResult",
    "ScoreBreakdown",
    # Selection
    "FrameSelector",
    "SelectionConfig",
    "ScoredFrame",
]
