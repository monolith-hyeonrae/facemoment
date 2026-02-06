"""Frame scorer for ranking frames by quality and aesthetics.

Combines multiple scoring components with configurable weights to produce
a final score for each frame. Supports hard filters for disqualification.

Scoring Components:
    - Technical (0.45): blur, exposure, sharpness, occlusion
    - Action (0.35): pose energy, expression, composition
    - Identity (0.20): face consistency, stability

Example:
    >>> scorer = FrameScorer()
    >>> result = scorer.score(face_obs=face_obs, quality_obs=quality_obs)
    >>> if not result.is_filtered:
    ...     print(f"Frame score: {result.total_score:.2f}")
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import logging

from facemoment.moment_detector.extractors.base import Observation

logger = logging.getLogger(__name__)


@dataclass
class ScoringConfig:
    """Configuration for frame scoring weights and thresholds.

    Attributes:
        weight_technical: Weight for technical quality score (default: 0.45).
        weight_action: Weight for action/aesthetics score (default: 0.35).
        weight_identity: Weight for identity safety score (default: 0.20).
        enable_hard_filters: Whether to apply hard filters (default: True).
        min_blur_score: Minimum blur score to pass filter (default: 30).
        min_face_confidence: Minimum face confidence (default: 0.5).
        max_head_yaw: Maximum head yaw angle for frontal preference (default: 45).
        max_head_pitch: Maximum head pitch angle (default: 30).
    """

    # Component weights (should sum to 1.0)
    weight_technical: float = 0.45
    weight_action: float = 0.35
    weight_identity: float = 0.20

    # Hard filter settings
    enable_hard_filters: bool = True
    min_blur_score: float = 30.0
    min_face_confidence: float = 0.5
    max_head_yaw: float = 45.0
    max_head_pitch: float = 30.0

    # Technical quality thresholds
    optimal_brightness_min: float = 80.0
    optimal_brightness_max: float = 180.0
    min_contrast: float = 30.0

    # Action/aesthetics thresholds
    frontal_yaw_bonus: float = 25.0  # Yaw within this gets bonus
    expression_boost_threshold: float = 0.3  # Expression intensity for boost


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of score components.

    Attributes:
        technical: Technical quality sub-scores.
        action: Action/aesthetics sub-scores.
        identity: Identity safety sub-scores.
    """

    technical: Dict[str, float] = field(default_factory=dict)
    action: Dict[str, float] = field(default_factory=dict)
    identity: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Convert to dictionary."""
        return {
            "technical": self.technical,
            "action": self.action,
            "identity": self.identity,
        }


@dataclass
class ScoreResult:
    """Result of frame scoring.

    Attributes:
        total_score: Final combined score [0, 1].
        technical_score: Technical quality score [0, 1].
        action_score: Action/aesthetics score [0, 1].
        identity_score: Identity safety score [0, 1].
        is_filtered: Whether frame was filtered out.
        filter_reason: Reason for filtering (if filtered).
        breakdown: Detailed score breakdown.
    """

    total_score: float
    technical_score: float
    action_score: float
    identity_score: float
    is_filtered: bool = False
    filter_reason: Optional[str] = None
    breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_score": self.total_score,
            "technical_score": self.technical_score,
            "action_score": self.action_score,
            "identity_score": self.identity_score,
            "is_filtered": self.is_filtered,
            "filter_reason": self.filter_reason,
            "breakdown": self.breakdown.to_dict(),
        }


# Type alias for filter function
FilterFunc = Callable[[Optional[Observation], Optional[Observation], Optional[Observation]], Tuple[bool, str]]


class FrameScorer:
    """Scores frames based on quality, aesthetics, and identity.

    Combines multiple scoring components with configurable weights.
    Supports hard filters for automatic disqualification.

    Args:
        config: Scoring configuration (default: ScoringConfig()).

    Example:
        >>> scorer = FrameScorer(ScoringConfig(weight_technical=0.5))
        >>> result = scorer.score(face_obs=face_obs, quality_obs=quality_obs)
        >>> print(f"Score: {result.total_score:.2f}")
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self._config = config or ScoringConfig()
        self._hard_filters: List[Tuple[str, FilterFunc]] = self._build_hard_filters()

    def _build_hard_filters(self) -> List[Tuple[str, FilterFunc]]:
        """Build list of hard filter functions."""
        filters = []

        # No face detected
        def no_face_filter(face_obs, pose_obs, quality_obs):
            if face_obs is None:
                return True, "no_face_obs"
            face_count = face_obs.signals.get("face_count", 0)
            if face_count == 0:
                return True, "no_face_detected"
            return False, ""

        filters.append(("no_face", no_face_filter))

        # Severe blur
        def blur_filter(face_obs, pose_obs, quality_obs):
            if quality_obs is None:
                return False, ""  # Can't check, don't filter
            blur_score = quality_obs.signals.get("blur_score", 100)
            if blur_score < self._config.min_blur_score:
                return True, f"severe_blur ({blur_score:.1f} < {self._config.min_blur_score})"
            return False, ""

        filters.append(("blur", blur_filter))

        # Low face confidence
        def confidence_filter(face_obs, pose_obs, quality_obs):
            if face_obs is None or not face_obs.faces:
                return False, ""  # Handled by no_face filter
            main_face = face_obs.faces[0]
            if main_face.confidence < self._config.min_face_confidence:
                return True, f"low_confidence ({main_face.confidence:.2f} < {self._config.min_face_confidence})"
            return False, ""

        filters.append(("confidence", confidence_filter))

        # Head cut-off (face not fully inside frame)
        def cutoff_filter(face_obs, pose_obs, quality_obs):
            if face_obs is None or not face_obs.faces:
                return False, ""
            main_face = face_obs.faces[0]
            if not main_face.inside_frame:
                return True, "head_cutoff"
            return False, ""

        filters.append(("cutoff", cutoff_filter))

        return filters

    def score(
        self,
        face_obs: Optional[Observation] = None,
        pose_obs: Optional[Observation] = None,
        quality_obs: Optional[Observation] = None,
        classifier_obs: Optional[Observation] = None,
    ) -> ScoreResult:
        """Score a frame based on available observations.

        Args:
            face_obs: Face detection observation.
            pose_obs: Pose estimation observation.
            quality_obs: Quality analysis observation.
            classifier_obs: Face classifier observation (for main face).

        Returns:
            ScoreResult with total score and breakdown.
        """
        breakdown = ScoreBreakdown()

        # Apply hard filters first
        if self._config.enable_hard_filters:
            for filter_name, filter_func in self._hard_filters:
                is_filtered, reason = filter_func(face_obs, pose_obs, quality_obs)
                if is_filtered:
                    return ScoreResult(
                        total_score=0.0,
                        technical_score=0.0,
                        action_score=0.0,
                        identity_score=0.0,
                        is_filtered=True,
                        filter_reason=reason,
                        breakdown=breakdown,
                    )

        # Calculate component scores
        technical_score, breakdown.technical = self._score_technical(
            face_obs, quality_obs
        )
        action_score, breakdown.action = self._score_action(
            face_obs, pose_obs, classifier_obs
        )
        identity_score, breakdown.identity = self._score_identity(
            face_obs, classifier_obs
        )

        # Weighted sum
        total_score = (
            self._config.weight_technical * technical_score
            + self._config.weight_action * action_score
            + self._config.weight_identity * identity_score
        )

        return ScoreResult(
            total_score=total_score,
            technical_score=technical_score,
            action_score=action_score,
            identity_score=identity_score,
            is_filtered=False,
            filter_reason=None,
            breakdown=breakdown,
        )

    def _score_technical(
        self,
        face_obs: Optional[Observation],
        quality_obs: Optional[Observation],
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate technical quality score.

        Components:
            - blur_score: Sharpness (Laplacian variance)
            - brightness_score: Optimal exposure
            - contrast_score: Image contrast

        Returns:
            Tuple of (score [0,1], breakdown dict).
        """
        breakdown = {}
        scores = []

        # Blur/sharpness score
        if quality_obs is not None:
            blur = quality_obs.signals.get("blur_score", 0)
            # Normalize: 0-200 range to 0-1, with diminishing returns above 100
            blur_score = min(1.0, blur / 100.0)
            breakdown["blur"] = blur_score
            scores.append(blur_score)

            # Brightness score (penalize over/under exposure)
            brightness = quality_obs.signals.get("brightness", 128)
            if brightness < self._config.optimal_brightness_min:
                # Under-exposed
                brightness_score = brightness / self._config.optimal_brightness_min
            elif brightness > self._config.optimal_brightness_max:
                # Over-exposed
                brightness_score = max(
                    0, 1 - (brightness - self._config.optimal_brightness_max) / 75
                )
            else:
                # Optimal range
                brightness_score = 1.0
            breakdown["brightness"] = brightness_score
            scores.append(brightness_score)

            # Contrast score
            contrast = quality_obs.signals.get("contrast", 0)
            contrast_score = min(1.0, contrast / 60.0)  # 60+ is good contrast
            breakdown["contrast"] = contrast_score
            scores.append(contrast_score)

        # Face region quality (if available)
        if face_obs is not None and face_obs.faces:
            main_face = face_obs.faces[0]
            # Confidence as quality proxy
            conf_score = main_face.confidence
            breakdown["face_confidence"] = conf_score
            scores.append(conf_score)

        if not scores:
            return 0.5, breakdown  # Neutral score if no data

        return sum(scores) / len(scores), breakdown

    def _score_action(
        self,
        face_obs: Optional[Observation],
        pose_obs: Optional[Observation],
        classifier_obs: Optional[Observation],
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate action/aesthetics score.

        Components:
            - face_direction: Frontal/45° preference
            - expression: Expression intensity
            - composition: Centering, margins

        Returns:
            Tuple of (score [0,1], breakdown dict).
        """
        breakdown = {}
        scores = []

        if face_obs is not None and face_obs.faces:
            main_face = face_obs.faces[0]

            # Face direction score (prefer frontal or slight angle)
            yaw = abs(main_face.yaw) if main_face.yaw else 0
            pitch = abs(main_face.pitch) if main_face.pitch else 0

            # Yaw: 0-25° is ideal, penalize beyond
            if yaw <= self._config.frontal_yaw_bonus:
                yaw_score = 1.0
            elif yaw <= self._config.max_head_yaw:
                yaw_score = 1.0 - (yaw - self._config.frontal_yaw_bonus) / (
                    self._config.max_head_yaw - self._config.frontal_yaw_bonus
                )
            else:
                yaw_score = 0.3  # Heavy penalty for extreme angles

            # Pitch: penalize looking up/down too much
            pitch_score = max(0.3, 1.0 - pitch / self._config.max_head_pitch)

            direction_score = (yaw_score + pitch_score) / 2
            breakdown["face_direction"] = direction_score
            scores.append(direction_score)

            # Expression score (interesting expressions are better)
            expression = main_face.expression if main_face.expression else 0
            happy = main_face.signals.get("em_happy", 0)
            neutral = main_face.signals.get("em_neutral", 1)

            # Boost for non-neutral expressions
            if expression > self._config.expression_boost_threshold:
                expr_score = 0.7 + 0.3 * min(1.0, expression)
            elif happy > 0.3:
                expr_score = 0.8 + 0.2 * happy
            else:
                expr_score = 0.5 + 0.3 * (1 - neutral)  # Less neutral = better

            breakdown["expression"] = expr_score
            scores.append(expr_score)

            # Composition score (centering)
            center_dist = main_face.center_distance
            # 0 = center, 0.5 = corner. Prefer center.
            composition_score = max(0.3, 1.0 - center_dist * 1.5)
            breakdown["composition"] = composition_score
            scores.append(composition_score)

        # Pose energy (if available)
        if pose_obs is not None:
            # For now, just check if pose is detected
            person_count = pose_obs.signals.get("person_count", 0)
            hands_raised = pose_obs.signals.get("hands_raised_count", 0)

            pose_score = 0.5  # Base score
            if person_count > 0:
                pose_score = 0.6
            if hands_raised > 0:
                pose_score = 0.9  # Bonus for raised hands

            breakdown["pose_energy"] = pose_score
            scores.append(pose_score)

        if not scores:
            return 0.5, breakdown

        return sum(scores) / len(scores), breakdown

    def _score_identity(
        self,
        face_obs: Optional[Observation],
        classifier_obs: Optional[Observation],
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate identity safety score.

        Components:
            - face_stability: Consistent face detection
            - role_confidence: Main face identification confidence

        Returns:
            Tuple of (score [0,1], breakdown dict).
        """
        breakdown = {}
        scores = []

        if face_obs is not None and face_obs.faces:
            main_face = face_obs.faces[0]

            # Face detection stability (based on confidence)
            stability_score = main_face.confidence
            breakdown["face_stability"] = stability_score
            scores.append(stability_score)

            # Inside frame bonus
            inside_score = 1.0 if main_face.inside_frame else 0.5
            breakdown["inside_frame"] = inside_score
            scores.append(inside_score)

        # Classifier-based scoring
        if classifier_obs is not None and classifier_obs.data:
            data = classifier_obs.data
            if hasattr(data, "main_face") and data.main_face:
                # Main face identified - good for identity
                main_conf = data.main_face.face.confidence
                track_len = data.main_face.track_length

                # Longer track = more stable identity
                track_score = min(1.0, track_len / 10.0)
                breakdown["track_stability"] = track_score
                scores.append(track_score)

                # Main face confidence
                breakdown["main_confidence"] = main_conf
                scores.append(main_conf)

        if not scores:
            return 0.5, breakdown

        return sum(scores) / len(scores), breakdown

    def add_filter(self, name: str, filter_func: FilterFunc) -> None:
        """Add a custom hard filter.

        Args:
            name: Filter name for identification.
            filter_func: Function(face_obs, pose_obs, quality_obs) -> (is_filtered, reason).
        """
        self._hard_filters.append((name, filter_func))

    def remove_filter(self, name: str) -> bool:
        """Remove a hard filter by name.

        Args:
            name: Filter name to remove.

        Returns:
            True if filter was found and removed.
        """
        for i, (filter_name, _) in enumerate(self._hard_filters):
            if filter_name == name:
                self._hard_filters.pop(i)
                return True
        return False

    @property
    def config(self) -> ScoringConfig:
        """Get current scoring configuration."""
        return self._config

    @config.setter
    def config(self, value: ScoringConfig) -> None:
        """Set scoring configuration."""
        self._config = value
