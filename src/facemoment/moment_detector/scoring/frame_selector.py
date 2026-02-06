"""Frame selector with diversity constraints.

Selects top K frames from scored frames while ensuring diversity
through time gap and similarity constraints.

Example:
    >>> selector = FrameSelector(min_time_gap_ms=200, max_frames=5)
    >>> scored_frames = [(frame1, score1), (frame2, score2), ...]
    >>> selected = selector.select(scored_frames)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import logging

from facemoment.moment_detector.scoring.frame_scorer import ScoreResult

logger = logging.getLogger(__name__)


@dataclass
class ScoredFrame:
    """A frame with its score result.

    Attributes:
        frame_id: Frame identifier.
        t_ns: Frame timestamp in nanoseconds.
        score_result: Scoring result from FrameScorer.
        observations: Original observations used for scoring.
        metadata: Additional metadata (e.g., pose keypoints for similarity).
    """

    frame_id: int
    t_ns: int
    score_result: ScoreResult
    observations: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Get total score."""
        return self.score_result.total_score

    @property
    def t_ms(self) -> float:
        """Get timestamp in milliseconds."""
        return self.t_ns / 1_000_000

    @property
    def t_sec(self) -> float:
        """Get timestamp in seconds."""
        return self.t_ns / 1_000_000_000


@dataclass
class SelectionConfig:
    """Configuration for frame selection.

    Attributes:
        max_frames: Maximum number of frames to select (default: 5).
        min_time_gap_ms: Minimum time gap between selected frames in ms (default: 200).
        min_score: Minimum score threshold for selection (default: 0.3).
        pose_similarity_threshold: Max pose similarity to consider frames different (default: 0.8).
        enable_pose_diversity: Whether to check pose similarity (default: False).
    """

    max_frames: int = 5
    min_time_gap_ms: float = 200.0
    min_score: float = 0.3
    pose_similarity_threshold: float = 0.8
    enable_pose_diversity: bool = False


class FrameSelector:
    """Selects best frames with diversity constraints.

    Ensures selected frames are:
    1. High scoring (above min_score)
    2. Temporally diverse (min_time_gap_ms apart)
    3. Optionally pose-diverse (different poses)

    Args:
        config: Selection configuration.

    Example:
        >>> selector = FrameSelector(SelectionConfig(max_frames=5, min_time_gap_ms=200))
        >>> selected = selector.select(scored_frames)
        >>> for frame in selected:
        ...     print(f"Frame {frame.frame_id}: {frame.score:.2f} @ {frame.t_sec:.2f}s")
    """

    def __init__(self, config: Optional[SelectionConfig] = None):
        self._config = config or SelectionConfig()

    def select(self, scored_frames: List[ScoredFrame]) -> List[ScoredFrame]:
        """Select top frames with diversity constraints.

        Args:
            scored_frames: List of scored frames to select from.

        Returns:
            List of selected frames, sorted by timestamp.
        """
        if not scored_frames:
            return []

        # Filter out low scores and filtered frames
        candidates = [
            f for f in scored_frames
            if not f.score_result.is_filtered and f.score >= self._config.min_score
        ]

        if not candidates:
            logger.debug("No candidates passed score threshold")
            return []

        # Sort by score descending
        candidates.sort(key=lambda f: f.score, reverse=True)

        # Greedy selection with diversity constraints
        selected: List[ScoredFrame] = []

        for candidate in candidates:
            if len(selected) >= self._config.max_frames:
                break

            # Check time gap constraint
            if not self._check_time_gap(candidate, selected):
                continue

            # Check pose diversity (optional)
            if self._config.enable_pose_diversity:
                if not self._check_pose_diversity(candidate, selected):
                    continue

            selected.append(candidate)

        # Sort by timestamp for output
        selected.sort(key=lambda f: f.t_ns)

        logger.debug(
            f"Selected {len(selected)} frames from {len(scored_frames)} "
            f"(candidates: {len(candidates)})"
        )

        return selected

    def _check_time_gap(
        self, candidate: ScoredFrame, selected: List[ScoredFrame]
    ) -> bool:
        """Check if candidate is far enough from all selected frames.

        Args:
            candidate: Frame to check.
            selected: Already selected frames.

        Returns:
            True if candidate passes time gap constraint.
        """
        if not selected:
            return True

        min_gap_ns = self._config.min_time_gap_ms * 1_000_000

        for frame in selected:
            gap = abs(candidate.t_ns - frame.t_ns)
            if gap < min_gap_ns:
                return False

        return True

    def _check_pose_diversity(
        self, candidate: ScoredFrame, selected: List[ScoredFrame]
    ) -> bool:
        """Check if candidate pose is different enough from selected frames.

        Uses keypoint positions to compute pose similarity.

        Args:
            candidate: Frame to check.
            selected: Already selected frames.

        Returns:
            True if candidate passes pose diversity constraint.
        """
        if not selected:
            return True

        candidate_pose = candidate.metadata.get("pose_keypoints")
        if candidate_pose is None:
            return True  # Can't check, allow

        for frame in selected:
            frame_pose = frame.metadata.get("pose_keypoints")
            if frame_pose is None:
                continue

            similarity = self._compute_pose_similarity(candidate_pose, frame_pose)
            if similarity > self._config.pose_similarity_threshold:
                return False

        return True

    def _compute_pose_similarity(
        self, pose1: List[List[float]], pose2: List[List[float]]
    ) -> float:
        """Compute similarity between two poses.

        Uses normalized keypoint distances.

        Args:
            pose1: First pose keypoints [[x, y, conf], ...].
            pose2: Second pose keypoints [[x, y, conf], ...].

        Returns:
            Similarity score [0, 1] where 1 = identical.
        """
        if len(pose1) != len(pose2):
            return 0.0

        total_dist = 0.0
        valid_count = 0

        for kp1, kp2 in zip(pose1, pose2):
            # Only compare if both have confidence
            if len(kp1) >= 3 and len(kp2) >= 3:
                if kp1[2] < 0.3 or kp2[2] < 0.3:
                    continue

            # Euclidean distance (normalized coords)
            dx = kp1[0] - kp2[0]
            dy = kp1[1] - kp2[1]
            dist = (dx * dx + dy * dy) ** 0.5
            total_dist += dist
            valid_count += 1

        if valid_count == 0:
            return 0.0

        avg_dist = total_dist / valid_count
        # Convert distance to similarity (0.5 distance = 0 similarity)
        similarity = max(0.0, 1.0 - avg_dist * 2)
        return similarity

    def select_from_window(
        self,
        scored_frames: List[ScoredFrame],
        window_start_ns: int,
        window_end_ns: int,
    ) -> List[ScoredFrame]:
        """Select frames from a specific time window.

        Useful for selecting best frames around a trigger event.

        Args:
            scored_frames: All scored frames.
            window_start_ns: Window start timestamp.
            window_end_ns: Window end timestamp.

        Returns:
            Selected frames within the window.
        """
        window_frames = [
            f for f in scored_frames
            if window_start_ns <= f.t_ns <= window_end_ns
        ]
        return self.select(window_frames)

    def select_around_triggers(
        self,
        scored_frames: List[ScoredFrame],
        trigger_times_ns: List[int],
        window_before_ms: float = 500.0,
        window_after_ms: float = 500.0,
        frames_per_trigger: int = 1,
    ) -> List[ScoredFrame]:
        """Select best frames around each trigger event.

        Args:
            scored_frames: All scored frames.
            trigger_times_ns: Trigger timestamps in nanoseconds.
            window_before_ms: Time window before trigger in ms.
            window_after_ms: Time window after trigger in ms.
            frames_per_trigger: Max frames to select per trigger.

        Returns:
            Selected frames around all triggers.
        """
        if not trigger_times_ns:
            return []

        # Create per-trigger config
        trigger_config = SelectionConfig(
            max_frames=frames_per_trigger,
            min_time_gap_ms=self._config.min_time_gap_ms,
            min_score=self._config.min_score,
        )
        trigger_selector = FrameSelector(trigger_config)

        all_selected: List[ScoredFrame] = []
        selected_ids = set()

        for trigger_ns in trigger_times_ns:
            window_start = trigger_ns - int(window_before_ms * 1_000_000)
            window_end = trigger_ns + int(window_after_ms * 1_000_000)

            window_frames = [
                f for f in scored_frames
                if window_start <= f.t_ns <= window_end
                and f.frame_id not in selected_ids
            ]

            selected = trigger_selector.select(window_frames)

            for frame in selected:
                if frame.frame_id not in selected_ids:
                    all_selected.append(frame)
                    selected_ids.add(frame.frame_id)

        # Sort by timestamp
        all_selected.sort(key=lambda f: f.t_ns)

        # Apply global max_frames limit
        if len(all_selected) > self._config.max_frames:
            # Re-sort by score and take top
            all_selected.sort(key=lambda f: f.score, reverse=True)
            all_selected = all_selected[: self._config.max_frames]
            all_selected.sort(key=lambda f: f.t_ns)

        return all_selected

    @property
    def config(self) -> SelectionConfig:
        """Get current selection configuration."""
        return self._config

    @config.setter
    def config(self, value: SelectionConfig) -> None:
        """Set selection configuration."""
        self._config = value
