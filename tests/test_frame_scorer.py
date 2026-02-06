"""Tests for FrameScorer and FrameSelector."""

import pytest
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from facemoment.moment_detector.scoring import (
    FrameScorer,
    ScoringConfig,
    ScoreResult,
    FrameSelector,
    SelectionConfig,
    ScoredFrame,
)
from facemoment.moment_detector.extractors.base import Observation, FaceObservation


def make_face_obs(
    face_count: int = 1,
    confidence: float = 0.9,
    yaw: float = 0.0,
    pitch: float = 0.0,
    inside_frame: bool = True,
    expression: float = 0.5,
    center_distance: float = 0.1,
    signals: Optional[Dict] = None,
) -> Observation:
    """Create a mock face observation."""
    faces = []
    for i in range(face_count):
        faces.append(
            FaceObservation(
                face_id=i,
                confidence=confidence,
                bbox=(0.3, 0.2, 0.4, 0.5),
                inside_frame=inside_frame,
                yaw=yaw,
                pitch=pitch,
                roll=0.0,
                area_ratio=0.1,
                center_distance=center_distance,
                expression=expression,
                signals=signals or {"em_happy": 0.3, "em_neutral": 0.5},
            )
        )

    return Observation(
        source="face",
        frame_id=0,
        t_ns=0,
        signals={"face_count": face_count, **(signals or {})},
        faces=faces,
    )


def make_quality_obs(
    blur_score: float = 100.0,
    brightness: float = 128.0,
    contrast: float = 50.0,
) -> Observation:
    """Create a mock quality observation."""
    return Observation(
        source="quality",
        frame_id=0,
        t_ns=0,
        signals={
            "blur_score": blur_score,
            "brightness": brightness,
            "contrast": contrast,
        },
    )


def make_pose_obs(
    person_count: int = 1,
    hands_raised_count: int = 0,
) -> Observation:
    """Create a mock pose observation."""
    return Observation(
        source="pose",
        frame_id=0,
        t_ns=0,
        signals={
            "person_count": person_count,
            "hands_raised_count": hands_raised_count,
        },
    )


class TestFrameScorer:
    """Tests for FrameScorer class."""

    def test_default_config(self):
        """Test scorer with default configuration."""
        scorer = FrameScorer()
        assert scorer.config.weight_technical == 0.45
        assert scorer.config.weight_action == 0.35
        assert scorer.config.weight_identity == 0.20

    def test_custom_config(self):
        """Test scorer with custom configuration."""
        config = ScoringConfig(
            weight_technical=0.5,
            weight_action=0.3,
            weight_identity=0.2,
        )
        scorer = FrameScorer(config)
        assert scorer.config.weight_technical == 0.5

    def test_score_good_frame(self):
        """Test scoring a high-quality frame."""
        scorer = FrameScorer()

        face_obs = make_face_obs(
            confidence=0.95,
            yaw=5.0,
            pitch=0.0,
            inside_frame=True,
            expression=0.6,
            center_distance=0.05,
        )
        quality_obs = make_quality_obs(
            blur_score=150.0,
            brightness=120.0,
            contrast=60.0,
        )

        result = scorer.score(face_obs=face_obs, quality_obs=quality_obs)

        assert not result.is_filtered
        assert result.total_score > 0.7  # Should be a high score
        assert result.technical_score > 0.8
        assert result.action_score > 0.6

    def test_hard_filter_no_face(self):
        """Test hard filter for no face detected."""
        scorer = FrameScorer()

        face_obs = make_face_obs(face_count=0)
        quality_obs = make_quality_obs()

        result = scorer.score(face_obs=face_obs, quality_obs=quality_obs)

        assert result.is_filtered
        assert "no_face" in result.filter_reason

    def test_hard_filter_severe_blur(self):
        """Test hard filter for severe blur."""
        scorer = FrameScorer()

        face_obs = make_face_obs()
        quality_obs = make_quality_obs(blur_score=10.0)  # Very blurry

        result = scorer.score(face_obs=face_obs, quality_obs=quality_obs)

        assert result.is_filtered
        assert "blur" in result.filter_reason

    def test_hard_filter_low_confidence(self):
        """Test hard filter for low face confidence."""
        scorer = FrameScorer()

        face_obs = make_face_obs(confidence=0.3)  # Low confidence
        quality_obs = make_quality_obs()

        result = scorer.score(face_obs=face_obs, quality_obs=quality_obs)

        assert result.is_filtered
        assert "confidence" in result.filter_reason

    def test_hard_filter_head_cutoff(self):
        """Test hard filter for head cut-off."""
        scorer = FrameScorer()

        face_obs = make_face_obs(inside_frame=False)
        quality_obs = make_quality_obs()

        result = scorer.score(face_obs=face_obs, quality_obs=quality_obs)

        assert result.is_filtered
        assert "cutoff" in result.filter_reason

    def test_disable_hard_filters(self):
        """Test disabling hard filters."""
        config = ScoringConfig(enable_hard_filters=False)
        scorer = FrameScorer(config)

        face_obs = make_face_obs(confidence=0.3)  # Would normally filter
        quality_obs = make_quality_obs()

        result = scorer.score(face_obs=face_obs, quality_obs=quality_obs)

        assert not result.is_filtered  # Filters disabled
        assert result.total_score > 0  # Still gets a score

    def test_technical_score_blur(self):
        """Test technical score is affected by blur."""
        scorer = FrameScorer()

        face_obs = make_face_obs()
        sharp_quality = make_quality_obs(blur_score=150.0)
        blurry_quality = make_quality_obs(blur_score=50.0)

        sharp_result = scorer.score(face_obs=face_obs, quality_obs=sharp_quality)
        blurry_result = scorer.score(face_obs=face_obs, quality_obs=blurry_quality)

        assert sharp_result.technical_score > blurry_result.technical_score

    def test_action_score_face_direction(self):
        """Test action score prefers frontal faces."""
        scorer = FrameScorer()
        quality_obs = make_quality_obs()

        frontal = make_face_obs(yaw=0.0, pitch=0.0)
        angled = make_face_obs(yaw=40.0, pitch=20.0)

        frontal_result = scorer.score(face_obs=frontal, quality_obs=quality_obs)
        angled_result = scorer.score(face_obs=angled, quality_obs=quality_obs)

        assert frontal_result.action_score > angled_result.action_score

    def test_action_score_expression(self):
        """Test action score is boosted by expression."""
        scorer = FrameScorer()
        quality_obs = make_quality_obs()

        expressive = make_face_obs(expression=0.8, signals={"em_happy": 0.7, "em_neutral": 0.1})
        neutral = make_face_obs(expression=0.1, signals={"em_happy": 0.0, "em_neutral": 0.9})

        expr_result = scorer.score(face_obs=expressive, quality_obs=quality_obs)
        neutral_result = scorer.score(face_obs=neutral, quality_obs=quality_obs)

        assert expr_result.breakdown.action["expression"] > neutral_result.breakdown.action["expression"]

    def test_action_score_pose_hands_raised(self):
        """Test action score is boosted by raised hands."""
        scorer = FrameScorer()

        face_obs = make_face_obs()
        quality_obs = make_quality_obs()

        hands_up = make_pose_obs(hands_raised_count=1)
        hands_down = make_pose_obs(hands_raised_count=0)

        up_result = scorer.score(face_obs=face_obs, quality_obs=quality_obs, pose_obs=hands_up)
        down_result = scorer.score(face_obs=face_obs, quality_obs=quality_obs, pose_obs=hands_down)

        assert up_result.breakdown.action["pose_energy"] > down_result.breakdown.action["pose_energy"]

    def test_identity_score_inside_frame(self):
        """Test identity score is affected by inside_frame."""
        scorer = FrameScorer()
        quality_obs = make_quality_obs()

        inside = make_face_obs(inside_frame=True)
        outside = make_face_obs(inside_frame=False)

        # Disable filters to test scoring
        scorer._config.enable_hard_filters = False

        inside_result = scorer.score(face_obs=inside, quality_obs=quality_obs)
        outside_result = scorer.score(face_obs=outside, quality_obs=quality_obs)

        assert inside_result.identity_score > outside_result.identity_score

    def test_score_breakdown(self):
        """Test score breakdown is populated."""
        scorer = FrameScorer()

        face_obs = make_face_obs()
        quality_obs = make_quality_obs()
        pose_obs = make_pose_obs()

        result = scorer.score(
            face_obs=face_obs,
            quality_obs=quality_obs,
            pose_obs=pose_obs,
        )

        # Check breakdown has expected keys
        assert "blur" in result.breakdown.technical
        assert "brightness" in result.breakdown.technical
        assert "face_direction" in result.breakdown.action
        assert "expression" in result.breakdown.action
        assert "face_stability" in result.breakdown.identity

    def test_add_custom_filter(self):
        """Test adding custom filter."""
        scorer = FrameScorer()

        # Add filter that rejects frames with low expression
        def low_expression_filter(face_obs, pose_obs, quality_obs):
            if face_obs and face_obs.faces:
                if face_obs.faces[0].expression < 0.2:
                    return True, "low_expression"
            return False, ""

        scorer.add_filter("low_expression", low_expression_filter)

        face_obs = make_face_obs(expression=0.1)
        quality_obs = make_quality_obs()

        result = scorer.score(face_obs=face_obs, quality_obs=quality_obs)

        assert result.is_filtered
        assert "low_expression" in result.filter_reason

    def test_remove_filter(self):
        """Test removing a filter."""
        scorer = FrameScorer()

        # Remove blur filter
        removed = scorer.remove_filter("blur")
        assert removed

        # Now blurry frame should not be filtered
        face_obs = make_face_obs()
        quality_obs = make_quality_obs(blur_score=10.0)

        result = scorer.score(face_obs=face_obs, quality_obs=quality_obs)

        assert not result.is_filtered or "blur" not in result.filter_reason

    def test_score_no_observations(self):
        """Test scoring with no observations."""
        scorer = FrameScorer()

        result = scorer.score()

        assert result.is_filtered
        assert "no_face" in result.filter_reason

    def test_to_dict(self):
        """Test ScoreResult.to_dict()."""
        scorer = FrameScorer()

        face_obs = make_face_obs()
        quality_obs = make_quality_obs()

        result = scorer.score(face_obs=face_obs, quality_obs=quality_obs)
        result_dict = result.to_dict()

        assert "total_score" in result_dict
        assert "breakdown" in result_dict
        assert "technical" in result_dict["breakdown"]


def make_scored_frame(
    frame_id: int,
    t_ns: int,
    score: float,
    is_filtered: bool = False,
    filter_reason: Optional[str] = None,
) -> ScoredFrame:
    """Create a mock scored frame."""
    return ScoredFrame(
        frame_id=frame_id,
        t_ns=t_ns,
        score_result=ScoreResult(
            total_score=score,
            technical_score=score,
            action_score=score,
            identity_score=score,
            is_filtered=is_filtered,
            filter_reason=filter_reason,
        ),
    )


class TestFrameSelector:
    """Tests for FrameSelector class."""

    def test_default_config(self):
        """Test selector with default configuration."""
        selector = FrameSelector()
        assert selector.config.max_frames == 5
        assert selector.config.min_time_gap_ms == 200.0

    def test_custom_config(self):
        """Test selector with custom configuration."""
        config = SelectionConfig(max_frames=10, min_time_gap_ms=500.0)
        selector = FrameSelector(config)
        assert selector.config.max_frames == 10
        assert selector.config.min_time_gap_ms == 500.0

    def test_select_top_k(self):
        """Test selecting top K frames by score."""
        selector = FrameSelector(SelectionConfig(max_frames=3, min_time_gap_ms=0))

        frames = [
            make_scored_frame(0, 0, 0.5),
            make_scored_frame(1, 100_000_000, 0.9),  # Highest
            make_scored_frame(2, 200_000_000, 0.7),
            make_scored_frame(3, 300_000_000, 0.8),  # Second
            make_scored_frame(4, 400_000_000, 0.6),
        ]

        selected = selector.select(frames)

        assert len(selected) == 3
        # Should have top 3 scores: 0.9, 0.8, 0.7
        scores = [f.score for f in selected]
        assert 0.9 in scores
        assert 0.8 in scores
        assert 0.7 in scores

    def test_time_gap_constraint(self):
        """Test minimum time gap between selected frames."""
        selector = FrameSelector(SelectionConfig(max_frames=5, min_time_gap_ms=200))

        # All high scores but too close in time
        frames = [
            make_scored_frame(0, 0, 0.9),
            make_scored_frame(1, 50_000_000, 0.85),  # 50ms - too close
            make_scored_frame(2, 100_000_000, 0.8),  # 100ms - too close
            make_scored_frame(3, 250_000_000, 0.75),  # 250ms - OK
            make_scored_frame(4, 300_000_000, 0.7),  # 50ms from #3 - too close
            make_scored_frame(5, 500_000_000, 0.65),  # 500ms - OK
        ]

        selected = selector.select(frames)

        # Should select: 0 (0ms), 3 (250ms), 5 (500ms)
        assert len(selected) == 3
        selected_ids = {f.frame_id for f in selected}
        assert 0 in selected_ids  # Highest score
        assert 3 in selected_ids  # Next available
        assert 5 in selected_ids  # Next available

    def test_min_score_threshold(self):
        """Test minimum score threshold."""
        selector = FrameSelector(SelectionConfig(min_score=0.5))

        frames = [
            make_scored_frame(0, 0, 0.9),
            make_scored_frame(1, 300_000_000, 0.4),  # Below threshold
            make_scored_frame(2, 600_000_000, 0.3),  # Below threshold
            make_scored_frame(3, 900_000_000, 0.6),
        ]

        selected = selector.select(frames)

        assert len(selected) == 2
        selected_ids = {f.frame_id for f in selected}
        assert 0 in selected_ids
        assert 3 in selected_ids
        assert 1 not in selected_ids
        assert 2 not in selected_ids

    def test_filtered_frames_excluded(self):
        """Test that filtered frames are excluded."""
        selector = FrameSelector()

        frames = [
            make_scored_frame(0, 0, 0.9),
            make_scored_frame(1, 300_000_000, 0.95, is_filtered=True, filter_reason="blur"),
            make_scored_frame(2, 600_000_000, 0.7),
        ]

        selected = selector.select(frames)

        selected_ids = {f.frame_id for f in selected}
        assert 1 not in selected_ids  # Filtered out despite high score

    def test_sorted_by_timestamp(self):
        """Test that output is sorted by timestamp."""
        selector = FrameSelector(SelectionConfig(min_time_gap_ms=0))

        frames = [
            make_scored_frame(3, 300_000_000, 0.6),
            make_scored_frame(1, 100_000_000, 0.9),
            make_scored_frame(2, 200_000_000, 0.8),
        ]

        selected = selector.select(frames)

        # Should be sorted by timestamp
        timestamps = [f.t_ns for f in selected]
        assert timestamps == sorted(timestamps)

    def test_empty_input(self):
        """Test with empty input."""
        selector = FrameSelector()
        selected = selector.select([])
        assert selected == []

    def test_all_below_threshold(self):
        """Test when all frames are below threshold."""
        selector = FrameSelector(SelectionConfig(min_score=0.9))

        frames = [
            make_scored_frame(0, 0, 0.5),
            make_scored_frame(1, 100_000_000, 0.6),
        ]

        selected = selector.select(frames)
        assert selected == []

    def test_select_from_window(self):
        """Test selecting from a time window."""
        selector = FrameSelector(SelectionConfig(max_frames=2, min_time_gap_ms=0))

        frames = [
            make_scored_frame(0, 0, 0.5),
            make_scored_frame(1, 500_000_000, 0.9),  # In window
            make_scored_frame(2, 600_000_000, 0.8),  # In window
            make_scored_frame(3, 700_000_000, 0.85),  # In window
            make_scored_frame(4, 1_000_000_000, 0.7),
        ]

        # Window: 400ms to 800ms
        selected = selector.select_from_window(
            frames,
            window_start_ns=400_000_000,
            window_end_ns=800_000_000,
        )

        assert len(selected) == 2
        selected_ids = {f.frame_id for f in selected}
        # Should pick top 2 from window: 1 (0.9) and 3 (0.85)
        assert 1 in selected_ids
        assert 3 in selected_ids

    def test_select_around_triggers(self):
        """Test selecting frames around trigger events."""
        selector = FrameSelector(SelectionConfig(max_frames=10, min_time_gap_ms=50))

        frames = [
            make_scored_frame(0, 0, 0.5),
            make_scored_frame(1, 450_000_000, 0.7),  # Near trigger 1
            make_scored_frame(2, 500_000_000, 0.9),  # At trigger 1
            make_scored_frame(3, 550_000_000, 0.6),  # Near trigger 1
            make_scored_frame(4, 1_000_000_000, 0.4),
            make_scored_frame(5, 1_450_000_000, 0.8),  # Near trigger 2
            make_scored_frame(6, 1_500_000_000, 0.75),  # At trigger 2
        ]

        triggers_ns = [500_000_000, 1_500_000_000]  # Two triggers

        selected = selector.select_around_triggers(
            frames,
            trigger_times_ns=triggers_ns,
            window_before_ms=100,
            window_after_ms=100,
            frames_per_trigger=2,
        )

        # Should have frames from both trigger windows
        assert len(selected) >= 2
        selected_ids = {f.frame_id for f in selected}
        # Should include best from each window
        assert 2 in selected_ids  # Best near trigger 1
        assert 5 in selected_ids or 6 in selected_ids  # Best near trigger 2

    def test_scored_frame_properties(self):
        """Test ScoredFrame helper properties."""
        frame = make_scored_frame(1, 1_500_000_000, 0.75)

        assert frame.score == 0.75
        assert frame.t_ms == 1500.0
        assert frame.t_sec == 1.5
