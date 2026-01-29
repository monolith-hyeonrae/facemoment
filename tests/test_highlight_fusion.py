"""Tests for HighlightFusion."""

import pytest

from facemoment.moment_detector.extractors.base import Observation, FaceObservation
from facemoment.moment_detector.fusion.highlight import HighlightFusion


def create_observation(
    frame_id: int = 0,
    t_ns: int = 0,
    face_count: int = 1,
    expression: float = 0.3,
    confidence: float = 0.9,
    yaw: float = 0.0,
    pitch: float = 0.0,
    inside_frame: bool = True,
    area_ratio: float = 0.05,
    center_distance: float = 0.1,
    quality_gate: float = 1.0,
    hand_wave: float = 0.0,
) -> Observation:
    """Create a test observation with specified parameters."""
    faces = []
    for i in range(face_count):
        faces.append(
            FaceObservation(
                face_id=i,
                confidence=confidence,
                bbox=(0.2 + i * 0.3, 0.2, 0.2, 0.3),
                inside_frame=inside_frame,
                yaw=yaw,
                pitch=pitch,
                roll=0.0,
                area_ratio=area_ratio,
                center_distance=center_distance,
                expression=expression,
            )
        )

    return Observation(
        source="test",
        frame_id=frame_id,
        t_ns=t_ns,
        signals={
            "face_count": face_count,
            "max_expression": expression,
            "quality_gate": quality_gate,
            "hand_wave_detected": hand_wave,
            "hand_wave_confidence": hand_wave * 0.9,
        },
        faces=faces,
    )


class TestHighlightFusion:
    def test_initial_state(self):
        """Test fusion initial state."""
        fusion = HighlightFusion()

        assert not fusion.is_gate_open
        assert not fusion.in_cooldown

    def test_gate_opens_with_good_conditions(self):
        """Test gate opens when conditions are met for duration."""
        fusion = HighlightFusion(
            gate_open_duration_sec=0.1,  # 100ms to open
        )

        # Simulate frames at 30fps with good conditions
        frame_interval_ns = 33_333_333  # ~30fps

        for i in range(10):  # 10 frames = ~333ms
            obs = create_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
            )
            fusion.update(obs)

        assert fusion.is_gate_open

    def test_gate_closes_with_bad_conditions(self):
        """Test gate closes after conditions fail for duration."""
        fusion = HighlightFusion(
            gate_open_duration_sec=0.05,
            gate_close_duration_sec=0.1,
        )

        frame_interval_ns = 33_333_333

        # Open the gate first
        for i in range(10):
            obs = create_observation(frame_id=i, t_ns=i * frame_interval_ns)
            fusion.update(obs)

        assert fusion.is_gate_open

        # Now send bad conditions (no faces)
        for i in range(10, 20):
            obs = create_observation(
                frame_id=i, t_ns=i * frame_interval_ns, face_count=0
            )
            fusion.update(obs)

        assert not fusion.is_gate_open

    def test_gate_rejects_wrong_face_count(self):
        """Test gate rejects too many or too few faces."""
        fusion = HighlightFusion(gate_open_duration_sec=0.01)

        # No faces
        obs = create_observation(face_count=0)
        result = fusion.update(obs)
        assert not fusion.is_gate_open

        fusion.reset()

        # Too many faces
        obs = create_observation(face_count=5)
        result = fusion.update(obs)
        assert not fusion.is_gate_open

    def test_gate_rejects_extreme_angles(self):
        """Test gate rejects extreme head angles."""
        fusion = HighlightFusion(
            yaw_max=25.0,
            pitch_max=20.0,
            gate_open_duration_sec=0.01,
        )

        # Extreme yaw
        obs = create_observation(yaw=30.0)
        fusion.update(obs)
        assert not fusion.is_gate_open

        fusion.reset()

        # Extreme pitch
        obs = create_observation(pitch=25.0)
        fusion.update(obs)
        assert not fusion.is_gate_open

    def test_gate_rejects_low_confidence(self):
        """Test gate rejects low confidence faces."""
        fusion = HighlightFusion(
            face_conf_threshold=0.7,
            gate_open_duration_sec=0.01,
        )

        obs = create_observation(confidence=0.5)
        fusion.update(obs)
        assert not fusion.is_gate_open

    def test_gate_rejects_quality_gate_closed(self):
        """Test gate respects quality extractor signal."""
        fusion = HighlightFusion(gate_open_duration_sec=0.01)

        obs = create_observation(quality_gate=0.0)
        fusion.update(obs)
        assert not fusion.is_gate_open

    def test_expression_spike_trigger(self):
        """Test trigger on expression spike."""
        fusion = HighlightFusion(
            gate_open_duration_sec=0.01,
            expression_z_threshold=1.5,
            consecutive_frames=2,
            cooldown_sec=0.5,
        )

        frame_interval_ns = 33_333_333
        triggered = False

        # First, establish baseline with low expression
        for i in range(15):
            obs = create_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                expression=0.2,
            )
            fusion.update(obs)

        # Now spike expression
        for i in range(15, 20):
            obs = create_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                expression=0.9,  # High spike
            )
            result = fusion.update(obs)
            if result.should_trigger:
                triggered = True
                assert result.reason == "expression_spike"
                break

        assert triggered

    def test_hand_wave_trigger(self):
        """Test trigger on hand wave detection."""
        fusion = HighlightFusion(
            gate_open_duration_sec=0.01,
            consecutive_frames=2,
            cooldown_sec=0.5,
        )

        frame_interval_ns = 33_333_333

        # Open gate first
        for i in range(5):
            obs = create_observation(frame_id=i, t_ns=i * frame_interval_ns)
            fusion.update(obs)

        # Now send hand wave signal
        triggered = False
        for i in range(5, 10):
            obs = create_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                hand_wave=1.0,
            )
            result = fusion.update(obs)
            if result.should_trigger:
                triggered = True
                assert result.reason == "hand_wave"
                break

        assert triggered

    def test_cooldown_prevents_rapid_triggers(self):
        """Test cooldown prevents rapid re-triggering."""
        fusion = HighlightFusion(
            gate_open_duration_sec=0.01,
            consecutive_frames=1,
            cooldown_sec=1.0,  # 1 second cooldown
        )

        frame_interval_ns = 33_333_333
        trigger_count = 0

        for i in range(60):  # 2 seconds of frames
            obs = create_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                expression=0.9,  # Always high
            )
            result = fusion.update(obs)
            if result.should_trigger:
                trigger_count += 1

        # Should trigger at most twice (initial + after 1s cooldown)
        assert trigger_count <= 2

    def test_consecutive_frames_required(self):
        """Test that trigger requires consecutive high frames."""
        fusion = HighlightFusion(
            gate_open_duration_sec=0.01,
            consecutive_frames=3,
            cooldown_sec=0.1,
        )

        frame_interval_ns = 33_333_333

        # Open gate
        for i in range(5):
            obs = create_observation(frame_id=i, t_ns=i * frame_interval_ns)
            fusion.update(obs)

        # Alternate high/low - should not trigger
        for i in range(5, 15):
            expr = 0.9 if i % 2 == 0 else 0.2
            obs = create_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                expression=expr,
            )
            result = fusion.update(obs)
            # Should not trigger due to alternating pattern
            if i < 10:  # Before we establish baseline
                assert not result.should_trigger

    def test_reset_clears_state(self):
        """Test reset clears all state."""
        fusion = HighlightFusion(
            gate_open_duration_sec=0.01,
            consecutive_frames=2,
        )

        frame_interval_ns = 33_333_333

        # First establish low baseline
        for i in range(10):
            obs = create_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                expression=0.2,
            )
            fusion.update(obs)

        # Then trigger with high expression spike
        for i in range(10, 20):
            obs = create_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                expression=0.95,
            )
            result = fusion.update(obs)
            if result.should_trigger:
                break

        assert fusion.is_gate_open
        assert fusion.in_cooldown

        fusion.reset()

        assert not fusion.is_gate_open
        assert not fusion.in_cooldown

    def test_trigger_includes_metadata(self):
        """Test that trigger result includes proper metadata."""
        fusion = HighlightFusion(
            gate_open_duration_sec=0.01,
            consecutive_frames=2,
            pre_sec=1.5,
            post_sec=2.5,
        )

        frame_interval_ns = 33_333_333

        # Establish baseline then trigger
        for i in range(10):
            obs = create_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                expression=0.2,
            )
            fusion.update(obs)

        result = None
        for i in range(10, 15):
            obs = create_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                expression=0.95,
            )
            result = fusion.update(obs)
            if result.should_trigger:
                break

        assert result is not None
        assert result.should_trigger
        assert result.trigger is not None
        assert result.trigger.label == "highlight"
        assert result.score > 0
        assert result.reason in ["expression_spike", "head_turn", "hand_wave"]

    def test_gate_hysteresis_prevents_flapping(self):
        """Test gate hysteresis prevents rapid open/close cycles."""
        fusion = HighlightFusion(
            gate_open_duration_sec=0.2,  # 200ms to open
            gate_close_duration_sec=0.2,  # 200ms to close
        )

        frame_interval_ns = 33_333_333  # ~30ms

        # Send alternating good/bad frames (faster than hysteresis)
        gate_changes = 0
        last_gate_state = False

        for i in range(30):  # ~1 second
            # Alternate every frame
            if i % 2 == 0:
                obs = create_observation(frame_id=i, t_ns=i * frame_interval_ns)
            else:
                obs = create_observation(
                    frame_id=i, t_ns=i * frame_interval_ns, face_count=0
                )

            fusion.update(obs)

            if fusion.is_gate_open != last_gate_state:
                gate_changes += 1
                last_gate_state = fusion.is_gate_open

        # Gate should not flap rapidly due to hysteresis
        assert gate_changes <= 2  # At most open once, close once

    def test_head_turn_detection(self):
        """Test head turn trigger detection."""
        fusion = HighlightFusion(
            gate_open_duration_sec=0.01,
            head_turn_velocity_threshold=20.0,  # deg/sec
            consecutive_frames=2,
            cooldown_sec=0.5,
        )

        frame_interval_ns = 33_333_333  # ~30fps, ~0.033s per frame

        # Open gate with stable head position
        for i in range(10):
            obs = create_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                yaw=0.0,
            )
            fusion.update(obs)

        # Rapid head turn (need > 20 deg/sec)
        # At 30fps, 2 degrees per frame = 60 deg/sec
        triggered = False
        for i in range(10, 20):
            obs = create_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                yaw=(i - 10) * 2.0,  # 2 degrees per frame
            )
            result = fusion.update(obs)
            if result.should_trigger and result.reason == "head_turn":
                triggered = True
                break

        assert triggered
