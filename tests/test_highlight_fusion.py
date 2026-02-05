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
    yaw: float = 15.0,  # Default to looking slightly away (avoid camera_gaze trigger)
    pitch: float = 10.0,  # Default to looking slightly away
    inside_frame: bool = True,
    area_ratio: float = 0.05,
    center_distance: float = 0.1,
    quality_gate: float = 1.0,
    hand_wave: float = 0.0,
    happy: float = None,  # Individual emotion overrides
    angry: float = None,
    neutral: float = None,
) -> Observation:
    """Create a test observation with specified parameters.

    If happy/angry/neutral are not specified, they are derived from expression:
    - High expression (>0.5) maps to high happy
    - Low expression (<0.3) maps to high neutral
    """
    # Derive emotions from expression if not specified
    if happy is None:
        happy = expression  # High expression = happy
    if angry is None:
        angry = 0.1  # Default low angry
    if neutral is None:
        neutral = max(0.0, 1.0 - expression)  # Inverse of expression

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
                signals={
                    "em_happy": happy,
                    "em_angry": angry,
                    "em_neutral": neutral,
                },
            )
        )

    return Observation(
        source="test",
        frame_id=frame_id,
        t_ns=t_ns,
        signals={
            "face_count": face_count,
            "max_expression": expression,
            "expression_happy": happy,
            "expression_angry": angry,
            "expression_neutral": neutral,
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
            spike_sustain_sec=0.1,  # Short sustain for testing
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

        # Now spike expression (need enough frames for sustained spike + consecutive)
        for i in range(15, 30):
            obs = create_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                expression=0.9,  # High spike
            )
            result = fusion.update(obs)
            if result.should_trigger:
                triggered = True
                assert result.trigger_reason == "expression_spike"
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
                assert result.trigger_reason == "hand_wave"
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
            spike_sustain_sec=0.1,  # Short sustain for testing
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

        # Then trigger with high expression spike (need enough frames for sustained spike)
        for i in range(10, 30):
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
            spike_sustain_sec=0.1,  # Short sustain for testing
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
        for i in range(10, 30):
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
        assert result.trigger_score > 0
        assert result.trigger_reason in ["expression_spike", "head_turn", "hand_wave"]

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
            if result.should_trigger and result.trigger_reason == "head_turn":
                triggered = True
                break

        assert triggered


class TestHighlightFusionPhase9:
    """Tests for camera gaze and passenger interaction triggers."""

    def test_camera_gaze_detection_looking_at_camera(self):
        """Test camera gaze triggers when looking directly at camera."""
        fusion = HighlightFusion(
            gate_open_duration_sec=0.01,
            gaze_yaw_threshold=10.0,
            gaze_pitch_threshold=15.0,
            gaze_score_threshold=0.5,
            consecutive_frames=2,
            cooldown_sec=0.5,
        )

        frame_interval_ns = 33_333_333

        # Open gate and establish baseline
        for i in range(10):
            obs = create_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                yaw=20.0,  # Not looking at camera initially
            )
            fusion.update(obs)

        # Now look at camera (yaw and pitch close to 0)
        triggered = False
        for i in range(10, 20):
            obs = create_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                yaw=2.0,  # Looking almost straight at camera
                pitch=3.0,
            )
            result = fusion.update(obs)
            if result.should_trigger and result.trigger_reason == "camera_gaze":
                triggered = True
                break

        assert triggered

    def test_camera_gaze_no_trigger_when_looking_away(self):
        """Test camera gaze does not trigger when looking away."""
        fusion = HighlightFusion(
            gate_open_duration_sec=0.01,
            gaze_yaw_threshold=10.0,
            gaze_pitch_threshold=15.0,
            gaze_score_threshold=0.5,
            consecutive_frames=2,
        )

        frame_interval_ns = 33_333_333

        # Process frames with head turned away
        camera_gaze_triggered = False
        for i in range(20):
            obs = create_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                yaw=15.0,  # Looking to the side
                pitch=0.0,
            )
            result = fusion.update(obs)
            if result.should_trigger and result.trigger_reason == "camera_gaze":
                camera_gaze_triggered = True

        assert not camera_gaze_triggered

    def test_passenger_interaction_detection(self):
        """Test passenger interaction when two people look at each other."""
        fusion = HighlightFusion(
            gate_open_duration_sec=0.01,
            interaction_yaw_threshold=15.0,
            consecutive_frames=2,
            cooldown_sec=0.5,
        )

        frame_interval_ns = 33_333_333

        # Create observation with 2 faces looking at each other
        def create_two_face_observation(
            frame_id: int,
            t_ns: int,
            left_yaw: float,
            right_yaw: float,
        ) -> Observation:
            faces = [
                FaceObservation(
                    face_id=0,
                    confidence=0.9,
                    bbox=(0.2, 0.2, 0.2, 0.3),  # Left face
                    inside_frame=True,
                    yaw=left_yaw,
                    pitch=0.0,
                    area_ratio=0.05,
                    center_distance=0.15,
                    expression=0.3,
                ),
                FaceObservation(
                    face_id=1,
                    confidence=0.9,
                    bbox=(0.6, 0.2, 0.2, 0.3),  # Right face
                    inside_frame=True,
                    yaw=right_yaw,
                    pitch=0.0,
                    area_ratio=0.05,
                    center_distance=0.15,
                    expression=0.3,
                ),
            ]
            return Observation(
                source="test",
                frame_id=frame_id,
                t_ns=t_ns,
                signals={
                    "face_count": 2,
                    "quality_gate": 1.0,
                },
                faces=faces,
            )

        # Open gate first with faces not looking at each other
        for i in range(10):
            obs = create_two_face_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                left_yaw=0.0,  # Looking forward
                right_yaw=0.0,
            )
            fusion.update(obs)

        # Now faces look at each other
        triggered = False
        for i in range(10, 20):
            obs = create_two_face_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                left_yaw=25.0,  # Left person looking right
                right_yaw=-25.0,  # Right person looking left
            )
            result = fusion.update(obs)
            if result.should_trigger and result.trigger_reason == "passenger_interaction":
                triggered = True
                break

        assert triggered

    def test_passenger_interaction_no_trigger_single_face(self):
        """Test passenger interaction does not trigger with single face."""
        fusion = HighlightFusion(
            gate_open_duration_sec=0.01,
            consecutive_frames=2,
        )

        frame_interval_ns = 33_333_333

        # Single face - should not trigger passenger interaction
        triggered_interaction = False
        for i in range(20):
            obs = create_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                face_count=1,
            )
            result = fusion.update(obs)
            if result.should_trigger and result.trigger_reason == "passenger_interaction":
                triggered_interaction = True

        assert not triggered_interaction

    def test_gesture_trigger_vsign(self):
        """Test gesture trigger for V-sign."""
        fusion = HighlightFusion(
            gate_open_duration_sec=0.01,
            consecutive_frames=2,
            cooldown_sec=0.5,
        )

        frame_interval_ns = 33_333_333

        def create_gesture_observation(
            frame_id: int,
            t_ns: int,
            gesture_type: str,
            gesture_confidence: float,
        ) -> Observation:
            return Observation(
                source="gesture",
                frame_id=frame_id,
                t_ns=t_ns,
                signals={
                    "face_count": 1,
                    "quality_gate": 1.0,
                    "gesture_detected": 1.0 if gesture_confidence > 0 else 0.0,
                    "gesture_confidence": gesture_confidence,
                },
                faces=[
                    FaceObservation(
                        face_id=0,
                        confidence=0.9,
                        bbox=(0.3, 0.2, 0.2, 0.3),
                        inside_frame=True,
                        yaw=15.0,  # Looking slightly away (avoid camera_gaze)
                        pitch=10.0,
                        area_ratio=0.05,
                        center_distance=0.1,
                        expression=0.3,
                    )
                ],
                metadata={
                    "gesture_type": gesture_type,
                },
            )

        # Open gate first
        for i in range(5):
            obs = create_observation(frame_id=i, t_ns=i * frame_interval_ns)
            fusion.update(obs)

        # Send V-sign gesture
        triggered = False
        for i in range(5, 15):
            obs = create_gesture_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                gesture_type="v_sign",
                gesture_confidence=0.9,
            )
            result = fusion.update(obs)
            if result.should_trigger and result.trigger_reason == "gesture_vsign":
                triggered = True
                break

        assert triggered

    def test_gesture_trigger_thumbsup(self):
        """Test gesture trigger for thumbs up."""
        fusion = HighlightFusion(
            gate_open_duration_sec=0.01,
            consecutive_frames=2,
            cooldown_sec=0.5,
        )

        frame_interval_ns = 33_333_333

        def create_gesture_observation(
            frame_id: int,
            t_ns: int,
            gesture_type: str,
            gesture_confidence: float,
        ) -> Observation:
            return Observation(
                source="gesture",
                frame_id=frame_id,
                t_ns=t_ns,
                signals={
                    "face_count": 1,
                    "quality_gate": 1.0,
                    "gesture_detected": 1.0 if gesture_confidence > 0 else 0.0,
                    "gesture_confidence": gesture_confidence,
                },
                faces=[
                    FaceObservation(
                        face_id=0,
                        confidence=0.9,
                        bbox=(0.3, 0.2, 0.2, 0.3),
                        inside_frame=True,
                        yaw=15.0,  # Looking slightly away (avoid camera_gaze)
                        pitch=10.0,
                        area_ratio=0.05,
                        center_distance=0.1,
                        expression=0.3,
                    )
                ],
                metadata={
                    "gesture_type": gesture_type,
                },
            )

        # Open gate first
        for i in range(5):
            obs = create_observation(frame_id=i, t_ns=i * frame_interval_ns)
            fusion.update(obs)

        # Send thumbs up gesture
        triggered = False
        for i in range(5, 15):
            obs = create_gesture_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                gesture_type="thumbs_up",
                gesture_confidence=0.9,
            )
            result = fusion.update(obs)
            if result.should_trigger and result.trigger_reason == "gesture_thumbsup":
                triggered = True
                break

        assert triggered

    def test_new_trigger_types_in_metadata(self):
        """Test that new trigger types include proper metadata."""
        fusion = HighlightFusion(
            gate_open_duration_sec=0.01,
            consecutive_frames=2,
        )

        frame_interval_ns = 33_333_333

        # Open gate
        for i in range(10):
            obs = create_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                yaw=15.0,
            )
            fusion.update(obs)

        # Trigger camera gaze
        result = None
        for i in range(10, 20):
            obs = create_observation(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                yaw=2.0,
                pitch=2.0,
            )
            result = fusion.update(obs)
            if result.should_trigger:
                break

        if result and result.should_trigger:
            assert result.trigger is not None
            assert result.trigger.label == "highlight"
            assert result.trigger.metadata is not None
            assert "reason" in result.trigger.metadata


class TestHighlightFusionMainOnly:
    """Tests for main-only mode (Phase 16)."""

    def test_main_only_triggers_on_main_face(self):
        """Test that only main face triggers in main-only mode."""
        from facemoment.moment_detector.extractors.face_classifier import (
            FaceClassifierOutput,
            ClassifiedFace,
        )

        fusion = HighlightFusion(
            gate_open_duration_sec=0.01,
            consecutive_frames=2,
            cooldown_sec=0.5,
            main_only=True,
            spike_sustain_sec=0.1,
        )

        frame_interval_ns = 33_333_333

        def create_two_face_obs(
            frame_id: int,
            t_ns: int,
            main_happy: float,
            passenger_happy: float,
        ) -> tuple:
            """Create observation and classifier observation."""
            main_face = FaceObservation(
                face_id=1,
                confidence=0.9,
                bbox=(0.35, 0.2, 0.25, 0.35),
                inside_frame=True,
                yaw=15.0,
                pitch=10.0,
                area_ratio=0.08,
                center_distance=0.1,
                signals={"em_happy": main_happy},
            )
            passenger_face = FaceObservation(
                face_id=2,
                confidence=0.85,
                bbox=(0.65, 0.3, 0.15, 0.25),
                inside_frame=True,
                yaw=15.0,
                pitch=10.0,
                area_ratio=0.04,
                center_distance=0.2,
                signals={"em_happy": passenger_happy},
            )

            obs = Observation(
                source="test",
                frame_id=frame_id,
                t_ns=t_ns,
                signals={"face_count": 2, "quality_gate": 1.0},
                faces=[main_face, passenger_face],
            )

            # Create classifier output
            cf_main = ClassifiedFace(
                face=main_face,
                role="main",
                confidence=0.9,
                track_length=50,
                avg_area=0.08,
            )
            cf_passenger = ClassifiedFace(
                face=passenger_face,
                role="passenger",
                confidence=0.7,
                track_length=30,
                avg_area=0.04,
            )

            classifier_data = FaceClassifierOutput(
                faces=[cf_main, cf_passenger],
                main_face=cf_main,
                passenger_faces=[cf_passenger],
            )

            classifier_obs = Observation(
                source="face_classifier",
                frame_id=frame_id,
                t_ns=t_ns,
                signals={},
                data=classifier_data,
            )

            return obs, classifier_obs

        # Baseline phase: both faces at low happy
        for i in range(15):
            obs, classifier_obs = create_two_face_obs(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                main_happy=0.2,
                passenger_happy=0.2,
            )
            fusion.update(obs, classifier_obs)

        # Now passenger has high happy, main stays low -> should NOT trigger
        for i in range(15, 30):
            obs, classifier_obs = create_two_face_obs(
                frame_id=i,
                t_ns=i * frame_interval_ns,
                main_happy=0.2,  # Main face stays low
                passenger_happy=0.9,  # Passenger is very happy
            )
            result = fusion.update(obs, classifier_obs)
            # Should not trigger because main face is not happy
            assert not result.should_trigger, f"Should not trigger on passenger (frame {i})"

    def test_main_only_ignores_passenger_expression(self):
        """Test main-only mode ignores passenger expressions."""
        from facemoment.moment_detector.extractors.face_classifier import (
            FaceClassifierOutput,
            ClassifiedFace,
        )

        fusion = HighlightFusion(
            gate_open_duration_sec=0.01,
            consecutive_frames=2,
            cooldown_sec=0.5,
            main_only=True,
            spike_sustain_sec=0.1,
        )

        frame_interval_ns = 33_333_333

        def create_obs(frame_id, t_ns, main_happy, passenger_happy):
            main_face = FaceObservation(
                face_id=1,
                confidence=0.9,
                bbox=(0.35, 0.2, 0.25, 0.35),
                yaw=15.0,
                pitch=10.0,
                area_ratio=0.08,
                signals={"em_happy": main_happy},
            )
            passenger_face = FaceObservation(
                face_id=2,
                confidence=0.85,
                bbox=(0.65, 0.3, 0.15, 0.25),
                yaw=15.0,
                pitch=10.0,
                area_ratio=0.04,
                signals={"em_happy": passenger_happy},
            )

            obs = Observation(
                source="test",
                frame_id=frame_id,
                t_ns=t_ns,
                signals={"face_count": 2, "quality_gate": 1.0},
                faces=[main_face, passenger_face],
            )

            cf_main = ClassifiedFace(
                face=main_face, role="main", confidence=0.9, track_length=50, avg_area=0.08,
            )
            classifier_data = FaceClassifierOutput(
                faces=[cf_main], main_face=cf_main, passenger_faces=[],
            )
            classifier_obs = Observation(
                source="face_classifier", frame_id=frame_id, t_ns=t_ns, signals={}, data=classifier_data,
            )

            return obs, classifier_obs

        # Baseline
        for i in range(15):
            obs, classifier_obs = create_obs(i, i * frame_interval_ns, 0.2, 0.2)
            fusion.update(obs, classifier_obs)

        # Main face becomes happy -> should trigger
        triggered = False
        for i in range(15, 35):
            obs, classifier_obs = create_obs(i, i * frame_interval_ns, 0.9, 0.2)
            result = fusion.update(obs, classifier_obs)
            if result.should_trigger:
                triggered = True
                break

        assert triggered, "Should trigger when main face is happy"
