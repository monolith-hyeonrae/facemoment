"""Tests for MomentDetector."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from facemoment import MomentDetector
from facemoment.moment_detector.extractors import DummyExtractor, Observation
from facemoment.moment_detector.fusion import DummyFusion


def create_test_video(path: Path, num_frames: int = 90, fps: int = 30) -> None:
    """Create a test video file (3 seconds at 30fps)."""
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 3) % 256
        frame[:, :, 1] = (i * 2) % 256
        frame[:, :, 2] = (i * 1) % 256
        cv2.putText(
            frame, f"F{i}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        writer.write(frame)

    writer.release()


class TestDummyExtractor:
    def test_extract_produces_observation(self):
        from visualbase import Frame
        import numpy as np

        extractor = DummyExtractor(num_faces=2, seed=42)

        # Create a dummy frame
        data = np.zeros((240, 320, 3), dtype=np.uint8)
        frame = Frame.from_array(data, frame_id=0, t_src_ns=0)

        obs = extractor.extract(frame)

        assert obs is not None
        assert obs.source == "dummy"
        assert obs.frame_id == 0
        assert len(obs.faces) == 2
        assert "max_expression" in obs.signals
        assert "face_count" in obs.signals

    def test_extractor_name(self):
        extractor = DummyExtractor(name="test_extractor")
        assert extractor.name == "test_extractor"

    def test_face_observation_values(self):
        from visualbase import Frame
        import numpy as np

        extractor = DummyExtractor(num_faces=1, seed=42)
        data = np.zeros((240, 320, 3), dtype=np.uint8)
        frame = Frame.from_array(data, frame_id=0, t_src_ns=0)

        obs = extractor.extract(frame)
        face = obs.faces[0]

        assert 0 <= face.confidence <= 1
        assert len(face.bbox) == 4
        assert all(0 <= v <= 1 for v in face.bbox)
        assert -90 <= face.yaw <= 90
        assert 0 <= face.expression <= 1


class TestDummyFusion:
    def test_fusion_initial_state(self):
        fusion = DummyFusion()
        assert not fusion.is_gate_open
        assert not fusion.in_cooldown

    def test_fusion_no_trigger_on_low_expression(self):
        fusion = DummyFusion(expression_threshold=0.9)

        # Create observation with low expression
        obs = Observation(
            source="test",
            frame_id=0,
            t_ns=0,
            signals={"max_expression": 0.3, "face_count": 1},
            faces=[],
        )

        result = fusion.update(obs)
        assert not result.should_trigger

    def test_fusion_reset(self):
        fusion = DummyFusion()

        # Process some observations
        for i in range(5):
            obs = Observation(
                source="test",
                frame_id=i,
                t_ns=i * 100_000_000,
                signals={"max_expression": 0.8, "face_count": 1},
                faces=[],
            )
            fusion.update(obs)

        # Reset
        fusion.reset()
        assert not fusion.in_cooldown


class TestMomentDetector:
    def test_process_file_no_triggers(self, tmp_path):
        """Test processing with low expression (no triggers)."""
        video_path = tmp_path / "test.mp4"
        clips_dir = tmp_path / "clips"
        create_test_video(video_path, num_frames=30, fps=30)

        # Use high threshold so no triggers fire
        detector = MomentDetector(
            extractors=[DummyExtractor(num_faces=1, spike_probability=0.0, seed=42)],
            fusion=DummyFusion(expression_threshold=0.99),
            clip_output_dir=clips_dir,
        )

        clips = detector.process_file(str(video_path), fps=10)

        assert detector.frames_processed > 0
        assert detector.triggers_fired == 0
        assert len(clips) == 0

    def test_process_file_with_triggers(self, tmp_path):
        """Test processing with high expression (triggers expected)."""
        video_path = tmp_path / "test.mp4"
        clips_dir = tmp_path / "clips"
        create_test_video(video_path, num_frames=90, fps=30)  # 3 seconds

        # Use low threshold and high spike probability to guarantee triggers
        detector = MomentDetector(
            extractors=[DummyExtractor(num_faces=1, spike_probability=0.5, seed=42)],
            fusion=DummyFusion(
                expression_threshold=0.5,
                consecutive_frames=2,
                cooldown_sec=1.0,
            ),
            clip_output_dir=clips_dir,
        )

        clips = detector.process_file(str(video_path), fps=10)

        assert detector.frames_processed > 0
        # With high spike probability, we should get at least one trigger
        assert detector.triggers_fired >= 1
        assert len(clips) >= 1

        # Check clip was created
        for clip in clips:
            if clip.success:
                assert clip.output_path.exists()

    def test_process_stream(self, tmp_path):
        """Test streaming mode."""
        video_path = tmp_path / "test.mp4"
        clips_dir = tmp_path / "clips"
        create_test_video(video_path, num_frames=30, fps=30)

        detector = MomentDetector(
            extractors=[DummyExtractor(num_faces=1, seed=42)],
            fusion=DummyFusion(expression_threshold=0.99),
            clip_output_dir=clips_dir,
        )

        frames_seen = 0
        for frame, result in detector.process_stream(str(video_path), fps=10):
            frames_seen += 1
            assert frame is not None

        assert frames_seen > 0

    def test_callbacks(self, tmp_path):
        """Test callback functionality."""
        video_path = tmp_path / "test.mp4"
        clips_dir = tmp_path / "clips"
        create_test_video(video_path, num_frames=30, fps=30)

        detector = MomentDetector(
            extractors=[DummyExtractor(num_faces=1, seed=42)],
            fusion=DummyFusion(expression_threshold=0.99),
            clip_output_dir=clips_dir,
        )

        frames_received = []
        observations_received = []

        detector.set_on_frame(lambda f: frames_received.append(f))
        detector.set_on_observation(lambda o: observations_received.append(o))

        detector.process_file(str(video_path), fps=10)

        assert len(frames_received) > 0
        assert len(observations_received) > 0

    def test_reset_stats(self, tmp_path):
        """Test stats reset."""
        video_path = tmp_path / "test.mp4"
        clips_dir = tmp_path / "clips"
        create_test_video(video_path, num_frames=30, fps=30)

        detector = MomentDetector(
            extractors=[DummyExtractor(seed=42)],
            fusion=DummyFusion(),
            clip_output_dir=clips_dir,
        )

        detector.process_file(str(video_path), fps=10)
        assert detector.frames_processed > 0

        detector.reset_stats()
        assert detector.frames_processed == 0
        assert detector.triggers_fired == 0


class TestE2E:
    """End-to-end tests: Frame → Analysis → Trigger → Clip."""

    def test_full_pipeline(self, tmp_path):
        """Test complete pipeline from video to clip."""
        video_path = tmp_path / "test.mp4"
        clips_dir = tmp_path / "clips"

        # Create longer video for better trigger chance
        create_test_video(video_path, num_frames=150, fps=30)  # 5 seconds

        detector = MomentDetector(
            extractors=[
                DummyExtractor(
                    name="face",
                    num_faces=2,
                    spike_probability=0.3,
                    seed=123,
                )
            ],
            fusion=DummyFusion(
                expression_threshold=0.6,
                consecutive_frames=2,
                cooldown_sec=1.5,
                pre_sec=1.0,
                post_sec=1.0,
            ),
            clip_output_dir=clips_dir,
        )

        trigger_events = []
        detector.set_on_trigger(lambda t, r: trigger_events.append((t, r)))

        clips = detector.process_file(str(video_path), fps=10)

        print(f"Frames processed: {detector.frames_processed}")
        print(f"Triggers fired: {detector.triggers_fired}")
        print(f"Clips created: {len(clips)}")

        # Verify pipeline works (may or may not trigger based on random)
        assert detector.frames_processed > 0

        # If triggers fired, verify clips
        for clip in clips:
            if clip.success:
                assert clip.output_path.exists()
                assert clip.duration_sec > 0
                print(f"Clip: {clip.output_path}, duration: {clip.duration_sec:.2f}s")
