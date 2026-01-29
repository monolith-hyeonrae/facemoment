"""Tests for process module (ExtractorProcess, FusionProcess)."""

import os
import tempfile
import threading
import time

import pytest
import numpy as np

from visualbase.core.frame import Frame
from visualbase.ipc.uds import UDSServer, UDSClient
from visualbase.ipc.messages import parse_obs_message

from facemoment.moment_detector.extractors.base import (
    BaseExtractor,
    Observation,
    FaceObservation,
)
from facemoment.moment_detector.fusion.base import BaseFusion, FusionResult
from facemoment.process.extractor import ExtractorProcess
from facemoment.process.fusion import FusionProcess


class MockExtractor(BaseExtractor):
    """Mock extractor for testing."""

    def __init__(self, name: str = "mock"):
        self._name = name
        self._call_count = 0

    @property
    def name(self) -> str:
        return self._name

    def extract(self, frame: Frame):
        self._call_count += 1
        if self._name == "face":
            return Observation(
                source="face",
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                faces=[
                    FaceObservation(
                        face_id=0,
                        confidence=0.95,
                        bbox=(0.1, 0.2, 0.3, 0.4),
                        expression=0.8,
                        yaw=5.0,
                        pitch=2.0,
                    ),
                ],
            )
        elif self._name == "pose":
            return Observation(
                source="pose",
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={
                    "hand_raised": 1.0,
                    "hand_wave": 0.0,
                    "wave_count": 0,
                    "confidence": 0.9,
                },
            )
        elif self._name == "quality":
            return Observation(
                source="quality",
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={
                    "blur_score": 100.0,
                    "brightness": 128.0,
                    "contrast": 0.5,
                    "quality_gate": 1.0,
                },
            )
        return None


class MockFusion(BaseFusion):
    """Mock fusion for testing."""

    def __init__(self):
        self._observations = []
        self._gate_open = True
        self._in_cooldown = False

    def update(self, observation: Observation) -> FusionResult:
        self._observations.append(observation)
        # Trigger every 5th observation
        should_trigger = len(self._observations) % 5 == 0
        return FusionResult(
            should_trigger=should_trigger,
            trigger=None,  # Would need actual Trigger for real test
            score=0.8 if should_trigger else 0.0,
            reason="test_trigger" if should_trigger else "",
        )

    def reset(self) -> None:
        self._observations.clear()

    @property
    def is_gate_open(self) -> bool:
        return self._gate_open

    @property
    def in_cooldown(self) -> bool:
        return self._in_cooldown


class TestExtractorProcess:
    """Tests for ExtractorProcess."""

    def test_observation_to_message_face(self):
        """Test conversion of face observation to OBS message."""
        extractor = MockExtractor(name="face")
        process = ExtractorProcess(
            extractor=extractor,
            input_fifo="/tmp/test.fifo",
            obs_socket="/tmp/test.sock",
        )

        # Create a test frame
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame.from_array(data, frame_id=1, t_src_ns=1_000_000_000)

        # Extract observation
        obs = extractor.extract(frame)
        assert obs is not None

        # Convert to message
        msg = process._observation_to_message(obs)
        assert msg is not None
        assert msg.startswith("OBS src=face")
        assert "frame=1" in msg
        assert "faces=1" in msg

        # Parse it back
        parsed = parse_obs_message(msg)
        assert parsed is not None
        assert parsed.src == "face"
        assert len(parsed.faces) == 1
        assert parsed.faces[0].expr == pytest.approx(0.8, rel=1e-2)

    def test_observation_to_message_pose(self):
        """Test conversion of pose observation to OBS message."""
        extractor = MockExtractor(name="pose")
        process = ExtractorProcess(
            extractor=extractor,
            input_fifo="/tmp/test.fifo",
            obs_socket="/tmp/test.sock",
        )

        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame.from_array(data, frame_id=1, t_src_ns=1_000_000_000)

        obs = extractor.extract(frame)
        msg = process._observation_to_message(obs)

        assert msg is not None
        assert msg.startswith("OBS src=pose")
        assert "poses=1" in msg

    def test_observation_to_message_quality(self):
        """Test conversion of quality observation to OBS message."""
        extractor = MockExtractor(name="quality")
        process = ExtractorProcess(
            extractor=extractor,
            input_fifo="/tmp/test.fifo",
            obs_socket="/tmp/test.sock",
        )

        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame.from_array(data, frame_id=1, t_src_ns=1_000_000_000)

        obs = extractor.extract(frame)
        msg = process._observation_to_message(obs)

        assert msg is not None
        assert msg.startswith("OBS src=quality")
        assert "blur:100.0" in msg

    def test_get_stats(self):
        """Test stats retrieval."""
        extractor = MockExtractor(name="face")
        process = ExtractorProcess(
            extractor=extractor,
            input_fifo="/tmp/test.fifo",
            obs_socket="/tmp/test.sock",
        )

        stats = process.get_stats()
        assert "frames_processed" in stats
        assert "obs_sent" in stats
        assert "errors" in stats
        assert "fps" in stats


class TestFusionProcess:
    """Tests for FusionProcess."""

    def test_obs_to_observation_face(self):
        """Test conversion of OBS message to Observation."""
        fusion = MockFusion()
        process = FusionProcess(
            fusion=fusion,
            obs_socket="/tmp/obs.sock",
            trig_socket="/tmp/trig.sock",
        )

        # Create mock OBS message
        from visualbase.ipc.messages import OBSMessage, FaceData
        obs_msg = OBSMessage(
            src="face",
            frame_id=1,
            t_ns=1_000_000_000,
            faces=[
                FaceData(id=0, conf=0.95, x=0.1, y=0.2, w=0.3, h=0.4, expr=0.8),
            ],
        )

        # Convert to Observation
        obs = process._obs_to_observation(obs_msg)

        assert obs is not None
        assert obs.source == "face"
        assert obs.frame_id == 1
        assert len(obs.faces) == 1
        assert obs.faces[0].expression == 0.8

    def test_obs_to_observation_pose(self):
        """Test conversion of pose OBS message to Observation."""
        fusion = MockFusion()
        process = FusionProcess(
            fusion=fusion,
            obs_socket="/tmp/obs.sock",
            trig_socket="/tmp/trig.sock",
        )

        from visualbase.ipc.messages import OBSMessage, PoseData
        obs_msg = OBSMessage(
            src="pose",
            frame_id=1,
            t_ns=1_000_000_000,
            poses=[
                PoseData(id=0, conf=0.9, hand_raised=True, hand_wave=False),
            ],
        )

        obs = process._obs_to_observation(obs_msg)

        assert obs is not None
        assert obs.source == "pose"
        assert obs.signals.get("hand_raised") == 1.0
        assert obs.signals.get("hand_wave") == 0.0

    def test_obs_to_observation_quality(self):
        """Test conversion of quality OBS message to Observation."""
        fusion = MockFusion()
        process = FusionProcess(
            fusion=fusion,
            obs_socket="/tmp/obs.sock",
            trig_socket="/tmp/trig.sock",
        )

        from visualbase.ipc.messages import OBSMessage, QualityData
        obs_msg = OBSMessage(
            src="quality",
            frame_id=1,
            t_ns=1_000_000_000,
            quality=QualityData(blur=100.0, brightness=128.0, contrast=0.5, gate_open=True),
        )

        obs = process._obs_to_observation(obs_msg)

        assert obs is not None
        assert obs.source == "quality"
        assert obs.signals.get("blur_score") == 100.0
        assert obs.signals.get("quality_gate") == 1.0

    def test_get_stats(self):
        """Test stats retrieval."""
        fusion = MockFusion()
        process = FusionProcess(
            fusion=fusion,
            obs_socket="/tmp/obs.sock",
            trig_socket="/tmp/trig.sock",
        )

        stats = process.get_stats()
        assert "obs_received" in stats
        assert "triggers_sent" in stats
        assert "errors" in stats
        assert "buffer_frames" in stats
