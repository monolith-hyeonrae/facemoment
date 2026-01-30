"""Tests for process module (ExtractorProcess, FusionProcess)."""

import os
import tempfile
import threading
import time
from typing import Optional, List

import pytest
import numpy as np

from visualbase.core.frame import Frame
from visualbase.ipc.uds import UDSServer, UDSClient
from visualbase.ipc.messages import parse_obs_message
from visualbase.ipc.interfaces import VideoReader, MessageSender

from facemoment.moment_detector.extractors.base import (
    BaseExtractor,
    Observation,
    FaceObservation,
)
from facemoment.moment_detector.fusion.base import BaseFusion, FusionResult
from facemoment.process.extractor import ExtractorProcess
from facemoment.process.fusion import FusionProcess
from facemoment.process.orchestrator import ExtractorOrchestrator


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


class MockVideoReader(VideoReader):
    """Mock video reader for testing interface-based initialization."""

    def __init__(self, frames: Optional[List[Frame]] = None):
        self._frames = frames or []
        self._index = 0
        self._is_open = False

    def open(self, timeout_sec: Optional[float] = None) -> bool:
        self._is_open = True
        self._index = 0
        return True

    def read(self) -> Optional[Frame]:
        if self._index < len(self._frames):
            frame = self._frames[self._index]
            self._index += 1
            return frame
        return None

    def close(self) -> None:
        self._is_open = False

    @property
    def is_open(self) -> bool:
        return self._is_open


class MockMessageSender(MessageSender):
    """Mock message sender for testing interface-based initialization."""

    def __init__(self):
        self._connected = False
        self._messages: List[str] = []

    def connect(self) -> bool:
        self._connected = True
        return True

    def send(self, message: str) -> bool:
        if self._connected:
            self._messages.append(message)
            return True
        return False

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected


class MockMessageReceiver:
    """Mock message receiver for testing interface-based initialization."""

    def __init__(self, messages: Optional[List[str]] = None):
        self._messages = messages or []
        self._running = False
        self._recv_index = 0

    def start(self) -> None:
        self._running = True
        self._recv_index = 0

    def recv(self, timeout: Optional[float] = None) -> Optional[str]:
        if self._recv_index < len(self._messages):
            msg = self._messages[self._recv_index]
            self._recv_index += 1
            return msg
        return None

    def recv_all(self, max_messages: int = 100) -> List[str]:
        result = self._messages[self._recv_index:self._recv_index + max_messages]
        self._recv_index += len(result)
        return result

    def stop(self) -> None:
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running


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

    def test_interface_based_initialization(self):
        """Test ExtractorProcess with interface-based dependency injection."""
        extractor = MockExtractor(name="face")
        reader = MockVideoReader()
        sender = MockMessageSender()

        process = ExtractorProcess(
            extractor=extractor,
            video_reader=reader,
            message_sender=sender,
        )

        # Verify process was created with injected dependencies
        assert process._reader is reader
        assert process._client is sender
        assert process._reader_provided is True
        assert process._client_provided is True

    def test_interface_process_frames(self):
        """Test frame processing with interface-based dependencies."""
        extractor = MockExtractor(name="face")

        # Create test frames
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frames = [
            Frame.from_array(data, frame_id=i, t_src_ns=i * 1_000_000_000)
            for i in range(3)
        ]

        reader = MockVideoReader(frames=frames)
        sender = MockMessageSender()

        process = ExtractorProcess(
            extractor=extractor,
            video_reader=reader,
            message_sender=sender,
            reconnect=False,  # Don't reconnect after frames are exhausted
        )

        # Run in a thread with timeout
        def run_with_stop():
            import time
            time.sleep(0.1)
            process.stop()

        stop_thread = threading.Thread(target=run_with_stop)
        stop_thread.start()

        process.run()
        stop_thread.join()

        # Verify frames were processed
        assert process._frames_processed == 3
        assert len(sender._messages) == 3

        # Verify OBS messages were sent correctly
        for msg in sender._messages:
            assert msg.startswith("OBS src=face")

    def test_missing_reader_and_path_raises(self):
        """Test that missing both video_reader and input_fifo raises ValueError."""
        extractor = MockExtractor(name="face")
        sender = MockMessageSender()

        with pytest.raises(ValueError, match="Either video_reader or input_fifo"):
            ExtractorProcess(
                extractor=extractor,
                message_sender=sender,
            )

    def test_missing_sender_and_path_raises(self):
        """Test that missing both message_sender and obs_socket raises ValueError."""
        extractor = MockExtractor(name="face")
        reader = MockVideoReader()

        with pytest.raises(ValueError, match="Either message_sender or obs_socket"):
            ExtractorProcess(
                extractor=extractor,
                video_reader=reader,
            )


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

    def test_interface_based_initialization(self):
        """Test FusionProcess with interface-based dependency injection."""
        fusion = MockFusion()
        obs_receiver = MockMessageReceiver()
        trig_sender = MockMessageSender()

        process = FusionProcess(
            fusion=fusion,
            obs_receiver=obs_receiver,
            trig_sender=trig_sender,
        )

        # Verify process was created with injected dependencies
        assert process._obs_server is obs_receiver
        assert process._trig_client is trig_sender
        assert process._obs_server_provided is True
        assert process._trig_client_provided is True

    def test_missing_obs_receiver_and_socket_raises(self):
        """Test that missing both obs_receiver and obs_socket raises ValueError."""
        fusion = MockFusion()
        trig_sender = MockMessageSender()

        with pytest.raises(ValueError, match="Either obs_receiver or obs_socket"):
            FusionProcess(
                fusion=fusion,
                trig_sender=trig_sender,
            )

    def test_missing_trig_sender_and_socket_raises(self):
        """Test that missing both trig_sender and trig_socket raises ValueError."""
        fusion = MockFusion()
        obs_receiver = MockMessageReceiver()

        with pytest.raises(ValueError, match="Either trig_sender or trig_socket"):
            FusionProcess(
                fusion=fusion,
                obs_receiver=obs_receiver,
            )


class TestExtractorOrchestrator:
    """Tests for ExtractorOrchestrator."""

    def test_initialization(self):
        """Test orchestrator initialization."""
        extractors = [MockExtractor(name="face"), MockExtractor(name="pose")]
        orchestrator = ExtractorOrchestrator(extractors)

        assert not orchestrator.is_initialized
        orchestrator.initialize()
        assert orchestrator.is_initialized
        assert orchestrator.extractor_names == ["face", "pose"]

        orchestrator.cleanup()
        assert not orchestrator.is_initialized

    def test_context_manager(self):
        """Test context manager interface."""
        extractors = [MockExtractor(name="face")]

        with ExtractorOrchestrator(extractors) as orchestrator:
            assert orchestrator.is_initialized

        assert not orchestrator.is_initialized

    def test_extract_all_parallel(self):
        """Test parallel extraction."""
        extractors = [
            MockExtractor(name="face"),
            MockExtractor(name="pose"),
            MockExtractor(name="quality"),
        ]

        with ExtractorOrchestrator(extractors, max_workers=3) as orchestrator:
            data = np.zeros((480, 640, 3), dtype=np.uint8)
            frame = Frame.from_array(data, frame_id=1, t_src_ns=1_000_000_000)

            observations = orchestrator.extract_all(frame)

            assert len(observations) == 3
            sources = {obs.source for obs in observations}
            assert sources == {"face", "pose", "quality"}

    def test_extract_sequential(self):
        """Test sequential extraction."""
        extractors = [MockExtractor(name="face"), MockExtractor(name="pose")]

        with ExtractorOrchestrator(extractors) as orchestrator:
            data = np.zeros((480, 640, 3), dtype=np.uint8)
            frame = Frame.from_array(data, frame_id=1, t_src_ns=1_000_000_000)

            observations = orchestrator.extract_sequential(frame)

            assert len(observations) == 2

    def test_get_stats(self):
        """Test stats retrieval."""
        extractors = [MockExtractor(name="face")]

        with ExtractorOrchestrator(extractors) as orchestrator:
            data = np.zeros((480, 640, 3), dtype=np.uint8)
            frame = Frame.from_array(data, frame_id=1, t_src_ns=1_000_000_000)

            orchestrator.extract_all(frame)
            orchestrator.extract_all(frame)

            stats = orchestrator.get_stats()

            assert stats["frames_processed"] == 2
            assert stats["total_observations"] == 2
            assert stats["errors"] == 0
            assert stats["timeouts"] == 0
            assert "avg_time_ms" in stats

    def test_empty_extractors_raises(self):
        """Test that empty extractor list raises ValueError."""
        with pytest.raises(ValueError, match="At least one extractor"):
            ExtractorOrchestrator([])

    def test_not_initialized_raises(self):
        """Test that extract without initialize raises RuntimeError."""
        extractors = [MockExtractor(name="face")]
        orchestrator = ExtractorOrchestrator(extractors)

        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame.from_array(data, frame_id=1, t_src_ns=1_000_000_000)

        with pytest.raises(RuntimeError, match="not initialized"):
            orchestrator.extract_all(frame)

    def test_multiple_frames(self):
        """Test processing multiple frames."""
        extractors = [MockExtractor(name="face")]

        with ExtractorOrchestrator(extractors) as orchestrator:
            data = np.zeros((480, 640, 3), dtype=np.uint8)

            for i in range(5):
                frame = Frame.from_array(data, frame_id=i, t_src_ns=i * 1_000_000_000)
                observations = orchestrator.extract_all(frame)
                assert len(observations) == 1
                assert observations[0].frame_id == i

            stats = orchestrator.get_stats()
            assert stats["frames_processed"] == 5

