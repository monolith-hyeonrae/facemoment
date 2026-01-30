"""Integration tests for A-B*-C process architecture.

Tests the full flow:
- ExtractorProcess with interface-based dependencies
- FusionProcess with interface-based dependencies
- ExtractorOrchestrator for Library mode
"""

import threading
import time
from typing import List, Optional

import pytest
import numpy as np

from visualbase.core.frame import Frame
from visualbase.ipc.interfaces import VideoReader, MessageSender, MessageReceiver

from facemoment.moment_detector.extractors.base import (
    BaseExtractor,
    Observation,
    FaceObservation,
)
from facemoment.moment_detector.fusion.base import BaseFusion, FusionResult, Trigger
from facemoment.process.extractor import ExtractorProcess
from facemoment.process.fusion import FusionProcess
from facemoment.process.orchestrator import ExtractorOrchestrator


class MockVideoReader(VideoReader):
    """Mock video reader that provides test frames."""

    def __init__(self, frames: Optional[List[Frame]] = None, delay_ms: float = 0):
        self._frames = frames or []
        self._index = 0
        self._is_open = False
        self._delay_ms = delay_ms

    def open(self, timeout_sec: Optional[float] = None) -> bool:
        self._is_open = True
        self._index = 0
        return True

    def read(self) -> Optional[Frame]:
        if self._delay_ms > 0:
            time.sleep(self._delay_ms / 1000)
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
    """Mock message sender that records sent messages."""

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


class MockMessageReceiver(MessageReceiver):
    """Mock message receiver that provides test messages."""

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


class MockExtractor(BaseExtractor):
    """Mock extractor for testing."""

    def __init__(self, name: str = "mock"):
        self._name = name
        self._call_count = 0

    @property
    def name(self) -> str:
        return self._name

    def extract(self, frame: Frame) -> Observation:
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
        return Observation(
            source=self._name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
        )


class MockFusion(BaseFusion):
    """Mock fusion that triggers every N observations."""

    def __init__(self, trigger_interval: int = 5):
        self._observations = []
        self._trigger_interval = trigger_interval
        self._gate_open = True
        self._in_cooldown = False

    def update(self, observation: Observation) -> FusionResult:
        self._observations.append(observation)
        should_trigger = len(self._observations) % self._trigger_interval == 0

        trigger = None
        if should_trigger:
            trigger = Trigger.point(
                event_time_ns=observation.t_ns,
                pre_sec=2.0,
                post_sec=1.0,
                label="test_trigger",
                score=0.85,
            )

        return FusionResult(
            should_trigger=should_trigger,
            trigger=trigger,
            score=0.85 if should_trigger else 0.0,
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


class TestExtractorProcessIntegration:
    """Integration tests for ExtractorProcess."""

    def test_full_extraction_pipeline(self):
        """Test complete extraction pipeline with mock interfaces."""
        # Create test frames
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frames = [
            Frame.from_array(data, frame_id=i, t_src_ns=i * 100_000_000)
            for i in range(10)
        ]

        reader = MockVideoReader(frames=frames)
        sender = MockMessageSender()
        extractor = MockExtractor(name="face")

        process = ExtractorProcess(
            extractor=extractor,
            video_reader=reader,
            message_sender=sender,
            reconnect=False,
        )

        # Run process in thread
        def run_with_stop():
            time.sleep(0.2)
            process.stop()

        stop_thread = threading.Thread(target=run_with_stop)
        stop_thread.start()

        process.run()
        stop_thread.join()

        # Verify results
        assert process._frames_processed == 10
        assert len(sender._messages) == 10

        # Verify message format
        for msg in sender._messages:
            assert msg.startswith("OBS src=face")
            assert "faces=1" in msg

    def test_multiple_extractor_types(self):
        """Test different extractor types produce correct message types."""
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frames = [Frame.from_array(data, frame_id=0, t_src_ns=0)]

        for ext_type in ["face", "pose", "quality"]:
            reader = MockVideoReader(frames=frames.copy())
            sender = MockMessageSender()
            extractor = MockExtractor(name=ext_type)

            process = ExtractorProcess(
                extractor=extractor,
                video_reader=reader,
                message_sender=sender,
                reconnect=False,
            )

            def run_with_stop():
                time.sleep(0.1)
                process.stop()

            stop_thread = threading.Thread(target=run_with_stop)
            stop_thread.start()
            process.run()
            stop_thread.join()

            assert len(sender._messages) == 1
            assert f"OBS src={ext_type}" in sender._messages[0]


class TestFusionProcessIntegration:
    """Integration tests for FusionProcess."""

    def test_obs_to_trig_pipeline(self):
        """Test OBS messages trigger TRIG messages correctly."""
        # Create OBS messages for 5 frames (trigger on frame 5)
        obs_messages = [
            f"OBS src=face frame={i} t={i*100_000_000} faces=1 "
            f"id:0,conf:0.95,x:0.1,y:0.2,w:0.3,h:0.4,expr:0.8"
            for i in range(5)
        ]

        obs_receiver = MockMessageReceiver(messages=obs_messages)
        trig_sender = MockMessageSender()
        fusion = MockFusion(trigger_interval=5)

        process = FusionProcess(
            fusion=fusion,
            obs_receiver=obs_receiver,
            trig_sender=trig_sender,
            alignment_window_ns=0,  # No alignment delay for testing
        )

        # Run briefly
        def run_with_stop():
            time.sleep(0.2)
            process.stop()

        stop_thread = threading.Thread(target=run_with_stop)
        stop_thread.start()
        process.run()
        stop_thread.join()

        # Verify OBS received
        assert process._obs_received == 5


class TestExtractorOrchestratorIntegration:
    """Integration tests for ExtractorOrchestrator."""

    def test_parallel_extraction(self):
        """Test parallel extraction produces correct results."""
        extractors = [
            MockExtractor(name="face"),
            MockExtractor(name="pose"),
            MockExtractor(name="quality"),
        ]

        with ExtractorOrchestrator(extractors, max_workers=3) as orchestrator:
            data = np.zeros((480, 640, 3), dtype=np.uint8)

            for i in range(5):
                frame = Frame.from_array(data, frame_id=i, t_src_ns=i * 100_000_000)
                observations = orchestrator.extract_all(frame)

                # Should get observations from all 3 extractors
                assert len(observations) == 3
                sources = {obs.source for obs in observations}
                assert sources == {"face", "pose", "quality"}

            stats = orchestrator.get_stats()
            assert stats["frames_processed"] == 5
            assert stats["total_observations"] == 15  # 3 per frame

    def test_sequential_vs_parallel_consistency(self):
        """Test sequential and parallel modes produce same results."""
        extractors = [MockExtractor(name="face"), MockExtractor(name="pose")]

        with ExtractorOrchestrator(extractors, max_workers=2) as orchestrator:
            data = np.zeros((480, 640, 3), dtype=np.uint8)
            frame = Frame.from_array(data, frame_id=0, t_src_ns=0)

            parallel_obs = orchestrator.extract_all(frame)
            sequential_obs = orchestrator.extract_sequential(frame)

            # Should have same sources
            parallel_sources = {obs.source for obs in parallel_obs}
            sequential_sources = {obs.source for obs in sequential_obs}
            assert parallel_sources == sequential_sources


class TestEndToEndFlow:
    """End-to-end tests for the complete A-B*-C flow."""

    def test_extractor_message_format(self):
        """Test extractor produces correctly formatted OBS messages.

        This tests the message format that would flow between
        ExtractorProcess and FusionProcess in the A-B*-C architecture.
        """
        # Create a single frame
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame.from_array(data, frame_id=1, t_src_ns=100_000_000)

        # Extract and convert to message
        extractor = MockExtractor(name="face")
        reader = MockVideoReader(frames=[frame])
        sender = MockMessageSender()

        process = ExtractorProcess(
            extractor=extractor,
            video_reader=reader,
            message_sender=sender,
            reconnect=False,
        )

        # Run in thread with stop
        def run_and_stop():
            time.sleep(0.1)
            process.stop()

        stop_thread = threading.Thread(target=run_and_stop)
        stop_thread.start()
        process.run()
        stop_thread.join()

        # Should have generated one OBS message
        assert len(sender._messages) == 1
        msg = sender._messages[0]

        # Verify message format
        assert msg.startswith("OBS src=face")
        assert "frame=1" in msg
        assert "faces=1" in msg

        # Now verify this message can be parsed by FusionProcess
        from visualbase.ipc.messages import parse_obs_message
        parsed = parse_obs_message(msg)
        assert parsed is not None
        assert parsed.src == "face"
        assert parsed.frame_id == 1
        assert len(parsed.faces) == 1

    def test_orchestrator_produces_fusable_observations(self):
        """Test ExtractorOrchestrator output can be used by Fusion."""
        extractors = [
            MockExtractor(name="face"),
            MockExtractor(name="pose"),
            MockExtractor(name="quality"),
        ]

        fusion = MockFusion(trigger_interval=3)

        with ExtractorOrchestrator(extractors, max_workers=3) as orchestrator:
            data = np.zeros((480, 640, 3), dtype=np.uint8)

            for i in range(6):
                frame = Frame.from_array(data, frame_id=i, t_src_ns=i * 100_000_000)
                observations = orchestrator.extract_all(frame)

                # Feed each observation to fusion
                for obs in observations:
                    result = fusion.update(obs)

                # Frame 0: 3 obs (trigger at obs 3)
                # Frame 1: 6 obs (trigger at obs 6)
                # ...

        # 6 frames * 3 extractors = 18 observations
        # Triggers at obs 3, 6, 9, 12, 15, 18 = 6 triggers
        assert len(fusion._observations) == 18
