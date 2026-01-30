"""Fusion process wrapper for C module.

Receives OBS messages from multiple extractors, runs the fusion engine,
and sends TRIG messages to the Ingest process.

Architecture:
    B1 (face) ──┐
    B2 (pose) ──┼──→ C (fusion) ──→ A (ingest)
    B3 (quality)┘
         OBS             TRIG

Supports interface-based dependency injection for swappable transports.

Example (legacy path-based):
    >>> from facemoment.moment_detector.fusion.highlight import HighlightFusion
    >>> process = FusionProcess(
    ...     fusion=HighlightFusion(),
    ...     obs_socket="/tmp/obs.sock",
    ...     trig_socket="/tmp/trig.sock",
    ... )
    >>> process.run()  # Blocking main loop

Example (interface-based):
    >>> from visualbase.ipc.factory import TransportFactory
    >>> obs_receiver = TransportFactory.create_message_receiver("uds", "/tmp/obs.sock")
    >>> trig_sender = TransportFactory.create_message_sender("uds", "/tmp/trig.sock")
    >>> process = FusionProcess(
    ...     fusion=HighlightFusion(),
    ...     obs_receiver=obs_receiver,
    ...     trig_sender=trig_sender,
    ... )
    >>> process.run()
"""

import signal
import time
import heapq
import logging
from typing import Optional, List, Dict, Callable
import threading
from collections import defaultdict

from visualbase.ipc.interfaces import MessageReceiver, MessageSender
from visualbase.ipc.factory import TransportFactory
from visualbase.ipc.messages import (
    parse_obs_message,
    OBSMessage,
    TRIGMessage,
)

from facemoment.moment_detector.extractors.base import Observation, FaceObservation
from facemoment.moment_detector.fusion.base import BaseFusion, FusionResult
from facemoment.observability import ObservabilityHub
from facemoment.observability.records import TimingRecord, SyncDelayRecord

logger = logging.getLogger(__name__)

# Get the global observability hub
_hub = ObservabilityHub.get_instance()


# Time window for observation alignment (100ms)
ALIGNMENT_WINDOW_NS = 100_000_000


class FusionProcess:
    """Wrapper for running fusion as an independent process.

    Receives OBS messages from extractors via MessageReceiver, converts them to
    Observations, runs the fusion engine, and sends TRIG messages to
    the ingest process via MessageSender.

    Supports two initialization modes:
    1. Interface-based: Pass MessageReceiver and MessageSender instances directly
    2. Legacy path-based: Pass obs_socket and trig_socket paths (auto-creates UDS)

    Args:
        fusion: The fusion engine instance.
        obs_receiver: MessageReceiver instance for receiving OBS messages.
        trig_sender: MessageSender instance for sending TRIG messages.
        obs_socket: (Legacy) Path to the UDS socket for receiving OBS messages.
        trig_socket: (Legacy) Path to the UDS socket for sending TRIG messages.
        message_transport: Transport type for messages ("uds", "zmq"). Default: "uds".
        alignment_window_ns: Time window for observation alignment.
        on_trigger: Optional callback for each trigger.
    """

    def __init__(
        self,
        fusion: BaseFusion,
        obs_receiver: Optional[MessageReceiver] = None,
        trig_sender: Optional[MessageSender] = None,
        obs_socket: Optional[str] = None,
        trig_socket: Optional[str] = None,
        message_transport: str = "uds",
        alignment_window_ns: int = ALIGNMENT_WINDOW_NS,
        on_trigger: Optional[Callable[[FusionResult], None]] = None,
    ):
        self._fusion = fusion
        self._alignment_window_ns = alignment_window_ns
        self._on_trigger = on_trigger

        # Store transport config
        self._message_transport = message_transport
        self._obs_path = obs_socket
        self._trig_path = trig_socket

        # Interface-based or legacy path-based initialization
        if obs_receiver is not None:
            self._obs_server: Optional[MessageReceiver] = obs_receiver
            self._obs_server_provided = True
        elif obs_socket is not None:
            self._obs_server = None  # Created in run()
            self._obs_server_provided = False
        else:
            raise ValueError("Either obs_receiver or obs_socket must be provided")

        if trig_sender is not None:
            self._trig_client: Optional[MessageSender] = trig_sender
            self._trig_client_provided = True
        elif trig_socket is not None:
            self._trig_client = None  # Created in run()
            self._trig_client_provided = False
        else:
            raise ValueError("Either trig_sender or trig_socket must be provided")

        self._running = False
        self._shutdown = threading.Event()

        # Observation buffer for alignment (keyed by frame_id)
        self._obs_buffer: Dict[int, Dict[str, OBSMessage]] = defaultdict(dict)
        self._frame_timestamps: Dict[int, int] = {}  # frame_id -> t_ns

        # Stats
        self._obs_received = 0
        self._triggers_sent = 0
        self._errors = 0
        self._start_time: Optional[float] = None

    def run(self) -> None:
        """Run the fusion process main loop.

        This method blocks until stop() is called or the process is
        interrupted.
        """
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._running = True
        self._start_time = time.monotonic()

        # Create OBS receiver if not provided
        if self._obs_server is None and self._obs_path is not None:
            self._obs_server = TransportFactory.create_message_receiver(
                self._message_transport, self._obs_path
            )

        # Start OBS server
        if self._obs_server is None:
            logger.error("No OBS receiver available")
            return
        self._obs_server.start()

        # Create TRIG sender if not provided
        if self._trig_client is None and self._trig_path is not None:
            self._trig_client = TransportFactory.create_message_sender(
                self._message_transport, self._trig_path
            )

        # Connect to TRIG socket
        if self._trig_client is None or not self._trig_client.connect():
            logger.error(f"Failed to connect to TRIG socket: {self._trig_path}")
            return

        logger.info(f"Fusion process started")
        logger.info(f"  OBS receiver: {self._obs_path or 'provided'}")
        logger.info(f"  TRIG sender: {self._trig_path or 'provided'}")

        try:
            while self._running and not self._shutdown.is_set():
                self._process_loop()

        finally:
            self._cleanup()

    def _process_loop(self) -> None:
        """Main processing loop iteration."""
        # Receive all pending OBS messages
        messages = self._obs_server.recv_all(max_messages=100)

        for msg in messages:
            self._handle_obs_message(msg)

        # Process aligned observations
        self._process_aligned_observations()

        # Small sleep to prevent busy-waiting
        if not messages:
            time.sleep(0.001)  # 1ms

    def _handle_obs_message(self, msg: str) -> None:
        """Handle a received OBS message."""
        obs_msg = parse_obs_message(msg)
        if obs_msg is None:
            logger.warning(f"Failed to parse OBS message: {msg[:100]}")
            return

        self._obs_received += 1

        # Store in buffer by frame_id and source
        frame_id = obs_msg.frame_id
        self._obs_buffer[frame_id][obs_msg.src] = obs_msg
        self._frame_timestamps[frame_id] = obs_msg.t_ns

    def _process_aligned_observations(self) -> None:
        """Process observations that have been aligned by frame_id."""
        if not self._obs_buffer:
            return

        # Get oldest frame_id
        oldest_frame = min(self._obs_buffer.keys())
        oldest_t_ns = self._frame_timestamps.get(oldest_frame, 0)

        # Current time estimate (most recent observation)
        current_t_ns = max(self._frame_timestamps.values()) if self._frame_timestamps else 0

        # Process observations older than alignment window
        frames_to_process = []
        for frame_id in list(self._obs_buffer.keys()):
            t_ns = self._frame_timestamps.get(frame_id, 0)
            delay_ns = current_t_ns - t_ns
            if delay_ns > self._alignment_window_ns:
                frames_to_process.append(frame_id)

                # Emit sync delay record if significant delay
                if _hub.enabled and delay_ns > self._alignment_window_ns * 1.5:
                    obs_sources = list(self._obs_buffer.get(frame_id, {}).keys())
                    expected_sources = {"face", "pose", "quality"}
                    missing_sources = list(expected_sources - set(obs_sources))
                    if missing_sources:
                        _hub.emit(SyncDelayRecord(
                            frame_id=frame_id,
                            expected_ns=int(self._alignment_window_ns),
                            actual_ns=int(delay_ns),
                            delay_ms=(delay_ns - self._alignment_window_ns) / 1_000_000,
                            waiting_for=missing_sources,
                        ))

        # Process in order
        for frame_id in sorted(frames_to_process):
            self._process_frame_observations(frame_id)
            del self._obs_buffer[frame_id]
            if frame_id in self._frame_timestamps:
                del self._frame_timestamps[frame_id]

    def _process_frame_observations(self, frame_id: int) -> None:
        """Process all observations for a single frame."""
        start_ns = time.perf_counter_ns() if _hub.enabled else 0

        obs_dict = self._obs_buffer.get(frame_id, {})
        if not obs_dict:
            return

        # Convert OBS messages to Observations and feed to fusion
        for src, obs_msg in obs_dict.items():
            observation = self._obs_to_observation(obs_msg)
            if observation:
                try:
                    result = self._fusion.update(observation)
                    if result.should_trigger and result.trigger:
                        self._send_trigger(result)

                except Exception as e:
                    logger.error(f"Fusion error: {e}")
                    self._errors += 1

        # Emit timing record for fusion process
        if _hub.enabled:
            processing_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            _hub.emit(TimingRecord(
                frame_id=frame_id,
                component="fusion_process",
                processing_ms=processing_ms,
                queue_depth=len(self._obs_buffer),
                threshold_ms=50.0,
                is_slow=processing_ms > 50.0,
            ))

    def _obs_to_observation(self, obs_msg: OBSMessage) -> Optional[Observation]:
        """Convert an OBSMessage to an Observation."""
        obs = Observation(
            source=obs_msg.src,
            frame_id=obs_msg.frame_id,
            t_ns=obs_msg.t_ns,
        )

        if obs_msg.src == "face":
            for face in obs_msg.faces:
                obs.faces.append(FaceObservation(
                    face_id=face.id,
                    confidence=face.conf,
                    bbox=(face.x, face.y, face.w, face.h),
                    yaw=face.yaw,
                    pitch=face.pitch,
                    expression=face.expr,
                ))

        elif obs_msg.src == "pose":
            for pose in obs_msg.poses:
                obs.signals["hand_raised"] = 1.0 if pose.hand_raised else 0.0
                obs.signals["hand_wave"] = 1.0 if pose.hand_wave else 0.0
                obs.signals["wave_count"] = float(pose.wave_count)
                obs.signals["confidence"] = pose.conf

        elif obs_msg.src == "quality":
            if obs_msg.quality:
                obs.signals["blur_score"] = obs_msg.quality.blur
                obs.signals["brightness"] = obs_msg.quality.brightness
                obs.signals["contrast"] = obs_msg.quality.contrast
                obs.signals["quality_gate"] = 1.0 if obs_msg.quality.gate_open else 0.0

        return obs

    def _send_trigger(self, result: FusionResult) -> None:
        """Send a TRIG message."""
        if not result.trigger:
            return

        trigger = result.trigger
        trig_msg = TRIGMessage(
            label=trigger.label or "PORTRAIT_HIGHLIGHT",
            t_start_ns=trigger.clip_start_ns,
            t_end_ns=trigger.clip_end_ns,
            faces=len(result.metadata.get("faces", [])),
            score=result.score,
            reason=result.reason,
        )

        if self._trig_client and self._trig_client.send(trig_msg.to_message()):
            self._triggers_sent += 1
            logger.info(
                f"Trigger sent: {trig_msg.label} score={trig_msg.score:.2f} "
                f"reason={trig_msg.reason}"
            )

            # Call trigger callback if set
            if self._on_trigger:
                self._on_trigger(result)

    def stop(self) -> None:
        """Stop the fusion process."""
        logger.info("Stopping fusion process")
        self._running = False
        self._shutdown.set()

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._obs_server:
            self._obs_server.stop()
            if not self._obs_server_provided:
                self._obs_server = None

        if self._trig_client:
            self._trig_client.disconnect()
            if not self._trig_client_provided:
                self._trig_client = None

        # Log stats
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        logger.info(
            f"Fusion process stopped: "
            f"{self._obs_received} obs, {self._triggers_sent} triggers, "
            f"{self._errors} errors in {elapsed:.1f}s"
        )

    def _signal_handler(self, signum, frame) -> None:
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    @property
    def is_running(self) -> bool:
        """Check if the process is running."""
        return self._running

    def get_stats(self) -> dict:
        """Get processing statistics."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        return {
            "obs_received": self._obs_received,
            "triggers_sent": self._triggers_sent,
            "errors": self._errors,
            "elapsed_sec": elapsed,
            "buffer_frames": len(self._obs_buffer),
        }
