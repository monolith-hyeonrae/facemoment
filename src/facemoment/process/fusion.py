"""Fusion process wrapper for C module.

Receives OBS messages from multiple extractors, runs the fusion engine,
and sends TRIG messages to the Ingest process.

Architecture:
    B1 (face) ──┐
    B2 (pose) ──┼──→ C (fusion) ──→ A (ingest)
    B3 (quality)┘
         OBS             TRIG

Example:
    >>> from facemoment.moment_detector.fusion.highlight import HighlightFusion
    >>> process = FusionProcess(
    ...     fusion=HighlightFusion(),
    ...     obs_socket="/tmp/obs.sock",
    ...     trig_socket="/tmp/trig.sock",
    ... )
    >>> process.run()  # Blocking main loop
"""

import signal
import time
import heapq
import logging
from typing import Optional, List, Dict, Callable
import threading
from collections import defaultdict

from visualbase.ipc.uds import UDSServer, UDSClient
from visualbase.ipc.messages import (
    parse_obs_message,
    OBSMessage,
    TRIGMessage,
)

from facemoment.moment_detector.extractors.base import Observation, FaceObservation
from facemoment.moment_detector.fusion.base import BaseFusion, FusionResult

logger = logging.getLogger(__name__)


# Time window for observation alignment (100ms)
ALIGNMENT_WINDOW_NS = 100_000_000


class FusionProcess:
    """Wrapper for running fusion as an independent process.

    Receives OBS messages from extractors via UDS, converts them to
    Observations, runs the fusion engine, and sends TRIG messages to
    the ingest process.

    Args:
        fusion: The fusion engine instance.
        obs_socket: Path to the UDS socket for receiving OBS messages.
        trig_socket: Path to the UDS socket for sending TRIG messages.
        alignment_window_ns: Time window for observation alignment.
        on_trigger: Optional callback for each trigger.
    """

    def __init__(
        self,
        fusion: BaseFusion,
        obs_socket: str,
        trig_socket: str,
        alignment_window_ns: int = ALIGNMENT_WINDOW_NS,
        on_trigger: Optional[Callable[[FusionResult], None]] = None,
    ):
        self._fusion = fusion
        self._obs_socket = obs_socket
        self._trig_socket = trig_socket
        self._alignment_window_ns = alignment_window_ns
        self._on_trigger = on_trigger

        self._obs_server: Optional[UDSServer] = None
        self._trig_client: Optional[UDSClient] = None
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

        # Start OBS server
        self._obs_server = UDSServer(self._obs_socket)
        self._obs_server.start()

        # Connect to TRIG socket
        self._trig_client = UDSClient(self._trig_socket)
        if not self._trig_client.connect():
            logger.error(f"Failed to connect to TRIG socket: {self._trig_socket}")
            return

        logger.info(f"Fusion process started")
        logger.info(f"  OBS socket: {self._obs_socket}")
        logger.info(f"  TRIG socket: {self._trig_socket}")

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
            if current_t_ns - t_ns > self._alignment_window_ns:
                frames_to_process.append(frame_id)

        # Process in order
        for frame_id in sorted(frames_to_process):
            self._process_frame_observations(frame_id)
            del self._obs_buffer[frame_id]
            if frame_id in self._frame_timestamps:
                del self._frame_timestamps[frame_id]

    def _process_frame_observations(self, frame_id: int) -> None:
        """Process all observations for a single frame."""
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
            self._obs_server = None

        if self._trig_client:
            self._trig_client.disconnect()
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
