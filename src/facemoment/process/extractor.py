"""Extractor process wrapper for B* modules.

Wraps a BaseExtractor to run as an independent process:
1. Reads frames from a VideoReader (from Ingest/A)
2. Runs extractor.extract() on each frame
3. Sends OBS messages via MessageSender (to Fusion/C)

Supports interface-based dependency injection for swappable transports.

Example (legacy path-based):
    >>> from facemoment.moment_detector.extractors.face import FaceExtractor
    >>> process = ExtractorProcess(
    ...     extractor=FaceExtractor(),
    ...     input_fifo="/tmp/vid_face.mjpg",
    ...     obs_socket="/tmp/obs.sock",
    ... )
    >>> process.run()  # Blocking main loop

Example (interface-based):
    >>> from visualbase.ipc.factory import TransportFactory
    >>> reader = TransportFactory.create_video_reader("fifo", "/tmp/vid_face.mjpg")
    >>> sender = TransportFactory.create_message_sender("uds", "/tmp/obs.sock")
    >>> process = ExtractorProcess(
    ...     extractor=FaceExtractor(),
    ...     video_reader=reader,
    ...     message_sender=sender,
    ... )
    >>> process.run()
"""

import signal
import time
import logging
from typing import Optional, Callable, Union, List
import threading

from visualbase.ipc.interfaces import VideoReader, MessageSender
from visualbase.ipc.factory import TransportFactory
from visualbase.ipc.messages import FaceOBS, PoseOBS, QualityOBS, FaceData, PoseData, QualityData
from visualbase import Frame

from facemoment.moment_detector.extractors.base import BaseExtractor, Observation
from facemoment.observability import ObservabilityHub
from facemoment.observability.records import TimingRecord, FrameDropRecord

logger = logging.getLogger(__name__)

# Get the global observability hub
_hub = ObservabilityHub.get_instance()


class ExtractorProcess:
    """Wrapper for running an extractor as an independent process.

    Reads frames from a VideoReader input, processes them with the extractor,
    and sends OBS messages to the fusion process via MessageSender.

    Supports two initialization modes:
    1. Interface-based: Pass VideoReader and MessageSender instances directly
    2. Legacy path-based: Pass input_fifo and obs_socket paths (auto-creates FIFO/UDS)

    Args:
        extractor: The extractor instance to use.
        video_reader: VideoReader instance for receiving frames.
        message_sender: MessageSender instance for sending OBS messages.
        input_fifo: (Legacy) Path to the FIFO for receiving frames.
        obs_socket: (Legacy) Path to the UDS socket for sending OBS messages.
        video_transport: Transport type for video ("fifo", "zmq"). Default: "fifo".
        message_transport: Transport type for messages ("uds", "zmq"). Default: "uds".
        reconnect: Whether to reconnect on reader disconnect.
        on_frame: Optional callback for each processed frame.
    """

    def __init__(
        self,
        extractor: BaseExtractor,
        video_reader: Optional[VideoReader] = None,
        message_sender: Optional[MessageSender] = None,
        input_fifo: Optional[str] = None,
        obs_socket: Optional[str] = None,
        video_transport: str = "fifo",
        message_transport: str = "uds",
        reconnect: bool = True,
        on_frame: Optional[Callable[[Frame, Observation], None]] = None,
    ):
        self._extractor = extractor
        self._reconnect = reconnect
        self._on_frame = on_frame

        # Store transport config for reconnection
        self._video_transport = video_transport
        self._message_transport = message_transport
        self._input_path = input_fifo
        self._obs_path = obs_socket

        # Interface-based or legacy path-based initialization
        if video_reader is not None:
            self._reader: Optional[VideoReader] = video_reader
            self._reader_provided = True
        elif input_fifo is not None:
            self._reader = None  # Created in run()
            self._reader_provided = False
        else:
            raise ValueError("Either video_reader or input_fifo must be provided")

        if message_sender is not None:
            self._client: Optional[MessageSender] = message_sender
            self._client_provided = True
        elif obs_socket is not None:
            self._client = None  # Created in run()
            self._client_provided = False
        else:
            raise ValueError("Either message_sender or obs_socket must be provided")

        self._running = False
        self._shutdown = threading.Event()

        # Stats
        self._frames_processed = 0
        self._obs_sent = 0
        self._errors = 0
        self._start_time: Optional[float] = None

        # Observability tracking
        self._dropped_frames: List[int] = []
        self._last_frame_id: Optional[int] = None

    def run(self) -> None:
        """Run the extractor process main loop.

        This method blocks until stop() is called or the process is
        interrupted.
        """
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._running = True
        self._start_time = time.monotonic()

        # Initialize extractor
        logger.info(f"Initializing extractor: {self._extractor.name}")
        self._extractor.initialize()

        # Create message sender if not provided
        if self._client is None and self._obs_path is not None:
            self._client = TransportFactory.create_message_sender(
                self._message_transport, self._obs_path
            )

        # Connect to OBS socket
        if self._client is None or not self._client.connect():
            logger.error(f"Failed to connect to OBS socket: {self._obs_path}")
            return

        try:
            while self._running and not self._shutdown.is_set():
                self._run_once()

                if not self._running:
                    break

                if self._reconnect:
                    logger.info("Reader disconnected, waiting to reconnect...")
                    time.sleep(1.0)
                else:
                    break

        finally:
            self._cleanup()

    def _run_once(self) -> None:
        """Run one session (until reader disconnects)."""
        # Create reader if not provided (legacy path-based mode)
        if not self._reader_provided:
            if self._input_path is None:
                logger.error("No input path configured")
                return
            self._reader = TransportFactory.create_video_reader(
                self._video_transport, self._input_path
            )

        if self._reader is None:
            logger.error("No video reader available")
            return

        if not self._reader.open():
            logger.warning(f"Failed to open reader: {self._input_path}")
            return

        logger.info(f"Connected to reader: {self._input_path or 'provided reader'}")

        try:
            for frame in self._reader:
                if not self._running or self._shutdown.is_set():
                    break

                self._process_frame(frame)

        except Exception as e:
            logger.error(f"Error in extractor loop: {e}")
            self._errors += 1

        finally:
            if self._reader:
                self._reader.close()
                # Only clear reader if we created it (not provided)
                if not self._reader_provided:
                    self._reader = None

    def _process_frame(self, frame: Frame) -> None:
        """Process a single frame."""
        start_ns = time.perf_counter_ns() if _hub.enabled else 0

        try:
            # Check for dropped frames
            if _hub.enabled and self._last_frame_id is not None:
                expected_frame_id = self._last_frame_id + 1
                if frame.frame_id > expected_frame_id:
                    dropped = list(range(expected_frame_id, frame.frame_id))
                    self._dropped_frames.extend(dropped)
                    _hub.emit(FrameDropRecord(
                        dropped_frame_ids=dropped,
                        reason="gap_in_sequence",
                    ))
            self._last_frame_id = frame.frame_id

            # Extract features
            obs = self._extractor.extract(frame)
            self._frames_processed += 1

            if obs is not None:
                # Convert to OBS message and send
                message = self._observation_to_message(obs)
                if message and self._client:
                    if self._client.send(message):
                        self._obs_sent += 1

                # Call frame callback if set
                if self._on_frame:
                    self._on_frame(frame, obs)

            # Emit timing record for process layer
            if _hub.enabled:
                processing_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
                _hub.emit(TimingRecord(
                    frame_id=frame.frame_id,
                    component=f"process_{self._extractor.name}",
                    processing_ms=processing_ms,
                    threshold_ms=100.0,
                    is_slow=processing_ms > 100.0,
                ))

        except Exception as e:
            logger.warning(f"Frame processing error: {e}")
            self._errors += 1

    def _observation_to_message(self, obs: Observation) -> Optional[str]:
        """Convert an Observation to an OBS message string."""
        src = obs.source

        if src == "face":
            faces = []
            for face_obs in obs.faces:
                x, y, w, h = face_obs.bbox
                faces.append(FaceData(
                    id=face_obs.face_id,
                    conf=face_obs.confidence,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    expr=face_obs.expression,
                    yaw=face_obs.yaw,
                    pitch=face_obs.pitch,
                ))
            return FaceOBS(
                frame_id=obs.frame_id,
                t_ns=obs.t_ns,
                faces=faces,
            ).to_message()

        elif src == "pose":
            poses = []
            # Extract pose data from signals
            hand_raised = obs.signals.get("hand_raised", 0) > 0.5
            hand_wave = obs.signals.get("hand_wave", 0) > 0.5
            wave_count = int(obs.signals.get("wave_count", 0))
            conf = obs.signals.get("confidence", 0.5)
            poses.append(PoseData(
                id=0,
                conf=conf,
                hand_raised=hand_raised,
                hand_wave=hand_wave,
                wave_count=wave_count,
            ))
            return PoseOBS(
                frame_id=obs.frame_id,
                t_ns=obs.t_ns,
                poses=poses,
            ).to_message()

        elif src == "quality":
            blur = obs.signals.get("blur_score", 0)
            brightness = obs.signals.get("brightness", 128)
            contrast = obs.signals.get("contrast", 0.5)
            gate_open = obs.signals.get("quality_gate", 0) > 0.5
            return QualityOBS(
                frame_id=obs.frame_id,
                t_ns=obs.t_ns,
                quality=QualityData(
                    blur=blur,
                    brightness=brightness,
                    contrast=contrast,
                    gate_open=gate_open,
                ),
            ).to_message()

        else:
            logger.warning(f"Unknown observation source: {src}")
            return None

    def stop(self) -> None:
        """Stop the extractor process."""
        logger.info(f"Stopping extractor process: {self._extractor.name}")
        self._running = False
        self._shutdown.set()

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._reader:
            self._reader.close()
            if not self._reader_provided:
                self._reader = None

        if self._client:
            self._client.disconnect()
            if not self._client_provided:
                self._client = None

        self._extractor.cleanup()

        # Log stats
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        fps = self._frames_processed / elapsed if elapsed > 0 else 0
        logger.info(
            f"Extractor '{self._extractor.name}' stopped: "
            f"{self._frames_processed} frames, {self._obs_sent} obs, "
            f"{fps:.1f} fps, {self._errors} errors"
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
            "frames_processed": self._frames_processed,
            "obs_sent": self._obs_sent,
            "errors": self._errors,
            "elapsed_sec": elapsed,
            "fps": self._frames_processed / elapsed if elapsed > 0 else 0,
        }
