"""Process module for A-B*-C architecture.

This module provides process wrappers for running extractors and fusion
as independent processes with IPC communication.

The generic IPC classes are provided by visualpath, and facemoment provides
domain-specific observation mappers for serialization.

Components:
- ExtractorProcess: Wraps a BaseExtractor for standalone IPC execution
- FusionProcess: Wraps a BaseFusion for standalone IPC execution
- ExtractorOrchestrator: Thread-parallel extractor execution for Library mode
- FacemomentMapper: Domain-specific Observation ↔ Message mappers

Factory Functions:
- create_extractor_process: Create ExtractorProcess with facemoment mapper
- create_fusion_process: Create FusionProcess with facemoment mapper

Example (using factory functions):
    >>> from facemoment.process import create_extractor_process
    >>> from facemoment.moment_detector.extractors.face import FaceExtractor
    >>>
    >>> process = create_extractor_process(
    ...     extractor=FaceExtractor(),
    ...     input_fifo="/tmp/vid_face.mjpg",
    ...     obs_socket="/tmp/obs.sock",
    ... )
    >>> process.run()

Example (direct use with custom mapper):
    >>> from facemoment.process import ExtractorProcess, FacemomentMapper
    >>>
    >>> process = ExtractorProcess(
    ...     extractor=FaceExtractor(),
    ...     observation_mapper=FacemomentMapper(),
    ...     video_reader=reader,
    ...     message_sender=sender,
    ... )
"""

from typing import Optional, Callable

# Re-export from visualpath
from visualpath.process import (
    ExtractorProcess,
    FusionProcess,
    ExtractorOrchestrator,
    ALIGNMENT_WINDOW_NS,
    # Mappers
    ObservationMapper,
    DefaultObservationMapper,
    CompositeMapper,
)

# Import facemoment-specific mappers
from facemoment.process.mappers import (
    FaceObservationMapper,
    PoseObservationMapper,
    QualityObservationMapper,
    FacemomentMapper,
)

# Re-export base types for convenience
from visualpath.core.extractor import BaseExtractor, Observation
from visualpath.core.fusion import BaseFusion, FusionResult
from visualbase.ipc.interfaces import VideoReader, MessageSender, MessageReceiver
from visualbase import Frame


def create_extractor_process(
    extractor: BaseExtractor,
    video_reader: Optional[VideoReader] = None,
    message_sender: Optional[MessageSender] = None,
    input_fifo: Optional[str] = None,
    obs_socket: Optional[str] = None,
    video_transport: str = "fifo",
    message_transport: str = "uds",
    reconnect: bool = True,
    on_frame: Optional[Callable[[Frame, Observation], None]] = None,
) -> ExtractorProcess:
    """Create an ExtractorProcess with facemoment-specific mapper.

    This is a convenience factory that automatically uses FacemomentMapper
    for Observation ↔ Message serialization.

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

    Returns:
        ExtractorProcess configured with FacemomentMapper.

    Example:
        >>> from facemoment.process import create_extractor_process
        >>> from facemoment.moment_detector.extractors.face import FaceExtractor
        >>>
        >>> process = create_extractor_process(
        ...     extractor=FaceExtractor(),
        ...     input_fifo="/tmp/vid_face.mjpg",
        ...     obs_socket="/tmp/obs.sock",
        ... )
        >>> process.run()
    """
    return ExtractorProcess(
        extractor=extractor,
        observation_mapper=FacemomentMapper(),
        video_reader=video_reader,
        message_sender=message_sender,
        input_fifo=input_fifo,
        obs_socket=obs_socket,
        video_transport=video_transport,
        message_transport=message_transport,
        reconnect=reconnect,
        on_frame=on_frame,
    )


def create_fusion_process(
    fusion: BaseFusion,
    obs_receiver: Optional[MessageReceiver] = None,
    trig_sender: Optional[MessageSender] = None,
    obs_socket: Optional[str] = None,
    trig_socket: Optional[str] = None,
    message_transport: str = "uds",
    alignment_window_ns: int = ALIGNMENT_WINDOW_NS,
    on_trigger: Optional[Callable[[FusionResult], None]] = None,
) -> FusionProcess:
    """Create a FusionProcess with facemoment-specific mapper.

    This is a convenience factory that automatically uses FacemomentMapper
    for Message → Observation deserialization.

    Args:
        fusion: The fusion engine instance.
        obs_receiver: MessageReceiver instance for receiving OBS messages.
        trig_sender: MessageSender instance for sending TRIG messages.
        obs_socket: (Legacy) Path to the UDS socket for receiving OBS messages.
        trig_socket: (Legacy) Path to the UDS socket for sending TRIG messages.
        message_transport: Transport type for messages ("uds", "zmq"). Default: "uds".
        alignment_window_ns: Time window for observation alignment.
        on_trigger: Optional callback for each trigger.

    Returns:
        FusionProcess configured with FacemomentMapper.

    Example:
        >>> from facemoment.process import create_fusion_process
        >>> from facemoment.moment_detector.fusion.highlight import HighlightFusion
        >>>
        >>> process = create_fusion_process(
        ...     fusion=HighlightFusion(),
        ...     obs_socket="/tmp/obs.sock",
        ...     trig_socket="/tmp/trig.sock",
        ... )
        >>> process.run()
    """
    return FusionProcess(
        fusion=fusion,
        observation_mapper=FacemomentMapper(),
        obs_receiver=obs_receiver,
        trig_sender=trig_sender,
        obs_socket=obs_socket,
        trig_socket=trig_socket,
        message_transport=message_transport,
        alignment_window_ns=alignment_window_ns,
        on_trigger=on_trigger,
    )


__all__ = [
    # Process wrappers (from visualpath)
    "ExtractorProcess",
    "FusionProcess",
    "ExtractorOrchestrator",
    "ALIGNMENT_WINDOW_NS",
    # Mappers (from visualpath)
    "ObservationMapper",
    "DefaultObservationMapper",
    "CompositeMapper",
    # Facemoment-specific mappers
    "FaceObservationMapper",
    "PoseObservationMapper",
    "QualityObservationMapper",
    "FacemomentMapper",
    # Factory functions
    "create_extractor_process",
    "create_fusion_process",
]
