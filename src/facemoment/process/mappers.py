"""Observation mappers for facemoment IPC communication.

This module provides domain-specific mappers for converting facemoment
Observations to/from visualbase IPC message formats.

Mapper implementations:
- FaceObservationMapper: Handles FaceObservation ↔ FaceOBS
- PoseObservationMapper: Handles pose Observation ↔ PoseOBS
- QualityObservationMapper: Handles quality Observation ↔ QualityOBS
- FacemomentMapper: Composite mapper that handles all facemoment observation types

Example:
    >>> from facemoment.process.mappers import FacemomentMapper
    >>> from visualpath.process import ExtractorProcess
    >>>
    >>> mapper = FacemomentMapper()
    >>> process = ExtractorProcess(
    ...     extractor=FaceExtractor(),
    ...     observation_mapper=mapper,
    ...     ...
    ... )
"""

from typing import Optional

from visualbase.ipc.messages import (
    FaceOBS,
    PoseOBS,
    QualityOBS,
    FaceData,
    PoseData,
    QualityData,
    parse_obs_message,
    OBSMessage,
)
from visualpath.process.mapper import ObservationMapper, CompositeMapper

from facemoment.moment_detector.extractors.base import Observation, FaceObservation


class FaceObservationMapper:
    """Mapper for FaceObservation ↔ FaceOBS message conversion.

    Handles serialization of face observations including:
    - Face bounding boxes
    - Confidence scores
    - Expression intensity
    - Head pose (yaw, pitch)

    Example:
        >>> mapper = FaceObservationMapper()
        >>> obs = FaceObservation(source="face", frame_id=1, ...)
        >>> message = mapper.to_message(obs)  # -> "OBS src=face ..."
        >>> restored = mapper.from_message(message)
    """

    def to_message(self, observation: Observation) -> Optional[str]:
        """Convert a face Observation to FaceOBS message.

        Args:
            observation: Observation with source="face".

        Returns:
            Serialized FaceOBS message string, or None if not a face observation.
        """
        if observation.source != "face":
            return None

        faces = []
        for face_obs in observation.faces:
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
            frame_id=observation.frame_id,
            t_ns=observation.t_ns,
            faces=faces,
        ).to_message()

    def from_message(self, message: str) -> Optional[Observation]:
        """Convert a FaceOBS message to Observation.

        Args:
            message: Raw OBS message string.

        Returns:
            Observation with face data, or None if not parseable.
        """
        obs_msg = parse_obs_message(message)
        if obs_msg is None or obs_msg.src != "face":
            return None

        obs = Observation(
            source="face",
            frame_id=obs_msg.frame_id,
            t_ns=obs_msg.t_ns,
        )

        for face in obs_msg.faces:
            obs.faces.append(FaceObservation(
                face_id=face.id,
                confidence=face.conf,
                bbox=(face.x, face.y, face.w, face.h),
                yaw=face.yaw,
                pitch=face.pitch,
                expression=face.expr,
            ))

        return obs


class PoseObservationMapper:
    """Mapper for pose Observation ↔ PoseOBS message conversion.

    Handles serialization of pose observations including:
    - Hand raised flag
    - Hand wave detection
    - Wave count
    - Confidence score

    Example:
        >>> mapper = PoseObservationMapper()
        >>> obs = Observation(source="pose", signals={"hand_raised": 1.0}, ...)
        >>> message = mapper.to_message(obs)
    """

    def to_message(self, observation: Observation) -> Optional[str]:
        """Convert a pose Observation to PoseOBS message.

        Args:
            observation: Observation with source="pose".

        Returns:
            Serialized PoseOBS message string, or None if not a pose observation.
        """
        if observation.source != "pose":
            return None

        # Extract pose data from signals
        hand_raised = observation.signals.get("hand_raised", 0) > 0.5
        hand_wave = observation.signals.get("hand_wave", 0) > 0.5
        wave_count = int(observation.signals.get("wave_count", 0))
        conf = observation.signals.get("confidence", 0.5)

        poses = [PoseData(
            id=0,
            conf=conf,
            hand_raised=hand_raised,
            hand_wave=hand_wave,
            wave_count=wave_count,
        )]

        return PoseOBS(
            frame_id=observation.frame_id,
            t_ns=observation.t_ns,
            poses=poses,
        ).to_message()

    def from_message(self, message: str) -> Optional[Observation]:
        """Convert a PoseOBS message to Observation.

        Args:
            message: Raw OBS message string.

        Returns:
            Observation with pose signals, or None if not parseable.
        """
        obs_msg = parse_obs_message(message)
        if obs_msg is None or obs_msg.src != "pose":
            return None

        obs = Observation(
            source="pose",
            frame_id=obs_msg.frame_id,
            t_ns=obs_msg.t_ns,
        )

        for pose in obs_msg.poses:
            obs.signals["hand_raised"] = 1.0 if pose.hand_raised else 0.0
            obs.signals["hand_wave"] = 1.0 if pose.hand_wave else 0.0
            obs.signals["wave_count"] = float(pose.wave_count)
            obs.signals["confidence"] = pose.conf

        return obs


class QualityObservationMapper:
    """Mapper for quality Observation ↔ QualityOBS message conversion.

    Handles serialization of quality observations including:
    - Blur score (laplacian variance)
    - Brightness (0-255)
    - Contrast (0-1)
    - Quality gate state

    Example:
        >>> mapper = QualityObservationMapper()
        >>> obs = Observation(source="quality", signals={"blur_score": 100}, ...)
        >>> message = mapper.to_message(obs)
    """

    def to_message(self, observation: Observation) -> Optional[str]:
        """Convert a quality Observation to QualityOBS message.

        Args:
            observation: Observation with source="quality".

        Returns:
            Serialized QualityOBS message string, or None if not a quality observation.
        """
        if observation.source != "quality":
            return None

        blur = observation.signals.get("blur_score", 0)
        brightness = observation.signals.get("brightness", 128)
        contrast = observation.signals.get("contrast", 0.5)
        gate_open = observation.signals.get("quality_gate", 0) > 0.5

        return QualityOBS(
            frame_id=observation.frame_id,
            t_ns=observation.t_ns,
            quality=QualityData(
                blur=blur,
                brightness=brightness,
                contrast=contrast,
                gate_open=gate_open,
            ),
        ).to_message()

    def from_message(self, message: str) -> Optional[Observation]:
        """Convert a QualityOBS message to Observation.

        Args:
            message: Raw OBS message string.

        Returns:
            Observation with quality signals, or None if not parseable.
        """
        obs_msg = parse_obs_message(message)
        if obs_msg is None or obs_msg.src != "quality":
            return None

        obs = Observation(
            source="quality",
            frame_id=obs_msg.frame_id,
            t_ns=obs_msg.t_ns,
        )

        if obs_msg.quality:
            obs.signals["blur_score"] = obs_msg.quality.blur
            obs.signals["brightness"] = obs_msg.quality.brightness
            obs.signals["contrast"] = obs_msg.quality.contrast
            obs.signals["quality_gate"] = 1.0 if obs_msg.quality.gate_open else 0.0

        return obs


class FacemomentMapper(CompositeMapper):
    """Composite mapper for all facemoment observation types.

    This mapper handles Face, Pose, and Quality observations using
    the appropriate specialized mappers.

    Example:
        >>> from facemoment.process import create_extractor_process
        >>> process = create_extractor_process(FaceExtractor())
        >>> # FacemomentMapper is automatically used
    """

    def __init__(self):
        """Initialize with all facemoment-specific mappers."""
        super().__init__([
            FaceObservationMapper(),
            PoseObservationMapper(),
            QualityObservationMapper(),
        ])


__all__ = [
    "FaceObservationMapper",
    "PoseObservationMapper",
    "QualityObservationMapper",
    "FacemomentMapper",
]
