"""Trace record data classes for FaceMoment observability.

This module defines FaceMoment-specific trace record types.
Generic record types are re-exported from visualpath.observability.

Record Categories:
- Extraction records: Frame-level extraction results (face-specific)
- Gate records: Quality gate state changes
- Trigger records: Trigger decisions and firings
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# Import and re-export base types from visualpath
from visualpath.observability import TraceLevel
from visualpath.observability.records import (
    TraceRecord,
    TimingRecord,
    FrameDropRecord,
    SyncDelayRecord,
    FPSRecord,
    SessionStartRecord,
    SessionEndRecord,
)


# =============================================================================
# FaceMoment-specific Extraction Records
# =============================================================================


@dataclass
class FrameExtractRecord(TraceRecord):
    """Record of frame extraction results.

    Emitted by extractors after processing each frame.
    Contains summary of what was extracted.
    """
    record_type: str = field(default="frame_extract", init=False)

    frame_id: int = 0
    t_ns: int = 0
    source: str = ""  # "face", "pose", "gesture"

    # Summary (NORMAL level)
    face_count: int = 0
    pose_count: int = 0
    gesture_detected: bool = False

    # Detailed signals (VERBOSE level)
    signals: Dict[str, float] = field(default_factory=dict)

    # Processing time
    processing_ms: float = 0.0


@dataclass
class FaceExtractDetail(TraceRecord):
    """Detailed face extraction record for VERBOSE level.

    Contains per-face information including pose angles,
    expression values, and tracking data.
    """
    record_type: str = field(default="face_extract_detail", init=False)
    min_level: TraceLevel = field(default=TraceLevel.VERBOSE, repr=False)

    frame_id: int = 0
    face_id: int = 0

    # Detection
    confidence: float = 0.0
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)  # x, y, w, h normalized

    # Pose
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    # Expression
    expression: float = 0.0
    dominant_emotion: str = ""

    # Tracking
    inside_frame: bool = True
    area_ratio: float = 0.0
    center_distance: float = 0.0


# =============================================================================
# Gate Records
# =============================================================================


@dataclass
class GateChangeRecord(TraceRecord):
    """Record of quality gate state transition.

    Emitted when the gate opens or closes due to quality conditions.
    """
    record_type: str = field(default="gate_change", init=False)

    frame_id: int = 0
    t_ns: int = 0

    # State change
    old_state: str = ""  # "open", "closed"
    new_state: str = ""

    # Hysteresis timing
    duration_ns: int = 0  # How long condition was met before change

    # Conditions at transition (VERBOSE)
    conditions: Dict[str, bool] = field(default_factory=dict)


@dataclass
class GateConditionRecord(TraceRecord):
    """Detailed gate condition check for VERBOSE level.

    Shows which conditions passed/failed on each frame.
    """
    record_type: str = field(default="gate_condition", init=False)
    min_level: TraceLevel = field(default=TraceLevel.VERBOSE, repr=False)

    frame_id: int = 0
    gate_open: bool = False

    # Individual condition results
    face_count_ok: bool = False
    confidence_ok: bool = False
    yaw_ok: bool = False
    pitch_ok: bool = False
    inside_frame_ok: bool = False
    area_ok: bool = False
    center_ok: bool = False
    quality_ok: bool = False

    # Values for failed conditions
    face_count: int = 0
    max_confidence: float = 0.0
    max_yaw: float = 0.0
    max_pitch: float = 0.0


# =============================================================================
# Trigger Records
# =============================================================================


@dataclass
class TriggerCandidate:
    """A candidate trigger with reason and score."""
    reason: str = ""
    score: float = 0.0
    source: str = ""  # Which extractor/detection produced it


@dataclass
class TriggerDecisionRecord(TraceRecord):
    """Record of trigger decision process.

    Emitted on each frame when gate is open, showing what
    trigger candidates were detected and why the final decision
    was made.
    """
    record_type: str = field(default="trigger_decision", init=False)

    frame_id: int = 0
    t_ns: int = 0

    # Gate state
    gate_open: bool = False
    in_cooldown: bool = False

    # Candidates evaluated
    candidates: List[Dict[str, Any]] = field(default_factory=list)

    # Consecutive counting
    consecutive_count: int = 0
    consecutive_required: int = 2

    # Decision
    decision: str = ""  # "triggered", "blocked_gate", "blocked_cooldown", "no_trigger", "consecutive_pending"

    # EWMA state (VERBOSE)
    ewma_values: Dict[int, float] = field(default_factory=dict)  # face_id -> ewma
    ewma_vars: Dict[int, float] = field(default_factory=dict)  # face_id -> variance


@dataclass
class TriggerFireRecord(TraceRecord):
    """Record of trigger being fired.

    Emitted when a trigger is confirmed and sent to ingest.
    Always emitted at MINIMAL level since triggers are important.
    """
    record_type: str = field(default="trigger_fire", init=False)
    min_level: TraceLevel = field(default=TraceLevel.MINIMAL, repr=False)

    frame_id: int = 0
    t_ns: int = 0
    event_t_ns: int = 0  # Event start time

    reason: str = ""
    score: float = 0.0

    # Clip timing
    pre_sec: float = 0.0
    post_sec: float = 0.0

    # Context
    face_count: int = 0
    consecutive_frames: int = 0

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Pathway Pipeline Monitoring Records
# =============================================================================


@dataclass
class PathwayFrameRecord(TraceRecord):
    """Per-frame pipeline summary for Pathway backend.

    Emitted at NORMAL level for each frame processed through the
    Pathway pipeline, summarizing extractor timings and fusion result.
    """
    record_type: str = field(default="pathway_frame", init=False)

    frame_id: int = 0
    t_ns: int = 0
    extractor_timings_ms: Dict[str, float] = field(default_factory=dict)
    total_frame_ms: float = 0.0
    observations_produced: List[str] = field(default_factory=list)
    observations_failed: List[str] = field(default_factory=list)
    fusion_ms: float = 0.0
    fusion_decision: str = ""  # "triggered"/"no_trigger"/"cooldown"/"gate_closed"


@dataclass
class ExtractorTimingRecord(TraceRecord):
    """Individual extractor timing with sub-component breakdown.

    Emitted at NORMAL level for each extractor on each frame.
    """
    record_type: str = field(default="extractor_timing", init=False)

    frame_id: int = 0
    extractor_name: str = ""
    processing_ms: float = 0.0
    produced_observation: bool = False
    sub_timings_ms: Dict[str, float] = field(default_factory=dict)


@dataclass
class ObservationMergeRecord(TraceRecord):
    """Detailed observation merge information.

    Emitted at VERBOSE level when observations from multiple extractors
    are merged for fusion input.
    """
    record_type: str = field(default="observation_merge", init=False)
    min_level: TraceLevel = field(default=TraceLevel.VERBOSE, repr=False)

    frame_id: int = 0
    input_sources: List[str] = field(default_factory=list)
    input_signal_counts: Dict[str, int] = field(default_factory=dict)
    merged_signal_keys: List[str] = field(default_factory=list)
    main_face_id: Optional[int] = None
    main_face_source: str = ""  # "classifier_obs"/"merged_signals"/"none"


@dataclass
class BackpressureRecord(TraceRecord):
    """Rolling performance statistics emitted periodically.

    Emitted at NORMAL level every N frames (default 50) with
    aggregated timing statistics for bottleneck analysis.
    """
    record_type: str = field(default="backpressure", init=False)

    start_frame: int = 0
    end_frame: int = 0
    frame_count: int = 0
    wall_time_sec: float = 0.0
    effective_fps: float = 0.0
    target_fps: float = 0.0
    fps_ratio: float = 0.0
    extractor_avg_ms: Dict[str, float] = field(default_factory=dict)
    extractor_max_ms: Dict[str, float] = field(default_factory=dict)
    extractor_p95_ms: Dict[str, float] = field(default_factory=dict)
    slowest_extractor: str = ""
    bottleneck_pct: float = 0.0
    fusion_avg_ms: float = 0.0


@dataclass
class PipelineStatsRecord(TraceRecord):
    """Session-level pipeline summary emitted at session end.

    Emitted at MINIMAL level with overall statistics.
    """
    record_type: str = field(default="pipeline_stats", init=False)
    min_level: TraceLevel = field(default=TraceLevel.MINIMAL, repr=False)

    total_frames: int = 0
    total_triggers: int = 0
    wall_time_sec: float = 0.0
    effective_fps: float = 0.0
    extractor_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    fusion_avg_ms: float = 0.0
    gate_open_pct: float = 0.0


__all__ = [
    # Re-exported from visualpath
    "TraceRecord",
    "TimingRecord",
    "FrameDropRecord",
    "SyncDelayRecord",
    "FPSRecord",
    "SessionStartRecord",
    "SessionEndRecord",
    # FaceMoment-specific
    "FrameExtractRecord",
    "FaceExtractDetail",
    "GateChangeRecord",
    "GateConditionRecord",
    "TriggerCandidate",
    "TriggerDecisionRecord",
    "TriggerFireRecord",
    # Pathway monitoring
    "PathwayFrameRecord",
    "ExtractorTimingRecord",
    "ObservationMergeRecord",
    "BackpressureRecord",
    "PipelineStatsRecord",
]
