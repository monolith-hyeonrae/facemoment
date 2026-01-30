"""Trace record data classes for observability.

This module defines all the trace record types used throughout
the FaceMoment observability system.

Record Categories:
- Extraction records: Frame-level extraction results
- Gate records: Quality gate state changes
- Trigger records: Trigger decisions and firings
- Timing records: Component performance metrics
- Sync records: Stream synchronization events
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import time
import json


# Forward reference for TraceLevel
from facemoment.observability import TraceLevel


@dataclass
class TraceRecord:
    """Base class for all trace records.

    All trace records have:
    - record_type: String identifying the record type
    - timestamp_ns: When the record was created (monotonic)
    - min_level: Minimum trace level required to emit this record

    Subclasses should set record_type as a class variable.
    """
    record_type: str = field(default="base", init=False)
    timestamp_ns: int = field(default_factory=lambda: time.perf_counter_ns())
    min_level: TraceLevel = field(default=TraceLevel.NORMAL, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        d = asdict(self)
        # Remove min_level from output (internal use only)
        d.pop("min_level", None)
        return d

    def to_json(self) -> str:
        """Convert record to JSON string.

        Returns:
            JSON-serialized record.
        """
        return json.dumps(self.to_dict(), default=str)


# =============================================================================
# Extraction Records
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
# Timing Records
# =============================================================================

@dataclass
class TimingRecord(TraceRecord):
    """Component processing time record.

    Emitted after each frame processing to track performance.
    """
    record_type: str = field(default="timing", init=False)
    min_level: TraceLevel = field(default=TraceLevel.NORMAL, repr=False)

    frame_id: int = 0
    component: str = ""  # "face", "pose", "gesture", "fusion", "orchestrator"

    processing_ms: float = 0.0
    queue_depth: int = 0

    # Thresholds for warnings
    threshold_ms: float = 50.0  # Default warning threshold
    is_slow: bool = False


@dataclass
class FrameDropRecord(TraceRecord):
    """Record of dropped frames.

    Emitted when frames are dropped due to processing delays.
    """
    record_type: str = field(default="frame_drop", init=False)

    dropped_frame_ids: List[int] = field(default_factory=list)
    reason: str = ""  # "timeout", "backpressure", "queue_full"

    # Context
    queue_depth: int = 0
    processing_ms: float = 0.0


@dataclass
class SyncDelayRecord(TraceRecord):
    """Record of observation synchronization delay.

    Emitted when fusion waits for observations from multiple extractors.
    """
    record_type: str = field(default="sync_delay", init=False)

    frame_id: int = 0
    expected_ns: int = 0
    actual_ns: int = 0
    delay_ms: float = 0.0

    waiting_for: List[str] = field(default_factory=list)  # ["gesture", "pose"]


@dataclass
class FPSRecord(TraceRecord):
    """Periodic FPS and performance summary.

    Emitted periodically (e.g., every 100 frames) with aggregate stats.
    """
    record_type: str = field(default="fps_summary", init=False)
    min_level: TraceLevel = field(default=TraceLevel.NORMAL, repr=False)

    # Frame range
    start_frame: int = 0
    end_frame: int = 0
    frame_count: int = 0

    # FPS
    actual_fps: float = 0.0
    target_fps: float = 0.0
    fps_ratio: float = 0.0  # actual / target

    # Latency stats (ms)
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    # Per-component average times
    component_avg_ms: Dict[str, float] = field(default_factory=dict)

    # Issues
    dropped_frames: int = 0
    slow_frames: int = 0  # Frames exceeding threshold


# =============================================================================
# Session Records
# =============================================================================

@dataclass
class SessionStartRecord(TraceRecord):
    """Record emitted when a processing session starts."""
    record_type: str = field(default="session_start", init=False)
    min_level: TraceLevel = field(default=TraceLevel.MINIMAL, repr=False)

    session_id: str = ""
    source_path: str = ""
    target_fps: float = 0.0

    # Configuration
    extractors: List[str] = field(default_factory=list)
    fusion_params: Dict[str, Any] = field(default_factory=dict)
    trace_level: str = ""


@dataclass
class SessionEndRecord(TraceRecord):
    """Record emitted when a processing session ends."""
    record_type: str = field(default="session_end", init=False)
    min_level: TraceLevel = field(default=TraceLevel.MINIMAL, repr=False)

    session_id: str = ""
    duration_sec: float = 0.0

    # Summary stats
    total_frames: int = 0
    total_triggers: int = 0
    total_dropped: int = 0
    avg_fps: float = 0.0


__all__ = [
    "TraceRecord",
    "FrameExtractRecord",
    "FaceExtractDetail",
    "GateChangeRecord",
    "GateConditionRecord",
    "TriggerCandidate",
    "TriggerDecisionRecord",
    "TriggerFireRecord",
    "TimingRecord",
    "FrameDropRecord",
    "SyncDelayRecord",
    "FPSRecord",
    "SessionStartRecord",
    "SessionEndRecord",
]
