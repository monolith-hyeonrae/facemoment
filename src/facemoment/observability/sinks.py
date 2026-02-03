"""Trace output sinks for observability.

This module re-exports sinks from visualpath and adds FaceMoment-specific
formatting for console output.

Sinks receive trace records and handle their output to various destinations:
- FileSink: JSONL file output
- ConsoleSink: Formatted console output
- MemorySink: In-memory buffer for testing/analysis
"""

import sys
from typing import List, Optional, TextIO

# Re-export sinks from visualpath
from visualpath.observability.sinks import (
    FileSink,
    NullSink,
    ConsoleSink as BaseConsoleSink,
    MemorySink as BaseMemorySink,
)
from visualpath.observability import Sink, TraceLevel
from visualpath.observability.records import TraceRecord, TimingRecord, FrameDropRecord, SyncDelayRecord

# Import facemoment-specific records
from facemoment.observability.records import (
    TriggerFireRecord,
    GateChangeRecord,
    BackpressureRecord,
    PathwayFrameRecord,
    PipelineStatsRecord,
)


class MemorySink(BaseMemorySink):
    """Memory sink with FaceMoment-specific helper methods.

    Extends the base MemorySink to add convenience methods for
    getting FaceMoment-specific record types like triggers.
    """

    def get_triggers(self) -> List[TriggerFireRecord]:
        """Get all trigger fire records.

        Returns:
            List of trigger fire records.
        """
        return [
            r for r in self.get_records()
            if isinstance(r, TriggerFireRecord)
        ]


class ConsoleSink(BaseConsoleSink):
    """Console sink with FaceMoment-specific formatting.

    Extends the base ConsoleSink to format FaceMoment-specific
    records like TriggerFireRecord and GateChangeRecord.
    """

    def _format_record(self, record: TraceRecord) -> Optional[str]:
        """Format a record for console output.

        Args:
            record: The trace record to format.

        Returns:
            Formatted string or None to skip output.
        """
        if isinstance(record, TriggerFireRecord):
            return self._format_trigger_fire(record)
        elif isinstance(record, GateChangeRecord):
            return self._format_gate_change(record)
        elif isinstance(record, BackpressureRecord):
            return self._format_backpressure(record)
        elif isinstance(record, PathwayFrameRecord):
            return self._format_pathway_frame(record)
        elif isinstance(record, PipelineStatsRecord):
            return self._format_pipeline_stats(record)
        elif isinstance(record, TimingRecord):
            return self._format_timing(record)
        elif isinstance(record, FrameDropRecord):
            return self._format_frame_drop(record)
        elif isinstance(record, SyncDelayRecord):
            return self._format_sync_delay(record)
        else:
            # Skip other record types for console
            return None

    def _format_trigger_fire(self, record: TriggerFireRecord) -> str:
        """Format trigger fire record."""
        tag = self._colorize("[TRIGGER]", "green")
        reason = self._colorize(record.reason, "cyan")
        return (
            f"{tag} Frame {record.frame_id}: {reason} "
            f"score={record.score:.2f} faces={record.face_count}"
        )

    def _format_gate_change(self, record: GateChangeRecord) -> str:
        """Format gate change record."""
        if record.new_state == "open":
            tag = self._colorize("[GATE]", "blue")
            state = self._colorize("OPEN", "green")
        else:
            tag = self._colorize("[GATE]", "blue")
            state = self._colorize("CLOSED", "yellow")

        duration_ms = record.duration_ns / 1_000_000
        return (
            f"{tag} Frame {record.frame_id}: {record.old_state} -> {state} "
            f"(after {duration_ms:.0f}ms)"
        )

    def _format_backpressure(self, record: BackpressureRecord) -> str:
        """Format backpressure/rolling performance record."""
        tag = self._colorize("[PERF]", "magenta")

        # Color FPS ratio
        if record.fps_ratio >= 0.9:
            fps_str = self._colorize(f"{record.effective_fps:.1f}fps ({record.fps_ratio:.0%})", "green")
        elif record.fps_ratio >= 0.7:
            fps_str = self._colorize(f"{record.effective_fps:.1f}fps ({record.fps_ratio:.0%})", "yellow")
        else:
            fps_str = self._colorize(f"{record.effective_fps:.1f}fps ({record.fps_ratio:.0%})", "red")

        bottleneck = ""
        if record.slowest_extractor:
            bottleneck = f" bottleneck={record.slowest_extractor} ({record.bottleneck_pct:.0f}%)"

        return (
            f"{tag} Frames {record.start_frame}-{record.end_frame}: "
            f"{fps_str}{bottleneck}"
        )

    def _format_pathway_frame(self, record: PathwayFrameRecord) -> Optional[str]:
        """Format pathway frame record (only slow frames >100ms)."""
        if record.total_frame_ms <= 100:
            return None

        tag = self._colorize("[FRAME]", "yellow")
        return (
            f"{tag} Frame {record.frame_id}: "
            f"{record.total_frame_ms:.0f}ms ({record.fusion_decision})"
        )

    def _format_pipeline_stats(self, record: PipelineStatsRecord) -> str:
        """Format pipeline session summary."""
        lines = []
        sep = "=" * 50
        lines.append(sep)
        lines.append("Pathway Pipeline Summary")
        lines.append(sep)
        lines.append(f"  Duration:      {record.wall_time_sec:.1f}s ({record.total_frames} frames)")
        lines.append(f"  Effective FPS: {record.effective_fps:.1f}")
        lines.append(f"  Triggers:      {record.total_triggers}")

        if record.extractor_stats:
            lines.append("")
            lines.append("  Extractor Performance:")
            header = f"  {'Extractor':<14} {'Avg ms':>8} {'P95 ms':>8} {'Max ms':>8} {'Errors':>8}"
            lines.append(header)
            lines.append("  " + "-" * 48)
            for name, stats in record.extractor_stats.items():
                avg = stats.get("avg_ms", 0)
                p95 = stats.get("p95_ms", 0)
                mx = stats.get("max_ms", 0)
                errs = int(stats.get("errors", 0))
                lines.append(f"  {name:<14} {avg:>8.1f} {p95:>8.1f} {mx:>8.1f} {errs:>8}")

        if record.fusion_avg_ms > 0:
            lines.append(f"  {'fusion':<14} {record.fusion_avg_ms:>8.1f}{'':>8}{'':>8}{'  -':>8}")

        lines.append("")
        lines.append(f"  Gate: open {record.gate_open_pct:.0f}% of frames")
        lines.append(sep)

        return "\n".join(lines)


__all__ = [
    "FileSink",
    "ConsoleSink",
    "MemorySink",
    "NullSink",
]
