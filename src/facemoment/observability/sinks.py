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


__all__ = [
    "FileSink",
    "ConsoleSink",
    "MemorySink",
    "NullSink",
]
