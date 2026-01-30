"""Tests for the observability system."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from facemoment.observability import (
    ObservabilityHub,
    TraceLevel,
    Sink,
    FileSink,
    ConsoleSink,
    MemorySink,
    NullSink,
)
from facemoment.observability.records import (
    TraceRecord,
    FrameExtractRecord,
    FaceExtractDetail,
    GateChangeRecord,
    GateConditionRecord,
    TriggerDecisionRecord,
    TriggerFireRecord,
    TimingRecord,
    FrameDropRecord,
    SyncDelayRecord,
    FPSRecord,
    SessionStartRecord,
    SessionEndRecord,
)


class TestTraceLevel:
    """Tests for TraceLevel enum."""

    def test_level_ordering(self):
        """Test that levels are correctly ordered."""
        assert TraceLevel.OFF < TraceLevel.MINIMAL
        assert TraceLevel.MINIMAL < TraceLevel.NORMAL
        assert TraceLevel.NORMAL < TraceLevel.VERBOSE

    def test_level_values(self):
        """Test numeric values of trace levels."""
        assert TraceLevel.OFF == 0
        assert TraceLevel.MINIMAL == 1
        assert TraceLevel.NORMAL == 2
        assert TraceLevel.VERBOSE == 3


class TestObservabilityHub:
    """Tests for ObservabilityHub singleton."""

    def setup_method(self):
        """Reset singleton before each test."""
        ObservabilityHub.reset_instance()

    def teardown_method(self):
        """Clean up after each test."""
        ObservabilityHub.reset_instance()

    def test_singleton_pattern(self):
        """Test that get_instance returns same instance."""
        hub1 = ObservabilityHub.get_instance()
        hub2 = ObservabilityHub.get_instance()
        assert hub1 is hub2

    def test_default_state_disabled(self):
        """Test that hub is disabled by default."""
        hub = ObservabilityHub.get_instance()
        assert not hub.enabled
        assert hub.level == TraceLevel.OFF

    def test_configure_enables_hub(self):
        """Test that configuring enables the hub."""
        hub = ObservabilityHub.get_instance()
        hub.configure(level=TraceLevel.NORMAL)

        assert hub.enabled
        assert hub.level == TraceLevel.NORMAL

    def test_configure_off_disables_hub(self):
        """Test that configuring to OFF disables the hub."""
        hub = ObservabilityHub.get_instance()
        hub.configure(level=TraceLevel.NORMAL)
        hub.configure(level=TraceLevel.OFF)

        assert not hub.enabled
        assert hub.level == TraceLevel.OFF

    def test_is_level_enabled(self):
        """Test level checking."""
        hub = ObservabilityHub.get_instance()
        hub.configure(level=TraceLevel.NORMAL)

        assert hub.is_level_enabled(TraceLevel.OFF)
        assert hub.is_level_enabled(TraceLevel.MINIMAL)
        assert hub.is_level_enabled(TraceLevel.NORMAL)
        assert not hub.is_level_enabled(TraceLevel.VERBOSE)

    def test_add_remove_sink(self):
        """Test adding and removing sinks."""
        hub = ObservabilityHub.get_instance()
        sink = MemorySink()

        hub.add_sink(sink)
        hub.configure(level=TraceLevel.MINIMAL)

        record = TriggerFireRecord(frame_id=1, reason="test", score=0.9)
        hub.emit(record)

        assert len(sink) == 1

        hub.remove_sink(sink)
        hub.emit(record)

        assert len(sink) == 1  # No new records

    def test_emit_respects_level(self):
        """Test that emit respects record minimum level."""
        hub = ObservabilityHub.get_instance()
        sink = MemorySink()
        hub.configure(level=TraceLevel.MINIMAL, sinks=[sink])

        # MINIMAL level record should be emitted
        trigger_record = TriggerFireRecord(frame_id=1, reason="test", score=0.9)
        hub.emit(trigger_record)

        # VERBOSE level record should not be emitted
        detail_record = FaceExtractDetail(frame_id=1, face_id=0)
        hub.emit(detail_record)

        records = sink.get_records()
        assert len(records) == 1
        assert records[0].record_type == "trigger_fire"

    def test_emit_when_disabled(self):
        """Test that emit does nothing when disabled."""
        hub = ObservabilityHub.get_instance()
        sink = MemorySink()
        hub.add_sink(sink)
        # Don't configure, so level is OFF

        record = TriggerFireRecord(frame_id=1, reason="test", score=0.9)
        hub.emit(record)

        assert len(sink) == 0

    def test_shutdown(self):
        """Test shutdown clears sinks and disables hub."""
        hub = ObservabilityHub.get_instance()
        sink = MemorySink()
        hub.configure(level=TraceLevel.NORMAL, sinks=[sink])

        hub.shutdown()

        assert not hub.enabled
        assert hub.level == TraceLevel.OFF


class TestTraceRecords:
    """Tests for trace record types."""

    def test_base_record_serialization(self):
        """Test TraceRecord to_dict and to_json."""
        record = FrameExtractRecord(
            frame_id=100,
            t_ns=1000000000,
            source="face",
            face_count=2,
            processing_ms=42.5,
        )

        d = record.to_dict()
        assert d["record_type"] == "frame_extract"
        assert d["frame_id"] == 100
        assert d["source"] == "face"
        assert d["face_count"] == 2
        assert d["processing_ms"] == 42.5
        assert "min_level" not in d  # Internal field should be removed

        json_str = record.to_json()
        parsed = json.loads(json_str)
        assert parsed["frame_id"] == 100

    def test_trigger_fire_record(self):
        """Test TriggerFireRecord fields."""
        record = TriggerFireRecord(
            frame_id=500,
            t_ns=5000000000,
            event_t_ns=4800000000,
            reason="expression_spike",
            score=0.85,
            pre_sec=2.0,
            post_sec=2.0,
            face_count=1,
            consecutive_frames=2,
        )

        assert record.record_type == "trigger_fire"
        assert record.min_level == TraceLevel.MINIMAL
        assert record.reason == "expression_spike"
        assert record.score == 0.85

    def test_gate_change_record(self):
        """Test GateChangeRecord fields."""
        record = GateChangeRecord(
            frame_id=100,
            t_ns=1000000000,
            old_state="closed",
            new_state="open",
            duration_ns=700000000,
        )

        assert record.record_type == "gate_change"
        assert record.old_state == "closed"
        assert record.new_state == "open"

    def test_timing_record(self):
        """Test TimingRecord fields."""
        record = TimingRecord(
            frame_id=50,
            component="face",
            processing_ms=55.0,
            threshold_ms=50.0,
            is_slow=True,
        )

        assert record.record_type == "timing"
        assert record.component == "face"
        assert record.is_slow is True

    def test_frame_drop_record(self):
        """Test FrameDropRecord fields."""
        record = FrameDropRecord(
            dropped_frame_ids=[101, 102, 103],
            reason="timeout",
        )

        assert record.record_type == "frame_drop"
        assert len(record.dropped_frame_ids) == 3

    def test_trigger_decision_record(self):
        """Test TriggerDecisionRecord with candidates."""
        record = TriggerDecisionRecord(
            frame_id=200,
            t_ns=2000000000,
            gate_open=True,
            in_cooldown=False,
            candidates=[
                {"reason": "expression_spike", "score": 0.7},
                {"reason": "head_turn", "score": 0.5},
            ],
            consecutive_count=1,
            consecutive_required=2,
            decision="consecutive_pending",
        )

        assert record.record_type == "trigger_decision"
        assert len(record.candidates) == 2
        assert record.decision == "consecutive_pending"


class TestFileSink:
    """Tests for FileSink."""

    def test_write_creates_file(self):
        """Test that writing creates the output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            sink = FileSink(str(path), buffer_size=1)

            record = TriggerFireRecord(frame_id=1, reason="test", score=0.9)
            sink.write(record)
            sink.flush()

            assert path.exists()
            content = path.read_text()
            assert "trigger_fire" in content

            sink.close()

    def test_buffered_writes(self):
        """Test that writes are buffered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            sink = FileSink(str(path), buffer_size=5)

            # Write 3 records (less than buffer)
            for i in range(3):
                record = TimingRecord(frame_id=i, component="test", processing_ms=10.0)
                sink.write(record)

            # File should exist but may be empty or partial
            sink.flush()

            content = path.read_text()
            lines = [l for l in content.strip().split("\n") if l]
            assert len(lines) == 3

            sink.close()

    def test_jsonl_format(self):
        """Test JSONL format correctness."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            sink = FileSink(str(path), buffer_size=1)

            records = [
                TriggerFireRecord(frame_id=1, reason="test1", score=0.9),
                TriggerFireRecord(frame_id=2, reason="test2", score=0.8),
            ]

            for r in records:
                sink.write(r)
            sink.close()

            content = path.read_text()
            lines = content.strip().split("\n")

            for i, line in enumerate(lines):
                parsed = json.loads(line)
                assert parsed["frame_id"] == i + 1


class TestMemorySink:
    """Tests for MemorySink."""

    def test_stores_records(self):
        """Test that records are stored in memory."""
        sink = MemorySink()

        for i in range(10):
            record = TimingRecord(frame_id=i, component="test", processing_ms=10.0)
            sink.write(record)

        assert len(sink) == 10

    def test_max_records_limit(self):
        """Test that max_records limit is enforced."""
        sink = MemorySink(max_records=5)

        for i in range(10):
            record = TimingRecord(frame_id=i, component="test", processing_ms=10.0)
            sink.write(record)

        assert len(sink) == 5
        records = sink.get_records()
        # Should have the last 5 records
        assert records[0].frame_id == 5

    def test_get_by_frame(self):
        """Test filtering records by frame ID."""
        sink = MemorySink()

        sink.write(TimingRecord(frame_id=1, component="face", processing_ms=40.0))
        sink.write(TimingRecord(frame_id=1, component="pose", processing_ms=20.0))
        sink.write(TimingRecord(frame_id=2, component="face", processing_ms=42.0))

        frame_1_records = sink.get_by_frame(1)
        assert len(frame_1_records) == 2

    def test_get_triggers(self):
        """Test getting trigger records."""
        sink = MemorySink()

        sink.write(TimingRecord(frame_id=1, component="face", processing_ms=40.0))
        sink.write(TriggerFireRecord(frame_id=2, reason="test", score=0.9))
        sink.write(TimingRecord(frame_id=3, component="face", processing_ms=41.0))

        triggers = sink.get_triggers()
        assert len(triggers) == 1
        assert triggers[0].frame_id == 2

    def test_get_timing_stats(self):
        """Test timing statistics calculation."""
        sink = MemorySink()

        # Add timing records for different components
        for i in range(5):
            sink.write(TimingRecord(
                frame_id=i,
                component="face",
                processing_ms=40.0 + i,
                is_slow=i > 3,
            ))
            sink.write(TimingRecord(
                frame_id=i,
                component="pose",
                processing_ms=20.0 + i,
                is_slow=False,
            ))

        stats = sink.get_timing_stats()

        assert "face" in stats
        assert "pose" in stats
        assert stats["face"]["count"] == 5
        assert stats["face"]["avg_ms"] == 42.0  # (40+41+42+43+44)/5
        assert stats["face"]["slow_count"] == 1
        assert stats["pose"]["avg_ms"] == 22.0

    def test_clear(self):
        """Test clearing records."""
        sink = MemorySink()

        for i in range(5):
            sink.write(TimingRecord(frame_id=i, component="test", processing_ms=10.0))

        assert len(sink) == 5
        sink.clear()
        assert len(sink) == 0


class TestConsoleSink:
    """Tests for ConsoleSink."""

    def test_formats_trigger_fire(self):
        """Test trigger fire formatting."""
        import io

        stream = io.StringIO()
        sink = ConsoleSink(stream=stream, color=False)

        record = TriggerFireRecord(
            frame_id=100,
            reason="expression_spike",
            score=0.85,
            face_count=1,
        )
        sink.write(record)

        output = stream.getvalue()
        assert "[TRIGGER]" in output
        assert "expression_spike" in output
        assert "0.85" in output

    def test_formats_gate_change(self):
        """Test gate change formatting."""
        import io

        stream = io.StringIO()
        sink = ConsoleSink(stream=stream, color=False)

        record = GateChangeRecord(
            frame_id=100,
            old_state="closed",
            new_state="open",
            duration_ns=700000000,
        )
        sink.write(record)

        output = stream.getvalue()
        assert "[GATE]" in output
        assert "closed" in output
        assert "open" in output or "OPEN" in output

    def test_timing_warning(self):
        """Test timing warning for slow frames."""
        import io

        stream = io.StringIO()
        sink = ConsoleSink(
            stream=stream,
            color=False,
            show_timing_warnings=True,
            timing_threshold_ms=50.0,
        )

        # Fast record - no output
        fast_record = TimingRecord(
            frame_id=1, component="face", processing_ms=30.0
        )
        sink.write(fast_record)
        assert "[TIMING]" not in stream.getvalue()

        # Slow record - should warn
        slow_record = TimingRecord(
            frame_id=2, component="face", processing_ms=60.0
        )
        sink.write(slow_record)
        assert "[TIMING]" in stream.getvalue()


class TestNullSink:
    """Tests for NullSink."""

    def test_discards_records(self):
        """Test that NullSink discards all records."""
        sink = NullSink()

        # Should not raise
        for i in range(100):
            sink.write(TimingRecord(frame_id=i, component="test", processing_ms=10.0))
        sink.flush()
        sink.close()


class TestIntegration:
    """Integration tests for observability system."""

    def setup_method(self):
        """Reset singleton before each test."""
        ObservabilityHub.reset_instance()

    def teardown_method(self):
        """Clean up after each test."""
        ObservabilityHub.reset_instance()

    def test_full_pipeline(self):
        """Test complete observability pipeline."""
        hub = ObservabilityHub.get_instance()
        memory_sink = MemorySink()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "trace.jsonl"
            file_sink = FileSink(str(file_path), buffer_size=1)

            hub.configure(
                level=TraceLevel.VERBOSE,
                sinks=[memory_sink, file_sink],
            )

            # Simulate a processing session
            hub.emit(SessionStartRecord(
                session_id="test-001",
                source_path="/test/video.mp4",
                target_fps=10.0,
            ))

            for frame_id in range(10):
                # Extract record
                hub.emit(FrameExtractRecord(
                    frame_id=frame_id,
                    source="face",
                    face_count=1,
                    processing_ms=40.0,
                ))

                # Timing record
                hub.emit(TimingRecord(
                    frame_id=frame_id,
                    component="face",
                    processing_ms=40.0,
                ))

                # Trigger on frame 5
                if frame_id == 5:
                    hub.emit(TriggerFireRecord(
                        frame_id=frame_id,
                        reason="expression_spike",
                        score=0.9,
                        face_count=1,
                    ))

            hub.emit(SessionEndRecord(
                session_id="test-001",
                duration_sec=1.0,
                total_frames=10,
                total_triggers=1,
            ))

            file_sink.close()

            # Verify memory sink
            records = memory_sink.get_records()
            assert len(records) > 0

            triggers = memory_sink.get_triggers()
            assert len(triggers) == 1
            assert triggers[0].frame_id == 5

            # Verify file sink
            content = file_path.read_text()
            lines = [l for l in content.strip().split("\n") if l]
            assert len(lines) > 0

            # All lines should be valid JSON
            for line in lines:
                parsed = json.loads(line)
                assert "record_type" in parsed

    def test_performance_minimal_overhead(self):
        """Test that tracing at OFF level has minimal overhead."""
        hub = ObservabilityHub.get_instance()
        # Leave at OFF level

        iterations = 10000
        start = time.perf_counter()

        for i in range(iterations):
            if hub.enabled:
                hub.emit(TimingRecord(frame_id=i, component="test", processing_ms=10.0))

        elapsed_off = time.perf_counter() - start

        # Now enable tracing
        hub.configure(level=TraceLevel.VERBOSE, sinks=[NullSink()])

        start = time.perf_counter()

        for i in range(iterations):
            if hub.enabled:
                hub.emit(TimingRecord(frame_id=i, component="test", processing_ms=10.0))

        elapsed_on = time.perf_counter() - start

        # OFF should be much faster than ON
        # Allow for some variance in timing
        assert elapsed_off < elapsed_on / 10  # OFF should be at least 10x faster

        hub.shutdown()
