"""Tests for PathwayMonitor and Pathway monitoring records."""

import time
from dataclasses import dataclass
from unittest.mock import Mock, patch, PropertyMock

import pytest

from facemoment.observability import ObservabilityHub, TraceLevel
from facemoment.observability.pathway_monitor import PathwayMonitor
from facemoment.observability.records import (
    BackpressureRecord,
    ExtractorTimingRecord,
    ObservationMergeRecord,
    PathwayFrameRecord,
    PipelineStatsRecord,
)
from facemoment.observability.sinks import ConsoleSink, MemorySink


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeFrame:
    frame_id: int = 0
    t_src_ns: int = 0


@dataclass
class FakeObservation:
    source: str = "face"
    signals: dict = None

    def __post_init__(self):
        if self.signals is None:
            self.signals = {}


@dataclass
class FakeFusionResult:
    should_trigger: bool = False
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FakeMainFace:
    face: Mock = None

    def __post_init__(self):
        if self.face is None:
            self.face = Mock(face_id=42)


@dataclass
class FakeClassifierData:
    main_face: FakeMainFace = None


def _make_hub(level: TraceLevel = TraceLevel.VERBOSE) -> tuple:
    """Create a fresh hub + memory sink for testing."""
    hub = ObservabilityHub.__new__(ObservabilityHub)
    hub._enabled = True
    hub._level = level
    hub._sinks = []
    hub._emit_lock = __import__("threading").Lock()
    sink = MemorySink()
    hub.add_sink(sink)
    return hub, sink


# ---------------------------------------------------------------------------
# TraceRecord Tests
# ---------------------------------------------------------------------------


class TestTraceRecords:
    """Tests for the 5 new TraceRecord types."""

    def test_pathway_frame_record_defaults(self):
        r = PathwayFrameRecord()
        assert r.record_type == "pathway_frame"
        assert r.frame_id == 0
        assert r.extractor_timings_ms == {}
        assert r.fusion_decision == ""

    def test_pathway_frame_record_with_values(self):
        r = PathwayFrameRecord(
            frame_id=10,
            t_ns=1000,
            extractor_timings_ms={"face": 32.4, "pose": 18.1},
            total_frame_ms=55.0,
            observations_produced=["face", "pose"],
            observations_failed=[],
            fusion_ms=2.1,
            fusion_decision="triggered",
        )
        assert r.frame_id == 10
        assert r.extractor_timings_ms["face"] == 32.4
        assert r.fusion_decision == "triggered"

    def test_pathway_frame_record_to_dict(self):
        r = PathwayFrameRecord(frame_id=5, total_frame_ms=42.0)
        d = r.to_dict()
        assert d["record_type"] == "pathway_frame"
        assert d["frame_id"] == 5
        assert "min_level" not in d  # min_level excluded from dict

    def test_extractor_timing_record(self):
        r = ExtractorTimingRecord(
            frame_id=1,
            extractor_name="face",
            processing_ms=32.5,
            produced_observation=True,
            sub_timings_ms={"detect": 20.0, "expression": 12.5},
        )
        assert r.record_type == "extractor_timing"
        assert r.extractor_name == "face"
        assert r.sub_timings_ms["detect"] == 20.0

    def test_observation_merge_record(self):
        r = ObservationMergeRecord(
            frame_id=1,
            input_sources=["face", "pose"],
            merged_signal_keys=["face_count", "hand_wave_detected"],
            main_face_id=42,
            main_face_source="classifier_obs",
        )
        assert r.record_type == "observation_merge"
        assert r.min_level == TraceLevel.VERBOSE
        assert r.main_face_id == 42

    def test_backpressure_record(self):
        r = BackpressureRecord(
            start_frame=0,
            end_frame=49,
            frame_count=50,
            wall_time_sec=5.0,
            effective_fps=10.0,
            target_fps=10.0,
            fps_ratio=1.0,
            extractor_avg_ms={"face": 30.0},
            slowest_extractor="face",
            bottleneck_pct=77.0,
        )
        assert r.record_type == "backpressure"
        assert r.fps_ratio == 1.0
        assert r.slowest_extractor == "face"

    def test_pipeline_stats_record(self):
        r = PipelineStatsRecord(
            total_frames=100,
            total_triggers=3,
            wall_time_sec=10.0,
            effective_fps=10.0,
            extractor_stats={"face": {"avg_ms": 30.0, "p95_ms": 45.0}},
            fusion_avg_ms=2.0,
            gate_open_pct=68.0,
        )
        assert r.record_type == "pipeline_stats"
        assert r.min_level == TraceLevel.MINIMAL
        assert r.total_triggers == 3
        assert r.gate_open_pct == 68.0


# ---------------------------------------------------------------------------
# PathwayMonitor Core Tests
# ---------------------------------------------------------------------------


class TestPathwayMonitorLifecycle:
    """Tests for PathwayMonitor frame lifecycle."""

    def test_basic_frame_lifecycle(self):
        hub, sink = _make_hub()
        monitor = PathwayMonitor(hub=hub, target_fps=10.0)

        frame = FakeFrame(frame_id=1, t_src_ns=100_000_000)
        monitor.begin_frame(frame)
        monitor.begin_extractor("face")
        obs = FakeObservation(source="face")
        monitor.end_extractor("face", obs)
        monitor.begin_fusion()
        result = FakeFusionResult(should_trigger=False, metadata={"state": "gate_closed"})
        monitor.end_fusion(result)
        monitor.end_frame(gate_open=False)

        stats = monitor.get_frame_stats()
        assert stats["frame_id"] == 1
        assert "face" in stats["extractor_timings_ms"]
        assert stats["fusion_decision"] == "gate_closed"

    def test_multiple_extractors(self):
        hub, sink = _make_hub()
        monitor = PathwayMonitor(hub=hub, target_fps=10.0)

        frame = FakeFrame(frame_id=1, t_src_ns=100_000_000)
        monitor.begin_frame(frame)

        for name in ["face", "pose", "quality"]:
            monitor.begin_extractor(name)
            obs = FakeObservation(source=name)
            monitor.end_extractor(name, obs)

        monitor.begin_fusion()
        monitor.end_fusion(FakeFusionResult())
        monitor.end_frame()

        stats = monitor.get_frame_stats()
        assert len(stats["extractor_timings_ms"]) == 3
        assert "face" in stats["extractor_timings_ms"]
        assert "pose" in stats["extractor_timings_ms"]
        assert "quality" in stats["extractor_timings_ms"]

    def test_failed_extractor(self):
        hub, sink = _make_hub()
        monitor = PathwayMonitor(hub=hub, target_fps=10.0)

        frame = FakeFrame(frame_id=1, t_src_ns=0)
        monitor.begin_frame(frame)
        monitor.begin_extractor("face")
        monitor.end_extractor("face", None)  # Failed
        monitor.end_frame()

        stats = monitor.get_frame_stats()
        assert "face" in stats["extractor_timings_ms"]

    def test_trigger_counting(self):
        hub, sink = _make_hub()
        monitor = PathwayMonitor(hub=hub, target_fps=10.0)

        # Frame 1: trigger
        frame = FakeFrame(frame_id=1, t_src_ns=0)
        monitor.begin_frame(frame)
        monitor.begin_fusion()
        monitor.end_fusion(FakeFusionResult(should_trigger=True))
        monitor.end_frame()

        # Frame 2: no trigger
        frame = FakeFrame(frame_id=2, t_src_ns=100_000_000)
        monitor.begin_frame(frame)
        monitor.begin_fusion()
        monitor.end_fusion(FakeFusionResult(should_trigger=False))
        monitor.end_frame()

        summary = monitor.get_summary()
        assert summary["total_frames"] == 2
        assert summary["total_triggers"] == 1

    def test_gate_open_tracking(self):
        hub, sink = _make_hub()
        monitor = PathwayMonitor(hub=hub, target_fps=10.0)

        for i in range(5):
            frame = FakeFrame(frame_id=i, t_src_ns=i * 100_000_000)
            monitor.begin_frame(frame)
            monitor.end_frame(gate_open=(i % 2 == 0))  # 0,2,4 = open

        summary = monitor.get_summary()
        assert summary["gate_open_pct"] == pytest.approx(60.0)


class TestPathwayMonitorStats:
    """Tests for stats accessors."""

    def _run_n_frames(self, monitor, n=10):
        """Helper to run N frames through monitor."""
        for i in range(n):
            frame = FakeFrame(frame_id=i, t_src_ns=i * 100_000_000)
            monitor.begin_frame(frame)
            monitor.begin_extractor("face")
            time.sleep(0.001)  # ~1ms
            monitor.end_extractor("face", FakeObservation())
            monitor.begin_extractor("quality")
            monitor.end_extractor("quality", FakeObservation(source="quality"))
            monitor.begin_fusion()
            monitor.end_fusion(FakeFusionResult())
            monitor.end_frame()

    def test_get_frame_stats_empty(self):
        hub, _ = _make_hub()
        monitor = PathwayMonitor(hub=hub)
        stats = monitor.get_frame_stats()
        assert stats == {}

    def test_get_frame_stats_has_expected_keys(self):
        hub, _ = _make_hub()
        monitor = PathwayMonitor(hub=hub, target_fps=10.0)
        self._run_n_frames(monitor, 3)

        stats = monitor.get_frame_stats()
        assert "frame_id" in stats
        assert "total_frame_ms" in stats
        assert "extractor_timings_ms" in stats
        assert "effective_fps" in stats
        assert "target_fps" in stats
        assert "fps_ratio" in stats
        assert "slowest_extractor" in stats
        assert "bottleneck_pct" in stats

    def test_get_rolling_stats(self):
        hub, _ = _make_hub()
        monitor = PathwayMonitor(hub=hub, target_fps=10.0)
        self._run_n_frames(monitor, 10)

        rolling = monitor.get_rolling_stats()
        assert rolling["effective_fps"] > 0
        assert "face" in rolling["extractor_avg_ms"]
        assert "face" in rolling["extractor_p95_ms"]
        assert "face" in rolling["extractor_max_ms"]
        assert rolling["slowest_extractor"] in ("face", "quality")

    def test_get_rolling_stats_insufficient_frames(self):
        hub, _ = _make_hub()
        monitor = PathwayMonitor(hub=hub)

        frame = FakeFrame(frame_id=0, t_src_ns=0)
        monitor.begin_frame(frame)
        monitor.end_frame()

        rolling = monitor.get_rolling_stats()
        assert rolling["effective_fps"] == 0.0

    def test_get_summary(self):
        hub, _ = _make_hub()
        monitor = PathwayMonitor(hub=hub, target_fps=10.0)
        self._run_n_frames(monitor, 5)

        summary = monitor.get_summary()
        assert summary["total_frames"] == 5
        assert summary["wall_time_sec"] > 0
        assert summary["effective_fps"] > 0
        assert "face" in summary["extractor_stats"]
        assert "avg_ms" in summary["extractor_stats"]["face"]
        assert "p95_ms" in summary["extractor_stats"]["face"]
        assert "max_ms" in summary["extractor_stats"]["face"]

    def test_main_face_tracking(self):
        hub, _ = _make_hub()
        monitor = PathwayMonitor(hub=hub)

        frame = FakeFrame(frame_id=1, t_src_ns=0)
        monitor.begin_frame(frame)

        # Record classifier with main face
        classifier_obs = Mock()
        classifier_obs.data = FakeClassifierData(main_face=FakeMainFace())
        monitor.record_classifier(classifier_obs)

        monitor.end_frame()

        stats = monitor.get_frame_stats()
        assert stats["main_face"] == "face#42"


# ---------------------------------------------------------------------------
# TraceRecord Emission Tests
# ---------------------------------------------------------------------------


class TestPathwayMonitorEmission:
    """Tests for TraceRecord emission gating."""

    def test_emits_pathway_frame_record(self):
        hub, sink = _make_hub(TraceLevel.NORMAL)
        monitor = PathwayMonitor(hub=hub)

        frame = FakeFrame(frame_id=1, t_src_ns=0)
        monitor.begin_frame(frame)
        monitor.end_frame()

        records = sink.get_records()
        frame_records = [r for r in records if isinstance(r, PathwayFrameRecord)]
        assert len(frame_records) == 1
        assert frame_records[0].frame_id == 1

    def test_emits_extractor_timing_record(self):
        hub, sink = _make_hub(TraceLevel.NORMAL)
        monitor = PathwayMonitor(hub=hub)

        frame = FakeFrame(frame_id=1, t_src_ns=0)
        monitor.begin_frame(frame)
        monitor.begin_extractor("face")
        monitor.end_extractor("face", FakeObservation())
        monitor.end_frame()

        records = sink.get_records()
        timing_records = [r for r in records if isinstance(r, ExtractorTimingRecord)]
        assert len(timing_records) == 1
        assert timing_records[0].extractor_name == "face"
        assert timing_records[0].produced_observation is True

    def test_emits_merge_record_at_verbose(self):
        hub, sink = _make_hub(TraceLevel.VERBOSE)
        monitor = PathwayMonitor(hub=hub)

        frame = FakeFrame(frame_id=1, t_src_ns=0)
        monitor.begin_frame(frame)

        obs_list = [FakeObservation(source="face"), FakeObservation(source="pose")]
        merged = FakeObservation(source="merged", signals={"face_count": 1, "hand_wave": 0.5})
        monitor.record_merge(obs_list, merged, main_face_id=42, main_face_source="classifier_obs")

        monitor.end_frame()

        records = sink.get_records()
        merge_records = [r for r in records if isinstance(r, ObservationMergeRecord)]
        assert len(merge_records) == 1
        assert merge_records[0].main_face_id == 42
        assert merge_records[0].input_sources == ["face", "pose"]

    def test_merge_record_not_emitted_at_normal(self):
        hub, sink = _make_hub(TraceLevel.NORMAL)
        monitor = PathwayMonitor(hub=hub)

        frame = FakeFrame(frame_id=1, t_src_ns=0)
        monitor.begin_frame(frame)

        obs_list = [FakeObservation(source="face")]
        merged = FakeObservation(source="merged")
        monitor.record_merge(obs_list, merged)
        monitor.end_frame()

        records = sink.get_records()
        merge_records = [r for r in records if isinstance(r, ObservationMergeRecord)]
        assert len(merge_records) == 0

    def test_emits_pipeline_stats_record(self):
        hub, sink = _make_hub(TraceLevel.MINIMAL)
        monitor = PathwayMonitor(hub=hub, target_fps=10.0)

        frame = FakeFrame(frame_id=1, t_src_ns=0)
        monitor.begin_frame(frame)
        monitor.end_frame()

        monitor.get_summary()

        records = sink.get_records()
        stats_records = [r for r in records if isinstance(r, PipelineStatsRecord)]
        assert len(stats_records) == 1
        assert stats_records[0].total_frames == 1

    def test_no_emission_when_hub_disabled(self):
        hub, sink = _make_hub(TraceLevel.OFF)
        hub._enabled = False
        monitor = PathwayMonitor(hub=hub)

        frame = FakeFrame(frame_id=1, t_src_ns=0)
        monitor.begin_frame(frame)
        monitor.begin_extractor("face")
        monitor.end_extractor("face", FakeObservation())
        monitor.end_frame()

        records = sink.get_records()
        assert len(records) == 0

    def test_backpressure_emitted_at_interval(self):
        hub, sink = _make_hub(TraceLevel.NORMAL)
        monitor = PathwayMonitor(hub=hub, target_fps=10.0, report_interval=5)

        for i in range(10):
            frame = FakeFrame(frame_id=i, t_src_ns=i * 100_000_000)
            monitor.begin_frame(frame)
            monitor.begin_extractor("face")
            monitor.end_extractor("face", FakeObservation())
            monitor.end_frame()

        records = sink.get_records()
        bp_records = [r for r in records if isinstance(r, BackpressureRecord)]
        assert len(bp_records) == 2  # At frame 5 and 10


# ---------------------------------------------------------------------------
# Stats Always Work (trace-level independent)
# ---------------------------------------------------------------------------


class TestPathwayMonitorStatsAlwaysWork:
    """Verify stats work even when tracing is OFF."""

    def test_frame_stats_with_tracing_off(self):
        hub, _ = _make_hub(TraceLevel.OFF)
        hub._enabled = False
        monitor = PathwayMonitor(hub=hub, target_fps=10.0)

        frame = FakeFrame(frame_id=1, t_src_ns=0)
        monitor.begin_frame(frame)
        monitor.begin_extractor("face")
        monitor.end_extractor("face", FakeObservation())
        monitor.begin_fusion()
        monitor.end_fusion(FakeFusionResult())
        monitor.end_frame()

        stats = monitor.get_frame_stats()
        assert stats["frame_id"] == 1
        assert "face" in stats["extractor_timings_ms"]

    def test_rolling_stats_with_tracing_off(self):
        hub, _ = _make_hub(TraceLevel.OFF)
        hub._enabled = False
        monitor = PathwayMonitor(hub=hub, target_fps=10.0)

        for i in range(5):
            frame = FakeFrame(frame_id=i, t_src_ns=i * 100_000_000)
            monitor.begin_frame(frame)
            monitor.begin_extractor("face")
            time.sleep(0.001)
            monitor.end_extractor("face", FakeObservation())
            monitor.end_frame()

        rolling = monitor.get_rolling_stats()
        assert rolling["effective_fps"] > 0
        assert "face" in rolling["extractor_avg_ms"]

    def test_summary_with_tracing_off(self):
        hub, _ = _make_hub(TraceLevel.OFF)
        hub._enabled = False
        monitor = PathwayMonitor(hub=hub)

        for i in range(3):
            frame = FakeFrame(frame_id=i, t_src_ns=i * 100_000_000)
            monitor.begin_frame(frame)
            monitor.end_frame()

        summary = monitor.get_summary()
        assert summary["total_frames"] == 3


# ---------------------------------------------------------------------------
# ConsoleSink Formatter Tests
# ---------------------------------------------------------------------------


class TestConsoleSinkFormatters:
    """Tests for new ConsoleSink formatters."""

    def _make_sink(self):
        return ConsoleSink(color=False)

    def test_format_backpressure(self):
        sink = self._make_sink()
        record = BackpressureRecord(
            start_frame=0,
            end_frame=49,
            frame_count=50,
            effective_fps=9.2,
            target_fps=10.0,
            fps_ratio=0.92,
            slowest_extractor="face",
            bottleneck_pct=77.0,
        )
        result = sink._format_record(record)
        assert "[PERF]" in result
        assert "9.2fps" in result
        assert "face" in result
        assert "77%" in result

    def test_format_pathway_frame_slow(self):
        sink = self._make_sink()
        record = PathwayFrameRecord(
            frame_id=42,
            total_frame_ms=105.0,
            fusion_decision="no_trigger",
        )
        result = sink._format_record(record)
        assert "[FRAME]" in result
        assert "105ms" in result
        assert "no_trigger" in result

    def test_format_pathway_frame_fast_skipped(self):
        sink = self._make_sink()
        record = PathwayFrameRecord(
            frame_id=42,
            total_frame_ms=50.0,
        )
        result = sink._format_record(record)
        assert result is None  # Fast frames skipped

    def test_format_pipeline_stats(self):
        sink = self._make_sink()
        record = PipelineStatsRecord(
            total_frames=100,
            total_triggers=2,
            wall_time_sec=10.0,
            effective_fps=10.0,
            extractor_stats={"face": {"avg_ms": 30.0, "p95_ms": 45.0, "max_ms": 80.0, "errors": 0}},
            fusion_avg_ms=2.0,
            gate_open_pct=68.0,
        )
        result = sink._format_record(record)
        assert "Pathway Pipeline Summary" in result
        assert "100 frames" in result
        assert "face" in result
        assert "68%" in result


# ---------------------------------------------------------------------------
# Fusion Decision Mapping Tests
# ---------------------------------------------------------------------------


class TestFusionDecisionMapping:
    """Tests for fusion result -> decision string mapping."""

    def test_triggered_decision(self):
        hub, _ = _make_hub()
        monitor = PathwayMonitor(hub=hub)
        frame = FakeFrame(frame_id=1, t_src_ns=0)
        monitor.begin_frame(frame)
        monitor.begin_fusion()
        monitor.end_fusion(FakeFusionResult(should_trigger=True))
        monitor.end_frame()

        stats = monitor.get_frame_stats()
        assert stats["fusion_decision"] == "triggered"

    def test_cooldown_decision(self):
        hub, _ = _make_hub()
        monitor = PathwayMonitor(hub=hub)
        frame = FakeFrame(frame_id=1, t_src_ns=0)
        monitor.begin_frame(frame)
        monitor.begin_fusion()
        monitor.end_fusion(FakeFusionResult(
            should_trigger=False,
            metadata={"state": "cooldown"},
        ))
        monitor.end_frame()

        stats = monitor.get_frame_stats()
        assert stats["fusion_decision"] == "cooldown"

    def test_gate_closed_decision(self):
        hub, _ = _make_hub()
        monitor = PathwayMonitor(hub=hub)
        frame = FakeFrame(frame_id=1, t_src_ns=0)
        monitor.begin_frame(frame)
        monitor.begin_fusion()
        monitor.end_fusion(FakeFusionResult(
            should_trigger=False,
            metadata={"state": "gate_closed"},
        ))
        monitor.end_frame()

        stats = monitor.get_frame_stats()
        assert stats["fusion_decision"] == "gate_closed"

    def test_no_trigger_decision(self):
        hub, _ = _make_hub()
        monitor = PathwayMonitor(hub=hub)
        frame = FakeFrame(frame_id=1, t_src_ns=0)
        monitor.begin_frame(frame)
        monitor.begin_fusion()
        monitor.end_fusion(FakeFusionResult(
            should_trigger=False,
            metadata={"state": "monitoring"},
        ))
        monitor.end_frame()

        stats = monitor.get_frame_stats()
        assert stats["fusion_decision"] == "no_trigger"

    def test_no_result_decision(self):
        hub, _ = _make_hub()
        monitor = PathwayMonitor(hub=hub)
        frame = FakeFrame(frame_id=1, t_src_ns=0)
        monitor.begin_frame(frame)
        monitor.begin_fusion()
        monitor.end_fusion(None)
        monitor.end_frame()

        stats = monitor.get_frame_stats()
        assert stats["fusion_decision"] == "no_result"


# ---------------------------------------------------------------------------
# Sub-timings Tests
# ---------------------------------------------------------------------------


class TestSubTimings:
    """Tests for extractor sub-timings propagation."""

    def test_sub_timings_passed_to_record(self):
        hub, sink = _make_hub(TraceLevel.NORMAL)
        monitor = PathwayMonitor(hub=hub)

        frame = FakeFrame(frame_id=1, t_src_ns=0)
        monitor.begin_frame(frame)
        monitor.begin_extractor("face")
        monitor.end_extractor("face", FakeObservation(), sub_timings={"detect": 20.0, "expression": 12.0})
        monitor.end_frame()

        records = sink.get_records()
        timing_records = [r for r in records if isinstance(r, ExtractorTimingRecord)]
        assert len(timing_records) == 1
        assert timing_records[0].sub_timings_ms == {"detect": 20.0, "expression": 12.0}

    def test_sub_timings_none(self):
        hub, sink = _make_hub(TraceLevel.NORMAL)
        monitor = PathwayMonitor(hub=hub)

        frame = FakeFrame(frame_id=1, t_src_ns=0)
        monitor.begin_frame(frame)
        monitor.begin_extractor("quality")
        monitor.end_extractor("quality", FakeObservation(source="quality"))
        monitor.end_frame()

        records = sink.get_records()
        timing_records = [r for r in records if isinstance(r, ExtractorTimingRecord)]
        assert len(timing_records) == 1
        assert timing_records[0].sub_timings_ms == {}
