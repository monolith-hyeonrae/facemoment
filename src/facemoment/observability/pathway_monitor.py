"""Pathway pipeline real-time monitoring.

Collects per-frame timing data, rolling statistics, and emits
TraceRecords for the Pathway backend debug loop.

Key design:
- get_frame_stats() and get_rolling_stats() always work (trace level independent)
- TraceRecord emission is gated by ObservabilityHub level
- BackpressureRecord emitted every `report_interval` frames
- PipelineStatsRecord emitted at session end via get_summary()
"""

import collections
import time
from typing import Any, Dict, List, Optional

from facemoment.observability import ObservabilityHub, TraceLevel
from facemoment.observability.records import (
    BackpressureRecord,
    ExtractorTimingRecord,
    ObservationMergeRecord,
    PathwayFrameRecord,
    PipelineStatsRecord,
)


class PathwayMonitor:
    """Real-time monitoring for Pathway pipeline.

    Responsibilities:
    1. Collect per-extractor timing (time.perf_counter_ns)
    2. Maintain rolling statistics (deque, last N frames)
    3. Provide overlay stats (get_frame_stats, get_rolling_stats)
    4. Emit TraceRecords when hub is enabled
    5. Emit BackpressureRecord every report_interval frames

    Args:
        hub: ObservabilityHub instance.
        target_fps: Expected input FPS for ratio calculation.
        report_interval: Frames between BackpressureRecord emissions.
    """

    def __init__(
        self,
        hub: Optional[ObservabilityHub] = None,
        target_fps: float = 10.0,
        report_interval: int = 50,
    ):
        self._hub = hub or ObservabilityHub.get_instance()
        self._target_fps = target_fps
        self._report_interval = report_interval

        # --- Current frame state ---
        self._frame_id: int = 0
        self._frame_t_ns: int = 0
        self._frame_start_ns: int = 0
        self._extractor_start_ns: int = 0
        self._extractor_timings: Dict[str, float] = {}  # name -> ms
        self._observations_produced: List[str] = []
        self._observations_failed: List[str] = []
        self._fusion_start_ns: int = 0
        self._fusion_ms: float = 0.0
        self._fusion_decision: str = ""
        self._classifier_main_face_id: Optional[int] = None

        # --- Rolling history (deque, maxlen=report_interval) ---
        self._history: collections.deque = collections.deque(maxlen=report_interval)

        # --- Session-level accumulators ---
        self._total_frames: int = 0
        self._total_triggers: int = 0
        self._session_start_ns: int = 0
        self._all_extractor_timings: Dict[str, List[float]] = collections.defaultdict(list)
        self._all_fusion_timings: List[float] = []
        self._gate_open_count: int = 0
        self._extractor_error_counts: Dict[str, int] = collections.defaultdict(int)

        # --- Backpressure tracking ---
        self._report_start_ns: int = 0
        self._report_start_frame: int = 0

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def begin_frame(self, frame: Any) -> None:
        """Start timing a new frame.

        Args:
            frame: Frame object (must have frame_id, t_src_ns).
        """
        now = time.perf_counter_ns()
        self._frame_start_ns = now
        self._frame_id = getattr(frame, "frame_id", 0)
        self._frame_t_ns = getattr(frame, "t_src_ns", 0)

        # Reset per-frame accumulators
        self._extractor_timings.clear()
        self._observations_produced.clear()
        self._observations_failed.clear()
        self._fusion_ms = 0.0
        self._fusion_decision = ""
        self._classifier_main_face_id = None

        if self._session_start_ns == 0:
            self._session_start_ns = now
            self._report_start_ns = now
            self._report_start_frame = self._frame_id

    def end_frame(self, gate_open: bool = False) -> None:
        """Finish timing a frame and record stats.

        Args:
            gate_open: Whether the quality gate is open this frame.
        """
        now = time.perf_counter_ns()
        total_ms = (now - self._frame_start_ns) / 1_000_000

        if gate_open:
            self._gate_open_count += 1

        # Build frame record dict (used for history and overlay)
        frame_stats = {
            "frame_id": self._frame_id,
            "t_ns": self._frame_t_ns,
            "extractor_timings_ms": dict(self._extractor_timings),
            "total_frame_ms": total_ms,
            "observations_produced": list(self._observations_produced),
            "observations_failed": list(self._observations_failed),
            "fusion_ms": self._fusion_ms,
            "fusion_decision": self._fusion_decision,
        }
        self._history.append(frame_stats)
        self._total_frames += 1

        if self._fusion_decision == "triggered":
            self._total_triggers += 1

        # Accumulate for session stats
        for name, ms in self._extractor_timings.items():
            self._all_extractor_timings[name].append(ms)
        if self._fusion_ms > 0:
            self._all_fusion_timings.append(self._fusion_ms)

        # Emit PathwayFrameRecord
        if self._hub.enabled:
            self._hub.emit(PathwayFrameRecord(
                frame_id=self._frame_id,
                t_ns=self._frame_t_ns,
                extractor_timings_ms=dict(self._extractor_timings),
                total_frame_ms=total_ms,
                observations_produced=list(self._observations_produced),
                observations_failed=list(self._observations_failed),
                fusion_ms=self._fusion_ms,
                fusion_decision=self._fusion_decision,
            ))

        # Emit BackpressureRecord periodically
        if self._total_frames % self._report_interval == 0 and self._total_frames > 0:
            self._emit_backpressure_record(now)

    # ------------------------------------------------------------------
    # Extractor timing
    # ------------------------------------------------------------------

    def begin_extractor(self, name: str) -> None:
        """Start timing an extractor.

        Args:
            name: Extractor name.
        """
        self._extractor_start_ns = time.perf_counter_ns()

    def end_extractor(
        self,
        name: str,
        obs: Any = None,
        sub_timings: Optional[Dict[str, float]] = None,
    ) -> None:
        """Finish timing an extractor.

        Args:
            name: Extractor name.
            obs: Observation result (None if extractor failed/produced nothing).
            sub_timings: Optional sub-component timings from the extractor.
        """
        elapsed_ms = (time.perf_counter_ns() - self._extractor_start_ns) / 1_000_000
        self._extractor_timings[name] = elapsed_ms

        if obs is not None:
            self._observations_produced.append(name)
        else:
            self._observations_failed.append(name)
            self._extractor_error_counts[name] += 1

        # Emit ExtractorTimingRecord
        if self._hub.enabled:
            self._hub.emit(ExtractorTimingRecord(
                frame_id=self._frame_id,
                extractor_name=name,
                processing_ms=elapsed_ms,
                produced_observation=obs is not None,
                sub_timings_ms=dict(sub_timings) if sub_timings else {},
            ))

    # ------------------------------------------------------------------
    # Merge & Classifier
    # ------------------------------------------------------------------

    def record_merge(
        self,
        observations: List[Any],
        merged: Any,
        main_face_id: Optional[int] = None,
        main_face_source: str = "none",
    ) -> None:
        """Record observation merge details.

        Args:
            observations: Input observations before merge.
            merged: Merged observation.
            main_face_id: Main face ID if available.
            main_face_source: Source of main_face_id.
        """
        self._classifier_main_face_id = main_face_id

        if self._hub.enabled:
            input_sources = [
                getattr(o, "source", "unknown") for o in observations
            ]
            input_signal_counts = {
                getattr(o, "source", "unknown"): len(getattr(o, "signals", {}))
                for o in observations
            }
            merged_signal_keys = list(getattr(merged, "signals", {}).keys())

            self._hub.emit(ObservationMergeRecord(
                frame_id=self._frame_id,
                input_sources=input_sources,
                input_signal_counts=input_signal_counts,
                merged_signal_keys=merged_signal_keys,
                main_face_id=main_face_id,
                main_face_source=main_face_source,
            ))

    def record_classifier(self, classifier_obs: Any) -> None:
        """Record classifier result for main face tracking.

        Args:
            classifier_obs: FaceClassifier observation.
        """
        if classifier_obs is not None and hasattr(classifier_obs, "data"):
            data = classifier_obs.data
            if data is not None and hasattr(data, "main_face") and data.main_face is not None:
                self._classifier_main_face_id = data.main_face.face.face_id

    # ------------------------------------------------------------------
    # Fusion timing
    # ------------------------------------------------------------------

    def begin_fusion(self) -> None:
        """Start timing fusion."""
        self._fusion_start_ns = time.perf_counter_ns()

    def end_fusion(self, result: Any = None) -> None:
        """Finish timing fusion and record decision.

        Args:
            result: FusionResult from fusion.update().
        """
        self._fusion_ms = (time.perf_counter_ns() - self._fusion_start_ns) / 1_000_000

        if result is None:
            self._fusion_decision = "no_result"
        elif getattr(result, "should_trigger", False):
            self._fusion_decision = "triggered"
        else:
            metadata = getattr(result, "metadata", {}) or {}
            state = metadata.get("state", "")
            if state == "cooldown":
                self._fusion_decision = "cooldown"
            elif state == "gate_closed":
                self._fusion_decision = "gate_closed"
            else:
                self._fusion_decision = "no_trigger"

    # ------------------------------------------------------------------
    # Stats accessors (always work, trace-level independent)
    # ------------------------------------------------------------------

    def get_frame_stats(self) -> Dict[str, Any]:
        """Get current frame statistics for overlay.

        Returns:
            Dict with frame timing, extractor breakdown, fusion decision.
            Always returns data regardless of trace level.
        """
        if not self._history:
            return {}

        current = self._history[-1]
        rolling = self.get_rolling_stats()

        # Determine bottleneck
        timings = current["extractor_timings_ms"]
        total_ext_ms = sum(timings.values()) if timings else 0
        slowest = max(timings, key=timings.get) if timings else ""
        bottleneck_pct = (
            (timings[slowest] / current["total_frame_ms"] * 100)
            if slowest and current["total_frame_ms"] > 0
            else 0.0
        )

        # Determine main face display
        main_face_str = ""
        if self._classifier_main_face_id is not None:
            main_face_str = f"face#{self._classifier_main_face_id}"

        return {
            # Current frame
            "frame_id": current["frame_id"],
            "total_frame_ms": current["total_frame_ms"],
            "extractor_timings_ms": current["extractor_timings_ms"],
            "fusion_ms": current["fusion_ms"],
            "fusion_decision": current["fusion_decision"],
            # Bottleneck
            "slowest_extractor": slowest,
            "bottleneck_pct": bottleneck_pct,
            # Main face
            "main_face": main_face_str,
            # Rolling stats
            "effective_fps": rolling.get("effective_fps", 0.0),
            "target_fps": self._target_fps,
            "fps_ratio": rolling.get("fps_ratio", 0.0),
        }

    def get_rolling_stats(self) -> Dict[str, Any]:
        """Get rolling statistics over recent frames.

        Returns:
            Dict with effective FPS, per-extractor avg/p95/max, bottleneck.
        """
        if len(self._history) < 2:
            return {
                "effective_fps": 0.0,
                "fps_ratio": 0.0,
                "extractor_avg_ms": {},
                "extractor_p95_ms": {},
                "extractor_max_ms": {},
            }

        frames = list(self._history)
        n = len(frames)

        # Effective FPS from wall time
        total_ms = sum(f["total_frame_ms"] for f in frames)
        effective_fps = (n * 1000.0 / total_ms) if total_ms > 0 else 0.0
        fps_ratio = effective_fps / self._target_fps if self._target_fps > 0 else 0.0

        # Per-extractor stats
        ext_timings: Dict[str, List[float]] = collections.defaultdict(list)
        fusion_timings: List[float] = []

        for f in frames:
            for name, ms in f["extractor_timings_ms"].items():
                ext_timings[name].append(ms)
            if f["fusion_ms"] > 0:
                fusion_timings.append(f["fusion_ms"])

        ext_avg = {}
        ext_p95 = {}
        ext_max = {}
        for name, vals in ext_timings.items():
            sorted_vals = sorted(vals)
            ext_avg[name] = sum(vals) / len(vals)
            ext_max[name] = sorted_vals[-1]
            p95_idx = min(int(len(sorted_vals) * 0.95), len(sorted_vals) - 1)
            ext_p95[name] = sorted_vals[p95_idx]

        fusion_avg = sum(fusion_timings) / len(fusion_timings) if fusion_timings else 0.0

        # Slowest extractor
        slowest = max(ext_avg, key=ext_avg.get) if ext_avg else ""
        avg_frame_ms = total_ms / n if n > 0 else 0
        bottleneck_pct = (
            (ext_avg[slowest] / avg_frame_ms * 100)
            if slowest and avg_frame_ms > 0
            else 0.0
        )

        return {
            "effective_fps": effective_fps,
            "fps_ratio": fps_ratio,
            "extractor_avg_ms": ext_avg,
            "extractor_p95_ms": ext_p95,
            "extractor_max_ms": ext_max,
            "fusion_avg_ms": fusion_avg,
            "slowest_extractor": slowest,
            "bottleneck_pct": bottleneck_pct,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get session-level summary for end-of-session output.

        Also emits PipelineStatsRecord if hub is enabled.

        Returns:
            Dict with total frames, triggers, FPS, per-extractor stats.
        """
        now = time.perf_counter_ns()
        wall_sec = (now - self._session_start_ns) / 1e9 if self._session_start_ns else 0
        effective_fps = self._total_frames / wall_sec if wall_sec > 0 else 0

        # Per-extractor stats
        ext_stats = {}
        for name, vals in self._all_extractor_timings.items():
            if not vals:
                continue
            sorted_vals = sorted(vals)
            p95_idx = min(int(len(sorted_vals) * 0.95), len(sorted_vals) - 1)
            ext_stats[name] = {
                "avg_ms": sum(vals) / len(vals),
                "p95_ms": sorted_vals[p95_idx],
                "max_ms": sorted_vals[-1],
                "errors": self._extractor_error_counts.get(name, 0),
            }

        fusion_avg = (
            sum(self._all_fusion_timings) / len(self._all_fusion_timings)
            if self._all_fusion_timings
            else 0.0
        )

        gate_open_pct = (
            self._gate_open_count / self._total_frames * 100
            if self._total_frames > 0
            else 0.0
        )

        summary = {
            "total_frames": self._total_frames,
            "total_triggers": self._total_triggers,
            "wall_time_sec": wall_sec,
            "effective_fps": effective_fps,
            "target_fps": self._target_fps,
            "extractor_stats": ext_stats,
            "fusion_avg_ms": fusion_avg,
            "gate_open_pct": gate_open_pct,
        }

        # Emit PipelineStatsRecord
        if self._hub.enabled:
            self._hub.emit(PipelineStatsRecord(
                total_frames=self._total_frames,
                total_triggers=self._total_triggers,
                wall_time_sec=wall_sec,
                effective_fps=effective_fps,
                extractor_stats={
                    name: {k: round(v, 1) for k, v in stats.items()}
                    for name, stats in ext_stats.items()
                },
                fusion_avg_ms=fusion_avg,
                gate_open_pct=gate_open_pct,
            ))

        return summary

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit_backpressure_record(self, now_ns: int) -> None:
        """Emit a BackpressureRecord for the recent window."""
        frames = list(self._history)
        if not frames:
            return

        n = len(frames)
        wall_sec = (now_ns - self._report_start_ns) / 1e9
        effective_fps = n / wall_sec if wall_sec > 0 else 0
        fps_ratio = effective_fps / self._target_fps if self._target_fps > 0 else 0

        # Per-extractor stats for this window
        ext_timings: Dict[str, List[float]] = collections.defaultdict(list)
        fusion_timings: List[float] = []

        for f in frames:
            for name, ms in f["extractor_timings_ms"].items():
                ext_timings[name].append(ms)
            if f["fusion_ms"] > 0:
                fusion_timings.append(f["fusion_ms"])

        ext_avg = {}
        ext_max = {}
        ext_p95 = {}
        for name, vals in ext_timings.items():
            sorted_vals = sorted(vals)
            ext_avg[name] = sum(vals) / len(vals)
            ext_max[name] = sorted_vals[-1]
            p95_idx = min(int(len(sorted_vals) * 0.95), len(sorted_vals) - 1)
            ext_p95[name] = sorted_vals[p95_idx]

        slowest = max(ext_avg, key=ext_avg.get) if ext_avg else ""
        avg_frame_ms = sum(f["total_frame_ms"] for f in frames) / n if n > 0 else 0
        bottleneck_pct = (
            (ext_avg[slowest] / avg_frame_ms * 100)
            if slowest and avg_frame_ms > 0
            else 0.0
        )

        fusion_avg = sum(fusion_timings) / len(fusion_timings) if fusion_timings else 0.0

        self._hub.emit(BackpressureRecord(
            start_frame=frames[0]["frame_id"],
            end_frame=frames[-1]["frame_id"],
            frame_count=n,
            wall_time_sec=wall_sec,
            effective_fps=effective_fps,
            target_fps=self._target_fps,
            fps_ratio=fps_ratio,
            extractor_avg_ms=ext_avg,
            extractor_max_ms=ext_max,
            extractor_p95_ms=ext_p95,
            slowest_extractor=slowest,
            bottleneck_pct=bottleneck_pct,
            fusion_avg_ms=fusion_avg,
        ))

        # Reset report window
        self._report_start_ns = now_ns
        self._report_start_frame = self._frame_id


__all__ = ["PathwayMonitor"]
