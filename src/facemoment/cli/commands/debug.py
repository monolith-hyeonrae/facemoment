"""Debug command for facemoment CLI.

Unified debug command with DebugSession that consolidates the three
debug modes (pathway, simple, distributed) into a single session class.
"""

import sys
from typing import List, Optional, Tuple, Dict, Any

from facemoment.cli.utils import (
    create_video_stream,
    check_ml_dependencies,
    setup_observability,
    cleanup_observability,
)


def run_debug(args):
    """Run unified debug session with selected extractors.

    Supports:
    - Single extractor: -e face, -e pose, -e quality, -e gesture
    - Multiple extractors: -e face,pose or -e all (default)
    - Raw video preview: -e raw (no analysis, verify video input)
    - Dummy mode: --no-ml (replaces legacy 'visualize' command)
    - Distributed mode: --distributed (uses VenvWorker for process isolation)
    - Backend selection: --backend pathway (default) or simple
    """
    # Parse extractor selection
    extractor_arg = getattr(args, 'extractor', 'all')
    selected = _parse_extractor_arg(extractor_arg)

    show_window = not getattr(args, 'no_window', False)
    if not show_window and not args.output:
        print("Error: --output is required when using --no-window")
        sys.exit(1)

    # Raw video preview mode (no analysis)
    if 'raw' in selected:
        _run_raw_preview(args, show_window)
        return

    # Check if distributed mode is requested
    distributed = getattr(args, 'distributed', False)
    config_path = getattr(args, 'config', None)
    venv_face = getattr(args, 'venv_face', None)
    venv_pose = getattr(args, 'venv_pose', None)
    venv_gesture = getattr(args, 'venv_gesture', None)

    if venv_face or venv_pose or venv_gesture or config_path:
        distributed = True

    if distributed:
        session = DistributedDebugSession(
            args, selected, show_window,
            config_path=config_path,
            venv_face=venv_face,
            venv_pose=venv_pose,
            venv_gesture=venv_gesture,
        )
    else:
        backend = getattr(args, 'backend', None)
        use_ml = args.use_ml

        if use_ml is False or backend == "simple":
            session = SimpleDebugSession(args, selected, show_window)
        else:
            # PathwayDebugSession: inline (default) or --backend pathway
            session = PathwayDebugSession(args, selected, show_window)

    session.run()


# ---------------------------------------------------------------------------
# DebugSession base
# ---------------------------------------------------------------------------

class DebugSession:
    """Base class for debug sessions.

    Consolidates the common setup/loop/teardown pattern shared by
    pathway, simple, and distributed debug modes.

    Subclasses override:
    - _setup_pipeline(): create extractors, fusion, monitor
    - _process_frame(): run extraction and fusion for one frame
    - _teardown_pipeline(): cleanup pipeline-specific resources
    - _print_summary(): print session-specific summary
    """

    def __init__(self, args, selected: List[str], show_window: bool):
        self.args = args
        self.selected = selected
        self.show_window = show_window
        self.profile_mode = getattr(args, 'profile', False)
        self.roi = _parse_roi(getattr(args, 'roi', None)) or (0.3, 0.1, 0.7, 0.6)
        self.backend_label = ""

        # Initialized in _setup()
        self.vb = None
        self.source = None
        self.stream = None
        self.hub = None
        self.file_sink = None
        self.visualizer = None
        self.writer = None
        self.writer_initialized = False
        self.frame_count = 0

        # Report data (Phase 19c)
        self.report_data: Optional[Dict[str, Any]] = None

    def run(self):
        """Execute the full debug session."""
        self._setup()
        try:
            self._loop()
        finally:
            self._teardown()

    def _setup(self):
        """Common setup: observability, video, visualizer."""
        import cv2
        from facemoment.moment_detector.visualize import DebugVisualizer

        # Observability
        trace_level = getattr(self.args, 'trace', 'off')
        trace_output = getattr(self.args, 'trace_output', None)
        self.hub, self.file_sink = setup_observability(trace_level, trace_output)

        # Video
        try:
            self.vb, self.source, self.stream = create_video_stream(
                self.args.path, fps=self.args.fps
            )
        except IOError:
            print(f"Error: Cannot open {self.args.path}")
            sys.exit(1)

        # Pipeline-specific setup (must run before header to detect actual backend)
        self._setup_pipeline()

        # Header (after pipeline setup so backend_label is accurate)
        self._print_header()

        # ROI info
        roi_pct = f"{int(self.roi[0]*100)}%-{int(self.roi[2]*100)}% x {int(self.roi[1]*100)}%-{int(self.roi[3]*100)}%"
        print(f"ROI: {roi_pct}")

        print("-" * 50)
        print("Controls: [q] quit, [r] reset, [space] pause")
        print("Layers:   [1] face [2] pose [3] ROI [4] stats")
        print("          [5] timeline [6] trigger [7] fusion [8] frame info")
        print("-" * 50)

        # Profile backend info
        if self.profile_mode:
            self._print_backend_info()

        # Visualizer
        self.visualizer = DebugVisualizer()

    def _print_header(self):
        """Print session header."""
        print(f"Debug: {self.args.path}")
        print(f"Frames: {self.source.frame_count}, FPS: {self.source.fps:.1f}")
        print(f"Extractors: {', '.join(self.selected)}")
        print(f"Backend: {self.backend_label.lower()}")
        print(f"Window: {'enabled' if self.show_window else 'disabled'}")
        if self.profile_mode:
            print(f"Profile: enabled")
        print("-" * 50)

    def _print_backend_info(self):
        """Print backend info in profile mode. Override in subclasses."""
        pass

    def _setup_pipeline(self):
        """Override: create pipeline-specific resources."""
        raise NotImplementedError

    def _loop(self):
        """Common frame loop."""
        import cv2

        for frame in self.stream:
            # Pipeline-specific processing
            result = self._process_frame(frame)
            if result is None:
                continue

            debug_image, timing_info = result

            # Initialize writer on first frame (canvas includes panels)
            if self.args.output and not self.writer_initialized:
                dh, dw = debug_image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.writer = cv2.VideoWriter(
                    self.args.output, fourcc, self.args.fps, (dw, dh)
                )
                self.writer_initialized = True

            if self.writer:
                self.writer.write(debug_image)

            if self.show_window:
                cv2.imshow("Debug", debug_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    self._on_reset()
                elif key == ord(" "):
                    cv2.waitKey(0)
                elif ord("1") <= key <= ord("8"):
                    self._toggle_layer(key - ord("0"))

            self.frame_count += 1

            # Progress output
            if self.profile_mode and timing_info:
                detect_ms = timing_info.get('detect_ms', 0)
                expr_ms = timing_info.get('expression_ms', 0)
                total_ms = timing_info.get('total_ms', 0)
                print(
                    f"\rFrame {frame.frame_id}: detect={detect_ms:.1f}ms, "
                    f"expression={expr_ms:.1f}ms, total={total_ms:.1f}ms    ",
                    end="", flush=True,
                )
            elif self.frame_count % 100 == 0:
                print(
                    f"\rFrame {self.frame_count}/{self.source.frame_count}",
                    end="", flush=True,
                )

        print()

    def _process_frame(self, frame) -> Optional[Tuple]:
        """Override: process one frame.

        Returns:
            (debug_image, timing_info) or None to skip frame.
        """
        raise NotImplementedError

    def _toggle_layer(self, layer_num: int):
        """Toggle a visualization layer by number (1-8)."""
        if self.visualizer is None:
            return
        from facemoment.moment_detector.visualize import DebugLayer
        try:
            layer = DebugLayer(layer_num)
        except ValueError:
            return
        new_state = self.visualizer.layers.toggle(layer)
        state_str = "ON" if new_state else "OFF"
        print(f"\n  Layer {layer_num} ({layer.name}): {state_str}")

    def _on_reset(self):
        """Called when user presses 'r'. Override to reset fusion/visualizer."""
        if self.visualizer:
            self.visualizer.reset()

    def _teardown(self):
        """Common teardown: cleanup video, writer, observability."""
        import cv2

        if self.vb:
            self.vb.disconnect()
        if self.writer:
            self.writer.release()
            print(f"Saved: {self.args.output}")
        if self.show_window:
            cv2.destroyAllWindows()

        self._teardown_pipeline()
        self._print_summary()

        # Generate HTML report if requested
        report_path = getattr(self.args, 'report', None)
        if report_path and self.report_data:
            try:
                from facemoment.moment_detector.visualize.report import generate_report
                generate_report(self.report_data, report_path)
                print(f"Report saved: {report_path}")
            except Exception as e:
                print(f"Warning: Failed to generate report: {e}")

        cleanup_observability(self.hub, self.file_sink)

        print(f"Processed {self.frame_count} frames")

    def _teardown_pipeline(self):
        """Override: cleanup pipeline-specific resources."""
        pass

    def _print_summary(self):
        """Override: print session-specific summary."""
        pass


# ---------------------------------------------------------------------------
# PathwayDebugSession
# ---------------------------------------------------------------------------

class PathwayDebugSession(DebugSession):
    """Debug session using FacemomentPipeline with PathwayMonitor.

    Default: inline processing (smooth frame-by-frame visualization).
    With --backend pathway: uses actual Pathway streaming engine via
    on_frame_result callback. Pathway batches frames which can cause
    stuttering in the visualization, but uses the real streaming pipeline.
    """

    def __init__(self, args, selected, show_window):
        super().__init__(args, selected, show_window)
        self.backend_label = "PATHWAY"
        self.pipeline = None
        self.monitor = None
        self._use_pathway = False
        self._force_pathway = getattr(args, 'backend', None) == 'pathway'

    def _setup_pipeline(self):
        from facemoment.pipeline.pathway_pipeline import FacemomentPipeline, PATHWAY_AVAILABLE
        from facemoment.observability.pathway_monitor import PathwayMonitor

        # Build extractor list
        extractor_names = []
        if 'pose' in self.selected or 'all' in self.selected:
            extractor_names.append('pose')
        if 'gesture' in self.selected or 'all' in self.selected:
            extractor_names.append('gesture')
        if 'face' in self.selected or 'all' in self.selected:
            extractor_names.append('face')
        if 'quality' in self.selected or 'all' in self.selected:
            extractor_names.append('quality')
        if not extractor_names:
            extractor_names = ['dummy']

        self.pipeline = FacemomentPipeline(
            extractors=extractor_names,
            fusion_config={"cooldown_sec": 2.0, "main_only": True},
            auto_inject_classifier=True,
        )
        self.pipeline.initialize()

        # Detect actual backend
        # Default: inline (smooth frame-by-frame visualization)
        # --backend pathway: force Pathway streaming engine
        if self._force_pathway:
            if not PATHWAY_AVAILABLE:
                self._use_pathway = False
                self.backend_label = "INLINE (pathway unavailable)"
                print("WARNING: Pathway not available — falling back to inline")
            elif self.pipeline.workers:
                self._use_pathway = False
                self.backend_label = "INLINE (subprocess workers active)"
                print("WARNING: ProcessWorkers active — falling back to inline")
            else:
                self._use_pathway = True
                self.backend_label = "PATHWAY"
        else:
            # Default: inline for smooth debug visualization
            self._use_pathway = False
            self.backend_label = "PATHWAY (inline)"

        # Print extractor status
        initialized_names = {ext.name for ext in self.pipeline.extractors}
        worker_names = set(self.pipeline.workers.keys())
        print("Extractors:")
        for name in extractor_names:
            if name in worker_names:
                print(f"  [+] {name}: enabled (subprocess)")
            elif name in initialized_names:
                print(f"  [+] {name}: enabled")
            else:
                print(f"  [-] {name}: failed to load")

        if 'face' in extractor_names:
            if 'face_classifier' in initialized_names:
                print(f"  [+] face_classifier: enabled (auto-injected)")
            else:
                print(f"  [-] face_classifier: failed to load")

        if not self.pipeline.extractors and not self.pipeline.workers:
            print("Error: No extractors available")
            self.vb.disconnect()
            cleanup_observability(self.hub, self.file_sink)
            sys.exit(1)

        if self.pipeline.fusion:
            print(f"  [+] fusion: HighlightFusion (main_only=True)")

        self.monitor = PathwayMonitor(hub=self.hub, target_fps=self.args.fps)

    def _print_backend_info(self):
        if self.pipeline:
            print("\nBackends:")
            for ext in self.pipeline.extractors:
                if hasattr(ext, 'get_backend_info'):
                    info = ext.get_backend_info()
                    for component, backend_name in info.items():
                        print(f"  {component.capitalize():12}: {backend_name}")
            print("-" * 50)

    def _loop(self):
        """Route to Pathway or inline loop based on availability."""
        if self._use_pathway:
            self._loop_pathway()
        else:
            self._loop_inline()

    def _loop_pathway(self):
        """Single-phase loop: visualize each frame as PathwayBackend processes it."""
        import cv2
        from visualpath.backends.pathway import PathwayBackend

        fusion = self.pipeline.fusion
        session = self  # capture for closure
        stop_requested = False

        def on_frame_result(frame, obs_list, fusion_result):
            nonlocal stop_requested
            if stop_requested:
                return

            obs_dict = {obs.source: obs for obs in obs_list}
            classifier_obs = obs_dict.get("face_classifier")
            face_obs = obs_dict.get("face") or obs_dict.get("dummy")
            is_gate_open = fusion.is_gate_open if fusion else False
            in_cooldown = fusion.in_cooldown if fusion else False

            debug_image = session.visualizer.create_debug_view(
                frame,
                face_obs=face_obs,
                pose_obs=obs_dict.get("pose"),
                quality_obs=obs_dict.get("quality"),
                classifier_obs=classifier_obs,
                fusion_result=fusion_result,
                is_gate_open=is_gate_open,
                in_cooldown=in_cooldown,
                roi=session.roi,
                backend_label=session.backend_label,
            )

            # Writer
            if session.args.output and not session.writer_initialized:
                dh, dw = debug_image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                session.writer = cv2.VideoWriter(
                    session.args.output, fourcc, session.args.fps, (dw, dh)
                )
                session.writer_initialized = True

            if session.writer:
                session.writer.write(debug_image)

            if session.show_window:
                cv2.imshow("Debug", debug_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    stop_requested = True
                elif key == ord(" "):
                    cv2.waitKey(0)
                elif ord("1") <= key <= ord("8"):
                    session._toggle_layer(key - ord("0"))

            session.frame_count += 1

        backend = PathwayBackend(window_ns=self.pipeline._window_ns)
        backend.run(
            frames=self.stream,
            extractors=self.pipeline.extractors,
            fusion=fusion,
            on_frame_result=on_frame_result,
        )

    def _loop_inline(self):
        """Inline frame-by-frame loop (fallback when Pathway unavailable)."""
        import cv2

        for frame in self.stream:
            result = self._process_frame_inline(frame)
            if result is None:
                continue

            debug_image, timing_info = result

            if self.args.output and not self.writer_initialized:
                dh, dw = debug_image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.writer = cv2.VideoWriter(
                    self.args.output, fourcc, self.args.fps, (dw, dh)
                )
                self.writer_initialized = True

            if self.writer:
                self.writer.write(debug_image)

            if self.show_window:
                cv2.imshow("Debug", debug_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    self._on_reset()
                elif key == ord(" "):
                    cv2.waitKey(0)
                elif ord("1") <= key <= ord("8"):
                    self._toggle_layer(key - ord("0"))

            self.frame_count += 1

            if self.profile_mode and timing_info:
                detect_ms = timing_info.get('detect_ms', 0)
                expr_ms = timing_info.get('expression_ms', 0)
                total_ms = timing_info.get('total_ms', 0)
                print(
                    f"\rFrame {frame.frame_id}: detect={detect_ms:.1f}ms, "
                    f"expression={expr_ms:.1f}ms, total={total_ms:.1f}ms    ",
                    end="", flush=True,
                )
            elif self.frame_count % 100 == 0:
                print(
                    f"\rFrame {self.frame_count}/{self.source.frame_count}",
                    end="", flush=True,
                )

        print()

    def _process_frame(self, frame):
        # Unused — _loop() is overridden. Kept for interface compatibility.
        return self._process_frame_inline(frame)

    def _process_frame_inline(self, frame):
        """Process one frame inline (no PathwayBackend)."""
        self.monitor.begin_frame(frame)

        observations = {}
        deps = {}
        classifier_obs = None

        # Inline extractors (deps accumulated)
        for ext in self.pipeline.extractors:
            try:
                extractor_deps = None
                if ext.depends:
                    extractor_deps = {
                        n: deps[n] for n in ext.depends if n in deps
                    }
                    # Composite "face" extractor satisfies "face_detect" dependency
                    if "face_detect" in ext.depends and "face_detect" not in extractor_deps and "face" in deps:
                        extractor_deps["face"] = deps["face"]
                self.monitor.begin_extractor(ext.name)
                try:
                    obs = ext.extract(frame, extractor_deps)
                except TypeError:
                    obs = ext.extract(frame)
                sub_timings = getattr(obs, "timing", None) if obs else None
                self.monitor.end_extractor(ext.name, obs, sub_timings=sub_timings)
                if obs:
                    observations[ext.name] = obs
                    deps[ext.name] = obs
                    if ext is self.pipeline._classifier:
                        classifier_obs = obs
            except Exception:
                self.monitor.end_extractor(ext.name, None)

        # Subprocess workers
        for name, worker in self.pipeline.workers.items():
            try:
                self.monitor.begin_extractor(name)
                result = worker.process(frame, deps=deps)
                if result.observation:
                    observations[name] = result.observation
                    deps[name] = result.observation
                self.monitor.end_extractor(name, result.observation)
            except Exception:
                self.monitor.end_extractor(name, None)

        if classifier_obs:
            self.monitor.record_classifier(classifier_obs)

        # Fusion
        fusion_result = None
        is_gate_open = False
        if self.pipeline.fusion:
            obs_list = list(observations.values())
            merged_obs = self.pipeline._merge_observations(obs_list, frame)

            main_face_id = None
            main_face_source = "none"
            if classifier_obs and hasattr(classifier_obs, "data") and classifier_obs.data:
                data = classifier_obs.data
                if hasattr(data, "main_face") and data.main_face:
                    main_face_id = data.main_face.face.face_id
                    main_face_source = "classifier_obs"
            elif hasattr(merged_obs, "signals") and "main_face_id" in merged_obs.signals:
                main_face_id = merged_obs.signals["main_face_id"]
                main_face_source = "merged_signals"

            self.monitor.record_merge(obs_list, merged_obs, main_face_id, main_face_source)

            self.monitor.begin_fusion()
            fusion_result = self.pipeline.fusion.update(merged_obs, classifier_obs=classifier_obs)
            self.monitor.end_fusion(fusion_result)

            is_gate_open = self.pipeline.fusion.is_gate_open

        self.monitor.end_frame(gate_open=is_gate_open)

        # Timing info for profile mode
        timing_info = None
        if self.profile_mode:
            face_obs = observations.get("face")
            if face_obs and face_obs.timing:
                timing_info = face_obs.timing

        debug_image = self.visualizer.create_debug_view(
            frame,
            face_obs=observations.get("face") or observations.get("dummy"),
            pose_obs=observations.get("pose"),
            quality_obs=observations.get("quality"),
            classifier_obs=classifier_obs,
            fusion_result=fusion_result,
            is_gate_open=is_gate_open,
            in_cooldown=self.pipeline.fusion.in_cooldown if self.pipeline.fusion else False,
            timing=timing_info if self.profile_mode else None,
            roi=self.roi,
            monitor_stats=self.monitor.get_frame_stats(),
            backend_label=self.backend_label,
        )

        return debug_image, timing_info

    def _on_reset(self):
        if self.pipeline and self.pipeline.fusion:
            self.pipeline.fusion.reset()
        super()._on_reset()

    def _teardown_pipeline(self):
        if self.pipeline:
            self.pipeline.cleanup()

    def _print_summary(self):
        if self.monitor:
            summary = self.monitor.get_summary()
            _print_pathway_summary(summary)
            self.report_data = _build_report_data(
                summary, self.visualizer, self.backend_label,
            )


# ---------------------------------------------------------------------------
# SimpleDebugSession
# ---------------------------------------------------------------------------

class SimpleDebugSession(DebugSession):
    """Debug session using simple/library mode with raw extractors."""

    def __init__(self, args, selected, show_window):
        super().__init__(args, selected, show_window)
        self.backend_label = "SIMPLE"
        self.extractors = []
        self.face_classifier = None
        self.fusion = None

    def _setup_pipeline(self):
        from facemoment.moment_detector.extractors import QualityExtractor
        from facemoment.moment_detector.extractors.face_classifier import FaceClassifierExtractor

        use_ml = self.args.use_ml
        ml_mode = "auto" if use_ml is None else ("enabled" if use_ml else "disabled")
        print(f"ML backends: {ml_mode}")

        extractor_status = {}

        # Load torch-based extractors first
        if 'pose' in self.selected or 'all' in self.selected:
            if use_ml is not False and _try_load_extractor('pose', self.extractors, self.args):
                extractor_status['pose'] = 'enabled'
            else:
                extractor_status['pose'] = 'disabled' if use_ml is not False else 'skipped'

        if 'gesture' in self.selected or 'all' in self.selected:
            if use_ml is not False and _try_load_extractor('gesture', self.extractors, self.args):
                extractor_status['gesture'] = 'enabled'
            else:
                extractor_status['gesture'] = 'disabled' if use_ml is not False else 'skipped'

        if 'face' in self.selected or 'all' in self.selected:
            if use_ml is False:
                from facemoment.moment_detector.extractors import DummyExtractor
                self.extractors.append(DummyExtractor(num_faces=2, spike_probability=0.1))
                extractor_status['face'] = 'dummy'
            elif _try_load_extractor('face', self.extractors, self.args):
                extractor_status['face'] = 'enabled'
            else:
                extractor_status['face'] = 'disabled'

        if 'quality' in self.selected or 'all' in self.selected:
            self.extractors.append(QualityExtractor())
            extractor_status['quality'] = 'enabled'

        # Face classifier
        if extractor_status.get('face') == 'enabled':
            self.face_classifier = FaceClassifierExtractor(
                min_track_frames=3, min_area_ratio=0.005, min_confidence=0.5,
            )
            extractor_status['face_classifier'] = 'enabled'

        for name, status in extractor_status.items():
            icon = "+" if status == 'enabled' else ("-" if status == 'disabled' else "o")
            print(f"  [{icon}] {name}: {status}")

        if not self.extractors:
            print("Error: No extractors available")
            sys.exit(1)

        # Fusion
        if any(e.name in ('face', 'dummy') for e in self.extractors):
            if any(e.name == 'face' for e in self.extractors):
                from facemoment.moment_detector.fusion import HighlightFusion
                self.fusion = HighlightFusion()
                print("  [+] fusion: HighlightFusion")
            else:
                from facemoment.moment_detector.fusion import DummyFusion
                self.fusion = DummyFusion()
                print("  [+] fusion: DummyFusion")

        # Initialize extractors
        for ext in self.extractors:
            if ext.name not in ('quality',):
                try:
                    ext.initialize()
                except Exception as e:
                    print(f"Warning: Failed to initialize {ext.name}: {e}")

        if self.face_classifier:
            self.face_classifier.initialize()

    def _print_backend_info(self):
        print("\nBackends:")
        for ext in self.extractors:
            if hasattr(ext, 'get_backend_info'):
                info = ext.get_backend_info()
                for component, backend_name in info.items():
                    print(f"  {component.capitalize():12}: {backend_name}")
        print("-" * 50)

    def _process_frame(self, frame):
        observations = {}
        for ext in self.extractors:
            try:
                obs = ext.extract(frame)
                if obs:
                    observations[ext.name] = obs
            except Exception:
                pass

        classifier_obs = None
        if self.face_classifier:
            face_obs = observations.get("face")
            if face_obs:
                try:
                    classifier_obs = self.face_classifier.extract(frame, {"face": face_obs})
                except Exception:
                    pass

        fusion_result = None
        if self.fusion:
            fusion_obs = observations.get("face") or observations.get("dummy")
            if fusion_obs:
                fusion_result = self.fusion.update(fusion_obs, classifier_obs=classifier_obs)

        timing_info = None
        if self.profile_mode:
            face_obs = observations.get("face")
            if face_obs and face_obs.timing:
                timing_info = face_obs.timing

        debug_image = self.visualizer.create_debug_view(
            frame,
            face_obs=observations.get("face") or observations.get("dummy"),
            pose_obs=observations.get("pose"),
            quality_obs=observations.get("quality"),
            classifier_obs=classifier_obs,
            fusion_result=fusion_result,
            is_gate_open=self.fusion.is_gate_open if self.fusion else False,
            in_cooldown=self.fusion.in_cooldown if self.fusion else False,
            timing=timing_info if self.profile_mode else None,
            roi=self.roi,
            backend_label="SIMPLE",
        )

        return debug_image, timing_info

    def _on_reset(self):
        if self.fusion:
            self.fusion.reset()
        super()._on_reset()

    def _teardown_pipeline(self):
        for ext in self.extractors:
            if hasattr(ext, 'cleanup'):
                ext.cleanup()
        if self.face_classifier:
            self.face_classifier.cleanup()


# ---------------------------------------------------------------------------
# DistributedDebugSession
# ---------------------------------------------------------------------------

class DistributedDebugSession(DebugSession):
    """Debug session using PipelineOrchestrator (distributed mode)."""

    def __init__(
        self, args, selected, show_window, *,
        config_path=None, venv_face=None, venv_pose=None, venv_gesture=None,
    ):
        super().__init__(args, selected, show_window)
        self.backend_label = "DISTRIBUTED"
        self.config_path = config_path
        self.venv_face = venv_face
        self.venv_pose = venv_pose
        self.venv_gesture = venv_gesture
        self.orchestrator = None
        self.temp_clip_dir = None
        self.trigger_count = 0

    def _print_header(self):
        print(f"Debug: {self.args.path}")
        print(f"Mode: DISTRIBUTED")
        print("-" * 50)

    def _setup_pipeline(self):
        import tempfile
        from facemoment.pipeline import (
            PipelineOrchestrator,
            PipelineConfig,
            ExtractorConfig,
            FusionConfig,
        )

        self.temp_clip_dir = tempfile.mkdtemp(prefix="facemoment_debug_")

        if self.config_path:
            print(f"Loading config from: {self.config_path}")
            config = PipelineConfig.from_yaml(self.config_path)
            config.clip_output_dir = self.temp_clip_dir
            config.fps = int(self.args.fps)
        else:
            include_face = 'face' in self.selected or 'all' in self.selected
            include_pose = 'pose' in self.selected or 'all' in self.selected
            include_gesture = 'gesture' in self.selected or 'all' in self.selected
            include_quality = 'quality' in self.selected or 'all' in self.selected

            extractors = []
            if include_face:
                extractors.append(ExtractorConfig(name="face", venv_path=self.venv_face))
            if include_pose:
                extractors.append(ExtractorConfig(name="pose", venv_path=self.venv_pose))
            if include_gesture and self.venv_gesture:
                extractors.append(ExtractorConfig(name="gesture", venv_path=self.venv_gesture))
            if include_quality:
                extractors.append(ExtractorConfig(name="quality"))
            if not extractors:
                extractors.append(ExtractorConfig(name="dummy"))

            config = PipelineConfig(
                extractors=extractors,
                fusion=FusionConfig(cooldown_sec=2.0),
                clip_output_dir=self.temp_clip_dir,
                fps=int(self.args.fps),
            )

        print("Extractors:")
        for ext_config in config.extractors:
            isolation = ext_config.effective_isolation.name
            venv = ext_config.venv_path or "(current)"
            print(f"  {ext_config.name}: {isolation} [{venv}]")

        print(f"Fusion: {config.fusion.name} (cooldown={config.fusion.cooldown_sec}s)")

        self.orchestrator = PipelineOrchestrator.from_config(config)

    def _loop(self):
        """Override loop to use orchestrator.run_stream()."""
        import cv2

        try:
            for frame, observations, fusion_result in self.orchestrator.run_stream(
                str(self.args.path), fps=int(self.args.fps)
            ):
                self.frame_count += 1

                obs_dict = {obs.source: obs for obs in observations}

                if fusion_result and fusion_result.should_trigger:
                    self.trigger_count += 1
                    event_time_sec = (
                        fusion_result.trigger.event_time_ns / 1e9
                        if fusion_result.trigger and fusion_result.trigger.event_time_ns
                        else 0
                    )
                    print(
                        f"\n  TRIGGER #{self.trigger_count}: {fusion_result.reason} "
                        f"(score={fusion_result.score:.2f}, t={event_time_sec:.2f}s)"
                    )

                is_gate_open = False
                in_cooldown = False
                if fusion_result:
                    in_cooldown = not fusion_result.should_trigger and fusion_result.trigger is None

                debug_image = self.visualizer.create_debug_view(
                    frame,
                    face_obs=obs_dict.get("face") or obs_dict.get("merged") or obs_dict.get("dummy"),
                    pose_obs=obs_dict.get("pose"),
                    quality_obs=obs_dict.get("quality"),
                    fusion_result=fusion_result,
                    is_gate_open=is_gate_open,
                    in_cooldown=in_cooldown,
                    backend_label="DISTRIBUTED",
                )

                if self.args.output and not self.writer_initialized:
                    dh, dw = debug_image.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self.writer = cv2.VideoWriter(
                        self.args.output, fourcc, self.args.fps, (dw, dh)
                    )
                    self.writer_initialized = True

                if self.writer:
                    self.writer.write(debug_image)

                if self.show_window:
                    cv2.imshow("Debug (Distributed)", debug_image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord(" "):
                        cv2.waitKey(0)
                    elif ord("1") <= key <= ord("8"):
                        self._toggle_layer(key - ord("0"))

                if self.frame_count % 100 == 0:
                    print(f"\rFrame {self.frame_count}...", end="", flush=True)

            print()

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nError during processing: {e}")
            import traceback
            traceback.print_exc()

    def _process_frame(self, frame):
        # Not used — _loop() is overridden
        pass

    def _teardown_pipeline(self):
        import shutil
        if self.temp_clip_dir:
            try:
                shutil.rmtree(self.temp_clip_dir)
            except Exception:
                pass

    def _print_summary(self):
        if self.orchestrator:
            stats = self.orchestrator.get_stats()
            print("-" * 50)
            print(f"Debug session complete")
            print(f"  Frames processed: {stats.frames_processed}")
            print(f"  Triggers fired: {stats.triggers_fired}")
            if stats.worker_stats:
                print("\nWorker statistics:")
                for name, ws in stats.worker_stats.items():
                    if ws["frames"] > 0:
                        avg_ms = ws["total_ms"] / ws["frames"]
                        print(f"  {name}: {ws['frames']} frames, avg {avg_ms:.1f}ms, errors: {ws['errors']}")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _print_pathway_summary(summary: dict) -> None:
    """Print Pathway pipeline session summary to console."""
    if not summary or summary.get("total_frames", 0) == 0:
        return

    sep = "=" * 50
    print()
    print(sep)
    print("Pathway Pipeline Summary")
    print(sep)
    print(f"  Duration:      {summary['wall_time_sec']:.1f}s ({summary['total_frames']} frames)")
    eff_fps = summary["effective_fps"]
    target = summary.get("target_fps", 0)
    ratio_str = f" / {target:.0f}" if target > 0 else ""
    print(f"  Effective FPS: {eff_fps:.1f}{ratio_str}")
    print(f"  Triggers:      {summary['total_triggers']}")

    ext_stats = summary.get("extractor_stats", {})
    if ext_stats:
        print()
        print("  Extractor Performance:")
        print(f"  {'Extractor':<14} {'Avg ms':>8} {'P95 ms':>8} {'Max ms':>8} {'Errors':>8}")
        print("  " + "-" * 48)
        for name, stats in ext_stats.items():
            avg = stats.get("avg_ms", 0)
            p95 = stats.get("p95_ms", 0)
            mx = stats.get("max_ms", 0)
            errs = int(stats.get("errors", 0))
            print(f"  {name:<14} {avg:>8.1f} {p95:>8.1f} {mx:>8.1f} {errs:>8}")

    fusion_avg = summary.get("fusion_avg_ms", 0)
    if fusion_avg > 0:
        print(f"  {'fusion':<14} {fusion_avg:>8.1f}")

    if ext_stats:
        slowest = max(ext_stats, key=lambda k: ext_stats[k].get("avg_ms", 0))
        total_avg = sum(s.get("avg_ms", 0) for s in ext_stats.values()) + fusion_avg
        if total_avg > 0:
            pct = ext_stats[slowest].get("avg_ms", 0) / total_avg * 100
            print(f"\n  Bottleneck: {slowest} ({pct:.0f}% of frame time)")

    gate_pct = summary.get("gate_open_pct", 0)
    print(f"  Gate: open {gate_pct:.0f}% of frames")
    print(sep)


def _build_report_data(
    summary: Optional[dict],
    visualizer,
    backend_label: str,
) -> Optional[Dict[str, Any]]:
    """Build report data dict from session results."""
    if not summary:
        return None
    return {
        "backend": backend_label,
        "summary": summary,
        "trigger_thumbs": (
            visualizer._stats_panel._trigger_thumbs if visualizer else []
        ),
    }


def _parse_extractor_arg(arg: str) -> List[str]:
    """Parse extractor argument into list of extractor names."""
    if arg == 'all':
        return ['all']
    if arg in ('raw', 'none'):
        return ['raw']

    parts = [p.strip().lower() for p in arg.split(',')]
    valid = {'face', 'pose', 'quality', 'gesture', 'all', 'raw', 'none'}
    for p in parts:
        if p not in valid:
            print(f"Warning: Unknown extractor '{p}', valid: {valid}")
    return [p for p in parts if p in valid]


def _run_raw_preview(args, show_window: bool):
    """Run raw video preview without any analysis."""
    import cv2

    try:
        vb, source, stream = create_video_stream(args.path, fps=args.fps)
    except IOError:
        print(f"Error: Cannot open {args.path}")
        sys.exit(1)

    duration_sec = source.frame_count / source.fps if source.fps > 0 else 0

    print("=" * 60)
    print("Raw Video Preview (No Analysis)")
    print("=" * 60)
    print(f"  File: {args.path}")
    print(f"  Resolution: {source.width} x {source.height}")
    print(f"  Native FPS: {source.fps:.2f}")
    print(f"  Preview FPS: {args.fps}")
    print(f"  Total frames: {source.frame_count}")
    print(f"  Duration: {duration_sec:.1f}s ({duration_sec/60:.1f}min)")
    print("-" * 60)
    print("Controls: [q] quit, [space] pause")
    print("-" * 60)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (source.width, source.height))

    frame_count = 0
    try:
        for frame in stream:
            image = frame.data
            overlay = image.copy()
            info_text = f"Frame: {frame.frame_id} | Time: {frame.t_src_ns / 1e9:.2f}s | {source.width}x{source.height}"
            cv2.putText(overlay, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(
                overlay, "[RAW PREVIEW - No Analysis]", (10, source.height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
            )

            if writer:
                writer.write(overlay)
            if show_window:
                cv2.imshow("Raw Preview", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    cv2.waitKey(0)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"\rFrame {frame_count}/{source.frame_count}", end="", flush=True)

        print()
    finally:
        vb.disconnect()
        if writer:
            writer.release()
            print(f"Saved: {args.output}")
        if show_window:
            cv2.destroyAllWindows()

    print(f"Previewed {frame_count} frames")


def _parse_roi(roi_str: Optional[str]) -> Optional[tuple]:
    """Parse ROI string to tuple."""
    if not roi_str:
        return None
    try:
        parts = [float(x.strip()) for x in roi_str.split(',')]
        if len(parts) != 4:
            print(f"Warning: Invalid ROI format '{roi_str}', expected x1,y1,x2,y2")
            return None
        x1, y1, x2, y2 = parts
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            print(f"Warning: Invalid ROI values '{roi_str}', must be 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
            return None
        return (x1, y1, x2, y2)
    except ValueError:
        print(f"Warning: Cannot parse ROI '{roi_str}'")
        return None


def _try_load_extractor(name: str, extractors: list, args) -> bool:
    """Try to load and add an extractor."""
    device = getattr(args, 'device', 'cuda:0')
    roi = _parse_roi(getattr(args, 'roi', None))

    try:
        if name == 'face':
            if not check_ml_dependencies("face"):
                return False
            from facemoment.moment_detector.extractors import FaceExtractor
            extractors.append(FaceExtractor(device=device, roi=roi))
            return True
        elif name == 'pose':
            if not check_ml_dependencies("pose"):
                return False
            from facemoment.moment_detector.extractors import PoseExtractor
            extractors.append(PoseExtractor(device=device))
            return True
        elif name == 'gesture':
            try:
                import mediapipe
            except ImportError:
                print("  GestureExtractor requires mediapipe")
                return False
            from facemoment.moment_detector.extractors import GestureExtractor
            extractors.append(GestureExtractor())
            return True
    except Exception as e:
        print(f"  Failed to load {name}: {e}")
        return False

    return False
