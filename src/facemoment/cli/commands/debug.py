"""Debug command for facemoment CLI.

Unified debug command that replaces separate debug-face, debug-pose, etc.
"""

import sys
from typing import List, Optional, Tuple

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
    """
    import cv2

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

    # If any venv path is provided, enable distributed mode
    if venv_face or venv_pose or venv_gesture or config_path:
        distributed = True

    if distributed:
        _run_distributed_debug(args, config_path, venv_face, venv_pose, venv_gesture, selected, show_window)
        return

    from facemoment.moment_detector.extractors import QualityExtractor
    from facemoment.moment_detector.extractors.face_classifier import FaceClassifierExtractor
    from facemoment.moment_detector.visualize import DebugVisualizer

    # Setup observability
    trace_level = getattr(args, 'trace', 'off')
    trace_output = getattr(args, 'trace_output', None)
    hub, file_sink = setup_observability(trace_level, trace_output)

    # Open video
    try:
        vb, source, stream = create_video_stream(args.path, fps=args.fps)
    except IOError:
        print(f"Error: Cannot open {args.path}")
        sys.exit(1)

    # Determine ML mode
    use_ml = args.use_ml
    ml_mode = "auto" if use_ml is None else ("enabled" if use_ml else "disabled")

    # Profile mode
    profile_mode = getattr(args, 'profile', False)

    print(f"Debug: {args.path}")
    print(f"Frames: {source.frame_count}, FPS: {source.fps:.1f}")
    print(f"Extractors: {', '.join(selected)}")
    print(f"ML backends: {ml_mode}")
    print(f"Window: {'enabled' if show_window else 'disabled'}")
    if profile_mode:
        print(f"Profile: enabled")
    print("-" * 50)

    # Load extractors
    extractors = []
    extractor_status = {}

    if 'face' in selected or 'all' in selected:
        if use_ml is False:
            # Dummy mode
            from facemoment.moment_detector.extractors import DummyExtractor
            extractors.append(DummyExtractor(num_faces=2, spike_probability=0.1))
            extractor_status['face'] = 'dummy'
        elif _try_load_extractor('face', extractors, args):
            extractor_status['face'] = 'enabled'
        else:
            extractor_status['face'] = 'disabled'

    if 'pose' in selected or 'all' in selected:
        if use_ml is not False and _try_load_extractor('pose', extractors, args):
            extractor_status['pose'] = 'enabled'
        else:
            extractor_status['pose'] = 'disabled' if use_ml is not False else 'skipped'

    if 'gesture' in selected or 'all' in selected:
        if use_ml is not False and _try_load_extractor('gesture', extractors, args):
            extractor_status['gesture'] = 'enabled'
        else:
            extractor_status['gesture'] = 'disabled' if use_ml is not False else 'skipped'

    if 'quality' in selected or 'all' in selected:
        extractors.append(QualityExtractor())
        extractor_status['quality'] = 'enabled'

    # Add FaceClassifierExtractor if face is enabled
    face_classifier = None
    if extractor_status.get('face') == 'enabled':
        face_classifier = FaceClassifierExtractor(
            min_track_frames=3,
            min_area_ratio=0.005,
            min_confidence=0.5,
        )
        extractor_status['face_classifier'] = 'enabled'

    # Print status
    for name, status in extractor_status.items():
        icon = "+" if status == 'enabled' else ("-" if status == 'disabled' else "o")
        print(f"  [{icon}] {name}: {status}")

    if not extractors:
        print("Error: No extractors available")
        sys.exit(1)

    # Setup fusion if face available
    fusion = None
    if any(e.name in ('face', 'dummy') for e in extractors):
        if any(e.name == 'face' for e in extractors):
            from facemoment.moment_detector.fusion import HighlightFusion
            fusion = HighlightFusion()
            print("  [+] fusion: HighlightFusion")
        else:
            from facemoment.moment_detector.fusion import DummyFusion
            fusion = DummyFusion()
            print("  [+] fusion: DummyFusion")

    # Parse ROI for visualization (use default if not specified)
    roi = _parse_roi(getattr(args, 'roi', None))
    if roi is None:
        # Use FaceExtractor's default ROI
        roi = (0.3, 0.1, 0.7, 0.6)
    roi_pct = f"{int(roi[0]*100)}%-{int(roi[2]*100)}% x {int(roi[1]*100)}%-{int(roi[3]*100)}%"
    print(f"ROI: {roi_pct}")

    print("-" * 50)
    print("Controls: [q] quit, [r] reset, [space] pause")
    print("-" * 50)

    # Initialize extractors
    for ext in extractors:
        if ext.name not in ('quality',):  # QualityExtractor doesn't need init
            try:
                ext.initialize()
            except Exception as e:
                print(f"Warning: Failed to initialize {ext.name}: {e}")

    # Initialize face classifier
    if face_classifier:
        face_classifier.initialize()

    # Print backend info in profile mode (after initialization)
    if profile_mode:
        print("\nBackends:")
        for ext in extractors:
            if hasattr(ext, 'get_backend_info'):
                info = ext.get_backend_info()
                for component, backend_name in info.items():
                    print(f"  {component.capitalize():12}: {backend_name}")
        print("-" * 50)

    visualizer = DebugVisualizer()

    # Setup output writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args.output, fourcc, args.fps, (source.width, source.height)
        )

    frame_count = 0
    try:
        for frame in stream:
            # Run extractors
            observations = {}
            for ext in extractors:
                try:
                    obs = ext.extract(frame)
                    if obs:
                        observations[ext.name] = obs
                except Exception as e:
                    pass  # Silent fail for individual extractors

            # Run face classifier if face observation is available
            classifier_obs = None
            if face_classifier:
                face_obs = observations.get("face")
                if face_obs:
                    try:
                        classifier_obs = face_classifier.extract(frame, {"face": face_obs})
                    except Exception as e:
                        pass  # Silent fail

            # Run fusion
            fusion_result = None
            if fusion:
                fusion_obs = observations.get("face") or observations.get("dummy")
                if fusion_obs:
                    fusion_result = fusion.update(fusion_obs, classifier_obs=classifier_obs)

            # Collect timing info for profile mode
            timing_info = None
            if profile_mode:
                face_obs = observations.get("face")
                if face_obs and face_obs.timing:
                    timing_info = face_obs.timing

            # Create debug view
            debug_image = visualizer.create_debug_view(
                frame,
                face_obs=observations.get("face") or observations.get("dummy"),
                pose_obs=observations.get("pose"),
                quality_obs=observations.get("quality"),
                classifier_obs=classifier_obs,
                fusion_result=fusion_result,
                is_gate_open=fusion.is_gate_open if fusion else False,
                in_cooldown=fusion.in_cooldown if fusion else False,
                timing=timing_info if profile_mode else None,
                roi=roi,
            )

            if writer:
                writer.write(debug_image)

            if show_window:
                cv2.imshow("Debug", debug_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r") and fusion:
                    fusion.reset()
                    visualizer.reset()
                elif key == ord(" "):
                    cv2.waitKey(0)

            frame_count += 1

            # Print timing info in profile mode
            if profile_mode and timing_info:
                detect_ms = timing_info.get('detect_ms', 0)
                expr_ms = timing_info.get('expression_ms', 0)
                total_ms = timing_info.get('total_ms', 0)
                print(f"\rFrame {frame.frame_id}: detect={detect_ms:.1f}ms, expression={expr_ms:.1f}ms, total={total_ms:.1f}ms    ", end="", flush=True)
            elif frame_count % 100 == 0:
                print(f"\rFrame {frame_count}/{source.frame_count}", end="", flush=True)

        print()

    finally:
        vb.disconnect()
        if writer:
            writer.release()
            print(f"Saved: {args.output}")
        if show_window:
            cv2.destroyAllWindows()

        for ext in extractors:
            if hasattr(ext, 'cleanup'):
                ext.cleanup()

        if face_classifier:
            face_classifier.cleanup()

        cleanup_observability(hub, file_sink)

    print(f"Processed {frame_count} frames")


def _parse_extractor_arg(arg: str) -> List[str]:
    """Parse extractor argument into list of extractor names."""
    if arg == 'all':
        return ['all']
    if arg in ('raw', 'none'):
        return ['raw']

    # Support comma-separated: "face,pose"
    parts = [p.strip().lower() for p in arg.split(',')]

    valid = {'face', 'pose', 'quality', 'gesture', 'all', 'raw', 'none'}
    for p in parts:
        if p not in valid:
            print(f"Warning: Unknown extractor '{p}', valid: {valid}")

    return [p for p in parts if p in valid]


def _run_raw_preview(args, show_window: bool):
    """Run raw video preview without any analysis.

    Shows video frames directly to verify:
    - Video file opens correctly
    - Frame dimensions and format
    - FPS and total duration
    - Visual quality of source
    """
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

    # Setup output writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args.output, fourcc, args.fps, (source.width, source.height)
        )

    frame_count = 0
    try:
        for frame in stream:
            image = frame.data

            # Add minimal overlay with frame info
            overlay = image.copy()
            info_text = f"Frame: {frame.frame_id} | Time: {frame.t_src_ns / 1e9:.2f}s | {source.width}x{source.height}"
            cv2.putText(
                overlay, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                overlay, "[RAW PREVIEW - No Analysis]", (10, source.height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2
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
    """Parse ROI string to tuple.

    Args:
        roi_str: ROI string in format "x1,y1,x2,y2"

    Returns:
        Tuple (x1, y1, x2, y2) or None if invalid
    """
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
    """Try to load and add an extractor. Returns True if successful."""
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


def _run_distributed_debug(
    args,
    config_path: Optional[str],
    venv_face: Optional[str],
    venv_pose: Optional[str],
    venv_gesture: Optional[str],
    selected: List[str],
    show_window: bool,
):
    """Run debug session in distributed mode using PipelineOrchestrator.

    Uses VenvWorker/InlineWorker for process isolation, visualizing
    the results in real-time.
    """
    import cv2
    import tempfile
    from pathlib import Path
    from facemoment.pipeline import (
        PipelineOrchestrator,
        PipelineConfig,
        ExtractorConfig,
        FusionConfig,
        create_default_config,
    )
    from facemoment.moment_detector.visualize import DebugVisualizer

    # Setup observability
    trace_level = getattr(args, 'trace', 'off')
    trace_output = getattr(args, 'trace_output', None)
    hub, file_sink = setup_observability(trace_level, trace_output)

    # Create temp output dir for clips (debug mode doesn't really need clips)
    temp_clip_dir = tempfile.mkdtemp(prefix="facemoment_debug_")

    print(f"Debug: {args.path}")
    print(f"Mode: DISTRIBUTED")
    print("-" * 50)

    # Load or create config
    if config_path:
        print(f"Loading config from: {config_path}")
        config = PipelineConfig.from_yaml(config_path)
        config.clip_output_dir = temp_clip_dir
        config.fps = int(args.fps)
    else:
        # Filter extractors based on selection
        include_face = 'face' in selected or 'all' in selected
        include_pose = 'pose' in selected or 'all' in selected
        include_gesture = 'gesture' in selected or 'all' in selected
        include_quality = 'quality' in selected or 'all' in selected

        extractors = []

        if include_face:
            extractors.append(ExtractorConfig(
                name="face",
                venv_path=venv_face,
            ))

        if include_pose:
            extractors.append(ExtractorConfig(
                name="pose",
                venv_path=venv_pose,
            ))

        if include_gesture and venv_gesture:
            extractors.append(ExtractorConfig(
                name="gesture",
                venv_path=venv_gesture,
            ))

        if include_quality:
            extractors.append(ExtractorConfig(name="quality"))

        # Fallback to dummy if no extractors
        if not extractors:
            extractors.append(ExtractorConfig(name="dummy"))

        config = PipelineConfig(
            extractors=extractors,
            fusion=FusionConfig(cooldown_sec=2.0),
            clip_output_dir=temp_clip_dir,
            fps=int(args.fps),
        )

    # Print extractor configuration
    print("Extractors:")
    for ext_config in config.extractors:
        isolation = ext_config.effective_isolation.name
        venv = ext_config.venv_path or "(current)"
        print(f"  {ext_config.name}: {isolation} [{venv}]")

    print(f"Fusion: {config.fusion.name} (cooldown={config.fusion.cooldown_sec}s)")
    print("-" * 50)
    print("Controls: [q] quit, [space] pause")
    print("-" * 50)

    # Create orchestrator
    orchestrator = PipelineOrchestrator.from_config(config)

    # Setup visualizer
    visualizer = DebugVisualizer()

    # Setup output writer
    writer = None
    # We'll initialize writer after we get the first frame to know dimensions

    frame_count = 0
    trigger_count = 0
    writer_initialized = False

    try:
        for frame, observations, fusion_result in orchestrator.run_stream(
            str(args.path), fps=int(args.fps)
        ):
            frame_count += 1

            # Initialize writer on first frame
            if args.output and not writer_initialized:
                h, w = frame.data.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
                writer_initialized = True

            # Convert observations list to dict for visualizer
            obs_dict = {}
            for obs in observations:
                obs_dict[obs.source] = obs

            # Check for triggers
            if fusion_result and fusion_result.should_trigger:
                trigger_count += 1
                event_time_sec = fusion_result.trigger.event_time_ns / 1e9 if fusion_result.trigger and fusion_result.trigger.event_time_ns else 0
                print(f"\n  TRIGGER #{trigger_count}: {fusion_result.reason} (score={fusion_result.score:.2f}, t={event_time_sec:.2f}s)")

            # Get fusion state for visualization
            # Note: orchestrator creates internal fusion, we need to track state
            is_gate_open = False
            in_cooldown = False
            if fusion_result:
                # Infer state from result
                in_cooldown = not fusion_result.should_trigger and fusion_result.trigger is None

            # Create debug view
            debug_image = visualizer.create_debug_view(
                frame,
                face_obs=obs_dict.get("face") or obs_dict.get("merged") or obs_dict.get("dummy"),
                pose_obs=obs_dict.get("pose"),
                quality_obs=obs_dict.get("quality"),
                fusion_result=fusion_result,
                is_gate_open=is_gate_open,
                in_cooldown=in_cooldown,
            )

            # Add distributed mode indicator
            cv2.putText(
                debug_image, "[DISTRIBUTED]", (debug_image.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2
            )

            # Show worker info
            worker_names = orchestrator.worker_names
            if worker_names:
                workers_text = f"Workers: {', '.join(worker_names)}"
                cv2.putText(
                    debug_image, workers_text, (debug_image.shape[1] - 250, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
                )

            if writer:
                writer.write(debug_image)

            if show_window:
                cv2.imshow("Debug (Distributed)", debug_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    cv2.waitKey(0)

            if frame_count % 100 == 0:
                print(f"\rFrame {frame_count}...", end="", flush=True)

        print()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if writer:
            writer.release()
            print(f"Saved: {args.output}")
        if show_window:
            cv2.destroyAllWindows()

        cleanup_observability(hub, file_sink)

        # Cleanup temp dir
        import shutil
        try:
            shutil.rmtree(temp_clip_dir)
        except Exception:
            pass

    # Print stats
    stats = orchestrator.get_stats()
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
