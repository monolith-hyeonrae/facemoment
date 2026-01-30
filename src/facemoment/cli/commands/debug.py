"""Debug command for facemoment CLI.

Unified debug command that replaces separate debug-face, debug-pose, etc.
"""

import sys
from typing import List, Optional

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

    from facemoment.moment_detector.extractors import QualityExtractor
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

            # Run fusion
            fusion_result = None
            if fusion:
                fusion_obs = observations.get("face") or observations.get("dummy")
                if fusion_obs:
                    fusion_result = fusion.update(fusion_obs)

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
                fusion_result=fusion_result,
                is_gate_open=fusion.is_gate_open if fusion else False,
                in_cooldown=fusion.in_cooldown if fusion else False,
                timing=timing_info if profile_mode else None,
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


def _try_load_extractor(name: str, extractors: list, args) -> bool:
    """Try to load and add an extractor. Returns True if successful."""
    device = getattr(args, 'device', 'cuda:0')

    try:
        if name == 'face':
            if not check_ml_dependencies("face"):
                return False
            from facemoment.moment_detector.extractors import FaceExtractor
            extractors.append(FaceExtractor(device=device))
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
