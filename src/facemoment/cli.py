"""Command-line interface for portrait981-moment."""

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Portrait981 - Portrait highlight capture"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # visualize command (legacy)
    viz_parser = subparsers.add_parser("visualize", help="Visualize detector on video")
    viz_parser.add_argument("path", help="Path to video file")
    viz_parser.add_argument(
        "--fps", type=int, default=10, help="Analysis FPS (default: 10)"
    )
    viz_parser.add_argument(
        "--faces", type=int, default=2, help="Number of simulated faces (default: 2)"
    )
    viz_parser.add_argument(
        "--threshold", type=float, default=0.7, help="Expression threshold (default: 0.7)"
    )
    viz_parser.add_argument(
        "--spike-prob", type=float, default=0.1, help="Spike probability (default: 0.1)"
    )
    viz_parser.add_argument(
        "--output-dir", "-o", type=str, default="./clips", help="Clip output directory"
    )

    # debug command - NEW: for module debugging with visualization
    debug_parser = subparsers.add_parser(
        "debug", help="Debug extractors/fusion with visualization"
    )
    debug_parser.add_argument("path", help="Path to video file")
    debug_parser.add_argument(
        "--fps", type=float, default=10.0, help="Analysis FPS (default: 10)"
    )
    debug_parser.add_argument(
        "--ml", action="store_true", dest="use_ml", default=None,
        help="Force ML backends (auto-detected by default)"
    )
    debug_parser.add_argument(
        "--no-ml", action="store_false", dest="use_ml",
        help="Force dummy backends (for testing without ML)"
    )
    debug_parser.add_argument(
        "--no-window", action="store_true", help="Disable interactive window"
    )
    debug_parser.add_argument(
        "--output", "-o", type=str, help="Save debug video to file"
    )
    debug_parser.add_argument(
        "--extractor",
        "-e",
        choices=["face", "pose", "quality", "all"],
        default="all",
        help="Which extractor to debug (default: all)"
    )

    # debug-face command - specific face extractor debugging
    face_debug_parser = subparsers.add_parser(
        "debug-face", help="Debug face extractor specifically"
    )
    face_debug_parser.add_argument("path", help="Path to video file or image")
    face_debug_parser.add_argument(
        "--fps", type=float, default=10.0, help="Analysis FPS (default: 10)"
    )
    face_debug_parser.add_argument(
        "--output", "-o", type=str, help="Save debug output (required if --no-window)"
    )
    face_debug_parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device (default: cuda:0)"
    )
    face_debug_parser.add_argument(
        "--no-window", action="store_true", help="Disable GUI window (headless mode)"
    )

    # debug-pose command - specific pose extractor debugging
    pose_debug_parser = subparsers.add_parser(
        "debug-pose", help="Debug pose extractor specifically"
    )
    pose_debug_parser.add_argument("path", help="Path to video file or image")
    pose_debug_parser.add_argument(
        "--fps", type=float, default=10.0, help="Analysis FPS (default: 10)"
    )
    pose_debug_parser.add_argument(
        "--output", "-o", type=str, help="Save debug output (required if --no-window)"
    )
    pose_debug_parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device (default: cuda:0)"
    )
    pose_debug_parser.add_argument(
        "--no-window", action="store_true", help="Disable GUI window (headless mode)"
    )

    # debug-quality command - specific quality extractor debugging
    quality_debug_parser = subparsers.add_parser(
        "debug-quality", help="Debug quality extractor specifically"
    )
    quality_debug_parser.add_argument("path", help="Path to video file or image")
    quality_debug_parser.add_argument(
        "--fps", type=float, default=10.0, help="Analysis FPS (default: 10)"
    )
    quality_debug_parser.add_argument(
        "--output", "-o", type=str, help="Save debug output (required if --no-window)"
    )
    quality_debug_parser.add_argument(
        "--no-window", action="store_true", help="Disable GUI window (headless mode)"
    )

    # benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark", help="Benchmark extractor performance"
    )
    bench_parser.add_argument("path", help="Path to video file")
    bench_parser.add_argument(
        "--frames", type=int, default=100, help="Number of frames to benchmark (default: 100)"
    )
    bench_parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device (default: cuda:0)"
    )
    bench_parser.add_argument(
        "--skip-pose", action="store_true", help="Skip pose extractor benchmark"
    )
    bench_parser.add_argument(
        "--expression-backend",
        choices=["auto", "hsemotion", "pyfeat", "none"],
        default="auto",
        help="Expression backend to benchmark (default: auto)"
    )

    # process command
    proc_parser = subparsers.add_parser("process", help="Process video and extract clips")
    proc_parser.add_argument("path", help="Path to video file")
    proc_parser.add_argument(
        "--fps", type=int, default=10, help="Analysis FPS (default: 10)"
    )
    proc_parser.add_argument(
        "--output-dir", "-o", type=str, default="./clips", help="Clip output directory"
    )
    proc_parser.add_argument(
        "--ml", action="store_true", dest="use_ml", default=None,
        help="Force ML backends (auto-detected by default)"
    )
    proc_parser.add_argument(
        "--no-ml", action="store_false", dest="use_ml",
        help="Force dummy backends (for testing)"
    )
    proc_parser.add_argument(
        "--report", type=str, help="Save processing report to JSON file"
    )
    proc_parser.add_argument(
        "--cooldown", type=float, default=2.0, help="Cooldown between triggers in seconds (default: 2.0)"
    )
    proc_parser.add_argument(
        "--head-turn-threshold", type=float, default=30.0,
        help="Head turn velocity threshold in deg/sec (default: 30.0, lower = more sensitive)"
    )
    # Legacy dummy mode options (hidden)
    proc_parser.add_argument("--faces", type=int, default=2, help=argparse.SUPPRESS)
    proc_parser.add_argument("--threshold", type=float, default=0.7, help=argparse.SUPPRESS)
    proc_parser.add_argument("--spike-prob", type=float, default=0.1, help=argparse.SUPPRESS)

    # extractor command (Phase 8: A-B*-C architecture)
    extractor_parser = subparsers.add_parser(
        "extractor",
        help="Run extractor process (B module) for A-B*-C architecture"
    )
    extractor_parser.add_argument(
        "type",
        choices=["face", "pose", "quality"],
        help="Extractor type to run"
    )
    extractor_parser.add_argument(
        "--input", type=str, required=True,
        help="Input FIFO path for receiving frames"
    )
    extractor_parser.add_argument(
        "--obs-socket", type=str, default="/tmp/obs.sock",
        help="UDS socket path for sending OBS messages"
    )
    extractor_parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device for ML inference (default: cuda:0)"
    )
    extractor_parser.add_argument(
        "--no-reconnect", action="store_true",
        help="Disable auto-reconnection on FIFO disconnect"
    )

    # fusion command (Phase 8: A-B*-C architecture)
    fusion_parser = subparsers.add_parser(
        "fusion",
        help="Run fusion process (C module) for A-B*-C architecture"
    )
    fusion_parser.add_argument(
        "--obs-socket", type=str, default="/tmp/obs.sock",
        help="UDS socket path for receiving OBS messages"
    )
    fusion_parser.add_argument(
        "--trig-socket", type=str, default="/tmp/trig.sock",
        help="UDS socket path for sending TRIG messages"
    )
    fusion_parser.add_argument(
        "--cooldown", type=float, default=2.0,
        help="Cooldown between triggers in seconds (default: 2.0)"
    )

    args = parser.parse_args()

    # Set up logging
    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.command == "visualize":
        from facemoment.tools import visualize

        visualize(
            args.path,
            fps=args.fps,
            num_faces=args.faces,
            expression_threshold=args.threshold,
            spike_probability=args.spike_prob,
            clip_output_dir=args.output_dir,
        )

    elif args.command == "debug":
        from facemoment.moment_detector.visualize import run_debug_session

        ml_mode = "auto" if args.use_ml is None else ("enabled" if args.use_ml else "disabled")
        print(f"Debug session: {args.path}")
        print(f"FPS: {args.fps}")
        print(f"ML backends: {ml_mode}")
        print("-" * 40)
        print("Controls: [q] quit, [r] reset, [space] pause")
        print("-" * 40)

        run_debug_session(
            args.path,
            output_path=args.output,
            fps=args.fps,
            use_ml_backends=args.use_ml,
            show_window=not args.no_window,
        )

    elif args.command == "debug-face":
        _run_face_debug(args)

    elif args.command == "debug-pose":
        _run_pose_debug(args)

    elif args.command == "debug-quality":
        _run_quality_debug(args)

    elif args.command == "benchmark":
        _run_benchmark(args)

    elif args.command == "process":
        _run_process(args)

    elif args.command == "extractor":
        _run_extractor(args)

    elif args.command == "fusion":
        _run_fusion(args)

    else:
        parser.print_help()
        sys.exit(1)


def _check_ml_dependencies(module_name: str, require_expression: bool = False) -> bool:
    """Check if ML dependencies are available."""
    deps = {
        "face": ["insightface", "onnxruntime"],
        "face_expression": ["insightface", "onnxruntime", "feat"],
        "pose": ["ultralytics"],
    }

    if module_name == "face" and require_expression:
        module_name = "face_expression"
    missing = []
    for dep in deps.get(module_name, []):
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)

    if missing:
        print(f"Error: ML dependencies not installed for {module_name} extractor.")
        print(f"Missing: {', '.join(missing)}")
        print()
        print("Install with:")
        print("  uv sync --extra ml")
        print()
        print("Or install individually:")
        print(f"  pip install {' '.join(missing)}")
        return False
    return True


def _run_face_debug(args):
    """Run face extractor debug session."""
    if not _check_ml_dependencies("face"):
        sys.exit(1)

    show_window = not getattr(args, "no_window", False)
    if not show_window and not args.output:
        print("Error: --output is required when using --no-window")
        sys.exit(1)

    import cv2
    import numpy as np
    from visualbase import Frame
    from facemoment.moment_detector.extractors import FaceExtractor
    from facemoment.moment_detector.visualize import ExtractorVisualizer

    print(f"Face Debug: {args.path}")
    print(f"Device: {args.device}")
    print(f"Window: {'enabled' if show_window else 'disabled'}")
    print("-" * 40)

    extractor = FaceExtractor(device=args.device)
    extractor.initialize()
    visualizer = ExtractorVisualizer()

    cap = cv2.VideoCapture(args.path)
    if not cap.isOpened():
        print(f"Error: Cannot open {args.path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, int(video_fps / args.fps))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))

    frame_id = 0
    try:
        while True:
            ret, image = cap.read()
            if not ret:
                break

            if frame_id % frame_skip != 0:
                frame_id += 1
                continue

            t_ns = int(frame_id / video_fps * 1e9)
            frame = Frame.from_array(image, frame_id=frame_id, t_src_ns=t_ns)

            obs = extractor.extract(frame)
            if obs:
                debug_image = visualizer.draw_face_observation(image, obs)
                # Print progress
                face_count = len(obs.faces)
                max_expr = obs.signals.get("max_expression", 0)
                print(f"\rFrame {frame_id}/{total_frames}: {face_count} faces, expr={max_expr:.2f}", end="")
            else:
                debug_image = image

            if writer:
                writer.write(debug_image)

            if show_window:
                cv2.imshow("Face Debug", debug_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    cv2.waitKey(0)

            frame_id += 1

        print()  # New line after progress

    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"Saved: {args.output}")
        if show_window:
            cv2.destroyAllWindows()
        extractor.cleanup()


def _run_pose_debug(args):
    """Run pose extractor debug session."""
    if not _check_ml_dependencies("pose"):
        sys.exit(1)

    show_window = not getattr(args, "no_window", False)
    if not show_window and not args.output:
        print("Error: --output is required when using --no-window")
        sys.exit(1)

    import cv2
    import numpy as np
    from visualbase import Frame
    from facemoment.moment_detector.extractors import PoseExtractor
    from facemoment.moment_detector.visualize import ExtractorVisualizer

    print(f"Pose Debug: {args.path}")
    print(f"Device: {args.device}")
    print(f"Window: {'enabled' if show_window else 'disabled'}")
    print("-" * 40)

    extractor = PoseExtractor(device=args.device)
    extractor.initialize()
    visualizer = ExtractorVisualizer()

    cap = cv2.VideoCapture(args.path)
    if not cap.isOpened():
        print(f"Error: Cannot open {args.path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, int(video_fps / args.fps))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))

    frame_id = 0
    try:
        while True:
            ret, image = cap.read()
            if not ret:
                break

            if frame_id % frame_skip != 0:
                frame_id += 1
                continue

            t_ns = int(frame_id / video_fps * 1e9)
            frame = Frame.from_array(image, frame_id=frame_id, t_src_ns=t_ns)

            obs = extractor.extract(frame)
            if obs:
                debug_image = visualizer.draw_pose_observation(image, obs)
                # Print progress
                person_count = int(obs.signals.get("person_count", 0))
                hands_up = int(obs.signals.get("hands_raised_count", 0))
                wave = obs.signals.get("hand_wave_detected", 0) > 0.5
                wave_str = " WAVE!" if wave else ""
                print(f"\rFrame {frame_id}/{total_frames}: {person_count} persons, {hands_up} hands up{wave_str}", end="")
            else:
                debug_image = image

            if writer:
                writer.write(debug_image)

            if show_window:
                cv2.imshow("Pose Debug", debug_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    cv2.waitKey(0)

            frame_id += 1

        print()  # New line after progress

    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"Saved: {args.output}")
        if show_window:
            cv2.destroyAllWindows()
        extractor.cleanup()


def _run_quality_debug(args):
    """Run quality extractor debug session."""
    show_window = not getattr(args, "no_window", False)
    if not show_window and not args.output:
        print("Error: --output is required when using --no-window")
        sys.exit(1)

    import cv2
    import numpy as np
    from visualbase import Frame
    from facemoment.moment_detector.extractors import QualityExtractor
    from facemoment.moment_detector.visualize import ExtractorVisualizer

    print(f"Quality Debug: {args.path}")
    print(f"Window: {'enabled' if show_window else 'disabled'}")
    print("-" * 40)

    extractor = QualityExtractor()
    visualizer = ExtractorVisualizer()

    cap = cv2.VideoCapture(args.path)
    if not cap.isOpened():
        print(f"Error: Cannot open {args.path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, int(video_fps / args.fps))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))

    frame_id = 0
    try:
        while True:
            ret, image = cap.read()
            if not ret:
                break

            if frame_id % frame_skip != 0:
                frame_id += 1
                continue

            t_ns = int(frame_id / video_fps * 1e9)
            frame = Frame.from_array(image, frame_id=frame_id, t_src_ns=t_ns)

            obs = extractor.extract(frame)
            if obs:
                debug_image = visualizer.draw_quality_observation(image, obs)
                # Print progress
                blur = obs.signals.get("blur_score", 0)
                bright = obs.signals.get("brightness", 0)
                gate = "OPEN" if obs.signals.get("quality_gate", 0) > 0.5 else "CLOSED"
                print(f"\rFrame {frame_id}/{total_frames}: blur={blur:.0f} bright={bright:.0f} gate={gate}", end="")
            else:
                debug_image = image

            if writer:
                writer.write(debug_image)

            if show_window:
                cv2.imshow("Quality Debug", debug_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    cv2.waitKey(0)

            frame_id += 1

        print()  # New line after progress

    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"Saved: {args.output}")
        if show_window:
            cv2.destroyAllWindows()


def _run_benchmark(args):
    """Run extractor performance benchmark."""
    import time
    import cv2
    import numpy as np
    from visualbase import Frame

    print(f"Benchmark: {args.path}")
    print(f"Frames: {args.frames}")
    print(f"Device: {args.device}")
    print("-" * 50)

    # Open video
    cap = cv2.VideoCapture(args.path)
    if not cap.isOpened():
        print(f"Error: Cannot open {args.path}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load frames into memory for consistent benchmarking
    print(f"Loading {args.frames} frames into memory...")
    frames = []
    frame_id = 0
    while len(frames) < args.frames:
        ret, image = cap.read()
        if not ret:
            break
        t_ns = int(frame_id / video_fps * 1e9)
        frames.append(Frame.from_array(image, frame_id=frame_id, t_src_ns=t_ns))
        frame_id += 1
    cap.release()

    if len(frames) < args.frames:
        print(f"Warning: Only {len(frames)} frames available")

    print(f"Loaded {len(frames)} frames")
    print("-" * 50)

    results = {}

    # Benchmark Face Detection
    print("\n[Face Detection - InsightFace SCRFD]")
    try:
        from facemoment.moment_detector.extractors.backends.face_backends import InsightFaceSCRFD

        face_backend = InsightFaceSCRFD()
        face_backend.initialize(args.device)

        # Warm up
        for f in frames[:5]:
            face_backend.detect(f.data)

        times = []
        all_faces = []
        for f in frames:
            start = time.perf_counter()
            faces = face_backend.detect(f.data)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            all_faces.append((f, faces))

        avg = sum(times) / len(times)
        results["face_detection"] = avg
        print(f"  Average: {avg:.1f}ms ({len(frames)} frames)")
        print(f"  Min/Max: {min(times):.1f}ms / {max(times):.1f}ms")
        face_backend.cleanup()

    except Exception as e:
        print(f"  Error: {e}")
        all_faces = [(f, []) for f in frames]

    # Benchmark Expression (HSEmotion or PyFeat)
    expression_backend_name = args.expression_backend
    if expression_backend_name == "auto":
        # Try HSEmotion first, then PyFeat
        try:
            from facemoment.moment_detector.extractors.backends.face_backends import HSEmotionBackend
            expression_backend_name = "hsemotion"
        except ImportError:
            expression_backend_name = "pyfeat"

    if expression_backend_name == "hsemotion":
        print("\n[Expression - HSEmotion (fast)]")
        try:
            from facemoment.moment_detector.extractors.backends.face_backends import HSEmotionBackend

            expr_backend = HSEmotionBackend()
            expr_backend.initialize(args.device)

            # Warm up
            for f, faces in all_faces[:5]:
                if faces:
                    expr_backend.analyze(f.data, faces)

            times = []
            faces_analyzed = 0
            for f, faces in all_faces:
                if faces:
                    start = time.perf_counter()
                    expr_backend.analyze(f.data, faces)
                    elapsed = (time.perf_counter() - start) * 1000
                    times.append(elapsed)
                    faces_analyzed += len(faces)

            if times:
                avg = sum(times) / len(times)
                results["expression_hsemotion"] = avg
                print(f"  Average: {avg:.1f}ms ({len(times)} frames with faces)")
                print(f"  Min/Max: {min(times):.1f}ms / {max(times):.1f}ms")
                print(f"  Faces analyzed: {faces_analyzed}")
            else:
                print("  No faces detected to analyze")
            expr_backend.cleanup()

        except Exception as e:
            print(f"  Error: {e}")

    elif expression_backend_name == "pyfeat":
        print("\n[Expression - PyFeat (accurate, slow)]")
        try:
            from facemoment.moment_detector.extractors.backends.face_backends import PyFeatBackend

            expr_backend = PyFeatBackend()
            expr_backend.initialize(args.device)

            # Warm up (only 1 frame for PyFeat since it's slow)
            for f, faces in all_faces[:1]:
                if faces:
                    expr_backend.analyze(f.data, faces)

            # Only benchmark a subset for PyFeat (it's very slow)
            subset = [(f, faces) for f, faces in all_faces if faces][:10]
            times = []
            faces_analyzed = 0
            for f, faces in subset:
                start = time.perf_counter()
                expr_backend.analyze(f.data, faces)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
                faces_analyzed += len(faces)

            if times:
                avg = sum(times) / len(times)
                results["expression_pyfeat"] = avg
                print(f"  Average: {avg:.1f}ms ({len(times)} frames - limited due to speed)")
                print(f"  Min/Max: {min(times):.1f}ms / {max(times):.1f}ms")
                print(f"  Faces analyzed: {faces_analyzed}")
            else:
                print("  No faces detected to analyze")
            expr_backend.cleanup()

        except Exception as e:
            print(f"  Error: {e}")

    elif expression_backend_name != "none":
        print(f"\n[Expression - {expression_backend_name}]")
        print("  Skipped (backend not available)")

    # Benchmark Pose
    if not args.skip_pose:
        print("\n[Pose - YOLO-Pose]")
        try:
            from facemoment.moment_detector.extractors.backends.pose_backends import YOLOPoseBackend

            pose_backend = YOLOPoseBackend()
            pose_backend.initialize(args.device)

            # Warm up
            for f in frames[:5]:
                pose_backend.detect(f.data)

            times = []
            for f in frames:
                start = time.perf_counter()
                pose_backend.detect(f.data)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

            avg = sum(times) / len(times)
            results["pose"] = avg
            print(f"  Average: {avg:.1f}ms ({len(frames)} frames)")
            print(f"  Min/Max: {min(times):.1f}ms / {max(times):.1f}ms")
            pose_backend.cleanup()

        except Exception as e:
            print(f"  Error: {e}")

    # Benchmark Quality
    print("\n[Quality - Blur/Brightness]")
    try:
        from facemoment.moment_detector.extractors import QualityExtractor

        quality_ext = QualityExtractor()

        # Warm up
        for f in frames[:5]:
            quality_ext.extract(f)

        times = []
        for f in frames:
            start = time.perf_counter()
            quality_ext.extract(f)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg = sum(times) / len(times)
        results["quality"] = avg
        print(f"  Average: {avg:.1f}ms ({len(frames)} frames)")
        print(f"  Min/Max: {min(times):.1f}ms / {max(times):.1f}ms")

    except Exception as e:
        print(f"  Error: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    total_ms = sum(results.values())
    fps = 1000 / total_ms if total_ms > 0 else 0

    for name, ms in results.items():
        print(f"  {name:25s}: {ms:7.1f}ms")
    print("-" * 50)
    print(f"  {'Total':25s}: {total_ms:7.1f}ms")
    print(f"  {'Estimated FPS':25s}: {fps:7.1f} fps")


def _run_process(args):
    """Run video processing and clip extraction."""
    import json
    import time
    from pathlib import Path
    from datetime import datetime

    from facemoment import MomentDetector
    from facemoment.moment_detector.extractors import DummyExtractor, QualityExtractor
    from facemoment.moment_detector.fusion import DummyFusion

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up extractors based on ML availability
    extractors = []
    fusion = None
    use_ml = args.use_ml
    ml_mode = "auto" if use_ml is None else ("enabled" if use_ml else "disabled")

    print(f"Processing: {args.path}")
    print(f"FPS: {args.fps}")
    print(f"Output: {output_dir}")
    print(f"ML backends: {ml_mode}")
    print("-" * 50)

    # Try to load and initialize ML extractors
    face_available = False
    pose_available = False
    if use_ml or use_ml is None:
        try:
            from facemoment.moment_detector.extractors import FaceExtractor
            face_ext = FaceExtractor()
            face_ext.initialize()  # Initialize now to catch errors early
            extractors.append(face_ext)
            face_available = True
            print("  FaceExtractor: enabled")
        except Exception as e:
            if use_ml:
                print(f"Error: FaceExtractor not available: {e}")
                sys.exit(1)
            print(f"  FaceExtractor: disabled ({type(e).__name__})")

        try:
            from facemoment.moment_detector.extractors import PoseExtractor
            pose_ext = PoseExtractor()
            pose_ext.initialize()  # Initialize now to catch errors early
            extractors.append(pose_ext)
            pose_available = True
            print("  PoseExtractor: enabled")
        except Exception as e:
            if use_ml:
                print(f"Error: PoseExtractor not available: {e}")
                sys.exit(1)
            print(f"  PoseExtractor: disabled ({type(e).__name__})")

    # Add QualityExtractor (always available)
    extractors.append(QualityExtractor())
    print("  QualityExtractor: enabled")

    # Fall back to dummy if no ML extractors
    if not face_available:
        extractors.insert(0, DummyExtractor(
            num_faces=args.faces,
            spike_probability=args.spike_prob,
        ))
        print("  DummyExtractor: enabled (fallback)")

    # Set up fusion
    if face_available:
        from facemoment.moment_detector.fusion import HighlightFusion
        fusion = HighlightFusion(
            cooldown_sec=args.cooldown,
            head_turn_velocity_threshold=args.head_turn_threshold,
        )
        print(f"  HighlightFusion: enabled (cooldown={args.cooldown}s, head_turn_threshold={args.head_turn_threshold}°/s)")
    else:
        fusion = DummyFusion(
            expression_threshold=args.threshold,
            consecutive_frames=3,
            cooldown_sec=args.cooldown,
        )
        print(f"  DummyFusion: enabled (threshold={args.threshold})")

    # Show available triggers
    print()
    print("Available triggers:")
    if face_available:
        # Check if expression backend is available by trying to get first extractor
        face_ext = extractors[0] if extractors and extractors[0].name == "face" else None
        has_expression = face_ext and getattr(face_ext, '_expression_backend', None) is not None
        print(f"  - expression_spike: {'enabled' if has_expression else 'DISABLED (install py-feat)'}")
        print(f"  - head_turn: enabled")
    if pose_available:
        print(f"  - hand_wave: enabled")
    if not face_available and not pose_available:
        print(f"  - dummy triggers only")

    print("-" * 50)

    # Create detector
    detector = MomentDetector(
        extractors=extractors,
        fusion=fusion,
        clip_output_dir=output_dir,
    )

    # Track metadata for each clip
    clip_metadata = []
    start_time = time.time()

    def on_trigger(trigger, result):
        """Callback to collect metadata when trigger fires."""
        meta = {
            "trigger_id": len(clip_metadata) + 1,
            "reason": result.reason,
            "score": result.score,
            "frame_id": trigger.frame_id,
            "timestamp_sec": trigger.t_ns / 1e9 if hasattr(trigger, 't_ns') else 0,
            "metadata": result.metadata,
        }
        clip_metadata.append(meta)
        print(f"\n  TRIGGER #{meta['trigger_id']}: {result.reason} (score={result.score:.2f}, frame={trigger.frame_id})")

    detector.set_on_trigger(on_trigger)

    # Progress tracking
    last_progress = [0]
    def on_frame(frame):
        """Show progress every 100 frames."""
        if frame.frame_id % 100 == 0 and frame.frame_id > last_progress[0]:
            last_progress[0] = frame.frame_id
            print(f"\r  Processing frame {frame.frame_id}...", end="", flush=True)

    detector.set_on_frame(on_frame)

    # Process video
    print("Processing video...")
    clips = detector.process_file(args.path, fps=args.fps)
    print()  # New line after progress

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"Processing complete in {elapsed:.1f}s")
    print(f"  Frames processed: {detector.frames_processed}")
    print(f"  Triggers fired: {detector.triggers_fired}")
    print(f"  Clips extracted: {len(clips)}")
    print()

    # Save metadata for each clip
    for i, clip in enumerate(clips):
        if clip.success and clip.output_path:
            clip_path = Path(clip.output_path)
            meta_path = clip_path.with_suffix(".json")

            # Get corresponding trigger metadata
            trigger_meta = clip_metadata[i] if i < len(clip_metadata) else {}

            full_meta = {
                "clip_id": clip_path.stem,
                "video_source": str(Path(args.path).resolve()),
                "created_at": datetime.now().isoformat(),
                "trigger": trigger_meta,
                "clip": {
                    "output_path": str(clip_path),
                    "duration_sec": clip.duration_sec,
                    "success": clip.success,
                },
            }

            with open(meta_path, "w") as f:
                json.dump(full_meta, f, indent=2, default=str)

            print(f"  [{i+1}] {clip_path.name} ({clip.duration_sec:.2f}s)")
            print(f"      Reason: {trigger_meta.get('reason', 'unknown')}")
        else:
            print(f"  [{i+1}] FAILED: {clip.error}")

    # Save processing report if requested
    if args.report:
        report = {
            "video_source": str(Path(args.path).resolve()),
            "processed_at": datetime.now().isoformat(),
            "settings": {
                "fps": args.fps,
                "cooldown_sec": args.cooldown,
                "ml_mode": ml_mode,
            },
            "results": {
                "frames_processed": detector.frames_processed,
                "triggers_fired": detector.triggers_fired,
                "clips_extracted": len([c for c in clips if c.success]),
                "processing_time_sec": elapsed,
            },
            "clips": [
                {
                    "clip_id": Path(c.output_path).stem if c.output_path else None,
                    "success": c.success,
                    "duration_sec": c.duration_sec,
                    "error": c.error,
                    "trigger": clip_metadata[i] if i < len(clip_metadata) else None,
                }
                for i, c in enumerate(clips)
            ],
        }

        report_path = Path(args.report)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print()
        print(f"Report saved: {report_path}")


def _run_extractor(args):
    """Run extractor process (B module) for A-B*-C architecture."""
    from facemoment.process.extractor import ExtractorProcess

    print(f"Extractor process: {args.type}")
    print(f"  Input FIFO: {args.input}")
    print(f"  OBS socket: {args.obs_socket}")
    print(f"  Device: {args.device}")
    print(f"  Reconnect: {not args.no_reconnect}")
    print("-" * 50)

    # Create extractor based on type
    if args.type == "face":
        if not _check_ml_dependencies("face"):
            sys.exit(1)
        from facemoment.moment_detector.extractors import FaceExtractor
        extractor = FaceExtractor(device=args.device)
        print("Using FaceExtractor (InsightFace + HSEmotion/PyFeat)")

    elif args.type == "pose":
        if not _check_ml_dependencies("pose"):
            sys.exit(1)
        from facemoment.moment_detector.extractors import PoseExtractor
        extractor = PoseExtractor(device=args.device)
        print("Using PoseExtractor (YOLO-Pose)")

    elif args.type == "quality":
        from facemoment.moment_detector.extractors import QualityExtractor
        extractor = QualityExtractor()
        print("Using QualityExtractor (blur/brightness)")

    else:
        print(f"Error: Unknown extractor type: {args.type}")
        sys.exit(1)

    # Create and run process
    process = ExtractorProcess(
        extractor=extractor,
        input_fifo=args.input,
        obs_socket=args.obs_socket,
        reconnect=not args.no_reconnect,
    )

    try:
        process.run()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        stats = process.get_stats()
        print(f"\nExtractor process stats:")
        print(f"  Frames: {stats['frames_processed']}")
        print(f"  OBS sent: {stats['obs_sent']}")
        print(f"  FPS: {stats['fps']:.1f}")
        print(f"  Errors: {stats['errors']}")


def _run_fusion(args):
    """Run fusion process (C module) for A-B*-C architecture."""
    from facemoment.process.fusion import FusionProcess
    from facemoment.moment_detector.fusion import HighlightFusion

    print(f"Fusion process")
    print(f"  OBS socket: {args.obs_socket}")
    print(f"  TRIG socket: {args.trig_socket}")
    print(f"  Cooldown: {args.cooldown}s")
    print("-" * 50)

    # Create fusion engine
    fusion = HighlightFusion(cooldown_sec=args.cooldown)
    print("Using HighlightFusion")

    # Create and run process
    process = FusionProcess(
        fusion=fusion,
        obs_socket=args.obs_socket,
        trig_socket=args.trig_socket,
    )

    try:
        process.run()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        stats = process.get_stats()
        print(f"\nFusion process stats:")
        print(f"  OBS received: {stats['obs_received']}")
        print(f"  Triggers sent: {stats['triggers_sent']}")
        print(f"  Errors: {stats['errors']}")


if __name__ == "__main__":
    main()
