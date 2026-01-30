"""Benchmark command for facemoment CLI."""

import sys
import time


def run_benchmark(args):
    """Run extractor performance benchmark."""
    import cv2
    from visualbase import Frame

    print(f"Benchmark: {args.path}")
    print(f"Frames: {args.frames}")
    print(f"Device: {args.device}")
    print("-" * 50)

    # Load frames into memory
    frames = []
    try:
        from visualbase import FileSource

        source = FileSource(args.path)
        source.open()

        if not hasattr(source, 'read') or not hasattr(source, 'close'):
            raise AttributeError("FileSource missing required methods")

        print(f"Loading {args.frames} frames into memory...")
        while len(frames) < args.frames:
            frame = source.read()
            if frame is None:
                break
            frames.append(frame)
        source.close()

    except (ImportError, AttributeError, TypeError) as e:
        import logging
        logging.warning(f"visualbase API incompatible, using cv2 fallback: {e}")

        cap = cv2.VideoCapture(args.path)
        if not cap.isOpened():
            print(f"Error: Cannot open {args.path}")
            sys.exit(1)

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Loading {args.frames} frames into memory...")
        frame_id = 0
        while len(frames) < args.frames:
            ret, image = cap.read()
            if not ret:
                break
            t_ns = int(frame_id / video_fps * 1e9)
            frames.append(Frame.from_array(image, frame_id=frame_id, t_src_ns=t_ns))
            frame_id += 1
        cap.release()

    except IOError:
        print(f"Error: Cannot open {args.path}")
        sys.exit(1)

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

    # Benchmark Expression
    expression_backend_name = args.expression_backend
    if expression_backend_name == "auto":
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

            for f, faces in all_faces[:1]:
                if faces:
                    expr_backend.analyze(f.data, faces)

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
