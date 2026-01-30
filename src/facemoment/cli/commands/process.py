"""Process command for facemoment CLI."""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

from facemoment.cli.utils import setup_observability, cleanup_observability


def run_process(args):
    """Run video processing and clip extraction."""
    from facemoment import MomentDetector
    from facemoment.moment_detector.extractors import DummyExtractor, QualityExtractor
    from facemoment.moment_detector.fusion import DummyFusion

    # Setup observability
    trace_level = getattr(args, 'trace', 'off')
    trace_output = getattr(args, 'trace_output', None)
    hub, file_sink = setup_observability(trace_level, trace_output)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up extractors based on ML availability
    extractors = []
    fusion = None
    use_ml = args.use_ml
    ml_mode = "auto" if use_ml is None else ("enabled" if use_ml else "disabled")
    gokart_mode = getattr(args, 'gokart', False)

    print(f"Processing: {args.path}")
    print(f"FPS: {args.fps}")
    print(f"Output: {output_dir}")
    print(f"ML backends: {ml_mode}")
    if gokart_mode:
        print(f"Mode: GOKART (Phase 9)")
    print("-" * 50)

    # Try to load and initialize ML extractors
    face_available = False
    pose_available = False
    gesture_available = False

    if use_ml or use_ml is None:
        try:
            from facemoment.moment_detector.extractors import FaceExtractor
            face_ext = FaceExtractor()
            face_ext.initialize()
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
            pose_ext.initialize()
            extractors.append(pose_ext)
            pose_available = True
            print("  PoseExtractor: enabled")
        except Exception as e:
            if use_ml:
                print(f"Error: PoseExtractor not available: {e}")
                sys.exit(1)
            print(f"  PoseExtractor: disabled ({type(e).__name__})")

        # Add GestureExtractor if gokart mode is enabled
        if gokart_mode:
            try:
                from facemoment.moment_detector.extractors import GestureExtractor
                gesture_ext = GestureExtractor()
                gesture_ext.initialize()
                extractors.append(gesture_ext)
                gesture_available = True
                print("  GestureExtractor: enabled (gokart)")
            except Exception as e:
                print(f"  GestureExtractor: disabled ({type(e).__name__})")

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
        if gokart_mode:
            fusion = HighlightFusion(
                cooldown_sec=args.cooldown,
                head_turn_velocity_threshold=args.head_turn_threshold,
                gaze_yaw_threshold=10.0,
                gaze_pitch_threshold=15.0,
                gaze_score_threshold=0.5,
                interaction_yaw_threshold=15.0,
            )
            print(f"  HighlightFusion: enabled (gokart mode, cooldown={args.cooldown}s)")
        else:
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
        face_ext = extractors[0] if extractors and extractors[0].name == "face" else None
        has_expression = face_ext and getattr(face_ext, '_expression_backend', None) is not None
        print(f"  - expression_spike: {'enabled' if has_expression else 'DISABLED (install py-feat)'}")
        print(f"  - head_turn: enabled")
        if gokart_mode:
            print(f"  - camera_gaze: enabled (gokart)")
            print(f"  - passenger_interaction: enabled (gokart)")
    if pose_available:
        print(f"  - hand_wave: enabled")
    if gesture_available:
        print(f"  - gesture_vsign: enabled (gokart)")
        print(f"  - gesture_thumbsup: enabled (gokart)")
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
        if frame.frame_id % 100 == 0 and frame.frame_id > last_progress[0]:
            last_progress[0] = frame.frame_id
            print(f"\r  Processing frame {frame.frame_id}...", end="", flush=True)

    detector.set_on_frame(on_frame)

    # Process video
    print("Processing video...")
    clips = detector.process_file(args.path, fps=args.fps)
    print()

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

    cleanup_observability(hub, file_sink)
