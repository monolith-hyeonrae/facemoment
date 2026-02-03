"""Process command for facemoment CLI."""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

from facemoment.cli.utils import setup_observability, cleanup_observability


def run_process(args):
    """Run video processing and clip extraction."""
    # Check if distributed mode is requested
    distributed = getattr(args, 'distributed', False)
    config_path = getattr(args, 'config', None)
    venv_face = getattr(args, 'venv_face', None)
    venv_pose = getattr(args, 'venv_pose', None)
    venv_gesture = getattr(args, 'venv_gesture', None)
    backend = getattr(args, 'backend', 'pathway')

    # If any venv path is provided, enable distributed mode
    if venv_face or venv_pose or venv_gesture or config_path:
        distributed = True

    if distributed:
        _run_distributed(args, config_path, venv_face, venv_pose, venv_gesture, backend)
    else:
        _run_library(args, backend)


def _run_distributed(args, config_path, venv_face, venv_pose, venv_gesture, backend="pathway"):
    """Run processing in distributed mode using PipelineOrchestrator."""
    from facemoment.pipeline import (
        PipelineOrchestrator,
        PipelineConfig,
        create_default_config,
    )
    from facemoment.pipeline.pathway_pipeline import PATHWAY_AVAILABLE

    # Setup observability
    trace_level = getattr(args, 'trace', 'off')
    trace_output = getattr(args, 'trace_output', None)
    hub, file_sink = setup_observability(trace_level, trace_output)

    output_dir = Path(args.output_dir)

    # Determine effective backend
    effective_backend = backend if PATHWAY_AVAILABLE or backend == "simple" else "simple"

    print(f"Processing: {args.path}")
    print(f"FPS: {args.fps}")
    print(f"Output: {output_dir}")
    print(f"Mode: DISTRIBUTED")
    print(f"Backend: {effective_backend}" + (" (pathway unavailable, using simple)" if backend == "pathway" and not PATHWAY_AVAILABLE else ""))
    print("-" * 50)

    # Load or create config
    if config_path:
        print(f"Loading config from: {config_path}")
        config = PipelineConfig.from_yaml(config_path)
        # Override output dir and fps from CLI if provided
        config.clip_output_dir = str(output_dir)
        config.fps = args.fps
        config.fusion.cooldown_sec = args.cooldown
    else:
        config = create_default_config(
            venv_face=venv_face,
            venv_pose=venv_pose,
            venv_gesture=venv_gesture,
            clip_output_dir=str(output_dir),
            fps=args.fps,
            cooldown_sec=args.cooldown,
            backend=effective_backend,
        )

    # Print extractor configuration
    print("Extractors:")
    for ext_config in config.extractors:
        isolation = ext_config.effective_isolation.name
        venv = ext_config.venv_path or "(current)"
        print(f"  {ext_config.name}: {isolation} [{venv}]")

    print(f"Fusion: {config.fusion.name} (cooldown={config.fusion.cooldown_sec}s)")
    print("-" * 50)

    # Create orchestrator
    orchestrator = PipelineOrchestrator.from_config(config)

    # Track metadata for each clip
    clip_metadata = []
    start_time = time.time()

    def on_trigger(trigger, result):
        event_time_sec = trigger.event_time_ns / 1e9 if trigger.event_time_ns else 0
        meta = {
            "trigger_id": len(clip_metadata) + 1,
            "reason": result.reason,
            "score": result.score,
            "timestamp_sec": event_time_sec,
            "metadata": result.metadata,
        }
        clip_metadata.append(meta)
        print(f"\n  TRIGGER #{meta['trigger_id']}: {result.reason} (score={result.score:.2f}, t={event_time_sec:.2f}s)")

    orchestrator.set_on_trigger(on_trigger)

    # Progress tracking
    last_progress = [0]
    def on_frame(frame):
        if frame.frame_id % 100 == 0 and frame.frame_id > last_progress[0]:
            last_progress[0] = frame.frame_id
            print(f"\r  Processing frame {frame.frame_id}...", end="", flush=True)

    orchestrator.set_on_frame(on_frame)

    # Process video
    print("Processing video...")
    try:
        clips = orchestrator.run(args.path, fps=args.fps)
    except Exception as e:
        print(f"\nError during processing: {e}")
        cleanup_observability(hub, file_sink)
        sys.exit(1)

    print()

    # Get stats
    stats = orchestrator.get_stats()
    elapsed = stats.processing_time_sec

    print("-" * 50)
    print(f"Processing complete in {elapsed:.1f}s")
    print(f"  Frames processed: {stats.frames_processed}")
    print(f"  Triggers fired: {stats.triggers_fired}")
    print(f"  Clips extracted: {stats.clips_extracted}")
    if stats.avg_frame_time_ms > 0:
        print(f"  Avg frame time: {stats.avg_frame_time_ms:.1f}ms")
    print()

    # Print worker stats
    if stats.worker_stats:
        print("Worker statistics:")
        for name, ws in stats.worker_stats.items():
            if ws["frames"] > 0:
                avg_ms = ws["total_ms"] / ws["frames"]
                print(f"  {name}: {ws['frames']} frames, avg {avg_ms:.1f}ms, errors: {ws['errors']}")

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
                "mode": "distributed",
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
            "mode": "distributed",
            "settings": {
                "fps": args.fps,
                "cooldown_sec": args.cooldown,
                "extractors": [
                    {
                        "name": ext.name,
                        "isolation": ext.effective_isolation.name,
                        "venv_path": ext.venv_path,
                    }
                    for ext in config.extractors
                ],
            },
            "results": {
                "frames_processed": stats.frames_processed,
                "triggers_fired": stats.triggers_fired,
                "clips_extracted": stats.clips_extracted,
                "processing_time_sec": elapsed,
                "avg_frame_time_ms": stats.avg_frame_time_ms,
            },
            "worker_stats": stats.worker_stats,
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


def _run_library(args, backend="pathway"):
    """Run processing in library mode (original implementation)."""
    from facemoment import MomentDetector
    from facemoment.moment_detector.extractors import DummyExtractor, QualityExtractor
    from facemoment.moment_detector.fusion import DummyFusion
    from facemoment.pipeline.pathway_pipeline import PATHWAY_AVAILABLE

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

    # Determine effective backend
    effective_backend = backend if PATHWAY_AVAILABLE or backend == "simple" else "simple"

    print(f"Processing: {args.path}")
    print(f"FPS: {args.fps}")
    print(f"Output: {output_dir}")
    print(f"Mode: LIBRARY")
    print(f"Backend: {effective_backend}" + (" (pathway unavailable, using simple)" if backend == "pathway" and not PATHWAY_AVAILABLE else ""))
    print(f"ML backends: {ml_mode}")
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

        # Try to load GestureExtractor if available
        try:
            from facemoment.moment_detector.extractors import GestureExtractor
            gesture_ext = GestureExtractor()
            gesture_ext.initialize()
            extractors.append(gesture_ext)
            gesture_available = True
            print("  GestureExtractor: enabled")
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
        fusion = HighlightFusion(
            cooldown_sec=args.cooldown,
            head_turn_velocity_threshold=args.head_turn_threshold,
        )
        print(f"  HighlightFusion: enabled (cooldown={args.cooldown}s)")
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
        print(f"  - camera_gaze: enabled")
        print(f"  - passenger_interaction: enabled")
    if pose_available:
        print(f"  - hand_wave: enabled")
    if gesture_available:
        print(f"  - gesture_vsign: enabled")
        print(f"  - gesture_thumbsup: enabled")
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
        event_time_sec = trigger.event_time_ns / 1e9 if trigger.event_time_ns else 0
        meta = {
            "trigger_id": len(clip_metadata) + 1,
            "reason": result.reason,
            "score": result.score,
            "timestamp_sec": event_time_sec,
            "metadata": result.metadata,
        }
        clip_metadata.append(meta)
        print(f"\n  TRIGGER #{meta['trigger_id']}: {result.reason} (score={result.score:.2f}, t={event_time_sec:.2f}s)")

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
                "mode": "library",
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
            "mode": "library",
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
