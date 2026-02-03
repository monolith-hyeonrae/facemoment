"""High-level API for facemoment.

Quick Start:
    >>> import facemoment as fm
    >>> result = fm.run("video.mp4")
    >>> print(f"Found {len(result.triggers)} highlights")

With options:
    >>> result = fm.run("video.mp4", fps=10, cooldown=3.0)
    >>> result = fm.run("video.mp4", output_dir="./clips")
    >>> result = fm.run("video.mp4", backend="pathway")  # Use Pathway backend
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

from visualbase import Trigger

from facemoment.moment_detector import MomentDetector
from facemoment.moment_detector.extractors import (
    FaceExtractor,
    PoseExtractor,
    GestureExtractor,
    QualityExtractor,
    DummyExtractor,
)
from facemoment.moment_detector.fusion import HighlightFusion, DummyFusion


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_FPS = 10
DEFAULT_COOLDOWN = 2.0
DEFAULT_BACKEND = "pathway"  # "pathway" or "simple"

EXTRACTORS = {
    "face": FaceExtractor,
    "pose": PoseExtractor,
    "gesture": GestureExtractor,
    "quality": QualityExtractor,
    "dummy": DummyExtractor,
}

FUSIONS = {
    "highlight": HighlightFusion,
    "dummy": DummyFusion,
}


# =============================================================================
# Result Type
# =============================================================================

@dataclass
class Result:
    """Result from fm.run()."""
    triggers: List[Trigger] = field(default_factory=list)
    frame_count: int = 0
    duration_sec: float = 0.0
    clips_extracted: int = 0


# =============================================================================
# High-level API
# =============================================================================

def run(
    video: Union[str, Path],
    *,
    extractors: Optional[Sequence[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    fps: int = DEFAULT_FPS,
    cooldown: float = DEFAULT_COOLDOWN,
    backend: str = DEFAULT_BACKEND,
    on_trigger: Optional[Callable[[Trigger], None]] = None,
) -> Result:
    """Process a video and return results.

    Args:
        video: Path to video file.
        extractors: Extractor names ["face", "pose", "gesture", "quality"].
                   None = use all ML extractors.
        output_dir: Directory for extracted clips. None = no clips.
        fps: Frames per second to process.
        cooldown: Seconds between triggers.
        backend: Execution backend ("pathway" or "simple"). Default: "pathway".
        on_trigger: Callback when a trigger fires.

    Returns:
        Result with triggers, frame_count, duration_sec, clips_extracted.

    Example:
        >>> result = fm.run("video.mp4")
        >>> result = fm.run("video.mp4", output_dir="./clips")
        >>> result = fm.run("video.mp4", extractors=["face", "pose"])
        >>> result = fm.run("video.mp4", backend="pathway")
    """
    from facemoment.pipeline.pathway_pipeline import (
        FacemomentPipeline,
        PATHWAY_AVAILABLE,
    )

    # Determine which backend to use
    use_pathway = backend == "pathway" and PATHWAY_AVAILABLE

    if use_pathway or backend == "pathway":
        # Use Pathway-based pipeline (with fallback to simple if unavailable)
        return _run_pathway(
            video=video,
            extractors=extractors,
            output_dir=output_dir,
            fps=fps,
            cooldown=cooldown,
            on_trigger=on_trigger,
        )
    else:
        # Use simple/library mode
        return _run_simple(
            video=video,
            extractors=extractors,
            output_dir=output_dir,
            fps=fps,
            cooldown=cooldown,
            on_trigger=on_trigger,
        )


def _run_pathway(
    video: Union[str, Path],
    extractors: Optional[Sequence[str]],
    output_dir: Optional[Union[str, Path]],
    fps: int,
    cooldown: float,
    on_trigger: Optional[Callable[[Trigger], None]],
) -> Result:
    """Run using Pathway-based pipeline."""
    from facemoment.pipeline.pathway_pipeline import FacemomentPipeline
    from facemoment.cli.utils import create_video_stream

    # Build extractor list
    extractor_names = list(extractors) if extractors else ["face", "pose", "gesture"]

    # Create pipeline
    pipeline = FacemomentPipeline(
        extractors=extractor_names,
        fusion_config={"cooldown_sec": cooldown, "main_only": True},
    )

    # Get frames from video
    vb, source, stream = create_video_stream(str(video), fps=fps)

    result = Result()

    try:
        # Collect frames
        frames = list(stream)
        result.frame_count = len(frames)
        result.duration_sec = result.frame_count / fps if fps > 0 else 0

        # Run pipeline
        triggers = pipeline.run(frames, on_trigger=on_trigger)
        result.triggers = triggers

        # Extract clips if output_dir specified
        if output_dir:
            from visualbase import VisualBase, FileSource
            clip_vb = VisualBase(clip_output_dir=Path(output_dir))
            clip_vb.connect(FileSource(str(video)))

            clips_extracted = 0
            for trigger in triggers:
                clip_result = clip_vb.trigger(trigger)
                if clip_result.success:
                    clips_extracted += 1

            clip_vb.disconnect()
            result.clips_extracted = clips_extracted

    finally:
        vb.disconnect()

    return result


def _run_simple(
    video: Union[str, Path],
    extractors: Optional[Sequence[str]],
    output_dir: Optional[Union[str, Path]],
    fps: int,
    cooldown: float,
    on_trigger: Optional[Callable[[Trigger], None]],
) -> Result:
    """Run using simple/library mode (MomentDetector)."""
    if extractors:
        extractor_list = [EXTRACTORS[n]() for n in extractors]
    else:
        extractor_list = [
            EXTRACTORS["face"](),
            EXTRACTORS["pose"](),
            EXTRACTORS["gesture"](),
            EXTRACTORS["quality"](),
        ]
    fusion = FUSIONS["highlight"](cooldown_sec=cooldown)

    detector = MomentDetector(
        extractors=extractor_list,
        fusion=fusion,
        clip_output_dir=Path(output_dir) if output_dir else None,
    )

    # Process
    result = Result()

    def collect_trigger(trigger, fusion_result):
        result.triggers.append(trigger)
        if on_trigger:
            on_trigger(trigger)

    detector.set_on_trigger(collect_trigger)
    clips = detector.process_file(str(video), fps=fps)

    result.frame_count = detector.frames_processed
    result.duration_sec = result.frame_count / fps if fps > 0 else 0
    result.clips_extracted = len([c for c in clips if c.success])

    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m facemoment.main <video_path> [--output-dir DIR]")
        print("\nExample:")
        print("  python -m facemoment.main video.mp4")
        print("  python -m facemoment.main video.mp4 --output-dir ./clips")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = None

    if "--output-dir" in sys.argv:
        idx = sys.argv.index("--output-dir")
        if idx + 1 < len(sys.argv):
            output_dir = sys.argv[idx + 1]

    print(f"Processing: {video_path}")
    result = run(video_path, output_dir=output_dir)

    print(f"\nResults:")
    print(f"  Frames processed: {result.frame_count}")
    print(f"  Duration: {result.duration_sec:.1f}s")
    print(f"  Triggers found: {len(result.triggers)}")
    print(f"  Clips extracted: {result.clips_extracted}")

    for i, trigger in enumerate(result.triggers, 1):
        print(f"\n  [{i}] {trigger.label} (score={trigger.score:.2f})")
