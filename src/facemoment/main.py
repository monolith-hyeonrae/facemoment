"""High-level API for facemoment.

Quick Start:
    >>> import facemoment as fm
    >>> result = fm.run("video.mp4")
    >>> print(f"Found {len(result.triggers)} highlights")

With options:
    >>> result = fm.run("video.mp4", fps=10, cooldown=3.0)
    >>> result = fm.run("video.mp4", output_dir="./clips")
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
        on_trigger: Callback when a trigger fires.

    Returns:
        Result with triggers, frame_count, duration_sec, clips_extracted.

    Example:
        >>> result = fm.run("video.mp4")
        >>> result = fm.run("video.mp4", output_dir="./clips")
        >>> result = fm.run("video.mp4", extractors=["face", "pose"])
    """

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
