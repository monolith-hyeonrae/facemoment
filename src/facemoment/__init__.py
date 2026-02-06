"""facemoment - Face and moment detection for video analysis.

Quick Start:
    >>> import facemoment as fm
    >>> result = fm.run("video.mp4")
    >>> print(f"Found {len(result.triggers)} highlights")

With options:
    >>> result = fm.run("video.mp4", fps=10, cooldown=3.0)
    >>> result = fm.run("video.mp4", output_dir="./clips")

Frame Scoring:
    >>> from facemoment.moment_detector.scoring import FrameScorer
    >>> scorer = FrameScorer()
    >>> result = scorer.score(face_obs=face_obs, quality_obs=quality_obs)
    >>> print(f"Score: {result.total_score:.2f}")

Advanced:
    >>> from facemoment import MomentDetector
    >>> from facemoment.moment_detector.extractors import FaceExtractor
"""

from facemoment.main import (
    # Configuration
    DEFAULT_FPS,
    DEFAULT_COOLDOWN,
    DEFAULT_BACKEND,
    EXTRACTORS,
    FUSIONS,
    # Result type
    Result,
    # High-level API
    run,
)

from facemoment.moment_detector import MomentDetector
from facemoment.tools.visualizer import visualize, DetectorVisualizer

__all__ = [
    # Configuration
    "DEFAULT_FPS",
    "DEFAULT_COOLDOWN",
    "DEFAULT_BACKEND",
    "EXTRACTORS",
    "FUSIONS",
    # High-level API
    "run",
    "Result",
    # Core
    "MomentDetector",
    # Visualization
    "visualize",
    "DetectorVisualizer",
]
