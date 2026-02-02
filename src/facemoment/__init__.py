"""facemoment - Face and moment detection for video analysis.

Quick Start:
    >>> import facemoment as fm
    >>> result = fm.run("video.mp4")
    >>> print(f"Found {len(result.triggers)} highlights")

With options:
    >>> result = fm.run("video.mp4", fps=10, cooldown=3.0)
    >>> result = fm.run("video.mp4", output_dir="./clips")

Advanced:
    >>> from facemoment import MomentDetector
    >>> from facemoment.moment_detector.extractors import FaceExtractor
"""

from facemoment.main import (
    # Configuration
    DEFAULT_FPS,
    DEFAULT_COOLDOWN,
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
