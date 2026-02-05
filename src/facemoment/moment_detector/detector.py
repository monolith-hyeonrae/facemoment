"""MomentDetector - Main class for highlight detection."""

from typing import List, Optional, Iterator, Callable
from pathlib import Path

from visualbase import VisualBase, FileSource, Frame, Trigger, ClipResult

from facemoment.moment_detector.extractors.base import BaseExtractor, Observation
from facemoment.moment_detector.fusion.base import BaseFusion


class MomentDetector:
    """Detects highlight moments in video streams.

    MomentDetector combines:
    - VisualBase for video input and clip extraction
    - Extractors (B modules) for feature analysis
    - Fusion (C module) for trigger decisions

    Args:
        extractors: List of feature extractors to use.
        fusion: Fusion module for trigger decisions.
        clip_output_dir: Directory for extracted clips.

    Example:
        >>> from facemoment.moment_detector.extractors import DummyExtractor
        >>> from facemoment.moment_detector.fusion import DummyFusion
        >>>
        >>> detector = MomentDetector(
        ...     extractors=[DummyExtractor(num_faces=2)],
        ...     fusion=DummyFusion(expression_threshold=0.8),
        ...     clip_output_dir=Path("./clips"),
        ... )
        >>>
        >>> results = detector.process_file("video.mp4", fps=10)
        >>> for clip_result in results:
        ...     print(f"Clip saved: {clip_result.output_path}")
    """

    def __init__(
        self,
        extractors: List[BaseExtractor],
        fusion: BaseFusion,
        clip_output_dir: Optional[Path] = None,
    ):
        self._extractors = extractors
        self._fusion = fusion
        self._clip_output_dir = Path(clip_output_dir) if clip_output_dir else Path("./clips")
        self._vb: Optional[VisualBase] = None

        # Callbacks
        self._on_frame: Optional[Callable[[Frame], None]] = None
        self._on_observation: Optional[Callable[[Observation], None]] = None
        self._on_trigger: Optional[Callable[[Trigger, Observation], None]] = None

        # Stats
        self._frames_processed: int = 0
        self._triggers_fired: int = 0

    def process_file(
        self,
        video_path: str,
        fps: int = 10,
        resolution: Optional[tuple[int, int]] = None,
    ) -> List[ClipResult]:
        """Process a video file and extract highlight clips.

        Args:
            video_path: Path to video file.
            fps: Analysis frame rate (default: 10).
            resolution: Analysis resolution (default: None = original).

        Returns:
            List of ClipResult for each extracted clip.
        """
        clips: List[ClipResult] = []

        # Initialize extractors
        for ext in self._extractors:
            ext.initialize()

        try:
            self._vb = VisualBase(clip_output_dir=self._clip_output_dir)
            self._vb.connect(FileSource(video_path))
            self._fusion.reset()

            for frame in self._vb.get_stream(fps=fps, resolution=resolution):
                self._frames_processed += 1

                # Callback
                if self._on_frame:
                    self._on_frame(frame)

                # Run extractors
                for extractor in self._extractors:
                    observation = extractor.extract(frame)

                    if observation is not None:
                        # Callback
                        if self._on_observation:
                            self._on_observation(observation)

                        # Feed to fusion
                        result = self._fusion.update(observation)

                        if result.should_trigger and result.trigger is not None:
                            self._triggers_fired += 1

                            # Callback
                            if self._on_trigger:
                                self._on_trigger(result.trigger, result)

                            # Extract clip
                            clip_result = self._vb.trigger(result.trigger)
                            clips.append(clip_result)

            self._vb.disconnect()

        finally:
            # Cleanup extractors
            for ext in self._extractors:
                ext.cleanup()

        return clips

    def process_stream(
        self,
        video_path: str,
        fps: int = 10,
        resolution: Optional[tuple[int, int]] = None,
    ) -> Iterator[tuple[Frame, Optional[Observation]]]:
        """Process video as a stream, yielding frames and fusion results.

        This is useful for real-time visualization and debugging.

        Args:
            video_path: Path to video file.
            fps: Analysis frame rate.
            resolution: Analysis resolution.

        Yields:
            Tuple of (frame, fusion_result). fusion_result is None if
            no observation was produced, or Observation from fusion.
        """
        # Initialize extractors
        for ext in self._extractors:
            ext.initialize()

        try:
            self._vb = VisualBase(clip_output_dir=self._clip_output_dir)
            self._vb.connect(FileSource(video_path))
            self._fusion.reset()

            for frame in self._vb.get_stream(fps=fps, resolution=resolution):
                self._frames_processed += 1

                fusion_result = None

                # Run extractors (use first extractor for simplicity)
                for extractor in self._extractors:
                    observation = extractor.extract(frame)

                    if observation is not None:
                        fusion_result = self._fusion.update(observation)

                        if fusion_result.should_trigger and fusion_result.trigger is not None:
                            self._triggers_fired += 1
                            # Extract clip in background
                            self._vb.trigger(fusion_result.trigger)

                yield frame, fusion_result

            self._vb.disconnect()

        finally:
            for ext in self._extractors:
                ext.cleanup()

    def set_on_frame(self, callback: Callable[[Frame], None]) -> None:
        """Set callback for each processed frame."""
        self._on_frame = callback

    def set_on_observation(self, callback: Callable[[Observation], None]) -> None:
        """Set callback for each observation."""
        self._on_observation = callback

    def set_on_trigger(
        self, callback: Callable[[Trigger, Observation], None]
    ) -> None:
        """Set callback for each trigger."""
        self._on_trigger = callback

    @property
    def frames_processed(self) -> int:
        """Number of frames processed."""
        return self._frames_processed

    @property
    def triggers_fired(self) -> int:
        """Number of triggers fired."""
        return self._triggers_fired

    @property
    def clip_output_dir(self) -> Path:
        """Output directory for clips."""
        return self._clip_output_dir

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self._frames_processed = 0
        self._triggers_fired = 0
