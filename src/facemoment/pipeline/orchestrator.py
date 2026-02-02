"""PipelineOrchestrator - A-B*-C-A distributed pipeline orchestration.

This module provides the main orchestrator for running the facemoment
pipeline in distributed mode, with each extractor potentially running
in its own virtual environment.

Architecture:
    A: Video Input (visualbase)
       │
       │ Frame
       ▼
    B* Extractors (VenvWorker/InlineWorker)
       ┌─────────┐  ┌─────────┐  ┌───────────┐
       │  face   │  │  pose   │  │  gesture  │
       │[venv-1] │  │[venv-2] │  │ [venv-3]  │
       └────┬────┘  └────┬────┘  └─────┬─────┘
            └────────────┴─────────────┘
                         │ Observations
                         ▼
    C: HighlightFusion
       - Gate check → Trigger detection
       │
       │ Trigger
       ▼
    A: Clip Output (visualbase.trigger())

Example:
    >>> from facemoment.pipeline import PipelineOrchestrator, ExtractorConfig
    >>>
    >>> configs = [
    ...     ExtractorConfig(name="face", venv_path="/opt/venv-face"),
    ...     ExtractorConfig(name="pose", venv_path="/opt/venv-pose"),
    ... ]
    >>>
    >>> orchestrator = PipelineOrchestrator(extractor_configs=configs)
    >>> clips = orchestrator.run("video.mp4", fps=10)
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from visualbase import ClipResult, Frame, Trigger, VisualBase, FileSource

from visualpath.core import IsolationLevel
from visualpath.process import (
    WorkerLauncher,
    BaseWorker,
    WorkerResult,
)
from visualpath.plugin import create_extractor, load_fusion

from facemoment.pipeline.config import ExtractorConfig, FusionConfig, PipelineConfig
from facemoment.moment_detector.fusion.base import BaseFusion, FusionResult
from facemoment.moment_detector.extractors.base import Observation

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Statistics from a pipeline run.

    Attributes:
        frames_processed: Total frames processed.
        triggers_fired: Number of triggers fired.
        clips_extracted: Number of clips successfully extracted.
        processing_time_sec: Total processing time in seconds.
        avg_frame_time_ms: Average time per frame in milliseconds.
        worker_stats: Per-worker statistics.
    """

    frames_processed: int = 0
    triggers_fired: int = 0
    clips_extracted: int = 0
    processing_time_sec: float = 0.0
    avg_frame_time_ms: float = 0.0
    worker_stats: Dict[str, Dict[str, Any]] = None

    def __post_init__(self):
        if self.worker_stats is None:
            self.worker_stats = {}


class PipelineOrchestrator:
    """Orchestrates the A-B*-C-A distributed pipeline.

    Manages the complete facemoment processing pipeline:
    - A: Video input via visualbase
    - B*: Multiple extractors running in parallel (potentially in separate venvs)
    - C: Fusion module for trigger decisions
    - A: Clip extraction via visualbase

    Args:
        extractor_configs: List of ExtractorConfig for each extractor.
        fusion: Optional pre-configured fusion instance. If None, uses HighlightFusion.
        fusion_config: Optional FusionConfig for creating fusion. Ignored if fusion is provided.
        clip_output_dir: Directory for extracted clips.

    Example:
        >>> orchestrator = PipelineOrchestrator(
        ...     extractor_configs=[
        ...         ExtractorConfig(name="face", venv_path="/opt/venv-face"),
        ...         ExtractorConfig(name="pose"),  # inline
        ...     ],
        ...     clip_output_dir=Path("./clips"),
        ... )
        >>>
        >>> clips = orchestrator.run("video.mp4", fps=10)
        >>> for clip in clips:
        ...     print(f"Extracted: {clip.output_path}")
    """

    def __init__(
        self,
        extractor_configs: List[ExtractorConfig],
        fusion: Optional[BaseFusion] = None,
        fusion_config: Optional[FusionConfig] = None,
        clip_output_dir: Optional[Path] = None,
    ):
        if not extractor_configs:
            raise ValueError("At least one extractor config is required")

        self._extractor_configs = extractor_configs
        self._fusion = fusion
        self._fusion_config = fusion_config or FusionConfig()
        self._clip_output_dir = Path(clip_output_dir) if clip_output_dir else Path("./clips")

        # Workers (created on start)
        self._workers: Dict[str, BaseWorker] = {}
        self._vb: Optional[VisualBase] = None
        self._initialized = False

        # Callbacks
        self._on_frame: Optional[Callable[[Frame], None]] = None
        self._on_observations: Optional[Callable[[List[Observation]], None]] = None
        self._on_trigger: Optional[Callable[[Trigger, FusionResult], None]] = None

        # Stats
        self._stats = PipelineStats()

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "PipelineOrchestrator":
        """Create orchestrator from a PipelineConfig.

        Args:
            config: Complete pipeline configuration.

        Returns:
            Configured PipelineOrchestrator.
        """
        return cls(
            extractor_configs=config.extractors,
            fusion_config=config.fusion,
            clip_output_dir=Path(config.clip_output_dir),
        )

    def _create_fusion(self) -> BaseFusion:
        """Create fusion instance from config."""
        if self._fusion is not None:
            return self._fusion

        # Try to load from entry points
        try:
            FusionClass = load_fusion(self._fusion_config.name)
            kwargs = {
                "cooldown_sec": self._fusion_config.cooldown_sec,
                **self._fusion_config.kwargs,
            }
            return FusionClass(**kwargs)
        except (KeyError, ImportError):
            # Fall back to importing directly
            from facemoment.moment_detector.fusion import HighlightFusion
            return HighlightFusion(
                cooldown_sec=self._fusion_config.cooldown_sec,
                **self._fusion_config.kwargs,
            )

    def _create_workers(self) -> None:
        """Create worker instances for each extractor."""
        for config in self._extractor_configs:
            isolation = config.effective_isolation

            # For INLINE/THREAD, we need to load the extractor
            extractor = None
            if isolation in (IsolationLevel.INLINE, IsolationLevel.THREAD):
                try:
                    extractor = create_extractor(config.name, **config.kwargs)
                except Exception as e:
                    logger.warning(
                        f"Failed to create extractor '{config.name}': {e}. "
                        f"Skipping this extractor."
                    )
                    continue

            worker = WorkerLauncher.create(
                level=isolation,
                extractor=extractor,
                venv_path=config.venv_path,
                extractor_name=config.name,
            )

            self._workers[config.name] = worker
            logger.info(
                f"Created worker for '{config.name}' with isolation={isolation.name}"
            )

    def _start_workers(self) -> None:
        """Start all worker instances.

        Workers that fail to start are removed from the workers dict.
        Raises if no workers start successfully.
        """
        failed_workers = []

        for name, worker in list(self._workers.items()):
            try:
                worker.start()
                logger.info(f"Started worker: {name}")
            except Exception as e:
                logger.warning(f"Failed to start worker '{name}': {e}. Skipping.")
                failed_workers.append(name)

        # Remove failed workers
        for name in failed_workers:
            del self._workers[name]

        if not self._workers:
            raise RuntimeError(
                f"No workers started successfully. Failed workers: {failed_workers}"
            )

    def _stop_workers(self) -> None:
        """Stop all worker instances."""
        for name, worker in self._workers.items():
            try:
                worker.stop()
                logger.info(f"Stopped worker: {name}")
            except Exception as e:
                logger.warning(f"Error stopping worker '{name}': {e}")

    def _process_frame(self, frame: Frame) -> List[Observation]:
        """Process a frame through all workers.

        Args:
            frame: Frame to process.

        Returns:
            List of observations from all workers.
        """
        observations = []

        for name, worker in self._workers.items():
            result = worker.process(frame)

            # Track timing stats
            if name not in self._stats.worker_stats:
                self._stats.worker_stats[name] = {
                    "frames": 0,
                    "total_ms": 0.0,
                    "errors": 0,
                }

            stats = self._stats.worker_stats[name]
            stats["frames"] += 1
            stats["total_ms"] += result.timing_ms

            if result.error:
                stats["errors"] += 1
                logger.warning(f"Worker '{name}' error: {result.error}")
            elif result.observation is not None:
                observations.append(result.observation)

        return observations

    def _merge_observations(
        self, observations: List[Observation], frame: Frame
    ) -> Observation:
        """Merge multiple observations into a single observation for fusion.

        Combines signals and metadata from all observations into one,
        using the first face-containing observation as the base.

        Args:
            observations: List of observations from different extractors.
            frame: The source frame.

        Returns:
            Merged observation.
        """
        if not observations:
            # Return empty observation
            return Observation(
                source="merged",
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={},
                faces=[],
                metadata={},
            )

        # Find the observation with face data (typically from face extractor)
        base_obs = None
        for obs in observations:
            if hasattr(obs, 'faces') and obs.faces:
                base_obs = obs
                break

        if base_obs is None:
            base_obs = observations[0]

        # Merge signals from all observations
        merged_signals = {}
        merged_metadata = {}

        for obs in observations:
            merged_signals.update(obs.signals)
            merged_metadata[obs.source] = obs.metadata

        # Create merged observation
        return Observation(
            source="merged",
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals=merged_signals,
            faces=getattr(base_obs, 'faces', []),
            metadata=merged_metadata,
        )

    def run(
        self,
        video_path: str,
        fps: int = 10,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> List[ClipResult]:
        """Run the complete pipeline on a video file.

        Args:
            video_path: Path to the input video file.
            fps: Analysis frame rate (default: 10).
            resolution: Analysis resolution (default: original).

        Returns:
            List of ClipResult for each extracted clip.

        Example:
            >>> clips = orchestrator.run("video.mp4", fps=10)
            >>> print(f"Extracted {len(clips)} clips")
        """
        clips: List[ClipResult] = []
        start_time = time.time()

        # Reset stats
        self._stats = PipelineStats()

        # Create fusion
        fusion = self._create_fusion()
        fusion.reset()

        # Create and start workers
        self._create_workers()
        self._start_workers()
        self._initialized = True

        try:
            # Connect to video source
            self._clip_output_dir.mkdir(parents=True, exist_ok=True)
            self._vb = VisualBase(clip_output_dir=self._clip_output_dir)
            self._vb.connect(FileSource(video_path))

            # Process frames
            for frame in self._vb.get_stream(fps=fps, resolution=resolution):
                self._stats.frames_processed += 1

                # Callback
                if self._on_frame:
                    self._on_frame(frame)

                # Process through all workers
                observations = self._process_frame(frame)

                # Callback
                if self._on_observations:
                    self._on_observations(observations)

                # Skip if no observations
                if not observations:
                    continue

                # Merge observations for fusion
                merged_obs = self._merge_observations(observations, frame)

                # Feed to fusion
                result = fusion.update(merged_obs)

                if result.should_trigger and result.trigger is not None:
                    self._stats.triggers_fired += 1

                    # Callback
                    if self._on_trigger:
                        self._on_trigger(result.trigger, result)

                    # Extract clip
                    clip_result = self._vb.trigger(result.trigger)
                    clips.append(clip_result)

                    if clip_result.success:
                        self._stats.clips_extracted += 1

            # Disconnect from video source
            self._vb.disconnect()

        finally:
            # Stop workers
            self._stop_workers()
            self._workers.clear()
            self._initialized = False

        # Calculate final stats
        elapsed = time.time() - start_time
        self._stats.processing_time_sec = elapsed
        if self._stats.frames_processed > 0:
            self._stats.avg_frame_time_ms = (elapsed * 1000) / self._stats.frames_processed

        return clips

    def run_stream(
        self,
        video_path: str,
        fps: int = 10,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> Iterator[Tuple[Frame, List[Observation], Optional[FusionResult]]]:
        """Run pipeline as a stream, yielding results for each frame.

        Useful for real-time visualization and debugging.

        Args:
            video_path: Path to the input video file.
            fps: Analysis frame rate.
            resolution: Analysis resolution.

        Yields:
            Tuple of (frame, observations, fusion_result).
            fusion_result is None if no trigger evaluation was performed.

        Example:
            >>> for frame, observations, result in orchestrator.run_stream("video.mp4"):
            ...     if result and result.should_trigger:
            ...         print(f"Trigger at frame {frame.frame_id}!")
        """
        # Reset stats
        self._stats = PipelineStats()

        # Create fusion
        fusion = self._create_fusion()
        fusion.reset()

        # Create and start workers
        self._create_workers()
        self._start_workers()
        self._initialized = True

        try:
            # Connect to video source
            self._clip_output_dir.mkdir(parents=True, exist_ok=True)
            self._vb = VisualBase(clip_output_dir=self._clip_output_dir)
            self._vb.connect(FileSource(video_path))

            for frame in self._vb.get_stream(fps=fps, resolution=resolution):
                self._stats.frames_processed += 1

                # Process through all workers
                observations = self._process_frame(frame)

                fusion_result = None
                if observations:
                    # Merge and feed to fusion
                    merged_obs = self._merge_observations(observations, frame)
                    fusion_result = fusion.update(merged_obs)

                    if fusion_result.should_trigger and fusion_result.trigger:
                        self._stats.triggers_fired += 1
                        # Extract clip in background
                        clip_result = self._vb.trigger(fusion_result.trigger)
                        if clip_result.success:
                            self._stats.clips_extracted += 1

                yield frame, observations, fusion_result

            self._vb.disconnect()

        finally:
            self._stop_workers()
            self._workers.clear()
            self._initialized = False

    def set_on_frame(self, callback: Callable[[Frame], None]) -> None:
        """Set callback for each processed frame.

        Args:
            callback: Function called with each frame.
        """
        self._on_frame = callback

    def set_on_observations(
        self, callback: Callable[[List[Observation]], None]
    ) -> None:
        """Set callback for observations from each frame.

        Args:
            callback: Function called with list of observations.
        """
        self._on_observations = callback

    def set_on_trigger(
        self, callback: Callable[[Trigger, FusionResult], None]
    ) -> None:
        """Set callback for each trigger event.

        Args:
            callback: Function called with trigger and fusion result.
        """
        self._on_trigger = callback

    def get_stats(self) -> PipelineStats:
        """Get processing statistics.

        Returns:
            PipelineStats with processing metrics.
        """
        return self._stats

    @property
    def is_initialized(self) -> bool:
        """Check if the orchestrator is initialized."""
        return self._initialized

    @property
    def worker_names(self) -> List[str]:
        """Get names of active workers."""
        return list(self._workers.keys())

    @property
    def clip_output_dir(self) -> Path:
        """Get the clip output directory."""
        return self._clip_output_dir
