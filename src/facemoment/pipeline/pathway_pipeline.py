"""Pathway pipeline builder for facemoment.

This module provides a Pathway-based pipeline that integrates facemoment's
extractors and fusion logic with visualpath's PathwayBackend.

Features:
- FaceClassifier auto-injection when face extractor is used
- HighlightFusion main_only mode support
- Observation merging for fusion
- Fallback to simple backend when Pathway is unavailable

Example:
    >>> from facemoment.pipeline.pathway_pipeline import FacemomentPipeline
    >>>
    >>> pipeline = FacemomentPipeline(
    ...     extractors=["face", "pose"],
    ...     fusion_config={"cooldown_sec": 2.0, "main_only": True},
    ... )
    >>> triggers = pipeline.run(frames)
"""

from typing import Callable, Dict, Iterator, List, Optional, Set, Union
import logging

from visualbase import Frame, Trigger

logger = logging.getLogger(__name__)

# CUDA conflict groups: extractors sharing the same CUDA runtime binding.
# If extractors from 2+ groups are active, the minority group must run in
# a subprocess to avoid symbol conflicts (e.g. onnxruntime-gpu vs torch).
_CUDA_GROUPS: Dict[str, Set[str]] = {
    "onnxruntime": {"face", "face_detect", "expression"},
    "torch": {"pose"},
}

# Check for Pathway availability
try:
    from visualpath.backends.pathway import PathwayBackend
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False


class FacemomentPipeline:
    """Pathway-based pipeline with facemoment-specific logic.

    Handles:
    - FaceExtractor -> FaceClassifier dependency
    - HighlightFusion main_only mode
    - Observation merging for fusion
    - Fallback to simple execution when Pathway unavailable

    Args:
        extractors: List of extractor names to use (default: ["face", "pose", "gesture"]).
        fusion_config: Configuration for HighlightFusion.
        window_ns: Window size for temporal joins (default: 100ms).
        auto_inject_classifier: Auto-inject FaceClassifier when face extractor is used.

    Example:
        >>> pipeline = FacemomentPipeline(
        ...     extractors=["face", "pose"],
        ...     fusion_config={"cooldown_sec": 2.0},
        ... )
        >>> triggers = pipeline.run(frames)
    """

    def __init__(
        self,
        extractors: Optional[List[str]] = None,
        fusion_config: Optional[Dict] = None,
        window_ns: int = 100_000_000,  # 100ms
        auto_inject_classifier: bool = True,
    ):
        self._extractor_names = extractors or ["face", "pose", "gesture"]
        self._fusion_config = fusion_config or {}
        self._window_ns = window_ns
        self._auto_inject_classifier = auto_inject_classifier
        self._extractors = []
        self._workers: Dict[str, "ProcessWorker"] = {}
        self._fusion = None
        self._classifier = None
        self._initialized = False
        self.actual_backend: Optional[str] = None

    # Extractors that depend on torch must initialize before onnxruntime-gpu.
    # If onnxruntime loads CUDA first, torch's libc10_cuda.so fails with
    # "undefined symbol: cudaGetDriverEntryPointByVersion".
    # This ordering is the fallback when pyzmq is unavailable for subprocess isolation.
    _TORCH_EXTRACTORS = frozenset({"pose"})

    @staticmethod
    def _detect_cuda_conflicts(names: List[str]) -> Set[str]:
        """Detect CUDA conflicts among active extractors.

        When extractors from 2+ CUDA groups are active (e.g. onnxruntime-gpu
        for face + torch for pose), the minority group is returned for
        subprocess isolation via ProcessWorker.

        Returns empty set if pyzmq is unavailable (falls back to ordering).

        Args:
            names: List of extractor names to check.

        Returns:
            Set of extractor names that should run in a subprocess.
        """
        try:
            import zmq  # noqa: F401
        except ImportError:
            logger.debug("pyzmq not available, skipping CUDA conflict detection")
            return set()

        # Map each extractor to its CUDA group
        active_groups: Dict[str, List[str]] = {}
        for name in names:
            for group, members in _CUDA_GROUPS.items():
                if name in members:
                    active_groups.setdefault(group, []).append(name)
                    break

        if len(active_groups) < 2:
            return set()

        # Isolate the smallest group (fewest extractors).
        # On tie, prefer isolating "torch" so onnxruntime stays in-process.
        _ISOLATE_PREFERENCE = {"torch": 0, "onnxruntime": 1}
        minority_group = min(
            active_groups,
            key=lambda g: (len(active_groups[g]), _ISOLATE_PREFERENCE.get(g, 99)),
        )
        isolated = set(active_groups[minority_group])
        logger.info(
            f"CUDA conflict detected: groups {list(active_groups.keys())}. "
            f"Isolating {minority_group} extractors {isolated} to subprocess."
        )
        return isolated

    def _build_extractors(self) -> List:
        """Build extractor instances with FaceClassifier auto-injection.

        When CUDA conflicts are detected and pyzmq is available, conflicting
        extractors are wrapped in ProcessWorker for subprocess isolation.
        Otherwise, torch-based extractors are ordered first as a fallback.

        Returns:
            List of inline extractor instances.
            Side effect: populates self._workers with ProcessWorker instances.
        """
        from visualpath.plugin import create_extractor

        isolated_names = self._detect_cuda_conflicts(self._extractor_names)

        # Sort remaining (non-isolated) names: torch first as fallback ordering
        inline_names = [n for n in self._extractor_names if n not in isolated_names]
        ordered_names = sorted(
            inline_names,
            key=lambda n: 0 if n in self._TORCH_EXTRACTORS else 1,
        )

        extractors = []
        has_face = False

        # Build inline extractors
        for name in ordered_names:
            try:
                ext = create_extractor(name)
                extractors.append(ext)
                if name in ("face", "face_detect"):
                    has_face = True
            except Exception as e:
                logger.warning(f"Failed to create extractor '{name}': {e}")

        # Build subprocess workers for isolated extractors
        self._workers = {}
        if isolated_names:
            try:
                from visualpath.process.launcher import ProcessWorker
                for name in isolated_names:
                    try:
                        worker = ProcessWorker(extractor_name=name)
                        self._workers[name] = worker
                        logger.info(f"Created ProcessWorker for '{name}'")
                    except Exception as e:
                        logger.warning(
                            f"Failed to create ProcessWorker for '{name}': {e}. "
                            f"Falling back to inline."
                        )
                        # Fallback: create inline extractor
                        try:
                            ext = create_extractor(name)
                            extractors.append(ext)
                        except Exception as e2:
                            logger.warning(f"Failed to create inline extractor '{name}': {e2}")
            except ImportError:
                logger.warning("ProcessWorker not available, using inline extractors")
                for name in isolated_names:
                    try:
                        ext = create_extractor(name)
                        extractors.append(ext)
                    except Exception as e:
                        logger.warning(f"Failed to create extractor '{name}': {e}")

        # Check if any isolated extractor provides face data
        for name in isolated_names:
            if name in ("face", "face_detect"):
                has_face = True

        # Auto-inject FaceClassifier when face extractor is used
        if has_face and self._auto_inject_classifier:
            try:
                from facemoment.moment_detector.extractors import FaceClassifierExtractor
                self._classifier = FaceClassifierExtractor()
                extractors.append(self._classifier)
                logger.debug("Auto-injected FaceClassifierExtractor")
            except Exception as e:
                logger.warning(f"Failed to create FaceClassifierExtractor: {e}")

        return extractors

    def _build_fusion(self):
        """Build HighlightFusion with facemoment-specific config.

        Returns:
            HighlightFusion instance.
        """
        from facemoment.moment_detector.fusion import HighlightFusion

        return HighlightFusion(
            main_only=self._fusion_config.get("main_only", True),
            cooldown_sec=self._fusion_config.get("cooldown_sec", 2.0),
            **{k: v for k, v in self._fusion_config.items()
               if k not in ("main_only", "cooldown_sec")},
        )

    def initialize(self) -> None:
        """Initialize extractors, workers, and fusion.

        Extractors that fail to initialize are gracefully skipped.
        Workers that fail to start are removed (extractor lost).
        """
        if self._initialized:
            return

        self._extractors = self._build_extractors()
        self._fusion = self._build_fusion()

        # Initialize inline extractors, removing those that fail
        initialized_extractors = []
        for ext in self._extractors:
            try:
                ext.initialize()
                initialized_extractors.append(ext)
            except Exception as e:
                logger.warning(f"Failed to initialize extractor '{ext.name}': {e}")
                # Update classifier reference if it was the one that failed
                if ext is self._classifier:
                    self._classifier = None

        self._extractors = initialized_extractors

        # Start subprocess workers
        failed_workers = []
        for name, worker in self._workers.items():
            try:
                worker.start()
                logger.info(f"Started ProcessWorker for '{name}'")
            except Exception as e:
                logger.warning(f"Failed to start ProcessWorker for '{name}': {e}")
                failed_workers.append(name)
        for name in failed_workers:
            del self._workers[name]

        self._initialized = True

    def cleanup(self) -> None:
        """Clean up extractors, workers, and fusion."""
        for ext in self._extractors:
            try:
                ext.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up extractor: {e}")

        for name, worker in self._workers.items():
            try:
                worker.stop()
            except Exception as e:
                logger.warning(f"Error stopping worker '{name}': {e}")

        self._extractors = []
        self._workers = {}
        self._fusion = None
        self._classifier = None
        self._initialized = False

    def run(
        self,
        frames: Union[Iterator[Frame], List[Frame]],
        on_trigger: Optional[Callable[[Trigger], None]] = None,
    ) -> List[Trigger]:
        """Run the facemoment pipeline.

        Uses PathwayBackend if available, otherwise falls back to simple execution.

        Args:
            frames: Iterator or list of Frame objects.
            on_trigger: Optional callback for each trigger.

        Returns:
            List of triggers that fired.
        """
        self.initialize()

        try:
            if PATHWAY_AVAILABLE:
                return self._run_pathway(frames, on_trigger)
            else:
                logger.warning("Pathway backend not available, falling back to simple execution")
                self.actual_backend = "simple"
                return self._run_simple(frames, on_trigger)
        finally:
            self.cleanup()

    def _run_pathway(
        self,
        frames: Union[Iterator[Frame], List[Frame]],
        on_trigger: Optional[Callable[[Trigger], None]] = None,
    ) -> List[Trigger]:
        """Run pipeline using PathwayBackend.

        Falls back to _run_simple() when subprocess workers are active,
        since PathwayBackend does not manage ProcessWorker instances.

        Args:
            frames: Iterator or list of Frame objects.
            on_trigger: Optional callback for each trigger.

        Returns:
            List of triggers that fired.
        """
        if self._workers:
            logger.warning(
                "ProcessWorkers active (%s), Pathway cannot manage subprocesses — falling back to simple execution",
                list(self._workers.keys()),
            )
            self.actual_backend = "simple"
            return self._run_simple(frames, on_trigger)

        try:
            backend = PathwayBackend(window_ns=self._window_ns)
        except ImportError:
            logger.warning("Pathway not installed, falling back to simple execution")
            self.actual_backend = "simple"
            return self._run_simple(frames, on_trigger)

        self.actual_backend = "pathway"

        # Ensure frames is an iterator
        frame_iter = iter(frames) if isinstance(frames, list) else frames

        return backend.run(
            frames=frame_iter,
            extractors=self._extractors,
            fusion=self._fusion,
            on_trigger=on_trigger,
        )

    def _run_simple(
        self,
        frames: Union[Iterator[Frame], List[Frame]],
        on_trigger: Optional[Callable[[Trigger], None]] = None,
    ) -> List[Trigger]:
        """Run pipeline using simple sequential execution.

        Fallback for when Pathway is not available. Uses generic deps
        accumulation following the Path._extract_with_deps() pattern.

        Args:
            frames: Iterator or list of Frame objects.
            on_trigger: Optional callback for each trigger.

        Returns:
            List of triggers that fired.
        """
        triggers = []
        frame_iter = iter(frames) if isinstance(frames, list) else frames

        for frame in frame_iter:
            deps = {}
            observations = []
            classifier_obs = None

            # 1) Inline extractors (ordered, deps accumulated)
            for ext in self._extractors:
                extractor_deps = None
                if ext.depends:
                    extractor_deps = {
                        name: deps[name]
                        for name in ext.depends
                        if name in deps
                    }
                    # Composite "face" extractor satisfies "face_detect" dependency
                    if "face_detect" in ext.depends and "face_detect" not in extractor_deps and "face" in deps:
                        extractor_deps["face"] = deps["face"]
                try:
                    obs = ext.extract(frame, extractor_deps)
                except TypeError:
                    obs = ext.extract(frame)

                if obs is not None:
                    observations.append(obs)
                    deps[ext.name] = obs
                    if ext is self._classifier:
                        classifier_obs = obs

            # 2) Subprocess workers (receive accumulated deps)
            for name, worker in self._workers.items():
                try:
                    result = worker.process(frame, deps=deps)
                    if result.observation:
                        observations.append(result.observation)
                        deps[name] = result.observation
                except Exception as e:
                    logger.warning(f"Worker '{name}' failed: {e}")

            # Skip if no observations
            if not observations:
                continue

            # 3) Merge observations for fusion
            merged_obs = self._merge_observations(observations, frame)

            # Run fusion with classifier observation
            result = self._fusion.update(merged_obs, classifier_obs=classifier_obs)

            if result.should_trigger and result.trigger:
                triggers.append(result.trigger)
                if on_trigger:
                    on_trigger(result.trigger)

        return triggers

    def _merge_observations(
        self,
        observations: List,
        frame: Frame,
    ):
        """Merge multiple observations into a single observation for fusion.

        Args:
            observations: List of observations from different extractors.
            frame: The source frame.

        Returns:
            Merged observation.
        """
        from facemoment.moment_detector.extractors.base import Observation

        if not observations:
            return Observation(
                source="merged",
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={},
                faces=[],
                metadata={},
            )

        # Find the observation with face data
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

            # Copy classifier main_face_id to signals for fusion
            if obs.source == "face_classifier" and obs.data is not None:
                if hasattr(obs.data, 'main_face') and obs.data.main_face is not None:
                    merged_signals["main_face_id"] = obs.data.main_face.face.face_id

        return Observation(
            source="merged",
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals=merged_signals,
            faces=getattr(base_obs, 'faces', []),
            metadata=merged_metadata,
        )

    @property
    def extractors(self) -> List:
        """Get current extractor instances."""
        return self._extractors

    @property
    def fusion(self):
        """Get current fusion instance."""
        return self._fusion

    @property
    def workers(self) -> Dict:
        """Get current subprocess worker instances."""
        return self._workers


__all__ = ["FacemomentPipeline", "PATHWAY_AVAILABLE"]
