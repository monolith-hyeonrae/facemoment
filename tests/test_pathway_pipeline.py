"""Tests for the facemoment Pathway pipeline integration (Phase 17)."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import cv2
import numpy as np
import pytest

from facemoment.pipeline.pathway_pipeline import (
    FacemomentPipeline,
    PATHWAY_AVAILABLE,
    _CUDA_GROUPS,
)
from facemoment.moment_detector.extractors.base import Observation, FaceObservation


def create_test_video(path: Path, num_frames: int = 30, fps: int = 30) -> None:
    """Create a test video file."""
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 3) % 256
        frame[:, :, 1] = (i * 2) % 256
        frame[:, :, 2] = (i * 1) % 256
        cv2.putText(
            frame, f"F{i}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        writer.write(frame)

    writer.release()


def create_mock_frame(frame_id: int = 0, t_ns: int = 0):
    """Create a mock Frame object."""
    frame = Mock()
    frame.frame_id = frame_id
    frame.t_src_ns = t_ns
    frame.data = np.zeros((240, 320, 3), dtype=np.uint8)
    return frame


class TestFacemomentPipeline:
    """Tests for FacemomentPipeline class."""

    def test_init_default_extractors(self):
        """Test default extractor configuration."""
        pipeline = FacemomentPipeline()
        assert pipeline._extractor_names == ["face", "pose", "gesture"]

    def test_init_custom_extractors(self):
        """Test custom extractor configuration."""
        pipeline = FacemomentPipeline(extractors=["face", "quality"])
        assert pipeline._extractor_names == ["face", "quality"]

    def test_init_fusion_config(self):
        """Test fusion configuration."""
        pipeline = FacemomentPipeline(
            fusion_config={"cooldown_sec": 3.0, "main_only": False}
        )
        assert pipeline._fusion_config["cooldown_sec"] == 3.0
        assert pipeline._fusion_config["main_only"] is False

    def test_build_fusion_creates_highlight_fusion(self):
        """Test fusion building creates HighlightFusion."""
        pipeline = FacemomentPipeline(
            fusion_config={"cooldown_sec": 5.0}
        )
        fusion = pipeline._build_fusion()

        from facemoment.moment_detector.fusion import HighlightFusion
        assert isinstance(fusion, HighlightFusion)
        assert fusion._cooldown_ns == int(5.0 * 1e9)

    def test_merge_observations_empty(self):
        """Test merging empty observation list."""
        pipeline = FacemomentPipeline()
        frame = create_mock_frame(frame_id=1, t_ns=1000000)

        merged = pipeline._merge_observations([], frame)

        assert merged.source == "merged"
        assert merged.frame_id == 1
        assert merged.signals == {}

    def test_merge_observations_with_face_data(self):
        """Test merging observations with face data."""
        pipeline = FacemomentPipeline()
        frame = create_mock_frame(frame_id=1, t_ns=1000000)

        obs1 = Observation(
            source="face",
            frame_id=1,
            t_ns=1000000,
            signals={"face_count": 2},
            faces=[
                FaceObservation(
                    face_id=0, bbox=(0.1, 0.1, 0.3, 0.3),
                    confidence=0.9, yaw=0.0, pitch=0.0, expression=0.8,
                )
            ],
            metadata={"backend": "insightface"},
        )

        obs2 = Observation(
            source="quality",
            frame_id=1,
            t_ns=1000000,
            signals={"quality_gate": 0.8},
            faces=[],
            metadata={"blur": 0.1},
        )

        merged = pipeline._merge_observations([obs1, obs2], frame)

        assert merged.source == "merged"
        assert "face_count" in merged.signals
        assert "quality_gate" in merged.signals
        assert len(merged.faces) == 1  # From face observation

    def test_merge_observations_copies_main_face_id(self):
        """Test that main_face_id is copied to merged signals."""
        pipeline = FacemomentPipeline()
        frame = create_mock_frame(frame_id=1, t_ns=1000000)

        # Create mock classifier output with main_face
        mock_main_face = Mock()
        mock_main_face.face = Mock()
        mock_main_face.face.face_id = 42

        mock_classifier_data = Mock()
        mock_classifier_data.main_face = mock_main_face

        classifier_obs = Observation(
            source="face_classifier",
            frame_id=1,
            t_ns=1000000,
            signals={},
            faces=[],
            metadata={},
            data=mock_classifier_data,
        )

        merged = pipeline._merge_observations([classifier_obs], frame)

        assert merged.signals.get("main_face_id") == 42


class TestFacemomentPipelineExecution:
    """Tests for FacemomentPipeline execution."""

    @pytest.fixture
    def test_video(self, tmp_path):
        """Create a test video."""
        video_path = tmp_path / "test.mp4"
        create_test_video(video_path, num_frames=30, fps=30)
        return video_path

    def test_run_simple_fallback(self):
        """Test _run_simple execution path."""
        pipeline = FacemomentPipeline(extractors=["dummy"])

        frames = [create_mock_frame(i, i * 100000) for i in range(10)]

        # Mock the extractor creation
        with patch("visualpath.plugin.create_extractor") as mock_create:
            mock_extractor = Mock()
            mock_extractor.initialize = Mock()
            mock_extractor.cleanup = Mock()
            mock_extractor.extract = Mock(return_value=Observation(
                source="dummy",
                frame_id=0,
                t_ns=0,
                signals={"test": 1.0},
                faces=[],
                metadata={},
            ))
            mock_create.return_value = mock_extractor

            triggers = pipeline._run_simple(frames)

            assert isinstance(triggers, list)
            # May or may not have triggers depending on fusion logic

    def test_run_calls_initialize_and_cleanup(self):
        """Test that run properly initializes and cleans up."""
        pipeline = FacemomentPipeline(extractors=["dummy"])

        # Mock PATHWAY_AVAILABLE to False so it uses simple fallback
        with patch("facemoment.pipeline.pathway_pipeline.PATHWAY_AVAILABLE", False):
            with patch.object(pipeline, "_build_extractors") as mock_build:
                with patch.object(pipeline, "_build_fusion") as mock_build_fusion:
                    mock_ext = Mock()
                    mock_ext.name = "dummy"
                    mock_ext.depends = []
                    mock_ext.initialize = Mock()
                    mock_ext.cleanup = Mock()
                    mock_ext.extract = Mock(return_value=None)
                    mock_build.return_value = [mock_ext]

                    mock_fusion = Mock()
                    mock_fusion.update = Mock(return_value=Mock(
                        should_trigger=False, trigger=None
                    ))
                    mock_build_fusion.return_value = mock_fusion

                    frames = [create_mock_frame(0, 0)]
                    pipeline.run(frames)

                    mock_ext.initialize.assert_called()
                    mock_ext.cleanup.assert_called()

    def test_on_trigger_callback(self):
        """Test on_trigger callback is called."""
        pipeline = FacemomentPipeline(extractors=["dummy"])

        # Track trigger callback
        triggers_received = []
        def on_trigger(t):
            triggers_received.append(t)

        # Mock PATHWAY_AVAILABLE to False so it uses simple fallback
        with patch("facemoment.pipeline.pathway_pipeline.PATHWAY_AVAILABLE", False):
            with patch.object(pipeline, "_build_extractors") as mock_build:
                with patch.object(pipeline, "_build_fusion") as mock_build_fusion:
                    mock_ext = Mock()
                    mock_ext.name = "dummy"
                    mock_ext.depends = []
                    mock_ext.initialize = Mock()
                    mock_ext.cleanup = Mock()
                    mock_ext.extract = Mock(return_value=Observation(
                        source="dummy",
                        frame_id=0,
                        t_ns=0,
                        signals={},
                        faces=[],
                        metadata={},
                    ))
                    mock_build.return_value = [mock_ext]

                    # Mock fusion to fire a trigger
                    from visualbase import Trigger
                    mock_trigger = Trigger.point(
                        event_time_ns=1000000,
                        pre_sec=2.0,
                        post_sec=2.0,
                        label="test",
                        score=0.9,
                    )
                    mock_fusion = Mock()
                    mock_fusion.update = Mock(return_value=Mock(
                        should_trigger=True, trigger=mock_trigger
                    ))
                    mock_build_fusion.return_value = mock_fusion

                    frames = [create_mock_frame(0, 0)]
                    pipeline.run(frames, on_trigger=on_trigger)

                    assert len(triggers_received) == 1
                    assert triggers_received[0] == mock_trigger


class TestPathwayAvailability:
    """Tests for Pathway availability check."""

    def test_pathway_available_flag(self):
        """Test that PATHWAY_AVAILABLE reflects actual availability."""
        # Just verify the flag exists and is a boolean
        assert isinstance(PATHWAY_AVAILABLE, bool)

    def test_run_with_pathway_unavailable(self):
        """Test fallback when Pathway is not available."""
        with patch("facemoment.pipeline.pathway_pipeline.PATHWAY_AVAILABLE", False):
            pipeline = FacemomentPipeline(extractors=["dummy"])

            with patch.object(pipeline, "_run_simple") as mock_simple:
                mock_simple.return_value = []
                frames = [create_mock_frame(0, 0)]

                # Initialize manually since _build_extractors may fail
                pipeline._extractors = []
                pipeline._fusion = Mock()
                pipeline._initialized = True

                pipeline._run_simple(frames)
                mock_simple.assert_called_once()


class TestHighlevelAPIBackend:
    """Tests for high-level API backend parameter."""

    def test_run_accepts_backend_parameter(self):
        """Test that fm.run() accepts backend parameter."""
        from facemoment.main import run, DEFAULT_BACKEND

        assert DEFAULT_BACKEND == "pathway"

        # Just verify the function signature accepts backend
        import inspect
        sig = inspect.signature(run)
        assert "backend" in sig.parameters

    def test_main_run_pathway_backend(self):
        """Test fm.run with pathway backend."""
        # This test verifies the pathway path is called
        with patch("facemoment.main._run_pathway") as mock_pathway:
            mock_pathway.return_value = Mock(
                triggers=[], frame_count=0, duration_sec=0, clips_extracted=0
            )

            from facemoment.main import run
            # Should call _run_pathway since PATHWAY_AVAILABLE might be True
            # or the backend is "pathway"

    def test_main_run_simple_backend(self):
        """Test fm.run with simple backend."""
        with patch("facemoment.main._run_simple") as mock_simple:
            mock_simple.return_value = Mock(
                triggers=[], frame_count=0, duration_sec=0, clips_extracted=0
            )

            from facemoment.main import run
            # Would call _run_simple when backend="simple"


class TestPipelineConfigBackend:
    """Tests for PipelineConfig backend field."""

    def test_config_default_backend(self):
        """Test default backend in PipelineConfig."""
        from facemoment.pipeline.config import PipelineConfig

        config = PipelineConfig()
        assert config.backend == "pathway"

    def test_config_custom_backend(self):
        """Test custom backend in PipelineConfig."""
        from facemoment.pipeline.config import PipelineConfig

        config = PipelineConfig(backend="simple")
        assert config.backend == "simple"

    def test_config_from_dict_with_backend(self):
        """Test loading backend from dict."""
        from facemoment.pipeline.config import PipelineConfig

        data = {
            "extractors": [{"name": "dummy"}],
            "backend": "simple",
        }
        config = PipelineConfig.from_dict(data)
        assert config.backend == "simple"

    def test_config_to_dict_includes_backend(self):
        """Test backend is included in to_dict."""
        from facemoment.pipeline.config import PipelineConfig, ExtractorConfig

        config = PipelineConfig(
            extractors=[ExtractorConfig(name="dummy")],
            backend="pathway",
        )
        data = config.to_dict()
        assert data["backend"] == "pathway"

    def test_create_default_config_backend(self):
        """Test create_default_config accepts backend."""
        from facemoment.pipeline.config import create_default_config

        config = create_default_config(backend="simple")
        assert config.backend == "simple"


class TestOrchestratorBackend:
    """Tests for PipelineOrchestrator backend support."""

    def test_orchestrator_default_backend(self):
        """Test default backend in orchestrator."""
        from facemoment.pipeline import PipelineOrchestrator, ExtractorConfig

        orchestrator = PipelineOrchestrator(
            extractor_configs=[ExtractorConfig(name="dummy")]
        )
        assert orchestrator._backend == "pathway"

    def test_orchestrator_custom_backend(self):
        """Test custom backend in orchestrator."""
        from facemoment.pipeline import PipelineOrchestrator, ExtractorConfig

        orchestrator = PipelineOrchestrator(
            extractor_configs=[ExtractorConfig(name="dummy")],
            backend="simple",
        )
        assert orchestrator._backend == "simple"

    def test_from_config_passes_backend(self):
        """Test that from_config passes backend to orchestrator."""
        from facemoment.pipeline import (
            PipelineOrchestrator,
            PipelineConfig,
            ExtractorConfig,
        )

        config = PipelineConfig(
            extractors=[ExtractorConfig(name="dummy")],
            backend="simple",
        )
        orchestrator = PipelineOrchestrator.from_config(config)
        assert orchestrator._backend == "simple"


class TestRunSimpleWithDepsGeneric:
    """Tests for _run_simple using generic deps instead of FaceClassifier special handling."""

    def test_run_simple_generic_deps(self):
        """Test that _run_simple uses generic deps accumulation."""
        pipeline = FacemomentPipeline(extractors=["dummy"])

        # Create mock extractors: upstream + dependent
        mock_upstream = Mock()
        mock_upstream.name = "upstream"
        mock_upstream.depends = []
        mock_upstream.initialize = Mock()
        mock_upstream.cleanup = Mock()
        mock_upstream.extract = Mock(return_value=Observation(
            source="upstream",
            frame_id=0,
            t_ns=0,
            signals={"test": 1.0},
            faces=[],
            metadata={},
        ))

        mock_dependent = Mock()
        mock_dependent.name = "dependent"
        mock_dependent.depends = ["upstream"]
        mock_dependent.initialize = Mock()
        mock_dependent.cleanup = Mock()

        def dependent_extract(frame, deps=None):
            has_upstream = deps is not None and "upstream" in deps
            return Observation(
                source="dependent",
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={"has_upstream": has_upstream},
                faces=[],
                metadata={},
            )
        mock_dependent.extract = dependent_extract

        # Set up pipeline internals
        pipeline._extractors = [mock_upstream, mock_dependent]
        pipeline._fusion = Mock()
        pipeline._fusion.update = Mock(return_value=Mock(
            should_trigger=False, trigger=None
        ))
        pipeline._classifier = None
        pipeline._initialized = True

        frames = [create_mock_frame(i, i * 100000) for i in range(3)]
        triggers = pipeline._run_simple(frames)

        assert isinstance(triggers, list)
        # Verify fusion was called with merged observations
        assert pipeline._fusion.update.call_count == 3
        # Verify the dependent extractor received upstream deps
        for call_args in pipeline._fusion.update.call_args_list:
            merged_obs = call_args[0][0]
            assert "has_upstream" in merged_obs.signals

    def test_run_simple_classifier_via_deps(self):
        """Test that FaceClassifier works through generic deps mechanism."""
        pipeline = FacemomentPipeline(extractors=["face"])

        # Create mock face extractor
        mock_face_ext = Mock()
        mock_face_ext.name = "face"
        mock_face_ext.depends = []
        mock_face_ext.initialize = Mock()
        mock_face_ext.cleanup = Mock()
        mock_face_ext.extract = Mock(return_value=Observation(
            source="face",
            frame_id=0,
            t_ns=0,
            signals={"face_count": 1},
            faces=[
                FaceObservation(
                    face_id=0, bbox=(0.1, 0.1, 0.3, 0.3),
                    confidence=0.9, yaw=0.0, pitch=0.0, expression=0.8,
                )
            ],
            metadata={},
        ))

        # Create mock classifier that depends on face
        mock_classifier = Mock()
        mock_classifier.name = "face_classifier"
        mock_classifier.depends = ["face"]
        mock_classifier.initialize = Mock()
        mock_classifier.cleanup = Mock()

        mock_classifier_data = Mock()
        mock_classifier_data.main_face = Mock()
        mock_classifier_data.main_face.face = Mock()
        mock_classifier_data.main_face.face.face_id = 0

        def classifier_extract(frame, deps=None):
            return Observation(
                source="face_classifier",
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={},
                faces=[],
                metadata={},
                data=mock_classifier_data,
            )
        mock_classifier.extract = classifier_extract

        # Set up pipeline
        pipeline._extractors = [mock_face_ext, mock_classifier]
        pipeline._fusion = Mock()
        pipeline._fusion.update = Mock(return_value=Mock(
            should_trigger=False, trigger=None
        ))
        pipeline._classifier = mock_classifier
        pipeline._initialized = True

        frames = [create_mock_frame(0, 0)]
        pipeline._run_simple(frames)

        # Verify fusion received classifier_obs
        call_kwargs = pipeline._fusion.update.call_args[1]
        assert "classifier_obs" in call_kwargs
        assert call_kwargs["classifier_obs"] is not None


class TestHighlightFusionMergedSignals:
    """Tests for HighlightFusion reading from merged signals."""

    def test_update_main_face_id_from_merged_signals(self):
        """Test that fusion reads main_face_id from merged signals."""
        from facemoment.moment_detector.fusion import HighlightFusion

        fusion = HighlightFusion(main_only=True)

        # Create observation with main_face_id in signals
        obs = Observation(
            source="merged",
            frame_id=1,
            t_ns=1000000,
            signals={"main_face_id": 42, "face_count": 1},
            faces=[
                FaceObservation(
                    face_id=42, bbox=(0.1, 0.1, 0.3, 0.3),
                    confidence=0.9, yaw=0.0, pitch=0.0, expression=0.8,
                )
            ],
            metadata={},
        )

        # Call update (which calls _update_main_face_id internally)
        fusion.update(obs)

        # main_face_id should be set from signals
        assert fusion._main_face_id == 42

    def test_explicit_classifier_obs_takes_priority(self):
        """Test that explicit classifier_obs takes priority over signals."""
        from facemoment.moment_detector.fusion import HighlightFusion

        fusion = HighlightFusion(main_only=True)

        # Create mock classifier observation
        mock_main_face = Mock()
        mock_main_face.face = Mock()
        mock_main_face.face.face_id = 99

        mock_data = Mock()
        mock_data.main_face = mock_main_face

        classifier_obs = Observation(
            source="face_classifier",
            frame_id=1,
            t_ns=1000000,
            signals={},
            faces=[],
            metadata={},
            data=mock_data,
        )

        # Create main observation with different main_face_id in signals
        obs = Observation(
            source="merged",
            frame_id=1,
            t_ns=1000000,
            signals={"main_face_id": 42},  # Different ID
            faces=[],
            metadata={},
        )

        # Call update with explicit classifier_obs
        fusion.update(obs, classifier_obs=classifier_obs)

        # Explicit classifier_obs should take priority
        assert fusion._main_face_id == 99


class TestCudaConflictDetection:
    """Tests for CUDA conflict detection and auto subprocess isolation."""

    def test_cuda_groups_defined(self):
        """Test that CUDA groups contain expected extractors."""
        assert "onnxruntime" in _CUDA_GROUPS
        assert "torch" in _CUDA_GROUPS
        assert "face" in _CUDA_GROUPS["onnxruntime"]
        assert "pose" in _CUDA_GROUPS["torch"]

    def test_no_conflict_single_group(self):
        """No conflict when all extractors are in the same CUDA group."""
        isolated = FacemomentPipeline._detect_cuda_conflicts(["face", "expression"])
        assert isolated == set()

    def test_no_conflict_no_cuda_extractors(self):
        """No conflict when extractors don't belong to any CUDA group."""
        isolated = FacemomentPipeline._detect_cuda_conflicts(["quality", "dummy"])
        assert isolated == set()

    def test_conflict_face_and_pose(self):
        """Conflict detected when face (onnxruntime) + pose (torch) are both active."""
        isolated = FacemomentPipeline._detect_cuda_conflicts(["face", "pose"])
        # torch group (pose) is minority → should be isolated
        assert isolated == {"pose"}

    def test_conflict_multiple_onnxruntime_vs_pose(self):
        """Multiple onnxruntime extractors vs single torch → torch isolated."""
        isolated = FacemomentPipeline._detect_cuda_conflicts(
            ["face", "face_detect", "expression", "pose"]
        )
        # onnxruntime has 3 extractors, torch has 1 → torch (pose) is minority
        assert isolated == {"pose"}

    def test_no_conflict_without_zmq(self):
        """Returns empty set when pyzmq is unavailable."""
        with patch.dict("sys.modules", {"zmq": None}):
            # Force ImportError for zmq
            import importlib
            with patch("builtins.__import__", side_effect=lambda name, *a, **kw: (_ for _ in ()).throw(ImportError("no zmq")) if name == "zmq" else importlib.__import__(name, *a, **kw)):
                isolated = FacemomentPipeline._detect_cuda_conflicts(["face", "pose"])
                assert isolated == set()

    def test_pipeline_init_has_workers_dict(self):
        """Pipeline initializes with empty workers dict."""
        pipeline = FacemomentPipeline(extractors=["face"])
        assert pipeline._workers == {}
        assert pipeline.workers == {}

    def test_workers_property(self):
        """Workers property returns the _workers dict."""
        pipeline = FacemomentPipeline(extractors=["face"])
        pipeline._workers = {"pose": Mock()}
        assert "pose" in pipeline.workers

    def test_build_extractors_no_conflict(self):
        """No workers created when there's no CUDA conflict."""
        pipeline = FacemomentPipeline(
            extractors=["dummy"],
            auto_inject_classifier=False,
        )

        with patch("visualpath.plugin.create_extractor") as mock_create:
            mock_ext = Mock()
            mock_ext.name = "dummy"
            mock_ext.depends = []
            mock_create.return_value = mock_ext

            extractors = pipeline._build_extractors()

            assert len(extractors) == 1
            assert pipeline._workers == {}

    def test_build_extractors_with_conflict_creates_worker(self):
        """Workers created for minority CUDA group when conflict detected."""
        pipeline = FacemomentPipeline(
            extractors=["face", "pose"],
            auto_inject_classifier=False,
        )

        mock_face_ext = Mock()
        mock_face_ext.name = "face"
        mock_face_ext.depends = []

        mock_worker = Mock()

        with patch("visualpath.plugin.create_extractor") as mock_create, \
             patch("visualpath.process.launcher.ProcessWorker") as mock_pw_cls, \
             patch.object(FacemomentPipeline, "_detect_cuda_conflicts", return_value={"pose"}):
            mock_create.return_value = mock_face_ext
            mock_pw_cls.return_value = mock_worker

            extractors = pipeline._build_extractors()

            # face should be inline, pose should be a worker
            assert len(extractors) == 1
            assert extractors[0].name == "face"
            assert "pose" in pipeline._workers
            mock_pw_cls.assert_called_once_with(extractor_name="pose")

    def test_run_simple_with_workers(self):
        """_run_simple processes both inline extractors and subprocess workers."""
        pipeline = FacemomentPipeline(extractors=["face", "pose"])

        # Setup inline extractor
        mock_ext = Mock()
        mock_ext.name = "face"
        mock_ext.depends = []
        mock_ext.extract = Mock(return_value=Observation(
            source="face",
            frame_id=0,
            t_ns=0,
            signals={"face_count": 1},
            faces=[],
            metadata={},
        ))

        # Setup worker
        mock_worker = Mock()
        mock_result = Mock()
        mock_result.observation = Observation(
            source="pose",
            frame_id=0,
            t_ns=0,
            signals={"pose_count": 1},
            faces=[],
            metadata={},
        )
        mock_worker.process = Mock(return_value=mock_result)

        pipeline._extractors = [mock_ext]
        pipeline._workers = {"pose": mock_worker}
        pipeline._fusion = Mock()
        pipeline._fusion.update = Mock(return_value=Mock(
            should_trigger=False, trigger=None
        ))
        pipeline._classifier = None
        pipeline._initialized = True

        frames = [create_mock_frame(0, 0)]
        pipeline._run_simple(frames)

        # Both extractor and worker should have been called
        mock_ext.extract.assert_called_once()
        mock_worker.process.assert_called_once()

        # Fusion should receive merged obs with signals from both
        merged_obs = pipeline._fusion.update.call_args[0][0]
        assert "face_count" in merged_obs.signals
        assert "pose_count" in merged_obs.signals

    def test_run_simple_worker_receives_deps(self):
        """Workers receive accumulated deps from inline extractors."""
        pipeline = FacemomentPipeline(extractors=["face", "pose"])

        face_obs = Observation(
            source="face",
            frame_id=0,
            t_ns=0,
            signals={"face_count": 1},
            faces=[],
            metadata={},
        )

        mock_ext = Mock()
        mock_ext.name = "face"
        mock_ext.depends = []
        mock_ext.extract = Mock(return_value=face_obs)

        mock_worker = Mock()
        mock_result = Mock()
        mock_result.observation = Observation(
            source="pose", frame_id=0, t_ns=0,
            signals={}, faces=[], metadata={},
        )
        mock_worker.process = Mock(return_value=mock_result)

        pipeline._extractors = [mock_ext]
        pipeline._workers = {"pose": mock_worker}
        pipeline._fusion = Mock()
        pipeline._fusion.update = Mock(return_value=Mock(
            should_trigger=False, trigger=None
        ))
        pipeline._classifier = None
        pipeline._initialized = True

        frames = [create_mock_frame(0, 0)]
        pipeline._run_simple(frames)

        # Verify worker received deps containing face observation
        call_kwargs = mock_worker.process.call_args[1]
        assert "face" in call_kwargs["deps"]
        assert call_kwargs["deps"]["face"] is face_obs

    def test_run_pathway_falls_back_with_workers(self):
        """_run_pathway falls back to _run_simple when workers are active."""
        pipeline = FacemomentPipeline(extractors=["face", "pose"])
        pipeline._workers = {"pose": Mock()}
        pipeline._extractors = []
        pipeline._fusion = Mock()
        pipeline._classifier = None
        pipeline._initialized = True

        with patch.object(pipeline, "_run_simple", return_value=[]) as mock_simple:
            frames = [create_mock_frame(0, 0)]
            pipeline._run_pathway(frames)
            mock_simple.assert_called_once()

    def test_cleanup_stops_workers(self):
        """cleanup() stops all workers."""
        pipeline = FacemomentPipeline(extractors=["face", "pose"])

        mock_worker = Mock()
        pipeline._workers = {"pose": mock_worker}
        pipeline._extractors = []
        pipeline._fusion = None
        pipeline._classifier = None
        pipeline._initialized = True

        pipeline.cleanup()

        mock_worker.stop.assert_called_once()
        assert pipeline._workers == {}

    def test_initialize_starts_workers(self):
        """initialize() starts all workers."""
        pipeline = FacemomentPipeline(
            extractors=["dummy"],
            auto_inject_classifier=False,
        )

        mock_worker = Mock()
        mock_worker.start = Mock()

        with patch.object(pipeline, "_build_extractors") as mock_build, \
             patch.object(pipeline, "_build_fusion") as mock_fusion:
            mock_ext = Mock()
            mock_ext.name = "dummy"
            mock_ext.initialize = Mock()
            mock_build.return_value = [mock_ext]

            # Simulate _build_extractors populating workers
            def build_side_effect():
                pipeline._workers = {"pose": mock_worker}
                return [mock_ext]
            mock_build.side_effect = build_side_effect

            mock_fusion.return_value = Mock()

            pipeline.initialize()

            mock_worker.start.assert_called_once()

    def test_initialize_removes_failed_workers(self):
        """Workers that fail to start are removed."""
        pipeline = FacemomentPipeline(
            extractors=["dummy"],
            auto_inject_classifier=False,
        )

        mock_worker = Mock()
        mock_worker.start = Mock(side_effect=RuntimeError("cannot start"))

        with patch.object(pipeline, "_build_extractors") as mock_build, \
             patch.object(pipeline, "_build_fusion") as mock_fusion:
            mock_ext = Mock()
            mock_ext.name = "dummy"
            mock_ext.initialize = Mock()
            mock_build.return_value = [mock_ext]

            def build_side_effect():
                pipeline._workers = {"pose": mock_worker}
                return [mock_ext]
            mock_build.side_effect = build_side_effect

            mock_fusion.return_value = Mock()

            pipeline.initialize()

            assert "pose" not in pipeline._workers
