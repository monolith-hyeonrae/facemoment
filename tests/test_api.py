"""Tests for facemoment high-level API."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestRunFunction:
    """Tests for fm.run() function."""

    def test_import(self):
        """Test that run can be imported."""
        import facemoment as fm
        assert hasattr(fm, "run")
        assert hasattr(fm, "Result")

    def test_run_returns_result(self):
        """Test run returns Result."""
        import facemoment as fm

        with patch.object(fm.MomentDetector, "process_file") as mock_process:
            mock_process.return_value = []

            result = fm.run("fake_video.mp4", extractors=["dummy"])

            assert isinstance(result, fm.Result)
            assert hasattr(result, "triggers")
            assert hasattr(result, "frame_count")
            assert hasattr(result, "duration_sec")
            assert hasattr(result, "clips_extracted")

    def test_run_with_callback(self):
        """Test run with on_trigger callback."""
        import facemoment as fm

        with patch.object(fm.MomentDetector, "process_file") as mock_process:
            mock_process.return_value = []

            callbacks = []
            result = fm.run("fake_video.mp4", extractors=["dummy"], on_trigger=lambda t: callbacks.append(t))

            assert isinstance(result, fm.Result)

    def test_run_with_output_dir(self):
        """Test run with output directory for clips."""
        import facemoment as fm

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(fm.MomentDetector, "process_file") as mock_process:
                mock_process.return_value = []

                result = fm.run("fake_video.mp4", output_dir=tmpdir, extractors=["dummy"])

                assert result.clips_extracted == 0

    def test_run_with_specific_extractors(self):
        """Test run with specific extractor list."""
        import facemoment as fm

        with patch.object(fm.MomentDetector, "process_file") as mock_process:
            mock_process.return_value = []

            result = fm.run("fake_video.mp4", extractors=["quality"])

            assert isinstance(result, fm.Result)


class TestConfiguration:
    """Tests for configuration constants."""

    def test_extractors_dict(self):
        """Test EXTRACTORS dict."""
        import facemoment as fm

        assert "face" in fm.EXTRACTORS
        assert "pose" in fm.EXTRACTORS
        assert "quality" in fm.EXTRACTORS
        assert fm.EXTRACTORS["face"].__name__ == "FaceExtractor"

    def test_fusions_dict(self):
        """Test FUSIONS dict."""
        import facemoment as fm

        assert "highlight" in fm.FUSIONS
        assert "dummy" in fm.FUSIONS
        assert fm.FUSIONS["highlight"].__name__ == "HighlightFusion"

    def test_default_values(self):
        """Test default configuration values."""
        import facemoment as fm

        assert fm.DEFAULT_FPS == 10
        assert fm.DEFAULT_COOLDOWN == 2.0


class TestExports:
    """Tests for __all__ exports."""

    def test_all_exports_exist(self):
        """Test all items in __all__ are actually exported."""
        import facemoment as fm

        for name in fm.__all__:
            assert hasattr(fm, name), f"Missing export: {name}"

    def test_core_exports(self):
        """Test core classes are exported."""
        import facemoment as fm

        assert callable(fm.run)
        assert fm.Result is not None
        assert fm.MomentDetector is not None
        assert callable(fm.visualize)
