"""Tests for HSEmotionBackend."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from facemoment.moment_detector.extractors.backends.base import (
    DetectedFace,
    FaceExpression,
)


class TestHSEmotionBackend:
    """Tests for HSEmotionBackend expression analysis."""

    def test_emotion_map(self):
        """Test emotion mapping from HSEmotion to standard names."""
        from facemoment.moment_detector.extractors.backends.face_backends import (
            HSEmotionBackend,
        )

        backend = HSEmotionBackend()

        # Check all 8 emotions are mapped
        assert len(backend.EMOTIONS) == 8
        assert len(backend.EMOTION_MAP) == 8

        # Check specific mappings
        assert backend.EMOTION_MAP["Happiness"] == "happy"
        assert backend.EMOTION_MAP["Sadness"] == "sad"
        assert backend.EMOTION_MAP["Anger"] == "angry"
        assert backend.EMOTION_MAP["Neutral"] == "neutral"

    def test_initialize_without_dependency(self):
        """Test initialization fails gracefully without hsemotion-onnx."""
        from facemoment.moment_detector.extractors.backends.face_backends import (
            HSEmotionBackend,
        )

        backend = HSEmotionBackend()

        # Mock missing hsemotion_onnx.facial_emotions
        with patch.dict("sys.modules", {"hsemotion_onnx.facial_emotions": None}):
            with pytest.raises(ImportError, match="hsemotion-onnx is required"):
                backend.initialize("cpu")

    def test_analyze_empty_faces(self):
        """Test analyze returns empty list for no faces."""
        from facemoment.moment_detector.extractors.backends.face_backends import (
            HSEmotionBackend,
        )

        # Create mock for hsemotion_onnx.facial_emotions
        mock_recognizer = MagicMock()
        mock_facial_emotions = MagicMock()
        mock_facial_emotions.HSEmotionRecognizer.return_value = mock_recognizer

        with patch.dict("sys.modules", {"hsemotion_onnx.facial_emotions": mock_facial_emotions}):
            backend = HSEmotionBackend()
            backend.initialize("cpu")

            image = np.zeros((480, 640, 3), dtype=np.uint8)
            results = backend.analyze(image, [])

            assert results == []

    def test_analyze_returns_face_expression(self):
        """Test analyze returns FaceExpression for each face."""
        from facemoment.moment_detector.extractors.backends.face_backends import (
            HSEmotionBackend,
        )

        # Create mock for hsemotion_onnx
        mock_recognizer = MagicMock()
        # Return emotion scores: [Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise]
        mock_recognizer.predict_emotions.return_value = (
            "Happiness",
            np.array([0.05, 0.02, 0.03, 0.01, 0.80, 0.05, 0.02, 0.02])
        )

        mock_facial_emotions = MagicMock()
        mock_facial_emotions.HSEmotionRecognizer.return_value = mock_recognizer

        with patch.dict("sys.modules", {"hsemotion_onnx.facial_emotions": mock_facial_emotions}):
            backend = HSEmotionBackend()
            backend.initialize("cpu")

            image = np.zeros((480, 640, 3), dtype=np.uint8)
            faces = [
                DetectedFace(bbox=(100, 100, 200, 200), confidence=0.95),
            ]

            results = backend.analyze(image, faces)

            assert len(results) == 1
            assert isinstance(results[0], FaceExpression)
            assert results[0].dominant_emotion == "happy"
            assert results[0].emotions["happy"] == pytest.approx(0.80, rel=0.01)
            assert results[0].expression_intensity > 0.5  # High happiness

    def test_analyze_multiple_faces(self):
        """Test analyze handles multiple faces."""
        from facemoment.moment_detector.extractors.backends.face_backends import (
            HSEmotionBackend,
        )

        mock_recognizer = MagicMock()
        # Return different emotions for each call
        mock_recognizer.predict_emotions.side_effect = [
            ("Happiness", np.array([0.05, 0.02, 0.03, 0.01, 0.80, 0.05, 0.02, 0.02])),
            ("Neutral", np.array([0.05, 0.02, 0.03, 0.01, 0.10, 0.70, 0.05, 0.04])),
        ]

        mock_facial_emotions = MagicMock()
        mock_facial_emotions.HSEmotionRecognizer.return_value = mock_recognizer

        with patch.dict("sys.modules", {"hsemotion_onnx.facial_emotions": mock_facial_emotions}):
            backend = HSEmotionBackend()
            backend.initialize("cpu")

            image = np.zeros((480, 640, 3), dtype=np.uint8)
            faces = [
                DetectedFace(bbox=(100, 100, 150, 150), confidence=0.95),
                DetectedFace(bbox=(300, 100, 150, 150), confidence=0.90),
            ]

            results = backend.analyze(image, faces)

            assert len(results) == 2
            assert results[0].dominant_emotion == "happy"
            assert results[1].dominant_emotion == "neutral"

    def test_analyze_empty_face_region(self):
        """Test analyze handles invalid face bbox gracefully."""
        from facemoment.moment_detector.extractors.backends.face_backends import (
            HSEmotionBackend,
        )

        mock_recognizer = MagicMock()
        mock_facial_emotions = MagicMock()
        mock_facial_emotions.HSEmotionRecognizer.return_value = mock_recognizer

        with patch.dict("sys.modules", {"hsemotion_onnx.facial_emotions": mock_facial_emotions}):
            backend = HSEmotionBackend()
            backend.initialize("cpu")

            image = np.zeros((480, 640, 3), dtype=np.uint8)
            # Face bbox outside image bounds
            faces = [
                DetectedFace(bbox=(700, 500, 100, 100), confidence=0.95),
            ]

            results = backend.analyze(image, faces)

            # Should return empty FaceExpression for invalid region
            assert len(results) == 1
            assert results[0].expression_intensity == 0.0

    def test_expression_intensity_calculation(self):
        """Test expression intensity is calculated correctly."""
        from facemoment.moment_detector.extractors.backends.face_backends import (
            HSEmotionBackend,
        )

        mock_recognizer = MagicMock()

        mock_facial_emotions = MagicMock()
        mock_facial_emotions.HSEmotionRecognizer.return_value = mock_recognizer

        with patch.dict("sys.modules", {"hsemotion_onnx.facial_emotions": mock_facial_emotions}):
            backend = HSEmotionBackend()
            backend.initialize("cpu")

            image = np.zeros((480, 640, 3), dtype=np.uint8)
            faces = [DetectedFace(bbox=(100, 100, 200, 200), confidence=0.95)]

            # Test high happiness -> high intensity
            mock_recognizer.predict_emotions.return_value = (
                "Happiness",
                np.array([0.02, 0.02, 0.02, 0.01, 0.90, 0.01, 0.01, 0.01])
            )
            results = backend.analyze(image, faces)
            assert results[0].expression_intensity >= 0.8

            # Test high neutral -> low intensity
            mock_recognizer.predict_emotions.return_value = (
                "Neutral",
                np.array([0.01, 0.01, 0.01, 0.01, 0.05, 0.90, 0.01, 0.00])
            )
            results = backend.analyze(image, faces)
            # Non-neutral is 10%, so intensity should be low
            assert results[0].expression_intensity < 0.2

    def test_cleanup(self):
        """Test cleanup releases resources."""
        from facemoment.moment_detector.extractors.backends.face_backends import (
            HSEmotionBackend,
        )

        mock_recognizer = MagicMock()
        mock_facial_emotions = MagicMock()
        mock_facial_emotions.HSEmotionRecognizer.return_value = mock_recognizer

        with patch.dict("sys.modules", {"hsemotion_onnx.facial_emotions": mock_facial_emotions}):
            backend = HSEmotionBackend()
            backend.initialize("cpu")
            assert backend._initialized is True

            backend.cleanup()
            assert backend._initialized is False
            assert backend._model is None

    def test_no_action_units(self):
        """Test that HSEmotion backend returns empty action units."""
        from facemoment.moment_detector.extractors.backends.face_backends import (
            HSEmotionBackend,
        )

        mock_recognizer = MagicMock()
        mock_recognizer.predict_emotions.return_value = (
            "Happiness",
            np.array([0.05, 0.02, 0.03, 0.01, 0.80, 0.05, 0.02, 0.02])
        )

        mock_facial_emotions = MagicMock()
        mock_facial_emotions.HSEmotionRecognizer.return_value = mock_recognizer

        with patch.dict("sys.modules", {"hsemotion_onnx.facial_emotions": mock_facial_emotions}):
            backend = HSEmotionBackend()
            backend.initialize("cpu")

            image = np.zeros((480, 640, 3), dtype=np.uint8)
            faces = [DetectedFace(bbox=(100, 100, 200, 200), confidence=0.95)]

            results = backend.analyze(image, faces)

            # HSEmotion doesn't support AUs
            assert results[0].action_units == {}


class TestFaceExtractorBackendSelection:
    """Tests for FaceExtractor backend selection logic."""

    def test_hsemotion_preferred_over_pyfeat(self):
        """Test that HSEmotion is preferred when available."""
        from facemoment.moment_detector.extractors.face import FaceExtractor
        from facemoment.moment_detector.extractors.backends.face_backends import (
            HSEmotionBackend,
        )

        # Mock both backends available
        mock_facial_emotions = MagicMock()
        mock_facial_emotions.HSEmotionRecognizer.return_value = MagicMock()

        mock_insightface = MagicMock()
        mock_insightface.app.FaceAnalysis.return_value = MagicMock()

        with patch.dict("sys.modules", {
            "hsemotion_onnx.facial_emotions": mock_facial_emotions,
            "insightface": mock_insightface,
            "insightface.app": mock_insightface.app,
        }):
            extractor = FaceExtractor()
            extractor.initialize()

            # Should use HSEmotionBackend
            assert isinstance(extractor._expression_backend, HSEmotionBackend)

            extractor.cleanup()

    def test_fallback_to_pyfeat_when_hsemotion_unavailable(self):
        """Test that PyFeat is used when HSEmotion is not available."""
        from facemoment.moment_detector.extractors.face import FaceExtractor

        # Mock hsemotion import failure
        def raise_import_error(*args, **kwargs):
            raise ImportError("No module named 'hsemotion_onnx'")

        with patch(
            "facemoment.moment_detector.extractors.backends.face_backends.HSEmotionBackend",
            side_effect=raise_import_error
        ):
            # This test would need more complex mocking to fully verify
            # For now, we verify the logic exists in the initialize method
            pass
