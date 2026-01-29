"""Tests for QualityExtractor."""

import numpy as np
import pytest

from visualbase import Frame

from facemoment.moment_detector.extractors.quality import QualityExtractor


class TestQualityExtractor:
    def test_extractor_name(self):
        """Test extractor name property."""
        extractor = QualityExtractor()
        assert extractor.name == "quality"

    def test_extract_sharp_image(self):
        """Test extraction on sharp image with edges."""
        # Create image with sharp edges
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add checkerboard pattern for high frequency content
        for i in range(0, 480, 20):
            for j in range(0, 640, 20):
                if (i // 20 + j // 20) % 2 == 0:
                    image[i : i + 20, j : j + 20] = 200

        extractor = QualityExtractor(blur_threshold=50.0)

        frame = Frame.from_array(image, frame_id=0, t_src_ns=0)
        obs = extractor.extract(frame)

        assert obs is not None
        assert obs.source == "quality"
        assert obs.signals["blur_score"] > 50.0  # Sharp image
        assert obs.signals["blur_quality"] >= 1.0

    def test_extract_blurry_image(self):
        """Test extraction on blurry image."""
        import cv2

        # Create sharp image then blur it
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(0, 480, 20):
            for j in range(0, 640, 20):
                if (i // 20 + j // 20) % 2 == 0:
                    image[i : i + 20, j : j + 20] = 200

        # Apply heavy blur
        blurred = cv2.GaussianBlur(image, (51, 51), 0)

        extractor = QualityExtractor(blur_threshold=100.0)

        frame = Frame.from_array(blurred, frame_id=0, t_src_ns=0)
        obs = extractor.extract(frame)

        assert obs.signals["blur_score"] < 100.0  # Blurry
        assert obs.signals["quality_gate"] == 0.0  # Gate closed due to blur

    def test_extract_dark_image(self):
        """Test extraction on dark image."""
        # Very dark image
        image = np.full((480, 640, 3), 20, dtype=np.uint8)

        extractor = QualityExtractor(brightness_low=50.0)

        frame = Frame.from_array(image, frame_id=0, t_src_ns=0)
        obs = extractor.extract(frame)

        assert obs.signals["brightness"] < 50.0
        assert obs.signals["brightness_quality"] < 1.0
        assert obs.metadata["is_well_lit"] is False

    def test_extract_bright_image(self):
        """Test extraction on overly bright image."""
        # Very bright image
        image = np.full((480, 640, 3), 240, dtype=np.uint8)

        extractor = QualityExtractor(brightness_high=200.0)

        frame = Frame.from_array(image, frame_id=0, t_src_ns=0)
        obs = extractor.extract(frame)

        assert obs.signals["brightness"] > 200.0
        assert obs.signals["brightness_quality"] < 1.0
        assert obs.metadata["is_well_lit"] is False

    def test_extract_optimal_brightness(self):
        """Test extraction on optimally lit image."""
        # Medium brightness
        image = np.full((480, 640, 3), 120, dtype=np.uint8)
        # Add some variation for contrast
        image[::2, ::2] = 140

        extractor = QualityExtractor(
            brightness_low=50.0,
            brightness_high=200.0,
        )

        frame = Frame.from_array(image, frame_id=0, t_src_ns=0)
        obs = extractor.extract(frame)

        assert 50.0 < obs.signals["brightness"] < 200.0
        assert obs.signals["brightness_quality"] == 1.0
        assert obs.metadata["is_well_lit"] is True

    def test_extract_low_contrast(self):
        """Test extraction on low contrast image."""
        # Very uniform image (low contrast)
        image = np.full((480, 640, 3), 128, dtype=np.uint8)
        # Add very small variation
        image[::2, ::2] = 130

        extractor = QualityExtractor(contrast_threshold=40.0)

        frame = Frame.from_array(image, frame_id=0, t_src_ns=0)
        obs = extractor.extract(frame)

        assert obs.signals["contrast"] < 40.0
        assert obs.metadata["has_contrast"] is False

    def test_extract_high_contrast(self):
        """Test extraction on high contrast image."""
        # High contrast image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[::2] = 255  # Alternating black and white rows

        extractor = QualityExtractor(contrast_threshold=40.0)

        frame = Frame.from_array(image, frame_id=0, t_src_ns=0)
        obs = extractor.extract(frame)

        assert obs.signals["contrast"] > 40.0
        assert obs.metadata["has_contrast"] is True

    def test_quality_gate_all_conditions_met(self):
        """Test quality gate opens when all conditions are met."""
        # Create image with good quality
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Checkerboard for sharpness
        for i in range(0, 480, 20):
            for j in range(0, 640, 20):
                if (i // 20 + j // 20) % 2 == 0:
                    image[i : i + 20, j : j + 20] = 150
                else:
                    image[i : i + 20, j : j + 20] = 80

        extractor = QualityExtractor(
            blur_threshold=50.0,
            brightness_low=50.0,
            brightness_high=200.0,
            contrast_threshold=20.0,
        )

        frame = Frame.from_array(image, frame_id=0, t_src_ns=0)
        obs = extractor.extract(frame)

        assert obs.signals["quality_gate"] == 1.0
        assert obs.metadata["is_sharp"] is True
        assert obs.metadata["is_well_lit"] is True
        assert obs.metadata["has_contrast"] is True

    def test_quality_score_range(self):
        """Test that quality scores are in valid range."""
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        extractor = QualityExtractor()

        frame = Frame.from_array(image, frame_id=0, t_src_ns=0)
        obs = extractor.extract(frame)

        assert 0.0 <= obs.signals["blur_quality"] <= 1.0
        assert 0.0 <= obs.signals["brightness_quality"] <= 1.0
        assert 0.0 <= obs.signals["contrast_quality"] <= 1.0
        assert 0.0 <= obs.signals["quality_score"] <= 1.0
        assert obs.signals["quality_gate"] in [0.0, 1.0]

    def test_grayscale_input(self):
        """Test extraction on grayscale image."""
        # Grayscale image
        image = np.random.randint(0, 256, (480, 640), dtype=np.uint8)

        extractor = QualityExtractor()

        frame = Frame.from_array(image, frame_id=0, t_src_ns=0)
        obs = extractor.extract(frame)

        assert obs is not None
        assert "blur_score" in obs.signals
        assert "brightness" in obs.signals
        assert "contrast" in obs.signals

    def test_frame_metadata(self):
        """Test that frame metadata is correctly passed through."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        extractor = QualityExtractor()

        frame = Frame.from_array(image, frame_id=42, t_src_ns=1_000_000_000)
        obs = extractor.extract(frame)

        assert obs.frame_id == 42
        assert obs.t_ns == 1_000_000_000
