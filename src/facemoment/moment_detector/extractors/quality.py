"""Quality extractor for image quality assessment."""

from typing import Optional
import logging

import cv2
import numpy as np

from visualbase import Frame

from facemoment.moment_detector.extractors.base import (
    Module,
    Observation,
)

logger = logging.getLogger(__name__)


class QualityExtractor(Module):
    """Extractor for image quality signals.

    Analyzes frames for quality metrics that affect composition:
    - Blur detection using Laplacian variance
    - Brightness analysis
    - Contrast analysis

    These signals are used by the fusion module to gate triggers,
    ensuring highlights are only captured from high-quality frames.

    Args:
        blur_threshold: Laplacian variance below this is considered blurry (default: 100).
        brightness_low: Brightness below this is considered too dark (default: 50).
        brightness_high: Brightness above this is considered too bright (default: 200).
        contrast_threshold: Contrast below this is considered low (default: 40).

    Example:
        >>> extractor = QualityExtractor()
        >>> obs = extractor.process(frame)
        >>> if obs.signals["quality_gate"] > 0.5:
        ...     print("Frame quality is acceptable")
    """

    def __init__(
        self,
        blur_threshold: float = 100.0,
        brightness_low: float = 50.0,
        brightness_high: float = 200.0,
        contrast_threshold: float = 40.0,
    ):
        self._blur_threshold = blur_threshold
        self._brightness_low = brightness_low
        self._brightness_high = brightness_high
        self._contrast_threshold = contrast_threshold

    @property
    def name(self) -> str:
        return "quality"

    def process(self, frame: Frame, deps=None) -> Optional[Observation]:
        """Extract quality signals from a frame.

        Args:
            frame: Input frame to analyze.
            deps: Not used (no dependencies).

        Returns:
            Observation with quality signals.
        """
        image = frame.data

        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Compute quality metrics
        blur_score = self._compute_blur_score(gray)
        brightness = self._compute_brightness(gray)
        contrast = self._compute_contrast(gray)

        # Normalize scores to [0, 1]
        blur_quality = min(1.0, blur_score / self._blur_threshold)
        brightness_quality = self._brightness_quality(brightness)
        contrast_quality = min(1.0, contrast / self._contrast_threshold)

        # Overall quality gate (all conditions must be met)
        quality_gate = min(blur_quality, brightness_quality, contrast_quality)

        # Binary gate decision
        is_sharp = blur_score >= self._blur_threshold
        is_well_lit = self._brightness_low <= brightness <= self._brightness_high
        has_contrast = contrast >= self._contrast_threshold
        gate_open = is_sharp and is_well_lit and has_contrast

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={
                # Raw measurements
                "blur_score": blur_score,
                "brightness": brightness,
                "contrast": contrast,
                # Normalized quality scores [0, 1]
                "blur_quality": blur_quality,
                "brightness_quality": brightness_quality,
                "contrast_quality": contrast_quality,
                # Overall quality
                "quality_score": quality_gate,
                "quality_gate": 1.0 if gate_open else 0.0,
            },
            metadata={
                "is_sharp": is_sharp,
                "is_well_lit": is_well_lit,
                "has_contrast": has_contrast,
            },
        )

    def _compute_blur_score(self, gray: np.ndarray) -> float:
        """Compute blur score using Laplacian variance.

        Higher values indicate sharper images.

        Args:
            gray: Grayscale image.

        Returns:
            Laplacian variance (higher = sharper).
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def _compute_brightness(self, gray: np.ndarray) -> float:
        """Compute mean brightness.

        Args:
            gray: Grayscale image.

        Returns:
            Mean pixel value [0, 255].
        """
        return float(np.mean(gray))

    def _compute_contrast(self, gray: np.ndarray) -> float:
        """Compute contrast as standard deviation.

        Args:
            gray: Grayscale image.

        Returns:
            Standard deviation of pixel values.
        """
        return float(np.std(gray))

    def _brightness_quality(self, brightness: float) -> float:
        """Convert brightness to quality score.

        Returns 1.0 when brightness is in optimal range,
        decreasing towards 0 at extremes.

        Args:
            brightness: Mean brightness [0, 255].

        Returns:
            Quality score [0, 1].
        """
        optimal_low = self._brightness_low
        optimal_high = self._brightness_high
        optimal_mid = (optimal_low + optimal_high) / 2
        optimal_range = (optimal_high - optimal_low) / 2

        if optimal_low <= brightness <= optimal_high:
            return 1.0

        if brightness < optimal_low:
            # Too dark
            return max(0.0, brightness / optimal_low)
        else:
            # Too bright
            return max(0.0, 1.0 - (brightness - optimal_high) / (255 - optimal_high))
