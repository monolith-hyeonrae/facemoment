"""Dummy extractor for testing."""

import random
from typing import Optional

from visualbase import Frame

from facemoment.moment_detector.extractors.base import (
    BaseExtractor,
    Observation,
    FaceObservation,
)


class DummyExtractor(BaseExtractor):
    """Dummy extractor that generates random observations for testing.

    This extractor simulates face detection with configurable behavior.
    Useful for testing the pipeline without real ML models.

    Args:
        name: Extractor name (default: "dummy").
        num_faces: Number of faces to simulate (default: 1).
        spike_probability: Probability of generating a high reaction (default: 0.1).
        seed: Random seed for reproducibility (default: None).

    Example:
        >>> extractor = DummyExtractor(num_faces=2, spike_probability=0.2)
        >>> obs = extractor.extract(frame)
        >>> print(f"Detected {len(obs.faces)} faces")
    """

    def __init__(
        self,
        name: str = "dummy",
        num_faces: int = 1,
        spike_probability: float = 0.1,
        seed: Optional[int] = None,
    ):
        self._name = name
        self._num_faces = num_faces
        self._spike_probability = spike_probability
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return self._name

    def extract(self, frame: Frame) -> Optional[Observation]:
        """Generate dummy observation with random values."""
        faces = []

        for i in range(self._num_faces):
            # Determine if this is a "spike" frame (high reaction)
            is_spike = self._rng.random() < self._spike_probability

            # Generate face position (centered, with slight variation)
            base_x = 0.3 + i * 0.2  # Spread faces horizontally
            x = base_x + self._rng.uniform(-0.05, 0.05)
            y = 0.3 + self._rng.uniform(-0.05, 0.05)
            w = 0.15 + self._rng.uniform(-0.02, 0.02)
            h = 0.2 + self._rng.uniform(-0.02, 0.02)

            # Calculate derived values
            area_ratio = w * h
            center_x, center_y = x + w / 2, y + h / 2
            center_distance = ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5

            # Expression: normally low, high on spike
            if is_spike:
                expression = self._rng.uniform(0.7, 1.0)
            else:
                expression = self._rng.uniform(0.1, 0.4)

            face = FaceObservation(
                face_id=i,
                confidence=self._rng.uniform(0.85, 0.99),
                bbox=(x, y, w, h),
                inside_frame=True,
                yaw=self._rng.uniform(-15, 15),
                pitch=self._rng.uniform(-10, 10),
                roll=self._rng.uniform(-5, 5),
                area_ratio=area_ratio,
                center_distance=center_distance,
                expression=expression,
                signals={
                    "smile": expression * self._rng.uniform(0.8, 1.0),
                    "attention": self._rng.uniform(0.6, 1.0),
                },
            )
            faces.append(face)

        # Aggregate signals
        max_expression = max(f.expression for f in faces) if faces else 0.0

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={
                "max_expression": max_expression,
                "face_count": len(faces),
            },
            faces=faces,
        )
