from facemoment.moment_detector.fusion.base import (
    BaseFusion,
    Module,
    Observation,
    Trigger,
)
from facemoment.moment_detector.fusion.dummy import DummyFusion
from facemoment.moment_detector.fusion.highlight import HighlightFusion

__all__ = [
    "Module",
    "Observation",
    "BaseFusion",
    "Trigger",
    "DummyFusion",
    "HighlightFusion",
]
