"""Base fusion interface for combining observations (C module).

This module re-exports the base classes from visualpath.
"""

# Re-export from visualpath
from visualpath.core.fusion import BaseFusion, FusionResult

# Re-export Trigger from visualbase for backward compatibility
from visualbase import Trigger

# Import Observation from facemoment for type hints
from facemoment.moment_detector.extractors.base import Observation  # noqa: F401

__all__ = ["BaseFusion", "FusionResult", "Trigger"]
