"""Base fusion interface for combining observations (C module).

This module re-exports the base classes from visualpath.

Usage:
    Inherit from Module and implement process():
    - Return Observation with trigger info in signals:
        - signals["should_trigger"]: bool
        - signals["trigger_score"]: float
        - signals["trigger_reason"]: str
        - metadata["trigger"]: Trigger object
"""

# Re-export Module from visualpath
from visualpath.core.module import Module

# Re-export Trigger from visualbase
from visualbase import Trigger

# Import Observation from facemoment (preferred return type)
from facemoment.moment_detector.extractors.base import Observation

# Backwards compatibility alias
BaseFusion = Module

__all__ = ["Module", "Observation", "BaseFusion", "Trigger"]
