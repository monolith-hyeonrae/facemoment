"""Process module for A-B*-C architecture.

This module provides process wrappers for running extractors and fusion
as independent processes with IPC communication.

- ExtractorProcess: Wraps a BaseExtractor for standalone IPC execution
- FusionProcess: Wraps a BaseFusion for standalone IPC execution
- ExtractorOrchestrator: Thread-parallel extractor execution for Library mode
"""

from facemoment.process.extractor import ExtractorProcess
from facemoment.process.fusion import FusionProcess
from facemoment.process.orchestrator import ExtractorOrchestrator

__all__ = [
    "ExtractorProcess",
    "FusionProcess",
    "ExtractorOrchestrator",
]
