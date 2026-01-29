"""Process module for A-B*-C architecture.

This module provides process wrappers for running extractors and fusion
as independent processes with IPC communication.

- ExtractorProcess: Wraps a BaseExtractor for standalone execution
- FusionProcess: Wraps a BaseFusion for standalone execution
"""

from facemoment.process.extractor import ExtractorProcess
from facemoment.process.fusion import FusionProcess

__all__ = [
    "ExtractorProcess",
    "FusionProcess",
]
