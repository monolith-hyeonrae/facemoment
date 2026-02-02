"""Pipeline module for facemoment A-B*-C-A distributed processing.

This module provides the orchestration layer for running the facemoment
pipeline in distributed mode, with each extractor potentially running
in its own virtual environment to avoid dependency conflicts.

Components:
- PipelineOrchestrator: Main orchestrator for A-B*-C-A workflow
- ExtractorConfig: Configuration for individual extractors
- FusionConfig: Configuration for the fusion module
- PipelineConfig: Complete pipeline configuration
- PipelineStats: Statistics from pipeline execution

Example:
    >>> from facemoment.pipeline import (
    ...     PipelineOrchestrator,
    ...     ExtractorConfig,
    ...     PipelineConfig,
    ... )
    >>>
    >>> # Simple usage with extractor configs
    >>> configs = [
    ...     ExtractorConfig(name="face", venv_path="/opt/venv-face"),
    ...     ExtractorConfig(name="pose", venv_path="/opt/venv-pose"),
    ... ]
    >>> orchestrator = PipelineOrchestrator(extractor_configs=configs)
    >>> clips = orchestrator.run("video.mp4", fps=10)
    >>>
    >>> # Using PipelineConfig
    >>> config = PipelineConfig.from_yaml("pipeline.yaml")
    >>> orchestrator = PipelineOrchestrator.from_config(config)
    >>> clips = orchestrator.run("video.mp4")
    >>>
    >>> # Using the default config helper
    >>> from facemoment.pipeline import create_default_config
    >>> config = create_default_config(
    ...     venv_face="/opt/venv-face",
    ...     venv_pose="/opt/venv-pose",
    ...     venv_gesture="/opt/venv-gesture",
    ... )
    >>> orchestrator = PipelineOrchestrator.from_config(config)
"""

from facemoment.pipeline.config import (
    ExtractorConfig,
    FusionConfig,
    PipelineConfig,
    create_default_config,
)
from facemoment.pipeline.orchestrator import (
    PipelineOrchestrator,
    PipelineStats,
)

# Re-export ClipResult for convenience
from visualbase import ClipResult

__all__ = [
    # Configuration
    "ExtractorConfig",
    "FusionConfig",
    "PipelineConfig",
    "create_default_config",
    # Orchestration
    "PipelineOrchestrator",
    "PipelineStats",
    # Results
    "ClipResult",
]
