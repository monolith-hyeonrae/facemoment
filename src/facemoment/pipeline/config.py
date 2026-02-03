"""Configuration classes for the facemoment pipeline.

This module provides configuration dataclasses for setting up
the A-B*-C-A distributed processing pipeline.

Example:
    >>> from facemoment.pipeline import ExtractorConfig, PipelineConfig
    >>> from visualpath.core import IsolationLevel
    >>>
    >>> config = PipelineConfig(
    ...     extractors=[
    ...         ExtractorConfig(name="face", venv_path="/opt/venv-face"),
    ...         ExtractorConfig(name="pose", venv_path="/opt/venv-pose"),
    ...         ExtractorConfig(name="quality", isolation=IsolationLevel.INLINE),
    ...     ],
    ...     clip_output_dir="./clips",
    ...     fps=10,
    ... )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from visualpath.core import IsolationLevel


@dataclass
class ExtractorConfig:
    """Configuration for a single extractor in the pipeline.

    Attributes:
        name: Entry point name of the extractor (e.g., "face", "pose", "gesture").
        venv_path: Path to virtual environment for VENV isolation.
            If provided and isolation is not set, defaults to VENV isolation.
        isolation: Isolation level for the extractor. Defaults to INLINE if
            no venv_path is provided, or VENV if venv_path is provided.
        kwargs: Additional keyword arguments passed to the extractor constructor.

    Example:
        >>> # Run face extractor in a separate venv
        >>> config = ExtractorConfig(
        ...     name="face",
        ...     venv_path="/opt/venvs/venv-face",
        ... )
        >>>
        >>> # Run quality extractor inline (same process)
        >>> config = ExtractorConfig(
        ...     name="quality",
        ...     isolation=IsolationLevel.INLINE,
        ... )
    """

    name: str
    venv_path: Optional[str] = None
    isolation: Optional[IsolationLevel] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set default isolation level based on venv_path."""
        if self.isolation is None:
            if self.venv_path:
                self.isolation = IsolationLevel.VENV
            else:
                self.isolation = IsolationLevel.INLINE

    @property
    def effective_isolation(self) -> IsolationLevel:
        """Get the effective isolation level."""
        return self.isolation or IsolationLevel.INLINE


@dataclass
class FusionConfig:
    """Configuration for the fusion module.

    Attributes:
        name: Entry point name of the fusion (e.g., "highlight").
        cooldown_sec: Cooldown between triggers in seconds.
        kwargs: Additional keyword arguments passed to the fusion constructor.

    Example:
        >>> config = FusionConfig(
        ...     name="highlight",
        ...     cooldown_sec=2.0,
        ...     kwargs={"head_turn_velocity_threshold": 30.0},
        ... )
    """

    name: str = "highlight"
    cooldown_sec: float = 2.0
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Complete configuration for the facemoment pipeline.

    Attributes:
        extractors: List of extractor configurations.
        fusion: Fusion module configuration.
        clip_output_dir: Directory for extracted clips.
        fps: Analysis frame rate.
        backend: Execution backend ("pathway" or "simple"). Default: "pathway".

    Example:
        >>> config = PipelineConfig(
        ...     extractors=[
        ...         ExtractorConfig(name="face", venv_path="/opt/venv-face"),
        ...         ExtractorConfig(name="pose", venv_path="/opt/venv-pose"),
        ...     ],
        ...     fusion=FusionConfig(cooldown_sec=2.0),
        ...     clip_output_dir="./clips",
        ...     fps=10,
        ...     backend="pathway",
        ... )
    """

    extractors: List[ExtractorConfig] = field(default_factory=list)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    clip_output_dir: str = "./clips"
    fps: int = 10
    backend: str = "pathway"  # "pathway" or "simple"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create PipelineConfig from a dictionary (e.g., loaded from YAML).

        Args:
            data: Dictionary with configuration data.

        Returns:
            PipelineConfig instance.

        Example:
            >>> import yaml
            >>> with open("pipeline.yaml") as f:
            ...     data = yaml.safe_load(f)
            >>> config = PipelineConfig.from_dict(data)
        """
        extractors = []
        for ext_data in data.get("extractors", []):
            isolation_str = ext_data.get("isolation")
            isolation = None
            if isolation_str:
                isolation = IsolationLevel.from_string(isolation_str)

            extractors.append(ExtractorConfig(
                name=ext_data["name"],
                venv_path=ext_data.get("venv_path"),
                isolation=isolation,
                kwargs=ext_data.get("kwargs", {}),
            ))

        fusion_data = data.get("fusion", {})
        fusion = FusionConfig(
            name=fusion_data.get("name", "highlight"),
            cooldown_sec=fusion_data.get("cooldown_sec", 2.0),
            kwargs=fusion_data.get("kwargs", {}),
        )

        return cls(
            extractors=extractors,
            fusion=fusion,
            clip_output_dir=data.get("clip_output_dir", data.get("output", {}).get("clip_dir", "./clips")),
            fps=data.get("fps", data.get("output", {}).get("fps", 10)),
            backend=data.get("backend", "pathway"),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """Load PipelineConfig from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            PipelineConfig instance.

        Raises:
            ImportError: If PyYAML is not installed.
            FileNotFoundError: If the file doesn't exist.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config support. "
                "Install it with: pip install pyyaml"
            )

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the config.
        """
        return {
            "extractors": [
                {
                    "name": ext.name,
                    "venv_path": ext.venv_path,
                    "isolation": ext.effective_isolation.name.lower(),
                    "kwargs": ext.kwargs,
                }
                for ext in self.extractors
            ],
            "fusion": {
                "name": self.fusion.name,
                "cooldown_sec": self.fusion.cooldown_sec,
                "kwargs": self.fusion.kwargs,
            },
            "clip_output_dir": self.clip_output_dir,
            "fps": self.fps,
            "backend": self.backend,
        }


def create_default_config(
    venv_face: Optional[str] = None,
    venv_pose: Optional[str] = None,
    venv_gesture: Optional[str] = None,
    clip_output_dir: str = "./clips",
    fps: int = 10,
    cooldown_sec: float = 2.0,
    backend: str = "pathway",
) -> PipelineConfig:
    """Create a default pipeline configuration.

    Creates a configuration with face, pose, and optionally gesture
    extractors based on provided venv paths.

    Args:
        venv_face: Path to venv for face extractor. If None, runs inline.
        venv_pose: Path to venv for pose extractor. If None, runs inline.
        venv_gesture: Path to venv for gesture extractor. If provided, adds gesture extractor.
        clip_output_dir: Directory for extracted clips.
        fps: Analysis frame rate.
        cooldown_sec: Cooldown between triggers.

    Returns:
        PipelineConfig with the specified extractors.

    Example:
        >>> config = create_default_config(
        ...     venv_face="/opt/venvs/venv-face",
        ...     venv_pose="/opt/venvs/venv-pose",
        ... )
    """
    extractors = []

    # Face extractor
    extractors.append(ExtractorConfig(
        name="face",
        venv_path=venv_face,
    ))

    # Pose extractor
    extractors.append(ExtractorConfig(
        name="pose",
        venv_path=venv_pose,
    ))

    # Gesture extractor (if venv provided)
    if venv_gesture:
        extractors.append(ExtractorConfig(
            name="gesture",
            venv_path=venv_gesture,
        ))

    # Quality extractor (always inline - lightweight)
    extractors.append(ExtractorConfig(
        name="quality",
        isolation=IsolationLevel.INLINE,
    ))

    return PipelineConfig(
        extractors=extractors,
        fusion=FusionConfig(cooldown_sec=cooldown_sec),
        clip_output_dir=clip_output_dir,
        fps=fps,
        backend=backend,
    )
