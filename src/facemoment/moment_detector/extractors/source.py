"""Source preprocessing stage definition.

Defines the processing steps for video input pipeline.
This is not an actual extractor but provides step definitions
for visualization and documentation purposes.
"""

from typing import List

from facemoment.moment_detector.extractors.base import ProcessingStep


class SourceProcessor:
    """Video source preprocessing stage.

    Defines the input pipeline steps from video file to Frame objects.
    This class is used for DAG visualization, not actual processing
    (processing is handled by visualbase).

    The source pipeline:
    1. video_decode: Read and decode video frames
    2. frame_sampling: Sample frames at target FPS (skip frames)
    3. resize (optional): Resize to target resolution
    4. frame_create: Create Frame object with metadata
    """

    # Class-level step definitions for DAG visualization
    _STEPS = [
        ProcessingStep(
            name="video_decode",
            description="Decode video frames from file",
            backend="visualbase/OpenCV",
            input_type="Video file path",
            output_type="Raw BGR frames",
        ),
        ProcessingStep(
            name="frame_sampling",
            description="Sample frames at target FPS (skip intermediate frames)",
            backend="visualbase",
            input_type="Raw frames @ source FPS",
            output_type="Sampled frames @ target FPS",
            depends_on=["video_decode"],
        ),
        ProcessingStep(
            name="resize",
            description="Resize to target resolution (if specified)",
            backend="OpenCV",
            input_type="Frame (original size)",
            output_type="Frame (target size)",
            optional=True,
            depends_on=["frame_sampling"],
        ),
        ProcessingStep(
            name="frame_create",
            description="Create Frame object with frame_id, t_src_ns metadata",
            backend="visualbase",
            input_type="BGR image + metadata",
            output_type="Frame object",
            depends_on=["resize"],
        ),
    ]

    @property
    def name(self) -> str:
        return "source"

    @property
    def processing_steps(self) -> List[ProcessingStep]:
        return self._STEPS


class BackendPreprocessor:
    """ML backend internal preprocessing.

    Defines common preprocessing steps that ML backends perform internally.
    This is informational only - actual preprocessing is handled by each backend.
    """

    # Class-level step definitions
    _STEPS = [
        ProcessingStep(
            name="color_convert",
            description="Convert BGR to RGB (if required by model)",
            backend="OpenCV/NumPy",
            input_type="BGR image",
            output_type="RGB image",
            optional=True,
        ),
        ProcessingStep(
            name="model_resize",
            description="Resize to model input size (e.g., 640x640 for SCRFD)",
            backend="Backend-specific",
            input_type="Image (any size)",
            output_type="Image (model input size)",
            depends_on=["color_convert"],
        ),
        ProcessingStep(
            name="normalize",
            description="Normalize pixel values (mean/std or 0-1 range)",
            backend="Backend-specific",
            input_type="Image [0-255]",
            output_type="Tensor (normalized)",
            depends_on=["model_resize"],
        ),
        ProcessingStep(
            name="inference",
            description="Run model inference (GPU/CPU)",
            backend="ONNX/PyTorch",
            input_type="Normalized tensor",
            output_type="Raw model output",
            depends_on=["normalize"],
        ),
        ProcessingStep(
            name="postprocess",
            description="Convert to original image coordinates",
            backend="Backend-specific",
            input_type="Raw model output",
            output_type="Detections in original coords",
            depends_on=["inference"],
        ),
    ]

    @property
    def name(self) -> str:
        return "backend_preprocess"

    @property
    def processing_steps(self) -> List[ProcessingStep]:
        return self._STEPS
