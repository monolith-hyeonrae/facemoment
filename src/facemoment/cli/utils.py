"""CLI utility functions and compatibility layer."""

import os
import sys


def suppress_qt_warnings():
    """Suppress Qt and OpenCV warnings for cleaner CLI output."""
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
    if os.environ.get("XDG_SESSION_TYPE") == "wayland":
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


class StderrFilter:
    """Filter stderr to suppress Qt/OpenCV warnings."""

    SUPPRESS_PATTERNS = (
        "QFontDatabase:",
        "Note that Qt no longer ships fonts",
        "XDG_SESSION_TYPE=wayland",
        "qt.qpa.",
    )

    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        if not any(p in text for p in self.SUPPRESS_PATTERNS):
            self._stream.write(text)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class VideoSourceInfo:
    """Video source metadata container for compatibility layer."""

    def __init__(self, fps: float, frame_count: int, width: int, height: int):
        self.fps = fps
        self.frame_count = frame_count
        self.width = width
        self.height = height


class LegacyVideoStream:
    """Fallback video stream using cv2.VideoCapture when visualbase API is incompatible."""

    def __init__(self, path: str, fps: float, resolution: tuple = None):
        import cv2
        from visualbase import Frame

        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {path}")

        self._video_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_skip = max(1, int(self._video_fps / fps)) if fps > 0 else 1
        self._resolution = resolution
        self._frame_id = 0
        self._Frame = Frame

        self.info = VideoSourceInfo(
            fps=self._video_fps,
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def __iter__(self):
        return self

    def __next__(self):
        import cv2

        while True:
            ret, image = self._cap.read()
            if not ret:
                raise StopIteration

            frame_id = self._frame_id
            self._frame_id += 1

            if frame_id % self._frame_skip != 0:
                continue

            if self._resolution:
                image = cv2.resize(image, self._resolution)

            t_ns = int(frame_id / self._video_fps * 1e9)
            return self._Frame.from_array(image, frame_id=frame_id, t_src_ns=t_ns)

    def disconnect(self):
        if self._cap:
            self._cap.release()
            self._cap = None


def create_video_stream(path: str, fps: float = 10.0, resolution: tuple = None):
    """Create visualbase video stream from file path.

    Falls back to cv2.VideoCapture if visualbase API is incompatible.

    Args:
        path: Path to the video file.
        fps: Target frame rate for processing.
        resolution: Optional (width, height) for resizing.

    Returns:
        Tuple of (vb, source, stream) where:
        - vb: VisualBase instance or LegacyVideoStream (call vb.disconnect() when done)
        - source: Object with metadata (fps, frame_count, width, height)
        - stream: Iterator[Frame] for processing
    """
    try:
        from visualbase import VisualBase, FileSource

        source = FileSource(path)
        source.open()

        required_attrs = ['fps', 'frame_count', 'width', 'height']
        for attr in required_attrs:
            if not hasattr(source, attr):
                raise AttributeError(f"FileSource missing '{attr}' attribute")

        vb = VisualBase()

        if not hasattr(vb, 'connect') or not hasattr(vb, 'get_stream'):
            raise AttributeError("VisualBase missing required methods")

        vb.connect(source)
        stream = vb.get_stream(fps=int(fps), resolution=resolution)
        return vb, source, stream

    except (ImportError, AttributeError, TypeError) as e:
        import logging
        logging.warning(f"visualbase API incompatible, using fallback: {e}")

        legacy = LegacyVideoStream(path, fps, resolution)
        return legacy, legacy.info, legacy


def check_ml_dependencies(module_name: str, require_expression: bool = False) -> bool:
    """Check if ML dependencies are available."""
    deps = {
        "face": ["insightface", "onnxruntime"],
        "face_expression": ["insightface", "onnxruntime", "feat"],
        "pose": ["ultralytics"],
    }

    if module_name == "face" and require_expression:
        module_name = "face_expression"
    missing = []
    for dep in deps.get(module_name, []):
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)

    if missing:
        print(f"Error: ML dependencies not installed for {module_name} extractor.")
        print(f"Missing: {', '.join(missing)}")
        print()
        print("Install with:")
        print("  uv sync --extra ml")
        print()
        print("Or install individually:")
        print(f"  pip install {' '.join(missing)}")
        return False
    return True


def setup_observability(trace_level: str, trace_output: str = None):
    """Configure observability based on CLI arguments.

    Args:
        trace_level: Trace level string ("off", "minimal", "normal", "verbose").
        trace_output: Optional path to output JSONL file.

    Returns:
        Tuple of (hub, file_sink) for cleanup. file_sink may be None.
    """
    from facemoment.observability import ObservabilityHub, TraceLevel, FileSink, ConsoleSink

    level_map = {
        "off": TraceLevel.OFF,
        "minimal": TraceLevel.MINIMAL,
        "normal": TraceLevel.NORMAL,
        "verbose": TraceLevel.VERBOSE,
    }

    level = level_map.get(trace_level, TraceLevel.OFF)
    hub = ObservabilityHub.get_instance()

    if level == TraceLevel.OFF:
        return hub, None

    sinks = []
    sinks.append(ConsoleSink())

    file_sink = None
    if trace_output:
        file_sink = FileSink(trace_output)
        sinks.append(file_sink)

    hub.configure(level=level, sinks=sinks)
    print(f"Observability: level={trace_level}", end="")
    if trace_output:
        print(f", output={trace_output}")
    else:
        print()

    return hub, file_sink


def cleanup_observability(hub, file_sink):
    """Clean up observability resources."""
    if hub is not None:
        hub.shutdown()
