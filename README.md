# facemoment

Face moment detection and highlight clip extraction.

## Quick Start

```python
import facemoment as fm

# Process a video (one-liner)
triggers = fm.run("video.mp4")
print(f"Found {len(triggers)} highlights")
```

### With Options

```python
# Adjust settings
triggers = fm.run(
    "video.mp4",
    fps=10,           # frames per second
    cooldown=3.0,     # seconds between triggers
)

# With callback
fm.run("video.mp4", on_trigger=lambda t: print(f"Trigger: {t.label}"))
```

### With Clip Extraction

```python
# Analyze and extract clips
result = fm.analyze("video.mp4", output_dir="./clips")
print(f"{len(result.triggers)} triggers, {result.clips_extracted} clips")
```

## Installation

```bash
# Basic
pip install facemoment

# With ML backends (recommended)
pip install facemoment[all]

# Or with uv
uv pip install -e ".[all]"
```

## API Reference

| Function | Description |
|----------|-------------|
| `fm.run(video)` | Process video, return triggers |
| `fm.analyze(video, output_dir)` | Process video with stats and clip extraction |

### Parameters

```python
fm.run(
    video,                    # Path to video file
    extractors=None,          # ["face", "pose", "gesture", "quality"] or None for auto
    fps=10,                   # Frames per second to process
    cooldown=2.0,             # Seconds between triggers
    use_ml=True,              # Use ML extractors (False for dummy)
    on_trigger=None,          # Callback function
)
```

---

## Advanced Usage

For complex use cases, use the class-based API:

```python
from facemoment import MomentDetector
from facemoment.moment_detector.extractors import FaceExtractor, PoseExtractor
from facemoment.moment_detector.fusion import HighlightFusion

detector = MomentDetector(
    extractors=[FaceExtractor(), PoseExtractor()],
    fusion=HighlightFusion(cooldown_sec=3.0),
    clip_output_dir="./clips",
)

clips = detector.process_file("video.mp4", fps=10)
```

## CLI Commands

```bash
# Process video
facemoment process video.mp4 -o ./clips

# Debug with visualization
facemoment debug video.mp4 -e face      # face only
facemoment debug video.mp4 -e face,pose # multiple
facemoment debug video.mp4 --no-ml      # dummy mode

# System info
facemoment info -v
```

## Architecture

Part of the Portrait981 system (A-B*-C architecture):

```
┌─────────────────────────────────────────────────────────────┐
│  fm.run("video.mp4")                                        │
│                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│  │   Source    │ ──► │  Extractors │ ──► │   Fusion    │   │
│  │  (frames)   │     │ face, pose  │     │ (triggers)  │   │
│  └─────────────┘     └─────────────┘     └─────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Extractors

| Extractor | Backend | Description |
|-----------|---------|-------------|
| `face` | InsightFace + HSEmotion | Face detection + expression |
| `pose` | YOLO-Pose | Body keypoints + gestures |
| `gesture` | MediaPipe | Hand gestures (V-sign, thumbsup) |
| `quality` | OpenCV | Blur, brightness, contrast |

## Trigger Types

| Trigger | Signal | Description |
|---------|--------|-------------|
| expression_spike | Adaptive EWMA | Happy emotion spike (sustained) |
| head_turn | Yaw velocity | Fast head rotation |
| hand_wave | Pose detection | Waving gesture |
| gesture_vsign | MediaPipe | V-sign hand pose |

## Dependencies

Core:
- Python >= 3.10
- NumPy, OpenCV
- visualbase, visualpath

ML (optional extras):
- `[face]`: InsightFace, HSEmotion, ONNX Runtime
- `[pose]`: Ultralytics (YOLO)
- `[gesture]`: MediaPipe

## Related Packages

- **visualbase**: Media streaming and clip extraction
- **visualpath**: Video analysis pipeline platform

## License

MIT
