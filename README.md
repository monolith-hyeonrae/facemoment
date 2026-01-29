# facemoment

Face moment detection and highlight clip extraction for 981park.

## Overview

Part of the Portrait981 system (A-B*-C architecture):

```
Camera → [A: Ingest] → [B*: Feature Extractors] → [C: Fusion] → Highlights
              ↓                    ↓                    ↓
          Ring Buffer         Observations          Triggers
              ↓                                         ↓
           4K Clips ←─────────────────────────────────────
```

This package implements the B* (Feature Extractors) and C (Fusion Engine) components.

## Features

- **Face Extraction**: InsightFace SCRFD for detection, HSEmotion for expression analysis
- **Pose Extraction**: YOLO-Pose for body keypoints and gesture detection
- **Quality Analysis**: Blur detection, brightness assessment
- **Highlight Fusion**: Multi-signal scoring for expression spikes, head turns, hand waves
- **Clip Extraction**: Integrates with visualbase for trigger-based clip saving

## Installation

```bash
pip install facemoment
```

For ML features (recommended):

```bash
pip install facemoment[ml]
```

Or with uv:

```bash
uv add facemoment --extra ml
```

## Quick Start

```python
from facemoment import MomentDetector
from facemoment.moment_detector.extractors import FaceExtractor, PoseExtractor, QualityExtractor
from facemoment.moment_detector.fusion import HighlightFusion

# Create detector with extractors
detector = MomentDetector(
    extractors=[
        FaceExtractor(),
        PoseExtractor(),
        QualityExtractor(),
    ],
    fusion=HighlightFusion(),
    clip_output_dir="./clips",
)

# Process video file
clips = detector.process_file("video.mp4", fps=10)

for clip in clips:
    if clip.success:
        print(f"Saved: {clip.output_path} ({clip.duration_sec:.2f}s)")
```

## CLI Commands

```bash
# Process video and extract highlight clips
facemoment process video.mp4 --fps 10 --output-dir ./clips

# Debug face extractor
facemoment debug-face video.mp4 --fps 10

# Debug pose extractor
facemoment debug-pose video.mp4 --fps 10

# Benchmark extractor performance
facemoment benchmark video.mp4 --frames 100

# Run interactive debug session
facemoment debug video.mp4 --fps 10
```

## Architecture

```
facemoment/
├── moment_detector/
│   ├── detector.py        # MomentDetector main class
│   ├── extractors/
│   │   ├── base.py        # BaseExtractor interface
│   │   ├── face.py        # FaceExtractor
│   │   ├── pose.py        # PoseExtractor
│   │   ├── quality.py     # QualityExtractor
│   │   └── backends/      # ML backend implementations
│   │       ├── face_backends.py   # InsightFace, HSEmotion, PyFeat
│   │       └── pose_backends.py   # YOLO-Pose
│   └── fusion/
│       ├── base.py        # BaseFusion interface
│       └── highlight.py   # HighlightFusion
├── tools/
│   └── visualizer.py      # Debug visualization
└── cli.py                 # Command-line interface
```

## Dependencies

Core:
- Python >= 3.10
- NumPy >= 1.24.0
- visualbase (for clip extraction)

ML (optional):
- InsightFace >= 0.7.3 (face detection)
- ONNX Runtime GPU >= 1.16.0
- Ultralytics >= 8.0.0 (pose detection)
- hsemotion-onnx >= 0.3 (expression analysis, fast)
- Py-Feat >= 0.6.0 (expression + AU analysis, slower)

## Trigger Types

| Trigger | Signal | Threshold |
|---------|--------|-----------|
| Expression Spike | Z-score vs EWMA baseline | > 2.0 |
| Head Turn | Yaw velocity | > 30 deg/sec |
| Hand Wave | Pose gesture detection | > 0.7 |

## Related Packages

- **visualbase**: Media streaming and clip extraction platform
- **appearance-vault**: (Future) Member-based clip storage
- **reportrait**: (Future) AI image/video reinterpretation

## License

MIT
