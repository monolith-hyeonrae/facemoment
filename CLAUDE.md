# FaceMoment - Claude Session Context

> 최종 업데이트: 2026-01-30
> 상태: **Phase 11 완료** (150 tests) - 의존성 분리 구조

## 프로젝트 역할

981파크 Portrait981 파이프라인의 **얼굴/장면 분석 모듈**:
- 얼굴 감지, 표정 분석, 포즈 추정
- 하이라이트 순간 감지 (트리거)
- 고카트 시나리오 특화 기능 (Phase 9)
- Observability 시스템 (Phase 10)
- **의존성 분리 구조** (Phase 11) - Worker별 독립 venv 지원

## 아키텍처

```
facemoment/
├── cli/                   # CLI 모듈 (리팩토링됨)
│   ├── __init__.py        # main(), argparse
│   ├── utils.py           # 공통 유틸리티, visualbase 호환성 레이어
│   └── commands/          # 명령어 핸들러
│       ├── info.py        # 시스템 정보
│       ├── debug.py       # 통합 디버그 (--profile 지원)
│       ├── process.py     # 클립 추출
│       └── benchmark.py   # 성능 측정
├── moment_detector/
│   ├── extractors/        # FaceExtractor, PoseExtractor, GestureExtractor
│   │   │                  # (lazy import로 독립 로딩 가능)
│   │   └── backends/      # InsightFace, HSEmotion, YOLOPose, MediaPipe
│   ├── fusion/            # HighlightFusion (트리거 결정)
│   ├── visualize.py       # 시각화 (DebugVisualizer, 타이밍 오버레이)
│   └── detector.py        # MomentDetector (오케스트레이터)
├── observability/         # Observability 시스템
│   ├── __init__.py        # ObservabilityHub, TraceLevel
│   ├── records.py         # TraceRecord 타입들
│   └── sinks.py           # FileSink, ConsoleSink, MemorySink
└── process/               # A-B*-C 분산 처리 (portrait981에서 사용)
    ├── extractor.py       # ExtractorProcess
    ├── fusion.py          # FusionProcess
    └── orchestrator.py    # ExtractorOrchestrator
```

## 의존성 구조 (분리됨)

Python ML 생태계의 의존성 충돌을 해결하기 위해 **worker별 독립 venv**를 지원합니다.

### Optional Dependencies

| Extra | 용도 | 주요 의존성 |
|-------|------|-------------|
| (base) | 모든 환경 | numpy, opencv-python, visualbase |
| `face` | Face worker | insightface, hsemotion-onnx, onnxruntime-gpu |
| `face-full` | Face worker (PyFeat) | insightface, py-feat |
| `pose` | Pose worker | ultralytics (torch) |
| `gesture` | Gesture worker | mediapipe |
| `cli` | CLI 도구 | pyqt6 |
| `zmq` | IPC 통신 | pyzmq |
| `all` | 개발/테스트 | face + pose + gesture + cli |

### Worker별 venv 생성

```bash
# Face worker 전용 (의존성 충돌 없음)
uv venv venv-face
source venv-face/bin/activate
uv pip install -e ".[face,zmq]"

# Pose worker 전용
uv venv venv-pose
source venv-pose/bin/activate
uv pip install -e ".[pose,zmq]"

# Gesture worker 전용
uv venv venv-gesture
source venv-gesture/bin/activate
uv pip install -e ".[gesture,zmq]"

# 개발/테스트 (전체 설치 - 충돌 가능성 있음)
uv sync --extra all --extra dev
```

### Import 패턴 (Lazy Loading)

```python
# 항상 가능 (base 의존성만)
from facemoment.moment_detector.extractors import BaseExtractor, Observation

# 해당 extra가 설치된 환경에서만 import
from facemoment.moment_detector.extractors.face import FaceExtractor      # [face]
from facemoment.moment_detector.extractors.pose import PoseExtractor      # [pose]
from facemoment.moment_detector.extractors.gesture import GestureExtractor # [gesture]
```

## 파이프라인 구조

```
Video Source (visualbase)
     │
     ▼
┌─────────────────────────────────────────┐
│              Extractors                 │
│  ┌─────────┐ ┌─────────┐ ┌───────────┐  │
│  │  Face   │ │  Pose   │ │  Gesture  │  │
│  │ [venv1] │ │ [venv2] │ │  [venv3]  │  │
│  └────┬────┘ └────┬────┘ └─────┬─────┘  │
│       └───────────┴────────────┘        │
│                   │ ZMQ/IPC             │
│                   ▼                     │
│  ┌─────────────────────────────────┐    │
│  │       HighlightFusion           │    │
│  │  Gate Check → Signal Analysis   │    │
│  │  → Trigger Decision             │    │
│  └─────────────┬───────────────────┘    │
└────────────────┼────────────────────────┘
                 │
                 ▼
           Trigger Event → Clip Extraction
```

## CLI 명령어

```bash
# 시스템 정보 확인
facemoment info                          # extractor/backend 상태, 파이프라인 구조
facemoment info -v                       # + GPU/ONNX 정보

# 디버그 (통합 명령어)
facemoment debug video.mp4               # 모든 extractor
facemoment debug video.mp4 -e raw        # 원본 비디오 프리뷰 (분석 없음)
facemoment debug video.mp4 -e face       # face만
facemoment debug video.mp4 -e pose       # pose만
facemoment debug video.mp4 -e face,pose  # 복수 선택
facemoment debug video.mp4 -e gesture    # gesture만
facemoment debug video.mp4 --no-ml       # dummy 모드 (ML 없이)
facemoment debug video.mp4 -o out.mp4    # 파일로 저장
facemoment debug video.mp4 -e face --profile  # 성능 프로파일링

# 클립 추출
facemoment process video.mp4 -o ./clips
facemoment process video.mp4 --gokart    # 고카트 모드
facemoment process video.mp4 --trace verbose --trace-output trace.jsonl

# 벤치마크
facemoment benchmark video.mp4 --frames 100
```

### --profile 모드

FaceExtractor 성능 프로파일링:

```bash
facemoment debug video.mp4 -e face --profile
```

출력:
```
Backends:
  Detection   : InsightFaceSCRFD [CUDA]
  Expression  : HSEmotionBackend
--------------------------------------------------
Frame 1: detect=42.3ms, expression=28.1ms, total=71.5ms
Frame 2: detect=38.7ms, expression=31.2ms, total=70.8ms
```

- 화면에 타이밍 오버레이 표시
- 색상 코딩: 녹색(빠름) / 노랑(보통) / 빨강(느림)

## Extractor 백엔드

| Extractor | Backend | 설명 | Extra |
|-----------|---------|------|-------|
| FaceExtractor | InsightFace SCRFD | 얼굴 감지 | `face` |
| FaceExtractor | HSEmotion | 표정 분석 (fast, ~30ms) | `face` |
| FaceExtractor | PyFeat | 표정 분석 (accurate, ~2000ms) | `face-full` |
| PoseExtractor | YOLO-Pose | 포즈 추정 | `pose` |
| GestureExtractor | MediaPipe Hands | 손/제스처 감지 | `gesture` |
| QualityExtractor | OpenCV | 블러/밝기/대비 | (base) |

## 트리거 유형

| 트리거 | 소스 | 설명 |
|--------|------|------|
| expression_spike | FaceExtractor | 표정 급변 |
| head_turn | FaceExtractor | 빠른 머리 회전 |
| hand_wave | PoseExtractor | 손 흔들기 |
| camera_gaze | HighlightFusion | 카메라 응시 (gokart) |
| passenger_interaction | HighlightFusion | 동승자 상호작용 (gokart) |
| gesture_vsign | GestureExtractor | V사인 (gokart) |
| gesture_thumbsup | GestureExtractor | 엄지척 (gokart) |

## Observability 시스템

### Trace Levels

| 레벨 | 용도 | 오버헤드 |
|------|------|----------|
| OFF | 프로덕션 기본 | 0% |
| MINIMAL | Trigger만 로깅 | <1% |
| NORMAL | 프레임 요약 + Gate 전환 | ~5% |
| VERBOSE | 모든 Signal + 타이밍 | ~15% |

### 사용법

```bash
facemoment process video.mp4 --trace normal
facemoment process video.mp4 --trace verbose --trace-output trace.jsonl

# 분석
cat trace.jsonl | jq 'select(.record_type=="trigger_fire")'
```

## 핵심 파일

| 파일 | 역할 |
|------|------|
| `cli/__init__.py` | CLI 메인, argparse (--profile 포함) |
| `cli/commands/debug.py` | 통합 debug 명령어, 프로파일링 로직 |
| `cli/utils.py` | 공통 유틸리티, visualbase 호환성 |
| `moment_detector/detector.py` | MomentDetector |
| `moment_detector/fusion/highlight.py` | HighlightFusion |
| `moment_detector/extractors/__init__.py` | Lazy import 패턴 |
| `moment_detector/extractors/face.py` | FaceExtractor (컴포넌트별 타이밍) |
| `moment_detector/visualize.py` | DebugVisualizer, 타이밍 오버레이 |
| `observability/__init__.py` | ObservabilityHub |

## visualbase 연동

```python
# CLI에서 visualbase 호환성 레이어 사용
from facemoment.cli.utils import create_video_stream

vb, source, stream = create_video_stream("video.mp4", fps=10)
try:
    for frame in stream:
        # frame.frame_id, frame.t_src_ns, frame.data
        process(frame)
finally:
    vb.disconnect()
```

- visualbase API 변경 시 자동 fallback (cv2.VideoCapture)
- FileSource 속성: fps, frame_count, width, height

## 테스트

```bash
uv sync --extra all --extra dev
uv run pytest tests/ -v            # 150 tests

# 특정 테스트
uv run pytest tests/test_gesture_extractor.py -v
uv run pytest tests/test_highlight_fusion.py -v
uv run pytest tests/test_observability.py -v
```

## 관련 패키지

- **visualbase**: 비디오 소스, 클립 추출, Frame 타입
- **portrait981**: 프로덕션 orchestrator (A-B*-C 분산 처리)

## 문서

- `docs/architecture-vision.md`: 플러그인 생태계 비전, 마이그레이션 로드맵
- `docs/phase-11-summary.md`: Phase 11 작업 요약 (의존성 분리)
- `docs/problems-and-solutions.md`: 알고리즘 문서 (EWMA, 히스테리시스 등)
- `docs/stream-synchronization.md`: 스트림 동기화 아키텍처

## 비전: 플러그인 생태계

현재 facemoment는 플랫폼 로직과 분석 로직이 혼재되어 있습니다. 장기적으로 3계층 구조로 분리할 계획입니다:

```
┌─────────────────────────────────────────────────────────┐
│  visualbase (미디어 소스)                                │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│  analysiscore (플랫폼 - 신규)                            │
│  - BaseExtractor, Observation 인터페이스                 │
│  - Plugin discovery, Worker orchestration               │
│  - Fusion framework, Observability                      │
└─────────────────────────────────────────────────────────┘
                           │
    ┌──────────────────────┼──────────────────────┐
    ▼                      ▼                      ▼
┌──────────┐        ┌──────────┐          ┌──────────┐
│facemoment│        │ plugin-A │          │ plugin-B │
│ (plugin) │        │  - OCR   │          │ - Scene  │
└──────────┘        └──────────┘          └──────────┘
```

상세 내용: [docs/architecture-vision.md](docs/architecture-vision.md)

## 다음 작업 우선순위

### 단기 (Phase 11 후속)
1. portrait981에서 worker별 독립 venv 실행 구조 적용
2. 실제 고카트 영상으로 테스트
3. 트리거 threshold 튜닝

### 중기 (Phase 12-13)
4. analysiscore 패키지 분리 (플랫폼 로직 추출)
5. 플러그인 discovery 구현 (entry_points 기반)

### 장기 (Phase 14-15)
6. facemoment 플러그인화
7. portrait981 analysiscore 전환
