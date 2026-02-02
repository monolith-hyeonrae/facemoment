# FaceMoment - Claude Session Context

> 최종 업데이트: 2026-02-02
> 상태: **Phase 16 완료** - 시각화 개선 (탑승자 구분, 상반신 포즈)

## 프로젝트 역할

981파크 Portrait981 파이프라인의 **얼굴/장면 분석 앱**:
- visualpath 프레임워크를 사용하여 구현
- 얼굴 감지, 표정 분석, 포즈 추정
- 하이라이트 순간 감지 (트리거)
- **on_trigger → 클립 저장 (비즈니스 로직 포함)**
- GR차량 시나리오 특화 기능 (Phase 9)
- Observability 시스템 (Phase 10)
- 의존성 분리 구조 (Phase 11) - Worker별 독립 venv 지원
- visualpath 플랫폼 분리 (Phase 12) - 플랫폼 로직을 visualpath로 분리
- IPC 프로세스 이동 (Phase 13) - ExtractorProcess/FusionProcess를 visualpath로 이동
- 독립 앱 (Phase 14) - PipelineOrchestrator로 완전한 A-B*-C-A 파이프라인 제공
- 의존성 기반 Extractor (Phase 15) - depends/deps로 extractor 간 데이터 전달
- **시각화 개선** (Phase 16) - 탑승자 구분 색상, 상반신 포즈 스켈레톤

## 아키텍처 위치

```
┌─────────────────────────────────────────────────────────┐
│  범용 레이어                                             │
│  visualbase → visualpath                                │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  981파크 특화 레이어                                     │
│  ┌─────────────┐      ┌─────────────┐                   │
│  │ facemoment  │ ───→ │ portrait981 │                   │
│  │ (분석 앱)   │      │ (통합 앱)   │                   │
│  └─────────────┘      └─────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

## 아키텍처

```
facemoment/
├── cli/                   # CLI 모듈 (리팩토링됨)
│   ├── __init__.py        # main(), argparse (--distributed 포함)
│   ├── utils.py           # 공통 유틸리티, visualbase 호환성 레이어
│   └── commands/          # 명령어 핸들러
│       ├── info.py        # 시스템 정보
│       ├── debug.py       # 통합 디버그 (--profile 지원)
│       ├── process.py     # 클립 추출 (Library/Distributed 모드)
│       └── benchmark.py   # 성능 측정
├── pipeline/              # A-B*-C-A 파이프라인 오케스트레이션 (Phase 14)
│   ├── __init__.py        # exports
│   ├── config.py          # ExtractorConfig, PipelineConfig
│   └── orchestrator.py    # PipelineOrchestrator
├── moment_detector/
│   ├── extractors/        # FaceExtractor, PoseExtractor, GestureExtractor
│   │   │                  # (lazy import로 독립 로딩 가능)
│   │   └── backends/      # InsightFace, HSEmotion, YOLOPose, MediaPipe
│   ├── fusion/            # HighlightFusion (트리거 결정)
│   ├── visualize.py       # 시각화 (DebugVisualizer, 타이밍 오버레이)
│   └── detector.py        # MomentDetector (Library 모드용)
├── observability/         # Observability 시스템 (visualpath 확장)
│   ├── __init__.py        # re-export from visualpath + facemoment sinks
│   ├── records.py         # 도메인 특화 TraceRecord (TriggerFireRecord 등)
│   └── sinks.py           # 확장 Sink (MemorySink, ConsoleSink with domain formatting)
└── process/               # A-B*-C 분산 처리 (Phase 13: visualpath로 이동)
    ├── __init__.py        # re-export from visualpath + 팩토리 함수
    └── mappers.py         # FacemomentMapper (Observation ↔ OBS 변환)
```

## A-B*-C-A 파이프라인 (Phase 14)

```
┌─────────────────────────────────────────────────────────────┐
│  facemoment process video.mp4 --distributed                 │
│                                                             │
│  A: Video Input (visualbase)                                │
│       │                                                     │
│       │ Frame                                               │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ B* Extractors (VenvWorker/InlineWorker)             │    │
│  │  ┌─────────┐  ┌─────────┐  ┌───────────┐           │    │
│  │  │  face   │  │  pose   │  │  gesture  │           │    │
│  │  │[venv-1] │  │[venv-2] │  │ [venv-3]  │           │    │
│  │  └────┬────┘  └────┬────┘  └─────┬─────┘           │    │
│  └───────┼────────────┼─────────────┼──────────────────┘    │
│          └────────────┴─────────────┘                       │
│                       │ Observations                        │
│                       ▼                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ C: HighlightFusion                                  │    │
│  │   - Gate check → Trigger detection                 │    │
│  └────────────────────────┬────────────────────────────┘    │
│                           │ Trigger                         │
│                           ▼                                 │
│  A: Clip Output (visualbase.trigger())                      │
│       └─→ clips/highlight_001.mp4                           │
└─────────────────────────────────────────────────────────────┘
```

### 사용법

```python
# Python API
from facemoment.pipeline import (
    PipelineOrchestrator,
    ExtractorConfig,
    PipelineConfig,
    create_default_config,
)

# 방법 1: 간단한 설정
configs = [
    ExtractorConfig(name="face", venv_path="/opt/venv-face"),
    ExtractorConfig(name="pose", venv_path="/opt/venv-pose"),
    ExtractorConfig(name="quality"),  # inline
]
orchestrator = PipelineOrchestrator(extractor_configs=configs)
clips = orchestrator.run("video.mp4", fps=10)

# 방법 2: PipelineConfig 사용
config = create_default_config(
    venv_face="/opt/venv-face",
    venv_pose="/opt/venv-pose",
    gokart_mode=True,
)
orchestrator = PipelineOrchestrator.from_config(config)
clips = orchestrator.run("video.mp4")

# 방법 3: YAML 설정 파일
config = PipelineConfig.from_yaml("pipeline.yaml")
orchestrator = PipelineOrchestrator.from_config(config)
```

```bash
# CLI
# Library 모드 (기존)
facemoment process video.mp4 -o ./clips

# Distributed 모드 (Phase 14)
facemoment process video.mp4 --distributed
facemoment process video.mp4 --venv-face /opt/venv-face
facemoment process video.mp4 --venv-face /opt/venv-face --venv-pose /opt/venv-pose
facemoment process video.mp4 --config pipeline.yaml
```

### 설정 파일 (pipeline.yaml)

```yaml
extractors:
  - name: face
    venv_path: /opt/venvs/venv-face
    isolation: venv
  - name: pose
    venv_path: /opt/venvs/venv-pose
    isolation: venv
  - name: quality
    isolation: inline

fusion:
  name: highlight
  cooldown_sec: 2.0

clip_output_dir: ./clips
fps: 10
gokart_mode: false
```

## High-Level API (Phase 15)

간단한 사용을 위한 고수준 API:

```python
import facemoment as fm

# 간단한 사용
result = fm.run("video.mp4")
print(f"Found {len(result.triggers)} highlights")

# 옵션 지정
result = fm.run("video.mp4", fps=10, cooldown=3.0, output_dir="./clips")

# 사용 가능한 extractor/fusion 확인
print(fm.EXTRACTORS.keys())  # ["face", "pose", "gesture", "quality", "dummy"]
print(fm.FUSIONS.keys())     # ["highlight", "dummy"]
```

### 설정 상수

```python
# facemoment/main.py 상단에 정의
DEFAULT_FPS = 10
DEFAULT_COOLDOWN = 2.0

EXTRACTORS = {
    "face": FaceExtractor,
    "pose": PoseExtractor,
    "gesture": GestureExtractor,
    "quality": QualityExtractor,
    "dummy": DummyExtractor,
}

FUSIONS = {
    "highlight": HighlightFusion,
    "dummy": DummyFusion,
}
```

## 의존성 기반 Extractor 아키텍처 (Phase 15)

Extractor 간 데이터 전달을 위한 `depends`/`deps` 시스템:

### 의존성 선언

```python
class ExpressionExtractor(BaseExtractor):
    depends = ["face_detect"]  # 이 extractor가 의존하는 extractor 이름

    def extract(self, frame, deps=None):
        # deps에서 의존 extractor의 결과 접근
        face_obs = deps["face_detect"] if deps else None
        face_data: FaceDetectOutput = face_obs.data

        # 타입 안전한 접근
        faces = face_data.faces
        detected_faces = face_data.detected_faces
```

### 분리된 Extractor 구조

| Extractor | depends | 역할 |
|-----------|---------|------|
| `FaceDetectionExtractor` | - | 얼굴 검출 (bbox, head pose) |
| `ExpressionExtractor` | `face_detect` | 표정 분석 (emotions) |
| `FaceClassifierExtractor` | `face_detect` | 역할 분류 (main/passenger/transient/noise) |
| `FaceExtractor` | - | 복합 (검출 + 표정, 하위 호환) |

### 타입 안전한 Output

```python
from facemoment.moment_detector.extractors import (
    FaceDetectOutput,
    ExpressionOutput,
    FaceClassifierOutput,
)

# 타입 힌트로 IDE 자동완성 지원
face_data: FaceDetectOutput = face_obs.data
faces = face_data.faces              # List[FaceObservation]
detected_faces = face_data.detected_faces  # List[DetectedFace]
image_size = face_data.image_size    # tuple[int, int]
```

### 의존성 검증

Path 초기화 시 자동 검증:

```python
from visualpath.core import Path

# 잘못된 순서 → ValueError
path = Path("test", extractors=[
    ExpressionExtractor(),      # depends=["face_detect"]
    FaceDetectionExtractor(),   # 이게 먼저 와야 함
])
path.initialize()
# ValueError: Extractor 'expression' depends on {'face_detect'},
#             but only set() is available.
```

### 의존성 그래프 확인

```bash
facemoment info --deps
```

출력:
```
[Dependency Tree]
  face_detect
  │   Face detection (bbox, head pose)
  ├── face_classifier
  │      Face role classification (main/passenger/transient)
  └── expression
         Expression analysis (emotions)
```

### FlowGraph와 통합

```python
from visualpath.flow import FlowGraphBuilder
from facemoment.moment_detector.extractors import (
    FaceDetectionExtractor,
    FaceClassifierExtractor,
    ExpressionExtractor,
)

graph = (FlowGraphBuilder()
    .source()
    .path("detect", extractors=[FaceDetectionExtractor()])
    .path("classify", extractors=[FaceClassifierExtractor()])  # deps 자동 전달
    .path("expr", extractors=[ExpressionExtractor()])          # deps 자동 전달
    .on_trigger(handle_trigger)
    .build())
```

## 시각화 (Phase 16)

### 탑승자 구분 시각화

FaceClassifierExtractor 결과를 역할별 색상으로 시각화:

| 역할 | 색상 (BGR) | 설명 |
|------|------------|------|
| `main` | 초록 `(0,255,0)` 두꺼운 선 | 주탑승자 |
| `passenger` | 주황 `(0,165,255)` | 동승자 |
| `transient` | 노랑 `(0,255,255)` | 일시적 검출 |
| `noise` | 회색 `(128,128,128)` | 오검출/노이즈 |

**debug 명령어에서 자동 적용**:
```bash
# face extractor 사용 시 FaceClassifier 자동 활성화
facemoment debug video.mp4 -e face
# → main=초록, passenger=주황, transient=노랑, noise=회색으로 표시
```

```python
from facemoment.moment_detector.visualize import DebugVisualizer

visualizer = DebugVisualizer()
image = visualizer.create_debug_view(
    frame,
    face_obs=face_obs,
    classifier_obs=classifier_obs,  # 역할별 색상 표시
    pose_obs=pose_obs,
)
```

### 상반신 포즈 랜드마크 시각화

COCO 17 keypoints 중 상반신 (0-10번):
- 머리: 코, 눈, 귀
- 상반신: 어깨, 팔꿈치, 손목

```
        ●          ← 코 (흰색)
       /|\
      ● ●         ← 눈 (노랑)
     /   \
    ●     ●       ← 귀 (노랑)

    ●─────●       ← 어깨 (초록)
    │     │
    ●     ●       ← 팔꿈치 (노랑)
    │     │
    ●     ●       ← 손목 (노랑, 큰 원)
```

스켈레톤 연결선은 하늘색으로 표시됩니다.

## FaceClassifierExtractor (Phase 15-16)

탑승자 역할 분류:

**제약사항**: 주탑승자 1명, 동승자 최대 1명

| 역할 | 조건 | 인원 |
|------|------|------|
| `main` | 안정적 위치 + 큰 얼굴 (주탑승자) | 정확히 1명 |
| `passenger` | 안정적 위치 + 두 번째 후보 (동승자) | 0~1명 |
| `transient` | 위치 변화 큼, 또는 짧은 등장 | 0~N명 |
| `noise` | 작은 얼굴, 낮은 confidence, 가장자리 | 0~N명 |

### 위치 안정성 기반 분류

카메라 위치가 고정되어 있으므로:
- **주탑승자/동승자**: 한번 자리 잡으면 탑승 끝까지 위치 변화 거의 없음
- **transient**: 지나가는 사람은 프레임마다 위치가 크게 변함

```python
# 위치 안정성 추적
position_drift = distance(current_pos, avg_pos)
if drift > 0.15:  # 15% 이상 이동 → transient
    return "transient"
```

### 점수 계산 가중치

| 요소 | 가중치 | 설명 |
|------|--------|------|
| 위치 안정성 | 40% | 가장 중요 (카메라 고정) |
| 얼굴 크기 | 30% | 큰 얼굴 우선 |
| 프레임 중앙 | 20% | 중앙에 가까울수록 |
| 프레임 내부 | 10% | 얼굴이 잘리지 않음 |

```python
from facemoment.moment_detector.extractors import FaceClassifierExtractor

classifier = FaceClassifierExtractor(
    min_track_frames=5,      # transient 판정 기준
    min_area_ratio=0.01,     # noise 판정 최소 크기
    min_confidence=0.5,      # noise 판정 최소 confidence
    main_zone=(0.3, 0.7),    # 주탑승자 영역 (x 좌표)
)

# 결과 활용
data: FaceClassifierOutput = obs.data
if data.main_face:
    print(f"주탑승자: face_id={data.main_face.face.face_id}")
for pf in data.passenger_faces:
    print(f"동승자: face_id={pf.face.face_id}")
print(f"일시적 검출: {data.transient_count}, 오검출: {data.noise_count}")
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

## CLI 명령어

```bash
# 시스템 정보 확인
facemoment info                          # extractor/backend 상태, 파이프라인 구조
facemoment info -v                       # + GPU/ONNX 정보

# 디버그 (통합 명령어)
facemoment debug video.mp4               # 모든 extractor + FaceClassifier 자동 적용
facemoment debug video.mp4 -e raw        # 원본 비디오 프리뷰 (분석 없음)
facemoment debug video.mp4 -e face       # face만 (+ classifier로 역할별 색상)
facemoment debug video.mp4 -e pose       # pose만 (상반신 스켈레톤 표시)
facemoment debug video.mp4 -e face,pose  # 복수 선택
facemoment debug video.mp4 -e gesture    # gesture만
facemoment debug video.mp4 --no-ml       # dummy 모드 (ML 없이)
facemoment debug video.mp4 -o out.mp4    # 파일로 저장
facemoment debug video.mp4 -e face --profile  # 성능 프로파일링

# 디버그 (Distributed 모드)
facemoment debug video.mp4 --distributed     # 분산 모드 디버그
facemoment debug video.mp4 --venv-face /opt/venv-face  # venv 지정
facemoment debug video.mp4 --config pipeline.yaml      # 설정 파일 사용

# 클립 추출 (Library 모드)
facemoment process video.mp4 -o ./clips
facemoment process video.mp4 --gokart    # GR차량 모드
facemoment process video.mp4 --trace verbose --trace-output trace.jsonl

# 클립 추출 (Distributed 모드 - Phase 14)
facemoment process video.mp4 --distributed
facemoment process video.mp4 --venv-face /opt/venv-face
facemoment process video.mp4 --venv-face /opt/venv-face --venv-pose /opt/venv-pose
facemoment process video.mp4 --config pipeline.yaml

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

## Extractor 목록

### 분리된 Extractor (Phase 15)

| Extractor | depends | Backend | 설명 |
|-----------|---------|---------|------|
| FaceDetectionExtractor | - | InsightFace SCRFD | 얼굴 검출 (bbox, head pose) |
| ExpressionExtractor | `face_detect` | HSEmotion/PyFeat | 표정 분석 |
| FaceClassifierExtractor | `face_detect` | (내장 로직) | 역할 분류 |

### 기존 Extractor

| Extractor | Backend | 설명 | Extra |
|-----------|---------|------|-------|
| FaceExtractor | InsightFace + HSEmotion | 복합 (검출+표정) | `face` |
| PoseExtractor | YOLO-Pose | 포즈 추정 | `pose` |
| GestureExtractor | MediaPipe Hands | 손/제스처 감지 | `gesture` |
| QualityExtractor | OpenCV | 블러/밝기/대비 | (base) |
| DummyExtractor | - | 테스트용 | (base) |

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

### Main-Only 모드 (Phase 16)

HighlightFusion은 기본적으로 **주탑승자만 트리거**:

```python
fusion = HighlightFusion(
    main_only=True,  # 기본값: 주탑승자만 트리거
)

# FaceClassifier 결과를 함께 전달
result = fusion.update(face_obs, classifier_obs=classifier_obs)
```

- 동승자의 표정/헤드턴은 무시됨
- 주탑승자가 식별되지 않은 경우 모든 얼굴 분석

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
| `cli/__init__.py` | CLI 메인, argparse (--distributed 포함) |
| `cli/commands/process.py` | Library/Distributed 모드 처리 |
| `cli/commands/debug.py` | 통합 debug 명령어, 프로파일링 로직 |
| `cli/utils.py` | 공통 유틸리티, visualbase 호환성 |
| `pipeline/orchestrator.py` | PipelineOrchestrator (A-B*-C-A) |
| `pipeline/config.py` | ExtractorConfig, PipelineConfig |
| `moment_detector/detector.py` | MomentDetector (Library 모드용) |
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
uv run pytest tests/ -v            # 177 tests (pipeline/integration 제외)

# 특정 테스트
uv run pytest tests/test_pipeline.py -v
uv run pytest tests/test_gesture_extractor.py -v
uv run pytest tests/test_highlight_fusion.py -v
uv run pytest tests/test_observability.py -v
```

## 관련 패키지

- **visualbase**: 미디어 I/O (범용)
- **visualpath**: 분석 프레임워크 (범용) - facemoment가 **사용**
- **portrait981**: 통합 오케스트레이터 - facemoment를 **호출**

## 문서

- `docs/phase-11-summary.md`: Phase 11 작업 요약 (의존성 분리)
- `docs/problems-and-solutions.md`: 981파크 분석 알고리즘 (EWMA, 히스테리시스, GR차량 트리거)
- visualpath/docs/architecture.md: 플러그인 생태계 아키텍처
- visualpath/docs/stream-synchronization.md: 스트림 동기화 아키텍처

## 아키텍처: 플랫폼-플러그인 분리 (Phase 12 완료)

플랫폼 로직이 **visualpath** 패키지로 분리되어 3계층 구조가 완성되었습니다:

```
┌─────────────────────────────────────────────────────────┐
│  visualbase (미디어 소스)                                │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│  visualpath (플랫폼) ✅ 구현 완료                         │
│  - BaseExtractor, Observation 인터페이스                 │
│  - Plugin discovery (entry_points 기반)                 │
│  - IsolationLevel, WorkerLauncher                       │
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

### visualpath에서 제공하는 기능

| 모듈 | 기능 |
|------|------|
| `visualpath.core` | BaseExtractor, Observation, BaseFusion, FusionResult |
| `visualpath.core.isolation` | IsolationLevel (INLINE, THREAD, PROCESS, VENV, CONTAINER) |
| `visualpath.process` | WorkerLauncher, BaseWorker, InlineWorker, VenvWorker 등 |
| `visualpath.process.mapper` | ObservationMapper, DefaultObservationMapper, CompositeMapper |
| `visualpath.process.ipc` | ExtractorProcess, FusionProcess |
| `visualpath.process.orchestrator` | ExtractorOrchestrator |
| `visualpath.observability` | TraceLevel, ObservabilityHub, Sink, TraceRecord |
| `visualpath.plugin` | discover_extractors, discover_fusions, PluginRegistry |

### facemoment의 플러그인 등록 (pyproject.toml)

```toml
[project.entry-points."visualpath.extractors"]
face = "facemoment.moment_detector.extractors.face:FaceExtractor"
pose = "facemoment.moment_detector.extractors.pose:PoseExtractor"
gesture = "facemoment.moment_detector.extractors.gesture:GestureExtractor"
quality = "facemoment.moment_detector.extractors.quality:QualityExtractor"
dummy = "facemoment.moment_detector.extractors.dummy:DummyExtractor"

[project.entry-points."visualpath.fusions"]
highlight = "facemoment.moment_detector.fusion.highlight:HighlightFusion"
dummy = "facemoment.moment_detector.fusion.dummy:DummyFusion"
```

## portrait981에서 facemoment 사용

```python
# portrait981에서 facemoment 사용
from facemoment.pipeline import (
    PipelineOrchestrator,
    ExtractorConfig,
    create_default_config,
)

# 방법 1: 기본 설정
config = create_default_config(
    venv_face="/opt/venvs/venv-face",
    venv_pose="/opt/venvs/venv-pose",
    gokart_mode=True,
)
orchestrator = PipelineOrchestrator.from_config(config)
clips = orchestrator.run("video.mp4", fps=10)

# 방법 2: CLI 호출
# facemoment process video.mp4 --distributed --config pipeline.yaml
```

## 다음 작업 우선순위

### 단기 (Phase 16 후속)
1. 실제 GR차량 영상으로 통합 테스트
2. FaceClassifier 파라미터 튜닝 (min_track_frames, min_area_ratio 등)

### 중기 (Phase 17+)
3. portrait981을 facemoment 호출로 단순화
4. 새로운 의존성 기반 extractor 개발 (HeadPoseExtractor 등)

### 완료됨 ✅
- analysiscore 패키지 분리 → **visualpath** 패키지로 구현 완료
- 플러그인 discovery 구현 (entry_points 기반) → 완료
- facemoment 플러그인화 → 완료
- IPC 프로세스 이동 (Phase 13): ExtractorProcess, FusionProcess를 visualpath로 이동
- 독립 앱 (Phase 14): PipelineOrchestrator로 A-B*-C-A 완전한 파이프라인
- 의존성 기반 Extractor (Phase 15):
  - visualpath: BaseExtractor.depends/deps, Path._validate_dependencies()
  - facemoment: FaceDetectionExtractor, ExpressionExtractor, FaceClassifierExtractor 분리
  - 타입 안전한 Output (FaceDetectOutput, ExpressionOutput, FaceClassifierOutput)
  - CLI `facemoment info --deps` 의존성 그래프 출력
- **시각화 및 Main-Only 모드 (Phase 16)**:
  - 탑승자 구분: main(초록), passenger(주황), transient(노랑), noise(회색)
  - 상반신 포즈 스켈레톤: 머리, 어깨, 팔꿈치, 손목 랜드마크
  - Observation.data 필드 추가로 PoseOutput 지원
  - DebugVisualizer.create_debug_view()에 classifier_obs 파라미터 추가
  - FaceClassifier 위치 안정성 기반 분류 (카메라 고정 활용)
  - HighlightFusion main_only 모드: 주탑승자만 트리거
  - **debug.py에서 FaceClassifier 자동 통합**: face extractor 사용 시 자동 활성화
