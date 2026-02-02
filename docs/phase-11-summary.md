# Phase 11 Summary: 의존성 분리 구조

> 완료일: 2026-01-30
> 테스트: 150 tests passing

## 개요

Phase 11에서는 Python ML 생태계의 의존성 충돌 문제를 해결하기 위해 **worker별 독립 venv**를 지원하는 구조로 리팩토링했습니다.

---

## 문제 상황

### ML 라이브러리 간 의존성 충돌

```
insightface (onnxruntime-gpu) ←→ ultralytics (torch)
                    ↑
              mediapipe (특정 protobuf 버전)
```

- InsightFace: `onnxruntime-gpu` 사용
- Ultralytics (YOLO): PyTorch 기반
- MediaPipe: 특정 protobuf 버전 요구

단일 venv에서 모든 라이브러리 설치 시 충돌 발생 가능.

---

## 해결 방안

### Optional Dependencies 구조

```toml
# pyproject.toml

[project.optional-dependencies]
# Base는 모든 환경에 포함 (의존성 명시 안 함)

# Worker별 독립 의존성
face = [
    "insightface>=0.7.0",
    "hsemotion-onnx>=0.3.1",
    "onnxruntime-gpu>=1.17.0",
]

face-full = [
    "insightface>=0.7.0",
    "py-feat>=0.6.0",
]

pose = [
    "ultralytics>=8.0.0",
]

gesture = [
    "mediapipe>=0.10.0",
]

# IPC 통신
zmq = ["pyzmq>=25.0.0"]

# CLI 도구
cli = ["pyqt6>=6.0.0"]

# 개발/테스트 (충돌 가능성 있음)
all = [
    "facemoment[face]",
    "facemoment[pose]",
    "facemoment[gesture]",
    "facemoment[cli]",
]
```

### Lazy Import 패턴

```python
# src/facemoment/moment_detector/extractors/__init__.py

from .base import BaseExtractor, Observation

# Lazy imports - 해당 extra 설치 시에만 import 가능
__all__ = [
    "BaseExtractor",
    "Observation",
    # 아래는 lazy import
    "FaceExtractor",     # [face]
    "PoseExtractor",     # [pose]
    "GestureExtractor",  # [gesture]
]

def __getattr__(name: str):
    if name == "FaceExtractor":
        from .face import FaceExtractor
        return FaceExtractor
    elif name == "PoseExtractor":
        from .pose import PoseExtractor
        return PoseExtractor
    elif name == "GestureExtractor":
        from .gesture import GestureExtractor
        return GestureExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

---

## Worker별 venv 생성

### Face Worker

```bash
uv venv venv-face
source venv-face/bin/activate
uv pip install -e ".[face,zmq]"
```

설치되는 패키지:
- numpy, opencv-python (base)
- insightface, hsemotion-onnx, onnxruntime-gpu
- pyzmq (IPC 통신)

### Pose Worker

```bash
uv venv venv-pose
source venv-pose/bin/activate
uv pip install -e ".[pose,zmq]"
```

설치되는 패키지:
- numpy, opencv-python (base)
- ultralytics (+ PyTorch)
- pyzmq (IPC 통신)

### Gesture Worker

```bash
uv venv venv-gesture
source venv-gesture/bin/activate
uv pip install -e ".[gesture,zmq]"
```

설치되는 패키지:
- numpy, opencv-python (base)
- mediapipe
- pyzmq (IPC 통신)

### 개발 환경 (전체)

```bash
uv sync --extra all --extra dev
```

충돌 가능성이 있으므로 프로덕션에서는 비권장.

---

## Import 패턴

### 항상 가능 (base 의존성)

```python
from facemoment.moment_detector.extractors import BaseExtractor, Observation
from facemoment.observability import ObservabilityHub, TraceLevel
from facemoment.process import ExtractorProcess, FusionProcess
```

### 조건부 Import

```python
# Face worker에서만 가능
try:
    from facemoment.moment_detector.extractors.face import FaceExtractor
except ImportError:
    FaceExtractor = None

# Pose worker에서만 가능
try:
    from facemoment.moment_detector.extractors.pose import PoseExtractor
except ImportError:
    PoseExtractor = None

# Gesture worker에서만 가능
try:
    from facemoment.moment_detector.extractors.gesture import GestureExtractor
except ImportError:
    GestureExtractor = None
```

### CLI에서의 동적 로딩

```python
# cli/commands/debug.py

def load_extractors(extractor_types: list[str]) -> list[BaseExtractor]:
    extractors = []

    for ext_type in extractor_types:
        if ext_type == "face":
            try:
                from facemoment.moment_detector.extractors.face import FaceExtractor
                extractors.append(FaceExtractor())
            except ImportError:
                print("Warning: FaceExtractor requires [face] extra")

        elif ext_type == "pose":
            try:
                from facemoment.moment_detector.extractors.pose import PoseExtractor
                extractors.append(PoseExtractor())
            except ImportError:
                print("Warning: PoseExtractor requires [pose] extra")

        # ... 등등

    return extractors
```

---

## 변경된 파일 목록

### 새로 생성된 파일

| 파일 | 역할 |
|------|------|
| `src/facemoment/cli/__init__.py` | CLI 메인, argparse |
| `src/facemoment/cli/utils.py` | 공통 유틸리티 |
| `src/facemoment/cli/commands/info.py` | 시스템 정보 명령어 |
| `src/facemoment/cli/commands/debug.py` | 통합 디버그 명령어 |
| `src/facemoment/cli/commands/process.py` | 클립 추출 명령어 |
| `src/facemoment/cli/commands/benchmark.py` | 벤치마크 명령어 |
| `src/facemoment/moment_detector/extractors/gesture.py` | GestureExtractor |
| `src/facemoment/moment_detector/extractors/backends/hand_backends.py` | MediaPipe 백엔드 |
| `src/facemoment/observability/__init__.py` | ObservabilityHub |
| `src/facemoment/observability/records.py` | TraceRecord 타입들 |
| `src/facemoment/observability/sinks.py` | 로깅 싱크들 |
| `src/facemoment/process/orchestrator.py` | ExtractorOrchestrator |

### 수정된 파일

| 파일 | 변경 내용 |
|------|-----------|
| `pyproject.toml` | optional dependencies 추가 |
| `src/facemoment/moment_detector/extractors/__init__.py` | lazy import 패턴 적용 |
| `src/facemoment/moment_detector/extractors/base.py` | 인터페이스 정리 |
| `src/facemoment/moment_detector/extractors/face.py` | 컴포넌트별 타이밍 추가 |
| `src/facemoment/moment_detector/extractors/pose.py` | lazy import 대응 |
| `src/facemoment/moment_detector/extractors/backends/__init__.py` | 백엔드 export 정리 |
| `src/facemoment/moment_detector/extractors/backends/base.py` | 프로토콜 정의 |
| `src/facemoment/moment_detector/extractors/backends/face_backends.py` | 백엔드 구현 |
| `src/facemoment/moment_detector/fusion/highlight.py` | Observability 연동 |
| `src/facemoment/moment_detector/visualize.py` | 타이밍 오버레이 추가 |
| `src/facemoment/process/__init__.py` | export 정리 |
| `src/facemoment/process/extractor.py` | orchestrator 연동 |
| `src/facemoment/process/fusion.py` | Observability 연동 |

### 삭제된 파일

| 파일 | 이유 |
|------|------|
| `src/facemoment/cli.py` | `cli/` 모듈로 리팩토링 |

---

## 테스트

### 새로 추가된 테스트

| 파일 | 테스트 내용 |
|------|-------------|
| `tests/test_gesture_extractor.py` | GestureExtractor 유닛 테스트 |
| `tests/test_observability.py` | Observability 시스템 테스트 |
| `tests/test_integration_process.py` | 통합 처리 테스트 |

### 테스트 실행

```bash
# 전체 테스트
uv run pytest tests/ -v

# 특정 모듈
uv run pytest tests/test_gesture_extractor.py -v
uv run pytest tests/test_observability.py -v
uv run pytest tests/test_highlight_fusion.py -v
```

---

## CLI 변경사항

### --profile 플래그 추가

FaceExtractor 성능 프로파일링:

```bash
facemoment debug video.mp4 -e face --profile
```

출력 예시:
```
Backends:
  Detection   : InsightFaceSCRFD [CUDA]
  Expression  : HSEmotionBackend
--------------------------------------------------
Frame 1: detect=42.3ms, expression=28.1ms, total=71.5ms
Frame 2: detect=38.7ms, expression=31.2ms, total=70.8ms
...
```

화면에 타이밍 오버레이:
- 녹색: 빠름 (<50ms)
- 노랑: 보통 (50-100ms)
- 빨강: 느림 (>100ms)

### Extractor 선택 확장

```bash
facemoment debug video.mp4 -e gesture    # gesture만
facemoment debug video.mp4 -e face,pose  # 복수 선택
facemoment debug video.mp4 -e raw        # 원본만 (분석 없음)
```

---

## 다음 단계

1. **portrait981에서 적용**: worker별 독립 venv 실행 구조
2. **실제 영상 테스트**: GR차량 영상으로 threshold 튜닝
3. **analysiscore 분리**: 플랫폼 로직 별도 패키지화 (Phase 12)

---

## 관련 문서

- [visualpath Architecture](../../visualpath/docs/architecture.md): 플러그인 생태계 아키텍처
- [Problems and Solutions](./problems-and-solutions.md): 981파크 분석 알고리즘
- [Stream Synchronization](../../visualpath/docs/stream-synchronization.md): 스트림 동기화 (visualpath)
