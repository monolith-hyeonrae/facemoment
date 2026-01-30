# FaceMoment Architecture Vision

> 문서 작성일: 2026-01-30
> 상태: Phase 11 완료 후 비전 정리

## 개요

이 문서는 facemoment 프로젝트의 장기적인 아키텍처 비전을 설명합니다. 현재 모놀리식 구조에서 플러그인 기반 생태계로의 발전 방향을 제시합니다.

---

## 현재 상태 (Phase 11)

### 패키지 구조

```
monolith/
├── visualbase/      # 미디어 소스 기반 레이어
├── facemoment/      # 영상 분석 + 플랫폼 로직 (혼재)
└── portrait981/     # 프로덕션 orchestrator
```

### 문제점

1. **결합도가 높음**: 플랫폼 로직(Fusion, IPC, Observability)과 분석 로직(Face/Pose/Gesture)이 한 패키지에 혼재
2. **확장성 제한**: 새로운 분석기(OCR, Scene Detection 등)를 추가하려면 facemoment 수정 필요
3. **의존성 관리**: ML 라이브러리 간 충돌로 worker별 venv 필요 (Phase 11에서 해결)

---

## 목표 아키텍처: 3계층 플러그인 생태계

```
┌─────────────────────────────────────────────────────────┐
│                    visualbase                            │
│              (미디어 소스 기반 레이어)                    │
│  - Frame, Source, Stream 추상화                          │
│  - 비디오/카메라/스트림 통합 인터페이스                   │
│  - 클립 추출                                             │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  analysiscore (신규)                     │
│              (분석 플랫폼 코어 레이어)                    │
│  - BaseExtractor, Observation 인터페이스                 │
│  - Plugin discovery & loading                            │
│  - Worker orchestration (ZMQ IPC)                        │
│  - Fusion framework                                      │
│  - Observability system                                  │
└─────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │facemoment│    │ plugin-A │    │ plugin-B │
    │ (plugin) │    │          │    │          │
    │  - Face  │    │  - OCR   │    │ - Object │
    │  - Pose  │    │  - Text  │    │ - Scene  │
    │  - Gesture│   │          │    │          │
    └──────────┘    └──────────┘    └──────────┘
```

### 계층별 역할

| 레이어 | 패키지 | 역할 | 의존성 |
|--------|--------|------|--------|
| **Media** | visualbase | 미디어 소스, Frame, 클립 추출 | opencv, numpy |
| **Platform** | analysiscore (신규) | 플러그인 프레임워크, IPC, Fusion | visualbase, pyzmq |
| **Plugin** | facemoment 등 | 순수 분석 로직 (Extractor 구현체) | analysiscore, ML libs |

---

## analysiscore 패키지 설계

### facemoment에서 분리할 컴포넌트

| 현재 위치 | 분리 대상 | 역할 |
|-----------|-----------|------|
| `extractors/base.py` | BaseExtractor, Observation | 플러그인 인터페이스 |
| `extractors/backends/base.py` | Backend protocols | 백엔드 인터페이스 |
| `fusion/base.py` (신규) | BaseFusion, FusionResult | Fusion 인터페이스 |
| `process/*.py` | Worker orchestration | 분산 처리 프레임워크 |
| `observability/` | 전체 | 모니터링 시스템 |

### analysiscore 구조 (예상)

```
analysiscore/
├── __init__.py              # 메인 API
├── interfaces/
│   ├── extractor.py         # BaseExtractor, Observation
│   ├── backend.py           # DetectionBackend, ExpressionBackend 등
│   └── fusion.py            # BaseFusion, FusionResult
├── orchestration/
│   ├── worker.py            # Worker 기반 클래스
│   ├── process.py           # ExtractorProcess, FusionProcess
│   └── ipc.py               # ZMQ IPC 통신
├── plugin/
│   ├── discovery.py         # entry_points 기반 발견
│   ├── loader.py            # 플러그인 로딩
│   └── registry.py          # Extractor 레지스트리
├── observability/
│   ├── __init__.py          # ObservabilityHub
│   ├── records.py           # TraceRecord 타입들
│   └── sinks.py             # FileSink, ConsoleSink 등
└── fusion/
    ├── base.py              # BaseFusion
    └── combiners.py         # 공통 결합 로직
```

---

## 플러그인 개발 가이드 (미래)

### entry_points 등록

플러그인은 `pyproject.toml`에서 entry_points로 등록됩니다:

```toml
[project.entry-points."analysiscore.extractors"]
face = "facemoment.extractors.face:FaceExtractor"
pose = "facemoment.extractors.pose:PoseExtractor"
gesture = "facemoment.extractors.gesture:GestureExtractor"

[project.entry-points."analysiscore.fusions"]
highlight = "facemoment.fusion.highlight:HighlightFusion"
```

### Extractor 구현 예시

```python
from analysiscore import BaseExtractor, Observation
from typing import Any
import numpy as np

class MyExtractor(BaseExtractor):
    """Custom extractor plugin example."""

    name = "my_extractor"
    version = "1.0.0"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        # 백엔드 초기화

    def extract(self, frame: np.ndarray, frame_id: int) -> list[Observation]:
        # 분석 로직
        results = self._analyze(frame)

        return [
            Observation(
                source="my_extractor",
                frame_id=frame_id,
                data=result,
                confidence=result.score,
            )
            for result in results
        ]

    def reset(self) -> None:
        # 상태 초기화
        pass
```

### Fusion 구현 예시

```python
from analysiscore import BaseFusion, FusionResult
from analysiscore.interfaces import Observation

class MyFusion(BaseFusion):
    """Custom fusion plugin example."""

    name = "my_fusion"

    def process(
        self,
        observations: dict[str, list[Observation]]
    ) -> FusionResult:
        # 여러 extractor 결과 결합
        triggers = self._analyze_patterns(observations)

        return FusionResult(
            triggers=triggers,
            metadata={"processed_at": time.time()}
        )
```

---

## facemoment 플러그인화 후 구조

analysiscore 분리 후 facemoment에 남는 것:

```
facemoment/
├── __init__.py              # 플러그인 메타데이터
├── extractors/
│   ├── face.py              # FaceExtractor 구현
│   ├── pose.py              # PoseExtractor 구현
│   └── gesture.py           # GestureExtractor 구현
├── backends/
│   ├── face_backends.py     # InsightFace, HSEmotion
│   ├── pose_backends.py     # YOLO-Pose
│   └── hand_backends.py     # MediaPipe
└── fusion/
    └── highlight.py         # HighlightFusion 구현
```

- 순수 분석 로직만 포함
- 플랫폼 로직은 analysiscore에 의존
- ML 라이브러리만 optional dependencies로 관리

---

## 마이그레이션 로드맵

### Phase 12: analysiscore 패키지 분리

1. **인터페이스 추출**
   - `BaseExtractor`, `Observation` → analysiscore
   - Backend protocols → analysiscore
   - `BaseFusion` 정의 → analysiscore

2. **orchestration 이동**
   - `process/extractor.py` → analysiscore
   - `process/fusion.py` → analysiscore
   - `process/orchestrator.py` → analysiscore

3. **observability 이동**
   - 전체 `observability/` 디렉토리 → analysiscore

### Phase 13: 플러그인 discovery 구현

1. **entry_points 기반 발견**
   ```python
   from importlib.metadata import entry_points

   def discover_extractors():
       eps = entry_points(group="analysiscore.extractors")
       return {ep.name: ep.load() for ep in eps}
   ```

2. **설정 파일로 활성화 제어**
   ```yaml
   # analysiscore.yaml
   extractors:
     enabled:
       - face
       - pose
     disabled:
       - gesture  # 비활성화
   ```

### Phase 14: facemoment 플러그인화

1. **analysiscore 의존성 추가**
   ```toml
   dependencies = ["analysiscore>=1.0.0"]
   ```

2. **entry_points 등록**

3. **import 경로 업데이트**
   - `from facemoment.moment_detector.extractors.base import BaseExtractor`
   - → `from analysiscore import BaseExtractor`

### Phase 15: portrait981 업데이트

1. **analysiscore 사용으로 전환**
2. **플러그인 동적 로딩**
3. **worker별 독립 venv 실행**

---

## 기대 효과

### 확장성

- 새 분석기 추가 시 별도 패키지로 개발 가능
- analysiscore나 facemoment 수정 불필요
- 독립적인 버전 관리 및 배포

### 유지보수성

- 관심사 분리로 코드 이해도 향상
- 테스트 격리 용이
- 의존성 충돌 최소화

### 재사용성

- analysiscore를 다른 프로젝트에서 재사용
- 플러그인 에코시스템 구축 가능

---

## 관련 문서

- [Phase 11 Summary](./phase-11-summary.md): 의존성 분리 구조 작업 요약
- [Problems and Solutions](./problems-and-solutions.md): 알고리즘 문서
- [Stream Synchronization](./stream-synchronization.md): 스트림 동기화 아키텍처
