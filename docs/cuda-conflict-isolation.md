# ML 의존성 충돌과 프로세스 격리

> facemoment 개발 과정에서 발견된 두 가지 실제 충돌 문제와,
> visualpath의 격리 메커니즘으로 해결한 과정을 기록합니다.

## 목차

1. [충돌 사례 1: onnxruntime GPU/CPU 패키지 충돌](#1-충돌-사례-1-onnxruntime-gpucpu-패키지-충돌)
2. [충돌 사례 2: onnxruntime-gpu vs PyTorch CUDA 런타임 충돌](#2-충돌-사례-2-onnxruntime-gpu-vs-pytorch-cuda-런타임-충돌)
3. [두 문제의 비교](#3-두-문제의-비교)
4. [visualpath가 제공하는 가치](#4-visualpath가-제공하는-가치)

---

## 1. 충돌 사례 1: onnxruntime GPU/CPU 패키지 충돌

### 배경

facemoment의 얼굴 분석은 두 단계로 구성됩니다:

| 단계 | 라이브러리 | onnxruntime 의존 |
|------|-----------|-----------------|
| 얼굴 검출 (face_detect) | insightface (SCRFD) | **onnxruntime-gpu** |
| 표정 분석 (expression) | hsemotion-onnx | **onnxruntime** (CPU) |

두 라이브러리를 같은 venv에 설치하면 문제가 발생합니다.

### 문제 현상

```bash
# 같은 venv에 둘 다 설치
pip install insightface    # → onnxruntime-gpu 설치됨
pip install hsemotion-onnx # → onnxruntime (CPU) 설치됨, GPU .so 덮어씀!
```

```
onnxruntime.capi.onnxruntime_pybind11_state.Fail:
  CUDA execution provider is not available.
  CUDA path: /usr/local/cuda
```

### 원인

`onnxruntime`과 `onnxruntime-gpu`는 **같은 Python 패키지명**(`onnxruntime`)을 사용하지만, 내부 공유 라이브러리(.so)가 다릅니다:

```
onnxruntime-gpu 설치 상태:
site-packages/onnxruntime/
├── capi/
│   ├── onnxruntime_pybind11_state.cpython-311.so
│   └── libonnxruntime_providers_cuda.so    ← GPU 지원
│   └── libonnxruntime_providers_shared.so  ← GPU 공유 라이브러리
└── ...

hsemotion-onnx 설치 후:
site-packages/onnxruntime/
├── capi/
│   ├── onnxruntime_pybind11_state.cpython-311.so  ← CPU 버전으로 덮어씀!
│   └── (libonnxruntime_providers_cuda.so 없음)    ← GPU .so 삭제됨
└── ...
```

pip은 `onnxruntime-gpu`와 `onnxruntime`을 서로 다른 패키지로 인식하면서도, 같은 디렉토리에 설치합니다. 나중에 설치된 쪽이 이전 .so 파일을 덮어쓰거나 제거합니다.

### 해결: VenvWorker (Phase 18)

같은 venv에서는 해결이 불가능합니다. **별도 venv**로 분리해야 합니다:

```
venv-face-detect/                    venv-expression/
├── onnxruntime-gpu                  ├── onnxruntime (CPU)
├── insightface                      ├── hsemotion-onnx
└── GPU .so 정상                     └── CPU .so 정상
```

visualpath의 `VenvWorker`가 이 격리를 제공합니다:

```yaml
# pipeline.yaml
extractors:
  - name: face_detect
    venv_path: /opt/venvs/venv-face-detect    # onnxruntime-gpu
    isolation: venv
  - name: expression
    venv_path: /opt/venvs/venv-expression     # onnxruntime CPU
    isolation: venv
  - name: face_classifier
    isolation: inline                         # 순수 Python, venv 불필요
```

```python
# facemoment 코드에서는 격리를 의식할 필요 없음
class ExpressionExtractor(BaseExtractor):
    depends = ["face_detect"]

    def extract(self, frame, deps=None):
        face_data = deps["face_detect"].data  # VenvWorker가 ZMQ로 전달
        # ... 표정 분석
```

### 추가 방어: override-dependencies

로컬 개발 환경에서는 venv 분리 없이 모든 패키지를 한 venv에 설치해야 할 때가 있습니다. 이 경우 CPU 버전이 GPU를 덮어쓰지 않도록 차단합니다:

```toml
# pyproject.toml
[tool.uv]
override-dependencies = [
    "onnxruntime ; sys_platform == 'never'",  # 불가능한 조건으로 CPU 버전 차단
]
```

이렇게 하면 `hsemotion-onnx`가 `onnxruntime` (CPU)를 요구해도 uv가 설치를 건너뛰고, 이미 설치된 `onnxruntime-gpu`가 유지됩니다. 단, 이 방법은 GPU가 있는 개발 환경에서만 유효합니다.

---

## 2. 충돌 사례 2: onnxruntime-gpu vs PyTorch CUDA 런타임 충돌

### 배경

얼굴 분석(onnxruntime-gpu)과 포즈 추정(PyTorch)을 동시에 실행하면 CUDA 런타임 수준에서 충돌합니다:

| Extractor | 프레임워크 | CUDA 바인딩 |
|-----------|-----------|------------|
| FaceExtractor (face_detect, expression) | onnxruntime-gpu | `libonnxruntime_providers_cuda.so` |
| PoseExtractor | PyTorch (ultralytics) | `libc10_cuda.so` (libtorch) |

### 문제 현상

```
ImportError: /opt/venv/lib/python3.11/site-packages/torch/lib/libc10_cuda.so:
  undefined symbol: cudaGetDriverEntryPointByVersion
```

또는:

```
ONNX Runtime: CUDA initialization failed
```

### 원인: 같은 프로세스 내 CUDA 런타임 이중 로드

onnxruntime-gpu와 PyTorch는 각각 자체적으로 CUDA 런타임을 로드합니다. 같은 프로세스에서 두 라이브러리가 CUDA를 초기화하면 공유 라이브러리 심볼이 충돌합니다.

```
프로세스 메모리 공간
┌─────────────────────────────────────────────────┐
│                                                 │
│  onnxruntime-gpu                                │
│  └─ libonnxruntime_providers_cuda.so            │
│     └─ libcudart.so.12 (CUDA 런타임 A)         │
│        ├─ cudaLaunchKernel        ──┐           │
│        ├─ cudaMemcpy                │ 심볼      │
│        └─ cudaGetDriverEntryPoint...│ 충돌!     │
│                                     │           │
│  PyTorch                            │           │
│  └─ libc10_cuda.so                  │           │
│     └─ libcudart.so.12 (CUDA 런타임 B)         │
│        ├─ cudaLaunchKernel        ──┘           │
│        ├─ cudaMemcpy                            │
│        └─ cudaGetDriverEntryPoint...            │
│                                                 │
│  → 같은 심볼이 두 번 로드됨                      │
│  → 먼저 로드된 쪽이 심볼 테이블을 점유            │
│  → 나중에 로드된 쪽에서 버전 불일치로 실패        │
└─────────────────────────────────────────────────┘
```

이 문제는 Python ML 생태계에서 광범위하게 보고되고 있습니다:
- [onnxruntime#13824](https://github.com/microsoft/onnxruntime/issues/13824)
- [PyTorch forums: CUDA conflicts with onnxruntime](https://discuss.pytorch.org/t/cuda-error-when-using-onnxruntime-and-pytorch/172722)

### 핵심 차이: 사례 1과 다른 계층의 문제

사례 1은 **pip 패키지 수준**의 충돌입니다. 같은 이름의 패키지(.so)가 덮어써지는 문제이므로, venv를 분리하면 해결됩니다.

사례 2는 **프로세스 수준**의 충돌입니다. 패키지가 정상적으로 설치되어 있어도, 같은 프로세스에서 두 CUDA 런타임이 로드되면 심볼이 충돌합니다. venv를 분리해도 같은 프로세스에서 import하면 충돌합니다.

```
                패키지 분리 (VenvWorker)    프로세스 분리 (ProcessWorker)
사례 1 (GPU/CPU)     ✅ 해결됨                  ✅ 해결됨 (과잉)
사례 2 (CUDA 런타임)  ❌ 해결 안 됨              ✅ 해결됨
```

### 임시 해결: 초기화 순서 제어

첫 번째 해결 시도는 torch를 onnxruntime보다 먼저 초기화하는 것이었습니다:

```python
# pathway_pipeline.py (Phase 17)
_TORCH_EXTRACTORS = frozenset({"pose"})

ordered_names = sorted(
    self._extractor_names,
    key=lambda n: 0 if n in self._TORCH_EXTRACTORS else 1,
)
# → ["pose", "face"] 순서로 초기화
```

#### 한계

| 문제 | 설명 |
|------|------|
| **취약성** | 초기화 순서에 의존하는 것은 본질적으로 불안정 |
| **CUDA 버전 의존** | 특정 CUDA/드라이버 조합에서만 우연히 동작 |
| **확장 불가** | 새로운 CUDA 기반 extractor 추가 시 순서 관리 복잡 |
| **디버깅 난이도** | 순서가 바뀌면 원인을 추적하기 어려운 에러 발생 |

### 근본 해결: ProcessWorker 자동 격리

visualpath의 `ProcessWorker`를 사용하여 충돌하는 extractor를 별도 프로세스에서 실행합니다. VenvWorker와 달리 별도 venv가 필요 없고, 현재 venv의 Python으로 subprocess를 실행합니다.

```
메인 프로세스                    subprocess (ProcessWorker)
┌──────────────────────┐       ┌──────────────────────┐
│                      │       │                      │
│  onnxruntime-gpu     │       │  PyTorch             │
│  ├─ face_detect      │  ZMQ  │  └─ PoseExtractor    │
│  ├─ expression       │◄─────►│                      │
│  └─ face_classifier  │  IPC  │  독립 CUDA 런타임     │
│                      │       │  → 심볼 충돌 없음      │
│  HighlightFusion     │       │                      │
│                      │       │                      │
└──────────────────────┘       └──────────────────────┘
```

#### 자동 감지 메커니즘

`_CUDA_GROUPS`로 충돌 그룹을 정의하고, 2개 이상 그룹이 활성화되면 소수 그룹을 자동 격리합니다:

```python
_CUDA_GROUPS = {
    "onnxruntime": {"face", "face_detect", "expression"},
    "torch": {"pose"},
}
```

```
사용자 요청: extractors=["face", "pose"]

_detect_cuda_conflicts()
  → onnxruntime 그룹: ["face"]  (1개)
  → torch 그룹: ["pose"]       (1개)
  → 2개 그룹 활성 → 충돌!
  → torch(소수) → {"pose"}를 ProcessWorker로 격리

결과:
  inline:  [face, face_classifier]  (메인 프로세스)
  workers: {pose: ProcessWorker}    (subprocess)
```

#### deps 전달

subprocess로 격리된 extractor도 다른 extractor의 결과를 받을 수 있습니다. inline extractor 실행 후 누적된 deps가 ZMQ IPC를 통해 worker에 전달됩니다:

```python
for frame in frames:
    deps = {}

    # 1) inline extractors
    for ext in self._extractors:
        obs = ext.extract(frame, deps)
        deps[ext.name] = obs

    # 2) subprocess workers (deps 전달)
    for name, worker in self._workers.items():
        result = worker.process(frame, deps=deps)
        deps[name] = result.observation
```

#### Fallback

pyzmq가 설치되지 않은 환경에서는 ProcessWorker를 사용할 수 없으므로, 기존 초기화 순서 워크어라운드로 자동 fallback합니다:

```
pyzmq 있음 → ProcessWorker 격리 (안전)
pyzmq 없음 → torch 먼저 초기화 (순서 워크어라운드)
```

---

## 3. 두 문제의 비교

facemoment에서 발견된 두 충돌은 서로 다른 계층에서 발생하며, 서로 다른 격리 수준으로 해결됩니다:

```
충돌 계층                해결에 필요한 격리

pip 패키지 (.so 덮어쓰기)  ──→  VenvWorker (별도 venv)
  └─ onnxruntime vs onnxruntime-gpu

CUDA 런타임 (심볼 충돌)    ──→  ProcessWorker (별도 프로세스)
  └─ onnxruntime-gpu vs PyTorch
```

| | 사례 1: GPU/CPU 패키지 | 사례 2: CUDA 런타임 |
|---|---|---|
| **충돌 원인** | 같은 이름의 pip 패키지가 .so 덮어씀 | 같은 프로세스에서 CUDA 심볼 이중 로드 |
| **발생 시점** | `pip install` (설치 시) | `import` / 모델 로드 (런타임) |
| **에러 메시지** | `CUDA execution provider is not available` | `undefined symbol: cudaGetDriverEntryPoint...` |
| **같은 venv에서 해결?** | ❌ 불가 | ❌ 불가 |
| **같은 프로세스에서 해결?** | ✅ 가능 (venv만 분리하면 됨) | ❌ 불가 |
| **필요한 격리** | **VenvWorker** (패키지 격리) | **ProcessWorker** (프로세스 격리) |
| **오버헤드** | 높음 (별도 venv 생성/관리) | 중간 (같은 venv, IPC만 추가) |

### 타임라인

```
Phase 11  venv 분리 구조 설계
Phase 15  face → face_detect + expression 분리
Phase 18  VenvWorker로 onnxruntime GPU/CPU 격리 (사례 1 해결)
          fine-grained extras (face-detect, expression)
Phase 19  ProcessWorker로 CUDA 런타임 자동 격리 (사례 2 해결)
          _detect_cuda_conflicts() 자동 감지
```

---

## 4. visualpath가 제공하는 가치

두 사례는 ML 파이프라인에 **플랫폼 레벨의 다층 격리**가 필요한 이유를 보여줍니다.

### 격리 수준 스펙트럼

visualpath는 충돌의 심각도에 따라 적절한 격리 수준을 선택할 수 있습니다:

```
격리 수준        해결하는 문제                        오버헤드
──────────────────────────────────────────────────────────
InlineWorker     충돌 없음                           없음
ThreadWorker     I/O 바운드, GIL 우회                낮음
ProcessWorker    CUDA 런타임 충돌 (사례 2)            중간 (IPC)
VenvWorker       pip 패키지 충돌 (사례 1)             높음 (별도 venv)
```

### 개별 앱이 직접 해결할 경우

각 분석 앱이 독자적으로 격리를 구현해야 합니다:

```
facemoment:  자체 subprocess 관리, ZMQ 통신, 직렬화, venv 관리...
plugin-ocr:  또 자체 subprocess 관리, ZMQ 통신, 직렬화...
plugin-scene: 또 자체 subprocess 관리...
```

- 코드 중복
- 버그 가능성 증가 (각 구현마다 edge case 처리)
- IPC 프로토콜 불일치
- 격리 수준 변경 시 대규모 리팩토링

### visualpath 플랫폼이 해결할 경우

```
visualpath (플랫폼)
├── InlineWorker      ← 같은 프로세스
├── ThreadWorker      ← 같은 프로세스, 별도 스레드
├── ProcessWorker     ← 같은 venv, 별도 프로세스 (CUDA 격리)
├── VenvWorker        ← 별도 venv, 별도 프로세스 (패키지 격리)
├── ZMQ IPC 프로토콜  ← 통일된 직렬화/역직렬화
└── deps 전달         ← extractor 간 의존성 자동 처리

앱 코드:
  ProcessWorker(extractor_name="pose")  ← 한 줄
  VenvWorker(extractor_name="expression", venv_path="...")  ← 한 줄
```

### 앱 코드는 격리를 모름

facemoment의 extractor 코드는 자신이 어떤 격리 수준에서 실행되는지 알 필요가 없습니다:

```python
class PoseExtractor(BaseExtractor):
    """포즈 추정. ProcessWorker 안에서 실행될 수도, InlineWorker로 실행될 수도 있음."""

    def extract(self, frame, deps=None):
        # 격리 방식과 무관한 동일한 코드
        results = self.model(frame.data)
        return Observation(source="pose", ...)
```

격리 결정은 파이프라인 설정 레벨에서 이루어집니다:

```yaml
# 설정만 바꾸면 격리 수준 변경
extractors:
  - name: pose
    isolation: inline    # 개발 환경: 빠르게 테스트
  - name: pose
    isolation: process   # 프로덕션: CUDA 격리
  - name: pose
    isolation: venv      # 극단적 격리: 별도 venv
    venv_path: /opt/venvs/venv-pose
```

### 정리

| 관점 | 앱 자체 해결 | visualpath 플랫폼 |
|------|-------------|-------------------|
| 구현 비용 | 높음 (ZMQ, 직렬화, 프로세스/venv 관리) | **한 줄** |
| 재사용성 | 없음 (앱별 구현) | 모든 플러그인이 공유 |
| deps 전달 | 직접 구현 | 플랫폼이 자동 처리 |
| 격리 수준 변경 | 대규모 리팩토링 | 설정 한 줄 변경 |
| 테스트 | 각 앱에서 개별 테스트 | 플랫폼 레벨에서 검증됨 |
| 새 충돌 유형 대응 | 각 앱에서 새로 구현 | 플랫폼에 Worker 추가 → 모든 앱 혜택 |

> **핵심**: ML 파이프라인에서 의존성 충돌은 피할 수 없는 현실이며,
> 충돌은 pip 패키지 수준부터 CUDA 런타임 수준까지 다양한 계층에서 발생합니다.
> visualpath는 이를 **다층 격리 스펙트럼**으로 해결하며,
> 앱 코드는 분석 로직에만 집중할 수 있습니다.
