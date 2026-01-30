# 스트림 동기화

FaceMoment의 A-B*-C 아키텍처에서 여러 데이터 스트림을 동기화하는 문제와 해결책을 설명합니다.

## 목차

1. [아키텍처 개요](#아키텍처-개요)
2. [visualbase와의 연동](#visualbase와의-연동)
3. [문제 1: Extractor 처리 시간 불균형](#문제-1-extractor-처리-시간-불균형)
4. [문제 2: 프레임 드롭과 백프레셔](#문제-2-프레임-드롭과-백프레셔)
5. [문제 3: OBS 동기화 지연](#문제-3-obs-동기화-지연)
6. [해결책: ExtractorOrchestrator](#해결책-extractororchestrator)
7. [해결책: 시간 윈도우 정렬](#해결책-시간-윈도우-정렬)
8. [Observability 연동](#observability-연동)
9. [설정 가이드](#설정-가이드)
10. [향후: Bytewax 연동 가능성](#향후-bytewax-연동-가능성)

---

## 아키텍처 개요

```
              ┌─────────────────────────────────┐
              │      visualbase (A 모듈)        │
              │  - VideoSource: 프레임 공급     │
              │  - ClipExtractor: 클립 추출     │
              │  - Daemon: ZMQ 프레임 배포      │
              └──────────────┬──────────────────┘
                             │
              ┌──────────────┼──────────────────┐
              ▼              ▼                  ▼
        ┌──────────┐   ┌──────────┐      ┌──────────┐
        │   Face   │   │   Pose   │      │ Gesture  │  (B* 모듈)
        │ Extractor│   │ Extractor│      │ Extractor│
        └────┬─────┘   └────┬─────┘      └────┬─────┘
             │              │                  │
             │    OBS 메시지│                  │
             └──────────────┼──────────────────┘
                            ▼
                      ┌──────────┐
                      │  Fusion  │  (C 모듈)
                      │  Process │
                      └────┬─────┘
                           │  TRIG 메시지
                           ▼
              ┌─────────────────────────────────┐
              │      visualbase (A 모듈)        │
              │  - ClipExtractor로 클립 생성    │
              └─────────────────────────────────┘
```

---

## visualbase와의 연동

### visualbase 주요 컴포넌트

| 컴포넌트 | 위치 | 역할 |
|----------|------|------|
| `VideoSource` | `sources/` | 파일/카메라/스트림에서 Frame 생성 |
| `ClipExtractor` | `core/` | TRIG 메시지 기반 클립 추출 |
| `Daemon` | `daemon.py` | ZMQ 기반 프레임 배포 서버 |
| `Frame` | `core/` | 타임스탬프 포함 프레임 데이터 |

### IPC 인터페이스 (`visualbase.ipc`)

FaceMoment는 visualbase의 IPC 인터페이스를 통해 통신합니다:

```python
from visualbase.ipc.interfaces import VideoReader, MessageSender, MessageReceiver
from visualbase.ipc.factory import TransportFactory

# 프레임 수신 (A→B)
reader = TransportFactory.create_video_reader("fifo", "/tmp/vid_face.mjpg")

# OBS 메시지 송신 (B→C)
sender = TransportFactory.create_message_sender("uds", "/tmp/obs.sock")

# TRIG 메시지 수신 (C→A)
receiver = TransportFactory.create_message_receiver("uds", "/tmp/trig.sock")
```

### 메시지 형식 (`visualbase.ipc.messages`)

| 메시지 | 방향 | 내용 |
|--------|------|------|
| `FaceOBS` | B→C | 얼굴 감지 결과 (bbox, yaw, pitch, expression) |
| `PoseOBS` | B→C | 포즈 감지 결과 (keypoints, hand_raised) |
| `QualityOBS` | B→C | 품질 지표 (blur, brightness, contrast) |
| `TRIGMessage` | C→A | 트리거 이벤트 (시작/끝 시간, reason, score) |

### Daemon 모드 연동

```bash
# visualbase daemon 시작
visualbase daemon --source video.mp4 --address tcp://*:5555

# facemoment에서 연결
facemoment attach tcp://localhost:5555
```

### 클립 추출 흐름

```
1. FaceMoment가 TRIG 메시지 생성
   TRIGMessage(t_start_ns, t_end_ns, reason, score)

2. visualbase ClipExtractor가 TRIG 수신
   clip = extractor.extract(source, t_start_ns, t_end_ns)

3. 클립 파일 저장
   clip.save(output_path)
```

---

## 문제 1: Extractor 처리 시간 불균형

각 Extractor의 처리 시간이 크게 다릅니다. Fusion에서 모든 결과를 기다리면 가장 느린 Extractor가 전체 파이프라인의 병목이 됩니다.

```
시간 →
        0ms      20ms     40ms     60ms     80ms    100ms
        │        │        │        │        │        │
Face    ├────────────────────────────┤                    42ms
        │        │        │        │        │        │
Pose    ├────────────┤                                    22ms
        │        │        │        │        │        │
Gesture ├────────────────────────────────────────────┤    58ms ← 병목!
        │        │        │        │        │        │
Quality ├────┤                                             8ms
        │        │        │        │        │        │
                                                     ▼
                                              Fusion 대기 완료
```

### 컴포넌트별 처리 시간

| Extractor | 일반 시간 | 변동성 | 원인 |
|-----------|-----------|--------|------|
| Face | 30-60ms | 중간 | InsightFace + 표정 분석 |
| Pose | 15-30ms | 낮음 | YOLOv8-Pose 최적화됨 |
| Gesture | 40-80ms | 높음 | MediaPipe Hands + 분류 |
| Quality | 5-10ms | 매우 낮음 | 순수 이미지 분석 |

**문제**: 같은 프레임에 대한 OBS가 서로 다른 시간에 Fusion에 도착합니다.

---

## 문제 2: 프레임 드롭과 백프레셔

처리 속도가 입력 속도를 따라가지 못하면 프레임이 누적되고, 결국 메모리 부족이나 지연이 발생합니다.

```
입력 FPS: 30        처리 FPS: 10
      │                  │
      ▼                  ▼
  ┌───────┐          ┌───────┐
  │Frame 1│──────────│Process│
  │Frame 2│          │       │
  │Frame 3│  ← 누적! │ 처리중│
  │Frame 4│          │       │
  │Frame 5│          │       │
  │Frame 6│          └───────┘
  │  ...  │
  └───────┘
      │
      ▼
  큐 깊이 증가 → 메모리 증가 → 지연 증가
```

### 프레임 드롭 시나리오

```
시간 →   T0      T1      T2      T3      T4      T5
         │       │       │       │       │       │
입력:    F1      F2      F3      F4      F5      F6
         │       │       │       │       │       │
         ▼       ▼       ▼       ▼       ▼       ▼
       ┌───────────────────────────────────────────┐
       │              프레임 큐                     │
       │  [F1] → [F2] → [F3] → ...                 │
       └────────────────┬──────────────────────────┘
                        │
                        ▼
       ┌───────────────────────────────────────────┐
       │         ExtractorProcess                  │
       │                                           │
       │  F1 처리중... (100ms 소요)                │
       │         │                                 │
       │         ▼                                 │
       │  F2, F3 스킵 (프레임 ID 점프 감지)        │  ← 드롭!
       │         │                                 │
       │         ▼                                 │
       │  F4 처리 시작                             │
       └───────────────────────────────────────────┘
```

### 드롭 전략

```
┌─────────────────────────────────────────────────────────────┐
│                    드롭 전략 선택                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Skip Oldest (기본)                                      │
│     큐가 가득 차면 가장 오래된 프레임 버림                   │
│     ┌─────────────────────────────┐                         │
│     │ [F1] [F2] [F3] [F4] [F5]    │ ← F6 도착               │
│     │  ↓                          │                         │
│     │ 버림 [F2] [F3] [F4] [F5] [F6]│                         │
│     └─────────────────────────────┘                         │
│                                                             │
│  2. Skip Intermediate                                       │
│     최신과 가장 오래된 것만 유지                             │
│     ┌─────────────────────────────┐                         │
│     │ [F1] [--] [--] [--] [F5]    │                         │
│     └─────────────────────────────┘                         │
│                                                             │
│  3. Keyframe Only                                           │
│     N 프레임마다 하나만 처리                                 │
│     ┌─────────────────────────────┐                         │
│     │ [F1] [--] [--] [F4] [--] [--]│                        │
│     └─────────────────────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**문제**: 모든 프레임에 모든 Extractor의 OBS가 있지 않습니다.

---

## 문제 3: OBS 동기화 지연

여러 Extractor에서 같은 프레임에 대한 OBS(Observation) 메시지가 서로 다른 시간에 Fusion에 도착합니다.

```
Frame #100 에 대한 OBS 도착 타이밍:

시간 →   0ms     20ms    40ms    60ms    80ms   100ms   120ms
         │       │       │       │       │       │       │
Quality  ●───────────────────────────────────────────────────
         │  도착 (8ms)
         │       │       │       │       │       │       │
Pose     ────────●───────────────────────────────────────────
                 │  도착 (22ms)
                 │       │       │       │       │       │
Face     ────────────────────────●───────────────────────────
                                 │  도착 (45ms)
                                 │       │       │       │
Gesture  ────────────────────────────────────────────●───────
                                                     │  도착 (95ms)
         │       │       │       │       │       │   │
         └───────┴───────┴───────┴───────┴───────┴───┘
                                                     │
                                          Alignment Window
                                             (100ms)
```

일부 감지는 시간에 걸친 패턴 분석이 필요합니다:

| 감지 | 윈도우 | 단위 |
|------|--------|------|
| 손 흔들기 | 500ms | 시간 기반 진동 패턴 |
| 표정 급변 | N/A | 프레임별 (히스토리 사용) |
| 머리 회전 | 100-500ms | 프레임 간 각속도 |

**문제**: 윈도우 기반 분석은 가변적인 지연을 추가합니다.

---

## 해결책: ExtractorOrchestrator

동일 프로세스 내 사용 시 Orchestrator가 병렬 실행 + 타임아웃으로 동기화를 단순화합니다.

```
                    ┌─────────────────────────────────────┐
                    │      ExtractorOrchestrator          │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
        ThreadPool               ThreadPool               ThreadPool
        ┌───────┐               ┌───────┐               ┌───────┐
        │ Face  │               │ Pose  │               │Gesture│
        └───┬───┘               └───┬───┘               └───┬───┘
            │                       │                       │
            │ 42ms                  │ 22ms                  │ 58ms
            ▼                       ▼                       ▼
        ┌───────────────────────────────────────────────────────┐
        │              as_completed(timeout=150ms)              │
        │                                                       │
        │   Face ──────────────┐                                │
        │   Pose ────┐         │                                │
        │   Quality ─┤         │                                │
        │            ▼         ▼                                │
        │          수집      수집                               │
        │                                      Gesture 도착     │
        │                                          │            │
        │                                          ▼            │
        │                                        수집           │
        └───────────────────────────────────────────────────────┘
                                      │
                                      ▼
                              List[Observation]
```

### 알고리즘: 타임아웃 기반 수집

```python
class ExtractorOrchestrator:
    def extract_all(self, frame):
        # 모든 Extractor를 스레드 풀에 제출
        futures = {executor.submit(ext.extract, frame): ext for ext in self._extractors}
        observations = []

        # 타임아웃으로 결과 수집
        for future in as_completed(futures, timeout=self._timeout):
            try:
                obs = future.result()
                observations.append(obs)
            except TimeoutError:
                # 늦은 Extractor는 건너뜀
                _hub.emit(FrameDropRecord(reason="timeout"))

        return observations
```

### 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `timeout` | 0.15 | 프레임당 최대 대기 시간 (초) |
| `max_workers` | 4 | ThreadPool 워커 수 |

### 트레이드오프

| 타임아웃 짧게 | 타임아웃 길게 |
|---------------|---------------|
| 빠른 응답 | 더 완전한 데이터 |
| 느린 Extractor 누락 | 전체 지연 증가 |
| 10 FPS 달성 용이 | 정확도 우선 |

---

## 해결책: 시간 윈도우 정렬

분산 처리 (A-B*-C) 환경에서 FusionProcess가 OBS를 정렬합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    FusionProcess                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  OBS 버퍼 (frame_id → List[OBS])                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Frame 98: [Face, Pose, Quality, Gesture] ✓ 완료     │   │
│  │ Frame 99: [Face, Pose, Quality, Gesture] ✓ 완료     │   │
│  │ Frame 100: [Quality, Pose, Face] ← Gesture 대기중    │   │
│  │ Frame 101: [Quality, Pose] ← Face, Gesture 대기중    │   │
│  │ Frame 102: [Quality] ← 도착 중...                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  워터마크: Frame 99 (100ms 경과)                            │
│            │                                                │
│            ▼                                                │
│  Frame 98, 99 처리 → Fusion 결정 → 결과 전송                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 알고리즘: 윈도우 기반 정렬

```python
ALIGNMENT_WINDOW_NS = 100_000_000  # 100ms

def _process_aligned_observations(self):
    current_t_ns = max(self._frame_timestamps.values())

    for frame_id in list(self._obs_buffer.keys()):
        t_ns = self._frame_timestamps[frame_id]
        age_ns = current_t_ns - t_ns

        if age_ns > ALIGNMENT_WINDOW_NS:
            # 100ms 지났으면 도착한 OBS로 처리
            observations = self._obs_buffer.pop(frame_id)
            self._process_frame(frame_id, observations)

            # 동기화 지연 기록
            if age_ns > ALIGNMENT_WINDOW_NS * 1.5:
                missing = self._get_missing_sources(frame_id)
                _hub.emit(SyncDelayRecord(
                    frame_id=frame_id,
                    delay_ms=(age_ns - ALIGNMENT_WINDOW_NS) / 1_000_000,
                    waiting_for=missing
                ))
```

### 동작 방식

1. **OBS 버퍼링**: 수신한 OBS 메시지를 frame_id별로 저장
2. **윈도우 대기**: 100ms 경과 전까지 프레임 처리 보류
3. **가용 데이터 처리**: 윈도우 내 도착한 OBS로 처리
4. **다음 진행**: 처리된 프레임을 버퍼에서 삭제

### 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `alignment_window_ns` | 100,000,000 | 정렬 윈도우 (100ms) |
| `late_arrival_threshold` | 1.5 | 지연 경고 배수 |

### 트레이드오프

| 현재 접근법 (단순 시간 윈도우) | |
|------|------|
| **장점** | **단점** |
| 단순한 구현 | 고정 지연 (100ms) |
| 현재 용도에 적합 | 늦게 도착하는 OBS 누락 |
| 낮은 오버헤드 | 순서 역전 처리 안됨 |

---

## Observability 연동

타이밍/동기화 관련 trace 레코드를 통해 문제를 진단합니다.

### 타이밍 레코드

```python
TimingRecord(
    frame_id=100,
    component="gesture",      # face, pose, gesture, fusion_process, orchestrator
    processing_ms=58.2,
    queue_depth=3,
)

FrameDropRecord(
    frame_id=1500,
    dropped_frame_ids=[1498, 1499],
    reason="backpressure",    # 또는 "timeout"
)

SyncDelayRecord(
    frame_id=100,
    delay_ms=45.0,
    waiting_for=["gesture"],
)
```

### 동기화 상태 시각화 (VERBOSE)

```
┌─────────────────────────────────────────────────────────────┐
│ Sync Status                                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Frame 100:  Face ✓   Pose ✓   Gesture ⏳   Quality ✓       │
│             ├──42ms──┤├─22ms─┤├──??ms──┤   ├──8ms──┤        │
│                                   │                         │
│                                   └─ 대기중 (55ms 경과)     │
│                                                             │
│ Alignment Window: ████████████████░░░░ 55/100ms            │
│                                                             │
│ 최근 지연:                                                  │
│   Frame 95: gesture +45ms                                   │
│   Frame 87: gesture +62ms                                   │
│   Frame 72: face +23ms                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 타이밍 분석 워크플로우

```
┌─────────────────────────────────────────────────────────────────────┐
│                     타이밍 분석 워크플로우                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. trace 수집                                                      │
│     facemoment process video.mp4 --trace normal -o trace.jsonl     │
│                                                                     │
│  2. 병목 식별                                                       │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │ $ cat trace.jsonl | jq -s '                             │    │
│     │     [.[] | select(.record_type=="timing")]              │    │
│     │     | group_by(.component)                              │    │
│     │     | map({                                             │    │
│     │         component: .[0].component,                      │    │
│     │         avg_ms: ([.[].processing_ms] | add / length),   │    │
│     │         max_ms: ([.[].processing_ms] | max)             │    │
│     │       })'                                               │    │
│     └─────────────────────────────────────────────────────────┘    │
│                                                                     │
│     결과:                                                           │
│     [                                                               │
│       {"component": "face", "avg_ms": 42.3, "max_ms": 68.1},       │
│       {"component": "gesture", "avg_ms": 55.7, "max_ms": 112.4},   │
│       {"component": "pose", "avg_ms": 21.8, "max_ms": 35.2}        │
│     ]                                                               │
│                        │                                            │
│                        ▼                                            │
│     gesture가 병목! → GPU 사용 또는 제외 검토                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Trace 명령어

```bash
# 동기화 지연이 발생한 프레임 찾기
cat trace.jsonl | jq 'select(.record_type=="sync_delay")'

# 느린 컴포넌트 확인 (50ms 초과)
cat trace.jsonl | jq 'select(.record_type=="timing") | select(.processing_ms > 50)'

# 프레임 드롭 원인 분석
cat trace.jsonl | jq 'select(.record_type=="frame_drop") | {reason, count: (.dropped_frame_ids | length)}'

# 동기화 지연이 잦은 컴포넌트
cat trace.jsonl | jq 'select(.record_type=="sync_delay") | .waiting_for[]' | sort | uniq -c | sort -rn

# 시간에 따른 큐 깊이 분석
cat trace.jsonl | jq 'select(.record_type=="timing") | {frame: .frame_id, queue: .queue_depth}'
```

### 일반적인 문제와 해결책

```
┌─────────────────────────────────────────────────────────────────────┐
│                        문제 진단 가이드                             │
├──────────────────┬─────────────────────┬────────────────────────────┤
│      증상        │        원인         │          해결책            │
├──────────────────┼─────────────────────┼────────────────────────────┤
│                  │                     │                            │
│ gesture 지연 많음│ CPU에서 MediaPipe   │ GPU 사용 또는              │
│                  │ 느림                │ gesture 제외               │
│                  │                     │                            │
├──────────────────┼─────────────────────┼────────────────────────────┤
│                  │                     │                            │
│ 큐 깊이 계속     │ 처리가 입력보다     │ target_fps 낮추거나        │
│ 증가            │ 느림                │ 프레임 스킵                │
│                  │                     │                            │
├──────────────────┼─────────────────────┼────────────────────────────┤
│                  │                     │                            │
│ OBS 누락        │ Extractor 타임아웃  │ timeout 늘리거나           │
│                  │                     │ 해당 Extractor 제외        │
│                  │                     │                            │
├──────────────────┼─────────────────────┼────────────────────────────┤
│                  │                     │                            │
│ 트리거 지연     │ alignment_window    │ 윈도우 줄이기              │
│                  │ 너무 김             │ (정확도 트레이드오프)      │
│                  │                     │                            │
├──────────────────┼─────────────────────┼────────────────────────────┤
│                  │                     │                            │
│ 불규칙한 FPS    │ GC 또는 I/O 스파이크│ 버퍼 크기 조정,            │
│                  │                     │ 메모리 프로파일링          │
│                  │                     │                            │
└──────────────────┴─────────────────────┴────────────────────────────┘
```

### 타이밍 측정 지점

| 위치 | 측정 내용 | 레코드 유형 |
|------|----------|-------------|
| Extractor.extract() | 단일 추출 시간 | `TimingRecord` component=face/pose/gesture |
| ExtractorProcess._process_frame() | 전체 프레임 처리 | `TimingRecord` component=process_face |
| ExtractorOrchestrator.extract_all() | 병렬 추출 | `TimingRecord` component=orchestrator |
| FusionProcess._process_frame_observations() | Fusion 결정 시간 | `TimingRecord` component=fusion_process |

모든 타이밍은 `time.perf_counter_ns()`를 사용하여 나노초 정밀도로 캡처됩니다.

---

## 설정 가이드

### 실시간 처리 (10 FPS 목표)

```python
FusionProcess(
    alignment_window_ns=100_000_000,  # 100ms - 10 FPS에서 한 프레임
)

ExtractorOrchestrator(
    timeout=0.15,  # 프레임당 최대 150ms
)
```

### 배치 처리 (오프라인)

```python
# 정확도를 위해 더 긴 윈도우 사용 가능
FusionProcess(
    alignment_window_ns=200_000_000,  # 200ms
)

ExtractorOrchestrator(
    timeout=1.0,  # 느린 처리 허용
)
```

### 저지연 (라이브 프리뷰)

```python
# 정확도보다 속도 우선
FusionProcess(
    alignment_window_ns=50_000_000,  # 50ms
)

# 느린 Extractor 건너뛰기
orchestrator = ExtractorOrchestrator(
    extractors=[FaceExtractor(), QualityExtractor()],  # gesture 제외
    timeout=0.08,
)
```

---

## 향후: Bytewax 연동 가능성

더 복잡한 스트리밍 시나리오에서 Bytewax가 제공할 수 있는 기능:

### 이벤트 시간 윈도잉

```python
from bytewax.dataflow import Dataflow
from bytewax.window import TumblingWindow, EventClock

flow = Dataflow("facemoment")

# 이벤트 시간 기반 윈도잉
clock = EventClock(
    lambda obs: obs.t_ns,
    wait_for_system_duration=timedelta(ms=200)
)

flow.collect_window("sync", clock, TumblingWindow(length=timedelta(ms=100)))
```

### 대안: 이벤트 시간 워터마크

```python
# 개념적 - 미구현
class WatermarkSynchronizer:
    def __init__(self, allowed_lateness_ms=200):
        self.allowed_lateness = allowed_lateness_ms
        self.watermark = 0  # 완료 보장된 최저 타임스탬프

    def advance_watermark(self, sources):
        # 워터마크 = min(각 소스의 최신 시간) - allowed_lateness
        self.watermark = min(s.latest_t for s in sources) - self.allowed_lateness
```

### 비교

| 기능 | 현재 | Bytewax 사용 시 |
|------|------|-----------------|
| 윈도우 정렬 | 수동 100ms | TumblingWindow 설정 가능 |
| 늦은 도착 처리 | 드롭 | `allowed_lateness` 파라미터 |
| 순서 역전 | 미처리 | EventClock이 이벤트 시간으로 정렬 |
| 워터마크 | 미구현 | 내장 |

### 구현 경로

1. **1단계** (현재): FusionProcess의 단순 시간 윈도우
2. **2단계**: `StreamSynchronizer` 인터페이스 추상화
3. **3단계**: Bytewax 백엔드 구현 (선택적)
