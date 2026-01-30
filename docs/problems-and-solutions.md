# 문제와 해결책

FaceMoment에서 하이라이트 순간을 감지하기 위해 사용하는 알고리즘과 기술을 설명합니다.

## 목차

1. [표정 급변 감지](#1-표정-급변-감지)
2. [품질 게이트와 히스테리시스](#2-품질-게이트와-히스테리시스)
3. [머리 회전 감지](#3-머리-회전-감지)
4. [카메라 응시 감지](#4-카메라-응시-감지)
5. [동승자 상호작용 감지](#5-동승자-상호작용-감지)
6. [연속 프레임 확인](#6-연속-프레임-확인)
7. [쿨다운 기간](#7-쿨다운-기간)

> **타이밍/동기화 문제**는 [stream-synchronization.md](stream-synchronization.md)를 참조하세요.

---

## 1. 표정 급변 감지

### 문제

웃음, 놀람 등 표정의 급격한 변화를 감지해야 합니다. 노이즈, 조명 변화, 개인별 기본 표정 차이를 구분해야 하는 것이 핵심 과제입니다.

```
표정값
  │
1.0├─────────────────────────────────────────────●─────
   │                                            ╱
   │                                           ╱
0.5├─────────────────────────────────────────╱─────────
   │                 ●──────●               ╱
   │                ╱        ╲             ╱
   │    ●──────────●          ╲───────────●
0.0├────┴──────────────────────────────────────────────
   │    │          │          │           │
   0   1초        2초        3초        4초    시간
                                          ▲
                                     급변 감지!
                                   (Z-score > 임계값)
```

### 알고리즘: EWMA + Z-Score

```
표정 급변 = Z-Score > 임계값 AND 표정값 > 0.5
```

**지수 가중 이동 평균 (EWMA):**
```python
ewma_new = ewma_old + alpha * (current_value - ewma_old)
ewma_var = (1 - alpha) * (ewma_var + alpha * delta^2)
z_score = (current_value - ewma) / sqrt(ewma_var)
```

### 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `ewma_alpha` | 0.1 | 평활 계수 (0-1). 클수록 빠르게 적응 |
| `expression_z_threshold` | 2.0 | 급변 판정 Z-score 임계값 |

### 트레이드오프

| alpha 높음 | alpha 낮음 |
|------------|------------|
| 변화에 빠르게 반응 | 더 안정적인 기준선 |
| 노이즈에 민감 | 빠른 표정 변화 놓칠 수 있음 |
| 활발한 피사체에 적합 | 미묘한 표정에 적합 |

| z_threshold 높음 | z_threshold 낮음 |
|------------------|------------------|
| 오탐 적음 | 더 많은 트리거 |
| 미묘한 표정 놓침 | 노이즈 오탐 증가 |

### 튜닝 가이드

1. **노이즈 많은 영상**: `ewma_alpha`를 0.15+, `z_threshold`를 2.5+로 증가
2. **미묘한 표정**: `z_threshold`를 1.5로 감소
3. **활발한 피사체**: `ewma_alpha`를 0.05로 감소

---

## 2. 품질 게이트와 히스테리시스

### 문제

적절한 화면 구성(얼굴 위치, 각도, 크기)일 때만 클립을 캡처해야 합니다. 경계 조건에서 게이트가 빠르게 켜졌다 꺼지면 짧고 품질 낮은 클립이 많이 생성됩니다.

```
품질 조건
충족 여부     히스테리시스 없이              히스테리시스 적용
              (떨림 발생)                    (안정적)

    ●─●   ●─●   ●─●                    ●───────────────●
   ╱   ╲ ╱   ╲ ╱   ╲                  ╱                 ╲
  ●     ●     ●     ●                ●                   ●
                                     │←  0.7초  →│
                                        열림 대기

게이트:  ON OFF ON OFF ON OFF         OFF ──→ ON ──────→ OFF
```

### 알고리즘: 히스테리시스 게이트

```
열림 조건: 모든 조건이 gate_open_duration_sec 동안 충족
닫힘 조건: 어떤 조건이라도 gate_close_duration_sec 동안 실패
```

**게이트 조건:**
1. 얼굴 수: 1-2명
2. 얼굴 신뢰도: > `face_conf_threshold`
3. 얼굴 각도: |yaw| < `yaw_max`, |pitch| < `pitch_max`
4. 얼굴 위치: 프레임 내부, 면적 비율 충분, 중앙 근처
5. 품질 지표: 블러, 밝기 적정 범위

### 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `gate_open_duration_sec` | 0.7 | 게이트 열림 대기 시간 |
| `gate_close_duration_sec` | 0.3 | 게이트 닫힘 대기 시간 |
| `face_conf_threshold` | 0.7 | 최소 얼굴 감지 신뢰도 |
| `yaw_max` | 25.0 | 최대 좌우 회전 각도 (도) |
| `pitch_max` | 20.0 | 최대 상하 각도 (도) |

### 트레이드오프

| 열림 대기 길게 | 열림 대기 짧게 |
|----------------|----------------|
| 안정적인 트리거 | 빠른 반응 |
| 짧은 좋은 순간 놓침 | 경계 노이즈 증가 |

| 닫힘 대기 짧게 | 닫힘 대기 길게 |
|----------------|----------------|
| 품질 저하에 빠르게 대응 | 일시적 가림에서 유지 |
| 가림에 민감 | 나쁜 프레임 포함 가능 |

### 비대칭 대기 시간인 이유

- **열림 (0.7초)**: 순간적으로 좋은 프레임에서 트리거 방지
- **닫힘 (0.3초)**: 품질 문제 발생 시 빠르게 중단

---

## 3. 머리 회전 감지

### 문제

피사체가 카메라 방향으로 머리를 돌리는 것을 감지합니다. 관심이나 참여를 나타내는 순간입니다.

```
Yaw 각도
   │
+45├───●
   │    ╲
   │     ╲  각속도 = Δyaw / Δt
   │      ╲    = 45° / 0.5s
   │       ╲   = 90°/s  > 30°/s (임계값)
 0 ├────────●─────────────────────
   │        ▲
   │    트리거!
   │  (카메라 방향으로 회전)
-45├───────────────────────────────
   │
   0      0.5초     1초      시간
```

### 알고리즘: 각속도 임계값

```python
angular_velocity = |yaw_current - yaw_previous| / dt
trigger = angular_velocity > threshold AND yaw가 0에 가까워지는 중
```

### 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `head_turn_velocity_threshold` | 30.0 | 각속도 (도/초) |

### 트레이드오프

| 임계값 높음 | 임계값 낮음 |
|-------------|-------------|
| 빠르고 의도적인 회전만 | 느린 머리 움직임도 포착 |
| 오탐 적음 | 트리거 증가 |

---

## 4. 카메라 응시 감지

### 문제

고카트 시나리오에서 피사체가 카메라를 직접 바라볼 때를 감지합니다. 인식과 참여를 나타냅니다.

```
                    Pitch
                      │
                   +15°├─────────────────────────
                      │         ╲
                      │          ╲  gaze_score
                      │           ╲  감소 영역
                      │            ╲
                    0°├─────────────●─────────────
                      │      ●───────────●
                      │       ╲ 고득점 ╱
                      │        ╲영역 ╱
                   -15°├─────────╲──╱───────────
                      │
                      └──────────┼──────────────
                              -10°  0°  +10°  Yaw

                      gaze_score = yaw_score × pitch_score
                                 = (1 - |yaw|/10) × (1 - |pitch|/15)
```

### 알고리즘: 각도 기반 점수

```python
yaw_score = max(0, 1 - |yaw| / yaw_threshold)
pitch_score = max(0, 1 - |pitch| / pitch_threshold)
gaze_score = yaw_score * pitch_score
trigger = gaze_score > score_threshold
```

### 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `gaze_yaw_threshold` | 10.0 | 최대 yaw (만점 기준) |
| `gaze_pitch_threshold` | 15.0 | 최대 pitch (만점 기준) |
| `gaze_score_threshold` | 0.5 | 트리거 최소 점수 |

### 트레이드오프

각도 임계값이 작을수록 정확한 카메라 정렬이 필요하지만 고품질 캡처가 됩니다.

---

## 5. 동승자 상호작용 감지

### 문제

2인 고카트 시나리오에서 동승자들이 서로를 바라볼 때를 감지합니다.

```
    ┌─────────────────────────────────────────────────┐
    │                    카메라 뷰                     │
    │                                                 │
    │     ┌───────┐                   ┌───────┐      │
    │     │ 왼쪽  │    ←── 서로 ──→   │ 오른쪽│      │
    │     │ 탑승자│       응시        │ 탑승자│      │
    │     │       │                   │       │      │
    │     │  yaw  │                   │  yaw  │      │
    │     │ +20°  │                   │ -25°  │      │
    │     └───────┘                   └───────┘      │
    │                                                 │
    │     interaction = left.yaw > 15° AND           │
    │                   right.yaw < -15°              │
    │                                                 │
    └─────────────────────────────────────────────────┘
```

### 알고리즘: 상대 Yaw 체크

```python
# 왼쪽 사람이 오른쪽 보기: 양의 yaw
# 오른쪽 사람이 왼쪽 보기: 음의 yaw
interaction = left_face.yaw > threshold AND right_face.yaw < -threshold
```

### 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `interaction_yaw_threshold` | 15.0 | "상대방 보기" 판정 최소 yaw |

---

## 6. 연속 프레임 확인

### 문제

단일 프레임 트리거는 노이즈나 일시적 조건으로 인한 오탐일 수 있습니다.

```
프레임:    F1    F2    F3    F4    F5    F6    F7
           │     │     │     │     │     │     │
트리거     │     ●     ●     │     ●     ●     ●
감지       │ spike spike│   spike spike spike
           │     │     │     │     │     │     │
연속       │     1     2     │     1     2     3
카운트     │     │     ▲     │     │     │     ▲
           │     │     │     │     │     │     │
                   └─ 2연속                └─ 3연속
                      실패!                   성공! → TRIGGER

                   (consecutive_frames=3 인 경우)
```

### 알고리즘: 프레임 카운팅

```python
if trigger_detected:
    if same_trigger_reason:
        consecutive_count += 1
    else:
        consecutive_count = 1

fire_trigger = consecutive_count >= required
```

### 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `consecutive_frames` | 2 | 필요한 연속 프레임 수 |

### 트레이드오프

| 프레임 많이 필요 | 프레임 적게 필요 |
|------------------|------------------|
| 높은 신뢰도 | 빠른 반응 |
| 트리거 지연 | 오탐 증가 |
| 짧은 순간 놓칠 수 있음 | 빠른 표정에 적합 |

---

## 7. 쿨다운 기간

### 문제

같은 이벤트에 대해 연속 트리거를 방지하여 중복 클립 생성을 막습니다.

```
시간 →   0초     1초     2초     3초     4초     5초
         │       │       │       │       │       │
트리거:  ●───────────────────────●───────────────●
         │  FIRE │       │       │  FIRE │       │  FIRE
         │       │       │       │       │       │
         └───────┴───────┘       └───────┴───────┘
         │←──쿨다운 2초──→│      │←──쿨다운 2초──→│

후보:    ●   ●   ●       ●       ●   ●   ●       ●
                 ▲               ▲       ▲
              차단됨          차단됨   차단됨
           (쿨다운 중)       (쿨다운 중)
```

### 알고리즘: 시간 기반 차단

```python
if last_trigger_time is not None:
    if current_time - last_trigger_time < cooldown:
        return blocked
```

### 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `cooldown_sec` | 2.0 | 트리거 간 최소 시간 (초) |

### 트레이드오프

| 쿨다운 길게 | 쿨다운 짧게 |
|-------------|-------------|
| 클립 적음, 더 구분됨 | 클립 많음, 겹칠 수 있음 |
| 2차 순간 놓칠 수 있음 | 더 많은 커버리지 |

---

## Observability 연동

모든 알고리즘은 디버깅과 튜닝을 위한 trace 레코드를 발생시킵니다.

### 레코드 유형

| 레코드 | 레벨 | 내용 |
|--------|------|------|
| `TriggerFireRecord` | MINIMAL | 발생한 트리거 |
| `GateChangeRecord` | NORMAL | 게이트 상태 전환 |
| `TriggerDecisionRecord` | NORMAL | 후보와 결정 과정 |
| `GateConditionRecord` | VERBOSE | 프레임별 게이트 조건 체크 |

> 타이밍 관련 레코드 (`TimingRecord`, `FrameDropRecord`, `SyncDelayRecord`)는
> [stream-synchronization.md](stream-synchronization.md#observability-연동)를 참조하세요.

### 예시: 표정 급변 민감도 분석

```bash
# trace에서 EWMA 값 추출
cat trace.jsonl | jq 'select(.record_type=="trigger_decision") | {frame: .frame_id, ewma: .ewma_values}'

# 트리거 vs 차단 결정 카운트
cat trace.jsonl | jq -r '.decision' | sort | uniq -c
```

---

## 부록: 파라미터 빠른 참조

### 프로덕션 기본값

```python
HighlightFusion(
    # 게이트
    face_conf_threshold=0.7,
    yaw_max=25.0,
    pitch_max=20.0,
    gate_open_duration_sec=0.7,
    gate_close_duration_sec=0.3,

    # 표정
    expression_z_threshold=2.0,
    ewma_alpha=0.1,

    # 머리 회전
    head_turn_velocity_threshold=30.0,

    # 타이밍
    cooldown_sec=2.0,
    consecutive_frames=2,

    # 고카트 전용
    gaze_yaw_threshold=10.0,
    gaze_pitch_threshold=15.0,
    gaze_score_threshold=0.5,
    interaction_yaw_threshold=15.0,
)
```

### 높은 민감도 모드

```python
# 더 많은 트리거, 낮은 임계값
HighlightFusion(
    expression_z_threshold=1.5,
    head_turn_velocity_threshold=20.0,
    consecutive_frames=1,
    cooldown_sec=1.5,
)
```

### 보수적 모드

```python
# 적지만 고품질 트리거
HighlightFusion(
    expression_z_threshold=2.5,
    consecutive_frames=3,
    cooldown_sec=3.0,
    gate_open_duration_sec=1.0,
)
```
