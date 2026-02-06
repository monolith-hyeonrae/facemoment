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
8. [프레임 스코어링](#8-프레임-스코어링)
9. [다양성 기반 프레임 선택](#9-다양성-기반-프레임-선택)

> **타이밍/동기화 문제**는 [visualpath/docs/stream-synchronization.md](../../visualpath/docs/stream-synchronization.md)를 참조하세요.

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

GR차량 시나리오에서 피사체가 카메라를 직접 바라볼 때를 감지합니다. 인식과 참여를 나타냅니다.

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

2인 GR차량 시나리오에서 동승자들이 서로를 바라볼 때를 감지합니다.

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

## 8. 프레임 스코어링

### 문제

트리거가 발생한 구간에서 여러 프레임 중 **가장 좋은 프레임**을 선택해야 합니다. 단순히 트리거 순간만 캡처하면 블러, 나쁜 각도, 눈 감김 등의 문제가 있을 수 있습니다.

```
트리거 구간 (±500ms)
         │←─────────────────────────→│
         │                           │
프레임:  F1   F2   F3   F4   F5   F6   F7   F8   F9
         │    │    │    │    │    │    │    │    │
품질:   0.3  0.5  0.8  0.9  0.7  0.4  0.6  0.8  0.5
              │         ▲
              │      최고 점수!
              │      → 선택
```

### 알고리즘: 가중 다중 컴포넌트 스코어링

세 가지 주요 컴포넌트를 가중치 합으로 결합:

```
total_score = w_tech × S_technical + w_action × S_action + w_identity × S_identity
```

```
┌──────────────────────────────────────────────────────────────────┐
│                     Frame Score 구조                              │
├─────────────────┬────────────────────────────────────────────────┤
│   Technical     │  blur, brightness, contrast, face_confidence   │
│   (0.45)        │  → 사진 품질의 기술적 측면                       │
├─────────────────┼────────────────────────────────────────────────┤
│   Action        │  face_direction, expression, composition,      │
│   (0.35)        │  pose_energy                                   │
│                 │  → 미적/동작적 측면                              │
├─────────────────┼────────────────────────────────────────────────┤
│   Identity      │  face_stability, inside_frame,                 │
│   (0.20)        │  track_stability, main_confidence              │
│                 │  → 인물 일관성/안정성                            │
└─────────────────┴────────────────────────────────────────────────┘
```

### Hard Filters (자동 탈락)

스코어링 전에 적용되어 명백히 나쁜 프레임을 즉시 탈락:

```
                  ┌─────────────────┐
    프레임 ──────→│ Hard Filters    │
                  ├─────────────────┤
                  │ 1. no_face      │──→ 얼굴 없음
                  │ 2. severe_blur  │──→ 블러 < 30
                  │ 3. low_conf     │──→ 신뢰도 < 0.5
                  │ 4. head_cutoff  │──→ 얼굴 잘림
                  └────────┬────────┘
                           │ 통과
                           ▼
                  ┌─────────────────┐
                  │ Score 계산      │
                  │ (3 컴포넌트)    │
                  └─────────────────┘
```

| 필터 | 조건 | 이유 |
|------|------|------|
| `no_face` | 얼굴 미검출 | 인물 사진 불가 |
| `severe_blur` | blur_score < 30 | 심하게 흔들림 |
| `low_confidence` | confidence < 0.5 | 불확실한 검출 |
| `head_cutoff` | inside_frame = false | 얼굴 일부 잘림 |

### 세부 스코어 계산

#### Technical Score (기술 품질)

```python
blur_score = min(1.0, blur / 100.0)  # 0~100 → 0~1

# 밝기: 최적 범위 80~180
if brightness < 80:
    brightness_score = brightness / 80
elif brightness > 180:
    brightness_score = max(0, 1 - (brightness - 180) / 75)
else:
    brightness_score = 1.0

contrast_score = min(1.0, contrast / 60.0)  # 60+ = good
```

#### Action Score (미적/동작)

```python
# 얼굴 각도: 정면(0~25°) 선호
if yaw <= 25°:
    yaw_score = 1.0
elif yaw <= 45°:
    yaw_score = 1.0 - (yaw - 25) / 20
else:
    yaw_score = 0.3  # 과도한 각도 페널티

# 표정: 비중립 표정 보너스
if expression > 0.3:
    expr_score = 0.7 + 0.3 × expression
elif happy > 0.3:
    expr_score = 0.8 + 0.2 × happy
else:
    expr_score = 0.5 + 0.3 × (1 - neutral)

# 구도: 중앙 선호
composition_score = max(0.3, 1.0 - center_distance × 1.5)

# 포즈: 손 들기 보너스
if hands_raised > 0:
    pose_score = 0.9
elif person_count > 0:
    pose_score = 0.6
```

#### Identity Score (인물 일관성)

```python
face_stability = confidence
inside_frame_score = 1.0 if inside_frame else 0.5

# FaceClassifier 결과 활용
track_stability = min(1.0, track_length / 10.0)  # 10프레임+ = 만점
main_confidence = main_face.confidence
```

### 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `weight_technical` | 0.45 | 기술 품질 가중치 |
| `weight_action` | 0.35 | 동작/미적 가중치 |
| `weight_identity` | 0.20 | 인물 일관성 가중치 |
| `min_blur_score` | 30.0 | 블러 필터 임계값 |
| `min_face_confidence` | 0.5 | 신뢰도 필터 임계값 |
| `max_head_yaw` | 45.0 | 최대 허용 yaw |
| `max_head_pitch` | 30.0 | 최대 허용 pitch |
| `frontal_yaw_bonus` | 25.0 | 정면 보너스 yaw 범위 |

### 트레이드오프

| 가중치 조정 | 효과 |
|-------------|------|
| Technical ↑ | 선명한 사진 우선, 동적 순간 놓칠 수 있음 |
| Action ↑ | 표정/동작 우선, 약간 흐려도 선택 |
| Identity ↑ | 안정적 인물 우선, 새로운 얼굴에 보수적 |

### 튜닝 가이드

1. **정적인 피사체**: `weight_technical`을 0.55로 증가
2. **활발한 피사체**: `weight_action`을 0.45로 증가
3. **다인 촬영**: `weight_identity`를 0.30으로 증가

---

## 9. 다양성 기반 프레임 선택

### 문제

Top K 프레임 선택 시, 점수만으로 선택하면 **비슷한 프레임이 여러 개** 선택될 수 있습니다.

```
점수 기준만 적용:          다양성 제약 적용:

프레임: F1  F2  F3  F4     프레임: F1  F2  F3  F4
시간:   0s  0.1s 0.2s 2s   시간:   0s  0.1s 0.2s 2s
점수:   0.9 0.88 0.85 0.7  점수:   0.9 0.88 0.85 0.7
        ▲   ▲   ▲                  ▲        ▲   ▲
     top3 선택                  시간 간격 200ms 충족
     (거의 같은 장면)           (다양한 장면)
```

### 알고리즘: Greedy Selection with Constraints

```python
selected = []
candidates.sort(by=score, descending=True)

for candidate in candidates:
    if len(selected) >= max_frames:
        break

    # 시간 간격 체크
    if not check_time_gap(candidate, selected):
        continue

    # (선택적) 포즈 다양성 체크
    if not check_pose_diversity(candidate, selected):
        continue

    selected.append(candidate)

return sorted(selected, by=timestamp)
```

```
점수 순위     시간 간격 체크     최종 선택
━━━━━━━━━    ━━━━━━━━━━━━━    ━━━━━━━━━
1. F3 (0.9)  ✓ (첫 번째)      ✓ 선택
2. F2 (0.88) ✗ (F3와 100ms)   ✗ 탈락
3. F5 (0.85) ✓ (F3와 300ms)   ✓ 선택
4. F1 (0.82) ✗ (F3와 150ms)   ✗ 탈락
5. F8 (0.78) ✓ (F5와 500ms)   ✓ 선택
```

### 시간 간격 제약

```python
min_gap_ns = min_time_gap_ms × 1_000_000

def check_time_gap(candidate, selected):
    for frame in selected:
        gap = abs(candidate.t_ns - frame.t_ns)
        if gap < min_gap_ns:
            return False  # 너무 가까움
    return True
```

### 포즈 다양성 제약 (선택적)

키포인트 위치를 비교하여 비슷한 포즈 필터링:

```python
def compute_pose_similarity(pose1, pose2):
    total_dist = 0
    valid_count = 0

    for kp1, kp2 in zip(pose1, pose2):
        if kp1.confidence < 0.3 or kp2.confidence < 0.3:
            continue
        dist = sqrt((kp1.x - kp2.x)² + (kp1.y - kp2.y)²)
        total_dist += dist
        valid_count += 1

    avg_dist = total_dist / valid_count
    similarity = max(0, 1.0 - avg_dist × 2)  # 0.5 distance = 0 similarity
    return similarity

# similarity > 0.8이면 같은 포즈로 간주하여 탈락
```

### 트리거 주변 선택

트리거 이벤트 주변의 최적 프레임을 선택:

```
                 트리거
                   │
      ←── 500ms ──→│←── 500ms ──→
                   │
프레임: F1  F2  F3  F4  F5  F6  F7  F8
        │   │   │   │   │   │   │   │
점수:   0.6 0.7 0.9 0.8 0.85 0.7 0.6 0.5
                ▲       ▲
              트리거 전  트리거 후
              최고점    2위
```

```python
def select_around_triggers(scored_frames, trigger_times_ns,
                           window_before_ms=500, window_after_ms=500,
                           frames_per_trigger=1):
    all_selected = []
    selected_ids = set()

    for trigger_ns in trigger_times_ns:
        window_start = trigger_ns - window_before_ms × 1_000_000
        window_end = trigger_ns + window_after_ms × 1_000_000

        # 윈도우 내 프레임 필터링 (이미 선택된 것 제외)
        window_frames = [f for f in scored_frames
                        if window_start <= f.t_ns <= window_end
                        and f.frame_id not in selected_ids]

        # Top K 선택
        selected = selector.select(window_frames)

        for frame in selected:
            all_selected.append(frame)
            selected_ids.add(frame.frame_id)

    return sorted(all_selected, by=timestamp)
```

### 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `max_frames` | 5 | 최대 선택 프레임 수 |
| `min_time_gap_ms` | 200 | 최소 시간 간격 (ms) |
| `min_score` | 0.3 | 최소 점수 임계값 |
| `pose_similarity_threshold` | 0.8 | 포즈 유사도 임계값 |
| `enable_pose_diversity` | false | 포즈 다양성 체크 활성화 |

### 트레이드오프

| 시간 간격 길게 | 시간 간격 짧게 |
|----------------|----------------|
| 더 다양한 장면 | 비슷한 장면 많음 |
| 최고 점수 프레임 놓칠 수 있음 | 높은 평균 점수 |
| 이벤트 전후 커버리지 | 특정 순간 집중 |

| max_frames 많이 | max_frames 적게 |
|-----------------|-----------------|
| 선택지 다양 | 엄선된 결과 |
| 후처리 부담 | 놓치는 장면 |

### 사용 예시

```python
from facemoment.moment_detector.scoring import (
    FrameScorer, FrameSelector, ScoredFrame,
    ScoringConfig, SelectionConfig,
)

# 스코어러 설정
scorer = FrameScorer(ScoringConfig(
    weight_technical=0.45,
    weight_action=0.35,
    weight_identity=0.20,
))

# 프레임 스코어링
scored_frames = []
for frame, observations in pipeline_results:
    result = scorer.score(
        face_obs=observations.get("face"),
        pose_obs=observations.get("pose"),
        quality_obs=observations.get("quality"),
    )
    if not result.is_filtered:
        scored_frames.append(ScoredFrame(
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            score_result=result,
        ))

# 다양성 기반 선택
selector = FrameSelector(SelectionConfig(
    max_frames=5,
    min_time_gap_ms=200,
    min_score=0.5,
))

# 방법 1: 전체에서 Top K
best_frames = selector.select(scored_frames)

# 방법 2: 트리거 주변에서 선택
trigger_times = [t.event_time_ns for t in triggers]
best_frames = selector.select_around_triggers(
    scored_frames, trigger_times,
    window_before_ms=500, window_after_ms=500,
    frames_per_trigger=1,
)

print(f"Selected {len(best_frames)} frames:")
for f in best_frames:
    print(f"  Frame {f.frame_id}: score={f.score:.2f} @ {f.t_sec:.2f}s")
```

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
> [visualpath/docs/stream-synchronization.md](../../visualpath/docs/stream-synchronization.md#observability-연동)를 참조하세요.

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

    # GR차량 전용
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

### 프레임 스코어링 기본값

```python
FrameScorer(ScoringConfig(
    # 컴포넌트 가중치 (합계 = 1.0)
    weight_technical=0.45,
    weight_action=0.35,
    weight_identity=0.20,

    # Hard filter 임계값
    enable_hard_filters=True,
    min_blur_score=30.0,
    min_face_confidence=0.5,
    max_head_yaw=45.0,
    max_head_pitch=30.0,

    # Technical 세부 설정
    optimal_brightness_min=80.0,
    optimal_brightness_max=180.0,
    min_contrast=30.0,

    # Action 세부 설정
    frontal_yaw_bonus=25.0,
    expression_boost_threshold=0.3,
))
```

### 프레임 선택 기본값

```python
FrameSelector(SelectionConfig(
    max_frames=5,
    min_time_gap_ms=200.0,
    min_score=0.3,
    pose_similarity_threshold=0.8,
    enable_pose_diversity=False,
))
```

### 선명한 사진 우선 모드

```python
# 정적인 피사체, 고해상도 출력용
FrameScorer(ScoringConfig(
    weight_technical=0.55,
    weight_action=0.30,
    weight_identity=0.15,
    min_blur_score=50.0,  # 더 엄격한 블러 필터
))
```

### 액션 캡처 모드

```python
# 활발한 피사체, 동적 순간 캡처
FrameScorer(ScoringConfig(
    weight_technical=0.35,
    weight_action=0.45,
    weight_identity=0.20,
    min_blur_score=20.0,  # 약간의 블러 허용
    expression_boost_threshold=0.2,  # 낮은 표정도 보너스
))

FrameSelector(SelectionConfig(
    max_frames=8,  # 더 많은 프레임
    min_time_gap_ms=100.0,  # 더 짧은 간격
))
```
