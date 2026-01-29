# 981park 미디어 플랫폼 개발 계획

## 1. 프로젝트 구조

### 개요
```
981park
├── visualbase (미디어 플랫폼)
│   - 시각 소스 연결, 버퍼 관리, 클립 생성
│
└── portrait981 (서비스)
    ├── MomentDetector (분석 스테이지) ← B-C
    ├── Clip Storage (저장)
    ├── AI Editor (편집/생성)
    └── Delivery (고객 전달)

(향후)
├── trackcam (트랙 클립 편집)
└── ocr-checkin (번호판 인식)
```

### 개발 방식
- **초기**: visualbase를 라이브러리로 개발
- **추후**: 필요시 별도 프로세스로 분리 가능

### visualbase의 본질
```
시각적 정보가 들어오는 곳 (Input)
시각적 정보가 잠시 머무는 곳 (Buffer)
시각적 정보를 가공해서 포장해주는 곳 (Clip-packaging)
```

---

## 2. 아키텍처

### visualbase (플랫폼) + 분석 서비스 분리
```
┌─────────────────────────────────────────────────────────────┐
│  visualbase (플랫폼)                                        │
├─────────────────────────────────────────────────────────────┤
│  INPUT        → BUFFER         → CLIP-PACKAGING            │
│  소스 연결       원본 보관         이벤트 처리              │
│  샘플링          시간 관리         클립 추출                │
│  전처리          조회 제공         원본 품질 출력           │
└─────────────────────────────────────────────────────────────┘
         │                              ▲
         │ 분석용 스트림                │ TRIG
         ▼                              │
┌─────────────────────────────────────────────────────────────┐
│  portrait981 (분석 서비스 = B* + C 세트)                    │
├─────────────────────────────────────────────────────────────┤
│  • visualbase에서 프레임 수신                               │
│  • 자체적으로 B1, B2, B3... 구성 (서비스가 결정)            │
│  • C에서 융합 → TRIG 생성                                  │
│  • TRIG를 visualbase에 전송                                │
└─────────────────────────────────────────────────────────────┘
```

### 핵심 원칙
- **visualbase는 분석 구조를 모름** - 단일 스트림만 제공
- **분석 서비스가 자체 fan-out** - B 모듈 개수/구성은 서비스 책임
- **B+C는 세트** - 분석과 융합은 항상 함께

---

## 3. visualbase 역할 상세

### 3.1 INPUT (소스 연결)
```
Sources:
├── FileSource    │ MP4, 영상 파일
├── CameraSource  │ USB/로컬 카메라
├── RTSPSource    │ IP 카메라, 스트림
└── (확장 가능)   │ 멀티카메라, HLS 등
```

### 3.2 SAMPLING & PREPROCESSING
```
• fps 조절 (30fps → 10fps 등)
• 해상도 조절 (4K → 720p 등)
• 포맷 변환 (YUV → RGB 등)
• 타임스탬프 부여 (t_src)
```

### 3.3 BUFFER (소스 버퍼 관리)
```
파일 모드:
  • 원본 파일 = 버퍼 (seek 가능)

스트림 모드:
  • 링버퍼 (tmpfs)에 원본 품질 세그먼트 보관
  • retention 관리
  • t_src 기준 조회 가능
```

### 3.4 CLIP-PACKAGING (이벤트 처리)
```
TRIG 수신 → 버퍼에서 해당 구간 조회 → 클립 생성

• POINT: event_time ± margin 구간 추출
• RANGE: [start_time, end_time] 구간 추출
• 원본 품질 유지
```

### 3.5 visualbase 인터페이스
```python
class VisualBase:
    # 소스 연결
    def connect(source_config) -> None

    # 분석용 스트림 (샘플링된)
    def get_stream(fps=10, resolution=(640,480)) -> Iterator[Frame]

    # 버퍼 조회 (원본 품질)
    def query_buffer(t_start_ns, t_end_ns) -> BufferSegment

    # 이벤트 처리
    def trigger(trig: Trigger) -> ClipResult
```

---

## 4. 동기화 전략

### 타임스탬프 원칙
- **모든 시간은 원본 기준 (t_src)** - 처리 지연과 무관
- 파일 모드: PTS (Presentation Timestamp)
- 스트림 모드: 캡처 시점 monotonic clock

### 지연 허용
```
A 프레임 캡처    분석 완료        TRIG 발생
    │               │               │
    ▼               ▼               ▼
t_src=0s ───────────────────────────────► 원본 시간축
                                    │
                 (10초 처리 지연)    │
                                    ▼
                        TRIG.event_time = 0s (10s 아님!)
```

### TRIG 유형
```python
# Point Event (순간 이벤트)
TRIG(type="POINT", event_time_ns=..., pre_sec=3.0, post_sec=2.0)
# → 클립: [event_time - pre, event_time + post]

# Range Event (구간 이벤트)
TRIG(type="RANGE", start_time_ns=..., end_time_ns=..., pre_sec=3.0, post_sec=2.0)
# → 클립: [start - pre, end + post]
```

---

## 5. 기술 스택

### 공통
- Python 3.10+
- NVIDIA GPU 활용

### visualbase
- GStreamer (비디오 처리)
- ZeroMQ (IPC)
- Shared Memory (프레임 전송)
- ffmpeg (클립 추출)

### portrait981 (분석 서비스)
- MediaPipe, InsightFace 등 (분석 라이브러리)
- 자체 B 모듈 구성

---

## 6. 프로젝트 디렉토리 구조

### 현재 구현된 구조 (Phase 3 완료)
```
monolith/
├── visualbase/                  # 플랫폼 (라이브러리)
│   ├── pyproject.toml
│   ├── src/visualbase/
│   │   ├── __init__.py
│   │   ├── api.py               # VisualBase 클래스 ✅
│   │   ├── cli.py               # CLI (play, info, clip) ✅
│   │   ├── types.py             # BGRImage 타입 ✅
│   │   ├── sources/
│   │   │   ├── base.py          # BaseSource ✅
│   │   │   └── file.py          # FileSource ✅
│   │   ├── core/
│   │   │   ├── frame.py         # Frame 데이터클래스 ✅
│   │   │   ├── sampler.py       # Sampler ✅
│   │   │   ├── timestamp.py     # 타임스탬프 유틸 ✅
│   │   │   └── buffer.py        # FileBuffer ✅
│   │   ├── packaging/
│   │   │   ├── trigger.py       # Trigger, TriggerType ✅
│   │   │   └── clipper.py       # Clipper, ClipResult ✅
│   │   └── tools/
│   │       └── viewer.py        # FrameViewer ✅
│   └── tests/
│       ├── test_file_source.py  # 12 tests ✅
│       └── test_trigger_clip.py # 15 tests ✅
│
└── portrait981/                 # 서비스
    ├── pyproject.toml
    ├── src/portrait981/
    │   ├── __init__.py
    │   ├── cli.py               # CLI (visualize, process) ✅
    │   ├── moment_detector/
    │   │   ├── detector.py      # MomentDetector ✅
    │   │   ├── extractors/
    │   │   │   ├── base.py      # BaseExtractor, Observation ✅
    │   │   │   └── dummy.py     # DummyExtractor ✅
    │   │   └── fusion/
    │   │       ├── base.py      # BaseFusion, FusionResult ✅
    │   │       └── dummy.py     # DummyFusion ✅
    │   └── tools/
    │       └── visualizer.py    # DetectorVisualizer ✅
    └── tests/
        └── test_moment_detector.py  # 12 tests ✅
```

### 향후 추가 예정
```
visualbase/src/visualbase/sources/
    ├── camera.py            # CameraSource (Phase 5)
    └── rtsp.py              # RTSPSource (Phase 5)

portrait981/src/portrait981/moment_detector/extractors/
    ├── face.py              # FaceExtractor (Phase 4)
    ├── quality.py           # QualityExtractor (Phase 4)
    └── gesture.py           # GestureExtractor (Phase 4)

portrait981/src/portrait981/moment_detector/fusion/
    └── highlight.py         # HighlightFusion (Phase 4)
```

---

## 7. 개발 단계

### Phase 1: visualbase 스켈레톤 ✅ 완료
**목표**: 기본 프레임 스트림 제공

- [x] 프로젝트 구조 생성
- [x] FileSource 구현 (MP4 읽기)
- [x] 기본 샘플링 (fps, resolution)
- [x] 타임스탬프 부여
- [x] 단독 테스트 (프레임 출력 확인)
- [x] FrameViewer 시각화 도구 추가

**테스트**: 12 passed

### Phase 2: visualbase 버퍼 + 클립 ✅ 완료
**목표**: TRIG → 클립 추출

- [x] 파일 모드 버퍼 (FileBuffer - 원본 파일 참조)
- [x] TRIG 메시지 정의 (Trigger, TriggerType)
- [x] 클립 추출 (Clipper - ffmpeg)
- [x] CLI clip 명령 추가
- [x] 수동 TRIG 테스트

**테스트**: 27 passed (누적)

### Phase 3: MomentDetector 스켈레톤 ✅ 완료
**목표**: visualbase 연동 + 더미 분석

- [x] MomentDetector 클래스 구조
- [x] visualbase 연동
- [x] DummyExtractor (더미 B)
- [x] DummyFusion (더미 C)
- [x] DetectorVisualizer 시각화 도구
- [x] CLI visualize/process 명령
- [x] E2E 테스트: 프레임 → 더미 분석 → TRIG → 클립

**테스트**: visualbase 27 + portrait981 12 = 39 passed

### Phase 4: MomentDetector 실제 분석 ⬜ 다음
**목표**: 실제 B, C 로직 구현

- [ ] FaceExtractor (얼굴/표정) - MediaPipe 또는 InsightFace
- [ ] QualityExtractor (품질) - 블러, 조명, 구도
- [ ] GestureExtractor (제스처)
- [ ] HighlightFusion (하이라이트 판정)

### Phase 5: visualbase 스트림 모드
**목표**: 실시간 입력 지원

- [ ] CameraSource / RTSPSource
- [ ] 링버퍼 구현 (tmpfs)
- [ ] 스트림 모드 클립 추출

---

## 8. 검증 전략

### 단위 테스트
- visualbase: 소스별 프레임 출력 확인
- portrait981: 각 Extractor 독립 테스트

### 통합 테스트
- visualbase 단독: 수동 TRIG → 클립 생성
- E2E: MP4 입력 → portrait981 분석 → 클립 출력

### 테스트 데이터
- 1280x720, 30fps, ~2분 MP4 파일
- 얼굴 2명, 다양한 표정/제스처 포함

---

## 9. 미결정 사항 (개발 중 결정)

- [ ] IPC 방식 상세 (ZMQ 패턴, shm 구조)
- [ ] 설정 파일 포맷 (YAML vs TOML)
- [ ] 로깅 포맷 및 레벨
- [ ] 링버퍼 크기 및 retention
