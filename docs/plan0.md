# Spec: A–B*–C–A Portrait Highlight Capture System (v1)
Goal: Extract impressive 5-second “live portrait” highlight scenes from a real-time camera feed using multiple feature extractor modules (B*) and a fusion module (C). Store ONLY event-surrounding 4K clips (not continuous 4K recording). Must run reliably 365 days, ~10 hours/day.

--------------------------------------------------------------------------------
1) System Overview (Processes & Responsibilities)

A (Ingest/Clock/Buffer/Recorder) – single authority for time and 4K buffering
- Captures camera ONCE.
- Produces:
  (a) 4K@30 hardware-encoded stream into a MEMORY ring buffer (tmpfs) as 1-second segments.
  (b) Proxy analysis streams (low-res / low-fps) fan-out to each B module continuously.
- Receives triggers from C with time windows (t_start/t_end) and saves a 4K clip around that window to persistent storage.
- Performs ring retention cleanup in tmpfs.

B* (Feature Extractors) – multiple independent GPU modules
- Input: continuous proxy stream from A (no clip boundaries).
- Output: observation messages (OBS) containing frame timestamps (from A timeline) and per-face/per-frame signals.
- Must send “event/observation time” based on the input frame time, NOT processing completion time.

C (Fusion/Decision Engine)
- Input: OBS streams from all B modules.
- Aligns observations by timestamp/frame_id and produces “highlight trigger” segments.
- Output: TRIG messages to A: fixed 5.0s highlight window with score and metadata.
- Applies gating (portrait composition quality) and reaction scoring (from any one person) with hysteresis and cooldown.

Data flow:
- Video: A -> B1..Bn (proxy)   (high bandwidth)
- Observations: B* -> C         (low bandwidth)
- Triggers: C -> A              (low bandwidth)

--------------------------------------------------------------------------------
2) IPC & Interfaces (Unix-friendly, simple)

2.1 Video fan-out (A -> B*)
- Use FIFO (named pipes) per B OR UDS stream per B.
- Recommended minimal: FIFO per B with MJPEG payload for easy frame boundary parsing.
  Example paths:
    /tmp/vid_b1.mjpg
    /tmp/vid_b2.mjpg
    ...
- Proxy stream must be CONTINUOUS (no clip segmentation) to support time-window behavior analysis.

2.2 Observations (B* -> C)
- Use Unix Domain Socket (UDS) datagram at: /tmp/obs.sock
- Each OBS is one datagram message. Loss tolerant; design C to handle occasional drops.

2.3 Triggers (C -> A)
- Use UDS datagram at: /tmp/trig.sock
- Each TRIG is one datagram message. A should treat as authoritative.

--------------------------------------------------------------------------------
3) Time & Sync Contract (Critical)

3.1 A is the Time Authority
- Every frame processed in the proxy pipeline must be associated with:
  - frame_id (monotonic increasing integer)
  - t_capture_ns (nanoseconds; monotonic clock recommended; or wallclock epoch_ns if required)
- B modules must propagate these timestamps in OBS.
- C uses (frame_id, t_capture_ns) to align signals. DO NOT use processing completion time.

3.2 Required semantics
- All OBS and TRIG timestamps reference the camera timeline (t_capture_ns).
- If B performs window-based inference (e.g., action recognition), it must report:
  - either the window center time, or best estimate of “event time” within the window,
  - but always in the same t_capture_ns domain.

--------------------------------------------------------------------------------
4) Video Handling & Buffering

4.1 4K Memory Ring Buffer (A)
- A must maintain a rolling buffer of hardware-encoded 4K segments in tmpfs:
  Directory: /dev/shm/ring4k/
  Segment format: MPEG-TS (.ts) preferred for robustness.
  Segment duration: 1 second
  Filename convention: %Y%m%d_%H%M%S.ts  (based on wallclock is acceptable if consistent)
- Retention:
  KEEP_SEC = 120 seconds (must be >= PRE_SEC + TRIGGER_LEN + worst_case_latency + margin)
  Example: PRE_SEC=10, TRIGGER_LEN=5, worst_latency=10 => >= 40; choose 120 for safety.

4.2 Persistent Clip Output (A)
- On TRIG(t_start, t_end= t_start+5s), A extracts from ring:
  clip_window = [t_start - PRE_SEC, t_end + POST_SEC]
  Default PRE_SEC=10, POST_SEC=1
- A concatenates the relevant .ts segments and remuxes to MP4 (prefer -c copy where possible).
- Output directory: /data/events/
- Output filename: <start_time>_<label>_<score>.mp4 + optional JSON metadata.

4.3 Proxy Streams (A -> B*)
- A creates per-B proxy with possibly different scale/fps:
  Examples:
    B1: 640px wide @ 10fps
    B2: 640px wide @ 5fps
    B3: 320px wide @ 2fps
- Proxy encoding: MJPEG stream in FIFO (simple) OR H.264 in MPEG-TS if preferred.
- Proxy must preserve a per-frame mapping to (frame_id, t_capture_ns).
  Options:
    - Embed metadata in a side-channel (recommended simple: A also sends a “frame map” to C or to B).
    - Or if B uses frames sequentially at known rate, B can attach timestamps derived from A-provided time-map messages.
  NOTE: For v1, simplest is:
    - A provides timestamps to B via an additional lightweight UDS datagram “FRAME_MAP” messages keyed by frame_id.
    - B echoes that in OBS. (See section 6.3 for formats.)

--------------------------------------------------------------------------------
5) Highlight Definition (Core Product Behavior)

5.1 Target highlight
- A highlight is a 5.0-second portrait scene.
- Subject count: 1 or 2 people (faces). If 2 people, BOTH must be well-framed.
- A highlight can trigger if BOTH are well-framed AND ANY ONE person shows strong reaction:
  - expression intensity spike vs baseline
  - expression change (transition)
  - head turn (noticeable yaw change)
  - hand wave gesture
- Output is a single TRIG with a fixed 5.0-second window plus score & reason.

5.2 Gating: “Portrait composition quality”
Evaluate over a short smoothing window (e.g., last 0.7s).
Gate conditions:
- face_count must be 1 or 2.
- For each visible face used:
  - face_conf >= T_FACE_CONF
  - inside_frame == true (not clipped)
  - face_area_ratio in [T_AREA_MIN, T_AREA_MAX]
  - abs(yaw) <= T_YAW_MAX, abs(pitch)<=T_PITCH_MAX, abs(roll)<=T_ROLL_MAX
  - center_distance <= T_CENTER_MAX
- If face_count==2:
  - BOTH faces must satisfy all above.
  - IoU(face1, face2) <= T_IOU_MAX (not heavily overlapping)
  - size_ratio(face1, face2) within [T_SIZE_RATIO_MIN, T_SIZE_RATIO_MAX]

Hysteresis:
- GateOpen if satisfied continuously for >= GATE_OPEN_SEC (default 0.7s)
- GateClose if violated continuously for >= GATE_CLOSE_SEC (default 0.3s)

5.3 Reaction signals (any one person can trigger)
Per face i, compute a reaction score in [0..1] from:
- expr_spike_i: z-score of expression intensity vs per-face baseline EWMA
- expr_change_i: magnitude of change in emotion/expr vector vs recent history
- head_turn_i: yaw change rate or cumulative yaw change over short window
- wave_i: hand-wave probability
Define:
  reaction_i = max(expr_spike_i, expr_change_i, head_turn_i, wave_i)
For 2 faces: reaction = max(reaction_1, reaction_2)

Trigger thresholding:
- Require reaction >= R_ON for M consecutive observations (e.g., 3 consecutive @ 10Hz => 0.3s).
- Start time selection:
  t_start = t_first_cross - LEAD_IN_SEC  (default LEAD_IN_SEC = 0.7s)
  t_end   = t_start + 5.0s

Cooldown:
- After a trigger completes, suppress new triggers for COOLDOWN_SEC (default 2.0s)
- Merge policy: optional; since length is fixed, prefer cooldown rather than merge in v1.

5.4 Scoring & reason
During [t_start, t_end]:
- peak_reaction = max_t(reaction(t))
- avg_quality   = avg_t(composition_quality(t))
final_score = 0.7*peak_reaction + 0.3*avg_quality
reason = argmax contributor at peak (wave / expr_spike / head_turn / expr_change)

--------------------------------------------------------------------------------
6) Message Formats (v1 text, one-line, copy/paste friendly)

All messages MUST include:
- frame_id (integer) when tied to a specific frame
- t_capture_ns (int64) for alignment

6.1 OBS message (B* -> C) [UDS datagram /tmp/obs.sock]
One message per observation tick (e.g., 10Hz). Use JSON or key=value.
Recommended minimal key=value single line:
OBS src=<BNAME> frame=<frame_id> t_ns=<t_capture_ns> faces=<n> \
  f0=id:<id>,conf:<c>,x:<x>,y:<y>,w:<w>,h:<h>,inside:<0/1>,yaw:<deg>,pitch:<deg>,roll:<deg>,area:<0..1>,center:<0..1>,expr:<0..1>,exprvec:<optional> \
  f1=...

Optional separate module observations:
OBS src=<BNAME> frame=<frame_id> t_ns=<t_capture_ns> wave_p=<0..1>
OBS src=<BNAME> frame=<frame_id> t_ns=<t_capture_ns> headturn_rate=<deg_per_s>
OBS src=<BNAME> frame=<frame_id> t_ns=<t_capture_ns> expr_change=<0..1>

6.2 TRIG message (C -> A) [UDS datagram /tmp/trig.sock]
TRIG label=PORTRAIT_HIGHLIGHT t_start_ns=<ns> t_end_ns=<ns> \
  faces=<1|2> face_ids=<id[,id]> score=<0..1> reason=<wave|expr_spike|expr_change|head_turn>

6.3 (Optional but recommended) FRAME_MAP message (A -> B* or A -> C)
If proxy frames do not inherently carry timestamps, A sends:
MAP stream=<BNAME> frame=<frame_id> t_ns=<t_capture_ns>
This allows B to attach exact A-timeline timestamps to its OBS.

--------------------------------------------------------------------------------
7) Implementation Requirements (Reliability / Ops)

- All processes managed by systemd (recommended) with Restart=always.
- A must keep running even if any B or C crashes.
- B modules must not block A; if FIFO backpressure occurs, A may drop proxy frames for that B.
- C must tolerate missing/delayed OBS datagrams.
- Logging:
  - Each process logs: drops, latency, trigger count, gate open ratio, clip save success/failure.
- Resource control:
  - B modules may be CPU/GPU heavy; use cgroups to cap memory to avoid system-wide OOM.
- Security:
  - UDS and FIFO paths should have restricted permissions (e.g., 0770) in production.

--------------------------------------------------------------------------------
8) Default Parameter Values (starting point)

KEEP_SEC = 120
PRE_SEC = 10
POST_SEC = 1
TRIGGER_LEN_SEC = 5.0
LEAD_IN_SEC = 0.7
COOLDOWN_SEC = 2.0

GATE_OPEN_SEC = 0.7
GATE_CLOSE_SEC = 0.3

T_FACE_CONF = 0.7
T_AREA_MIN = 0.02      (2% of frame)
T_AREA_MAX = 0.35      (35% of frame)
T_YAW_MAX = 25 deg
T_PITCH_MAX = 20 deg
T_ROLL_MAX = 20 deg
T_CENTER_MAX = 0.35    (normalized distance)
T_IOU_MAX = 0.2
T_SIZE_RATIO_MIN = 0.5
T_SIZE_RATIO_MAX = 2.0

R_ON = 0.75
M_CONSEC = 3

--------------------------------------------------------------------------------
9) Acceptance Criteria (Definition of Done)

1) With a live camera feed, system runs for hours without manual intervention.
2) When two people sit in front:
   - No triggers if either face is clipped/out of frame or badly angled.
   - Triggers occur when BOTH faces are well-framed AND one person waves/smiles strongly/turns head markedly.
3) Trigger produces:
   - TRIG message with t_start/t_end and metadata.
   - A saves a persistent 4K clip containing [t_start-PRE, t_end+POST].
4) If B or C is restarted mid-run, A continues buffering; triggers resume when modules come back.
5) No continuous 4K disk writing except when saving trigger clips.

--------------------------------------------------------------------------------
10) Notes / Future Extensions (non-blocking)
- Replace text messages with protobuf for schema evolution.
- Add “pair tracking” stabilization in C to lock consistent two-face pair.
- Add ranking: if frequent highlights, keep top-K per hour by score.
- Add quality signals: lighting, sharpness, exposure, background clutter.
