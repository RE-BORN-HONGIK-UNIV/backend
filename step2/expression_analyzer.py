"""표정(blendshape) 기반 분석 — 미소 빈도 / 긴장(찡그림) 빈도.

MediaPipe FaceLandmarker가 뽑아주는 52개 ARKit 스타일 blendshape 중,
비언어적 소통 코칭에 의미 있는 신호만 골라 쓴다.
- 미소: mouthSmileLeft / mouthSmileRight
- 긴장: browDownLeft / browDownRight(미간 찌푸림), eyeSquintLeft / eyeSquintRight(눈 찡그림)
"""

SMILE_KEYS = ["mouthSmileLeft", "mouthSmileRight"]
TENSION_KEYS = ["browDownLeft", "browDownRight", "eyeSquintLeft", "eyeSquintRight"]
JAW_OPEN_KEY = "jawOpen"

# 근거자료 없는 잠정치 — MediaPipe blendshape 점수 스케일에 대한 엔지니어링 추정.
# 팀 자체 라벨링 데이터(눈으로 봐도 웃음/긴장이 맞는지)로 검증·조정 필요 (2025-09 재검토).
SMILE_THRESHOLD = 0.35
TENSION_THRESHOLD = 0.4

# 2026-09-12 라벨링 검증(3인 48프레임)에서 발견: mouthSmile이 말하느라 입을
# 벌린 순간(발화 조음)에도 높게 반응해서 오탐(20/48)이 심했음. jawOpen이 이
# 임계값을 넘으면 그 프레임의 미소 점수를 0으로 무시하도록 게이팅해서
# 오탐을 20건→5건, 정확도 56.2%→87.5%로 줄임 (evaluate_threshold.py 재검증).
# 표본이 작아 엄밀한 최적값은 아니고, "일단 훨씬 나아짐" 수준의 잠정치.
JAW_OPEN_GATE = 0.05


def _avg(blendshapes, keys):
    vals = [blendshapes.get(k, 0.0) for k in keys]
    return sum(vals) / len(vals) if vals else 0.0


def compute_expression_series(frames):
    """프레임 리스트 → 시간별 (t, smile_score, tension_score) 리스트.
    얼굴 미검출 프레임은 (t, None, None). 입을 벌린(말하는) 순간의 미소 오탐을
    줄이기 위해 jawOpen이 JAW_OPEN_GATE를 넘으면 그 프레임의 미소 점수는 0으로 본다."""
    series = []
    for f in frames:
        bs = f.get("blendshapes")
        if bs is None:
            series.append((f["t"], None, None))
            continue
        smile = 0.0 if bs.get(JAW_OPEN_KEY, 0.0) > JAW_OPEN_GATE else _avg(bs, SMILE_KEYS)
        series.append((f["t"], smile, _avg(bs, TENSION_KEYS)))
    return series


def summarize_expression(series):
    """전체 구간에서 미소/긴장 임계값을 넘긴 프레임 비율."""
    valid = [(smile, tension) for _, smile, tension in series if smile is not None]
    if not valid:
        return {"smile_ratio": 0.0, "tension_ratio": 0.0, "frame_count": 0}

    smile_hits = sum(1 for smile, _ in valid if smile >= SMILE_THRESHOLD)
    tension_hits = sum(1 for _, tension in valid if tension >= TENSION_THRESHOLD)
    n = len(valid)
    return {
        "smile_ratio": round(smile_hits / n, 4),
        "tension_ratio": round(tension_hits / n, 4),
        "frame_count": n,
    }


def _merge_short_segments(segments, min_duration=0.3):
    """min_duration보다 짧은 구간은 이전 구간에 흡수 (gaze_analyzer와 동일한 방식)."""
    if not segments:
        return segments
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        too_short = (seg["end"] - seg["start"]) < min_duration
        same_type = seg["type"] == prev["type"]
        if too_short or same_type:
            prev["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    return merged


def detect_expression_segments(series, min_segment_sec=0.3):
    """시간별 (t, smile, tension) 시리즈 → 'smile'/'tension'/'neutral' 연속 구간 리스트.
    프론트에서 "이 순간 다시 보기" 용으로 쓰는 타임스탬프 — gaze의 fixation/aversion
    세그먼트와 같은 목적. 같은 프레임에서 둘 다 임계값을 넘기면 미소를 우선한다."""
    segments = []
    current_state = None
    seg_start = None
    last_t = None

    for t, smile, tension in series:
        if smile is None:
            state = "neutral"
        elif smile >= SMILE_THRESHOLD:
            state = "smile"
        elif tension >= TENSION_THRESHOLD:
            state = "tension"
        else:
            state = "neutral"

        if state != current_state:
            if current_state is not None:
                segments.append({"type": current_state, "start": seg_start, "end": t})
            current_state = state
            seg_start = t
        last_t = t

    if current_state is not None and last_t is not None:
        segments.append({"type": current_state, "start": seg_start, "end": last_t})

    return _merge_short_segments(segments, min_duration=min_segment_sec)
