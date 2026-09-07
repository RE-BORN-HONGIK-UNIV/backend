"""표정(blendshape) 기반 분석 — 미소 빈도 / 긴장(찡그림) 빈도.

MediaPipe FaceLandmarker가 뽑아주는 52개 ARKit 스타일 blendshape 중,
비언어적 소통 코칭에 의미 있는 신호만 골라 쓴다.
- 미소: mouthSmileLeft / mouthSmileRight
- 긴장: browDownLeft / browDownRight(미간 찌푸림), eyeSquintLeft / eyeSquintRight(눈 찡그림)
"""

SMILE_KEYS = ["mouthSmileLeft", "mouthSmileRight"]
TENSION_KEYS = ["browDownLeft", "browDownRight", "eyeSquintLeft", "eyeSquintRight"]

SMILE_THRESHOLD = 0.35
TENSION_THRESHOLD = 0.4


def _avg(blendshapes, keys):
    vals = [blendshapes.get(k, 0.0) for k in keys]
    return sum(vals) / len(vals) if vals else 0.0


def compute_expression_series(frames):
    """프레임 리스트 → 시간별 (t, smile_score, tension_score) 리스트.
    얼굴 미검출 프레임은 (t, None, None)."""
    series = []
    for f in frames:
        bs = f.get("blendshapes")
        if bs is None:
            series.append((f["t"], None, None))
            continue
        series.append((f["t"], _avg(bs, SMILE_KEYS), _avg(bs, TENSION_KEYS)))
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
