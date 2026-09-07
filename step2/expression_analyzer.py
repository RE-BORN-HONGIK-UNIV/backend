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
