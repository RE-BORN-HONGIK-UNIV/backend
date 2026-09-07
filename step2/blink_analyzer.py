import numpy as np

def eye_aspect_ratio(eye_pts):
    # eye_pts: 6개 (x,y) 좌표, [outer, top1, top2, inner, bottom1, bottom2] 순서
    p1, p2, p3, p4, p5, p6 = eye_pts
    vertical = np.linalg.norm(np.array(p2) - np.array(p6)) + np.linalg.norm(np.array(p3) - np.array(p5))
    horizontal = np.linalg.norm(np.array(p1) - np.array(p4))
    return vertical / (2.0 * horizontal + 1e-6)


def compute_ear_series(frames, left_idx, right_idx):
    """프레임 리스트 → 시간별 평균 EAR 값 리스트"""
    series = []
    for f in frames:
        if f["landmarks"] is None:
            series.append((f["t"], None))
            continue
        left_pts = f["landmarks"][left_idx][:, :2]
        right_pts = f["landmarks"][right_idx][:, :2]
        ear = (eye_aspect_ratio(left_pts) + eye_aspect_ratio(right_pts)) / 2
        series.append((f["t"], ear))
    return series


def detect_blinks(ear_series, baseline_ear, drop_ratio=0.7, min_blink_duration=0.08):
    """baseline_ear 대비 drop_ratio 이하로 떨어진 연속 구간을 blink로 판정.
    (blendshape가 없는 영상을 위한 폴백 — 카메라 각도/조명에 덜 강건함)"""
    threshold = baseline_ear * drop_ratio
    blinks = []
    in_blink = False
    blink_start = None

    for t, ear in ear_series:
        if ear is None:
            continue
        if ear < threshold and not in_blink:
            in_blink = True
            blink_start = t
        elif ear >= threshold and in_blink:
            in_blink = False
            duration = t - blink_start
            if duration >= min_blink_duration:
                blinks.append({"start": blink_start, "end": t})
    return blinks


def compute_blink_blendshape_series(frames):
    """MediaPipe의 eyeBlinkLeft/Right blendshape 평균 → 시간별 (t, score) 리스트.
    EAR(눈꺼풀 좌표 기하학)보다 카메라 각도·조명에 훨씬 강건한, 학습된 blink 신호.
    blendshape가 없는 프레임(또는 얼굴 미검출)은 (t, None)."""
    series = []
    for f in frames:
        bs = f.get("blendshapes")
        if bs is None:
            series.append((f["t"], None))
            continue
        score = (bs.get("eyeBlinkLeft", 0.0) + bs.get("eyeBlinkRight", 0.0)) / 2
        series.append((f["t"], score))
    return series


def detect_blinks_from_blendshape(blink_series, threshold=0.5, min_blink_duration=0.08):
    """blink blendshape score가 threshold 이상으로 올라간 연속 구간을 blink로 판정."""
    blinks = []
    in_blink = False
    blink_start = None

    for t, score in blink_series:
        if score is None:
            continue
        if score >= threshold and not in_blink:
            in_blink = True
            blink_start = t
        elif score < threshold and in_blink:
            in_blink = False
            duration = t - blink_start
            if duration >= min_blink_duration:
                blinks.append({"start": blink_start, "end": t})
    return blinks