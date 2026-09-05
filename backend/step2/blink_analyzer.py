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
    """baseline_ear 대비 drop_ratio 이하로 떨어진 연속 구간을 blink로 판정"""
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