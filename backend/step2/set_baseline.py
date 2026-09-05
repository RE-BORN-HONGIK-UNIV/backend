import numpy as np

def calibrate_baseline_ear(calibration_frames, left_idx, right_idx):
    """"눈 뜨고 정면 보기" 5초 구간 프레임으로 개인별 baseline EAR 계산"""
    from .blink_analyzer import compute_ear_series
    series = compute_ear_series(calibration_frames, left_idx, right_idx)
    values = [ear for _, ear in series if ear is not None]
    return float(np.median(values))  # 중앙값이 평균보다 이상치에 강함