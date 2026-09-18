import numpy as np

# 캘리브레이션 구간에서 얼굴이 한 번도 안 잡혔을 때 쓰는 폴백값 (일반적인 눈 뜬 상태 EAR 근사치).
# 이 경우 blink 판정 정확도는 떨어지지만, 최소한 분석 자체가 죽지는 않게 한다.
DEFAULT_BASELINE_EAR = 0.3


def calibrate_baseline_ear(calibration_frames, left_idx, right_idx):
    """"눈 뜨고 정면 보기" 5초 구간 프레임으로 개인별 baseline EAR 계산.
    캘리브레이션 구간에서 얼굴이 전혀 검출되지 않으면 DEFAULT_BASELINE_EAR로 폴백한다."""
    from .blink_analyzer import compute_ear_series
    series = compute_ear_series(calibration_frames, left_idx, right_idx)
    values = [ear for _, ear in series if ear is not None]
    if not values:
        return DEFAULT_BASELINE_EAR
    return float(np.median(values))  # 중앙값이 평균보다 이상치에 강함


def calibrate_baseline_gaze(calibration_frames, pose_idx, left_iris_idx, right_iris_idx, left_eye_idx, right_eye_idx):
    """"카메라를 정면으로 봐주세요" 캘리브레이션 구간에서 개인별 baseline yaw/pitch 계산.
    웹캠 위치가 얼굴 정면이 아닌 사람(예: 노트북 웹캠을 올려다보는 각도)의 시선 판정 편향을 보정하기 위함.
    얼굴이 전혀 검출되지 않으면 (0.0, 0.0)으로 폴백한다."""
    from .gaze_analyzer import estimate_head_pose
    yaws, pitches = [], []
    for f in calibration_frames:
        if f["landmarks"] is None:
            continue
        yaw, pitch, _ = estimate_head_pose(f["landmarks"], f["frame_size"], pose_idx)
        yaws.append(yaw)
        pitches.append(pitch)
    if not yaws:
        return 0.0, 0.0
    return float(np.median(yaws)), float(np.median(pitches))


def calibrate_baseline_tension(calibration_frames):
    """"정면 보기" 5초 캘리브레이션 구간의 blendshape로 개인별 baseline 긴장 점수 계산.

    1차 검증(ACCURACY_NOTES.md)에서 resting face가 원래 미간이 좁은 인물 1명에게
    라벨이 쏠려 "원래 이런 얼굴"과 "진짜 긴장"을 절대 임계값(TENSION_THRESHOLD)만으로
    구분하지 못한 문제가 있었음 — blink/gaze처럼 baseline 대비 편차로 판정하면 이
    개인차가 캘리브레이션 단계에서 상쇄된다. 얼굴이 전혀 검출되지 않으면 0.0으로
    폴백(=캘리브레이션 이전과 동일하게 절대 임계값처럼 동작)."""
    from .expression_analyzer import TENSION_KEYS, _avg
    values = [_avg(f["blendshapes"], TENSION_KEYS) for f in calibration_frames if f.get("blendshapes")]
    if not values:
        return 0.0
    return float(np.median(values))