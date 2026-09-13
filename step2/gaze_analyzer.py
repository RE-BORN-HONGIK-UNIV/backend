import cv2
import numpy as np

# 표준 3D 얼굴 모델 좌표 (일반적으로 쓰이는 6점 근사 모델, 단위: mm 상대값)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),        # 코끝
    (0.0, -330.0, -65.0),   # 턱
    (-225.0, 170.0, -135.0),# 왼쪽 눈 바깥쪽 코너
    (225.0, 170.0, -135.0), # 오른쪽 눈 바깥쪽 코너
    (-150.0, -150.0, -125.0), # 왼쪽 입꼬리
    (150.0, -150.0, -125.0),  # 오른쪽 입꼬리
], dtype=np.float64)


def estimate_head_pose(landmarks_2d, frame_size, pose_idx):
    w, h = frame_size
    image_points = np.array([
        landmarks_2d[pose_idx["nose_tip"]][:2],
        landmarks_2d[pose_idx["chin"]][:2],
        landmarks_2d[pose_idx["left_eye_corner"]][:2],
        landmarks_2d[pose_idx["right_eye_corner"]][:2],
        landmarks_2d[pose_idx["left_mouth"]][:2],
        landmarks_2d[pose_idx["right_mouth"]][:2],
    ], dtype=np.float64)

    focal_length = w
    camera_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, _ = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs
    )
    rmat, _ = cv2.Rodrigues(rotation_vec)
    yaw, pitch, roll = cv2.RQDecomp3x3(rmat)[0]

    if yaw > 0:
        yaw = yaw - 180
    else:
        yaw = yaw + 180

    return yaw, pitch, roll


def is_looking_at_camera(yaw, pitch, iris_offset_x, iris_offset_y,
                          baseline_yaw=0.0, baseline_pitch=0.0,
                          yaw_thresh=10, pitch_thresh=10, iris_thresh=0.15):
    """머리 방향 + 눈동자 상대위치 둘 다 고려 - 고개는 정면인데 눈만 돌린 경우도 잡음.
    baseline_yaw/pitch: set_baseline.calibrate_baseline_gaze()로 잡은 개인별 "정면" 기준값.
    웹캠이 얼굴 정면이 아닌 위치에 있는 사람의 시선 판정 편향을 보정한다.

    yaw_thresh/pitch_thresh 근거 (2026-09-12, 이전 15도에서 조정):
    "Cone of Direct Gaze" 연구(Bell Labs 1969 계열, 화상회의 환경 재현 연구까지
    반복 검증) — 사람이 "눈이 마주쳤다"고 인지하는 각도 범위는 수평 4.5도,
    수직 5.5도. 단, 이 연구는 머리+눈을 합친 최종 시선 방향을 측정한 거고
    여기서는 머리 방향(yaw/pitch)과 눈동자 위치(iris_offset)를 따로 측정해서
    AND로 합치는 구조라 4.5/5.5를 그대로 쓸 수는 없음 — 10도는 "이전 15도보다
    문헌에 훨씬 가깝게 좁힌" 절충값. 실제 영상 3개(48프레임 라벨링과 별개로
    시선 3개 영상 전체 재생 검토)로 재검증: 기존 15도에서 놓치던 실제 시선
    회피 구간(예: 3.5~3.8초, 11.9~12.4초)을 10도에서 정확히 잡아냈고, 이미
    잘 잡던 다른 두 영상은 결과가 거의 안 바뀜(과잉 오탐 없음). 그래도 표본이
    작아 정밀한 최적값은 아님 — 더 다양한 인물로 검증 필요."""
    return (abs(yaw - baseline_yaw) < yaw_thresh and abs(pitch - baseline_pitch) < pitch_thresh
            and abs(iris_offset_x) < iris_thresh and abs(iris_offset_y) < iris_thresh)


def compute_iris_offset(landmarks, left_iris_idx, right_iris_idx, left_eye_idx, right_eye_idx):
    """눈 중심 대비 iris(눈동자) 중심의 상대적 치우침 (정규화됨)"""
    def offset_for(iris_idx, eye_idx):
        iris_center = landmarks[iris_idx][:, :2].mean(axis=0)
        eye_pts = landmarks[eye_idx][:, :2]
        eye_center = eye_pts.mean(axis=0)
        eye_width = np.linalg.norm(eye_pts[0] - eye_pts[3]) + 1e-6
        return (iris_center - eye_center) / eye_width
    l = offset_for(left_iris_idx, left_eye_idx)
    r = offset_for(right_iris_idx, right_eye_idx)
    avg = (l + r) / 2
    return avg[0], avg[1]  # (x_offset, y_offset)


def _smooth(prev, new, alpha=0.3):
    """(yaw, pitch, ox, oy) 튜플에 대한 EMA 스무딩 — 프레임 잡음으로 인한 순간적인
    fixation/aversion 뒤집힘(flicker)을 줄인다."""
    if prev is None:
        return new
    return tuple(alpha * n + (1 - alpha) * p for n, p in zip(new, prev))


def _merge_short_segments(segments, min_duration=0.3):
    """min_duration보다 짧은 구간은 노이즈로 보고 이전 구간에 흡수하고,
    그 결과 같은 타입이 연속되면 하나로 합친다 (hysteresis 효과)."""
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


def detect_gaze_segments(frames, pose_idx, left_iris_idx, right_iris_idx, left_eye_idx, right_eye_idx,
                          baseline_yaw=0.0, baseline_pitch=0.0,
                          smoothing_alpha=0.3, min_segment_sec=0.3):
    """프레임 시퀀스 → fixation/aversion 연속 구간 리스트.

    - 얼굴 미검출 프레임은 aversion으로 처리한다 (화면 밖으로 나간 것도 회피로 봄 —
      예전엔 그냥 건너뛰어서 그 시간이 통째로 누락됐음).
    - yaw/pitch/눈동자offset은 EMA로 스무딩해서 프레임 잡음으로 인한 flicker를 줄인다.
    - baseline_yaw/pitch: calibrate_baseline_gaze()로 잡은 개인별 "정면" 기준.
    - 마지막에 min_segment_sec보다 짧게 쪼개진 구간은 이전 구간에 흡수한다.
    """
    segments = []
    current_state = None
    seg_start = None
    last_t = None
    smoothed = None

    for f in frames:
        if f["landmarks"] is None:
            state = "aversion"
        else:
            yaw, pitch, _ = estimate_head_pose(f["landmarks"], f["frame_size"], pose_idx)
            ox, oy = compute_iris_offset(f["landmarks"], left_iris_idx, right_iris_idx, left_eye_idx, right_eye_idx)
            smoothed = _smooth(smoothed, (yaw, pitch, ox, oy), alpha=smoothing_alpha)
            s_yaw, s_pitch, s_ox, s_oy = smoothed
            looking = is_looking_at_camera(s_yaw, s_pitch, s_ox, s_oy, baseline_yaw, baseline_pitch)
            state = "fixation" if looking else "aversion"

        #print(f"[DEBUG] t={f['t']:.2f} state={state}")

        if state != current_state:
            if current_state is not None:
                segments.append({"type": current_state, "start": seg_start, "end": f["t"]})
            current_state = state
            seg_start = f["t"]
        last_t = f["t"]

    # 루프 끝난 뒤 마지막까지 이어진 구간도 저장
    if current_state is not None and last_t is not None:
        segments.append({"type": current_state, "start": seg_start, "end": last_t})

    return _merge_short_segments(segments, min_duration=min_segment_sec)