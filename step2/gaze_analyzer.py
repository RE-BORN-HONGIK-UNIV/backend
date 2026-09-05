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
                          yaw_thresh=15, pitch_thresh=15, iris_thresh=0.15):
    """머리 방향 + 눈동자 상대위치 둘 다 고려 - 고개는 정면인데 눈만 돌린 경우도 잡음"""
    return (abs(yaw) < yaw_thresh and abs(pitch) < pitch_thresh
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


def detect_gaze_segments(frames, pose_idx, left_iris_idx, right_iris_idx, left_eye_idx, right_eye_idx):
    """프레임 시퀀스 → fixation/aversion 연속 구간 리스트"""
    segments = []
    current_state = None
    seg_start = None
    last_t = None

    for f in frames:
        if f["landmarks"] is None:
            continue
        yaw, pitch, _ = estimate_head_pose(f["landmarks"], f["frame_size"], pose_idx)
        ox, oy = compute_iris_offset(f["landmarks"], left_iris_idx, right_iris_idx, left_eye_idx, right_eye_idx)
        looking = is_looking_at_camera(yaw, pitch, ox, oy)
        state = "fixation" if looking else "aversion"

        #fprint(f"[DEBUG] t={f['t']:.2f} yaw={yaw:.1f} pitch={pitch:.1f} ox={ox:.3f} oy={oy:.3f} looking={looking}")

        if state != current_state:
            if current_state is not None:
                segments.append({"type": current_state, "start": seg_start, "end": f["t"]})
            current_state = state
            seg_start = f["t"]
        last_t = f["t"]

    # 루프 끝난 뒤 마지막까지 이어진 구간도 저장
    if current_state is not None and last_t is not None:
        segments.append({"type": current_state, "start": seg_start, "end": last_t})

    return segments