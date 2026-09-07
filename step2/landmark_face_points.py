import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# 눈 주변 6점 EAR용 인덱스 (478 landmark 기준, 왼쪽 눈)
LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]

POSE_LANDMARK_IDX = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}
LEFT_IRIS_IDX = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDX = [473, 474, 475, 476, 477]

MODEL_PATH = "step2/face_landmarker.task"

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True,  # 표정(미소·찡그림 등) 52종 점수 — expression_analyzer.py 에서 사용
)


def extract_landmarks_from_video(video_path, sample_fps=15):
    """영상을 읽어서 프레임별 landmark 좌표 + 표정 blendshape 리스트를 반환.
    반환: [{ "t": 초, "landmarks": (478,3) ndarray 또는 None, "frame_size": (w,h),
             "blendshapes": {카테고리명: score(0~1)} 또는 None }, ...]
    """
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, round(orig_fps / sample_fps))

    results_list = []
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_interval == 0:
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((frame_idx / orig_fps) * 1000)

                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                t_sec = frame_idx / orig_fps

                if result.face_landmarks:
                    lm = result.face_landmarks[0]
                    coords = np.array([[p.x * w, p.y * h, p.z * w] for p in lm])
                    blendshapes = None
                    if result.face_blendshapes:
                        blendshapes = {c.category_name: c.score for c in result.face_blendshapes[0]}
                    results_list.append({
                        "t": t_sec, "landmarks": coords, "frame_size": (w, h),
                        "blendshapes": blendshapes,
                    })
                else:
                    results_list.append({
                        "t": t_sec, "landmarks": None, "frame_size": (w, h),
                        "blendshapes": None,
                    })
            frame_idx += 1
    cap.release()
    return results_list