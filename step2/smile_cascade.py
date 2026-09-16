"""OpenCV 내장 Haar Cascade 기반 미소 검출 — blendshape(mouthSmile) 방식이 "말하기"와
"웃기"를 구분 못 하는 문제(ACCURACY_NOTES.md "2차 검증" 참고, jawOpen 게이팅·랜드마크
기하학·지속시간 필터 3종 전부 실패)의 대안으로 검토 중인 모듈.

opencv-python(-headless)가 원래 제공하는 고전적인 Viola-Jones 캐스케이드 분류기라
mediapipe/torch 같은 새 무거운 의존성이 전혀 추가되지 않는다 — cv2는 이미 필수
의존성. 캐스케이드 XML 2개(frontalface, smile)는 OpenCV 공식 레포에서 받은 것으로
합쳐서 약 1.1MB (step2/haarcascade_*.xml).

2026-09-16 실측(면접 영상 1개·13프레임, ACCURACY_NOTES.md "2차 검증" 참고):
minNeighbors 5~15 구간에서 정확도 76.9%(기존 blendshape+jawOpen 게이팅 방식
69.2% 대비 개선), minNeighbors=5~6에서는 recall 100%(진짜 미소 4건 전부 검출,
대신 오탐 늘어남). 다만 표본이 여전히 1영상·1인이라 이 결과만으로 프로덕션
파이프라인(expression_analyzer.py)에 바로 통합하지는 않음 — 별도 검증 모듈로
분리해두고, 통합 여부는 더 많은 인물로 재검증 후 결정.
"""
import os
import cv2

_HAAR_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_CASCADE_PATH = os.path.join(_HAAR_DIR, "haarcascade_frontalface.xml")
SMILE_CASCADE_PATH = os.path.join(_HAAR_DIR, "haarcascade_smile.xml")

_face_cascade = None
_smile_cascade = None


def _get_cascades():
    global _face_cascade, _smile_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        _smile_cascade = cv2.CascadeClassifier(SMILE_CASCADE_PATH)
    return _face_cascade, _smile_cascade


def detect_smile_in_frame(frame_bgr, min_neighbors=10):
    """한 프레임(BGR 이미지)에서 얼굴 하관(입 주변) 영역에 스마일 패턴이 있는지 검출.
    반환: 검출된 스마일 영역 개수(cv2 CascadeClassifier의 원시 출력 — 0이면 미검출,
    얼굴 자체가 안 잡히면 None). min_neighbors가 낮을수록 recall은 높아지고 오탐도
    늘어남 (실측 근거는 모듈 docstring 참고)."""
    face_cascade, smile_cascade = _get_cascades()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # 얼굴 하관(세로 중간~턱)만 잘라서 검색 — 눈/눈썹 패턴을 입으로 오검출하는 걸
    # 줄이는 표준 관행 (OpenCV 공식 예제와 동일)
    roi = gray[y + int(h * 0.5):y + h, x:x + w]
    smiles = smile_cascade.detectMultiScale(roi, scaleFactor=1.7, minNeighbors=min_neighbors, minSize=(20, 20))
    return len(smiles)


def compute_smile_haar_series(video_path, sample_fps=15, min_neighbors=10):
    """`landmark_face_points.extract_landmarks_from_video`와 동일한 샘플링 간격으로
    영상을 별도로 다시 읽어서, 프레임별 Haar 스마일 검출 개수 시계열을 만든다.
    반환: [(t, count_or_None), ...].

    mediapipe 파이프라인과 별개의 video read pass — 아직 expression_analyzer.py의
    프로덕션 판정 로직에 통합되지 않은 검증용 함수다 (모듈 docstring 참고)."""
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, round(orig_fps / sample_fps))

    series = []
    frame_idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_interval == 0:
            t = frame_idx / orig_fps
            series.append((t, detect_smile_in_frame(frame, min_neighbors=min_neighbors)))
        frame_idx += 1
    cap.release()
    return series
