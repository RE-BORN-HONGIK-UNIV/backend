"""POST /analyze/gaze-blink (업로드 영상 분석) 라우트.

mediapipe가 필요한 landmark 추출(extract_landmarks_from_video)만 가짜로 바꿔 끼우고, 그 뒤의
품질 게이트 → 점수 계산 → DB 저장 → 응답은 실제 코드를 탄다. 영상 편집(ffmpeg)·머리 자세
추정(cv2.solvePnP)은 이 테스트의 관심사가 아니라서 결과가 정해진 값을 돌려주도록 막아둔다 —
이 값들의 정확도는 2층(정확도 검증 하네스) 몫이다.
"""
import io
import sys
import types

import numpy as np
import pytest

FAKE_SEGMENTS = [
    {"type": "fixation", "start": 0.0, "end": 4.0},
    {"type": "aversion", "start": 4.0, "end": 4.5},
]


def _frames(detected_flags, step=0.5):
    """얼굴 검출 여부 리스트 → extract_landmarks_from_video와 같은 모양의 프레임 리스트."""
    rng = np.random.default_rng(0)
    frames = []
    for i, detected in enumerate(detected_flags):
        if detected:
            frames.append({
                "t": i * step,
                "landmarks": rng.random((478, 3)) * 100,
                "frame_size": (640, 480),
                "blendshapes": {"eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.0, "mouthSmileLeft": 0.6,
                                "mouthSmileRight": 0.6, "jawOpen": 0.0, "browDownLeft": 0.0, "browDownRight": 0.0},
            })
        else:
            frames.append({"t": i * step, "landmarks": None, "frame_size": (640, 480), "blendshapes": None})
    return frames


@pytest.fixture()
def set_frames(app_module, monkeypatch):
    """분석 라우트가 읽을 프레임을 정하는 함수를 돌려준다."""
    state = {"frames": []}
    fake = types.ModuleType("step2.landmark_face_points")
    fake.extract_landmarks_from_video = lambda path: state["frames"]
    for name in ("LEFT_EYE_EAR_IDX", "RIGHT_EYE_EAR_IDX", "LEFT_IRIS_IDX", "RIGHT_IRIS_IDX"):
        setattr(fake, name, [33, 160, 158, 133, 153, 144])
    fake.POSE_LANDMARK_IDX = {}
    monkeypatch.setitem(sys.modules, "step2.landmark_face_points", fake)

    import step2.gaze_analyzer as gaze_analyzer
    monkeypatch.setattr(gaze_analyzer, "estimate_head_pose", lambda *a, **k: (0.0, 0.0, None))
    monkeypatch.setattr(gaze_analyzer, "detect_gaze_segments", lambda *a, **k: [dict(s) for s in FAKE_SEGMENTS])
    monkeypatch.setattr(app_module, "build_highlight_clip", lambda *a, **k: None)

    def set_(frames):
        state["frames"] = frames

    return set_


def _post(client, headers):
    return client.post(
        "/analyze/gaze-blink",
        data={"file": (io.BytesIO(b"fake-video-bytes"), "clip.mp4")},
        content_type="multipart/form-data",
        headers=headers,
    )


def _stage2_count(app_module):
    with app_module.app.app_context():
        return app_module.Stage2Result.query.count()


def test_missing_file_returns_400(client, auth_headers, set_frames):
    res = client.post("/analyze/gaze-blink", headers=auth_headers)
    assert res.status_code == 400


@pytest.mark.parametrize(
    "label,flags",
    [
        ("프레임 없음", []),
        ("얼굴 0%", [False] * 20),
        ("얼굴 40%", [True] * 8 + [False] * 12),
    ],
)
def test_undetected_face_is_rejected_422_and_not_saved(client, auth_headers, set_frames, app_module, label, flags):
    """얼굴이 거의 안 잡힌 영상이 그럴듯한 점수로 저장돼 이력·3단계 난이도를 오염시키면 안 된다."""
    set_frames(_frames(flags))
    res = _post(client, auth_headers)
    assert res.status_code == 422, label
    body = res.get_json()
    assert body["code"] == "FACE_NOT_DETECTED"
    assert body["error"]  # 사용자에게 보여줄 안내 문구
    assert _stage2_count(app_module) == 0


def test_successful_analysis_returns_all_metrics_and_saves(client, auth_headers, set_frames, app_module):
    set_frames(_frames([True] * 20))
    res = _post(client, auth_headers)
    assert res.status_code == 200, res.get_data(as_text=True)
    body = res.get_json()

    assert set(body) == {"blink", "gaze", "expression", "previous"}
    assert body["previous"] is None
    assert body["blink"]["status"] in ("정상", "빈번", "과응시")
    assert 0 <= body["blink"]["score"] <= 100
    assert body["gaze"]["segments"] == FAKE_SEGMENTS
    assert body["expression"]["smile_ratio"] == 1.0  # 모든 프레임이 미소 임계값(0.35) 이상
    assert body["blink"]["highlight"] is None
    assert _stage2_count(app_module) == 1


def test_second_analysis_returns_previous_and_latest_reflects_it(client, auth_headers, set_frames):
    set_frames(_frames([True] * 20))
    assert _post(client, auth_headers).status_code == 200
    second = _post(client, auth_headers).get_json()
    assert second["previous"] is not None
    assert second["previous"]["smileRatio"] == 1.0

    latest = client.get("/analyze/stage2/latest", headers=auth_headers).get_json()["result"]
    assert latest["expressionStatus"] in ("편안함", "보통", "긴장됨")


def test_analysis_failure_returns_500_without_leaking_traceback(client, auth_headers, set_frames, app_module, monkeypatch):
    set_frames(_frames([True] * 20))
    import step2.gaze_analyzer as gaze_analyzer

    def boom(*a, **k):
        raise RuntimeError("secret internal detail /srv/app/step2/x.py")

    monkeypatch.setattr(gaze_analyzer, "detect_gaze_segments", boom)
    res = _post(client, auth_headers)
    assert res.status_code == 500
    body = res.get_json()
    assert set(body) == {"error"}
    assert "secret internal detail" not in res.get_data(as_text=True)
    assert _stage2_count(app_module) == 0
