"""step2 파이프라인(영상 → 랜드마크 추출 → 깜빡임/시선/표정 점수) 자동 회귀 테스트.

manual_test_step2.py(print만 찍고 사람이 눈으로 확인하던 수동 스크립트)를
pytest로 전환 — assert로 대체해서 다음에 이 파이프라인을 건드릴 때(임계값
변경, blendshape 키 추가 등) 회귀를 자동으로 잡아낼 수 있게 한다.

실제 얼굴 영상이 있어야 도는 통합 테스트라 TESTING.md의 "2층"(정확도 검증
하네스)과 같은 성격 — mediapipe 풀버전이 필요하고, 저작권 있는 영상을 레포에
커밋할 수 없어서 CI에는 못 올린다(`.github/workflows/ci.yml`의 test-pure-logic
job은 이 파일을 실행 목록에 넣지 않음 — 안 넣는 한 blanket 실행에도 안 걸림).
로컬에 테스트 영상이 있을 때만 환경변수로 경로를 넘겨서 실행한다:

    STEP2_TEST_VIDEO=/path/to/video.mp4 pytest tests/test_step2_pipeline.py -v

환경변수가 없으면(CI, 또는 영상 없는 로컬) 전부 자동 skip된다. mediapipe/cv2
임포트도 fixture 안에서 지연 임포트해서, 무거운 의존성이 없는 환경에서
`pytest tests/`를 통째로 돌려도 이 파일 때문에 collection 단계에서 죽지 않는다.

**주의**: 여기서 하는 검증은 "파이프라인이 안 깨지고 끝까지 돌면서 값이
말이 되는 범위에 있는가"(구조적 회귀)이지, "임계값이 정확한가"(정확도)가
아니다 — 정확도 검증은 여전히 tests/labeling/ + ACCURACY_NOTES.md 몫이다.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VIDEO_PATH = os.environ.get("STEP2_TEST_VIDEO")

pytestmark = pytest.mark.skipif(
    not VIDEO_PATH,
    reason="STEP2_TEST_VIDEO 환경변수로 실제 영상 경로를 지정해야 실행됨 (docs/TESTING.md 참고)",
)


@pytest.fixture(scope="module")
def pipeline_result():
    from step2.landmark_face_points import (
        extract_landmarks_from_video, LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX,
        POSE_LANDMARK_IDX, LEFT_IRIS_IDX, RIGHT_IRIS_IDX,
    )
    from step2.blink_analyzer import (
        compute_ear_series, detect_blinks,
        compute_blink_blendshape_series, detect_blinks_from_blendshape,
    )
    from step2.gaze_analyzer import detect_gaze_segments
    from step2.expression_analyzer import compute_expression_series, summarize_expression
    from step2.scoring import score_blink_rate, score_gaze_segments, score_expression
    from step2.set_baseline import calibrate_baseline_ear, calibrate_baseline_gaze

    frames = extract_landmarks_from_video(VIDEO_PATH)
    duration_sec = frames[-1]["t"] if frames else 0

    calib_frames = [f for f in frames if f["t"] <= 5.0]
    baseline_ear = calibrate_baseline_ear(calib_frames, LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX)
    baseline_yaw, baseline_pitch = calibrate_baseline_gaze(
        calib_frames, POSE_LANDMARK_IDX, LEFT_IRIS_IDX, RIGHT_IRIS_IDX,
        LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX,
    )

    # blink: EAR 방식 vs blendshape 방식 둘 다 계산해서 실제 라우트(app.py)와
    # 같은 우선순위(blendshape 있으면 그걸 씀)로 최종값을 고른다.
    ear_series = compute_ear_series(frames, LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX)
    blinks_ear = detect_blinks(ear_series, baseline_ear)
    blink_series = compute_blink_blendshape_series(frames)
    has_blendshape = any(score is not None for _, score in blink_series)
    blinks_bs = detect_blinks_from_blendshape(blink_series) if has_blendshape else []
    blinks = blinks_bs if has_blendshape else blinks_ear
    blink_result = score_blink_rate(len(blinks), duration_sec)

    gaze_segments = detect_gaze_segments(
        frames, POSE_LANDMARK_IDX, LEFT_IRIS_IDX, RIGHT_IRIS_IDX,
        LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX,
        baseline_yaw=baseline_yaw, baseline_pitch=baseline_pitch,
    )
    gaze_result = score_gaze_segments(gaze_segments)

    expression_series = compute_expression_series(frames)
    expression_summary = summarize_expression(expression_series)
    expression_result = score_expression(
        expression_summary["smile_ratio"], expression_summary["tension_ratio"]
    )

    return {
        "frames": frames,
        "duration_sec": duration_sec,
        "blinks_ear": blinks_ear,
        "blinks_bs": blinks_bs,
        "has_blendshape": has_blendshape,
        "blink_result": blink_result,
        "gaze_segments": gaze_segments,
        "gaze_result": gaze_result,
        "expression_summary": expression_summary,
        "expression_result": expression_result,
    }


def test_frames_extracted_with_positive_duration(pipeline_result):
    assert len(pipeline_result["frames"]) > 0
    assert pipeline_result["duration_sec"] > 0


def test_face_detected_in_most_frames(pipeline_result):
    # 테스트 영상은 얼굴이 화면에 계속 나오는 걸 전제로 함 — 검출률이 낮으면
    # 랜드마크 모델/전처리가 깨졌다는 신호.
    frames = pipeline_result["frames"]
    detected = sum(1 for f in frames if f["landmarks"] is not None)
    rate = detected / len(frames)
    assert rate > 0.5, f"얼굴 검출률이 너무 낮음: {detected}/{len(frames)} ({rate:.0%})"


def test_blink_result_shape_and_range(pipeline_result):
    result = pipeline_result["blink_result"]
    assert result["rate_per_min"] >= 0
    assert 0 <= result["score"] <= 100
    assert result["status"] in ("정상", "빈번", "과응시")


def test_gaze_segments_cover_video_without_gaps(pipeline_result):
    segments = pipeline_result["gaze_segments"]
    duration = pipeline_result["duration_sec"]
    assert len(segments) > 0
    # 구간들이 서로 이어져서 영상 처음부터 끝까지 빈틈없이 덮어야 함
    assert segments[0]["start"] == pytest.approx(0, abs=0.5)
    assert segments[-1]["end"] == pytest.approx(duration, abs=0.5)
    for a, b in zip(segments, segments[1:]):
        assert a["end"] == pytest.approx(b["start"], abs=0.01)


def test_gaze_result_shape_and_range(pipeline_result):
    result = pipeline_result["gaze_result"]
    assert result["avg_fixation_sec"] >= 0
    assert 0 <= result["score"] <= 100


def test_expression_result_shape_and_range(pipeline_result):
    summary = pipeline_result["expression_summary"]
    result = pipeline_result["expression_result"]
    assert 0 <= summary["smile_ratio"] <= 1
    assert 0 <= summary["tension_ratio"] <= 1
    assert 0 <= result["smile_score"] <= 100
    assert 0 <= result["tension_score"] <= 100
    assert 0 <= result["score"] <= 100
    assert result["status"] in ("긴장됨", "편안함", "보통")
