"""수동 1회성 테스트 스크립트 — step2 파이프라인을 실제 영상으로 검증.
사용: python _manual_test_step2.py "<video_path>"
"""
import sys
import time

from step2.landmark_face_points import (
    extract_landmarks_from_video, LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX,
    POSE_LANDMARK_IDX, LEFT_IRIS_IDX, RIGHT_IRIS_IDX,
)
from step2.blink_analyzer import compute_ear_series, detect_blinks
from step2.gaze_analyzer import detect_gaze_segments
from step2.expression_analyzer import compute_expression_series, summarize_expression
from step2.scoring import score_blink_rate, score_gaze_segments, score_expression
from step2.set_baseline import calibrate_baseline_ear

video_path = sys.argv[1]
print(f"[1/5] 영상 로드 + 랜드마크 추출 중: {video_path}")
t0 = time.time()
frames = extract_landmarks_from_video(video_path)
print(f"      프레임 {len(frames)}개, {time.time() - t0:.1f}초 소요")
detected = sum(1 for f in frames if f["landmarks"] is not None)
print(f"      얼굴 검출된 프레임: {detected}/{len(frames)}")

duration_sec = frames[-1]["t"] if frames else 0
print(f"[2/5] 영상 길이: {duration_sec:.1f}초")

calib_frames = [f for f in frames if f["t"] <= 5.0]
baseline_ear = calibrate_baseline_ear(calib_frames, LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX)
print(f"[3/5] baseline EAR: {baseline_ear:.4f}")

ear_series = compute_ear_series(frames, LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX)
blinks = detect_blinks(ear_series, baseline_ear)
blink_result = score_blink_rate(len(blinks), duration_sec)
print(f"[4/5] 깜빡임: {len(blinks)}회 -> {blink_result}")

gaze_segments = detect_gaze_segments(
    frames, POSE_LANDMARK_IDX, LEFT_IRIS_IDX, RIGHT_IRIS_IDX,
    LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX,
)
gaze_result = score_gaze_segments(gaze_segments)
print(f"      시선 세그먼트 {len(gaze_segments)}개 -> {gaze_result}")

expression_series = compute_expression_series(frames)
expression_summary = summarize_expression(expression_series)
expression_result = score_expression(
    expression_summary["smile_ratio"], expression_summary["tension_ratio"]
)
print(f"[5/5] 표정 -> {expression_summary} -> {expression_result}")
