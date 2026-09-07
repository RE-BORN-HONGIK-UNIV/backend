"""수동 1회성 테스트 스크립트 — step2 파이프라인을 실제 영상으로 검증.
새 테스트 영상이 생겨도 파일을 새로 만들지 말고 이 스크립트를 재사용할 것.
사용: python tests/manual_test_step2.py "<video_path>"
"""
import os
import sys
import time

# tests/ 밖(backend 루트)에서도 `step2` 패키지를 import할 수 있게 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

video_path = sys.argv[1]
print(f"[1/6] 영상 로드 + 랜드마크 추출 중: {video_path}")
t0 = time.time()
frames = extract_landmarks_from_video(video_path)
print(f"      프레임 {len(frames)}개, {time.time() - t0:.1f}초 소요")
detected = sum(1 for f in frames if f["landmarks"] is not None)
print(f"      얼굴 검출된 프레임: {detected}/{len(frames)}")

duration_sec = frames[-1]["t"] if frames else 0
print(f"[2/6] 영상 길이: {duration_sec:.1f}초")

calib_frames = [f for f in frames if f["t"] <= 5.0]
baseline_ear = calibrate_baseline_ear(calib_frames, LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX)
baseline_yaw, baseline_pitch = calibrate_baseline_gaze(
    calib_frames, POSE_LANDMARK_IDX, LEFT_IRIS_IDX, RIGHT_IRIS_IDX, LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX,
)
print(f"[3/6] baseline EAR: {baseline_ear:.4f} / baseline yaw: {baseline_yaw:.1f} / baseline pitch: {baseline_pitch:.1f}")

# blink: EAR 방식 vs blendshape 방식 비교
ear_series = compute_ear_series(frames, LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX)
blinks_ear = detect_blinks(ear_series, baseline_ear)
blink_series = compute_blink_blendshape_series(frames)
has_blendshape = any(score is not None for _, score in blink_series)
blinks_bs = detect_blinks_from_blendshape(blink_series) if has_blendshape else []
print(f"[4/6] 깜빡임 — EAR 방식: {len(blinks_ear)}회 / blendshape 방식: {len(blinks_bs)}회 (blendshape 있음: {has_blendshape})")
blinks = blinks_bs if has_blendshape else blinks_ear
blink_result = score_blink_rate(len(blinks), duration_sec)
print(f"      최종 사용({'blendshape' if has_blendshape else 'EAR'}): {blink_result}")

gaze_segments = detect_gaze_segments(
    frames, POSE_LANDMARK_IDX, LEFT_IRIS_IDX, RIGHT_IRIS_IDX, LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX,
    baseline_yaw=baseline_yaw, baseline_pitch=baseline_pitch,
)
gaze_result = score_gaze_segments(gaze_segments)
print(f"[5/6] 시선 세그먼트 {len(gaze_segments)}개 -> {gaze_result}")
for seg in gaze_segments:
    print(f"      {seg['type']:9s} {seg['start']:.2f}s ~ {seg['end']:.2f}s ({seg['end']-seg['start']:.2f}s)")

expression_series = compute_expression_series(frames)
expression_summary = summarize_expression(expression_series)
expression_result = score_expression(
    expression_summary["smile_ratio"], expression_summary["tension_ratio"]
)
print(f"[6/6] 표정 -> {expression_summary} -> {expression_result}")
