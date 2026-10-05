import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step2.quality import MIN_FACE_DETECTED_RATIO, check_analyzable
from step2.scoring import score_blink_rate


def _frames(detected_flags, step=0.1):
    """얼굴 검출 여부 리스트 → extract_landmarks_from_video와 같은 모양의 프레임 리스트."""
    return [{"t": round(i * step, 3), "landmarks": object() if d else None} for i, d in enumerate(detected_flags)]


def test_all_detected_passes():
    ok, message, ratio = check_analyzable(_frames([True] * 20))
    assert ok and message is None and ratio == 1.0


def test_empty_frames_rejected():
    ok, message, ratio = check_analyzable([])
    assert not ok and ratio == 0.0 and message


def test_zero_length_video_rejected():
    """프레임이 하나뿐이면 마지막 t가 0 → 깜빡임 비율 계산이 0으로 나누게 된다."""
    ok, message, _ = check_analyzable(_frames([True]))
    assert not ok and message


def test_no_face_at_all_rejected_with_actionable_message():
    ok, message, ratio = check_analyzable(_frames([False] * 20))
    assert not ok and ratio == 0.0
    assert "0%" in message and "다시 촬영" in message


def test_boundary_just_below_rejected_at_threshold_passes():
    below = [True] * 49 + [False] * 51
    at = [True] * 50 + [False] * 50
    assert not check_analyzable(_frames(below))[0]
    assert MIN_FACE_DETECTED_RATIO == 0.5
    assert check_analyzable(_frames(at))[0]


def test_blink_rate_zero_duration_raises_clear_error():
    """이전엔 ZeroDivisionError(→ 500)였다. 호출부가 놓쳐도 원인이 분명한 오류로 막는다."""
    with pytest.raises(ValueError):
        score_blink_rate(0, 0)
