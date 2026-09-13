import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step2.scoring import score_blink_rate, score_expression, score_gaze_segment, score_gaze_segments


def test_blink_rate_normal_range():
    result = score_blink_rate(blink_count=20, duration_sec=60)
    assert result["status"] == "정상"
    assert result["score"] == 100
    assert result["rate_per_min"] == 20.0


def test_blink_rate_frequent():
    result = score_blink_rate(blink_count=35, duration_sec=60)
    assert result["status"] == "빈번"
    assert result["score"] == 75.0


def test_blink_rate_staring():
    result = score_blink_rate(blink_count=5, duration_sec=60)
    assert result["status"] == "과응시"
    assert result["score"] == 75.0


def test_gaze_segment_ideal_range():
    assert score_gaze_segment(3) == 100
    assert score_gaze_segment(5) == 100


def test_gaze_segment_too_short():
    assert score_gaze_segment(1.5) == 35.0


def test_gaze_segment_too_long():
    assert score_gaze_segment(7) == 80


def test_gaze_segments_no_fixation():
    result = score_gaze_segments([{"type": "aversion", "start": 0, "end": 5}])
    assert result == {"avg_fixation_sec": 0, "score": 0}


def test_gaze_segments_with_fixation():
    segments = [
        {"type": "fixation", "start": 0, "end": 4},
        {"type": "aversion", "start": 4, "end": 5},
        {"type": "fixation", "start": 5, "end": 6},
    ]
    result = score_gaze_segments(segments)
    assert result["avg_fixation_sec"] == 2.5
    assert result["score"] == 61.7


def test_expression_relaxed():
    result = score_expression(smile_ratio=0.3, tension_ratio=0.05)
    assert result["status"] == "편안함"


def test_expression_tense():
    result = score_expression(smile_ratio=0.0, tension_ratio=0.5)
    assert result["status"] == "긴장됨"


def test_expression_neutral():
    result = score_expression(smile_ratio=0.1, tension_ratio=0.1)
    assert result["status"] == "보통"
