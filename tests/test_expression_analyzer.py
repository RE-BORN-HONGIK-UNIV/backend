import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step2.expression_analyzer import detect_expression_segments, summarize_expression


def test_summarize_expression_empty_series():
    result = summarize_expression([(0.0, None, None)])
    assert result == {"smile_ratio": 0.0, "tension_ratio": 0.0, "frame_count": 0}


def test_summarize_expression_counts_ratios():
    series = [
        (0.0, 0.5, 0.1),  # smile hit (>=0.35)
        (0.1, 0.1, 0.5),  # tension hit (>=0.4)
        (0.2, 0.1, 0.1),  # neither
        (0.3, 0.4, 0.0),  # smile hit
    ]
    result = summarize_expression(series)
    assert result["frame_count"] == 4
    assert result["smile_ratio"] == 0.5
    assert result["tension_ratio"] == 0.25


def test_detect_expression_segments_merges_short_neutral_blip():
    series = [
        (0.0, 0.5, 0.0),
        (0.1, 0.5, 0.0),
        (0.15, 0.0, 0.0),  # 0.05s짜리 neutral blip -> min_segment_sec(0.3)보다 짧아서 흡수됨
        (0.2, 0.5, 0.0),
        (0.6, 0.5, 0.0),
    ]
    segments = detect_expression_segments(series, min_segment_sec=0.3)
    assert len(segments) == 1
    assert segments[0]["type"] == "smile"
    assert segments[0]["start"] == 0.0
    assert segments[0]["end"] == 0.6
