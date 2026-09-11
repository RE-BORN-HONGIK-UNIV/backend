import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step2.blink_analyzer import detect_blinks, detect_blinks_from_blendshape


def test_detect_blinks_ear_finds_dip_below_threshold():
    baseline = 0.3  # threshold = 0.3 * 0.7 = 0.21
    series = [
        (0.0, 0.30), (0.1, 0.30), (0.2, 0.10),
        (0.3, 0.10), (0.4, 0.30), (0.5, 0.30),
    ]
    blinks = detect_blinks(series, baseline_ear=baseline)
    assert len(blinks) == 1
    assert blinks[0] == {"start": 0.2, "end": 0.4}


def test_detect_blinks_ear_ignores_too_short_dip():
    baseline = 0.3
    series = [(0.0, 0.30), (0.05, 0.10), (0.10, 0.30)]  # dip 0.05s < min_blink_duration(0.08)
    assert detect_blinks(series, baseline_ear=baseline) == []


def test_detect_blinks_ear_skips_none_frames():
    baseline = 0.3
    series = [(0.0, 0.30), (0.1, None), (0.2, 0.10), (0.3, 0.30)]
    blinks = detect_blinks(series, baseline_ear=baseline)
    assert len(blinks) == 1


def test_detect_blinks_from_blendshape_finds_high_score():
    series = [
        (0.0, 0.1), (0.1, 0.1), (0.2, 0.8), (0.3, 0.8), (0.4, 0.1), (0.5, 0.1),
    ]
    blinks = detect_blinks_from_blendshape(series)
    assert len(blinks) == 1
    assert blinks[0] == {"start": 0.2, "end": 0.4}
