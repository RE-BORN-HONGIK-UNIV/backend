import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step1.report_detail import build_report_detail, energy_bars, merge_windows, prolong_events

SR = 16000


def _tone(sec, amp):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def test_energy_bars_fading_voice_goes_down():
    y = np.concatenate([_tone(1, 0.5), _tone(1, 0.25), _tone(1, 0.1)])
    bars = energy_bars(y, SR, n=3)
    assert bars[0] == 100
    assert bars[0] > bars[1] > bars[2]
    assert abs(bars[2] - 20) <= 2


def test_energy_bars_ignores_quiet_tail():
    # 말이 끝난 뒤의 작은 잡음(-40dB)은 막대에 들어가지 않는다
    y = np.concatenate([_tone(2, 0.5), _tone(2, 0.005)])
    assert min(energy_bars(y, SR, n=4)) >= 95


def test_energy_bars_too_short_is_empty():
    assert energy_bars(_tone(0.1, 0.5), SR) == []


def test_merge_windows_joins_overlapping_and_skips_low():
    windows = [(0.0, 0.9), (1.0, 0.8), (2.0, 0.1), (5.0, 0.7)]
    assert merge_windows(windows, 0.5, 3.0) == [(0.0, 4.0, 0.9), (5.0, 8.0, 0.7)]


def test_prolong_event_picks_longest_word():
    words = [
        {"text": "저는", "start": 0.0, "end": 1.6},
        {"text": "요리하는", "start": 1.7, "end": 2.2},
        {"text": "좋아해요", "start": 2.6, "end": 3.2},
    ]
    assert prolong_events([(0.0, 0.8)], 0.5, 3.0, words) == [{"text": "저는", "start": 0.0, "end": 1.6}]


def test_build_report_detail_without_cnn_models():
    y = _tone(3, 0.3)
    fillers = [{"text": "음", "start": 0.5, "end": 0.8}]
    pauses = [{"start": 1.0, "end": 2.0, "duration": 1.0, "type": "pause"},
              {"start": 1.0, "end": 2.5, "duration": 1.5, "type": "long_pause"}]
    out = build_report_detail(y, SR, fillers, pauses, [], {})
    assert out["filler_events"] == [{"text": "음", "start": 0.5, "end": 0.8}]
    assert out["long_pauses"] == [{"start": 1.0, "end": 2.5, "duration": 1.5}]
    assert out["prolong_detail"] is None and out["tremor_detail"] is None  # 화면은 '준비 중'
    assert len(out["energy_detail"]["sentence_bars"]) == 10


def test_build_report_detail_tremor_curve_is_probability():
    cnn = {"tremor": {"windows": [(0.0, 0.2), (1.0, 0.75)], "threshold": 0.681, "win_sec": 3.0}}
    out = build_report_detail(_tone(3, 0.3), SR, [], [], [], cnn)
    assert out["tremor_detail"] == {"curve": [0.2, 0.75]}
