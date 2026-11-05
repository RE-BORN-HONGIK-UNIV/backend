import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step1.energy_rule import drop_prolonged, faded
from step3.anxiety import anxiety_score, blink_item, gaze_item, tier


def test_core_items_weigh_four_times_aux():
    # 멈춤(2) 100점 긴장 0, 채움말(0.5) 0점 긴장 100 → 50 / 2.5 = 20
    assert anxiety_score({"pause": 100, "filler": 0}) == 20.0


def test_missing_items_drop_out():
    assert anxiety_score({"pause": 40, "gaze": None, "smile": 0}) == 60.0
    assert anxiety_score({}) is None


def test_tier_bounds():
    assert [tier(s) for s in (None, 60, 59.9, 30, 29.9)] == ["warmup", "warmup", "standard", "standard", "practice"]


def test_energy_rule_sees_fade_inside_window():
    t = np.arange(48000) / 16000
    tone = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    assert faded(tone * np.linspace(1.0, 0.1, 48000, dtype=np.float32), 6.0)  # 끝으로 갈수록 20 dB 작아짐
    assert not faded(tone * 0.5, 6.0)


def test_energy_drops_windows_caught_as_prolongation():
    energy = [(0.0, 0.9), (1.0, 0.8), (2.0, 0.7)]
    prolong = [(0.0, 0.5), (1.0, 0.1), (2.0, 0.36)]
    assert drop_prolonged(energy, prolong, 0.36) == [(0.0, 0.0), (1.0, 0.8), (2.0, 0.0)]


def test_gaze_counts_only_looking_away():
    assert gaze_item(9.0) == 100.0  # 오래 바라본 것은 긴장이 아니다
    assert gaze_item(1.5) == 35.0


def test_blink_counts_only_frequent():
    assert blink_item(5) == 100.0   # 적은 깜빡임은 깎지 않는다
    assert blink_item(40) == 50.0
