import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step3.anxiety import anxiety_score, blink_item, gaze_item, tier


def test_core_items_weigh_four_times_aux():
    # 멈춤(2) 100점 긴장 0, 채움말(0.5) 0점 긴장 100 → 50 / 2.5 = 20
    assert anxiety_score({"pause": 100, "filler": 0}) == 20.0


def test_missing_items_drop_out():
    assert anxiety_score({"pause": 40, "gaze": None, "smile": 0}) == 60.0
    assert anxiety_score({}) is None


def test_tier_bounds():
    assert [tier(s) for s in (None, 60, 59.9, 30, 29.9)] == ["warmup", "warmup", "standard", "standard", "practice"]


def test_gaze_counts_only_looking_away():
    assert gaze_item(9.0) == 100.0  # 오래 바라본 것은 긴장이 아니다
    assert gaze_item(1.5) == 35.0


def test_blink_counts_only_frequent():
    assert blink_item(5) == 100.0   # 적은 깜빡임은 깎지 않는다
    assert blink_item(40) == 50.0
