import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step1.stage1_score import overall_score


def _scores(**overrides):
    base = {"stability": 80, "fluency": 80, "pause_ctrl": 80, "continuity": 80, "calm": 80}
    return {**base, **overrides}


def test_overall_is_mean_of_five_axes():
    assert overall_score(_scores(stability=100, fluency=60, pause_ctrl=80, continuity=70, calm=90)) == 80


def test_overall_rounds_half_up_like_js_math_round():
    # 평균 80.5 — 프론트 Math.round(80.5)=81 과 같아야 한다 (파이썬 round()는 80으로 내려감)
    assert overall_score(_scores(stability=85, fluency=80, pause_ctrl=80, continuity=80, calm=77.5)) == 81


def test_overall_accepts_extra_keys():
    # /analyze 응답 scores에 축이 아닌 키가 섞여도 5축만 본다
    assert overall_score({**_scores(), "extra": 0}) == 80


def test_overall_missing_axis_raises():
    s = _scores()
    del s["calm"]
    with pytest.raises(KeyError):
        overall_score(s)
