import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step2.live_payload import parse_live_result


def _valid(**overrides):
    base = {
        "blinkRatePerMin": 18, "blinkStatus": "정상", "blinkScore": 100,
        "avgFixationSec": 3.6, "gazeScore": 74,
        "smileRatio": 0.12, "tensionRatio": 0.08,
        "expressionScore": 68, "expressionStatus": "보통",
    }
    return {**base, **overrides}


def test_valid_payload_is_normalized_to_floats():
    out = parse_live_result(_valid())
    assert out["blinkRatePerMin"] == 18.0 and isinstance(out["blinkRatePerMin"], float)
    assert out["blinkStatus"] == "정상"


def test_boundary_values_are_accepted():
    out = parse_live_result(_valid(blinkScore=0, smileRatio=1, tensionRatio=0, gazeScore=100))
    assert out["smileRatio"] == 1.0


@pytest.mark.parametrize("body", [None, [], "x", 3])
def test_non_object_body_rejected(body):
    with pytest.raises(ValueError):
        parse_live_result(body)


@pytest.mark.parametrize("key", [
    "blinkRatePerMin", "blinkScore", "avgFixationSec", "gazeScore",
    "smileRatio", "tensionRatio", "expressionScore", "blinkStatus", "expressionStatus",
])
def test_missing_field_rejected(key):
    data = _valid()
    del data[key]
    with pytest.raises(ValueError):
        parse_live_result(data)


@pytest.mark.parametrize("key,value", [
    ("blinkScore", 101), ("gazeScore", -1), ("smileRatio", 1.5), ("tensionRatio", -0.1),
    ("blinkRatePerMin", 301), ("avgFixationSec", -2),
])
def test_out_of_range_rejected(key, value):
    with pytest.raises(ValueError):
        parse_live_result(_valid(**{key: value}))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_rejected(value):
    with pytest.raises(ValueError):
        parse_live_result(_valid(gazeScore=value))


@pytest.mark.parametrize("value", [True, "80", None])
def test_non_number_rejected(value):
    # bool은 isinstance(int)라서 따로 막지 않으면 1.0으로 저장된다
    with pytest.raises(ValueError):
        parse_live_result(_valid(blinkScore=value))


def test_status_must_be_short_nonblank_string():
    for bad in ("", "   ", "가" * 21, 5):
        with pytest.raises(ValueError):
            parse_live_result(_valid(blinkStatus=bad))
