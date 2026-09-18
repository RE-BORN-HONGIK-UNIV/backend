import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step2.expression_analyzer import TENSION_KEYS
from step2.set_baseline import calibrate_baseline_tension


def _frame(tension_val, has_face=True):
    if not has_face:
        return {"blendshapes": None}
    return {"blendshapes": {k: tension_val for k in TENSION_KEYS}}


def test_calibrate_baseline_tension_uses_median():
    frames = [_frame(0.3), _frame(0.5), _frame(0.4)]
    assert calibrate_baseline_tension(frames) == 0.4


def test_calibrate_baseline_tension_ignores_faceless_frames():
    frames = [_frame(0.0, has_face=False), _frame(0.5), _frame(0.5)]
    assert calibrate_baseline_tension(frames) == 0.5


def test_calibrate_baseline_tension_falls_back_to_zero_when_no_face_detected():
    frames = [_frame(0.0, has_face=False), _frame(0.0, has_face=False)]
    assert calibrate_baseline_tension(frames) == 0.0
