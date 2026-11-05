"""에너지 조각 판정 규칙. librosa가 필요해 CI 목록에는 넣지 않는다(analyze_filler 테스트와 같음)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step1.energy_rule import drop_prolonged, faded


def test_energy_rule_sees_fade_inside_window():
    t = np.arange(48000) / 16000
    tone = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    assert faded(tone * np.linspace(1.0, 0.1, 48000, dtype=np.float32), 6.0)  # 끝으로 갈수록 20 dB 작아짐
    assert not faded(tone * 0.5, 6.0)


def test_energy_drops_windows_caught_as_prolongation():
    energy = [(0.0, 0.9), (1.0, 0.8), (2.0, 0.7)]
    prolong = [(0.0, 0.5), (1.0, 0.1), (2.0, 0.36)]
    assert drop_prolonged(energy, prolong, 0.36) == [(0.0, 0.0), (1.0, 0.8), (2.0, 0.0)]
