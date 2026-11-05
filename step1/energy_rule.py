"""
Step 1 · 에너지(작아짐) 조각 판정 규칙 (순수 함수).

CNN이 작아짐이라고 한 3초 조각도 두 경우는 빼고 인정한다.
1) 같은 조각을 연장 모델이 연장으로 잡았을 때: CNN이 말 늘이기를 작아짐으로 헷갈리는 오탐이 대부분이라서.
2) 조각 안에서 앞 1/3보다 뒤 1/3이 fade_db(2 dB)만큼도 작아지지 않았을 때.
긴 답변에서도 맞도록 녹음 전체가 아니라 조각마다 본다. 틀린 지적을 줄이되 진짜 작아짐은 놓치지 않는 값을
검증으로 골랐다(근거와 숫자: docs/anxiety_근거.md). fade_delta는 voice-cnn check_friend.py와 같다.
"""
from __future__ import annotations

import librosa
import numpy as np

HOP = 160


def frame_db(y):
    """프레임별 음량(dB)."""
    return 20 * np.log10(librosa.feature.rms(y=y, frame_length=400, hop_length=HOP)[0] + 1e-9)


def fade_delta(db):
    """말소리 구간을 셋으로 나눠 앞 1/3과 뒤 1/3의 음절 크기(상위 10% dB) 차이(클수록 잘 작아짐)."""
    # 앞뒤 잡음은 말로 잡지 않도록 가장 큰 소리 -20dB 안쪽만, 쉬는 틈은 빼려고 상위 10%로 비교
    idx = np.flatnonzero(db > db.max() - 20)
    third = (idx[-1] - idx[0]) // 3
    if third == 0:  # 말소리가 너무 짧으면 비교할 수 없다
        return 0.0
    return round(np.percentile(db[idx[0]:idx[0] + third], 90) - np.percentile(db[idx[-1] - third:idx[-1] + 1], 90), 1)


def faded(chunk, fade_db):
    """이 조각이 fade_db 이상 작아졌는가."""
    return fade_delta(frame_db(chunk)) >= fade_db


def drop_prolonged(energy_windows, prolong_windows, prolong_threshold):
    """연장으로 잡힌 조각(같은 시작초)의 에너지 확률을 0으로 바꾼 (시작초, 확률) 목록."""
    prolonged = {s for s, p in prolong_windows if p >= prolong_threshold}
    return [(s, 0.0 if s in prolonged else p) for s, p in energy_windows]
