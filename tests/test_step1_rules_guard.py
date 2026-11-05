"""채움말·멈춤 규칙이 Whisper 환각과 짧은 답변에서 크게 틀린 지적을 하지 않는지 본다.
librosa가 필요해서 CI(가벼운 의존성만 설치)에서는 건너뛴다. Whisper는 가짜 모델로 대신한다.
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("librosa")
sf = pytest.importorskip("soundfile")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step1.analyze_filler import get_word_segments
from step1.analyze_pause import analyze_pause

SR = 16000


class LoopingWhisper:
    """늘인 소리 끝에서 길이 0인 "그"를 반복해 지어내는 가짜 Whisper."""

    def transcribe(self, *_args, **_kwargs):
        loop = [{"word": " 그", "start": 3.12, "end": 3.12}] * 20
        return {"segments": [{"words": [{"word": " 경험으로", "start": 0.3, "end": 1.0}, *loop]}]}


def test_zero_length_words_are_dropped():
    assert [w["text"] for w in get_word_segments(np.zeros(SR, np.float32), LoopingWhisper())] == ["경험으로"]


def test_short_answer_pause_is_not_inflated(tmp_path):
    tone = 0.5 * np.sin(2 * np.pi * 200 * np.arange(SR) / SR).astype(np.float32)
    path = tmp_path / "a.wav"
    sf.write(path, np.concatenate([tone, np.zeros(int(1.5 * SR), np.float32), tone]), SR)  # 3.5초 답, 1.5초 멈춤 한 번
    r = analyze_pause(str(path))
    assert r["long_pause_count"] == 1
    assert 85 < r["pause_score"] < 95  # 30초로 나눠 약 9점 감점 (3.5초로 나누면 약 23점)
