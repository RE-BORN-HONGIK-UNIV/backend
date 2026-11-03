"""analyze_filler()가 Whisper 단어 목록을 돌려주는지, 그 목록이 리포트의 연장 단어까지 이어지는지 본다.

예전에는 반환값에 'words'가 없어서 app.py의 filler_result.get('words', [])가 늘 빈 목록이 됐고,
리포트의 연장 단어(prolong_detail.events[].text)가 오류 없이 항상 ""로 나갔다.
librosa가 필요해서 CI(가벼운 의존성만 설치)에서는 건너뛴다. Whisper는 가짜 모델로 대신한다.
"""
import os
import sys

import numpy as np
import pytest

librosa = pytest.importorskip("librosa")
sf = pytest.importorskip("soundfile")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step1.analyze_filler import analyze_filler
from step1.report_detail import prolong_events

SR = 16000


class FakeWhisper:
    """transcribe()가 정해진 단어 타임스탬프를 돌려주는 가짜 Whisper."""

    def transcribe(self, *_args, **_kwargs):
        return {"segments": [{"words": [
            {"word": " 저는", "start": 0.2, "end": 0.6},
            {"word": " 그래서요", "start": 0.8, "end": 2.4},
            {"word": " 좋아요.", "start": 2.6, "end": 3.0},
        ]}]}


@pytest.fixture
def wav_path(tmp_path):
    t = np.arange(int(3.2 * SR)) / SR
    path = tmp_path / "answer.wav"
    sf.write(path, (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), SR)
    return str(path)


def test_analyze_filler_returns_normalized_words(wav_path):
    result = analyze_filler(wav_path, whisper_model=FakeWhisper())
    # 한글만 남기는 정규화(_normalize_word)를 거친 단어가 시간순으로 나온다
    assert result["words"] == [
        {"text": "저는", "start": 0.2, "end": 0.6},
        {"text": "그래서요", "start": 0.8, "end": 2.4},
        {"text": "좋아요", "start": 2.6, "end": 3.0},
    ]


def test_words_reach_prolong_event_text(wav_path):
    result = analyze_filler(wav_path, whisper_model=FakeWhisper())
    # 0~3초 조각에서 연장이 잡히면, 그 안에서 가장 길게 발음된 단어가 연장 단어가 된다
    events = prolong_events([(0.0, 0.9)], 0.5, 3.0, result["words"])
    assert events == [{"text": "그래서요", "start": 0.8, "end": 2.4}]
