"""step1/whisper_guard.py — Whisper 전사를 한 번에 하나만(동시 요청이 단어 시각 계산을 깨뜨리던 문제)."""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step1.whisper_guard import transcribe  # noqa: E402


class FakeModel:
    """동시에 몇 개가 전사 중인지 센다."""

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self._count = threading.Lock()

    def transcribe(self, audio, **kwargs):
        with self._count:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self._count:
            self.active -= 1
        return {"text": audio, "kwargs": kwargs}


def test_transcriptions_never_overlap():
    model = FakeModel()
    threads = [threading.Thread(target=transcribe, args=(model, f"a{i}"), kwargs={"language": "ko"}) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert model.max_active == 1


def test_passes_arguments_and_result_through():
    assert transcribe(FakeModel(), "x", language="ko", word_timestamps=True) == {
        "text": "x", "kwargs": {"language": "ko", "word_timestamps": True}}
