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


class _Hooked:
    def __init__(self):
        self._forward_hooks = {}


class _Block:
    def __init__(self):
        self.cross_attn = _Hooked()


class PoisonedModel:
    """단어 시각 계산이 한 번 실패해 cross_attn에 hook이 남은 모델 — hook이 남아 있으면 전사가 실패한다."""

    def __init__(self, fail_once=False):
        self.decoder = type("D", (), {})()
        self.decoder.blocks = [_Block(), _Block()]
        self.fail_once = fail_once

    def transcribe(self, audio, **kwargs):
        if any(b.cross_attn._forward_hooks for b in self.decoder.blocks):
            raise TypeError("'NoneType' object is not subscriptable")
        if self.fail_once:
            self.fail_once = False
            self.decoder.blocks[0].cross_attn._forward_hooks[1] = object()  # 실패하며 hook을 남김
            raise TypeError("alignment failed")
        return {"text": audio}


def test_leftover_hooks_from_a_failed_alignment_do_not_break_the_next_request():
    model = PoisonedModel()
    model.decoder.blocks[1].cross_attn._forward_hooks[7] = object()  # 이전 실패로 남은 hook
    assert transcribe(model, "ok") == {"text": "ok"}


def test_a_failure_does_not_poison_later_requests():
    model = PoisonedModel(fail_once=True)
    try:
        transcribe(model, "first")
    except TypeError:
        pass
    assert transcribe(model, "second") == {"text": "second"}
