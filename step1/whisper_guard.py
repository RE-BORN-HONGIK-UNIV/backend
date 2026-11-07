# -*- coding: utf-8 -*-
"""Whisper 모델을 한 번에 한 요청만 쓰게 하는 잠금.

왜: 서버(gunicorn --threads 2)는 요청 두 개를 동시에 처리하는데 Whisper 모델은 하나를 같이 쓴다. 단어 시각
(word_timestamps=True — 채움말 판정·캘리브레이션)을 잴 때 Whisper는 모델의 cross-attention에 임시 hook을 달고
SDPA를 잠시 끄는데(whisper/timing.py disable_sdpa — 클래스 전체에 걸리는 설정), 그동안 다른 요청이 같은 모델로
전사하면 그 hook이 다른 요청의 출력(SDPA라 attention 가중치가 None)을 받아
`TypeError: 'NoneType' object is not subscriptable`로 죽는다. 2026-11-07 배포 서버에서 1단계 목소리 확인
(/api/stage1/calibrate)이 다른 분석과 겹쳐 500으로 실패한 것을 로그로 확인함.

게다가 Whisper(20250625 timing.find_alignment)는 그 임시 hook을 성공했을 때만 떼고(try/finally가 아님), 한 번 실패하면
hook이 같이 쓰는 모델에 그대로 남는다. 그러면 그 뒤의 모든 전사가 남은 hook에서 같은 오류로 죽는다 — 서버를 다시
켜기 전까지 1단계 목소리 확인·분석이 계속 실패한 이유. 그래서 전사 전후로 남은 hook을 떼어 낸다(_clear_stale_alignment_hooks).

그래서 모든 Whisper 전사를 이 잠금으로 줄 세운다. 동시에 들어오면 뒤 요청이 앞 요청을 기다리지만(전사 한 번은
보통 수 초~수십 초), gunicorn timeout(300초) 안이라 실패하지 않고, 틀린 결과·오류가 나는 것보다 낫다.
"""
import threading

_LOCK = threading.Lock()


def _clear_stale_alignment_hooks(model):
    """단어 시각 계산이 실패해 남은 cross-attention hook을 뗀다. 잠금 안에서만 부른다(다른 전사가 없을 때).
    Whisper가 정상일 때 cross_attn 자체에 다는 hook은 단어 시각 계산의 이 hook뿐이라(kv cache hook은 key/value에 닮)
    여기 남아 있는 건 모두 실패하고 남은 것이다."""
    decoder = getattr(model, "decoder", None)
    for block in getattr(decoder, "blocks", None) or []:
        hooks = getattr(getattr(block, "cross_attn", None), "_forward_hooks", None)
        if hooks:
            hooks.clear()


def transcribe(model, audio, **kwargs):
    """model.transcribe(audio, **kwargs)를 잠근 채로 부른다. 전후로 남은 hook을 떼어 한 번의 실패가 다음 요청으로 번지지 않게 한다."""
    with _LOCK:
        _clear_stale_alignment_hooks(model)
        try:
            return model.transcribe(audio, **kwargs)
        finally:
            _clear_stale_alignment_hooks(model)
