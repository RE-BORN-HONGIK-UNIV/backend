"""1단계 · Whisper 잠금 — 같이 쓰는 Whisper 모델을 한 번에 한 요청만 쓰게 줄 세운다.
두 요청이 동시에 단어 시각을 재면 서로의 임시 hook이 엉켜 오류가 나고, 실패한 hook이 남아 다음 요청까지 망가뜨린다.
원인 추적 기록: docs/step1_설계노트.md
"""
import threading

_LOCK = threading.Lock()


def _clear_stale_alignment_hooks(model):
    """단어 시각 계산이 실패해 남은 cross-attention hook을 뗀다(잠금 안에서만 부른다)."""
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
