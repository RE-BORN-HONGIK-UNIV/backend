"""
Step 3 · 질문 음성 변환 (ElevenLabs TTS).

- 질문 텍스트를 받아 mp3 오디오 바이트를 반환한다.
- ELEVENLABS_API_KEY 가 없거나 호출이 실패하면 None을 반환한다.
  → 호출측(app.py /interview/tts)이 에러 응답을 주고, 프론트는 오디오 없이
    바로 다음 단계(녹화)로 넘어가도록 폴백한다. (화면 흐름은 절대 안 끊기게)
"""
from __future__ import annotations

import os
import logging

log = logging.getLogger(__name__)

try:
    from elevenlabs.client import ElevenLabs
    _client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY")) if os.environ.get("ELEVENLABS_API_KEY") else None
except ImportError:  # elevenlabs 미설치 환경에서도 서버는 떠야 함
    ElevenLabs = None
    _client = None

# 기본 제공(premade) 보이스 중 하나. 다른 보이스 쓰고 싶으면 .env에 ELEVENLABS_VOICE_ID로 교체.
VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
MODEL_ID = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")


def synthesize_speech(text: str, timeout: float = 20.0) -> bytes | None:
    """질문 텍스트 → mp3 바이트. 실패 시 None."""
    if _client is None or not text:
        return None
    try:
        audio_stream = _client.text_to_speech.convert(
            voice_id=VOICE_ID,
            model_id=MODEL_ID,
            text=text,
            output_format="mp3_44100_128",
        )
        return b"".join(audio_stream)
    except Exception as e:  # 네트워크/키/쿼터 등 — 화면 흐름은 절대 안 깨지게
        log.warning("TTS 생성 실패: %s", e)
        return None