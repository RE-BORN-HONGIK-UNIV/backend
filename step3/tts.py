"""
Step 3 · 면접관 음성 변환 (Typecast TTS).

- 질문 텍스트를 받아 mp3 오디오 바이트를 반환한다.
- TYPECAST_API_KEY 가 없거나 호출이 실패하면 None을 반환한다.
  → 호출측(app.py /interview/tts)이 에러 응답을 주고, 프론트는 오디오 없이
    바로 다음 단계(녹화)로 넘어가도록 폴백한다. (화면 흐름은 절대 안 끊기게)
- 면접관(tier)마다 다른 음성을 쓸 수 있도록 음성 ID를 tier별로 분리.
  지금은 세 값 모두 같은 ID로 두고, 목소리 정해지면 .env 값만 교체하면 됨.
"""
from __future__ import annotations

import os
import logging

log = logging.getLogger(__name__)

try:
    from typecast import Typecast
    from typecast.models import LanguageCode, Output, TTSRequest

    _api_key = os.environ.get("TYPECAST_API_KEY")
    _client = Typecast(api_key=_api_key) if _api_key else None
except ImportError:  # typecast-python 미설치 환경에서도 서버는 떠야 함
    Typecast = None
    _client = None

MODEL = os.environ.get("TYPECAST_MODEL", "ssfm-v30")

# 프론트 difficulty.ts의 DifficultyTier와 같은 키 사용 (warmup / standard / practice)
VOICE_IDS = {
    "warmup": os.environ.get("TYPECAST_VOICE_WARMUP"),
    "standard": os.environ.get("TYPECAST_VOICE_STANDARD"),
    "practice": os.environ.get("TYPECAST_VOICE_PRACTICE"),
}
DEFAULT_TIER = "standard"


def _get_voice_id(tier: str | None) -> str | None:
    """tier에 맞는 음성 ID. 없거나 비어 있으면 standard 음성으로 대체."""
    return VOICE_IDS.get(tier or DEFAULT_TIER) or VOICE_IDS[DEFAULT_TIER]


def synthesize_speech(text: str, timeout: float = 20.0, tier: str | None = None) -> bytes | None:
    """
    질문 텍스트 → mp3 바이트. 실패 시 None.
    tier를 안 넘기면 standard 음성 사용 (기존 호출 코드는 수정 없이 그대로 동작).
    timeout은 기존 함수 시그니처 유지용. 현재 SDK 기본 타임아웃 사용.
    """
    voice_id = _get_voice_id(tier)
    if _client is None or not text or not voice_id:
        return None
    try:
        response = _client.text_to_speech(
            TTSRequest(
                text=text,
                voice_id=voice_id,
                model=MODEL,
                language=LanguageCode.KOR,
                # 기존 ElevenLabs 때와 같은 mp3로 받아서 app.py·프론트는 수정 불필요
                output=Output(audio_format="mp3"),
            )
        )
        return response.audio_data
    except Exception as e:  # 네트워크/키/크레딧 부족 등 — 화면 흐름은 절대 안 깨지게
        log.warning("TTS 생성 실패: %s", e)
        return None