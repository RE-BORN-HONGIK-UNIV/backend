"""3단계 · 음성 변환(Typecast TTS) — 면접관·하루의 말을 mp3로 바꾼다. 키가 없거나 실패하면 None(프론트는 소리 없이 다음으로 넘어감).
난이도별로 다른 목소리를 쓸 수 있게 나눠 뒀지만 지금은 면접관 목소리 하나를 같이 쓴다.
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

# 키는 프론트 difficulty.ts의 DifficultyTier와 같다. 면접관은 난이도와 상관없이 미리 녹음한 mp3와 같은 "형진" 목소리
INTERVIEWER_VOICE = "tc_682e8798603b4e9ed84074f5"
VOICE_IDS = {
    "warmup": os.environ.get("TYPECAST_VOICE_WARMUP") or INTERVIEWER_VOICE,
    "standard": os.environ.get("TYPECAST_VOICE_STANDARD") or INTERVIEWER_VOICE,
    "practice": os.environ.get("TYPECAST_VOICE_PRACTICE") or INTERVIEWER_VOICE,
    # 1·2단계 대화 상대 하루(경량 웹과 같은 "서현" 목소리). 목소리 ID는 비밀이 아니라 기본값으로 둔다.
    "haru": os.environ.get("TYPECAST_VOICE_HARU") or "tc_69f2e455ea79fd197aa0476f",
}
DEFAULT_TIER = "standard"


def _get_voice_id(tier: str | None) -> str | None:
    """tier에 맞는 목소리 ID를 돌려준다. 없으면 standard 목소리."""
    return VOICE_IDS.get(tier or DEFAULT_TIER) or VOICE_IDS[DEFAULT_TIER]


def synthesize_speech(text: str, timeout: float = 20.0, tier: str | None = None) -> bytes | None:
    """글을 mp3 바이트로 바꾼다. 실패하면 None. timeout은 예전 호출과 맞추려고 남긴 인자(지금은 SDK 기본값 사용)."""
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
                # 예전(ElevenLabs)과 같은 mp3로 받아 app.py·프론트를 그대로 쓴다
                output=Output(audio_format="mp3"),
            )
        )
        return response.audio_data
    except Exception as e:  # 네트워크/키/크레딧 부족 등 — 화면 흐름은 절대 안 깨지게
        log.warning("TTS 생성 실패: %s", e)
        return None