"""
LLM 호출 공통 헬퍼 (순수 함수).

effort(생각·토큰 사용량) 설정을 SDK 버전과 무관하게 넘기기 위한 것. `messages.create(output_config=...)`처럼
키워드 인자로 직접 넘기면, requirements.txt가 고정한 anthropic==0.69.0에는 그 인자가 없어서 요청을 보내기도
전에 TypeError가 난다 — 그리고 호출부가 `except Exception`으로 폴백하는 구조라 **에러가 삼켜져서 AI가
한 번도 안 쓰이는데 아무도 모르는** 상태가 된다(질문 생성·1단계 코칭이 그랬다).
`extra_body`는 모든 SDK 버전에서 요청 본문에 그대로 합쳐지므로 안전하다.
"""
from __future__ import annotations

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def effort_options(level: str) -> dict:
    """`messages.create(**effort_options("low"), ...)` 형태로 쓴다."""
    if level not in EFFORT_LEVELS:
        raise ValueError(f"effort는 {EFFORT_LEVELS} 중 하나여야 합니다: {level!r}")
    return {"extra_body": {"output_config": {"effort": level}}}
