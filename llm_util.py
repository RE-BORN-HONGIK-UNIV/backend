"""공통 · AI 호출 도우미 — effort(생각·토큰 사용량) 설정을 SDK 버전과 무관하게 extra_body로 넘긴다.
키워드 인자로 넘기면 고정된 anthropic 0.69.0에서 오류가 나고 폴백에 삼켜져 AI가 한 번도 안 쓰인 적이 있다(docs/공통_설계노트.md).
"""
from __future__ import annotations

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def effort_options(level: str) -> dict:
    """`messages.create(**effort_options("low"), ...)` 형태로 쓴다."""
    if level not in EFFORT_LEVELS:
        raise ValueError(f"effort는 {EFFORT_LEVELS} 중 하나여야 합니다: {level!r}")
    return {"extra_body": {"output_config": {"effort": level}}}
