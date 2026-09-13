"""
Step 3 · 면접 질문 생성 (난이도 tier 기반).

- 프론트에서 tier(warmup/standard/practice)와 지금까지 나온 질문 목록을 보내면
  다음 질문 1개를 생성해 돌려준다.
- ANTHROPIC_API_KEY 가 없거나 호출이 실패하면 tier별 고정 질문 리스트에서
  아직 안 나온 걸 하나 골라 폴백한다. (프론트 QUESTION_BANK와 동일한 목록 —
  백엔드가 죽어도 화면은 안 죽게)
- STT 연동 전까지는 "이전 답변 내용"은 넘기지 않고, tier + 질문 이력만 사용한다.
  STT 붙으면 previous_answer 파라미터를 추가해서 꼬리질문에 활용하면 된다.
"""
from __future__ import annotations

import os
import logging
import random

log = logging.getLogger(__name__)

try:
    import anthropic
    _client = anthropic.Anthropic() if os.environ.get("ANTHROPIC_API_KEY") else None
except ImportError:  # anthropic 미설치 환경에서도 서버는 떠야 함
    anthropic = None
    _client = None

# 모델은 환경변수로 교체 가능 (claude-opus-5 / claude-haiku-4-5 등)
MODEL = os.environ.get("INTERVIEW_MODEL", "claude-sonnet-5")

COMMON_RULES = """당신은 'Re-born'이라는 비대면 발화 코칭 앱의 AI 모의면접관입니다.
대상은 대면 면접에 대한 두려움으로 취업에 어려움을 겪어온 고립·은둔 청년입니다.

공통 규칙:
- 반드시 한국어로, 질문 딱 1개만 출력합니다.
- 질문 앞뒤에 설명, 인사말, 따옴표, 번호를 붙이지 않습니다. 질문 문장 하나만 출력합니다.
- 존댓말을 쓰되 딱딱한 격식체가 아니라 편안하게 대화하듯 묻습니다.
- 이미 나온 질문과 겹치지 않게 합니다."""

TIER_INSTRUCTIONS = {
    "warmup": (
        "현재 난이도는 워밍업입니다 (통합 불안도 낮음, 긴장도 높은 상태로 해석).\n"
        "자기소개, 취미, 지원 동기처럼 답하기 쉬운 질문만 합니다. 절대 압박하지 않습니다."
    ),
    "standard": (
        "현재 난이도는 표준입니다.\n"
        "실제 채용 면접에서 흔히 나오는 수준의 질문을 합니다."
    ),
    "practice": (
        "현재 난이도는 실전입니다 (통합 불안도 높음, 실전 대응력을 키울 준비가 된 상태로 해석).\n"
        "구체적인 근거나 경험을 요구하는, 조금 더 깊이 있는 질문을 합니다."
    ),
}

# 프론트 difficulty.ts의 QUESTION_BANK와 동일 — API 실패 시 안전망
FALLBACK_QUESTIONS = {
    "warmup": [
        "간단하게 자기소개 먼저 해주시겠어요?",
        "요즘 관심 있게 보고 있는 게 있다면 편하게 말씀해주세요.",
        "오늘 이 자리에 오면서 어떤 마음이었는지 궁금해요.",
    ],
    "standard": [
        "이 직무에 지원하게 된 계기를 말씀해주시겠어요?",
        "최근에 어려운 상황을 해결했던 경험이 있다면 소개해주세요.",
        "본인의 강점을 하나 꼽는다면 무엇인가요?",
    ],
    "practice": [
        "이 직무에 본인이 적합하다고 생각하는 구체적인 근거는 무엇인가요?",
        "실패했던 경험과 그로부터 배운 점을 말씀해주세요.",
        "지금 말씀하신 강점을 실제로 발휘했던 순간을 좀 더 구체적으로 설명해주시겠어요?",
    ],
}


def _fallback_question(tier: str, previous_questions: list[str]) -> str:
    pool = FALLBACK_QUESTIONS.get(tier, FALLBACK_QUESTIONS["standard"])
    remaining = [q for q in pool if q not in previous_questions]
    if not remaining:
        remaining = pool
    return random.choice(remaining)


def _user_prompt(tier: str, previous_questions: list[str], previous_answer: str | None = None) -> str:
    instr = TIER_INSTRUCTIONS.get(tier, TIER_INSTRUCTIONS["standard"])
    asked = "\n".join(f"- {q}" for q in previous_questions) if previous_questions else "(아직 없음)"

    answer_section = ""
    if previous_answer:
        answer_section = (
            f"\n\n[방금 사용자 답변]\n{previous_answer}\n\n"
            "이 답변 내용과 자연스럽게 이어지는 질문을 만들어주세요. "
            "답변에 더 자세히 물어볼 만한 부분이 있으면 꼬리질문으로, "
            "없으면 자연스러운 다음 질문으로 넘어가세요."
        )

    return (
        f"{instr}\n\n"
        f"[이미 나온 질문]\n{asked}"
        f"{answer_section}\n\n"
        "다음 질문을 1개만 생성해주세요."
    )


def generate_question(
    tier: str,
    previous_questions: list[str] | None = None,
    previous_answer: str | None = None,
    timeout: float = 15.0,
) -> tuple[str, str]:
    """(질문, source) 반환. source는 'llm' 또는 'fallback'."""
    previous_questions = previous_questions or []

    if _client is None:
        return _fallback_question(tier, previous_questions), "fallback"

    try:
        resp = _client.with_options(timeout=timeout).messages.create(
            model=MODEL,
            max_tokens=200,
            output_config={"effort": "low"},  # 질문 하나 — 저비용/저지연
            system=[{
                "type": "text",
                "text": COMMON_RULES,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": _user_prompt(tier, previous_questions, previous_answer)}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if text:
            return text, "llm"
        return _fallback_question(tier, previous_questions), "fallback"
    except Exception as e:  # 네트워크/키/쿼터 등 — 화면 흐름은 절대 안 깨지게
        log.warning("Step3 질문 생성 실패: %s", e)
        return _fallback_question(tier, previous_questions), "fallback"