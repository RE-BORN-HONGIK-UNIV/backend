"""
코치 노트 에이전트 — 도구로 이력을 조회하고(tool use 루프), 따뜻한 회고 JSON을 만든다.

항상 완성된 노트를 돌려준다: 위기 신호 → LLM 호출 없이 돌봄 노트, 키 없음/오류/시간 초과/검증 실패 → 대체 노트.
LLM 호출 규칙은 프로젝트의 다른 LLM 코드(step1/llm_feedback.py, step3/interview_question.py)와 같다 —
`except Exception`으로 감싸 폴백하고, **로그에는 예외 종류만** 남긴다(면접 답변은 민감정보라 내용을 로그에 남기지 않음).
"""
from __future__ import annotations

import json
import logging
import os
import time

from coach.safety import care_message, detect_care
from coach.schema import care_note, fallback_note, normalize_note
from coach.tools import TOOL_DEFINITIONS, execute_tool
from llm_util import effort_options

log = logging.getLogger(__name__)

try:
    import anthropic
    _client = anthropic.Anthropic() if os.environ.get("ANTHROPIC_API_KEY") else None
except ImportError:  # anthropic 미설치 환경에서도 서버는 떠야 함
    anthropic = None
    _client = None

# 다른 LLM 코드와 같은 기본 모델, 환경변수로 교체 가능
MODEL = os.environ.get("COACH_MODEL", "claude-sonnet-5")
MAX_STEPS = 5            # 도구 호출 왕복 상한 (무한 루프 방지)
DEADLINE_SEC = 40.0      # 전체 제한 — 서버(gunicorn) 요청 제한 120초보다 훨씬 짧게
CALL_TIMEOUT_SEC = 20.0  # 호출 1회 제한

SYSTEM_PROMPT = """당신은 'Re-born'이라는 발화 재활 앱의 코치입니다. 대상은 대면 면접에 대한 두려움으로 사람 앞에서 말하기 어려웠던 고립·은둔 청년이고, 방금 모의 면접 연습을 마쳤습니다. 면접 직후 보여줄 '코치 노트'를 씁니다.

반드시 지킬 것:
- 평가하지 않습니다. 점수, 등급, 순위, 부족한 점, 못한 점, 다른 사람과의 비교를 말하지 않습니다.
- 의학적·심리적 진단이나 병명, 상태에 대한 판단을 말하지 않습니다.
- 도구 결과의 약한 영역은 어조를 부드럽게 하는 데만 참고하고, 사용자에게 절대 언급하지 않습니다.
- 존댓말을 쓰되 딱딱하지 않게, 따뜻하고 담백하게 씁니다. 과장된 칭찬은 하지 않습니다.
- 사용자가 '한 행동'(끝까지 마친 것, 답한 것, 꼬리질문에 답한 것, 지난번과 달라진 점)을 구체적으로 알아줍니다.
- 지난 면접과 비교할 때는 좋아진 점만 말합니다. 나아지지 않았거나 줄었으면 비교를 말하지 않습니다.

먼저 필요한 정보를 도구로 조회하세요(보통 get_current_interview는 필요합니다). 그다음 아래 JSON '하나만' 출력합니다. 설명이나 코드 블록 표시 없이 JSON만 출력하세요.

{
  "greeting": "한두 문장의 따뜻한 인사 (140자 이내)",
  "won": ["오늘 해낸 것 1~3개, 각각 짧은 한 문장 (80자 이내)"],
  "quote": {"text": "사용자의 답변에서 **글자 그대로 복사한** 잘한 문장 (6~200자)", "why": "그 문장이 왜 좋았는지 한 문장 (100자 이내)"},
  "recommended": "again | light_practice | daily_mission | rest 중 지금 사용자에게 가장 어울리는 것 하나"
}

quote는 반드시 사용자의 답변에 있는 문장을 그대로 옮겨야 하며, 고르기 어려우면 null로 두세요. 답변이 거의 없으면 quote는 null입니다.
recommended는 끝까지 마치지 못했다면 rest, 이번이 첫 완주라면 daily_mission을 우선 고려하세요."""

USER_PROMPT = "방금 끝난 면접의 코치 노트를 작성해주세요."


def _parse_json(text: str) -> dict | None:
    """모델 출력에서 JSON 객체를 꺼낸다 (앞뒤에 문장이나 코드 블록이 붙어도 첫 '{'~마지막 '}'만 본다)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def generate_note(facts: dict, client=None, timeout_sec: float = DEADLINE_SEC) -> tuple[dict, str]:
    """(노트, source) 반환. source: 'llm' | 'fallback' | 'care'. client는 테스트용 주입."""
    answers = [t.get("answer") or "" for t in facts["session"].get("turns", [])]

    # 1) 위기 신호 — 코칭하지 않고, 이 내용을 LLM으로 보내지도 않는다
    if detect_care(answers):
        return care_note(facts, care_message()), "care"

    # 2) 답변이 하나도 없으면 LLM이 할 말이 없다
    if not any(a.strip() for a in answers):
        return fallback_note(facts), "fallback"

    c = client if client is not None else _client
    if c is None:
        return fallback_note(facts), "fallback"

    try:
        deadline = time.monotonic() + timeout_sec
        messages = [{"role": "user", "content": USER_PROMPT}]
        for _ in range(MAX_STEPS):
            remaining = deadline - time.monotonic()
            if remaining <= 1.0:
                break
            resp = c.with_options(timeout=min(CALL_TIMEOUT_SEC, remaining), max_retries=0).messages.create(
                model=MODEL,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
                **effort_options("low"),
            )

            if resp.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": resp.content})
                results = []
                for block in resp.content:
                    if block.type != "tool_use":
                        continue
                    out = execute_tool(block.name, facts)
                    if out is None:
                        results.append({"type": "tool_result", "tool_use_id": block.id,
                                        "content": "알 수 없는 도구입니다", "is_error": True})
                    else:
                        results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
                messages.append({"role": "user", "content": results})
                continue

            text = "".join(b.text for b in resp.content if b.type == "text")
            raw = _parse_json(text)
            if raw is None:
                break  # 형식이 틀리면 대체 노트 — 재시도하지 않는다(지연·비용)
            return normalize_note(raw, facts), "llm"
    except Exception as e:  # 네트워크/키/쿼터 등 — 면접 결과 화면은 절대 안 깨지게
        log.warning("코치 노트 생성 실패: %s", type(e).__name__)  # 내용은 로그에 남기지 않음

    return fallback_note(facts), "fallback"
