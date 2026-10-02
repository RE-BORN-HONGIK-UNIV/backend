"""
맞춤 연습 — 방금 면접에서 "한 번 더 해볼 만한 질문"을 고르고, 그 질문을 다시 답하는 데 도움이 되는 힌트를 만든다.

원칙은 코치 노트와 같다: 무엇을 연습할지(질문 고르기)는 코드가 정하고, AI는 힌트 문구만 쓰며, AI 출력은 전부
검증하고, 실패하면 일반 힌트로 대체한다. 힌트는 **답을 대신 써주지 않는다** — 말하는 순서와 시작 문장 틀(빈칸 포함)만
주고, 사용자의 경험을 AI가 지어내지 못하게 한다.
"""
from __future__ import annotations

import json
import logging
import time

from coach.safety import detect_care, has_banned_term
from llm_util import effort_options

log = logging.getLogger(__name__)

MAX_QUESTION_LEN = 1000
MAX_PREVIOUS_ANSWER_LEN = 5000
PREVIOUS_ANSWER_FOR_AI = 600   # AI에는 이전 답변의 앞부분만 (맞춤 힌트엔 충분하고, 전송량을 줄임)
MIN_STEPS, MAX_STEPS = 2, 3
MAX_OPENING_LEN = 120
MAX_STEP_LEN = 80
HINT_TIMEOUT_SEC = 20.0

# 사용자 1명이 힌트를 요청할 수 있는 횟수 — 요청마다 AI 비용이 들어서 연타·악용을 막는다. 정상 연습엔 넉넉한 값.
RATE_LIMIT_PER_HOUR = 20


# ── 연습할 질문 고르기 (코드가 정함) ──────────────────────────────────────

def pick_practice_turn(turns: list[dict]) -> int | None:
    """한 번 더 답해볼 질문의 순번(0부터). 답을 못 남긴 질문이 있으면 그중 첫 번째, 없으면 가장 짧게 답한 질문
    (동점이면 앞선 것). 질문이 하나도 없으면 None. '못했다'는 평가가 아니라 "한 번 더 해볼 만한" 질문을 고르는 것이다."""
    if not turns:
        return None
    lengths = [len((t.get("answer") or "").strip()) for t in turns]
    for i, n in enumerate(lengths):
        if n == 0:
            return i
    return min(range(len(lengths)), key=lambda i: (lengths[i], i))


# ── 요청 검증 ────────────────────────────────────────────────────────────

def parse_hint_request(data) -> tuple[str, str]:
    """힌트 요청 → (질문, 이전 답변). 이전 답변은 없을 수 있다(""). 잘못되면 ValueError(사용자용 메시지)."""
    if not isinstance(data, dict):
        raise ValueError("요청 본문이 올바르지 않습니다")
    question = data.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question이 필요합니다")
    question = question.strip()
    if len(question) > MAX_QUESTION_LEN:
        raise ValueError(f"question은 {MAX_QUESTION_LEN}자 이하여야 합니다")
    previous = data.get("previous_answer", "")
    if previous is None:
        previous = ""
    if not isinstance(previous, str):
        raise ValueError("previous_answer는 문자열이어야 합니다")
    previous = previous.strip()
    if len(previous) > MAX_PREVIOUS_ANSWER_LEN:
        raise ValueError(f"previous_answer는 {MAX_PREVIOUS_ANSWER_LEN}자 이하여야 합니다")
    return question, previous


# ── 힌트 (대체 · 검증) ───────────────────────────────────────────────────

def fallback_hint() -> dict:
    """AI를 못 쓸 때의 일반 힌트 — 어떤 질문에도 쓸 수 있는 '결론 → 이유 → 사례' 순서."""
    return {
        "opening": "저는 ___라고 생각해요. 왜냐하면 ___ 때문이에요.",
        "steps": [
            "먼저 한 문장으로 결론부터 말해보세요.",
            "그렇게 생각한 이유나, 떠오르는 경험 하나를 이어서 말해보세요.",
            "마지막으로 그 경험에서 얻은 점이나 앞으로 해보고 싶은 것을 한마디로 정리해보세요.",
        ],
    }


def normalize_hint(raw) -> dict:
    """AI가 만든 힌트를 검증해 완성된 힌트로. 필드 단위로 통과 못 하면 그 필드만 일반 힌트로 바꾼다."""
    base = fallback_hint()
    if not isinstance(raw, dict):
        return base

    opening = raw.get("opening")
    if isinstance(opening, str):
        opening = " ".join(opening.split())
    # 빈칸(___)이 없으면 AI가 답을 통째로 써준 것 — 사용자가 스스로 채워 말하게 하려는 취지에 어긋나서 버린다
    opening_ok = (
        isinstance(opening, str) and "___" in opening and len(opening) <= MAX_OPENING_LEN and not has_banned_term(opening)
    )

    steps: list[str] = []
    if isinstance(raw.get("steps"), list):
        for s in raw["steps"]:
            if isinstance(s, str):
                s = " ".join(s.split())
                if s and len(s) <= MAX_STEP_LEN and not has_banned_term(s):
                    steps.append(s)
            if len(steps) >= MAX_STEPS:
                break

    return {
        "opening": opening if opening_ok else base["opening"],
        "steps": steps if len(steps) >= MIN_STEPS else base["steps"],
    }


SYSTEM_PROMPT = """당신은 'Re-born'이라는 발화 재활 앱의 코치입니다. 사람 앞에서 말하기 어려웠던 고립·은둔 청년이 모의 면접 질문 하나를 한 번 더 연습합니다. 스스로 말할 수 있도록 '말하는 순서'와 '시작 문장 틀'만 줍니다.

반드시 지킬 것:
- 답을 대신 써주지 않습니다. 사용자의 경험이나 사실을 지어내거나 단정하지 않습니다.
- 평가하지 않습니다. 점수, 부족한 점, 못한 점, 진단, 다른 사람과의 비교를 말하지 않습니다.
- 존댓말로 부드럽고 담백하게 씁니다.
- opening은 사용자가 빈칸(___)을 채워 말할 수 있는 한 문장 틀입니다. 반드시 ___ 를 포함하세요 (120자 이내).
- steps는 이 질문에 맞게 구체적으로 쓴 2~3개의 짧은 안내입니다 (각 80자 이내). 예: "먼저 그때의 상황을 한 문장으로 말해보세요."
- 이전 답변이 있으면 그 내용을 바탕으로 '더 구체적으로 말해볼 부분'을 steps에 자연스럽게 녹입니다(이전 답변을 비판하지 않습니다). 이전 답변이 없으면 일반적인 순서를 안내합니다.

설명이나 코드 블록 없이 아래 JSON만 출력하세요.
{"opening": "...___...", "steps": ["...", "..."]}"""


def _parse_json(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def generate_hint(question: str, previous_answer: str, client=None) -> tuple[dict, str]:
    """(힌트, source) 반환. source: 'llm' | 'fallback'. 항상 완성된 힌트를 준다.
    위기 신호가 있는 질문·답변은 AI로 보내지 않는다. 로그엔 예외 종류만 남긴다(답변은 민감정보)."""
    if detect_care([question, previous_answer]):
        return fallback_hint(), "fallback"

    if client is None:
        from coach import agent  # 호출 시점에 가져와서 테스트에서 agent._client를 바꿔 끼울 수 있게
        client = agent._client
    if client is None:
        return fallback_hint(), "fallback"

    from coach.agent import MODEL
    user = f"[질문]\n{question}\n\n[이전 답변]\n{previous_answer[:PREVIOUS_ANSWER_FOR_AI] or '(없음)'}"
    try:
        resp = client.with_options(timeout=HINT_TIMEOUT_SEC, max_retries=0).messages.create(
            model=MODEL,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
            **effort_options("low"),
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        raw = _parse_json(text)
        if raw is not None:
            return normalize_hint(raw), "llm"
    except Exception as e:  # 네트워크/키/쿼터 등 — 연습 화면은 절대 안 깨지게
        log.warning("연습 힌트 생성 실패: %s", type(e).__name__)  # 내용은 로그에 남기지 않음
    return fallback_hint(), "fallback"


# ── 호출 제한 ────────────────────────────────────────────────────────────

class RateLimiter:
    """사용자별 슬라이딩 윈도우 호출 제한 (메모리). 서버가 workers=1이라 프로세스 안에서 충분하고, 재배포하면
    초기화되지만 비용 폭주를 막는 용도로는 이걸로 충분하다."""

    def __init__(self, limit: int = RATE_LIMIT_PER_HOUR, window_sec: float = 3600.0):
        self.limit, self.window = limit, window_sec
        self._hits: dict[object, list[float]] = {}

    def allow(self, key, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        recent = [t for t in self._hits.get(key, []) if now - t < self.window]
        if len(recent) >= self.limit:
            self._hits[key] = recent
            return False
        recent.append(now)
        self._hits[key] = recent
        return True
