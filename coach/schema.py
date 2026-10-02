"""
코치 노트의 모양·검증·대체 노트 (순수 함수, DB·LLM 없음).

설계 원칙: **실행 가능한 조언(다음 한 걸음 카드, 일상 미션)은 코드가 정하고, LLM은 따뜻한 말(인사·"해낸 것"·
인용)만 쓴다.** LLM이 만든 문구는 전부 검증을 거치고(길이, 평가·진단 표현, 인용이 실제 답변에 있는지),
통과 못 하면 대체 문구로 바꾼다 — 노트는 항상 완성된 채로 나온다.

facts 형태 (app.py가 DB에서 모아 넘김):
{
  "session":  {"id": int, "tier": str, "completed": bool,
               "turns": [{"kind": "main"|"follow_up", "question": str, "answer": str}]},
  "previous_sessions": [{"id": int, "tier": str, "completed": bool, "question_count": int,
                         "answered_count": int}],          # 최근 순, 이번 면접 제외
  "stage_scores": {"stage1": {축: 0~100} | None, "stage2": {축: 0~100} | None},
}
"""
from __future__ import annotations

import re

from coach.missions import pick_mission
from coach.practice import pick_practice_turn
from coach.safety import has_banned_term

KINDS = ("again", "light_practice", "daily_mission", "rest")

MAX_GREETING_LEN = 140
MAX_WON_ITEMS = 3
MAX_WON_LEN = 80
MIN_QUOTE_LEN = 6      # 너무 짧은 인용("네", "음 그래서")은 칭찬 근거가 못 된다
MAX_QUOTE_LEN = 200
MAX_WHY_LEN = 100

# 약하다고 볼 기준 — 프론트 getTier의 "안정적" 경계(70)를 재사용한 휴리스틱 (step3/interview_question.py WEAK_BELOW와 같은 값)
WEAK_BELOW = 70


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def answered_count(turns: list[dict]) -> int:
    return sum(1 for t in turns if (t.get("answer") or "").strip())


def completed_count(facts: dict) -> int:
    """끝까지 마친 면접 수(이번 면접 포함)."""
    prev = sum(1 for s in facts.get("previous_sessions", []) if s.get("completed"))
    return prev + (1 if facts["session"].get("completed") else 0)


# ── 다음 한 걸음 카드 (코드가 정함) ─────────────────────────────────────────

def weakest_stage_path(stage_scores: dict) -> str:
    """'가볍게 연습' 카드가 보낼 단계: 점수가 가장 낮은 항목이 속한 단계. 기록이 없으면 1단계."""
    lowest: tuple[float, str] | None = None
    for stage, path in (("stage1", "/voice"), ("stage2", "/face")):
        for v in (stage_scores.get(stage) or {}).values():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and (lowest is None or v < lowest[0]):
                lowest = (float(v), path)
    return lowest[1] if lowest else "/voice"


def _practice_card(facts: dict) -> dict:
    """'연습' 카드. 방금 면접에서 한 번 더 답해볼 질문이 있으면 맞춤 연습 화면(그 질문을 힌트와 함께 다시 답하기)으로,
    없으면(질문이 하나도 없는 비정상 면접) 예전처럼 1·2단계 연습 화면으로 보낸다. 점수나 약한 부분의 이름은 문구에 넣지 않는다."""
    idx = pick_practice_turn(facts["session"].get("turns", []))
    if idx is not None:
        return {
            "kind": "light_practice",
            "title": "이 질문 다시 답해보기",
            "body": "방금 질문 하나를 힌트와 함께 한 번 더 답해봐요. 천천히, 편하게요.",
            "practiceTurn": idx,  # 프론트가 면접 중 메모리에 모아둔 질문·답변에서 이 순번을 찾아 연습 화면에 넘긴다
        }
    return {
        "kind": "light_practice",
        "title": "가볍게 연습",
        "body": "부담 없이 목소리나 표정으로 몸을 풀어봐요. 짧게 해도 충분해요.",
        "path": weakest_stage_path(facts.get("stage_scores") or {}),
    }


def build_cards(facts: dict) -> list[dict]:
    """다음 한 걸음 카드 4장. 점수나 약한 부분의 이름은 카드 문구에 넣지 않는다(규정하는 말이 되므로)."""
    session = facts["session"]
    return [
        {
            "kind": "again",
            "title": "한 번 더 해보기",
            "body": "방금 해본 흐름을 한 번 더 이어가 봐요. 면접관을 바꿔볼 수도 있어요.",
        },
        _practice_card(facts),
        {
            "kind": "daily_mission",
            "title": "현실로 한 걸음",
            "body": pick_mission(completed_count(facts), int(session.get("id") or 0)),
        },
        {
            "kind": "rest",
            "title": "오늘은 여기까지",
            "body": "충분히 잘했어요. 쉬는 것도 연습의 일부예요.",
        },
    ]


def default_recommended(facts: dict) -> str:
    """강조할 카드: 끝까지 못 했으면 쉬기, 첫 완주면 일상 미션, 그 뒤로는 한 번 더."""
    if not facts["session"].get("completed"):
        return "rest"
    return "daily_mission" if completed_count(facts) <= 1 else "again"


# ── 대체 노트 (LLM 없이) ────────────────────────────────────────────────────

def fallback_note(facts: dict) -> dict:
    """AI를 못 쓸 때(키 없음·오류·검증 실패)도 항상 완성된 노트를 준다. 사실(횟수·완주)만 말한다."""
    session = facts["session"]
    turns = session.get("turns", [])
    won: list[str] = []
    if session.get("completed"):
        won.append("면접을 끝까지 마쳤어요")
    n = answered_count(turns)
    if n:
        won.append(f"{n}개의 질문에 답했어요")
    if any(t.get("kind") == "follow_up" and (t.get("answer") or "").strip() for t in turns):
        won.append("꼬리질문에도 답해봤어요")
    if completed_count(facts) <= 1 and session.get("completed"):
        won.insert(0, "첫 면접 연습을 해냈어요")
    return {
        "greeting": "수고하셨어요. 오늘도 한 걸음 해냈어요.",
        "won": (won or ["면접 연습을 시작해봤어요"])[:MAX_WON_ITEMS],
        "quote": None,
        "recommended": default_recommended(facts),
        "cards": build_cards(facts),
        "care": None,
    }


def care_note(facts: dict, care: dict) -> dict:
    """위기 신호가 있을 때의 노트: 코칭 없이 돌봄 안내와 '쉬기'만."""
    return {
        "greeting": care["greeting"],
        "won": [],
        "quote": None,
        "recommended": "rest",
        "cards": [c for c in build_cards(facts) if c["kind"] == "rest"],
        "care": {"body": care["body"], "resources": care["resources"]},
    }


# ── LLM 출력 검증 ───────────────────────────────────────────────────────────

def _clean_text(v, max_len: int) -> str | None:
    """사람이 읽는 문구로 쓸 수 있으면 정리해서, 아니면 None(→ 대체 문구)."""
    if not isinstance(v, str):
        return None
    s = _norm_ws(v)
    if not s or len(s) > max_len or has_banned_term(s):
        return None
    return s


def verified_quote(raw, answers: list[str]) -> dict | None:
    """인용은 **실제 답변에 그대로 있는 문장일 때만** 쓴다(LLM이 지어낸 인용은 버림). 사용자 자신의 말이라도
    평가·부정적 표현이 들어 있으면 칭찬 근거로 되돌려주지 않는다."""
    if not isinstance(raw, dict):
        return None
    text = raw.get("text")
    why = _clean_text(raw.get("why"), MAX_WHY_LEN)
    if not isinstance(text, str) or why is None:
        return None
    quote = _norm_ws(text)
    if not (MIN_QUOTE_LEN <= len(quote) <= MAX_QUOTE_LEN) or has_banned_term(quote):
        return None
    if not any(quote in _norm_ws(a) for a in answers):
        return None
    return {"text": quote, "why": why}


def normalize_note(raw, facts: dict) -> dict:
    """LLM이 만든 JSON을 검증해서 완성된 노트로. 필드별로 통과 못 하면 그 필드만 대체 문구로 바꾼다."""
    base = fallback_note(facts)
    if not isinstance(raw, dict):
        return base

    greeting = _clean_text(raw.get("greeting"), MAX_GREETING_LEN)

    won: list[str] = []
    raw_won = raw.get("won")
    if isinstance(raw_won, list):
        for item in raw_won:
            cleaned = _clean_text(item, MAX_WON_LEN)
            if cleaned:
                won.append(cleaned)
            if len(won) >= MAX_WON_ITEMS:
                break

    answers = [t.get("answer") or "" for t in facts["session"].get("turns", [])]
    recommended = raw.get("recommended")

    return {
        "greeting": greeting or base["greeting"],
        "won": won or base["won"],
        "quote": verified_quote(raw.get("quote"), answers),
        "recommended": recommended if recommended in KINDS else base["recommended"],
        "cards": base["cards"],  # 카드는 항상 코드가 정한 것
        "care": None,
    }
