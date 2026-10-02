"""
코치 에이전트가 쓰는 도구 (순수 함수, DB 없음).

도구는 app.py가 DB에서 미리 모아 둔 `facts`를 들여다보는 얇은 창이다. 도구에 user_id 같은 인자를 받지 않는
것이 핵심 — facts가 이미 **그 면접의 주인** 것만 담고 있어서, 에이전트가 다른 사용자의 데이터를 조회할 방법이
구조적으로 없다. 1·2단계 점수는 숫자를 주지 않고 "기록 여부와 약한 영역 이름"만 준다: 숫자를 모르면 LLM이 점수를
말해버릴 수도 없다.
"""
from __future__ import annotations

import json

from coach.schema import WEAK_BELOW, completed_count

# 약한 영역을 사용자에게 보이는 말로 부를 때의 이름 (에이전트가 어조를 조절하는 데만 쓰고 언급은 금지)
AXIS_LABELS = {
    "stability": "목소리 안정", "fluency": "말의 유창함", "pause_ctrl": "말 사이 멈춤",
    "continuity": "소리의 이어짐", "calm": "말의 힘",
    "blink": "눈 깜빡임", "gaze": "시선", "expression": "표정",
}

MAX_ANSWER_CHARS_FOR_TOOL = 1500  # 도구 결과가 너무 길어지지 않게 (검증은 원문 전체로 한다)

TOOL_DEFINITIONS = [
    {
        "name": "get_current_interview",
        "description": "방금 끝난 면접의 면접관 난이도, 완주 여부, 그리고 질문과 사용자의 답변 텍스트를 순서대로 가져온다. "
                       "'오늘 해낸 것'과 인용할 문장을 고르려면 이 도구가 필요하다.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_previous_sessions",
        "description": "이번 이전의 면접 기록 요약(최근 순, 최대 5개): 완주 여부, 질문 수, 답변한 질문 수. 질문·답변 본문은 없다. "
                       "'지난번과 달라진 점'을 말하려면 이 도구를 쓴다.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_stage_scores",
        "description": "사용자의 1단계(음성)·2단계(표정·시선) 분석 기록 유무와, 상대적으로 약하게 나온 영역의 이름. "
                       "점수 숫자는 주지 않는다. 말투와 부담을 조절할 때만 참고하고 사용자에게 절대 언급하지 않는다.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_practice_stats",
        "description": "지금까지 끝까지 마친 면접 횟수(이번 포함)와 이번이 첫 완주인지.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


def _weak_areas(stage_scores: dict) -> list[str]:
    out: list[tuple[float, str]] = []
    for stage in ("stage1", "stage2"):
        for axis, v in (stage_scores.get(stage) or {}).items():
            if axis in AXIS_LABELS and isinstance(v, (int, float)) and not isinstance(v, bool) and v < WEAK_BELOW:
                out.append((float(v), AXIS_LABELS[axis]))
    return [label for _, label in sorted(out)]


def execute_tool(name: str, facts: dict) -> str | None:
    """도구 실행 → JSON 문자열. 모르는 도구면 None(호출부가 is_error로 처리)."""
    session = facts["session"]
    if name == "get_current_interview":
        turns = [
            {
                "kind": t.get("kind"),
                "question": t.get("question"),
                "answer": (t.get("answer") or "")[:MAX_ANSWER_CHARS_FOR_TOOL],
            }
            for t in session.get("turns", [])
        ]
        payload = {"tier": session.get("tier"), "completed": bool(session.get("completed")), "turns": turns}
    elif name == "get_previous_sessions":
        payload = {"sessions": facts.get("previous_sessions", [])[:5]}
    elif name == "get_stage_scores":
        scores = facts.get("stage_scores") or {}
        payload = {
            "stage1_recorded": scores.get("stage1") is not None,
            "stage2_recorded": scores.get("stage2") is not None,
            "weak_areas": _weak_areas(scores)[:2],
        }
    elif name == "get_practice_stats":
        n = completed_count(facts)
        payload = {"completed_count": n, "is_first_completed_interview": n == 1 and bool(session.get("completed"))}
    else:
        return None
    return json.dumps(payload, ensure_ascii=False)
