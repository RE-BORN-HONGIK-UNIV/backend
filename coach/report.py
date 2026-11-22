"""
리포트 3·4번 칸 AI — 다듬을 대답만 다듬은 말(3번)과 다음 연습 계획(4번)을 Claude 한 번 호출로 만든다.

고르기·숫자·검사는 코드가 하고 Claude는 문장만 쓴다. 어떤 대답을 다듬을지는 브라우저가 보낸 질문별 말버릇·멈춤
숫자와 서버의 대답 글자로 코드가 고른다(말버릇 표시는 브라우저가 만들어서 서버엔 숫자가 없음).
AI가 실패해도 항상 완성된 결과를 돌려준다(빈 다듬은 말·빈 계획 → 프론트가 내가 한 말·기본 팁으로 보여 줌).
"""
from __future__ import annotations

import json
import logging
import re

from coach.agent import MODEL, _client, _parse_json
from coach.safety import care_message, detect_care, has_banned_term
from llm_util import effort_options

log = logging.getLogger(__name__)

TIMEOUT_SEC = 40.0  # 서버 요청 제한(120초)보다 짧게

# 다듬을 대답 고르는 기준 (spec 7-4)
MIN_FILLERS = 2          # 말버릇 2번 이상
MIN_PAUSES = 2           # 긴 멈춤(1.2초 이상, step1/analyze_pause.py) 2번 이상
LONG_SENTENCE = 120      # 한 문장이 이 글자 수를 넘으면 너무 긴 문장
MIN_ANSWER_LEN = 30      # 이보다 짧은 대답은 다듬을 게 없다
MAX_REWRITES = 4         # 비용 상한

MAX_CHANGES = 2
MAX_CHANGE_LEN = 60
MAX_HOW_LEN = 90
MAX_PLAN = 2

# 계획 후보로 받을 수 있는 습관 (프론트 AxisKey + 얼굴 3가지)
PLAN_KEYS = ("stability", "fluency", "pause_ctrl", "continuity", "calm", "blink", "gaze", "expr")
MAX_LINE_LEN = 120
MAX_STATS = 30

SYSTEM_PROMPT = """당신은 면접 연습 앱 'Re-born'의 코치입니다. 사람 앞에서 말하기가 어려웠던 청년이 방금 모의 면접을 마쳤습니다.
두 가지를 씁니다.

1) rewrites: 주어진 대답 각각을 '같은 사람이 같은 내용을 더 또렷하게 말한 버전'으로 다듬습니다.
- 내용은 그대로 둡니다. 원래 대답에 없는 경험·사실·숫자·이름을 절대 더하지 않습니다.
- 바꿀 수 있는 것: 문장 나누기, 결론을 먼저 말하기, 순서 정리, '음·어·약간·그냥' 같은 말버릇 빼기, '~입니다/~했습니다' 말투.
- 길이는 원래와 비슷하거나 더 짧게. 말하듯 자연스럽게.
- changes: 무엇을 바꿨는지 짧은 두 줄 (예: "결론을 첫 문장으로 옮겼어요", "'음'을 빼고 문장을 둘로 나눴어요").

2) plan: 후보 습관 중 이번 대답에서 연습 효과가 가장 클 것 최대 2개를 고르고, 각각 바로 해 볼 방법을 한 문장으로 씁니다.
- 이 사람의 실제 대답을 짚어 구체적으로 (예: "질문이 끝나면 숨을 한 번 쉬고 '저는'으로 첫 문장을 시작해 봐요").
- key는 후보에 있는 것만 씁니다.

지킬 것: 평가하지 않습니다(점수·등급·부족·실패·잘못 같은 말 금지). 진단·병명을 말하지 않습니다. 존댓말로 따뜻하고 담백하게.

아래 JSON 하나만 출력합니다. 설명이나 코드 블록 표시 없이.
{"rewrites": [{"id": 0, "text": "다듬은 대답", "changes": ["바꾼 점", "바꾼 점"]}], "plan": [{"key": "후보 key", "how": "방법 한 문장"}]}"""


def parse_report_request(data) -> tuple[list[dict], list[dict]]:
    """요청 본문 → (질문별 말버릇 수·긴 멈춤 수, 계획 후보). 형식이 틀리면 ValueError. 숫자는 고르는 데만 쓰여 잘못 와도 망가지는 건 없다."""
    if not isinstance(data, dict):
        raise ValueError("요청 본문이 올바르지 않습니다")
    stats_raw, cand_raw = data.get("turns", []), data.get("candidates", [])
    if not isinstance(stats_raw, list) or not isinstance(cand_raw, list) or len(stats_raw) > MAX_STATS:
        raise ValueError("turns·candidates는 목록이어야 합니다")

    def num(v, hi):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (0 <= v <= hi):
            raise ValueError("숫자가 올바르지 않습니다")
        return v

    stats = []
    for s in stats_raw:
        if not isinstance(s, dict) or not isinstance(s.get("question"), str):
            raise ValueError("turns 항목이 올바르지 않습니다")
        stats.append({"question": s["question"], "fillers": int(num(s.get("fillers", 0), 500)),
                      "pauses": int(num(s.get("pauses", 0), 500))})
    cands, seen = [], set()
    for c in cand_raw:
        if not isinstance(c, dict) or c.get("key") not in PLAN_KEYS or not isinstance(c.get("line"), str):
            raise ValueError("candidates 항목이 올바르지 않습니다")
        if c["key"] not in seen:
            seen.add(c["key"])
            cands.append({"key": c["key"], "line": c["line"][:MAX_LINE_LEN]})
    return stats, cands


def longest_sentence(text: str) -> int:
    """문장부호로 나눈 가장 긴 문장의 글자 수."""
    return max((len(s.strip()) for s in re.split(r"[.?!。\n]", text)), default=0)


def pick_rewrite_turns(turns: list[dict], stats: list[dict]) -> list[dict]:
    """다듬을 대답: 말버릇 2번 이상 / 긴 멈춤 2번 이상 / 너무 긴 문장. 너무 짧은 대답은 뺀다. 질문 글로 짝을 맞춘다."""
    by_q = {s["question"]: s for s in stats}
    picked = []
    for t in turns:
        answer = (t.get("answer") or "").strip()
        if len(answer) < MIN_ANSWER_LEN:
            continue
        s = by_q.get(t.get("question"), {})
        if (s.get("fillers", 0) >= MIN_FILLERS or s.get("pauses", 0) >= MIN_PAUSES
                or longest_sentence(answer) > LONG_SENTENCE):
            picked.append({"question": t["question"], "answer": answer,
                           "fillers": s.get("fillers", 0), "pauses": s.get("pauses", 0)})
    return picked[:MAX_REWRITES]


def keeps_content(original: str, text: str) -> bool:
    """다듬은 말이 원래 대답에 없는 내용을 더하지 않았는지. 숫자는 원래 대답에 있어야 하고,
    낱말 앞 두 글자가 원래 대답에 없는 낱말이 30%를 넘으면 버린다.
    shortcut: 낱말 앞 두 글자 비교라 바꿔 쓴 말도 새 낱말로 셀 수 있다, 너무 많이 버려지면 형태소 분석으로 바꾼다."""
    flat = re.sub(r"\s+", "", original)
    if any(d not in flat for d in re.findall(r"\d+", text)):
        return False
    words = [w for w in re.findall(r"[가-힣A-Za-z]+", text) if len(w) >= 2]
    if not words:
        return False
    new = sum(1 for w in words if w[:2] not in flat)
    return new / len(words) <= 0.3


def _clean(v, max_len: int) -> str | None:
    if not isinstance(v, str):
        return None
    s = re.sub(r"\s+", " ", v).strip()
    return s if s and len(s) <= max_len and not has_banned_term(s) else None


def normalize_result(raw: dict, picked: list[dict], cands: list[dict]) -> dict:
    """Claude 출력 검사: 통과한 다듬은 말·계획만 남긴다."""
    rewrites = []
    for r in raw.get("rewrites") or []:
        if not isinstance(r, dict) or not isinstance(r.get("id"), int) or not (0 <= r["id"] < len(picked)):
            continue
        orig = picked[r["id"]]["answer"]
        text = _clean(r.get("text"), int(len(orig) * 1.3) + 20)
        if text is None or not keeps_content(orig, text) or any(x["question"] == picked[r["id"]]["question"] for x in rewrites):
            continue
        changes = [c for c in (_clean(c, MAX_CHANGE_LEN) for c in (r.get("changes") or [])[:MAX_CHANGES]) if c]
        rewrites.append({"question": picked[r["id"]]["question"], "text": text, "changes": changes})

    keys = {c["key"] for c in cands}
    plan = []
    for p in raw.get("plan") or []:
        how = _clean(p.get("how"), MAX_HOW_LEN) if isinstance(p, dict) else None
        if how and p.get("key") in keys and all(x["key"] != p["key"] for x in plan):
            plan.append({"key": p["key"], "how": how})
    return {"rewrites": rewrites, "plan": plan[:MAX_PLAN]}


def generate_report_coach(turns: list[dict], stats: list[dict], cands: list[dict], client=None) -> tuple[dict, str]:
    """(결과, source) 반환. 결과: {picked, rewrites, plan, care}. source: 'llm' | 'fallback' | 'care'."""
    if detect_care([t.get("answer") or "" for t in turns]):
        return {"picked": [], "rewrites": [], "plan": [], "care": care_message()}, "care"

    picked = pick_rewrite_turns(turns, stats)
    result = {"picked": [p["question"] for p in picked], "rewrites": [], "plan": [], "care": None}
    c = client if client is not None else _client
    if c is None or (not picked and not cands):
        return result, "fallback"

    payload = {
        "answers_to_rewrite": [{"id": i, "question": p["question"], "answer": p["answer"],
                                "fillers": p["fillers"], "long_pauses": p["pauses"]}
                               for i, p in enumerate(picked)],
        "plan_candidates": cands,
        "all_answers": [{"question": t.get("question"), "answer": (t.get("answer") or "")[:1500]} for t in turns],
    }
    try:
        resp = c.with_options(timeout=TIMEOUT_SEC, max_retries=0).messages.create(
            model=MODEL,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            **effort_options("low"),
        )
        raw = _parse_json("".join(b.text for b in resp.content if b.type == "text"))
        if raw is not None:
            return {**result, **normalize_result(raw, picked, cands)}, "llm"
    except Exception as e:  # 리포트는 절대 안 깨지게, 내용은 로그에 남기지 않음
        log.warning("리포트 코치 생성 실패: %s", type(e).__name__)
    return result, "fallback"
