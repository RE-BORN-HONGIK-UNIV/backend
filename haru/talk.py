"""하루 · 대화 — 1·2단계 대화 상대 하루가 방금 대답에 한 문장 반응하고 다음 질문으로 잇는 말과, "하루가 이렇게 이해했어요" 카드 요약을 만든다.
질문 순서는 화면(frontend features/haru/script.ts)이 정하고 AI는 반응 문장만 쓴다. 위기 신호면 AI로 보내지 않고, 실패해도 정해 둔 문장으로 항상 응답한다.
경량 웹에서 옮기며 바꾼 점: docs/haru_설계노트.md
"""
from __future__ import annotations

import json
import re
import logging
import os

from coach.safety import detect_care, has_banned_term
from llm_util import effort_options

log = logging.getLogger(__name__)

try:
    import anthropic
    _client = anthropic.Anthropic() if os.environ.get("ANTHROPIC_API_KEY") else None
except ImportError:  # anthropic 미설치 환경에서도 서버는 떠야 함
    _client = None

MODEL = os.environ.get("HARU_MODEL", "claude-sonnet-5")
TIMEOUT_SEC = 12.0  # 대화 중 기다리는 시간이라 짧게 — 넘으면 정해 둔 문장으로 바로 이어간다

MAX_TEXT_LEN = 300       # 질문·대화 기록 한 줄
MAX_ANSWER_LEN = 800     # 사용자 대답 (말로 한 대답을 받아쓴 글)
MAX_HISTORY = 6          # 최근 몇 줄까지 AI에 넘길지
MAX_SAY_LEN = 160        # 하루가 소리 내어 읽을 말 — 두 문장 80자 안팎을 요청하고, 넘치면 버린다
MAX_PROFILE_ITEMS = 8
MAX_PROFILE_VALUE_LEN = 40

# 하루가 대답을 듣고 붙이는 짧은 공감 한마디 — AI를 못 쓸 때 돌려 가며 쓴다 (경량 웹 NODS와 같은 문장)
NODS = ("그렇군요, 이야기해 줘서 고마워요.", "좋아요, 잘 들었어요.", "천천히 말해 줘서 고마워요.")  # 미리 받아 둔 하루 mp3와 같은 문장 ("아, 그랬군요."는 어색해서 뺌)
CLOSING = "오늘 이야기 나눠 줘서 고마워요."
CARE_LINE = "이야기해 줘서 고마워요. 마음이 힘들면 언제든 쉬어 가도 괜찮아요."

SYSTEM = """너는 말하기 연습 웹 "Re-born"의 대화 상대 "하루"다. 사용자는 말하기와 면접이 긴장되는 청년이고, 천천히 한 단계씩 연습하고 있다.
꼭 물어야 할 질문 순서는 화면이 정해 둔다. 너는 방금 대답에 짧게 반응하고, 다음 질문으로 자연스럽게 이어 주는 말만 만든다.

규칙
- 반응은 대답 내용을 한 가지 짚는 한 문장. 그다음 다음 질문을 뜻은 그대로, 말투만 자연스럽게 붙인다.
- 평가·점수·조언을 하지 않는다. 말버릇, 긴장, 목소리, 분석 이야기를 하지 않는다.
- 사용자의 처지(고립, 오랜만, 단절, 공백기)를 짐작하거나 묻지 않는다. 사용자가 먼저 꺼내면 탓하지 않고 받아 준다.
- 대답이 질문을 비켜 갔거나 비어 있고, 아직 다시 묻지 않았다면 retry를 true로 한다. 가볍게 받아 준 뒤 같은 질문을 더 쉬운 말로 한 번 다시 묻는다. 이미 다시 물었다면 retry는 false로 하고 넘어간다.
- 다음 질문이 없으면(마지막) 고맙다는 짧은 마무리 한 문장만 한다.
- 힘든 마음을 털어놓으면 다그치지 않고 따뜻하게 받아 준 뒤 천천히 다음으로 간다.
- 말투: 면접 연습을 도와주는 친절한 진행자의 해요체. 따뜻하지만 수다나 긴 맞장구는 줄이고, 질문은 또렷하게 한다.
- 전체 길이는 두 문장, 80자 안팎. 소리 내어 읽을 말이라 이모지·괄호·목록을 쓰지 않는다.

설명이나 코드 블록 없이 아래 JSON만 출력한다.
{"say": "...", "retry": false}"""

PROFILE_SYSTEM = """너는 말하기 연습 웹 "Re-born"의 대화 상대 "하루"다. 1·2단계에서 사용자와 나눈 대화를 항목별로 짧게 정리한다.
이 정리는 사용자에게 "하루가 이렇게 이해했어요"라고 보여 주고, 사용자가 확인·수정한 뒤 실전 면접 질문의 소재로 쓴다.

규칙
- 항목마다 사용자의 대답에서 핵심만 15~30자 정도의 명사구나 짧은 문장으로 정리한다. 대답에 없는 내용을 지어내지 않는다.
- 대답이 비었거나 알아들을 수 없으면 그 항목은 빈 문자열로 둔다.
- 평가·진단·조언 표현을 쓰지 않는다. 사용자의 처지를 짐작해서 덧붙이지 않는다.
- 이모지·괄호를 쓰지 않는다.

설명이나 코드 블록 없이, 받은 항목 순서 그대로 아래 JSON만 출력한다.
{"values": ["...", "..."]}"""


def _clip(v, n: int) -> str:
    """값을 글로 바꿔 앞뒤 공백을 지우고 n글자로 자른다."""
    return str(v or "").strip()[:n]


def _parse_json(text: str) -> dict | None:
    """AI 응답 글에서 첫 '{'부터 마지막 '}'까지를 JSON 객체로 읽는다. 실패하면 None."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


# ── 반응 한마디 ─────────────────────────────────────────────────────────

def parse_reply_request(data) -> dict:
    """요청 본문 검증·정리. 형식이 틀리면 ValueError."""
    if not isinstance(data, dict):
        raise ValueError("JSON 본문이 필요합니다")
    question = _clip(data.get("question"), MAX_TEXT_LEN)
    if not question:
        raise ValueError("question이 필요합니다")
    history = data.get("history") or []
    if not isinstance(history, list):
        raise ValueError("history는 목록이어야 합니다")
    index = data.get("index", 0)
    return {
        "question": question,
        "answer": _clip(data.get("answer"), MAX_ANSWER_LEN),
        "next": _clip(data.get("next"), MAX_TEXT_LEN) or None,
        "retried": bool(data.get("retried")),
        "index": index if isinstance(index, int) and index >= 0 else 0,
        "history": [
            {"ai": bool(h.get("ai")), "text": _clip(h.get("text"), MAX_TEXT_LEN)}
            for h in history[-MAX_HISTORY:] if isinstance(h, dict)
        ],
    }


def reply_prompt(req: dict) -> str:
    """최근 대화·방금 질문·대답을 AI에게 보낼 글로 만든다."""
    history = "\n".join(f"{'하루' if h['ai'] else '사용자'}: {h['text']}" for h in req["history"])
    return (
        f"최근 대화:\n{history or '(없음)'}\n\n"
        f"방금 질문: {req['question']}\n"
        f"사용자 대답: {req['answer'] or '(말이 글로 잡히지 않았다)'}\n"
        f"이미 다시 물었나: {'예' if req['retried'] else '아니오'}\n"
        f"다음 질문: {req['next'] or '(없음, 마지막)'}"
    )


def fallback_reply(req: dict, care: bool = False) -> dict:
    """AI 없이 쓰는 반응: 공감 한마디 + 다음 질문(마지막이면 마무리 인사). 다시 묻지 않는다."""
    nod = CARE_LINE if care else NODS[req["index"] % len(NODS)]
    say = f"{nod} {req['next']}" if req["next"] else (CARE_LINE if care else CLOSING)
    return {"say": say, "retry": False}


def normalize_reply(raw, req: dict) -> dict | None:
    """AI 출력 검증. 통과 못 하면 None(→ 폴백). 이미 다시 물었으면 retry는 강제로 False."""
    if not isinstance(raw, dict):
        return None
    say = raw.get("say")
    if not isinstance(say, str):
        return None
    say = " ".join(say.split())
    if not say or len(say) > MAX_SAY_LEN or has_banned_term(say):
        return None
    return {"say": say, "retry": bool(raw.get("retry")) and not req["retried"]}


def generate_reply(req: dict, client=None) -> tuple[dict, str]:
    """(반응, source) 반환. source: 'llm' | 'fallback' | 'care'. 항상 완성된 반응을 준다."""
    if detect_care([req["answer"]]):
        return fallback_reply(req, care=True), "care"
    client = client if client is not None else _client
    if client is None:
        return fallback_reply(req), "fallback"
    try:
        resp = client.with_options(timeout=TIMEOUT_SEC, max_retries=0).messages.create(
            model=MODEL,
            max_tokens=300,
            system=SYSTEM,
            messages=[{"role": "user", "content": reply_prompt(req)}],
            **effort_options("low"),
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        out = normalize_reply(_parse_json(text), req)
        if out is not None:
            return out, "llm"
    except Exception as e:  # 네트워크/키/쿼터 등 — 대화는 정해 둔 문장으로 계속 간다
        log.warning("하루 반응 생성 실패: %s", type(e).__name__)  # 내용은 로그에 남기지 않음
    return fallback_reply(req), "fallback"


# ── "하루가 이렇게 이해했어요" 요약 ─────────────────────────────────────

def parse_profile_request(data) -> list[dict]:
    """[{label, question, answer}] 검증·정리. 형식이 틀리면 ValueError."""
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError("items가 필요합니다")
    out = []
    for it in items[:MAX_PROFILE_ITEMS]:
        if not isinstance(it, dict) or not _clip(it.get("label"), 40):
            raise ValueError("각 항목에 label이 필요합니다")
        out.append({
            "label": _clip(it.get("label"), 40),
            "question": _clip(it.get("question"), MAX_TEXT_LEN),
            "answer": _clip(it.get("answer"), MAX_ANSWER_LEN),
        })
    return out


NAME_LABEL = "부를 이름"
# "○○라고 불러 줘"·"제 이름은 ○○예요"에서 이름만 뽑는다('라고' 앞은 통째로 — 이름+이인지 글자로는 못 가른다)
_NAME_PATTERNS = [
    re.compile(r"([가-힣A-Za-z]{1,10})\s*라고\s*(?:불러|해|하면|부르)"),
    re.compile(r"(?:제|내|저의|나의)\s*이름은\s*([가-힣A-Za-z]{1,10}?)\s*(?:이에요|예요|입니다|이야|야|이요|요|이고|고)?(?:[.!~\s]|$)"),
]
_NAME_ENDINGS = ("이에요", "예요", "입니다", "이요", "요")


def extract_name(answer: str) -> str:
    """부를 이름 대답에서 이름만 뽑는다(AI 없이). 못 뽑으면 빈 문자열."""
    a = " ".join((answer or "").split())
    for pat in _NAME_PATTERNS:
        m = pat.search(a)
        if m:
            return m.group(1)
    # "지우요"처럼 한 낱말 대답이면 끝맺음만 떼고 그대로
    word = re.sub(r"[.!~?\s]+$", "", a)
    if word and " " not in word and len(word) <= 8:
        for end in _NAME_ENDINGS:
            if word.endswith(end) and len(word) > len(end):
                return word[: -len(end)]
        return word
    return ""


def fallback_value(answer: str, label: str = "") -> str:
    """AI 없이 대답 앞부분을 그대로 쓴다(사용자가 카드에서 고친다). 부를 이름은 이름만 뽑는다."""
    if label == NAME_LABEL:
        name = extract_name(answer)
        if name:
            return name
    a = " ".join(answer.split())
    return a if len(a) <= MAX_PROFILE_VALUE_LEN else a[:MAX_PROFILE_VALUE_LEN - 1] + "…"


def profile_prompt(items: list[dict]) -> str:
    """카드 항목별 질문과 대답을 AI에게 보낼 글로 만든다."""
    return "\n\n".join(
        f"[{i + 1}. {it['label']}]\n질문: {it['question']}\n대답: {it['answer'] or '(없음)'}"
        for i, it in enumerate(items)
    )


def normalize_profile(raw, items: list[dict]) -> list[str] | None:
    """AI가 준 값 목록 검증. 개수가 다르면 전체 폴백, 항목 하나가 이상하면 그 항목만 대답 원문으로."""
    vals = raw.get("values") if isinstance(raw, dict) else None
    if not isinstance(vals, list) or len(vals) != len(items):
        return None
    out = []
    for v, it in zip(vals, items):
        v = " ".join(v.split()) if isinstance(v, str) else ""
        if not it["answer"]:
            v = ""  # 대답이 없던 항목을 AI가 채우지 못하게
        elif not v or len(v) > MAX_PROFILE_VALUE_LEN or has_banned_term(v):
            v = fallback_value(it["answer"], it["label"])
        out.append(v)
    return out


def summarize_profile(items: list[dict], client=None) -> tuple[list[dict], str]:
    """([{label, value}], source) 반환. 위기 신호가 있으면 AI로 보내지 않고 원문 앞부분을 쓴다."""
    def pack(values):
        return [{"label": it["label"], "value": v} for it, v in zip(items, values)]

    fallback = pack([fallback_value(it["answer"], it["label"]) for it in items])
    if detect_care([it["answer"] for it in items]):
        return fallback, "care"
    client = client if client is not None else _client
    if client is None:
        return fallback, "fallback"
    try:
        resp = client.with_options(timeout=20.0, max_retries=0).messages.create(
            model=MODEL,
            max_tokens=600,
            system=PROFILE_SYSTEM,
            messages=[{"role": "user", "content": profile_prompt(items)}],
            **effort_options("low"),
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        values = normalize_profile(_parse_json(text), items)
        if values is not None:
            return pack(values), "llm"
    except Exception as e:
        log.warning("하루 요약 생성 실패: %s", type(e).__name__)
    return fallback, "fallback"
