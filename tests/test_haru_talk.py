import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from haru.talk import (
    CARE_LINE, CLOSING, MAX_PROFILE_VALUE_LEN, MAX_SAY_LEN, NODS,
    fallback_reply, fallback_value, generate_reply, normalize_profile, normalize_reply,
    parse_profile_request, parse_reply_request, reply_prompt, summarize_profile,
)


class Fake:
    def __init__(self, *texts, error=None):
        self.texts, self.error, self.calls = list(texts), error, []
        self.messages = types.SimpleNamespace(create=self._create)

    def with_options(self, **kw):
        return self

    def _create(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=self.texts.pop(0))])


def req(**kw):
    base = {"question": "요즘 어떻게 지내세요?", "answer": "요리를 배우고 있어요", "next": "관심 있는 건 뭐예요?",
            "retried": False, "index": 1, "history": []}
    base.update(kw)
    return parse_reply_request(base)


# ── 요청 검증 ───────────────────────────────────────────────────────────

def test_reply_request_needs_a_question():
    with pytest.raises(ValueError):
        parse_reply_request({"answer": "네"})
    with pytest.raises(ValueError):
        parse_reply_request("not a dict")


def test_reply_request_clips_long_text_and_keeps_recent_history():
    r = req(answer="가" * 5000, history=[{"ai": i % 2 == 0, "text": f"줄{i}"} for i in range(20)])
    assert len(r["answer"]) == 800
    assert [h["text"] for h in r["history"]] == [f"줄{i}" for i in range(14, 20)]


def test_reply_request_empty_next_means_last_question():
    assert req(next="").get("next") is None


# ── 폴백 ───────────────────────────────────────────────────────────────

def test_fallback_nods_then_asks_next_question():
    out = fallback_reply(req(index=2))
    assert out == {"say": f"{NODS[2]} 관심 있는 건 뭐예요?", "retry": False}


def test_fallback_on_last_question_is_a_closing_line():
    assert fallback_reply(req(next=None))["say"] == CLOSING


def test_crisis_answer_is_not_sent_to_ai():
    client = Fake('{"say": "x", "retry": false}')
    out, source = generate_reply(req(answer="요즘은 그냥 죽고 싶어요"), client=client)
    assert source == "care" and client.calls == []
    assert out["say"].startswith(CARE_LINE) and out["retry"] is False


def test_no_client_falls_back():
    from haru import talk
    saved, talk._client = talk._client, None
    try:
        out, source = generate_reply(req())
    finally:
        talk._client = saved
    assert source == "fallback" and out["say"].endswith("관심 있는 건 뭐예요?")


# ── AI 출력 검증 ────────────────────────────────────────────────────────

def test_llm_reply_is_used_and_effort_goes_through_extra_body():
    client = Fake('{"say": "요리를 배우시는군요. 요즘 관심 있는 건 뭐예요?", "retry": false}')
    out, source = generate_reply(req(), client=client)
    assert source == "llm" and out["say"].startswith("요리를")
    call = client.calls[0]
    assert "output_config" not in call  # anthropic==0.69.0엔 이 키워드가 없음 — extra_body로만
    assert call["extra_body"]["output_config"]["effort"] == "low"
    assert "요리를 배우고 있어요" in call["messages"][0]["content"]


def test_retry_is_forced_off_once_already_retried():
    out = normalize_reply({"say": "쉬운 말로 다시 물어볼게요.", "retry": True}, req(retried=True))
    assert out["retry"] is False


@pytest.mark.parametrize("raw", [
    None, {"retry": True}, {"say": ""}, {"say": "가" * (MAX_SAY_LEN + 1)},
    {"say": "말투에 긴장이 보여서 점수는 45점이에요."},  # 평가 표현
])
def test_bad_ai_output_is_rejected(raw):
    assert normalize_reply(raw, req()) is None


def test_bad_ai_output_falls_back_and_api_error_falls_back():
    out, source = generate_reply(req(), client=Fake("JSON 아님"))
    assert source == "fallback" and out["retry"] is False
    out, source = generate_reply(req(), client=Fake(error=RuntimeError("boom")))
    assert source == "fallback"


def test_prompt_marks_empty_answer_and_last_question():
    p = reply_prompt(req(answer="", next=None))
    assert "(말이 글로 잡히지 않았다)" in p and "(없음, 마지막)" in p


# ── 이해 카드 요약 ─────────────────────────────────────────────────────

ITEMS = [
    {"label": "부를 이름", "question": "뭐라고 불러 드리면 될까요?", "answer": "지우라고 불러 주세요"},
    {"label": "관심사", "question": "관심 있는 건 뭐예요?", "answer": ""},
]


def test_profile_request_needs_items_with_labels():
    with pytest.raises(ValueError):
        parse_profile_request({"items": []})
    with pytest.raises(ValueError):
        parse_profile_request({"items": [{"answer": "x"}]})
    assert parse_profile_request({"items": ITEMS})[0]["label"] == "부를 이름"


def test_fallback_value_shortens_long_answers():
    assert fallback_value("짧은 대답") == "짧은 대답"
    long = fallback_value("가" * 100)
    assert len(long) == MAX_PROFILE_VALUE_LEN and long.endswith("…")


def test_profile_values_cannot_invent_content_for_empty_answers():
    items = parse_profile_request({"items": ITEMS})
    assert normalize_profile({"values": ["지우", "요리"]}, items) == ["지우", ""]


def test_profile_wrong_count_is_rejected_and_bad_item_uses_answer():
    items = parse_profile_request({"items": ITEMS})
    assert normalize_profile({"values": ["지우"]}, items) is None
    assert normalize_profile({"values": ["진단상 불안 증상", ""]}, items)[0] == "지우라고 불러 주세요"


def test_profile_llm_and_crisis_paths():
    items = parse_profile_request({"items": ITEMS})
    out, source = summarize_profile(items, client=Fake(json.dumps({"values": ["지우", ""]}, ensure_ascii=False)))
    assert source == "llm" and out == [{"label": "부를 이름", "value": "지우"}, {"label": "관심사", "value": ""}]

    client = Fake('{"values": ["x", "y"]}')
    crisis = parse_profile_request({"items": [{"label": "걱정", "question": "q", "answer": "자해하고 싶어요"}]})
    _, source = summarize_profile(crisis, client=client)
    assert source == "care" and client.calls == []
