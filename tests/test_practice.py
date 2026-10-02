import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coach.practice import (
    MAX_OPENING_LEN, MAX_STEP_LEN, RATE_LIMIT_PER_HOUR, RateLimiter, fallback_hint, generate_hint,
    normalize_hint, parse_hint_request, pick_practice_turn,
)
from coach.safety import has_banned_term
from coach.schema import build_cards


def turn(answer, kind="main"):
    return {"kind": kind, "question": "질문", "answer": answer}


# ── 연습할 질문 고르기 ──────────────────────────────────────────────────

def test_picks_the_first_unanswered_question():
    assert pick_practice_turn([turn("길게 잘 답했어요 정말로"), turn(""), turn("짧음"), turn("   ")]) == 1


def test_picks_the_shortest_answer_when_everything_was_answered_earliest_on_ties():
    assert pick_practice_turn([turn("열 글자가 넘는 답변이에요"), turn("짧아요"), turn("짧아요")]) == 1


def test_no_turns_means_nothing_to_practice():
    assert pick_practice_turn([]) is None


def test_whitespace_only_counts_as_unanswered():
    assert pick_practice_turn([turn("답했어요 충분히"), turn(" \n ")]) == 1


# ── 요청 검증 ───────────────────────────────────────────────────────────

def test_request_parses_question_and_optional_previous_answer():
    assert parse_hint_request({"question": " 강점이 뭔가요? ", "previous_answer": " 꾸준함 "}) == ("강점이 뭔가요?", "꾸준함")
    assert parse_hint_request({"question": "q"}) == ("q", "")
    assert parse_hint_request({"question": "q", "previous_answer": None}) == ("q", "")


@pytest.mark.parametrize("body", [None, [], {}, {"question": ""}, {"question": "  "}, {"question": 3},
                                  {"question": "q", "previous_answer": 5}, {"question": "가" * 1001},
                                  {"question": "q", "previous_answer": "가" * 5001}])
def test_bad_requests_are_rejected(body):
    with pytest.raises(ValueError):
        parse_hint_request(body)


# ── 힌트 검증 ───────────────────────────────────────────────────────────

GOOD = {"opening": "제가 ___에서 ___을 맡았을 때 가장 기억에 남는 건 ___예요.",
        "steps": ["그때의 상황을 한 문장으로 말해보세요.", "그 안에서 내가 한 일을 말해보세요.", "결과나 배운 점으로 마무리해보세요."]}


def test_good_hint_passes_through():
    assert normalize_hint(GOOD) == GOOD


def test_opening_without_a_blank_is_dropped_because_it_would_write_the_answer_for_the_user():
    hint = normalize_hint({**GOOD, "opening": "저는 팀 프로젝트에서 자료 정리를 맡아 꾸준히 했어요."})
    assert hint["opening"] == fallback_hint()["opening"]
    assert hint["steps"] == GOOD["steps"]            # 나머지 필드는 그대로


def test_each_field_falls_back_independently():
    hint = normalize_hint({"opening": "점수가 낮아요 ___", "steps": ["좋은 안내예요", "점수를 올려보세요", "다른 안내예요"]})
    assert hint["opening"] == fallback_hint()["opening"]           # 금지어
    assert hint["steps"] == ["좋은 안내예요", "다른 안내예요"]         # 나쁜 항목만 빠짐 (2개 이상이라 유지)


def test_too_few_valid_steps_use_the_fallback_steps_and_steps_are_capped_at_three():
    assert normalize_hint({**GOOD, "steps": ["하나뿐이에요"]})["steps"] == fallback_hint()["steps"]
    assert len(normalize_hint({**GOOD, "steps": [f"안내 {i}번이에요" for i in range(9)]})["steps"]) == 3


def test_length_limits():
    assert normalize_hint({**GOOD, "opening": "가" * MAX_OPENING_LEN + "___"})["opening"] == fallback_hint()["opening"]
    assert normalize_hint({**GOOD, "steps": ["가" * (MAX_STEP_LEN + 1), "나" * (MAX_STEP_LEN + 1)]})["steps"] == fallback_hint()["steps"]


@pytest.mark.parametrize("raw", [None, [], "text", 3, {}])
def test_non_object_or_empty_gives_the_fallback_hint(raw):
    assert normalize_hint(raw) == fallback_hint()


def test_fallback_hint_is_itself_valid_and_free_of_evaluation_words():
    h = fallback_hint()
    assert "___" in h["opening"] and 2 <= len(h["steps"]) <= 3
    assert not any(has_banned_term(t) for t in [h["opening"], *h["steps"]])
    assert normalize_hint(h) == h


# ── AI 호출 (가짜 AI) ───────────────────────────────────────────────────

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


def test_llm_hint_is_validated_and_effort_goes_through_extra_body():
    client = Fake(json.dumps(GOOD, ensure_ascii=False))
    hint, source = generate_hint("강점이 뭔가요?", "꾸준함이에요", client=client)
    assert source == "llm" and hint == GOOD
    call = client.calls[0]
    assert call["extra_body"] == {"output_config": {"effort": "low"}} and "output_config" not in call
    assert "강점이 뭔가요?" in call["messages"][0]["content"] and "꾸준함이에요" in call["messages"][0]["content"]


def test_only_the_beginning_of_a_long_previous_answer_is_sent_to_the_ai():
    client = Fake(json.dumps(GOOD, ensure_ascii=False))
    generate_hint("q", "가" * 3000, client=client)
    assert "가" * 600 in client.calls[0]["messages"][0]["content"]
    assert "가" * 601 not in client.calls[0]["messages"][0]["content"]


def test_no_previous_answer_is_marked_as_none():
    client = Fake(json.dumps(GOOD, ensure_ascii=False))
    generate_hint("q", "", client=client)
    assert "(없음)" in client.calls[0]["messages"][0]["content"]


@pytest.mark.parametrize("client", [Fake("JSON이 아니에요"), Fake(error=RuntimeError("boom 비밀 내용"))])
def test_unusable_output_or_errors_give_the_fallback_hint(client):
    hint, source = generate_hint("q", "답변", client=client)
    assert source == "fallback" and hint == fallback_hint()


def test_errors_never_log_the_answer(caplog):
    with caplog.at_level("WARNING"):
        generate_hint("q", "아주 개인적인 이야기 987", client=Fake(error=RuntimeError("보낸 내용: 아주 개인적인 이야기 987")))
    assert "아주 개인적인 이야기" not in caplog.text and "RuntimeError" in caplog.text


def test_care_signal_in_question_or_answer_is_never_sent_to_the_ai():
    for q, a in (("q", "요즘 죽고 싶어요"), ("자해 경험이 있나요", "")):
        client = Fake()
        hint, source = generate_hint(q, a, client=client)
        assert source == "fallback" and client.calls == [] and hint == fallback_hint()


def test_without_a_client_the_fallback_hint_is_used(monkeypatch):
    from coach import agent
    monkeypatch.setattr(agent, "_client", None)
    assert generate_hint("q", "a")[1] == "fallback"


# ── 호출 제한 ───────────────────────────────────────────────────────────

def test_rate_limiter_blocks_after_the_limit_and_recovers_after_the_window():
    rl = RateLimiter(limit=3, window_sec=100)
    assert [rl.allow("u", now=t) for t in (0, 1, 2)] == [True, True, True]
    assert rl.allow("u", now=3) is False
    assert rl.allow("u", now=50) is False
    assert rl.allow("u", now=101) is True           # 첫 호출이 창 밖으로 나가 다시 가능


def test_rate_limiter_is_per_user():
    rl = RateLimiter(limit=1, window_sec=100)
    assert rl.allow("a", now=0) and not rl.allow("a", now=1)
    assert rl.allow("b", now=1)


def test_default_limit_is_generous_enough_for_normal_practice():
    assert RATE_LIMIT_PER_HOUR >= 10


# ── 코치 노트 카드와의 연결 ─────────────────────────────────────────────

def facts(answers):
    turns = [{"kind": "main", "question": f"질문{i}", "answer": a} for i, a in enumerate(answers)]
    return {"session": {"id": 7, "tier": "standard", "completed": True, "turns": turns},
            "previous_sessions": [], "stage_scores": {"stage1": {"fluency": 10}, "stage2": None}}


def test_practice_card_points_to_the_turn_to_retry_without_naming_weak_areas():
    card = next(c for c in build_cards(facts(["충분히 길게 답했어요 정말", ""])) if c["kind"] == "light_practice")
    assert card["practiceTurn"] == 1 and "path" not in card
    assert card["title"] == "이 질문 다시 답해보기"
    assert "유창" not in card["body"] and "점수" not in card["body"]


def test_practice_card_falls_back_to_the_stage_screen_when_there_is_no_question():
    card = next(c for c in build_cards(facts([])) if c["kind"] == "light_practice")
    assert "practiceTurn" not in card and card["path"] == "/voice"
