import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step3.session_payload import (
    MAX_ANSWER_LEN,
    MAX_QUESTION_LEN,
    parse_answer,
    parse_session_start,
    parse_turn,
)


@pytest.mark.parametrize("tier", ["warmup", "standard", "practice"])
def test_session_start_accepts_known_tiers(tier):
    assert parse_session_start({"tier": tier}) == tier


@pytest.mark.parametrize("body", [None, [], "x", {}, {"tier": "hard"}, {"tier": None}, {"tier": 1}])
def test_session_start_rejects_bad_input(body):
    with pytest.raises(ValueError):
        parse_session_start(body)


def test_turn_returns_kind_and_stripped_question():
    assert parse_turn({"kind": "follow_up", "question": "  왜 그렇게 하셨나요?  "}) == ("follow_up", "왜 그렇게 하셨나요?")


@pytest.mark.parametrize("body", [
    None, [], {"kind": "main"}, {"question": "q"},
    {"kind": "other", "question": "q"},
    {"kind": "main", "question": ""}, {"kind": "main", "question": "   "},
    {"kind": "main", "question": 3},
])
def test_turn_rejects_bad_input(body):
    with pytest.raises(ValueError):
        parse_turn(body)


def test_turn_question_length_limit_boundary():
    parse_turn({"kind": "main", "question": "가" * MAX_QUESTION_LEN})  # 경계값은 통과
    with pytest.raises(ValueError):
        parse_turn({"kind": "main", "question": "가" * (MAX_QUESTION_LEN + 1)})


def test_answer_is_stripped_and_empty_is_allowed():
    assert parse_answer({"answer": "  안녕하세요  "}) == "안녕하세요"
    assert parse_answer({"answer": ""}) == ""        # 음성 인식 실패 = 답변 건너뜀 기록
    assert parse_answer({"answer": "   "}) == ""


@pytest.mark.parametrize("body", [None, [], {}, {"answer": None}, {"answer": 5}, {"answer": ["a"]}])
def test_answer_rejects_non_string(body):
    with pytest.raises(ValueError):
        parse_answer(body)


def test_answer_length_limit_boundary():
    parse_answer({"answer": "가" * MAX_ANSWER_LEN})
    with pytest.raises(ValueError):
        parse_answer({"answer": "가" * (MAX_ANSWER_LEN + 1)})
