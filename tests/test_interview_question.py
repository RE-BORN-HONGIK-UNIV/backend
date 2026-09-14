import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step3.interview_question import _fallback_question, _user_prompt, FALLBACK_QUESTIONS


def test_fallback_question_picks_from_tier_pool():
    q = _fallback_question("warmup", previous_questions=[])
    assert q in FALLBACK_QUESTIONS["warmup"]


def test_fallback_question_avoids_already_asked():
    pool = FALLBACK_QUESTIONS["standard"]
    already_asked = pool[:-1]  # 마지막 1개만 남김
    q = _fallback_question("standard", previous_questions=already_asked)
    assert q == pool[-1]


def test_fallback_question_resets_pool_when_all_asked():
    pool = FALLBACK_QUESTIONS["practice"]
    q = _fallback_question("practice", previous_questions=list(pool))
    assert q in pool  # 다 나왔으면 처음부터 다시 고름 — 빈 리스트에서 고르다 죽지 않음


def test_fallback_question_unknown_tier_uses_standard_pool():
    q = _fallback_question("unknown-tier", previous_questions=[])
    assert q in FALLBACK_QUESTIONS["standard"]


def test_user_prompt_lists_previous_questions():
    prompt = _user_prompt("standard", previous_questions=["첫 질문?"])
    assert "첫 질문?" in prompt
    assert "이미 나온 질문" in prompt


def test_user_prompt_marks_no_previous_questions():
    prompt = _user_prompt("warmup", previous_questions=[])
    assert "(아직 없음)" in prompt


def test_user_prompt_includes_previous_answer_when_given():
    prompt = _user_prompt("standard", previous_questions=[], previous_answer="이렇게 답했어요")
    assert "이렇게 답했어요" in prompt
    assert "꼬리질문" in prompt


def test_user_prompt_omits_answer_section_when_none():
    prompt = _user_prompt("standard", previous_questions=[], previous_answer=None)
    assert "방금 사용자 답변" not in prompt
