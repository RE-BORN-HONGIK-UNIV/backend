import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step3.interview_question import (
    FALLBACK_FOLLOW_UPS,
    FALLBACK_QUESTIONS,
    FOLLOW_UP_INSTRUCTIONS,
    _fallback_question,
    _user_prompt,
)


# ── 폴백 질문: 기본 질문(main) ──────────────────────────────────────

def test_fallback_question_picks_from_tier_pool():
    q = _fallback_question("warmup", previous_questions=[], mode="main")
    assert q in FALLBACK_QUESTIONS["warmup"]


def test_fallback_question_avoids_already_asked():
    pool = FALLBACK_QUESTIONS["standard"]
    already_asked = pool[:-1]  # 마지막 1개만 남김
    q = _fallback_question("standard", previous_questions=already_asked, mode="main")
    assert q == pool[-1]


def test_fallback_question_resets_pool_when_all_asked():
    pool = FALLBACK_QUESTIONS["practice"]
    q = _fallback_question("practice", previous_questions=list(pool), mode="main")
    assert q in pool  # 다 나왔으면 처음부터 다시 고름 — 빈 리스트에서 고르다 죽지 않음


def test_fallback_question_unknown_tier_uses_standard_pool():
    q = _fallback_question("unknown-tier", previous_questions=[], mode="main")
    assert q in FALLBACK_QUESTIONS["standard"]


def test_fallback_question_skips_self_intro_when_already_done():
    # 첫 질문(면접관별로 문장이 다른 자기소개)에서 이미 했으면, 폴백이 자기소개를 다시 묻지 않는다.
    # 문장이 달라서 "이미 나온 질문" 완전 일치 비교로는 걸러지지 않기 때문에 따로 검사한다.
    asked = ["먼저 간단하게 자기소개 부탁드려요. 편하게 말해주시면 돼요."]
    for _ in range(30):  # 무작위 선택이라 반복해서 확인
        q = _fallback_question("warmup", previous_questions=asked, mode="main")
        assert "자기소개" not in q


# ── 폴백 질문: 꼬리질문(follow_up) ──────────────────────────────────

def test_fallback_follow_up_uses_follow_up_pool_not_main_pool():
    for tier in ("warmup", "standard", "practice"):
        q = _fallback_question(tier, previous_questions=[], mode="follow_up")
        assert q in FALLBACK_FOLLOW_UPS[tier]
        assert q not in FALLBACK_QUESTIONS[tier]


def test_fallback_follow_up_unknown_tier_uses_standard_pool():
    q = _fallback_question("unknown-tier", previous_questions=[], mode="follow_up")
    assert q in FALLBACK_FOLLOW_UPS["standard"]


# ── 프롬프트 조립 ────────────────────────────────────────────────────

def test_user_prompt_lists_previous_questions():
    prompt = _user_prompt("standard", previous_questions=["첫 질문?"], previous_answer=None, mode="main")
    assert "첫 질문?" in prompt
    assert "이미 나온 질문" in prompt


def test_user_prompt_marks_no_previous_questions():
    prompt = _user_prompt("warmup", previous_questions=[], previous_answer=None, mode="main")
    assert "(아직 없음)" in prompt


def test_user_prompt_follow_up_includes_answer_and_tier_depth():
    prompt = _user_prompt(
        "practice", previous_questions=[], previous_answer="이렇게 답했어요", mode="follow_up"
    )
    assert "이렇게 답했어요" in prompt
    assert "꼬리질문" in prompt
    assert FOLLOW_UP_INSTRUCTIONS["practice"] in prompt  # 난이도별로 파고드는 깊이가 다름


def test_user_prompt_main_asks_new_topic_and_ignores_answer():
    # 기본 질문은 직전 답변을 파고들면 안 되므로, 답변이 넘어와도 프롬프트에 넣지 않는다.
    prompt = _user_prompt(
        "standard", previous_questions=[], previous_answer="이렇게 답했어요", mode="main"
    )
    assert "새로운 기본 질문" in prompt
    assert "이렇게 답했어요" not in prompt
    assert "방금 사용자 답변" not in prompt
