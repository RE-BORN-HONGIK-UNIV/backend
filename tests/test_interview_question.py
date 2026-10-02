import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step3.interview_question import (
    FALLBACK_FOLLOW_UPS,
    FALLBACK_QUESTIONS,
    FOLLOW_UP_INSTRUCTIONS,
    MAX_WEAK_AXES,
    WEAK_BELOW,
    _fallback_question,
    _profile_section,
    _user_prompt,
    _weak_axes,
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


# ── 1·2단계 세부 점수 반영 (질문 방식 조절) ─────────────────────────────

S1_OK = {"stability": 90, "fluency": 90, "pause_ctrl": 90, "continuity": 90, "calm": 90}
S2_OK = {"blink": 90, "gaze": 90, "expression": 90}


def _profile(s1=None, s2=None):
    return {"stage1": {**S1_OK, **(s1 or {})}, "stage2": {**S2_OK, **(s2 or {})}}


def test_no_profile_or_all_stable_adds_no_section():
    assert _profile_section(None) == ""
    assert _profile_section({}) == ""
    assert _profile_section(_profile()) == ""


def test_weak_axes_picks_only_below_threshold():
    # 경계: WEAK_BELOW 미만만 약함, 딱 그 값은 약하지 않음
    assert _weak_axes(_profile(s1={"fluency": WEAK_BELOW - 0.1})) == ["fluency"]
    assert _weak_axes(_profile(s1={"fluency": WEAK_BELOW})) == []


def test_weak_axes_limited_to_lowest_two_in_ascending_order():
    p = _profile(s1={"fluency": 50, "calm": 30}, s2={"gaze": 60})  # 약한 항목 3개
    axes = _weak_axes(p)
    assert len(axes) == MAX_WEAK_AXES
    assert axes == ["calm", "fluency"]  # 가장 낮은 2개, 낮은 순


def test_weak_axes_combines_stage1_and_stage2():
    assert _weak_axes(_profile(s1={"fluency": 40}, s2={"gaze": 20})) == ["gaze", "fluency"]


def test_weak_axes_handles_missing_stage_and_bad_values():
    # 한 단계 기록만 있어도 동작, 숫자가 아닌 값·모르는 축·bool은 무시
    assert _weak_axes({"stage1": None, "stage2": {"gaze": 10}}) == ["gaze"]
    assert _weak_axes({"stage1": {"fluency": "낮음", "unknown": 1, "calm": True}, "stage2": None}) == []


def test_profile_section_never_leaks_scores_or_axis_keys():
    # LLM이 사용자에게 점수·약점을 말하지 못하게, 숫자와 내부 키 이름은 프롬프트에 넣지 않는다.
    section = _profile_section(_profile(s1={"fluency": 41.5}, s2={"gaze": 23}))
    assert section
    for leaked in ("41", "23", "fluency", "gaze", "stage1", "stage2"):
        assert leaked not in section
    assert "절대 언급하지 않습니다" in section


def test_user_prompt_includes_profile_section_in_both_modes():
    p = _profile(s1={"fluency": 40})
    main = _user_prompt("standard", [], None, "main", p)
    follow = _user_prompt("standard", [], "답변", "follow_up", p)
    assert "질문 방식 참고" in main and "질문 방식 참고" in follow


def test_user_prompt_without_profile_is_unchanged():
    assert "질문 방식 참고" not in _user_prompt("standard", [], None, "main")
    assert _user_prompt("standard", [], None, "main") == _user_prompt("standard", [], None, "main", None)
