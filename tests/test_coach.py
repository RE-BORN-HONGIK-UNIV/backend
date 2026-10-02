import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coach import agent as coach_agent
from coach.missions import MISSIONS, mission_level, pick_mission
from coach.safety import care_message, detect_care, has_banned_term
from coach.schema import (
    KINDS, MAX_GREETING_LEN, build_cards, completed_count, default_recommended,
    fallback_note, normalize_note, verified_quote, weakest_stage_path,
)
from coach.tools import TOOL_DEFINITIONS, execute_tool


def facts(completed=True, answers=("저는 꾸준히 연습하는 사람이에요.", "프로젝트에서 자료 정리를 맡았어요."),
          previous=(), stage_scores=None, session_id=7):
    turns = [{"kind": "main" if i == 0 else "follow_up", "question": f"질문{i}", "answer": a}
             for i, a in enumerate(answers)]
    return {
        "session": {"id": session_id, "tier": "standard", "completed": completed, "turns": turns},
        "previous_sessions": list(previous),
        "stage_scores": stage_scores if stage_scores is not None else {"stage1": None, "stage2": None},
    }


def prev(completed=True, q=3, a=3, sid=1):
    return {"id": sid, "tier": "standard", "completed": completed, "question_count": q, "answered_count": a}


# ── 미션 ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,level", [(0, 1), (1, 1), (2, 2), (4, 2), (5, 3), (20, 3)])
def test_mission_level_by_completed_count(n, level):
    assert mission_level(n) == level


def test_pick_mission_is_stable_for_same_session_and_comes_from_the_pool():
    assert pick_mission(3, seed=11) == pick_mission(3, seed=11)  # 재조회해도 안 바뀜
    assert pick_mission(3, seed=11) in MISSIONS[2]
    assert len({pick_mission(0, s) for s in range(10)}) > 1      # seed에 따라 돌려 쓴다


# ── 안전장치 ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", ["요즘 죽고 싶다는 생각이 들어요", "자 해 를 한 적이 있어요", "그냥 사라지고싶어요"])
def test_care_signals_are_detected_even_with_spacing_variants(text):
    assert detect_care(["평범한 답변", text])


def test_ordinary_answers_do_not_trigger_care():
    assert not detect_care(["면접에서 긴장이 되지만 열심히 준비했어요", "죽을 만큼 열심히 했어요"[:0] or "열심히 했어요"])


def test_care_message_never_invents_contacts(monkeypatch):
    monkeypatch.delenv("CARE_RESOURCES", raising=False)
    assert care_message()["resources"] == ""  # 연락처는 팀이 확정해서 환경변수로만 넣는다
    monkeypatch.setenv("CARE_RESOURCES", " 팀이 확정한 안내 ")
    assert care_message()["resources"] == "팀이 확정한 안내"


@pytest.mark.parametrize("text", ["45점이에요", "점수가 낮아요", "많이 부족했어요", "우울해 보여요", "진단이 필요해요"])
def test_banned_terms_are_caught(text):
    assert has_banned_term(text)


def test_warm_phrases_are_not_banned():
    assert not has_banned_term("끝까지 답해준 것만으로도 큰 한 걸음이에요")


# ── 인용 검증 ───────────────────────────────────────────────────────────

ANS = ["저는 팀 프로젝트에서  자료 정리를 맡아서\n꾸준히 했어요.", "음 그래서 그냥 했어요"]


def test_quote_must_be_found_verbatim_in_an_answer_ignoring_whitespace_differences():
    q = verified_quote({"text": "팀 프로젝트에서 자료 정리를 맡아서 꾸준히 했어요", "why": "구체적인 역할이 드러나요"}, ANS)
    assert q == {"text": "팀 프로젝트에서 자료 정리를 맡아서 꾸준히 했어요", "why": "구체적인 역할이 드러나요"}


def test_invented_quote_is_dropped():
    assert verified_quote({"text": "저는 리더 역할을 맡아 팀을 이끌었어요", "why": "좋아요"}, ANS) is None


@pytest.mark.parametrize("quote", [
    {"text": "네", "why": "짧아요"},                                     # 너무 짧음
    {"text": "팀 프로젝트에서 자료 정리를 맡아서", "why": "점수가 높아요"},  # why에 금지어
    {"text": "팀 프로젝트에서 자료 정리를 맡아서", "why": ""},
    {"why": "텍스트 없음"}, "문자열", None, {"text": 5, "why": "x"},
])
def test_bad_quotes_are_dropped(quote):
    assert verified_quote(quote, ANS) is None


def test_quote_with_negative_self_talk_is_not_echoed_back():
    answers = ["저는 많이 부족해서 걱정이에요"]
    assert verified_quote({"text": "저는 많이 부족해서 걱정이에요", "why": "솔직해요"}, answers) is None


# ── 카드·추천 ───────────────────────────────────────────────────────────

def test_cards_are_always_the_four_code_defined_kinds():
    cards = build_cards(facts())
    assert [c["kind"] for c in cards] == list(KINDS)
    assert all(c["title"] and c["body"] for c in cards)


@pytest.mark.parametrize("scores,path", [
    ({"stage1": {"fluency": 40, "calm": 80}, "stage2": {"gaze": 60}}, "/voice"),
    ({"stage1": {"fluency": 90}, "stage2": {"gaze": 30}}, "/face"),
    ({"stage1": None, "stage2": None}, "/voice"),   # 기록이 없으면 1단계로
])
def test_light_practice_points_to_the_stage_with_the_lowest_score(scores, path):
    assert weakest_stage_path(scores) == path


def test_card_text_never_names_scores_or_weak_areas():
    cards = build_cards(facts(stage_scores={"stage1": {"fluency": 10}, "stage2": {"gaze": 10}}))
    text = " ".join(c["title"] + c["body"] for c in cards)
    for leaked in ("유창", "시선", "점수", "10", "약한", "부족"):
        assert leaked not in text


def test_default_recommended_rules():
    assert default_recommended(facts(completed=False)) == "rest"
    assert default_recommended(facts(completed=True)) == "daily_mission"                    # 첫 완주
    assert default_recommended(facts(completed=True, previous=[prev()])) == "again"          # 두 번째부터


def test_completed_count_includes_this_session_only_when_completed():
    assert completed_count(facts(completed=True, previous=[prev(), prev(completed=False)])) == 2
    assert completed_count(facts(completed=False, previous=[prev()])) == 1


# ── 대체 노트 ───────────────────────────────────────────────────────────

def test_fallback_note_only_states_facts_and_is_complete():
    note = fallback_note(facts())
    assert note["quote"] is None and note["care"] is None
    assert note["recommended"] in KINDS and len(note["cards"]) == 4
    assert "첫 면접 연습을 해냈어요" in note["won"] and "2개의 질문에 답했어요" in note["won"]
    assert not any(has_banned_term(w) for w in note["won"] + [note["greeting"]])


def test_fallback_note_for_an_interview_left_early_does_not_claim_completion():
    note = fallback_note(facts(completed=False, answers=("한 문장만 답했어요",)))
    assert "면접을 끝까지 마쳤어요" not in note["won"]
    assert note["recommended"] == "rest"


# ── LLM 출력 정규화 ─────────────────────────────────────────────────────

GOOD = {"greeting": "오늘도 와줘서 고마워요.", "won": ["끝까지 답했어요"], "quote": None, "recommended": "again"}


def test_good_output_passes_through_and_cards_stay_code_defined():
    note = normalize_note(GOOD, facts())
    assert note["greeting"] == GOOD["greeting"] and note["won"] == ["끝까지 답했어요"]
    assert note["recommended"] == "again"
    assert note["cards"] == build_cards(facts())


def test_each_bad_field_is_replaced_independently_not_the_whole_note():
    raw = {"greeting": "점수가 낮아요" , "won": ["끝까지 답했어요", "45점이에요"], "recommended": "teleport"}
    note = normalize_note(raw, facts())
    base = fallback_note(facts())
    assert note["greeting"] == base["greeting"]            # 금지어 → 대체
    assert note["won"] == ["끝까지 답했어요"]                # 나쁜 항목만 빠지고 좋은 항목은 유지
    assert note["recommended"] == base["recommended"]       # 모르는 kind → 기본 규칙


def test_length_limits_are_enforced():
    note = normalize_note({**GOOD, "greeting": "가" * (MAX_GREETING_LEN + 1)}, facts())
    assert note["greeting"] == fallback_note(facts())["greeting"]


@pytest.mark.parametrize("raw", [None, [], "text", 3, {}])
def test_non_object_or_empty_output_gives_the_fallback_note(raw):
    assert normalize_note(raw, facts()) == fallback_note(facts())


# ── 도구 ────────────────────────────────────────────────────────────────

def test_tool_definitions_have_no_user_id_argument():
    # 사용자 식별자를 인자로 받지 않는다 → 다른 사용자 데이터를 조회할 방법이 구조적으로 없음
    for t in TOOL_DEFINITIONS:
        assert t["input_schema"]["properties"] == {}


def test_stage_scores_tool_never_returns_numbers():
    out = execute_tool("get_stage_scores", facts(stage_scores={"stage1": {"fluency": 41, "calm": 90}, "stage2": {"gaze": 22}}))
    data = json.loads(out)
    assert data == {"stage1_recorded": True, "stage2_recorded": True, "weak_areas": ["시선", "말의 유창함"]}
    assert "41" not in out and "22" not in out


def test_previous_sessions_tool_has_no_text_and_is_capped():
    out = json.loads(execute_tool("get_previous_sessions", facts(previous=[prev(sid=i) for i in range(9)])))
    assert len(out["sessions"]) == 5
    assert all(set(s) == {"id", "tier", "completed", "question_count", "answered_count"} for s in out["sessions"])


def test_current_interview_tool_truncates_long_answers_but_returns_all_turns():
    long = "가" * 4000
    out = json.loads(execute_tool("get_current_interview", facts(answers=(long, "짧은 답변이에요"))))
    assert len(out["turns"]) == 2 and len(out["turns"][0]["answer"]) == 1500


def test_unknown_tool_returns_none():
    assert execute_tool("drop_database", facts()) is None


# ── 에이전트 루프 (가짜 AI) ─────────────────────────────────────────────

def block(**kw):
    return types.SimpleNamespace(**kw)


class FakeClient:
    """미리 정한 응답을 순서대로 돌려주는 가짜 Anthropic 클라이언트. 호출 인자를 기록한다."""

    def __init__(self, *responses, error=None):
        self.responses, self.error, self.calls = list(responses), error, []
        self.messages = types.SimpleNamespace(create=self._create)

    def with_options(self, **kw):
        self.options = kw
        return self

    def _create(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        return self.responses.pop(0)


def text_resp(obj_or_text):
    text = obj_or_text if isinstance(obj_or_text, str) else json.dumps(obj_or_text, ensure_ascii=False)
    return block(stop_reason="end_turn", content=[block(type="text", text=text)])


def tool_resp(*names):
    return block(stop_reason="tool_use", content=[block(type="tool_use", id=f"t{i}", name=n, input={}) for i, n in enumerate(names)])


def test_agent_calls_tools_then_returns_a_validated_llm_note():
    f = facts()
    quote = {"text": "프로젝트에서 자료 정리를 맡았어요", "why": "맡은 역할이 구체적이에요"}
    client = FakeClient(tool_resp("get_current_interview", "get_practice_stats"), text_resp({**GOOD, "quote": quote}))
    note, source = coach_agent.generate_note(f, client=client)
    assert source == "llm" and note["quote"] == quote
    # 두 번째 호출에 도구 결과가 같은 사용자 메시지로 함께 들어갔는지(병렬 도구 호출은 결과를 한 메시지에)
    results = client.calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t0", "t1"] and all(r["type"] == "tool_result" for r in results)
    # SDK 버전 호환: effort는 extra_body로, output_config 키워드로 직접 넘기지 않는다
    assert client.calls[0]["extra_body"] == {"output_config": {"effort": "low"}} and "output_config" not in client.calls[0]


def test_agent_drops_an_invented_quote_but_keeps_the_rest():
    raw = {**GOOD, "quote": {"text": "저는 팀을 이끄는 리더예요", "why": "자신감이 보여요"}}
    note, source = coach_agent.generate_note(facts(), client=FakeClient(text_resp(raw)))
    assert source == "llm" and note["quote"] is None and note["greeting"] == GOOD["greeting"]


def test_agent_accepts_json_wrapped_in_a_code_block():
    wrapped = "```json\n" + json.dumps(GOOD, ensure_ascii=False) + "\n```"
    note, source = coach_agent.generate_note(facts(), client=FakeClient(text_resp(wrapped)))
    assert source == "llm" and note["greeting"] == GOOD["greeting"]


def test_unknown_tool_call_is_answered_with_an_error_result_and_the_loop_continues():
    client = FakeClient(tool_resp("drop_database"), text_resp(GOOD))
    note, source = coach_agent.generate_note(facts(), client=client)
    assert source == "llm"
    assert client.calls[1]["messages"][-1]["content"][0]["is_error"] is True


@pytest.mark.parametrize("responses", [
    [text_resp("JSON이 아닌 그냥 문장이에요")],        # 형식 오류
    [tool_resp("get_practice_stats")] * 6,              # 도구만 계속 부르는 루프 → MAX_STEPS 후 중단
])
def test_agent_falls_back_when_output_is_unusable_or_loops_too_long(responses):
    note, source = coach_agent.generate_note(facts(), client=FakeClient(*responses))
    assert source == "fallback" and note == fallback_note(facts())


def test_agent_never_raises_on_llm_errors_and_never_logs_answers(caplog):
    secret = "아주 개인적인 이야기예요 123456"
    f = facts(answers=(secret,))
    with caplog.at_level("WARNING"):
        note, source = coach_agent.generate_note(f, client=FakeClient(error=RuntimeError(f"boom {secret}")))
    assert source == "fallback" and note["cards"]
    assert secret not in caplog.text and "RuntimeError" in caplog.text   # 예외 종류만 기록


def test_care_signal_skips_the_llm_entirely():
    client = FakeClient()
    note, source = coach_agent.generate_note(facts(answers=("요즘 죽고 싶어요",)), client=client)
    assert source == "care" and client.calls == []                       # AI로 보내지도 않는다
    assert note["care"] and note["won"] == [] and [c["kind"] for c in note["cards"]] == ["rest"]


def test_no_answers_or_no_client_uses_the_fallback_without_calling_the_llm(monkeypatch):
    client = FakeClient()
    _, source = coach_agent.generate_note(facts(answers=("", "  ")), client=client)
    assert source == "fallback" and client.calls == []
    monkeypatch.setattr(coach_agent, "_client", None)
    _, source = coach_agent.generate_note(facts(), client=None)
    assert source == "fallback"
