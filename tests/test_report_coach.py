import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coach.report import (
    generate_report_coach, keeps_content, normalize_result, parse_report_request, pick_rewrite_turns,
)
from tests.test_coach import FakeClient, text_resp

LONG = "음 저는 카페에서 일 년 동안 아르바이트를 했고 음 손님 응대를 주로 맡았어요 그래서 바쁠 때도 차분하게 하려고 했어요."
TURNS = [
    {"question": "자기소개 해 주세요", "answer": LONG},
    {"question": "장점은요?", "answer": "꼼꼼한 편이에요."},  # 짧은 대답
    {"question": "갈등 경험은요?", "answer": "팀 과제에서 의견이 갈렸을 때 각자 이유를 듣고 일정표를 다시 짜서 맞췄어요."},
]
STATS = [
    {"question": "자기소개 해 주세요", "fillers": 2, "pauses": 1},
    {"question": "장점은요?", "fillers": 5, "pauses": 4},
    {"question": "갈등 경험은요?", "fillers": 0, "pauses": 0},
]
CANDS = [{"key": "fluency", "line": "'음' 5번 중 3번이 첫머리였어요"}]


def test_parse_rejects_bad_shapes_and_unknown_keys():
    with pytest.raises(ValueError):
        parse_report_request([])
    with pytest.raises(ValueError):
        parse_report_request({"turns": [{"question": "q", "fillers": -1}]})
    with pytest.raises(ValueError):
        parse_report_request({"candidates": [{"key": "score", "line": "x"}]})
    stats, cands = parse_report_request({"turns": STATS, "candidates": CANDS + CANDS})
    assert len(stats) == 3 and cands == CANDS  # 같은 key는 한 번만


def test_pick_only_flagged_and_long_enough_answers():
    picked = pick_rewrite_turns(TURNS, STATS)
    assert [p["question"] for p in picked] == ["자기소개 해 주세요"]  # 짧은 대답·멀쩡한 대답은 빠짐


def test_two_long_pauses_alone_pick_an_answer():
    stats = [{"question": "갈등 경험은요?", "fillers": 0, "pauses": 2}]
    assert [p["question"] for p in pick_rewrite_turns(TURNS, stats)] == ["갈등 경험은요?"]


def test_long_sentence_is_picked_without_any_stats():
    long_one = [{"question": "q", "answer": "가" * 130}]
    assert len(pick_rewrite_turns(long_one, [])) == 1


def test_rewrite_that_adds_new_facts_or_numbers_is_rejected():
    assert keeps_content(LONG, "저는 카페에서 일 년 동안 아르바이트를 했습니다. 손님 응대를 주로 맡았고, 바쁠 때도 차분하게 하려고 했어요.")
    assert not keeps_content(LONG, "저는 3년 동안 카페를 운영했습니다.")
    assert not keeps_content(LONG, "대기업 마케팅 부서에서 인턴으로 근무하며 매출 분석 보고서를 작성했습니다.")


def test_normalize_drops_bad_items():
    picked = pick_rewrite_turns(TURNS, STATS)
    raw = {
        "rewrites": [
            {"id": 0, "text": "저는 카페에서 일 년 동안 아르바이트를 했습니다. 손님 응대를 주로 맡았어요.", "changes": ["'음'을 뺐어요", "x" * 100]},
            {"id": 5, "text": "없는 질문"},
        ],
        "plan": [{"key": "fluency", "how": "숨을 한 번 쉬고 '저는'으로 시작해 봐요."},
                 {"key": "gaze", "how": "후보에 없는 습관"},
                 {"key": "fluency", "how": "중복"}],
    }
    out = normalize_result(raw, picked, CANDS)
    assert [r["question"] for r in out["rewrites"]] == ["자기소개 해 주세요"]
    assert out["rewrites"][0]["changes"] == ["'음'을 뺐어요"]
    assert [p["key"] for p in out["plan"]] == ["fluency"]


def test_banned_terms_in_plan_are_dropped():
    out = normalize_result({"plan": [{"key": "fluency", "how": "말버릇이 부족해요"}]}, [], CANDS)
    assert out["plan"] == []


def test_generate_calls_claude_once_and_returns_checked_result():
    client = FakeClient(text_resp({
        "rewrites": [{"id": 0, "text": "저는 카페에서 일 년 동안 아르바이트를 했습니다. 손님 응대를 주로 맡았어요.", "changes": ["문장을 나눴어요"]}],
        "plan": [{"key": "fluency", "how": "질문이 끝나면 숨을 한 번 쉬고 시작해 봐요."}],
    }))
    result, source = generate_report_coach(TURNS, STATS, CANDS, client=client)
    assert source == "llm" and len(client.calls) == 1
    assert result["picked"] == ["자기소개 해 주세요"]
    assert len(result["rewrites"]) == 1 and len(result["plan"]) == 1


def test_errors_and_bad_output_fall_back_but_keep_the_picked_list():
    for client in (FakeClient(error=RuntimeError("x")), FakeClient(text_resp("JSON이 아님"))):
        result, source = generate_report_coach(TURNS, STATS, CANDS, client=client)
        assert source == "fallback" and result["rewrites"] == [] and result["picked"] == ["자기소개 해 주세요"]


def test_care_signal_skips_claude():
    client = FakeClient()
    turns = TURNS + [{"question": "q", "answer": "요즘 죽고 싶다는 생각이 들어요"}]
    result, source = generate_report_coach(turns, STATS, CANDS, client=client)
    assert source == "care" and client.calls == [] and result["care"]["body"]


def test_nothing_to_do_skips_claude():
    client = FakeClient()
    _, source = generate_report_coach(TURNS[1:], STATS, [], client=client)
    assert source == "fallback" and client.calls == []


# ── API (flask가 있는 환경에서만) ─────────────────────────────────────────

@pytest.fixture(scope="module")
def api(tmp_path_factory):
    pytest.importorskip("flask")
    pytest.importorskip("flask_sqlalchemy")
    pytest.importorskip("librosa")
    os.environ.setdefault("DATABASE_URL", f"sqlite:///{tmp_path_factory.mktemp('db') / 'report.db'}")
    os.environ.setdefault("SECRET_KEY", "test-secret")
    import app as app_module
    return app_module


def test_report_coach_api_saves_once_and_is_deleted_with_the_session(api):
    c = api.app.test_client()
    auth = {"Authorization": f"Bearer {c.post('/api/guest').get_json()['token']}"}
    sid = c.post("/interview/sessions", json={"tier": "standard"}, headers=auth).get_json()["id"]
    tid = c.post(f"/interview/sessions/{sid}/turns", json={"kind": "main", "question": TURNS[0]["question"]}, headers=auth).get_json()["id"]
    c.put(f"/interview/sessions/{sid}/turns/{tid}/answer", json={"answer": LONG}, headers=auth)

    url = f"/interview/sessions/{sid}/report-coach"
    assert c.post(url, json={"turns": "x"}, headers=auth).status_code == 400
    first = c.post(url, json={"turns": STATS, "candidates": CANDS}, headers=auth).get_json()
    assert first["result"]["picked"] == [TURNS[0]["question"]]
    again = c.post(url, json={}, headers=auth).get_json()  # 저장된 것 — 본문이 달라도 그대로
    assert again == first

    c.delete(f"/interview/sessions/{sid}", headers=auth)
    with api.app.app_context():
        assert api.ReportCoach.query.filter_by(session_id=sid).first() is None
