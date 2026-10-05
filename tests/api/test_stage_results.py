"""1·2단계 결과 저장/조회 API — 3단계 난이도·대시보드·"지난번 대비" 비교가 이 값들을 읽는다."""
import datetime

import pytest

LIVE_PAYLOAD = {
    "blinkRatePerMin": 18.0,
    "blinkStatus": "정상",
    "blinkScore": 100,
    "avgFixationSec": 3.5,
    "gazeScore": 90,
    "smileRatio": 0.3,
    "tensionRatio": 0.1,
    "expressionScore": 80,
    "expressionStatus": "편안함",
}


def _user_id(app_module):
    with app_module.app.app_context():
        return app_module.User.query.first().id


# ── 인증 ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/analyze/stage1/latest"),
        ("get", "/analyze/stage2/latest"),
        ("post", "/analyze/gaze-blink/live"),
        ("post", "/analyze/gaze-blink"),
    ],
)
def test_stage_routes_require_login(client, method, path):
    assert getattr(client, method)(path).status_code == 401


# ── 1단계 최신 조회 ──────────────────────────────────────────────────────

def test_stage1_latest_is_null_when_no_result(client, auth_headers):
    res = client.get("/analyze/stage1/latest", headers=auth_headers)
    assert res.status_code == 200
    assert res.get_json() == {"result": None}


def test_stage1_latest_returns_saved_entry(client, auth_headers, app_module):
    uid = _user_id(app_module)
    with app_module.app.app_context():
        app_module.db.session.add(app_module.Stage1Result(
            user_id=uid, stability=70, fluency=60, pause_ctrl=50, continuity=40, calm=30, overall_score=50,
        ))
        app_module.db.session.commit()
    result = client.get("/analyze/stage1/latest", headers=auth_headers).get_json()["result"]
    assert result["stability"] == 70
    assert result["overallScore"] == 50


# ── 2단계 실시간 결과 저장 (/live) ──────────────────────────────────────

def test_live_save_stores_result_and_returns_no_previous_first(client, auth_headers, app_module):
    res = client.post("/analyze/gaze-blink/live", json=LIVE_PAYLOAD, headers=auth_headers)
    assert res.status_code == 200
    assert res.get_json() == {"previous": None}
    with app_module.app.app_context():
        assert app_module.Stage2Result.query.count() == 1


def test_live_save_returns_previous_entry_on_second_save(client, auth_headers):
    client.post("/analyze/gaze-blink/live", json=LIVE_PAYLOAD, headers=auth_headers)
    second = {**LIVE_PAYLOAD, "blinkRatePerMin": 25.0}
    res = client.post("/analyze/gaze-blink/live", json=second, headers=auth_headers)
    previous = res.get_json()["previous"]
    assert previous["blinkRatePerMin"] == 18.0  # 방금 저장한 값이 아니라 "직전" 기록


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: {k: v for k, v in p.items() if k != "blinkScore"},   # 필드 누락
        lambda p: {**p, "blinkScore": 101},                             # 범위 초과
        lambda p: {**p, "smileRatio": -0.1},                            # 범위 미만
        lambda p: {**p, "gazeScore": True},                             # bool은 숫자로 안 받음
        lambda p: {**p, "blinkRatePerMin": float("nan")},               # NaN
        lambda p: {**p, "expressionStatus": ""},                        # 빈 상태 문자열
        lambda p: {**p, "blinkStatus": "x" * 21},                       # 컬럼 길이 초과
    ],
)
def test_live_save_rejects_invalid_payload_without_saving(client, auth_headers, app_module, mutate):
    res = client.post("/analyze/gaze-blink/live", json=mutate(LIVE_PAYLOAD), headers=auth_headers)
    assert res.status_code == 400
    assert "error" in res.get_json()
    with app_module.app.app_context():
        assert app_module.Stage2Result.query.count() == 0


def test_live_save_rejects_non_json_body(client, auth_headers):
    res = client.post("/analyze/gaze-blink/live", data="not json", headers=auth_headers)
    assert res.status_code == 400


# ── 2단계 최신 조회 ──────────────────────────────────────────────────────

def test_stage2_latest_is_null_when_no_result(client, auth_headers):
    assert client.get("/analyze/stage2/latest", headers=auth_headers).get_json() == {"result": None}


def test_stage2_latest_returns_newest_with_overall_average(client, auth_headers, app_module):
    uid = _user_id(app_module)
    base = datetime.datetime(2026, 10, 1, 12, 0, 0)
    with app_module.app.app_context():
        for i, (blink, gaze, expr) in enumerate([(10, 20, 30), (60, 90, 30)]):
            app_module.db.session.add(app_module.Stage2Result(
                user_id=uid, created_at=base + datetime.timedelta(days=i),
                blink_rate_per_min=15, blink_status="정상", blink_score=blink,
                avg_fixation_sec=3, gaze_score=gaze,
                smile_ratio=0.2, tension_ratio=0.1, expression_score=expr, expression_status="보통",
            ))
        app_module.db.session.commit()

    result = client.get("/analyze/stage2/latest", headers=auth_headers).get_json()["result"]
    assert (result["blinkScore"], result["gazeScore"], result["expressionScore"]) == (60, 90, 30)
    assert result["overallScore"] == 60.0  # (60 + 90 + 30) / 3


def test_stage2_latest_only_sees_own_results(client, auth_headers, app_module):
    """다른 사용자의 기록이 내 최신 조회에 섞이면 안 된다."""
    client.post("/api/signup", json={"email": "other@example.com", "password": "pw-67890", "name": "타인", "terms_agreed": True})
    token = client.post("/api/login", json={"email": "other@example.com", "password": "pw-67890"}).get_json()["token"]
    other_headers = {"Authorization": f"Bearer {token}"}

    client.post("/analyze/gaze-blink/live", json=LIVE_PAYLOAD, headers=other_headers)
    assert client.get("/analyze/stage2/latest", headers=auth_headers).get_json() == {"result": None}
