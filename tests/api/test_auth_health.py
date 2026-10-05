from tests.api.conftest import SIGNUP_BODY


def test_health_reports_models_not_loaded_without_heavy_deps(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] == "ok"
    # 모델은 첫 분석 요청 때 지연 로드 — /health만 쓰는 워커는 torch/whisper를 안 올린다
    assert body["cnn_model_loaded"] is False
    assert body["whisper_loaded"] is False


def test_signup_creates_user(client):
    res = client.post("/api/signup", json=SIGNUP_BODY)
    assert res.status_code == 201


def test_signup_requires_core_fields(client):
    res = client.post("/api/signup", json={"email": "a@b.c", "password": "x", "terms_agreed": True})
    assert res.status_code == 400


def test_signup_requires_terms_agreement(client):
    res = client.post("/api/signup", json={**SIGNUP_BODY, "terms_agreed": False})
    assert res.status_code == 400


def test_signup_duplicate_email_conflicts(client):
    assert client.post("/api/signup", json=SIGNUP_BODY).status_code == 201
    assert client.post("/api/signup", json=SIGNUP_BODY).status_code == 409


def test_login_returns_token_that_authenticates(client, auth_headers):
    res = client.get("/analyze/stage2/latest", headers=auth_headers)
    assert res.status_code == 200


def test_login_wrong_password_rejected(client):
    client.post("/api/signup", json=SIGNUP_BODY)
    res = client.post("/api/login", json={"email": SIGNUP_BODY["email"], "password": "wrong"})
    assert res.status_code == 401


def test_invalid_token_is_treated_as_logged_out(client):
    res = client.get("/analyze/stage2/latest", headers={"Authorization": "Bearer not-a-jwt"})
    assert res.status_code == 401
