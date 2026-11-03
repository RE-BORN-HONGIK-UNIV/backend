"""로그인 없이 시작하기(POST /api/guest) — 게스트 토큰으로 로그인이 필요한 API를 그대로 쓸 수 있는지 본다.

app.py는 flask·librosa를 모듈 최상단에서 import해서 CI의 가벼운 테스트 목록(numpy·opencv만 설치)에서는
건너뛴다. DB는 임시 sqlite를 쓴다.
"""
import os
import sys

import pytest

pytest.importorskip("flask")
pytest.importorskip("flask_sqlalchemy")
pytest.importorskip("librosa")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def app_module(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "guest.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["SECRET_KEY"] = "test-secret"
    import app as app_module
    return app_module


@pytest.fixture
def client(app_module):
    app_module._guest_limiter._hits.clear()
    return app_module.app.test_client()


def test_guest_gets_token_that_works_on_login_required_api(client):
    res = client.post("/api/guest")
    assert res.status_code == 201
    body = res.get_json()
    assert body["guest"] is True and body["name"] == "게스트" and body["token"]

    auth = {"Authorization": f"Bearer {body['token']}"}
    # 로그인이 필요한 API — 토큰 없이는 401, 게스트 토큰이면 통과
    assert client.get("/analyze/stage1/latest").status_code == 401
    assert client.get("/analyze/stage1/latest", headers=auth).status_code == 200


def test_each_guest_is_a_separate_account_no_one_can_log_in_to(client, app_module):
    a = client.post("/api/guest").get_json()["token"]
    b = client.post("/api/guest").get_json()["token"]
    assert a != b
    with app_module.app.app_context():
        guests = app_module.User.query.filter(
            app_module.User.email.like(f"%@{app_module.GUEST_EMAIL_DOMAIN}")
        ).all()
    assert len({g.email for g in guests}) == len(guests) >= 2
    # 비밀번호는 무작위라 이메일을 알아도 로그인할 수 없다
    res = client.post("/api/login", json={"email": guests[0].email, "password": ""})
    assert res.status_code == 401


def test_guest_creation_is_rate_limited_per_ip(client, app_module):
    headers = {"X-Forwarded-For": "203.0.113.7, 10.0.0.1"}
    for _ in range(app_module._guest_limiter.limit):
        assert client.post("/api/guest", headers=headers).status_code == 201
    assert client.post("/api/guest", headers=headers).status_code == 429
    # 다른 IP는 영향 없음
    assert client.post("/api/guest", headers={"X-Forwarded-For": "198.51.100.2"}).status_code == 201
