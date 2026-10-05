"""API 통합 테스트(3층) 공용 픽스처 — Flask test client + 인메모리 SQLite.

app.py는 torch/whisper/mediapipe를 라우트 안에서 지연 import하도록 바뀌어서(docs/TESTING.md 3층),
`/health`·인증·`/analyze/stage*/latest` 같은 가벼운 라우트는 무거운 모델 없이 테스트할 수 있다.
영상 분석 라우트(/analyze/gaze-blink)는 mediapipe가 필요한 landmark 추출(extract_landmarks_from_video)만
가짜로 바꿔 끼우고 나머지(품질 게이트·점수 계산·DB 저장·응답)는 실제 코드를 탄다.

의존성(flask 등)이 없는 환경(1층 CI job, 로컬 최소 설치)에선 이 디렉터리 전체가 skip된다.
CI에서는 별도 job(`test-api`)이 필요한 패키지를 설치하고 실행한다 — 순수 로직 job과 분리한 이유는
librosa·matplotlib 등 app.py가 최상단에서 import하는 패키지 설치가 무겁기 때문.
"""
import os
import sys

import pytest

for _mod in ("flask", "flask_sqlalchemy", "flask_cors", "jwt", "librosa", "matplotlib", "PIL", "dotenv"):
    pytest.importorskip(_mod)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# app.py는 import 시점에 DB/SECRET_KEY를 읽는다 — 개발자 로컬 .env의 실제 DB에 붙지 않도록
# (load_dotenv는 이미 설정된 환경변수를 덮어쓰지 않음) import 전에 테스트용 값으로 고정한다.
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["SECRET_KEY"] = "test-secret-key-for-api-tests-0123456789"  # HS256 권장 길이(32바이트) 이상


@pytest.fixture(scope="session")
def app_module():
    import app as app_module

    return app_module


@pytest.fixture()
def client(app_module):
    return app_module.app.test_client()


@pytest.fixture(autouse=True)
def clean_db(app_module):
    """테스트마다 모든 테이블을 비운다 (인메모리 SQLite는 세션 내내 공유됨)."""
    yield
    db = app_module.db
    with app_module.app.app_context():
        db.session.remove()
        for table in reversed(db.metadata.sorted_tables):
            db.session.execute(table.delete())
        db.session.commit()


SIGNUP_BODY = {
    "email": "tester@example.com",
    "password": "pw-12345",
    "name": "테스터",
    "terms_agreed": True,
}


@pytest.fixture()
def auth_headers(client):
    """실제 /api/signup → /api/login을 거쳐 얻은 Bearer 헤더 (JWT 발급·검증 경로까지 함께 탄다)."""
    assert client.post("/api/signup", json=SIGNUP_BODY).status_code == 201
    res = client.post("/api/login", json={"email": SIGNUP_BODY["email"], "password": SIGNUP_BODY["password"]})
    return {"Authorization": f"Bearer {res.get_json()['token']}"}
