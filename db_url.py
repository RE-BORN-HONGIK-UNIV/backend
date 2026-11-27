"""공통 · DB 주소 정리 — DATABASE_URL의 드라이버를 psycopg2로 못 박는다(SQLAlchemy 2.1부터 기본 드라이버가 바뀌어 배포가 깨진 적이 있음).
app.py와 떼어 둔 건 CI 테스트에서 DB 없이 불러오기 위해서다. 원인 기록: docs/공통_설계노트.md
"""
from __future__ import annotations

_DRIVER = "postgresql+psycopg2://"
# Render는 "postgres://"(SQLAlchemy 1.4+가 버린 형식), 로컬은 "postgresql://"을 주므로 둘 다 psycopg2로 맞춘다
_BARE_SCHEMES = ("postgres://", "postgresql://")


def normalize_database_url(url: str) -> str:
    """드라이버 없는 postgres(ql):// 주소만 psycopg2로 바꾸고, 드라이버를 적었거나 다른 DB면 그대로 둔다."""
    for scheme in _BARE_SCHEMES:
        if url.startswith(scheme):
            return _DRIVER + url[len(scheme):]
    return url
