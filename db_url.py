"""
DATABASE_URL 정규화 (순수 함수).

SQLAlchemy는 드라이버를 안 적은 `postgresql://`에 "기본 드라이버"를 붙이는데, 2.1부터 그 기본값이
psycopg2에서 psycopg(v3)로 바뀌었다. 이 프로젝트는 psycopg2-binary만 설치하므로(requirements.txt),
requirements.txt가 SQLAlchemy 버전을 고정하지 않은 채 배포하면 어느 날부터 빌드가
`ModuleNotFoundError: No module named 'psycopg'`로 죽는다(2026-10 Render 배포 실패 원인).
SQLAlchemy 버전에 기본값을 맡기지 않고 드라이버를 psycopg2로 못 박는다.

app.py에 두지 않은 이유: app.py는 import 시 DB·matplotlib 등이 필요해서 CI의 순수 로직
테스트에서 불러올 수 없다.
"""
from __future__ import annotations

_DRIVER = "postgresql+psycopg2://"
# Render가 주는 connectionString은 "postgres://"(SQLAlchemy 1.4+는 이 스킴을 버림),
# 로컬/수동 입력은 드라이버 없는 "postgresql://" — 둘 다 psycopg2로 통일.
_BARE_SCHEMES = ("postgres://", "postgresql://")


def normalize_database_url(url: str) -> str:
    """드라이버가 없는 postgres(ql):// URL만 psycopg2 드라이버로 바꾼다.
    이미 `+드라이버`를 명시했거나(postgresql+psycopg2://, postgresql+psycopg:// 등) 다른 DB(sqlite 등)면 그대로 둔다."""
    for scheme in _BARE_SCHEMES:
        if url.startswith(scheme):
            return _DRIVER + url[len(scheme):]
    return url
