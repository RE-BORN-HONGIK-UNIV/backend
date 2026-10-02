import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_url import normalize_database_url

REST = "user:pass@host:5432/db?sslmode=require"


@pytest.mark.parametrize("scheme", ["postgres://", "postgresql://"])
def test_bare_postgres_schemes_get_psycopg2_driver(scheme):
    # SQLAlchemy 2.1부터 드라이버 없는 postgresql://의 기본이 psycopg(v3)라서, 명시하지 않으면 배포가 죽는다
    assert normalize_database_url(scheme + REST) == "postgresql+psycopg2://" + REST


@pytest.mark.parametrize("url", [
    "postgresql+psycopg2://" + REST,
    "postgresql+psycopg://" + REST,   # 이미 드라이버를 명시한 건 존중
    "sqlite:///test.db",
    "mysql+pymysql://u:p@h/db",
])
def test_explicit_driver_or_other_db_is_left_alone(url):
    assert normalize_database_url(url) == url


def test_only_the_scheme_is_rewritten_not_the_rest_of_the_url():
    # 비밀번호나 경로에 "postgres://" 비슷한 문자열이 있어도 앞의 스킴만 바뀐다
    url = "postgres://u:postgres://x@h/postgresql://db"
    assert normalize_database_url(url) == "postgresql+psycopg2://u:postgres://x@h/postgresql://db"
