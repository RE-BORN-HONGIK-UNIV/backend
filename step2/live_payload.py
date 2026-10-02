"""
Step 2 · 실시간 촬영 결과 요약의 서버 검증 (순수 함수).

실시간 모드는 영상이 서버로 오지 않고 브라우저가 계산한 숫자 요약만 온다
(프라이버시 + 서버 메모리 부담 회피). 그 숫자를 Stage2Result에 그대로 저장하기 전에
형식·범위만 검증한다 — 점수를 서버에서 재계산하지 않는 이유는 원본 프레임/세그먼트가
없어서이고, 값은 본인 기록용이라 조작해도 본인 이력에만 영향을 준다.

app.py에 두지 않은 이유: app.py는 import 시 DB/matplotlib 등이 필요해서 CI의 순수
로직 테스트에서 불러올 수 없다.
"""
from __future__ import annotations

import math

# (키, 최솟값, 최댓값) — 범위는 "물리적으로 말이 되는" 느슨한 상한 (임계값 판정이 아님)
_NUMERIC_FIELDS = (
    ("blinkRatePerMin", 0, 300),
    ("blinkScore", 0, 100),
    ("avgFixationSec", 0, 3600),
    ("gazeScore", 0, 100),
    ("smileRatio", 0, 1),
    ("tensionRatio", 0, 1),
    ("expressionScore", 0, 100),
)
_STATUS_FIELDS = ("blinkStatus", "expressionStatus")
_STATUS_MAX_LEN = 20  # Stage2Result.*_status 컬럼 길이


def parse_live_result(data) -> dict:
    """요청 JSON → Stage2Result 컬럼 dict. 잘못된 입력이면 ValueError(사용자용 메시지)."""
    if not isinstance(data, dict):
        raise ValueError("요청 본문이 올바르지 않습니다")

    out = {}
    for key, lo, hi in _NUMERIC_FIELDS:
        v = data.get(key)
        # bool은 int의 하위 타입이라 따로 걸러야 한다 (True가 1.0으로 저장되는 것 방지)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{key}는 숫자여야 합니다")
        if not math.isfinite(v) or not (lo <= v <= hi):
            raise ValueError(f"{key}는 {lo}~{hi} 범위여야 합니다")
        out[key] = float(v)

    for key in _STATUS_FIELDS:
        v = data.get(key)
        if not isinstance(v, str) or not v.strip() or len(v) > _STATUS_MAX_LEN:
            raise ValueError(f"{key}는 1~{_STATUS_MAX_LEN}자 문자열이어야 합니다")
        out[key] = v.strip()

    return out
