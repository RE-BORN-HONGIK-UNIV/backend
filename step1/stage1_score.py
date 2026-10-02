"""
Step 1 · 종합 점수 계산 (순수 함수).

프론트 `features/voice/feedback.ts`의 `overallScore`와 같은 공식 — 5개 축 점수의
산술 평균을 반올림한 값. DB에 쌓는 1단계 종합 점수가 화면에 보이는 점수와
어긋나지 않게, 공식을 바꾸려면 양쪽을 같이 바꿀 것.

app.py에 두지 않은 이유: app.py는 import 시점에 DB·matplotlib 등을 요구해서
CI의 순수 로직 테스트에서 불러올 수 없다.
"""
from __future__ import annotations

SCORE_KEYS = ("stability", "fluency", "pause_ctrl", "continuity", "calm")


def overall_score(scores: dict) -> int:
    """5개 축 점수 평균(반올림). 프론트 overallScore()와 동일 (JS Math.round: .5는 올림)."""
    avg = sum(float(scores[k]) for k in SCORE_KEYS) / len(SCORE_KEYS)
    return int(avg + 0.5)
