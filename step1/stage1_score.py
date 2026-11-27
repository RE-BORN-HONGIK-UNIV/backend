"""1단계 · 종합 점수 — 5개 축 평균(프론트 features/voice/feedback.ts의 overallScore와 같은 공식)과 CNN 습관 점수를 계산한다.
공식을 바꾸면 프론트도 같이 바꾼다. app.py와 떼어 둔 건 CI 테스트에서 DB 없이 불러오기 위해서다.
"""
from __future__ import annotations

SCORE_KEYS = ("stability", "fluency", "pause_ctrl", "continuity", "calm")


def overall_score(scores: dict) -> int:
    """5개 축 점수 평균(반올림). 프론트 overallScore()와 동일 (JS Math.round: .5는 올림)."""
    avg = sum(float(scores[k]) for k in SCORE_KEYS) / len(SCORE_KEYS)
    return int(avg + 0.5)


def window_score(wins: list, threshold: float) -> float:
    """CNN 습관 점수: 기준값을 넘은 3초 조각의 비율만큼 깎는다(조각이 없으면 100). 최고 조각 하나로 매기지 않는 이유는 설계노트."""
    if not wins:
        return 100.0
    hit = sum(1 for _, p in wins if p >= threshold)
    return round(100 * (1 - hit / len(wins)), 1)
