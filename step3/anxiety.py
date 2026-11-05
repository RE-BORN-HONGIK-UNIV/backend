"""
Step 3 · 불안도와 실전 면접 난이도 (순수 함수).

불안도 = Σ 가중치 × (100 − 항목 점수) ÷ (값이 있는 항목의 가중치 합). 0~100, 높을수록 긴장.
항목 점수는 1·2단계가 이미 저장한 0~100 점수(높을수록 안정)이고, 사용자에게는 보여 주지 않는다.
가중치는 불안·면접 인상 연구의 근거 강도로 정한 초기값: 근거가 비교적 강한 긴 멈춤·시선 2,
나머지 0.5. 미소는 연구 결과가 엇갈려 뺀다. 떨림은 시험을 통과(PR-AUC 0.77)해 0.5로 넣는다.
"""
from __future__ import annotations

WEIGHTS = {
    "pause": 2.0,       # 긴 멈춤 (1단계 pause_ctrl)
    "gaze": 2.0,        # 시선 (2단계 avg_fixation_sec → gaze_item)
    "filler": 0.5,      # 채움말 (1단계 fluency)
    "prolong": 0.5,     # 말끝 늘이기 (1단계 continuity)
    "energy": 0.5,      # 끝까지 또렷하게 (1단계 calm)
    "tremor": 0.5,      # 떨림 (1단계 stability)
    "blink": 0.5,       # 깜빡임 (2단계 blink_rate_per_min → blink_item)
    "tension": 0.5,     # 미간 긴장 (2단계 100 × (1 − tension_ratio))
}


def gaze_item(avg_fixation_sec: float) -> float:
    """시선 피함만 긴장으로 본다. 화면 점수(step2/scoring.py)는 5초 넘게 바라보면 깎지만 그건 인상 기준이라 여기선 100."""
    return 100.0 if avg_fixation_sec >= 3 else avg_fixation_sec / 3 * 70


def blink_item(rate_per_min: float) -> float:
    """잦은 깜빡임(분당 30회 초과)만 긴장으로 본다. 불안 근거는 많을 때뿐이라 적은 쪽은 깎지 않는다."""
    return max(0.0, 100 - max(0.0, rate_per_min - 30) * 5)


def anxiety_score(items: dict) -> float | None:
    """값이 있는 항목만으로 불안도를 낸다. 값이 하나도 없으면 None."""
    have = {k: float(v) for k, v in items.items() if k in WEIGHTS and v is not None}
    if not have:
        return None
    total = sum(WEIGHTS[k] for k in have)
    return round(sum(WEIGHTS[k] * (100 - v) for k, v in have.items()) / total, 1)


def tier(score: float | None) -> str:
    """불안도 60 이상 편안(warmup) / 30~60 보통(standard) / 30 미만 도전(practice). 값이 없으면 가장 쉬운 난이도."""
    if score is None or score >= 60:
        return "warmup"
    return "standard" if score >= 30 else "practice"
