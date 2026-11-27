"""3단계 · 불안도와 난이도 — 1·2단계 점수로 불안도(0~100, 높을수록 긴장)를 내고 면접 난이도를 정한다(사용자에게는 숫자를 보여 주지 않음).
불안도 = Σ 가중치 × (100 − 항목 점수) ÷ 값이 있는 항목의 가중치 합. 가중치는 근거가 강한 긴 멈춤·시선 2, 나머지 0.5.
가중치와 경계값 근거: docs/anxiety_근거.md (문헌은 docs/REFERENCES.md)
"""
from __future__ import annotations

WEIGHTS = {
    "pause": 2.0,       # 침묵 조절력 (1단계 pause_ctrl)
    "gaze": 2.0,        # 시선 (2단계 avg_fixation_sec → gaze_item)
    "filler": 0.5,      # 발화 유창성 (1단계 fluency)
    "prolong": 0.5,     # 발화 지속성 (1단계 continuity)
    "energy": 0.5,      # 발화 에너지 (1단계 calm)
    "tremor": 0.5,      # 음성 안정성 (1단계 stability)
    "blink": 0.5,       # 깜빡임 (2단계 blink_rate_per_min → blink_item)
    "tension": 0.5,     # 미간 긴장 (2단계 100 × (1 − tension_ratio))
}


def gaze_item(avg_fixation_sec: float) -> float:
    """시선 피함만 긴장으로 본다. 화면 점수(step2/scoring.py)는 5초 넘게 바라보면 깎지만 그건 인상 기준이라 여기선 100."""
    return 100.0 if avg_fixation_sec >= 3 else avg_fixation_sec / 3 * 70


def blink_item(rate_per_min: float) -> float:
    """잦은 깜빡임(분당 30회 초과)만 긴장으로 본다. 불안 근거는 많을 때뿐이라 적은 쪽은 깎지 않는다."""
    return max(0.0, 100 - max(0.0, rate_per_min - 30) * 5)


def worse(*scores: float | None) -> float | None:
    """같은 항목의 1·2단계 점수 중 더 긴장한 쪽(낮은 점수). 판단이 틀려도 쉬운 난이도로 가게 한다."""
    have = [s for s in scores if s is not None]
    return min(have) if have else None


def anxiety_score(items: dict) -> float | None:
    """값이 있는 항목만으로 불안도를 낸다. 값이 하나도 없으면 None."""
    have = {k: float(v) for k, v in items.items() if k in WEIGHTS and v is not None}
    if not have:
        return None
    total = sum(WEIGHTS[k] for k in have)
    return round(sum(WEIGHTS[k] * (100 - v) for k, v in have.items()) / total, 1)


def tier(score: float | None) -> str:
    """불안도 20 이상 편안(warmup) / 5~20 보통(standard) / 5 미만 도전(practice). 값이 없으면 가장 쉬운 난이도(경계값은 잠정)."""
    if score is None or score >= 20:
        return "warmup"
    return "standard" if score >= 5 else "practice"
