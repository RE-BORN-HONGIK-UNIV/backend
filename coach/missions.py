"""코치 · 일상 미션 — 면접 밖에서 해 볼 아주 작은 행동 목록(노출 연습 단계). AI는 새로 만들지 않고 여기서 고르기만 한다.
초안이라 전문가 검토 전에는 확정하지 않는다.
"""
from __future__ import annotations

# 단계가 올라갈수록 조금 더 사람과 가까운 행동. 각 단계 안의 미션은 서로 비슷한 난이도.
MISSIONS: dict[int, tuple[str, ...]] = {
    1: (
        "오늘 거울 앞에서 소리 내어 인사 한마디 해보기",
        "좋아하는 글 한 문장을 소리 내어 읽어보기",
        "내 이름과 좋아하는 것 하나를 소리 내어 말해보기",
    ),
    2: (
        "소리 내어 1분 동안 자기소개를 해보기 (녹음해서 들어봐도 좋아요)",
        "가게에서 '감사합니다' 한마디 해보기",
        "오늘 하루 있었던 일을 소리 내어 두 문장으로 말해보기",
    ),
    3: (
        "가까운 사람에게 오늘 연습한 이야기 한 가지를 말해보기",
        "질문 하나를 정해서 3분 동안 소리 내어 답해보기",
        "전화나 메시지로 짧은 안부를 먼저 건네보기",
    ),
}


def mission_level(completed_count: int) -> int:
    """지금까지 끝까지 마친 면접 횟수로 단계를 정한다: 0~1회 → 1단계, 2~4회 → 2단계, 5회 이상 → 3단계."""
    if completed_count >= 5:
        return 3
    if completed_count >= 2:
        return 2
    return 1


def pick_mission(completed_count: int, seed: int) -> str:
    """단계에 맞는 미션 하나. seed(면접 id)로 돌려 쓰므로 같은 면접엔 항상 같은 미션이 나온다(재조회해도 안 바뀜)."""
    pool = MISSIONS[mission_level(completed_count)]
    return pool[seed % len(pool)]
