"""
코치 노트 안전장치 (순수 함수).

1) 위기 신호 감지: 답변에 자해·극단적 표현이 있으면 코칭이 아니라 돌봄 안내로 넘어가고 **LLM을 호출하지 않는다**
   (그 내용을 AI로 보내지도 않고, AI가 코칭 투로 응답하지도 못하게).
   **키워드 방식의 최선 노력(best-effort)일 뿐 임상적 판단이 아니다** — 못 잡는 표현이 많다. 확정된 위기 대응
   절차와 안내 기관은 팀이 정해야 한다(여기에 연락처를 임의로 넣지 않는다 → 환경변수 CARE_RESOURCES).
2) 금지어: 평가·진단처럼 들리는 표현이 AI 출력에 섞이면 그 문구를 버린다 (재활 플랫폼이라 점수·병명·"부족" 같은
   말로 사용자를 규정하면 안 됨).
"""
from __future__ import annotations

import os
import re

# 띄어쓰기 변형을 잡으려고 공백을 제거한 뒤 비교한다
_CARE_PATTERNS = (
    "죽고싶", "죽을래", "죽어버리", "자살", "자해", "살기싫", "살고싶지않", "사라지고싶", "끝내고싶", "목숨",
)

# AI 출력에서 막을 표현 — 점수·등급·진단·치료 용어, 사용자를 깎아내리는 말
_BANNED_SUBSTRINGS = (
    "점수", "등급", "순위", "부족", "실패", "못했", "못 했", "나쁘", "한심", "잘못",
    "우울", "장애", "진단", "증상", "치료", "병원", "약물", "공황", "트라우마",
)
_SCORE_PATTERN = re.compile(r"\d+\s*점")  # "45점" 같은 점수 노출


def detect_care(texts: list[str]) -> bool:
    """답변 텍스트들에서 위기 신호가 감지되면 True."""
    for t in texts:
        squeezed = re.sub(r"\s+", "", t or "")
        if any(p in squeezed for p in _CARE_PATTERNS):
            return True
    return False


def has_banned_term(text: str) -> bool:
    """AI가 쓴 문구에 평가·진단 표현이 있으면 True."""
    return bool(_SCORE_PATTERN.search(text)) or any(b in text for b in _BANNED_SUBSTRINGS)


def care_message() -> dict:
    """돌봄 안내 문구. 연락처는 팀이 확정해서 환경변수 CARE_RESOURCES에 넣는다(비어 있으면 일반 안내만)."""
    return {
        "greeting": "오늘 이야기해줘서 고마워요. 마음이 많이 힘들다면 혼자 견디지 않아도 괜찮아요.",
        "body": "가까운 사람이나 전문 상담에 도움을 요청해 보세요. 지금 당장 도움이 필요하다면 주변에 바로 알려주세요.",
        "resources": os.environ.get("CARE_RESOURCES", "").strip(),
    }
