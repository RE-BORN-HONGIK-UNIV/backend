"""
Step 1 · LLM 기반 발화 코칭 피드백 생성.

- /analyze 결과(점수 dict)를 받아 공감형 코칭 문구 한 문단을 생성한다.
- ANTHROPIC_API_KEY 가 없거나 호출이 실패하면 None 을 반환한다.
  → 호출측(app.py /analyze/feedback)이 기존 템플릿 문구로 폴백한다.
- 오디오/이름/전사문은 넘기지 않는다. 점수와 집계 숫자만 사용.
"""
from __future__ import annotations

import os
import logging

log = logging.getLogger(__name__)

try:
    import anthropic
    _client = anthropic.Anthropic() if os.environ.get("ANTHROPIC_API_KEY") else None
except ImportError:  # anthropic 미설치 환경에서도 서버는 떠야 함
    anthropic = None
    _client = None

# 모델은 환경변수로 교체 가능 (claude-sonnet-5 / claude-haiku-4-5 등)
MODEL = os.environ.get("FEEDBACK_MODEL", "claude-opus-5")

AXIS_LABELS = {
    "stability": "음성 안정성",
    "fluency": "발화 유창성",
    "pause_ctrl": "침묵 조절력",
    "continuity": "발화 지속성",
    "calm": "발화 에너지",
}

SYSTEM_PROMPT = """당신은 고립·은둔 청년의 발화 재활을 돕는 '보이스 터치' 코치입니다.

말투 규칙:
- 따뜻하고 차분하게, 사용자를 판단하지 않습니다.
- "못했다", "부족하다" 같은 표현 대신 "이 부분을 함께 연습해봐요" 식으로 말합니다.
- 존댓말, 부드러운 구어체. 이모지·마크다운·목록·제목은 쓰지 않습니다.
- 지표 이름(음성 안정성 등)은 써도 되지만 점수 숫자를 나열하지 않습니다.

출력: 3~4문장, 120~200자 한 문단.
순서는 잘한 점 1가지 → 함께 연습할 점 1가지 → 다음 훈련 한 줄 제안."""


def _user_prompt(result: dict) -> str:
    scores = result.get("scores", {})
    fd = result.get("filler_detail", {})
    pd = result.get("pause_detail", {})

    score_lines = "\n".join(
        f"- {AXIS_LABELS.get(k, k)}: {v}점" for k, v in scores.items()
    )
    detail = (
        f"채움말 {fd.get('filler_count', 0)}회 / 소리구간 {fd.get('sound_segment_count', 0)}개, "
        f"1.2초 이상 불안한 멈춤 {pd.get('anxious_pause_count', 0)}회, "
        f"발화 길이 {pd.get('total_duration_sec', 0)}초"
    )
    note = ""
    if result.get("demo_mode"):
        note = (
            "\n참고: 음성 모델 파일이 없어 음성 안정성·발화 지속성·발화 에너지 점수는 "
            "정확하지 않습니다. 이 세 지표는 단정하지 말고 참고로만 언급하세요."
        )

    return (
        "이번 발화 분석 결과입니다.\n\n"
        f"[오각형 점수]\n{score_lines}\n\n"
        f"[상세]\n{detail}{note}\n\n"
        "이 결과를 바탕으로 코칭 피드백을 한 문단 작성해주세요."
    )


def generate_feedback(result: dict, timeout: float = 20.0) -> str | None:
    """점수 dict → 코칭 문단. 실패 시 None."""
    if _client is None:
        return None
    try:
        resp = _client.with_options(timeout=timeout).messages.create(
            model=MODEL,
            max_tokens=800,
            output_config={"effort": "low"},  # 짧은 생성 — 저비용/저지연
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # 시스템 프롬프트 캐시
            }],
            messages=[{"role": "user", "content": _user_prompt(result)}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return text or None
    except Exception as e:  # 네트워크/키/쿼터 등 — 분석 흐름은 절대 안 깨지게
        log.warning("Step1 LLM 피드백 생성 실패: %s", e)
        return None
