"""
Step 3 · 면접 질문 생성 (난이도 tier + 질문 종류 mode 기반).

- 프론트에서 tier(warmup/standard/practice), mode(main/follow_up), 지금까지 나온 질문 목록,
  방금 답변(previous_answer)을 보내면 다음 질문 1개를 생성해 돌려준다.
- mode
  - main: 새 주제의 기본 질문. 방금 답변을 파고들지 않는다.
  - follow_up: 방금 답변을 바탕으로 한 꼬리질문. tier가 높을수록 더 깊게 파고든다.
- 첫 질문(자기소개)은 프론트에서 고정으로 보내고, 이 함수는 두 번째 질문부터 사용된다.
- ANTHROPIC_API_KEY 가 없거나 호출이 실패하면 고정 질문 리스트에서 폴백한다.
  (백엔드가 죽어도 화면은 안 죽게)
"""
from __future__ import annotations

import os
import logging
import random

log = logging.getLogger(__name__)

try:
    import anthropic
    _client = anthropic.Anthropic() if os.environ.get("ANTHROPIC_API_KEY") else None
except ImportError:  # anthropic 미설치 환경에서도 서버는 떠야 함
    anthropic = None
    _client = None

# 모델은 환경변수로 교체 가능 (claude-opus-5 / claude-haiku-4-5 등)
MODEL = os.environ.get("INTERVIEW_MODEL", "claude-sonnet-5")

COMMON_RULES = """당신은 'Re-born'이라는 비대면 발화 코칭 앱의 AI 모의면접관입니다.
대상은 대면 면접에 대한 두려움으로 취업에 어려움을 겪어온 고립·은둔 청년입니다.

공통 규칙:
- 반드시 한국어로, 질문 딱 1개만 출력합니다.
- 질문 앞뒤에 설명, 인사말, 따옴표, 번호를 붙이지 않습니다. 질문 문장 하나만 출력합니다.
- 존댓말을 쓰되 딱딱한 격식체가 아니라 편안하게 대화하듯 묻습니다.
- 이미 나온 질문과 겹치지 않게 합니다. 자기소개는 첫 질문에서 이미 했으므로 다시 묻지 않습니다.
- 비꼬거나 공격적인 말투, 답변을 깎아내리는 표현은 난이도와 상관없이 절대 쓰지 않습니다."""

# 난이도별 전체 분위기 (기본 질문·꼬리질문 공통)
# tier는 1·2단계 결과로 정해짐: 긴장이 많이 보일수록 warmup, 안정적일수록 practice
TIER_INSTRUCTIONS = {
    "warmup": (
        "현재 난이도는 워밍업입니다. 아직 말하기에 긴장이 많이 보이는 단계입니다.\n"
        "취미, 관심사, 지원 동기처럼 답하기 쉬운 질문만 합니다. 절대 압박하지 않습니다.\n"
        "공백기, 은둔 경험 같은 민감한 개인사는 묻지 않습니다."
    ),
    "standard": (
        "현재 난이도는 표준입니다.\n"
        "실제 채용 면접에서 흔히 나오는 수준의 질문을 합니다.\n"
        "공백기, 은둔 경험 같은 민감한 개인사는 묻지 않습니다."
    ),
    "practice": (
        "현재 난이도는 실전입니다. 비교적 안정적으로 말할 수 있어 실전 대응력을 키울 단계입니다.\n"
        "구체적인 근거나 경험을 요구하는, 조금 더 깊이 있는 질문을 합니다.\n"
        "실제 면접처럼 공백기 질문도 할 수 있지만, 캐묻거나 탓하는 말투가 아니라 "
        "그 시간을 어떻게 보냈고 무엇을 준비했는지 설명할 기회를 주는 방식으로 묻습니다."
    ),
}

# 꼬리질문을 얼마나 깊게 파고들지 난이도별로 다르게
FOLLOW_UP_INSTRUCTIONS = {
    "warmup": (
        "답변에서 좋았던 점을 짧게 짚은 뒤, 부담 없이 조금만 더 이야기해볼 수 있는 가벼운 질문을 합니다."
    ),
    "standard": (
        "답변 중 구체적이지 않은 부분 하나를 골라, 실제 예시나 그때 상황을 조금 더 물어봅니다. "
        "압박하지 않고 자연스럽게 이어가듯 묻습니다."
    ),
    "practice": (
        "답변의 근거, 이유, 결과를 한 단계 더 파고듭니다. "
        "'열심히', '많이'처럼 모호한 표현이 있으면 구체적인 행동이나 수치를 묻고, "
        "다른 방법도 있었을 텐데 왜 그렇게 판단했는지처럼 사고 과정을 묻습니다."
    ),
}

# API 실패 시 안전망. 기본 질문은 프론트 difficulty.ts의 QUESTION_BANK와 동일
FALLBACK_QUESTIONS = {
    "warmup": [
        "간단하게 자기소개 먼저 해주시겠어요?",
        "요즘 관심 있게 보고 있는 게 있다면 편하게 말씀해주세요.",
        "오늘 이 자리에 오면서 어떤 마음이었는지 궁금해요.",
    ],
    "standard": [
        "이 직무에 지원하게 된 계기를 말씀해주시겠어요?",
        "최근에 어려운 상황을 해결했던 경험이 있다면 소개해주세요.",
        "본인의 강점을 하나 꼽는다면 무엇인가요?",
    ],
    "practice": [
        "이 직무에 본인이 적합하다고 생각하는 구체적인 근거는 무엇인가요?",
        "실패했던 경험과 그로부터 배운 점을 말씀해주세요.",
        "지금 말씀하신 강점을 실제로 발휘했던 순간을 좀 더 구체적으로 설명해주시겠어요?",
    ],
}

FALLBACK_FOLLOW_UPS = {
    "warmup": ["방금 이야기 좋았어요. 그 부분을 조금만 더 들려주실 수 있을까요?"],
    "standard": ["방금 말씀하신 내용을 실제 예시와 함께 조금 더 설명해주실 수 있을까요?"],
    "practice": ["방금 말씀하신 내용에서, 그렇게 판단한 구체적인 근거를 말씀해주시겠어요?"],
}


# ── 1·2단계 세부 점수 → 질문 방식 조절 ─────────────────────────────────
# 점수 "숫자"는 LLM에 넘기지 않고, 낮게 나온 항목에 맞는 "질문 방식 지침"만 넘긴다.
# 숫자나 약점 이름이 프롬프트에 없으면 LLM이 그걸 사용자에게 말해버릴 여지가 구조적으로 줄고
# (발화·사회불안 사용자에게 "채움말이 많으시네요" 같은 말은 상처가 될 수 있음),
# 프롬프트에도 "절대 언급하지 않는다"는 규칙을 한 번 더 넣는다.
#
# 기준: 점수는 모두 0~100, 높을수록 안정적. WEAK_BELOW 미만이면 "약한 항목"으로 보는데,
# 70은 프론트 getTier()가 "실전 난이도(안정적)"로 넘어가는 경계와 같은 값이라 의미를 맞춘 것.
# 정확도 검증으로 정한 임계값이 아니라 난이도 경계를 재사용한 휴리스틱이므로, 바꿀 땐 실제
# 사용자 질문을 보고 조정할 것. 항목이 많이 약해도 질문이 과해지지 않게 가장 낮은 2개만 반영.
WEAK_BELOW = 70
MAX_WEAK_AXES = 2

# (축 키, 상태 설명, 질문 방식 지침) — 1단계(음성) 5축, 2단계(표정·시선) 3축
_AXIS_GUIDANCE = {
    "stability": (
        "목소리가 떨리기 쉬운 편",
        "짧고 부담 없는 질문으로, 답을 천천히 시작해도 괜찮은 분위기를 줍니다.",
    ),
    "fluency": (
        "말 사이에 군더더기가 늘기 쉬운 편",
        "한 번에 한 가지만 묻고, 한두 문장으로 정리해 답할 수 있는 질문을 합니다.",
    ),
    "pause_ctrl": (
        "말 도중 길게 멈추기 쉬운 편",
        "질문을 여러 개로 쪼개지 말고 하나만 묻고, 생각할 시간이 충분하다고 느끼게 묻습니다.",
    ),
    "continuity": (
        "소리가 늘어지거나 끊기기 쉬운 편",
        "답이 구체적인 사실 하나로 모이는, 범위가 좁은 질문을 합니다.",
    ),
    "calm": (
        "말의 힘이 오르내리기 쉬운 편",
        "편안하게 이야기하듯 답할 수 있는 질문을 합니다.",
    ),
    # 표정·시선은 질문 내용으로 바꿀 수 있는 게 거의 없어서, 어투와 부담을 낮추는 정도로만 반영
    "blink": (
        "긴장이 눈 깜빡임에 드러나기 쉬운 편",
        "질문을 짧고 친근한 어투로, 압박 없이 던집니다.",
    ),
    "gaze": (
        "시선을 유지하기 어려운 편",
        "질문을 짧고 친근한 어투로, 압박 없이 던집니다.",
    ),
    "expression": (
        "표정이 굳기 쉬운 편",
        "질문을 짧고 친근한 어투로, 압박 없이 던집니다.",
    ),
}


def _weak_axes(profile: dict | None) -> list[str]:
    """profile에서 WEAK_BELOW 미만인 항목을 낮은 점수 순으로 최대 MAX_WEAK_AXES개.

    profile 형태: {"stage1": {축: 점수, ...} | None, "stage2": {축: 점수, ...} | None}
    (알 수 없는 키·숫자가 아닌 값은 무시 — 서버 DB 값이라 신뢰하지만 프롬프트가 깨지면 안 됨)
    """
    if not profile:
        return []
    scored: list[tuple[float, str]] = []
    for stage in ("stage1", "stage2"):
        for axis, v in (profile.get(stage) or {}).items():
            if axis in _AXIS_GUIDANCE and isinstance(v, (int, float)) and not isinstance(v, bool) and v < WEAK_BELOW:
                scored.append((float(v), axis))
    scored.sort(key=lambda t: t[0])
    return [axis for _, axis in scored[:MAX_WEAK_AXES]]


def _profile_section(profile: dict | None) -> str:
    """프롬프트에 붙일 "질문 방식 조절" 섹션. 약한 항목이 없으면 빈 문자열."""
    axes = _weak_axes(profile)
    if not axes:
        return ""
    lines = "\n".join(f"- {_AXIS_GUIDANCE[a][0]}: {_AXIS_GUIDANCE[a][1]}" for a in axes)
    return (
        "[질문 방식 참고 — 이 사용자가 긴장을 보이기 쉬운 부분]\n"
        f"{lines}\n"
        "위 내용은 질문의 방식과 부담을 조절하는 데만 참고합니다. "
        "분석 결과, 점수, 약한 부분을 질문이나 말투에서 절대 언급하지 않습니다.\n\n"
    )


# ── 지난 면접과 겹치지 않게 ─────────────────────────────────────────────
# 같은 사람이 여러 번 연습해도 매번 새로운 질문을 받도록, 지난 면접에서 했던 "기본 질문"을 프롬프트에
# 넣어 피하게 한다. 꼬리질문은 그때그때 답변에 따라 달라지는 질문이라 대상에서 뺀다(비슷한 꼬리질문이
# 다시 나오는 건 자연스럽고, 막으면 오히려 답변을 못 파고든다). 답변 내용은 넘기지 않고 질문 문장만 넘긴다.
# 개수는 프롬프트가 길어지지 않게 최근 것만 — 너무 많이 넘기면 LLM이 피할 주제를 못 찾아 질문이 어색해진다.
MAX_PAST_QUESTIONS = 15


def _past_to_avoid(past_questions: list[str] | None, previous_questions: list[str]) -> list[str]:
    """지난 면접 질문 중 이번 프롬프트에 넣을 것: 이번 면접에서 이미 나온 것(별도 섹션으로 들어감)과
    자기소개(첫 질문에서 이미 했고 공통 규칙으로 금지)는 빼고, 중복을 없애 최근 순으로 최대 MAX개."""
    seen = set(previous_questions)
    out: list[str] = []
    for q in past_questions or []:
        if q in seen or "자기소개" in q:
            continue
        seen.add(q)
        out.append(q)
        if len(out) >= MAX_PAST_QUESTIONS:
            break
    return out


def _past_section(past_questions: list[str] | None, previous_questions: list[str], mode: str) -> str:
    """프롬프트에 붙일 "지난 면접 질문" 섹션. 기본 질문 차례가 아니거나 피할 질문이 없으면 빈 문자열."""
    if mode != "main":
        return ""
    avoid = _past_to_avoid(past_questions, previous_questions)
    if not avoid:
        return ""
    lines = "\n".join(f"- {q}" for q in avoid)
    return (
        "[지난 면접에서 이미 했던 질문 — 이번에는 이와 겹치지 않는 새로운 주제로 질문합니다]\n"
        f"{lines}\n\n"
    )


def _fallback_question(
    tier: str, previous_questions: list[str], mode: str, past_questions: list[str] | None = None
) -> str:
    if mode == "follow_up":
        return random.choice(FALLBACK_FOLLOW_UPS.get(tier, FALLBACK_FOLLOW_UPS["standard"]))

    pool = FALLBACK_QUESTIONS.get(tier, FALLBACK_QUESTIONS["standard"])
    remaining = [q for q in pool if q not in previous_questions]
    # 첫 질문에서 자기소개를 이미 했으면 자기소개 질문은 제외 (문장이 달라서 위 비교로는 안 걸림)
    if any("자기소개" in q for q in previous_questions):
        remaining = [q for q in remaining if "자기소개" not in q]
    if not remaining:
        remaining = pool
    # 지난 면접에서 나온 적 없는 질문을 우선한다. 다 나왔던 거라면(고정 질문은 난이도당 3개뿐) 그냥 고른다.
    fresh = [q for q in remaining if q not in (past_questions or [])]
    return random.choice(fresh or remaining)


def _user_prompt(
    tier: str,
    previous_questions: list[str],
    previous_answer: str | None,
    mode: str,
    profile: dict | None = None,
    past_questions: list[str] | None = None,
) -> str:
    instr = TIER_INSTRUCTIONS.get(tier, TIER_INSTRUCTIONS["standard"])
    asked = "\n".join(f"- {q}" for q in previous_questions) if previous_questions else "(아직 없음)"

    if mode == "follow_up":
        follow = FOLLOW_UP_INSTRUCTIONS.get(tier, FOLLOW_UP_INSTRUCTIONS["standard"])
        task = (
            f"[방금 사용자 답변]\n{previous_answer}\n\n"
            "지금은 꼬리질문 차례입니다. 방금 답변 내용을 바탕으로 이어지는 질문을 만들어주세요.\n"
            f"{follow}\n"
            "답변 속 표현을 한 구절 자연스럽게 짚으며 시작하면 좋습니다 (예: '말씀하신 ~에서').\n"
            "답변이 아주 짧거나 내용이 거의 없으면, 부담 없이 조금 더 이야기해달라고 부드럽게 요청합니다."
        )
    else:
        task = (
            "지금은 새로운 기본 질문 차례입니다. "
            "방금 답변을 더 파고들지 말고, 아직 다루지 않은 새로운 주제로 질문해주세요."
        )

    return (
        f"{instr}\n\n"
        f"[이미 나온 질문]\n{asked}\n\n"
        f"{_profile_section(profile)}"
        f"{_past_section(past_questions, previous_questions, mode)}"
        f"{task}\n\n"
        "다음 질문을 1개만 생성해주세요."
    )


def generate_question(
    tier: str,
    previous_questions: list[str] | None = None,
    previous_answer: str | None = None,
    mode: str = "main",
    timeout: float = 15.0,
    profile: dict | None = None,
    past_questions: list[str] | None = None,
) -> tuple[str, str]:
    """(질문, source) 반환. source는 'llm' 또는 'fallback'.
    profile: 1·2단계 세부 점수(_weak_axes 참고) — 약한 항목에 맞춰 질문 방식만 조절한다.
    past_questions: 지난 면접의 기본 질문(최근 순) — 겹치지 않는 새 주제로 묻게 한다. 폴백 경로에서는
    고정 질문 중 지난번에 안 나온 것을 우선 고른다."""
    previous_questions = previous_questions or []

    # 꼬리질문인데 답변 텍스트가 없으면(인식 실패 등) 파고들 내용이 없으므로 기본 꼬리질문으로
    if mode == "follow_up" and not previous_answer:
        return _fallback_question(tier, previous_questions, mode, past_questions), "fallback"

    if _client is None:
        return _fallback_question(tier, previous_questions, mode, past_questions), "fallback"

    try:
        resp = _client.with_options(timeout=timeout).messages.create(
            model=MODEL,
            max_tokens=200,
            output_config={"effort": "low"},  # 질문 하나 — 저비용/저지연
            system=[{
                "type": "text",
                "text": COMMON_RULES,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{
                "role": "user",
                "content": _user_prompt(tier, previous_questions, previous_answer, mode, profile, past_questions),
            }],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if text:
            return text, "llm"
        return _fallback_question(tier, previous_questions, mode, past_questions), "fallback"
    except Exception as e:  # 네트워크/키/쿼터 등 — 화면 흐름은 절대 안 깨지게
        log.warning("Step3 질문 생성 실패: %s", e)
        return _fallback_question(tier, previous_questions, mode, past_questions), "fallback"