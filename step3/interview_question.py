"""3단계 · 면접 질문 생성 — 난이도(tier)와 질문 종류(기본 main / 꼬리 follow_up)에 맞춰 다음 질문 하나를 만든다.
첫 질문(자기소개)은 프론트가 고정으로 내고, AI 키가 없거나 실패하면 미리 녹음해 둔 고정 질문으로 대신한다.
프롬프트 설계 배경: docs/step3_설계노트.md
"""
from __future__ import annotations

import os
import logging
import random

from llm_util import effort_options
from coach.safety import detect_care

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

# 난이도별 분위기(기본·꼬리질문 공통) — 긴장이 많이 보일수록 warmup, 안정적일수록 practice
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

# 꼬리질문을 얼마나 깊게 파고들지 난이도별로
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

# AI 실패 시 쓰는 고정 질문 — 미리 녹음한 면접관 mp3와 글자까지 같아야 그 목소리로 나온다(프론트 prerecorded.ts)
FALLBACK_QUESTIONS = {
    "warmup": [
        "요즘 관심 있게 보고 있는 게 있다면 편하게 말씀해 주세요.",
        "오늘 이 자리에 오면서 어떤 마음이었는지 궁금해요.",
        "쉬는 날에는 주로 어떻게 시간을 보내세요?",
        "최근에 작게라도 해낸 일이 있다면 들려주시겠어요?",
        "스스로 생각하기에 본인은 어떤 사람인가요?",
        "일을 한다면 어떤 분위기의 곳에서 일하고 싶으세요?",
    ],
    "standard": [
        "하고 싶은 일에 관심을 갖게 된 계기를 말씀해 주시겠어요?",
        "최근에 어려운 상황을 해결했던 경험이 있다면 소개해 주세요.",
        "본인의 강점을 하나 꼽는다면 무엇인가요?",
        "다른 사람과 함께 무언가를 해 본 경험이 있다면 말씀해 주세요.",
        "일할 때 가장 중요하다고 생각하는 것은 무엇인가요?",
        "앞으로 어떤 모습으로 일하고 싶으신가요?",
    ],
    "practice": [
        "하고 싶은 일에 본인이 잘 맞는다고 생각하는 구체적인 근거는 무엇인가요?",
        "실패했던 경험과 그로부터 배운 점을 말씀해 주세요.",
        "의견이 다른 사람과 일을 맞춰 가야 했던 경험이 있나요?",
        "본인의 약점은 무엇이고, 어떻게 보완하고 있나요?",
        "해야 할 일이 한꺼번에 몰리면 어떻게 순서를 정하시나요?",
        "일을 시작한다면 처음 석 달 동안 무엇을 해 보고 싶으신가요?",
    ],
}

# 어떤 대답 뒤에도 어울리는 고정 꼬리질문 — 한 면접 안에서 겹치지 않게 순서대로 쓴다
FALLBACK_FOLLOW_UPS = {
    "warmup": [
        "방금 이야기 좋았어요. 그 부분을 조금만 더 들려주실 수 있을까요?",
        "그때 어떤 기분이 들었는지 궁금해요.",
        "그 이야기에서 가장 기억에 남는 게 있다면 말씀해 주세요.",
        "천천히 하셔도 돼요. 예를 하나만 들어 주시겠어요?",
    ],
    "standard": [
        "방금 말씀하신 내용을 실제 예시와 함께 조금 더 설명해 주실 수 있을까요?",
        "그때 본인은 구체적으로 어떤 역할을 하셨나요?",
        "그렇게 생각하게 된 이유를 조금 더 말씀해 주시겠어요?",
        "그 경험이 지금의 본인에게 어떤 도움이 됐나요?",
    ],
    "practice": [
        "방금 말씀하신 내용에서, 그렇게 판단한 구체적인 근거를 말씀해 주시겠어요?",
        "다시 그 상황이 온다면 무엇을 다르게 하시겠어요?",
        "그 결과를 스스로는 어떻게 평가하시나요?",
        "그 경험을 앞으로 일할 때 어떻게 살릴 수 있을까요?",
    ],
}

# 자기소개 바로 뒤의 고정 꼬리질문 — 위 목록은 경험 이야기를 전제로 해서("그때 어떤 역할을…") 자기소개 뒤에는 뜬금없다
INTRO_FOLLOW_UPS = {
    "warmup": ["방금 이야기 좋았어요. 그 부분을 조금만 더 들려주실 수 있을까요?"],
    "standard": ["방금 이야기 좋았어요. 그 부분을 조금만 더 들려주실 수 있을까요?"],
    "practice": [
        "방금 말씀하신 내용에서, 그렇게 판단한 구체적인 근거를 말씀해 주시겠어요?",
        "그 경험을 앞으로 일할 때 어떻게 살릴 수 있을까요?",
    ],
}


def _in_intro_block(tier: str, previous_questions: list[str]) -> bool:
    """지금까지가 첫 질문(자기소개)과 그 꼬리질문뿐인지 — 첫 질문은 프론트가 고정으로 낸다."""
    if not previous_questions or not any(k in previous_questions[0] for k in ("자기소개", "본인 소개")):
        return False
    intro = INTRO_FOLLOW_UPS.get(tier, INTRO_FOLLOW_UPS["standard"])
    return all(q in intro for q in previous_questions[1:])


# ── 1·2단계 세부 점수 → 질문 방식 조절 ─────────────────────────────────
# 점수 숫자는 AI에 넘기지 않고, 낮게 나온 항목에 맞는 "질문 방식 지침"만 넘긴다(이유: 설계노트)
# 70 미만을 약한 항목으로 보고(잠정 기준), 질문이 과해지지 않게 가장 낮은 2개만 반영한다
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
    # 표정·시선은 질문 내용으로 바꿀 게 거의 없어 어투와 부담만 낮춘다
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
    """{"stage1": {축: 점수}, "stage2": {...}}에서 70 미만 항목을 낮은 순으로 최대 2개 돌려준다(이상한 값은 무시)."""
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
    """프롬프트에 붙일 "질문 방식 조절" 부분을 만든다. 약한 항목이 없으면 빈 문자열."""
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
# 여러 번 연습해도 새 질문을 받도록 지난 면접의 기본 질문(꼬리질문·답변은 빼고)을 최근 것만 넘긴다
MAX_PAST_QUESTIONS = 15


def _past_to_avoid(past_questions: list[str] | None, previous_questions: list[str]) -> list[str]:
    """지난 면접 질문 중 이번에 이미 나온 것과 자기소개를 빼고, 중복 없이 최근 순으로 최대 MAX_PAST_QUESTIONS개."""
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
    """프롬프트에 붙일 "지난 면접 질문" 부분을 만든다. 기본 질문 차례가 아니거나 피할 질문이 없으면 빈 문자열."""
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


# ── 하루와 나눈 이야기 ──────────────────────────────────────────────────
# 1·2단계에서 확인받은 자기 이야기를 기본 질문의 소재로만 쓴다(위기 신호가 있으면 보내지 않음, coach/safety.py)
MAX_ABOUT_ITEMS = 8
MAX_ABOUT_LEN = 80


def clean_about_me(about_me) -> list[str]:
    """요청의 about_me(["항목: 내용", ...])를 정리한다. 형식이 틀리거나 위기 신호가 있으면 빈 목록."""
    if not isinstance(about_me, list):
        return []
    out = [" ".join(str(x).split())[:MAX_ABOUT_LEN] for x in about_me if isinstance(x, str) and x.strip()]
    out = out[:MAX_ABOUT_ITEMS]
    return [] if detect_care(out) else out


def _about_section(about_me: list[str] | None, mode: str) -> str:
    """프롬프트에 붙일 "미리 들려준 자기 이야기" 부분을 만든다. 기본 질문 차례가 아니거나 내용이 없으면 빈 문자열."""
    if mode != "main" or not about_me:
        return ""
    lines = "\n".join(f"- {x}" for x in about_me)
    return (
        "[사용자가 미리 들려준 자기 이야기 — 사용자가 직접 확인한 내용]\n"
        f"{lines}\n"
        "새 기본 질문은 가능하면 이 이야기 중 아직 다루지 않은 것을 소재로 삼습니다(예: 해 본 일, 관심사, 장점). "
        "면접에 대한 걱정은 질문 소재로 쓰지 말고 질문의 부담을 조절하는 데만 참고합니다. "
        "적혀 있지 않은 내용을 사실처럼 덧붙이지 않습니다.\n\n"
    )


def _fallback_question(
    tier: str, previous_questions: list[str], mode: str, past_questions: list[str] | None = None
) -> str:
    """고정 질문 중 이번 면접에서 안 나온 것을 고른다(기본 질문은 지난 면접에도 안 나온 것을 우선)."""
    if mode == "follow_up":
        intro = _in_intro_block(tier, previous_questions)
        pool = (INTRO_FOLLOW_UPS if intro else FALLBACK_FOLLOW_UPS).get(tier, FALLBACK_FOLLOW_UPS["standard"])
        unused = [q for q in pool if q not in previous_questions]
        return (unused or pool)[0]

    pool = FALLBACK_QUESTIONS.get(tier, FALLBACK_QUESTIONS["standard"])
    remaining = [q for q in pool if q not in previous_questions]
    # 첫 질문에서 자기소개를 이미 했으면 자기소개 질문은 제외 (문장이 달라서 위 비교로는 안 걸림)
    if any("자기소개" in q for q in previous_questions):
        remaining = [q for q in remaining if "자기소개" not in q]
    if not remaining:
        remaining = pool
    # 지난 면접에서 나온 적 없는 질문을 우선한다. 다 나왔던 거라면(고정 질문은 난이도당 6개뿐) 그냥 고른다.
    fresh = [q for q in remaining if q not in (past_questions or [])]
    return random.choice(fresh or remaining)


def _user_prompt(
    tier: str,
    previous_questions: list[str],
    previous_answer: str | None,
    mode: str,
    profile: dict | None = None,
    past_questions: list[str] | None = None,
    about_me: list[str] | None = None,
) -> str:
    """난이도·이미 나온 질문·참고 정보를 모아 AI에게 보낼 글을 만든다."""
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
        f"{_about_section(about_me, mode)}"
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
    about_me: list[str] | None = None,
) -> tuple[str, str]:
    """다음 질문과 출처('llm' 또는 'fallback')를 돌려준다. profile·past_questions·about_me는 질문 방식·주제 참고용."""
    previous_questions = previous_questions or []

    # 꼬리질문인데 대답 글이 없으면(인식 실패 등) 파고들 내용이 없어 고정 꼬리질문으로
    if mode == "follow_up" and not previous_answer:
        return _fallback_question(tier, previous_questions, mode, past_questions), "fallback"

    if _client is None:
        return _fallback_question(tier, previous_questions, mode, past_questions), "fallback"

    try:
        resp = _client.with_options(timeout=timeout).messages.create(
            model=MODEL,
            max_tokens=200,
            **effort_options("low"),  # 질문 하나 — 저비용/저지연 (SDK 버전과 무관하게 넘기는 법은 llm_util.py)
            system=[{
                "type": "text",
                "text": COMMON_RULES,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{
                "role": "user",
                "content": _user_prompt(
                    tier, previous_questions, previous_answer, mode, profile, past_questions, about_me,
                ),
            }],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if text:
            return text, "llm"
        return _fallback_question(tier, previous_questions, mode, past_questions), "fallback"
    except Exception as e:  # 네트워크/키/쿼터 등 — 화면 흐름은 절대 안 깨지게
        log.warning("Step3 질문 생성 실패: %s", e)
        return _fallback_question(tier, previous_questions, mode, past_questions), "fallback"