"""
Step 3 · 면접 기록(세션·질문·답변) 요청 검증 (순수 함수).

면접 중 질문/답변 텍스트를 서버에 저장하는 API의 입력 검증을 모아둔다. 답변 텍스트는 발화·사회불안
사용자의 말이라 민감할 수 있으므로, 저장하는 값의 형식·길이를 서버에서 엄격히 제한한다.

app.py에 두지 않은 이유: app.py는 import 시 DB·matplotlib 등이 필요해서 CI의 순수 로직
테스트에서 불러올 수 없다.
"""
from __future__ import annotations

TIERS = ("warmup", "standard", "practice")
KINDS = ("main", "follow_up")

# 길이 상한 — 질문은 LLM이 만든 한 문장(여유 있게), 답변은 음성 인식 텍스트(3분 안팎 발화 기준 여유)
MAX_QUESTION_LEN = 1000
MAX_ANSWER_LEN = 5000
# 한 면접에서 저장할 수 있는 질문 수 상한 — 정상 면접(기본 3개 + 꼬리질문)보다 훨씬 크게 잡아
# 정상 사용엔 영향 없고, 잘못된 클라이언트가 무한히 쌓는 것만 막는다
MAX_TURNS_PER_SESSION = 30


def parse_session_start(data) -> str:
    """면접 시작 요청 → 검증된 tier. 잘못되면 ValueError(사용자용 메시지)."""
    if not isinstance(data, dict):
        raise ValueError("요청 본문이 올바르지 않습니다")
    tier = data.get("tier")
    if tier not in TIERS:
        raise ValueError("tier는 'warmup' | 'standard' | 'practice' 중 하나여야 합니다")
    return tier


def parse_turn(data) -> tuple[str, str]:
    """질문 저장 요청 → (kind, question)."""
    if not isinstance(data, dict):
        raise ValueError("요청 본문이 올바르지 않습니다")
    kind = data.get("kind")
    if kind not in KINDS:
        raise ValueError("kind는 'main' | 'follow_up' 중 하나여야 합니다")
    question = data.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question이 필요합니다")
    question = question.strip()
    if len(question) > MAX_QUESTION_LEN:
        raise ValueError(f"question은 {MAX_QUESTION_LEN}자 이하여야 합니다")
    return kind, question


def parse_answer(data) -> str:
    """답변 저장 요청 → 답변 텍스트. 음성 인식이 실패한 빈 답변("")도 허용한다(답변을 건너뛴 기록)."""
    if not isinstance(data, dict):
        raise ValueError("요청 본문이 올바르지 않습니다")
    answer = data.get("answer")
    if not isinstance(answer, str):
        raise ValueError("answer는 문자열이어야 합니다")
    answer = answer.strip()
    if len(answer) > MAX_ANSWER_LEN:
        raise ValueError(f"answer는 {MAX_ANSWER_LEN}자 이하여야 합니다")
    return answer
