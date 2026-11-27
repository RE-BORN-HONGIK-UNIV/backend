"""3단계 · 면접 기록 요청 검증 — 면접 중 질문·대답 글을 저장하는 API의 입력 형식과 길이를 엄격히 확인한다(대답은 민감할 수 있어서).
app.py와 떼어 둔 건 CI 테스트에서 DB 없이 불러오기 위해서다.
"""
from __future__ import annotations

TIERS = ("warmup", "standard", "practice")
KINDS = ("main", "follow_up")

# 길이 상한 — 질문은 AI가 만든 한 문장, 대답은 3분 안팎 음성 인식 글 기준으로 여유 있게
MAX_QUESTION_LEN = 1000
MAX_ANSWER_LEN = 5000
# 한 면접의 질문 수 상한 — 정상 면접보다 훨씬 크게 잡아 잘못된 요청이 무한히 쌓이는 것만 막는다
MAX_TURNS_PER_SESSION = 30


def parse_session_start(data) -> str:
    """면접 시작 요청에서 tier를 꺼낸다. 잘못되면 ValueError(사용자용 메시지)."""
    if not isinstance(data, dict):
        raise ValueError("요청 본문이 올바르지 않습니다")
    tier = data.get("tier")
    if tier not in TIERS:
        raise ValueError("tier는 'warmup' | 'standard' | 'practice' 중 하나여야 합니다")
    return tier


def parse_turn(data) -> tuple[str, str]:
    """질문 저장 요청에서 (kind, question)을 꺼낸다."""
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
    """대답 저장 요청에서 대답 글을 꺼낸다. 인식이 실패한 빈 대답("")도 건너뛴 기록으로 받는다."""
    if not isinstance(data, dict):
        raise ValueError("요청 본문이 올바르지 않습니다")
    answer = data.get("answer")
    if not isinstance(answer, str):
        raise ValueError("answer는 문자열이어야 합니다")
    answer = answer.strip()
    if len(answer) > MAX_ANSWER_LEN:
        raise ValueError(f"answer는 {MAX_ANSWER_LEN}자 이하여야 합니다")
    return answer
