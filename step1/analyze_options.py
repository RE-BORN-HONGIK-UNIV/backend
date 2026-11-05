"""/analyze 요청 옵션 — 어느 단계의 녹음을 분석하는지(순수 함수, 테스트용으로 app.py에서 분리).

/analyze는 원래 1단계 전용이라 분석할 때마다 Stage1Result(1단계 최신 점수)로 저장한다. 이 점수는 3단계 면접
난이도와 대시보드 1단계 완료에 쓰이므로, 3단계 면접 대답을 같은 분석에 보낼 때(2026-11 경량 웹 리포트의
1→2→3단계 흐름·질문별 말버릇 표시) 저장하면 1단계 점수가 3단계 녹음으로 덮인다. 그래서 단계를 받아
1단계일 때만 저장한다. 값이 없거나 이상하면 예전처럼 1단계로 본다(하위 호환).
"""

ANALYZE_STAGES = (1, 2, 3)


def parse_analyze_stage(raw) -> int:
    """form-data 'stage' 값 → 1·2·3. 없거나 알 수 없는 값이면 1."""
    try:
        stage = int(str(raw).strip())
    except (TypeError, ValueError):
        return 1
    return stage if stage in ANALYZE_STAGES else 1


def saves_stage1_result(stage: int) -> bool:
    """1단계 녹음만 Stage1Result(3단계 난이도·대시보드 기준)로 저장한다."""
    return stage == 1
