"""1단계 · 분석 요청 옵션 — /analyze로 들어온 녹음이 몇 단계 것인지 읽고, 1단계 녹음만 1단계 점수로 저장하게 한다.
3단계 대답도 같은 분석을 쓰는데, 그때 저장하면 1단계 점수가 덮이기 때문이다. 값이 없거나 이상하면 1단계로 본다.
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
