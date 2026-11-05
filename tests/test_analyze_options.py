import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step1.analyze_options import parse_analyze_stage, saves_stage1_result


def test_stage_defaults_to_1_for_missing_or_bad_values():
    # 예전 프론트(필드 없음)는 그대로 1단계로 저장돼야 한다
    for raw in (None, "", "abc", "0", "4", " "):
        assert parse_analyze_stage(raw) == 1


def test_stage_parses_known_values():
    assert parse_analyze_stage("3") == 3
    assert parse_analyze_stage(" 2 ") == 2
    assert parse_analyze_stage(1) == 1


def test_only_stage1_saves_stage1_result():
    # 3단계 대답 분석이 1단계 최신 점수(면접 난이도 기준)를 덮지 않아야 한다
    assert saves_stage1_result(1) is True
    assert saves_stage1_result(2) is False
    assert saves_stage1_result(3) is False
