import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step1.llm_feedback import _user_prompt


def _base_result():
    return {
        "scores": {"stability": 80, "fluency": 65},
        "filler_detail": {"filler_count": 3, "sound_segment_count": 12},
        "pause_detail": {"anxious_pause_count": 1, "total_duration_sec": 45.2},
    }


def test_user_prompt_includes_axis_labels_not_raw_keys():
    prompt = _user_prompt(_base_result())
    assert "음성 안정성: 80점" in prompt
    assert "발화 유창성: 65점" in prompt


def test_user_prompt_includes_detail_counts():
    prompt = _user_prompt(_base_result())
    assert "채움말 3회" in prompt
    assert "1.2초 이상 불안한 멈춤 1회" in prompt


def test_user_prompt_adds_demo_mode_disclaimer_when_flagged():
    result = _base_result()
    result["demo_mode"] = True
    prompt = _user_prompt(result)
    assert "음성 모델 파일이 없어" in prompt


def test_user_prompt_omits_disclaimer_when_not_demo_mode():
    prompt = _user_prompt(_base_result())
    assert "음성 모델 파일이 없어" not in prompt


def test_user_prompt_handles_missing_optional_fields():
    # scores/filler_detail/pause_detail이 비어있어도 죽지 않아야 함
    prompt = _user_prompt({})
    assert "채움말 0회" in prompt
