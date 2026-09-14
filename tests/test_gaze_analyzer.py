import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step2.gaze_analyzer import is_looking_at_camera


def test_looking_at_camera_within_all_thresholds():
    assert is_looking_at_camera(yaw=5, pitch=-5, iris_offset_x=0.05, iris_offset_y=-0.05) is True


def test_not_looking_when_yaw_exceeds_threshold():
    assert is_looking_at_camera(yaw=11, pitch=0, iris_offset_x=0.0, iris_offset_y=0.0) is False


def test_not_looking_when_pitch_exceeds_threshold():
    assert is_looking_at_camera(yaw=0, pitch=-11, iris_offset_x=0.0, iris_offset_y=0.0) is False


def test_not_looking_when_iris_offset_exceeds_threshold():
    assert is_looking_at_camera(yaw=0, pitch=0, iris_offset_x=0.16, iris_offset_y=0.0) is False


def test_baseline_shifts_the_accepted_center():
    # baseline_yaw=20 인 사람에게는 yaw=25(차이 5)가 정면
    assert is_looking_at_camera(
        yaw=25, pitch=0, iris_offset_x=0.0, iris_offset_y=0.0, baseline_yaw=20,
    ) is True
    # 같은 yaw=25라도 baseline 없이(0 기준) 보면 차이가 25라 회피로 판정
    assert is_looking_at_camera(yaw=25, pitch=0, iris_offset_x=0.0, iris_offset_y=0.0) is False


def test_custom_thresholds_override_defaults():
    # 기본 yaw_thresh(10)라면 통과할 값이지만, 더 엄격한 5로 좁히면 회피로 판정
    assert is_looking_at_camera(
        yaw=8, pitch=0, iris_offset_x=0.0, iris_offset_y=0.0, yaw_thresh=5,
    ) is False
