"""tests/labeling/live_session.py — 2단계 하루 통화 정확도 검증 하네스의 순수 함수(표준 라이브러리만)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "labeling"))

from live_session import (  # noqa: E402
    answer_windows, binary_metrics, eye_closures, gaze_prediction, measured_windows, sample_every,
    smoothed_measured_rows,
)

BASE = {"base_yaw": 0.0, "base_pitch": 0.0, "base_iris_x": 0.0, "base_iris_y": 0.2}


def row(t, answering, iris_y=0.2, face=True):
    return {"t": t, "answering": answering, "face": face, "yaw": 0.0, "pitch": 0.0,
            "iris_x": 0.0, "iris_y": iris_y if face else None, **BASE}


def test_answer_windows_and_trim_like_the_app():
    rows = [row(t / 2, 1 <= t / 2 <= 6) for t in range(0, 20)]
    assert answer_windows(rows) == [(1.0, 6.0)]
    # 앱처럼 앞 1초·뒤 1초를 뺀다
    assert measured_windows(rows) == [(2.0, 5.0)]
    # 2초짜리 대답은 남는 게 없어 뺀다
    assert measured_windows([row(0, True), row(2, True), row(3, False)]) == []


def test_iris_baseline_makes_looking_at_haru_count_as_looking():
    # 하루를 보느라 눈동자가 아래로 0.2 치우친 사람(캘리브레이션 기준도 0.2)
    rows = smoothed_measured_rows([row(t / 2, True) for t in range(0, 12)])
    assert rows and all(gaze_prediction(r, use_iris_baseline=True) == 1 for r in rows)
    # 예전 방식(눈동자 기준 0)은 같은 장면을 회피로 셌다
    assert all(gaze_prediction(r, use_iris_baseline=False) == 0 for r in rows)


def test_face_missing_counts_as_aversion():
    rows = smoothed_measured_rows([row(t / 2, True, face=(t != 5)) for t in range(0, 12)])
    missing = [r for r in rows if not r["face"]]
    assert missing and all(gaze_prediction(r) == 0 for r in missing)


def test_sample_every_and_metrics():
    picked = sample_every([{"t": t / 10} for t in range(0, 35)], step_sec=1.0)
    assert [p["t"] for p in picked] == [0.0, 1.0, 2.0, 3.0]
    m = binary_metrics([(1, 1), (1, 0), (0, 0), (0, 1), (1, None)])
    assert (m["tp"], m["fp"], m["tn"], m["fn"], m["n"]) == (1, 1, 1, 1, 4)
    assert m["accuracy"] == 0.5


def _closure_rows(spans, smile=0.0, end=10.0):
    """0.05초 간격 장면 — spans 안이면 눈 감김(0.9), 밖이면 뜸(0.1). 대답은 0~end초 내내."""
    rows, t = [], 0.0
    while t <= end + 1e-9:
        closed = any(a <= t < b for a, b in spans)
        rows.append({"t": round(t, 2), "answering": True, "calibrating": False, "face": True,
                     "blink_score": 0.9 if closed else 0.1, "smile_raw": smile})
        t += 0.05
    return rows


def test_blink_up_to_0_4s_and_longer_closure_becomes_aversion():
    rows = _closure_rows([(2.0, 2.2), (5.0, 8.3)])
    blinks, long_closures = eye_closures(rows)
    assert len(blinks) == 1 and abs(blinks[0][0] - 2.0) < 1e-6
    assert len(long_closures) == 1 and abs(long_closures[0][1] - 8.3) <= 0.06  # 0.05초 간격 장면


def test_long_closure_while_smiling_is_not_aversion():
    rows = _closure_rows([(5.0, 5.6)], smile=0.45)
    assert eye_closures(rows) == ([], [])
