"""2단계 하루 통화(실시간 분석) 정확도 검증용 순수 함수 — 표준 라이브러리만 쓴다(CI 유닛테스트 대상).

프론트 2단계 통화를 주소 뒤 `?debug=1`로 하면 통화를 끝낼 때 영상(webm)과 장면별 값 표(CSV)가 내려받아진다
(frontend src/features/face/live/useDebugRecorder.ts). 이 모듈은 그 CSV를 읽어
  - 앱과 똑같이 "대답하는 동안, 앞뒤 1초 제외" 구간을 잘라 내고 (useLiveFaceSession measureOnlyWhenAnswering)
  - 앱과 똑같이 대답마다 시선 스무딩(EMA α=0.3)을 새로 시작해 (gaze.ts smoothGaze, closeAnswerWindow)
  - 시선 판정을 눈동자 보정 있음/없음 두 방식으로 다시 계산한다(headPose.ts isLookingAtCamera)
판정을 CSV에 고정해 두지 않고 원래 값으로 다시 계산하는 이유: 같은 기록으로 여러 방식·임계값을 비교하려고.
"""
from __future__ import annotations

import csv

# 앱(frontend useLiveFaceSession.ts)과 같은 값
ANSWER_LEAD_SKIP_SEC = 1.0
ANSWER_TAIL_TRIM_SEC = 1.0
GAZE_SMOOTHING_ALPHA = 0.3
YAW_THRESH = 10.0
PITCH_THRESH = 10.0
IRIS_THRESH = 0.15


def _num(v):
    v = (v or "").strip()
    return float(v) if v else None


def read_debug_csv(path):
    """프론트가 내려준 CSV → dict 목록(숫자 칸은 float/None, 0·1 칸은 bool)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            row = {k: _num(v) for k, v in r.items()}
            for k in ("calibrating", "answering", "face"):
                row[k] = bool(row.get(k))
            rows.append(row)
    return rows


def answer_windows(rows):
    """연속으로 answering인 구간 → [(start, end)] (세션 시각, 앞뒤 자르기 전)."""
    windows, start, last = [], None, None
    for r in rows:
        if r["answering"]:
            if start is None:
                start = r["t"]
            last = r["t"]
        elif start is not None:
            windows.append((start, last))
            start = None
    if start is not None:
        windows.append((start, last))
    return windows


def measured_windows(rows, lead=ANSWER_LEAD_SKIP_SEC, tail=ANSWER_TAIL_TRIM_SEC):
    """앱이 실제로 재는 구간 — 대답마다 앞 lead초·뒤 tail초를 뺀다. 남는 게 없으면 뺀다."""
    out = []
    for a, b in answer_windows(rows):
        s, e = a + lead, b - tail
        if e > s:
            out.append((s, e))
    return out


def is_looking(sy, sp, six, siy, base, use_iris_baseline=True,
               yaw_thresh=YAW_THRESH, pitch_thresh=PITCH_THRESH, iris_thresh=IRIS_THRESH):
    """headPose.ts isLookingAtCamera와 같은 판정. use_iris_baseline=False면 예전 방식(눈동자 기준 0)."""
    bix = base["base_iris_x"] if use_iris_baseline else 0.0
    biy = base["base_iris_y"] if use_iris_baseline else 0.0
    return (
        abs(sy - base["base_yaw"]) < yaw_thresh
        and abs(sp - base["base_pitch"]) < pitch_thresh
        and abs(six - bix) < iris_thresh
        and abs(siy - biy) < iris_thresh
    )


def smoothed_measured_rows(rows, alpha=GAZE_SMOOTHING_ALPHA):
    """앱이 재는 구간의 장면만, 스무딩한 시선 값(s_yaw 등)을 붙여 돌려준다.
    대답(구간)마다 스무딩을 새로 시작하고, 얼굴이 안 잡힌 장면은 스무딩을 건너뛴다(앱과 같음 — 판정은 회피)."""
    out = []
    for s, e in measured_windows(rows):
        prev = None
        for r in rows:
            if not (s <= r["t"] <= e):
                continue
            row = dict(r)
            if r["face"] and r["yaw"] is not None:
                cur = (r["yaw"], r["pitch"], r["iris_x"], r["iris_y"])
                prev = cur if prev is None else tuple(alpha * c + (1 - alpha) * p for c, p in zip(cur, prev))
                row["s_yaw"], row["s_pitch"], row["s_iris_x"], row["s_iris_y"] = prev
            else:
                row["s_yaw"] = row["s_pitch"] = row["s_iris_x"] = row["s_iris_y"] = None
            out.append(row)
    return out


def gaze_prediction(row, use_iris_baseline=True, **thresh):
    """한 장면의 시선 판정 1(봄)/0(회피). 얼굴 미검출이면 0(앱과 같음 — 화면 밖도 회피로 봄)."""
    if row.get("s_yaw") is None:
        return 0
    return 1 if is_looking(row["s_yaw"], row["s_pitch"], row["s_iris_x"], row["s_iris_y"], row,
                           use_iris_baseline=use_iris_baseline, **thresh) else 0


def sample_every(rows, step_sec=1.0):
    """장면을 step_sec 간격으로 고른다(사람이 라벨링할 양을 줄이려고)."""
    picked, next_t = [], None
    for r in rows:
        if next_t is None or r["t"] >= next_t:
            picked.append(r)
            next_t = r["t"] + step_sec
    return picked


def binary_metrics(pairs):
    """[(예측, 정답)] → 정확도·precision·recall. 정답이 비어 있는 줄은 건너뛴다."""
    tp = fp = tn = fn = 0
    for pred, label in pairs:
        if label is None:
            continue
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and not label:
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    return {
        "n": total, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round((tp + tn) / total, 3) if total else None,
        "precision": round(tp / (tp + fp), 3) if (tp + fp) else None,
        "recall": round(tp / (tp + fn), 3) if (tp + fn) else None,
    }


# ── 2026-11 판정 변경(프론트 blink.ts MAX_BLINK_DURATION · gaze.ts markClosuresAsAversion과 같은 규칙) ──
BLINK_SCORE_THRESH = 0.5
MIN_BLINK_SEC = 0.08
MAX_BLINK_SEC = 0.4
SMILE_THRESH = 0.35


def eye_closures(rows):
    """앱이 재는 구간 안의 눈 감김 → (깜빡임 목록, 시선 이탈로 넘길 긴 감김 목록). 각 항목은 (시작, 끝).

    - 0.08~0.4초: 깜빡임(Schiffman 2001: 깜빡임 0.1~0.4초)
    - 0.4초 초과: 깜빡임이 아님. 그동안 원래 미소 값이 0.35 미만이면 아래를 본 것으로 보고 시선 이탈(Becker & Fuchs 1988),
      웃는 중이면(눈이 가늘어짐) 넘기지 않는다
    - 대답(구간)이 끝날 때 감겨 있던 건 버린다(앱과 같음)
    """
    blinks, long_closures = [], []
    for s, e in measured_windows(rows):
        start, smile_max = None, 0.0
        for r in rows:
            if not (s <= r["t"] <= e) or r.get("blink_score") is None:
                continue
            score, smile = r["blink_score"], r.get("smile_raw") or 0.0
            if start is not None or score >= BLINK_SCORE_THRESH:
                smile_max = max(smile_max, smile)
            if score >= BLINK_SCORE_THRESH and start is None:
                start = r["t"]
            elif score < BLINK_SCORE_THRESH and start is not None:
                d = r["t"] - start
                if d >= MIN_BLINK_SEC:
                    if d <= MAX_BLINK_SEC:
                        blinks.append((start, r["t"]))
                    elif smile_max < SMILE_THRESH:
                        long_closures.append((start, r["t"]))
                start, smile_max = None, 0.0
            if start is None:
                smile_max = 0.0
    return blinks, long_closures


def baseline_tension(rows):
    """찡그림 판정의 평소 값 — 프론트와 같게 캘리브레이션 중(두 단계면 '하루의 눈 보기' 단계만) 미간 값의 중앙값.
    CSV에 base_tension 칸이 있으면(2026-11 이후 기록) 그 값을 쓰고, 없으면(예전 기록) 직접 계산한다."""
    for r in reversed(rows):
        if r.get("base_tension") is not None and r["base_tension"] > 0:
            return r["base_tension"]
    vals = []
    for r in rows:
        if not r.get("calibrating"):
            continue
        if r.get("calib_step") not in (None, 1.0):  # 두 단계면 눈 보기(1)만
            continue
        v = r.get("brow_down") if r.get("brow_down") is not None else r.get("tension_app")
        if v is not None:
            vals.append(v)
    if not vals:
        return 0.0
    vals.sort()
    m = len(vals) // 2
    return vals[m] if len(vals) % 2 else (vals[m - 1] + vals[m]) / 2


def tension_delta(row, base):
    """평소 대비 미간 값(프론트 tensionApp과 같음). 예전 기록은 tension_app이 원래 값이라 base를 뺀다."""
    raw = row.get("brow_down")
    if raw is None:
        raw = row.get("tension_app")
    return None if raw is None else raw - base
