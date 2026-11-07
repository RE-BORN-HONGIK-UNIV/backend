"""2단계 하루 통화 라벨링 결과(extract_live_session.py로 만든 labels.csv, 여러 개 가능) → 판정 정확도.

  - 시선: 눈동자 보정 있음(지금 앱) vs 없음(예전) 비교 + 눈동자 폭(0.15) 주변 탐색
  - 미소: 지금 앱 값(입 벌림 0.05 넘으면 0, 원래 점수 0.5 이상은 예외) vs 입 벌림 기준을 바꾼 방식 비교
  - 긴장·깜빡임: 지금 임계값과 0.05 간격 탐색
여러 사람 파일을 합쳐서 본다 — 한 사람 데이터로만 숫자를 바꾸지 말 것(step2/ACCURACY_NOTES.md 긴장 임계값 사례).

사용법: python tests/labeling/evaluate_live.py labels1.csv [labels2.csv ...]
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_session import binary_metrics, gaze_prediction  # noqa: E402

SMILE_THRESHOLD = 0.35
SMILE_HIGH_CONFIDENCE_BYPASS = 0.5
TENSION_THRESHOLD = 0.4
BLINK_THRESHOLD = 0.5


def _f(v):
    v = (v or "").strip()
    return float(v) if v else None


def _label(v):
    v = (v or "").strip()
    return int(v) if v in ("0", "1") else None


def load(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                row = {k: _f(v) if k not in ("image", "session") and not k.startswith("label_") else v for k, v in r.items()}
                for k in ("label_gaze", "label_smile", "label_tension", "label_blink"):
                    row[k] = _label(r.get(k))
                row["face"] = bool(row.get("face"))
                rows.append(row)
    return rows


def show(title, m):
    print(f"  {title:<38} n={m['n']:<4} 정확도 {m['accuracy']}  precision {m['precision']}  recall {m['recall']}  (tp {m['tp']} fp {m['fp']} tn {m['tn']} fn {m['fn']})")


def sweep(rows, score_key, label_key, current, lo=0.05, hi=0.95):
    best = None
    t = lo
    while t <= hi + 1e-9:
        m = binary_metrics([(1 if (r[score_key] or 0) >= t else 0, r[label_key]) for r in rows if r[score_key] is not None])
        if m["accuracy"] is not None and (best is None or m["accuracy"] > best[1]["accuracy"]):
            best = (round(t, 2), m)
        t += 0.05
    if best:
        show(f"가장 정확한 임계값 {best[0]} (지금 {current})", best[1])


def smile_with_gate(r, jaw_gate):
    """미소 판정 값 — jaw_gate=None이면 입 벌림 처리 없이 원래 점수."""
    raw, jaw = r["smile_raw"], r["jaw_open"]
    if raw is None:
        return None
    if jaw_gate is None or raw >= SMILE_HIGH_CONFIDENCE_BYPASS or (jaw or 0) <= jaw_gate:
        return raw
    return 0.0


def main(paths):
    rows = load(paths)
    sessions = sorted({r["session"] for r in rows})
    print(f"파일 {len(paths)}개 · 세션 {len(sessions)}개 · 장면 {len(rows)}개\n")

    print("■ 시선 (label_gaze: 하루를 보고 있으면 1)")
    show("눈동자 보정 있음 (지금 앱)", binary_metrics([(int(r["gaze_app"] or 0), r["label_gaze"]) for r in rows]))
    show("눈동자 보정 없음 (예전)", binary_metrics([(int(r["gaze_noiris"] or 0), r["label_gaze"]) for r in rows]))
    for th in (0.10, 0.15, 0.20, 0.25):
        show(f"  보정 있음 · 눈동자 폭 ±{th}", binary_metrics([(gaze_prediction(r, True, iris_thresh=th), r["label_gaze"]) for r in rows]))
    print()

    print("■ 미소 (label_smile: 웃고 있으면 1)")
    show(f"지금 앱 (입 벌림 0.05, ≥{SMILE_THRESHOLD})", binary_metrics([(1 if (r["smile_app"] or 0) >= SMILE_THRESHOLD else 0, r["label_smile"]) for r in rows if r["smile_app"] is not None]))
    for gate in (0.1, 0.2, 0.3, None):
        vals = [(smile_with_gate(r, gate), r["label_smile"]) for r in rows]
        name = "입 벌림 처리 없음" if gate is None else f"입 벌림 기준 {gate}"
        show(f"{name} (≥{SMILE_THRESHOLD})", binary_metrics([(1 if v >= SMILE_THRESHOLD else 0, lab) for v, lab in vals if v is not None]))
    sweep(rows, "smile_app", "label_smile", SMILE_THRESHOLD)
    print()

    print("■ 긴장 (label_tension: 굳은·찌푸린 표정이면 1)")
    show(f"지금 앱 (≥{TENSION_THRESHOLD})", binary_metrics([(1 if (r["tension_app"] or 0) >= TENSION_THRESHOLD else 0, r["label_tension"]) for r in rows if r["tension_app"] is not None]))
    sweep(rows, "tension_app", "label_tension", TENSION_THRESHOLD)
    if any(r.get("tension_delta") is not None for r in rows):
        # 2026-11 판정 구조: 평소(캘리브레이션) 대비 — 사람마다 무표정 값을 뺀다. 0.02 간격으로 더 촘촘히 본다
        print("  · 평소 대비(tension_delta) 기준 탐색 — 여러 사람의 찡그림이 고르게 섞였는지 먼저 확인할 것")
        best = None
        t = 0.02
        while t <= 0.5 + 1e-9:
            m = binary_metrics([(1 if r["tension_delta"] >= t else 0, r["label_tension"]) for r in rows if r.get("tension_delta") is not None])
            if m["accuracy"] is not None:
                show(f"    평소 대비 ≥{round(t, 2)}", m)
                if best is None or m["accuracy"] > best[1]["accuracy"]:
                    best = (round(t, 2), m)
            t += 0.02
        if best:
            show(f"  가장 정확한 평소 대비 기준 {best[0]}", best[1])
    print()

    print("■ 깜빡임 (label_blink: 눈을 감는 중이면 1 — 사진 한 장 기준이라 참고용)")
    show(f"지금 앱 (≥{BLINK_THRESHOLD})", binary_metrics([(1 if (r["blink_score"] or 0) >= BLINK_THRESHOLD else 0, r["label_blink"]) for r in rows if r["blink_score"] is not None]))
    sweep(rows, "blink_score", "label_blink", BLINK_THRESHOLD)
    print()
    print("※ 숫자를 바꾸기 전에: 여러 사람(세션)이 고르게 섞였는지, 한 사람이 대부분이 아닌지 확인하고 결과를 step2/ACCURACY_NOTES.md에 남길 것.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
