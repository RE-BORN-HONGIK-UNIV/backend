"""라벨링된 CSV를 읽어서 현재 임계값의 정확도를 확인하고, 가장 정확도 높은
임계값을 탐색한다. numpy/torch 등 무거운 의존성 없이 표준 라이브러리만 씀.

사용법: python tests/labeling/evaluate_threshold.py <labels.csv>
"""
import csv
import sys


def evaluate(rows, score_key, label_key, threshold):
    tp = fp = tn = fn = skipped = 0
    for r in rows:
        label, score = r[label_key].strip(), r[score_key].strip()
        if label == "" or score == "":
            skipped += 1
            continue
        label = int(label)
        predicted = 1 if float(score) >= threshold else 0
        if predicted == 1 and label == 1:
            tp += 1
        elif predicted == 1 and label == 0:
            fp += 1
        elif predicted == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "skipped": skipped,
        "accuracy": round((tp + tn) / total, 3) if total else None,
        "precision": round(tp / (tp + fp), 3) if (tp + fp) else None,
        "recall": round(tp / (tp + fn), 3) if (tp + fn) else None,
    }


def sweep_best_threshold(rows, score_key, label_key):
    best = None
    for i in range(5, 96, 5):
        result = evaluate(rows, score_key, label_key, i / 100)
        if result["accuracy"] is not None and (best is None or result["accuracy"] > best["accuracy"]):
            best = result
    return best


def report(rows, name, score_key, label_key, current_threshold):
    print(f"=== {name} (현재 임계값 {current_threshold}) ===")
    print(evaluate(rows, score_key, label_key, current_threshold))
    print("최적 임계값 탐색(0.05~0.95, 0.05 간격):")
    print(sweep_best_threshold(rows, score_key, label_key))
    print()


def main(csv_path):
    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    # 2026-09-18(6차 검증): smile_score_app/tension_score_app이 앱이 실제로 쓰는 값
    # (게이팅·바이패스·baseline 보정 전부 반영) — extract_label_candidates.py
    # 갱신판(compute_expression_series 직접 호출)으로 생성한 CSV라면 이 컬럼이 있음.
    # 구버전 CSV(smile_score/tension_score만 있는 raw 값)와의 하위호환을 위해
    # 컬럼이 없으면 raw 쪽으로 폴백해서 리포트한다.
    if rows and "smile_score_app" in rows[0]:
        report(rows, "미소(앱과 동일)", "smile_score_app", "label_smile", 0.35)
    else:
        report(rows, "미소(raw, 구버전 CSV — 게이팅 전 값이라 앱과 다를 수 있음)", "smile_score", "label_smile", 0.35)

    if rows and "tension_score_app" in rows[0]:
        report(rows, "긴장(앱과 동일)", "tension_score_app", "label_tension", 0.4)
    else:
        report(rows, "긴장(raw, 구버전 CSV — baseline 보정 전 값이라 앱과 다를 수 있음)", "tension_score", "label_tension", 0.4)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python evaluate_threshold.py <labels.csv>")
        sys.exit(1)
    main(sys.argv[1])
