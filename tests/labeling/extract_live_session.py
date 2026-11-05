"""2단계 하루 통화 기록(?debug=1로 내려받은 webm + csv) → 라벨링용 사진 + labels.csv.

앱이 실제로 재는 구간(대답하는 동안, 앞뒤 1초 제외)에서 1초마다 장면을 골라 영상에서 사진을 뽑고, 그 장면의
앱 값(스무딩한 시선 값·미소·긴장·깜빡임)을 표에 적는다. 사람이 사진을 보고 label_* 칸에 1/0을 채운 뒤
evaluate_live.py로 정확도를 계산한다. 사진·영상은 얼굴이 담긴 민감정보라 검증이 끝나면 지울 것.

사용법:
  python tests/labeling/extract_live_session.py <영상.webm> <값.csv> [출력폴더] [--all]
  --all: 대답 구간과 상관없이 전체에서 1초마다(통화 흐름 전체를 보고 싶을 때)
"""
import csv
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_session import gaze_prediction, read_debug_csv, sample_every, smoothed_measured_rows  # noqa: E402

LABEL_COLUMNS = ["label_gaze", "label_smile", "label_tension", "label_blink"]


def main(video_path, csv_path, out_dir, use_all=False):
    rows = read_debug_csv(csv_path)
    if use_all:
        picked = sample_every([dict(r, s_yaw=r["yaw"], s_pitch=r["pitch"], s_iris_x=r["iris_x"], s_iris_y=r["iris_y"]) for r in rows])
    else:
        picked = sample_every(smoothed_measured_rows(rows))
    picked = [r for r in picked if r.get("video_t") is not None]
    if not picked:
        print("뽑을 장면이 없어요 — 대답한 구간이 없거나(전부 넘김) 영상 시각(video_t)이 비어 있어요. --all로 다시 해 보세요.")
        return
    os.makedirs(out_dir, exist_ok=True)
    session = os.path.splitext(os.path.basename(csv_path))[0]

    # 브라우저 녹화(webm)는 길이 정보가 없어 원하는 시각으로 바로 이동이 안 되는 경우가 많아, 처음부터 읽으며 고른다
    cap = cv2.VideoCapture(video_path)
    targets = sorted(picked, key=lambda r: r["video_t"])
    out_rows, i = [], 0
    while i < len(targets):
        ok, frame = cap.read()
        if not ok:
            break
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        while i < len(targets) and pos >= targets[i]["video_t"]:
            r = targets[i]
            name = f"{session}_t{r['t']:.2f}.jpg"
            cv2.imwrite(os.path.join(out_dir, name), frame)
            out_rows.append({
                "image": name, "session": session, "t": round(r["t"], 2), "video_t": round(r["video_t"], 2),
                "face": int(bool(r["face"])),
                # 앱 판정(지금 코드 = 눈동자 보정 있음)과 예전 방식(눈동자 보정 없음)
                "gaze_app": gaze_prediction(r, use_iris_baseline=True),
                "gaze_noiris": gaze_prediction(r, use_iris_baseline=False),
                **{k: ("" if r.get(k) is None else round(r[k], 4)) for k in (
                    "s_yaw", "s_pitch", "s_iris_x", "s_iris_y", "base_yaw", "base_pitch", "base_iris_x", "base_iris_y",
                    "smile_app", "smile_raw", "jaw_open", "tension_app", "blink_score")},
                **{k: "" for k in LABEL_COLUMNS},
            })
            i += 1
    cap.release()

    labels_path = os.path.join(out_dir, "labels.csv")
    with open(labels_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"{len(out_rows)}장 → {out_dir}/ (사진 + labels.csv)")
    print("사진을 보고 labels.csv의 label_gaze(하루를 봄 1) / label_smile(웃음 1) / label_tension(굳은 표정 1) /")
    print("label_blink(눈 감는 중 1)에 1 또는 0을 적은 뒤: python tests/labeling/evaluate_live.py labels.csv [더 있으면 여러 개]")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    main(args[0], args[1], args[2] if len(args) > 2 else "live_labels", use_all="--all" in sys.argv)
