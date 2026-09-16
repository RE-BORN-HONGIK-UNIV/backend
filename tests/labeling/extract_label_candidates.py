"""2단계 임계값 검증용 라벨링 후보 추출.

영상에서 1초 간격으로 프레임을 뽑아 이미지로 저장하고, 그 프레임의
blendshape 기반 미소/긴장/깜빡임 점수를 CSV에 같이 적어둔다. 사람이
이미지를 직접 보고 label_smile / label_tension / label_blink 칸에
0 또는 1(정답)을 채워넣으면, evaluate_threshold.py로 현재 임계값
(미소 0.35 / 긴장 0.4, step2/expression_analyzer.py)이 실제로 맞는지
확인할 수 있다. (근거 검증 배경은 step2/ACCURACY_NOTES.md 참고)

사용법: python tests/labeling/extract_label_candidates.py <video_path> [output_dir]
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2

from step2.expression_analyzer import SMILE_KEYS, TENSION_KEYS, JAW_OPEN_KEY
from step2.landmark_face_points import extract_landmarks_from_video

SAMPLE_INTERVAL_SEC = 1.0


def _avg(blendshapes, keys):
    vals = [blendshapes.get(k, 0.0) for k in keys]
    return sum(vals) / len(vals) if vals else 0.0


def main(video_path, output_dir="label_candidates"):
    os.makedirs(output_dir, exist_ok=True)
    frames = extract_landmarks_from_video(video_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    rows = []
    next_t = 0.0
    for f in frames:
        if f["t"] < next_t:
            continue
        next_t += SAMPLE_INTERVAL_SEC

        cap.set(cv2.CAP_PROP_POS_FRAMES, round(f["t"] * fps))
        ok, img = cap.read()
        if not ok:
            continue

        img_name = f"t{f['t']:.2f}.jpg"
        cv2.imwrite(os.path.join(output_dir, img_name), img)

        bs = f.get("blendshapes")
        rows.append({
            "image": img_name,
            "t": round(f["t"], 2),
            "smile_score": round(_avg(bs, SMILE_KEYS), 4) if bs else "",
            "tension_score": round(_avg(bs, TENSION_KEYS), 4) if bs else "",
            "blink_score": round((bs.get("eyeBlinkLeft", 0.0) + bs.get("eyeBlinkRight", 0.0)) / 2, 4) if bs else "",
            # jawOpen 게이팅이 미소를 잘못 지우는 경우를 따로 진단할 수 있게 —
            # 2026-09-16 미소 미검출 재조사(ACCURACY_NOTES.md "2차 검증") 때 이 컬럼이
            # 없어서 매번 별도 스크립트를 급조해야 했음
            "jaw_open_score": round(bs.get(JAW_OPEN_KEY, 0.0), 4) if bs else "",
            "label_smile": "",    # 이미지 보고 웃는 게 맞으면 1, 아니면 0
            "label_tension": "",  # 긴장한 표정이 맞으면 1, 아니면 0
            "label_blink": "",    # 눈 감는 중이면 1, 아니면 0
        })

    cap.release()

    if not rows:
        print("추출된 프레임이 없습니다 — 영상 경로/길이를 확인하세요.")
        return

    csv_path = os.path.join(output_dir, "labels.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)}개 프레임 → {output_dir}/ 에 이미지 + labels.csv 생성 완료.")
    print("이미지 보면서 label_smile / label_tension / label_blink 칸을 0 또는 1로 채운 뒤")
    print("evaluate_threshold.py labels.csv 로 넘기세요.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python extract_label_candidates.py <video_path> [output_dir]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "label_candidates")
