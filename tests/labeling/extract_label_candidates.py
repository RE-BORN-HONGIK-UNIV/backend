"""2단계 임계값 검증용 라벨링 후보 추출.

영상에서 1초 간격으로 프레임을 뽑아 이미지로 저장하고, 그 프레임의
blendshape 기반 미소/긴장/깜빡임 점수를 CSV에 같이 적어둔다. 사람이
이미지를 직접 보고 label_smile / label_tension / label_blink 칸에
0 또는 1(정답)을 채워넣으면, evaluate_threshold.py로 현재 임계값
(미소 0.35 / 긴장 0.4, step2/expression_analyzer.py)이 실제로 맞는지
확인할 수 있다. (근거 검증 배경은 step2/ACCURACY_NOTES.md 참고)

2026-09-18(6차 검증, practical-curie 세션에서 발견한 버그 반영): smile_score/
tension_score는 원래 jawOpen 게이팅·SMILE_HIGH_CONFIDENCE_BYPASS 이전의 raw
값이었다 — 앱이 실제로 쓰는 값(app.py가 compute_expression_series로 계산하는
값)과 달라서 이 컬럼만으로는 게이팅 효과를 검증할 수 없었다. 수식을 하네스에
따로 재구현해 컬럼을 추가하는 대신, compute_expression_series를 그대로
호출해 smile_score_app/tension_score_app에 "앱이 실제로 보는 값"을 적는다 —
프로덕션 로직과 하네스가 따로 놀면서 벌어지는 이런 종류의 버그를 원천적으로
막기 위함. raw 값(게이팅 전)도 진단용으로 같이 남긴다.

baseline-relative 긴장 판정(calibrate_baseline_tension)은 6차 검증에서 시도했다가
기각됨(browDown 단독 신호에 얹으면 과보정 — ACCURACY_NOTES.md 참고) — app.py는
이 보정을 쓰지 않으므로 smile_score_app/tension_score_app도 baseline 없이
compute_expression_series(frames) 그대로 계산해 앱과 정확히 같은 값을 담는다.
baseline 적용 시 값이 궁금할 때를 위해 tension_score_baseline_adjusted 컬럼을
진단용으로 별도 남긴다(앱이 쓰는 값 아님, 참고용).

사용법: python tests/labeling/extract_label_candidates.py <video_path> [output_dir]
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2

from step2.expression_analyzer import SMILE_KEYS, TENSION_KEYS, JAW_OPEN_KEY, compute_expression_series
from step2.landmark_face_points import extract_landmarks_from_video
from step2.set_baseline import calibrate_baseline_tension
from step2.smile_cascade import detect_smile_in_frame

SAMPLE_INTERVAL_SEC = 1.0
CALIBRATION_SEC = 5.0  # app.py의 캘리브레이션 구간(앞 5초)과 동일하게 맞춤


def _avg(blendshapes, keys):
    vals = [blendshapes.get(k, 0.0) for k in keys]
    return sum(vals) / len(vals) if vals else 0.0


def main(video_path, output_dir="label_candidates"):
    os.makedirs(output_dir, exist_ok=True)
    frames = extract_landmarks_from_video(video_path)

    calib_frames = [f for f in frames if f["t"] <= CALIBRATION_SEC]
    baseline_tension = calibrate_baseline_tension(calib_frames)
    print(f"baseline_tension = {baseline_tension:.4f} (앞 {CALIBRATION_SEC}초 기준, 참고용 — app.py는 현재 이 보정을 쓰지 않음)")

    # 앱과 정확히 같은 함수·같은 인자(baseline 없음)로 한 번에 계산 —
    # 하네스가 게이팅/바이패스 로직을 따로 재구현하지 않는다.
    app_series = compute_expression_series(frames)
    # 참고용 — baseline 보정을 적용하면 값이 어떻게 달라지는지(6차 검증 결과 앱에는 미적용)
    baseline_series = compute_expression_series(frames, baseline_tension=baseline_tension)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    rows = []
    next_t = 0.0
    for f, (_, smile_app, tension_app), (_, _, tension_baseline) in zip(frames, app_series, baseline_series):
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
            # 앱이 실제로 판정에 쓰는 값(게이팅/바이패스 반영, baseline 보정은 미적용 — 위 docstring 참고)
            "smile_score_app": round(smile_app, 4) if smile_app is not None else "",
            "tension_score_app": round(tension_app, 4) if tension_app is not None else "",
            # 진단용 raw 값 — 게이팅 전, 어떤 프레임에서 왜 보정됐는지 비교할 때 씀
            "smile_score_raw": round(_avg(bs, SMILE_KEYS), 4) if bs else "",
            "tension_score_raw": round(_avg(bs, TENSION_KEYS), 4) if bs else "",
            # 진단용 — baseline 보정을 적용했다면 어떤 값이었을지(앱은 이 값을 쓰지 않음)
            "tension_score_baseline_adjusted": round(tension_baseline, 4) if tension_baseline is not None else "",
            "blink_score": round((bs.get("eyeBlinkLeft", 0.0) + bs.get("eyeBlinkRight", 0.0)) / 2, 4) if bs else "",
            # jawOpen 게이팅이 미소를 잘못 지우는 경우를 따로 진단할 수 있게 —
            # 2026-09-16 미소 미검출 재조사(ACCURACY_NOTES.md "2차 검증") 때 이 컬럼이
            # 없어서 매번 별도 스크립트를 급조해야 했음
            "jaw_open_score": round(bs.get(JAW_OPEN_KEY, 0.0), 4) if bs else "",
            # blendshape와 별개 신호(Haar Cascade, step2/smile_cascade.py) — 2026-09-16
            # "2차 검증" 대안 후보, min_neighbors=6/10/15 세 값으로 같이 뽑아서 비교 가능하게
            "smile_haar_n6": detect_smile_in_frame(img, min_neighbors=6),
            "smile_haar_n10": detect_smile_in_frame(img, min_neighbors=10),
            "smile_haar_n15": detect_smile_in_frame(img, min_neighbors=15),
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
