"""Step 2 · 분석 결과를 저장·점수화해도 되는 영상인지 판단하는 품질 게이트 (순수 함수).

얼굴이 거의 안 잡힌 영상을 그대로 점수화하면 표정은 "미소 0·긴장 0 → 보통 50점",
시선은 fixation 구간이 없어 0점처럼 그럴듯한 숫자가 나오고, 그 값이 Stage2Result에
저장돼 "지난번 대비" 비교와 3단계 난이도(추천 면접관)에까지 반영된다 — 그래서 저장
전에 거절하고 다시 촬영하게 한다. 프론트 실시간 모드의 같은 기준:
frontend `src/features/face/live/quality.ts`.

app.py에 두지 않은 이유: app.py는 import 시 DB 등이 필요해서 CI의 순수 로직
테스트에서 불러올 수 없다.
"""
from __future__ import annotations

# 얼굴이 검출된 프레임 비율의 최소값. 0.5는 실측으로 최적화한 숫자가 아니라 "절반 이상
# 안 보이면 판정 근거가 부족하다"는 보수적 기준이다 — 프론트 캘리브레이션 확인 UI
# ("얼굴이 잘 인식됐어요" 기준 0.5)와 맞췄다. 팀 촬영 영상이 쌓이면 검출률 분포를 보고
# tests/labeling 하네스로 조정할 것 (docs/TESTING.md 2층).
MIN_FACE_DETECTED_RATIO = 0.5


def check_analyzable(frames) -> tuple[bool, str | None, float]:
    """landmark 추출 결과 프레임 리스트 → (분석 가능 여부, 거절 사유 메시지, 검출 비율).

    frames는 extract_landmarks_from_video의 반환 형식 — 얼굴 미검출 프레임은
    "landmarks"가 None이다. 사유 메시지는 사용자에게 그대로 보여줄 수 있는 문장
    (원인 추정이 아니라 다음에 할 행동 중심)."""
    if not frames or frames[-1]["t"] <= 0:
        return False, "영상에서 분석할 장면을 찾지 못했어요. 영상이 손상됐거나 너무 짧을 수 있어요. 다른 영상으로 다시 시도해 주세요.", 0.0

    detected = sum(1 for f in frames if f["landmarks"] is not None)
    ratio = detected / len(frames)
    if ratio < MIN_FACE_DETECTED_RATIO:
        return (
            False,
            f"영상에서 얼굴이 {round(ratio * 100)}%만 인식돼서 정확한 분석이 어려워요. "
            "얼굴이 화면 안에 들어오고 조명이 밝은지 확인한 뒤 다시 촬영해 주세요.",
            ratio,
        )
    return True, None, ratio
