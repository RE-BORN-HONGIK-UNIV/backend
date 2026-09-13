def score_blink_rate(blink_count, duration_sec):
    """분당 깜빡임 정상 범위 근거 (2025-09 재검토):
    - Bentivoglio et al. (1997, Movement Disorders) — 안정(rest) 상태 평균 17회/분,
      **대화 중엔 평균 26회/분로 증가**, 범위 10.5~32.5회/분.
    - Cluster analysis 기반 후속 연구(PMID 17099391) — 개인차가 2.8~48회/분까지 벌어짐 보고.
    2단계 영상은 카메라를 보고 말하는, 안정 상태보다 대화 상황에 가까우므로
    대화 중 관찰 범위(10.5~32.5)를 반올림해 기준으로 삼는다 (이전엔 안정 상태 기준 12~20을
    그대로 썼는데, 그러면 정상적인 대화 중 깜빡임도 '빈번'으로 오판정할 위험이 있었음).
    다만 개인차가 워낙 커서 이 10/30 컷오프도 정밀한 임상 기준이 아니라 느슨한 휴리스틱 —
    팀 자체 라벨링 데이터로 검증 전까지는 참고용으로만 취급할 것."""
    rate_per_min = blink_count / (duration_sec / 60)
    if rate_per_min > 30:
        status = "빈번"
        score = max(0, 100 - (rate_per_min - 30) * 5)
    elif rate_per_min < 10:
        status = "과응시"
        score = max(0, 100 - (10 - rate_per_min) * 5)
    else:
        status = "정상"
        score = 100
    return {"rate_per_min": round(rate_per_min, 1), "status": status, "score": round(score, 1)}


def score_gaze_segment(duration_sec):
    """3~5초를 이상적 구간으로 보는 근거: Binetti et al. (2016, Royal Society Open Science,
    런던 과학박물관 방문객 약 500명 대상 실험) — 선호 응시 지속시간 평균 3.2~3.3초,
    선호 범위 2~5초, 1초 미만·9초 초과는 아무도 선호하지 않음.
    주의: 이 연구는 '상대가 나를 바라보는 시간'에 대한 선호도를 측정한 것이라,
    이 앱처럼 '카메라(상대)를 내가 바라보는' 상황과는 방향이 반대다 — 방향은 다르지만
    참고할 수 있는 유일한 정량 근거라 우선 채택. 정확한 매칭은 아니므로 향후 자체 데이터로 검증 필요."""
    if duration_sec < 3:
        return duration_sec / 3 * 70
    if duration_sec <= 5:
        return 100
    return max(50, 100 - (duration_sec - 5) * 10)


def score_gaze_segments(segments):
    fixation_segs = [s for s in segments if s["type"] == "fixation"]
    durations = [s["end"] - s["start"] for s in fixation_segs]
    if not durations:
        return {"avg_fixation_sec": 0, "score": 0}
    scores = [score_gaze_segment(d) for d in durations]
    return {
        "avg_fixation_sec": round(sum(durations) / len(durations), 2),
        "score": round(sum(scores) / len(scores), 1),
    }


def score_expression(smile_ratio, tension_ratio):
    """미소는 많을수록, 긴장(찡그림)은 적을수록 좋은 점수로 환산.
    주의: smile_ratio/tension_ratio를 만들어내는 SMILE_THRESHOLD/TENSION_THRESHOLD
    (expression_analyzer.py)는 심리학 문헌 근거가 아니라 MediaPipe blendshape 점수 스케일에 대한
    엔지니어링 컷오프다 — "근거자료"라고 부를 수 있는 값이 아니므로, 팀 라벨링 데이터로 캘리브레이션
    필요 (이 스코어 공식의 계수 250/200도 같은 이유로 잠정치)."""
    smile_score = min(100, smile_ratio * 250)  # 미소 프레임 비율 40%면 만점
    tension_score = max(0, 100 - tension_ratio * 200)  # 긴장 프레임 비율 50%면 0점

    if tension_ratio >= 0.4:
        status = "긴장됨"
    elif smile_ratio >= 0.2:
        status = "편안함"
    else:
        status = "보통"

    return {
        "smile_score": round(smile_score, 1),
        "tension_score": round(tension_score, 1),
        "score": round((smile_score + tension_score) / 2, 1),
        "status": status,
    }