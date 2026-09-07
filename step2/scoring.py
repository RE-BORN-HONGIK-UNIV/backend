def score_blink_rate(blink_count, duration_sec):
    rate_per_min = blink_count / (duration_sec / 60)
    if rate_per_min > 20:
        status = "빈번"
        score = max(0, 100 - (rate_per_min - 20) * 5)
    elif rate_per_min < 12:
        status = "과응시"
        score = max(0, 100 - (12 - rate_per_min) * 5)
    else:
        status = "정상"
        score = 100
    return {"rate_per_min": round(rate_per_min, 1), "status": status, "score": round(score, 1)}


def score_gaze_segment(duration_sec):
    """3~5초를 이상적 구간으로 보는 근거자료 기준"""
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
    """미소는 많을수록, 긴장(찡그림)은 적을수록 좋은 점수로 환산."""
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