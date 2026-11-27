"""1단계 · 멈춤 분석 — 답변 안의 무음을 찾아 0.25~1.2초는 일반 멈춤, 1.2초 이상은 긴 멈춤으로 센다.
녹음 앞뒤 침묵은 빼고, 긴 멈춤만으로 불안을 판단하지 않고 다른 습관과 함께 본다.
기준값 근거: docs/pause_threshold_근거.md (문헌은 docs/REFERENCES.md)
"""

import librosa


# -----------------------------------------------------------------------------
# 설정값
# -----------------------------------------------------------------------------

MIN_PAUSE_DURATION = 0.25
# 0.25초 미만 무음은 발음·숨 고르기 사이라 멈춤으로 세지 않는다

LONG_PAUSE_THRESHOLD = 1.2
# 문헌의 문장 경계 쉼과 정상 발화 무음 분포를 보고 오탐이 적게 잡은 잠정값

MIN_RATE_DURATION_SEC = 30.0
# 분당 빈도의 최소 분모 — 5초 답의 멈춤 한 번이 분당 12번으로 부풀지 않게

SILENCE_TOP_DB = 30
# 가장 큰 소리보다 30dB 이상 작으면 무음으로 본다(채움말 분석과 같은 기준)


# -----------------------------------------------------------------------------
# 내부 함수
# -----------------------------------------------------------------------------

def _get_internal_pause_events(non_silent_intervals, sr):
    """소리 구간 사이의 침묵을 멈춤 이벤트로 바꾼다(맨 앞·맨 뒤 침묵은 녹음 조작 시간이라 자연히 빠진다)."""
    events = []

    if len(non_silent_intervals) < 2:
        return events

    for index in range(1, len(non_silent_intervals)):
        previous_end = non_silent_intervals[index - 1][1] / sr
        current_start = non_silent_intervals[index][0] / sr
        duration = current_start - previous_end

        if duration < MIN_PAUSE_DURATION:
            continue

        pause_type = (
            "long_pause"
            if duration >= LONG_PAUSE_THRESHOLD
            else "pause"
        )

        events.append({
            "start": round(previous_end, 3),
            "end": round(current_start, 3),
            "duration": round(duration, 3),
            "type": pause_type,
        })

    return events


def _calculate_pause_score(long_pause_rate_per_min, long_pause_ratio):
    """긴 멈춤의 분당 횟수와 시간 비율로 점수를 낸다(일반 멈춤은 생각 정리일 수 있어 깎지 않는 잠정 공식)."""
    burden = (
        2.0 * long_pause_rate_per_min
        + 100.0 * long_pause_ratio
    )

    return round(max(0, min(100, 100 - burden)), 1)


# -----------------------------------------------------------------------------
# 공개 진입점
# -----------------------------------------------------------------------------

def analyze_pause(audio_path, total_duration_sec=None, calibration=None):
    """녹음 하나의 멈춤 점수·개수·분당 빈도·구간 목록을 돌려준다. 실패하면 모든 값이 0인 결과."""
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_duration = librosa.get_duration(y=y, sr=sr)

        total_duration = (
            float(total_duration_sec)
            if total_duration_sec is not None
            else audio_duration
        )
        total_duration = max(total_duration, 0.001)

        # 시끄러운 곳이면 무음 기준을 5dB 늘려 배경소음을 말소리로 보지 않게 한다(잠정값)
        top_db = SILENCE_TOP_DB
        if calibration and calibration.get("background_peak_db", -120) > -40:
            top_db = SILENCE_TOP_DB + 5

        non_silent_intervals = librosa.effects.split(
            y,
            top_db=top_db,
        )

        events = _get_internal_pause_events(non_silent_intervals, sr)
        long_pause_events = [
            event for event in events
            if event["type"] == "long_pause"
        ]

        pause_count = len(events)
        long_pause_count = len(long_pause_events)
        long_pause_total_sec = sum(
            event["duration"] for event in long_pause_events
        )

        # 짧은 답에서 멈춤 한 번이 크게 부풀지 않게 최소 30초로 나눈다
        rate_duration = max(total_duration, MIN_RATE_DURATION_SEC)
        pause_rate_per_min = pause_count / (rate_duration / 60.0)
        long_pause_rate_per_min = long_pause_count / (rate_duration / 60.0)
        long_pause_ratio = long_pause_total_sec / rate_duration

        score = _calculate_pause_score(
            long_pause_rate_per_min=long_pause_rate_per_min,
            long_pause_ratio=long_pause_ratio,
        )

        return {
            # 통합 분석 / 화면용 키
            "score": score,
            "pause_count": pause_count,
            "long_pause_count": long_pause_count,
            "pause_rate_per_min": round(pause_rate_per_min, 2),
            "long_pause_rate_per_min": round(long_pause_rate_per_min, 2),
            "long_pause_total_sec": round(long_pause_total_sec, 2),
            "long_pause_ratio": round(long_pause_ratio, 4),
            "total_duration_sec": round(total_duration, 2),
            "events": events,

            # 기존 app.py 호환용 alias
            "pause_score": score,
            "anxious_pause_count": long_pause_count,
            "anxious_pause_total_sec": round(long_pause_total_sec, 2),
            "anxious_pause_ratio": round(long_pause_ratio, 4),
        }

    except Exception as error:
        print(f"[ERROR] pause 분석 실패: {error}")

        return {
            "score": 0,
            "pause_count": 0,
            "long_pause_count": 0,
            "pause_rate_per_min": 0,
            "long_pause_rate_per_min": 0,
            "long_pause_total_sec": 0,
            "long_pause_ratio": 0,
            "total_duration_sec": 0,
            "events": [],
            "pause_score": 0,
            "anxious_pause_count": 0,
            "anxious_pause_total_sec": 0,
            "anxious_pause_ratio": 0,
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) <= 1:
        print("사용법: python analyze_pause.py <오디오파일경로>")
        raise SystemExit(1)

    result = analyze_pause(sys.argv[1])

    print(f"점수: {result['score']}")
    print(
        f"전체 멈춤: {result['pause_count']}건 | "
        f"긴 멈춤: {result['long_pause_count']}건"
    )
    print(
        f"분당 멈춤: {result['pause_rate_per_min']}건 | "
        f"분당 긴 멈춤: {result['long_pause_rate_per_min']}건"
    )

    for event in result["events"]:
        print(
            f"[{event['type']:>10}] "
            f"{event['start']:.2f}s ~ {event['end']:.2f}s "
            f"({event['duration']:.2f}s)"
        )
