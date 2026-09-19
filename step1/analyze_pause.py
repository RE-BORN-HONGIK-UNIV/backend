# -*- coding: utf-8 -*-
"""
Step1 · 멈춤(pause) 분석

원칙
- 답변 내부의 무음 구간만 분석한다.
- 0.25초 미만의 무음은 조음·호흡·짧은 발화 전환 가능성이 커 제외한다.
- 0.25초 이상 1.2초 미만의 무음은 일반 멈춤으로 기록한다.
- 1.2초 이상의 무음은 긴 멈춤(long pause)으로 분류해 점수에 반영한다.
- 녹음 시작 전과 종료 후의 침묵은 답변 중 멈춤이 아니므로 제외한다.

근거 요약
- Goldman-Eisler(1968), De Jong & Bosker(2013): 250ms는 조음성 짧은
  공백과 분석 대상 침묵을 구분하는 대표적 하한값이다.
- Frontiers in Psychology(2022): 영어 문장부호 위치의 계획된 휴지에서
  0.6초 및 0.6~1.2초가 자연스러움 평가와 관련됨.
- 자체 정상 발화 무음 463개 분석: 평균 0.342초, 평균+1σ = 0.597초.
  오탐을 줄이는 보수적 기준으로 1.2초 이상을 긴 멈춤으로 설정했다.

주의
- 긴 멈춤은 불안을 단독 진단하는 기준이 아니다.
- Step1의 연장·에너지 변동·떨림·채움말과 결합하는 발화 안정성 보조 지표다.
"""

import librosa


# -----------------------------------------------------------------------------
# 설정값
# -----------------------------------------------------------------------------

MIN_PAUSE_DURATION = 0.25
# 250ms 미만 무음은 조음·호흡·짧은 발화 전환 가능성이 크므로 제외.

LONG_PAUSE_THRESHOLD = 1.2
# 1.2초 이상 발화 내부 무음은 '긴 멈춤'으로 분류.
# 문헌의 계획된 문장 경계 쉼 범위와 자체 정상 발화 무음 분포를 참고한
# 보수적 초기값이며, 실제 서비스 녹음으로 재검증·보정한다.

SILENCE_TOP_DB = 30
# librosa.effects.split의 상대 음량 기반 비무음 구간 분리 기준.
# 파일의 최대 참조 레벨보다 30dB 이상 낮은 구간을 무음으로 본다.
# 채움말 분석과 동일한 기준을 사용한다.


# -----------------------------------------------------------------------------
# 내부 함수
# -----------------------------------------------------------------------------

def _get_internal_pause_events(non_silent_intervals, sr):
    """비무음 구간 사이의 내부 침묵을 멈춤 이벤트로 변환한다.

    첫 비무음 구간 전 / 마지막 비무음 구간 후 침묵은 녹음 시작·종료
    조작 시간일 수 있으므로 여기서 자동으로 제외된다.
    """
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
    """긴 멈춤 빈도와 전체 시간 비율을 이용한 잠정 점수식.

    일반 멈춤(0.25~1.2초)은 자연스러운 생각 정리일 수 있으므로 점수에서
    직접 감점하지 않고 기록만 한다. 긴 멈춤만 강하게 반영한다.

    계수는 표준화된 임상 기준이 아니라 오각형 차트용 잠정 운영값이므로,
    실제 서비스 녹음과 수동 라벨 비교 후 재보정해야 한다.
    """
    burden = (
        2.0 * long_pause_rate_per_min
        + 100.0 * long_pause_ratio
    )

    return round(max(0, min(100, 100 - burden)), 1)


# -----------------------------------------------------------------------------
# 공개 진입점
# -----------------------------------------------------------------------------

def analyze_pause(audio_path, total_duration_sec=None):
    """음성 파일에서 답변 내부 멈춤을 분석한다.

    Args:
        audio_path: 분석할 음성 파일 경로.
        total_duration_sec: 분당 빈도 계산에 쓸 답변 길이. None이면 파일 전체 길이 사용.

    Returns:
        dict:
            score: 오각형 차트용 멈춤 점수(높을수록 안정적)
            pause_count: 0.25초 이상 모든 내부 멈춤 수
            long_pause_count: 1.2초 이상 긴 멈춤 수
            pause_rate_per_min: 모든 분석 대상 멈춤의 분당 빈도
            long_pause_rate_per_min: 긴 멈춤의 분당 빈도
            long_pause_total_sec: 긴 멈춤 총 지속시간
            long_pause_ratio: 전체 답변 시간 중 긴 멈춤 비율
            events: 각 멈춤의 시간 구간·길이·유형
    """
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_duration = librosa.get_duration(y=y, sr=sr)

        total_duration = (
            float(total_duration_sec)
            if total_duration_sec is not None
            else audio_duration
        )
        total_duration = max(total_duration, 0.001)

        non_silent_intervals = librosa.effects.split(
            y,
            top_db=SILENCE_TOP_DB,
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

        pause_rate_per_min = pause_count / (total_duration / 60.0)
        long_pause_rate_per_min = long_pause_count / (total_duration / 60.0)
        long_pause_ratio = long_pause_total_sec / total_duration

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
