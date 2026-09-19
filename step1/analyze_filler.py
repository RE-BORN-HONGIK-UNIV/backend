# -*- coding: utf-8 -*-
"""
채움말(filler) 검출 로직

구성 (3갈래 검출 후 병합):
1) 고신뢰 채움말 (어/음/아/에/애) - VERBATIM_PROMPT로 텍스트 복원 유도,
   독립 소리구간이면 채움말로 확정, 다른 단어와 뭉쳐 있으면 제외(연장으로 간주)
2) 음향 기반 채움말 - 소리구간에서 Whisper 단어가 설명 못 하는 잔여 구간 탐지,
   ZCR 개인 기준값으로 음성/숨소리 구분
3) 조건부 채움말 (그/저/뭐/막 등) - Whisper 텍스트 매칭 + 독립 소리구간일 때만 인정

CNN(prolongation/tremor/energy)은 이미 전체 오디오를 슬라이딩 윈도우로 훑고
있으므로, 이 파일이 CNN을 직접 호출하거나 구간을 실시간으로 넘기지는 않는다.
다만 뭉쳐서 채움말 집계에서 제외된 구간은 cnn_prolongation_candidates로
기록해두어, 나중에 CNN 재학습 결과와 "이 구간을 CNN이 실제로 연장으로
잡아냈는지" 수동 대조하는 용도로만 쓴다.

각 임계값의 근거(논문/코퍼스 인용)는 /docs/filler_threshold_근거.md 참고.
"""

import re
import librosa


# -----------------------------------------------------------------------------
# 설정값 (잠정치 - 근거는 /docs/filler_threshold_근거.md 참고)
# -----------------------------------------------------------------------------

ACOUSTIC_MIN_DUR = 0.12    # 음향 기반 채움말 후보 최소 길이 (짧은 잡음 제거용)
LIBROSA_TOP_DB = 30        # 비무음 구간 분리 임계값 (librosa 기본 60dB보다 공격적)

ZCR_TOLERANCE_RATIO = 1.5     # 개인 채움말 ZCR 기준값의 이 배수 이내면 인정
ZCR_FALLBACK_THRESHOLD = 0.15 # 개인 기준값이 없을 때만 쓰는 예비값

HIGH_CONFIDENCE_FILLERS = {"어", "음", "아", "에", "애"}
CONDITIONAL_FILLERS = {"그", "저", "뭐", "막", "좀", "이제", "인제", "이렇게", "일단"}


# -----------------------------------------------------------------------------
# 전사 / 음향 공통 유틸
# -----------------------------------------------------------------------------

def _normalize_word(word: str) -> str:
    """Whisper 출력에서 한글만 남긴다. 조사 붙은 단어는 후보 사전과
    자동으로 안 맞아서 걸러짐(다의어 오탐 1차 방어)."""
    return re.sub(r"[^\uac00-\ud7a3]", "", word).strip()


VERBATIM_PROMPT = (
    "다음은 발표 음성의 축어 전사입니다. 화자가 말하는 중간에 내는 "
    "\"음\", \"어\", \"아\", \"에\" 같은 망설임 소리를 절대 생략하지 말고 "
    "들리는 그대로 표기하세요. "
    "예시: \"음... 그래서 저는\", \"어... 이 프로젝트는\", \"아 그게\". "
    "문장을 매끄럽게 다듬지 말고, 망설임 소리가 들리면 반드시 그 자리에 표기하세요."
)


def get_word_segments(audio_path_or_array, whisper_model):
    """Whisper 전사 결과에서 단어별 text/start/end를 반환한다."""
    result = whisper_model.transcribe(
        audio_path_or_array,
        language="ko",
        word_timestamps=True,
        initial_prompt=VERBATIM_PROMPT,
        temperature=0,
    )

    words = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            text = _normalize_word(word.get("word", ""))
            if text:
                words.append({
                    "text": text,
                    "start": float(word["start"]),
                    "end": float(word["end"]),
                })

    words.sort(key=lambda item: item["start"])
    return words


def get_sound_intervals(y, sr):
    """상대 음량 기준으로 비무음 소리구간을 반환한다."""
    intervals = librosa.effects.split(y, top_db=LIBROSA_TOP_DB)
    return [(start / sr, end / sr) for start, end in intervals]


def _compute_zcr(y, sr, start, end):
    """구간 평균 ZCR을 반환한다. 너무 짧은 구간은 None."""
    segment = y[int(start * sr):int(end * sr)]
    if len(segment) < int(0.02 * sr):
        return None
    zcr = librosa.feature.zero_crossing_rate(segment)[0]
    return float(zcr.mean())


def _overlapping_sound_intervals(word, sound_intervals):
    """단어 시간과 겹치는 비무음 구간들을 반환한다."""
    return [
        interval for interval in sound_intervals
        if interval[0] < word["end"] and interval[1] > word["start"]
    ]


def _is_merged_with_other_words(word, words, sound_intervals):
    """후보 단어가 다른 Whisper 단어와 동일 비무음 구간을 공유하는지 판정.
    공유하면 다음 단어에 이어붙은 연장(prolongation)으로 보고 채움말에서 제외."""
    containing = _overlapping_sound_intervals(word, sound_intervals)
    return any(
        any(
            other is not word
            and other["start"] < interval_end
            and other["end"] > interval_start
            for other in words
        )
        for interval_start, interval_end in containing
    )


# -----------------------------------------------------------------------------
# ZCR 개인 기준값
# -----------------------------------------------------------------------------

def _get_personal_zcr_baseline(y, sr, high_confidence_events):
    """독립 고신뢰 채움말의 평균 ZCR을 이번 녹음의 채움말 기준으로 사용한다."""
    zcrs = [
        z for e in high_confidence_events
        if (z := _compute_zcr(y, sr, e["start"], e["end"])) is not None
    ]
    return sum(zcrs) / len(zcrs) if zcrs else None


def _get_personal_breath_baseline(y, sr, sound_intervals, first_word_start, last_word_end):
    """첫 단어 전/마지막 단어 후 소리에서 숨소리·노이즈 ZCR 기준을 추정한다."""
    zcrs = []
    for start, end in sound_intervals:
        leading = (start, min(end, first_word_start))
        trailing = (max(start, last_word_end), end)
        for part_start, part_end in (leading, trailing):
            if part_end - part_start < ACOUSTIC_MIN_DUR:
                continue
            z = _compute_zcr(y, sr, part_start, part_end)
            if z is not None:
                zcrs.append(z)
    return sum(zcrs) / len(zcrs) if zcrs else None


def _is_voiced(y, sr, start, end, filler_baseline, breath_baseline):
    """후보 구간이 채움말성 목소리에 가까운지 판정한다."""
    zcr = _compute_zcr(y, sr, start, end)
    if zcr is None:
        return False
    if filler_baseline is not None and breath_baseline is not None:
        return abs(zcr - filler_baseline) < abs(zcr - breath_baseline)
    if filler_baseline is not None:
        return zcr < filler_baseline * ZCR_TOLERANCE_RATIO
    return zcr < ZCR_FALLBACK_THRESHOLD


# -----------------------------------------------------------------------------
# 캘리브레이션 (1단계 시작 전 5초 무음 + 10초 낭독 + 1문장 자유발화)
# -----------------------------------------------------------------------------
# 실제 면접 답변 녹음 "안에서" 개인 기준값을 추정하면, 그 녹음에 진짜 채움말이
# 하나도 없거나(개인 채움말 ZCR을 못 구함) 무음 구간이 짧을 때(숨소리 기준값이
# 부정확할 때) 예비값(ZCR_FALLBACK_THRESHOLD)으로 폴백되는 문제가 있었다.
# 답변 녹음 전에 짧은 캘리브레이션 구간을 따로 받으면, 매번 안정적으로
# "이 사람의 배경소음/입력음량/채움말 ZCR" 기준을 확보할 수 있다.
#
# 캘리브레이션 스크립트는 평범한 문장이면 됨 (예: "안녕하세요, 오늘 날씨가
# 좋네요. 편하게 몇 마디 해보세요." 같은 자연스러운 낭독+자유발화) 
# 자유발화 중 자연스럽게 "음/어"가 나오면 개인 채움말 ZCR도 덤으로 잡히고,
# 안 나오면 배경소음/숨소리 기준값만 확보하고 채움말 쪽은 그대로 폴백 처리.
#
# 주의: 이 결과(오각형 차트, 불안도 점수)에는 캘리브레이션 구간 자체를
# 포함하지 않는다 - 어디까지나 기준값 추정용.

CALIBRATION_SILENCE_SEC = 5.0   # 배경소음/입력음량 측정용 무음 구간
CLIPPING_AMPLITUDE = 0.99       # 이 값 이상이면 클리핑(입력 과다)으로 판단


def analyze_calibration(calibration_audio_path, whisper_model=None):
    """캘리브레이션 오디오(5초 무음 + 10초 낭독 + 1문장 자유발화)를 분석해서
    이 사람/이 녹음 환경의 기준값을 뽑아낸다.

    반환값은 analyze_filler()의 calibration 인자로 그대로 넘기면 된다.
    """
    model = whisper_model or _get_default_whisper_model()

    y, sr = librosa.load(calibration_audio_path, sr=16000, mono=True)
    total_dur = len(y) / sr

    # 1) 배경소음 구간 (맨 앞 CALIBRATION_SILENCE_SEC 초로 가정)
    silence_end_sample = int(min(CALIBRATION_SILENCE_SEC, total_dur) * sr)
    silence_segment = y[:silence_end_sample]

    background_rms = float(librosa.feature.rms(y=silence_segment)[0].mean()) if len(silence_segment) else 0.0
    background_peak_db = float(librosa.amplitude_to_db([background_rms])[0]) if background_rms > 0 else -120.0
    background_zcr = _compute_zcr(y, sr, 0.0, silence_end_sample / sr)

    # 2) 클리핑 체크 (전체 파일 기준 - 낭독/자유발화 구간 포함, 입력 장비 문제 조기 발견용)
    clipped_ratio = float((abs(y) >= CLIPPING_AMPLITUDE).mean())

    # 3) 입력 음량 (전체 파일 기준 peak dB)
    overall_peak_db = float(librosa.amplitude_to_db([float(abs(y).max())])[0]) if len(y) else -120.0

    # 4) 낭독+자유발화 구간(무음 이후)에서 개인 채움말 ZCR 추정 (기회가 되면).
    #    스크립트에 "음/어"를 일부러 넣으라고 시키지 않으므로, 자연스럽게
    #    나온 경우에만 잡힘 - 없으면 None 반환되고 호출부(analyze_filler)가
    #    기존처럼 예비값으로 폴백한다.
    speech_start_sec = silence_end_sample / sr
    word_segments = get_word_segments(calibration_audio_path, model)
    speech_words = [w for w in word_segments if w["start"] >= speech_start_sec]
    sound_intervals = get_sound_intervals(y, sr)

    high_confidence_events, _ = detect_high_confidence_fillers(speech_words, sound_intervals)
    personal_filler_zcr = _get_personal_zcr_baseline(y, sr, high_confidence_events)

    return {
        "background_noise_rms": round(background_rms, 6),
        "background_peak_db": round(background_peak_db, 2),
        "background_zcr": round(background_zcr, 4) if background_zcr is not None else None,
        "clipped_ratio": round(clipped_ratio, 4),
        "is_clipping": clipped_ratio > 0.001,  # 전체 샘플의 0.1% 이상이 클리핑이면 경고
        "overall_peak_db": round(overall_peak_db, 2),
        "personal_filler_zcr": round(personal_filler_zcr, 4) if personal_filler_zcr is not None else None,
        "personal_breath_zcr": round(background_zcr, 4) if background_zcr is not None else None,
    }


# -----------------------------------------------------------------------------
# 채움말 검출
# -----------------------------------------------------------------------------

def detect_high_confidence_fillers(words, sound_intervals):
    """어/음/아/에/애 - 독립 소리구간이면 채움말, 다른 단어와 뭉쳐 있으면 제외.
    뭉쳐서 제외된 것은 cnn_prolongation_candidates로 따로 기록한다
    (CNN을 직접 호출하지는 않음 - 나중에 CNN 재학습 결과와 수동 대조하기 위한
    기록용. "이 시간대를 CNN이 실제로 prolongation으로 잡아냈는지" 검증 목적)."""
    events = []
    cnn_candidates = []
    for word in words:
        if word["text"] not in HIGH_CONFIDENCE_FILLERS:
            continue
        if _is_merged_with_other_words(word, words, sound_intervals):
            cnn_candidates.append({
                "text": word["text"], "start": word["start"], "end": word["end"], "source": "high_confidence",
            })
            continue
        events.append({"text": word["text"], "start": word["start"], "end": word["end"], "type": "high"})
    return events, cnn_candidates


def detect_conditional_fillers(words, sound_intervals):
    """그/저/뭐/막 등 - 실제 단어일 수 있으므로 독립 소리구간일 때만 채움말로 인정.
    뭉쳐서 제외된 것도 cnn_prolongation_candidates로 기록 (용도는 위와 동일)."""
    events = []
    cnn_candidates = []
    for word in words:
        if word["text"] not in CONDITIONAL_FILLERS:
            continue
        if _is_merged_with_other_words(word, words, sound_intervals):
            cnn_candidates.append({
                "text": word["text"], "start": word["start"], "end": word["end"], "source": "conditional",
            })
            continue
        events.append({"text": word["text"], "start": word["start"], "end": word["end"], "type": "conditional"})
    return events, cnn_candidates


def _subtract_words_from_interval(interval_start, interval_end, words):
    """소리구간에서 Whisper 단어가 커버하는 시간을 빼고 남은 구간을 반환한다."""
    covering = sorted(
        [
            (max(word["start"], interval_start), min(word["end"], interval_end))
            for word in words
            if word["start"] < interval_end and word["end"] > interval_start
        ],
        key=lambda item: item[0],
    )
    leftover = []
    cursor = interval_start
    for word_start, word_end in covering:
        if word_start > cursor:
            leftover.append((cursor, word_start))
        cursor = max(cursor, word_end)
    if cursor < interval_end:
        leftover.append((cursor, interval_end))
    return leftover


def detect_acoustic_fillers(y, sr, sound_intervals, words, filler_baseline, breath_baseline):
    """Whisper가 설명하지 못한 유성 잔여 구간을 비언어적 채움말 후보로 검출한다."""
    if not words:
        return []

    first_word_start = words[0]["start"]
    last_word_end = words[-1]["end"]
    events = []

    for start, end in sound_intervals:
        clipped_start = max(start, first_word_start)
        clipped_end = min(end, last_word_end)
        if clipped_start >= clipped_end:
            continue

        for leftover_start, leftover_end in _subtract_words_from_interval(clipped_start, clipped_end, words):
            if leftover_end - leftover_start < ACOUSTIC_MIN_DUR:
                continue
            if not _is_voiced(y, sr, leftover_start, leftover_end, filler_baseline, breath_baseline):
                continue
            events.append({
                "text": "(비언어적 채움말)",
                "start": leftover_start,
                "end": leftover_end,
                "type": "acoustic",
            })

    return events


# -----------------------------------------------------------------------------
# 실행 진입점
# -----------------------------------------------------------------------------

_default_whisper_model = None


def _get_default_whisper_model():
    """whisper_model을 받지 않았을 때만 기본 Whisper 모델을 1회 로드한다."""
    global _default_whisper_model
    if _default_whisper_model is None:
        import whisper
        _default_whisper_model = whisper.load_model("base")
    return _default_whisper_model


def analyze_filler(audio_path, whisper_model=None, total_duration_sec=None, calibration=None):
    """채움말 분석 진입점. app.py 호출부와의 시그니처/반환키 호환 유지.

    calibration: analyze_calibration()의 반환값을 그대로 넘기면, 개인 ZCR
    기준값을 이 녹음 안에서 추정하는 대신 캘리브레이션 값을 사용한다.
    없으면(=캘리브레이션 단계가 아직 프론트/라우트에 연결 안 된 경우) 기존처럼
    이 녹음 자체에서 추정 - 하위 호환 유지."""
    model = whisper_model or _get_default_whisper_model()

    word_segments = get_word_segments(audio_path, model)
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    sound_intervals = get_sound_intervals(y, sr)

    conditional_events, conditional_cnn_candidates = detect_conditional_fillers(word_segments, sound_intervals)
    high_confidence_events, high_confidence_cnn_candidates = detect_high_confidence_fillers(word_segments, sound_intervals)
    cnn_prolongation_candidates = sorted(
        conditional_cnn_candidates + high_confidence_cnn_candidates, key=lambda c: c["start"],
    )

    if calibration is not None:
        filler_baseline = calibration.get("personal_filler_zcr")
        breath_baseline = calibration.get("personal_breath_zcr")
    else:
        first_word_start = word_segments[0]["start"] if word_segments else 0.0
        last_word_end = word_segments[-1]["end"] if word_segments else 0.0
        filler_baseline = _get_personal_zcr_baseline(y, sr, high_confidence_events)
        breath_baseline = _get_personal_breath_baseline(y, sr, sound_intervals, first_word_start, last_word_end)

    acoustic_events = detect_acoustic_fillers(
        y, sr, sound_intervals, word_segments, filler_baseline, breath_baseline,
    )

    all_events = sorted(
        conditional_events + high_confidence_events + acoustic_events,
        key=lambda event: event["start"],
    )

    if total_duration_sec is None:
        total_duration_sec = word_segments[-1]["end"] if word_segments else len(y) / sr
    total_duration_sec = max(float(total_duration_sec), 0.001)  # 0초 파일 등 예외 방어

    filler_rate_per_min = len(all_events) / (total_duration_sec / 60.0)
    sound_segment_count = len(sound_intervals)
    filler_ratio = len(all_events) / sound_segment_count if sound_segment_count else 0.0

    # 잠정 점수식 - 실제 서비스 라벨 데이터로 재보정 필요
    if filler_rate_per_min <= 5:
        score = 100 - filler_rate_per_min * 2
    else:
        score = 90 - (filler_rate_per_min - 5) * 6
    score = max(0, min(100, round(score, 1)))

    return {
        "score": score,
        "filler_rate_per_min": round(filler_rate_per_min, 2),
        "total_count": len(all_events),
        "high_confidence_count": sum(1 for e in all_events if e["type"] == "high"),
        "conditional_count": sum(1 for e in all_events if e["type"] == "conditional"),
        "acoustic_count": sum(1 for e in all_events if e["type"] == "acoustic"),
        "personal_filler_zcr": round(filler_baseline, 4) if filler_baseline is not None else None,
        "personal_breath_zcr": round(breath_baseline, 4) if breath_baseline is not None else None,
        "events": all_events,
        "cnn_prolongation_candidates": cnn_prolongation_candidates,  # CNN 미호출, 나중에 수동 대조용 기록
        # app.py 기존 호환 alias
        "total_duration_sec": round(total_duration_sec, 2),
        "filler_count": len(all_events),
        "fluency_score": score,
        "sound_segment_count": sound_segment_count,
        "filler_ratio": round(filler_ratio, 4),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 1:
        print("사용법: python analyze_filler.py <오디오파일경로>")
        raise SystemExit(1)

    result = analyze_filler(sys.argv[1])
    print(f"점수: {result['score']}")
    print(f"개인 채움말 ZCR: {result['personal_filler_zcr']} | 개인 숨소리 ZCR: {result['personal_breath_zcr']}")
    print(f"분당 채움말: {result['filler_rate_per_min']}개")
    print(f"총 {result['total_count']}건 (고신뢰 {result['high_confidence_count']} / "
          f"조건부 {result['conditional_count']} / 음향 기반 {result['acoustic_count']})")
    for event in result["events"]:
        print(f"[{event['type']:>11}] {event['start']:.2f}s ~ {event['end']:.2f}s '{event['text']}'")

    if result["cnn_prolongation_candidates"]:
        print(f"\nCNN 연장 후보 (뭉쳐서 채움말 집계에서 제외됨, {len(result['cnn_prolongation_candidates'])}건):")
        for c in result["cnn_prolongation_candidates"]:
            print(f"  {c['start']:.2f}s ~ {c['end']:.2f}s '{c['text']}' (source={c['source']})")