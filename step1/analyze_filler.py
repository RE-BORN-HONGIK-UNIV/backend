# -*- coding: utf-8 -*-
"""
채움말(filler) 검출 로직

구성 (3갈래 검출):
1) 고신뢰 채움말 (어/음/아/에/애) - VERBATIM_PROMPT로 텍스트 복원 유도,
   독립 소리구간이면 채움말로 확정, 다른 단어와 뭉쳐 있으면 제외(연장으로 간주)
2) 조건부 채움말 (그/저/뭐/막 등) - Whisper 텍스트 매칭 + 독립 소리구간일 때만 인정
3) 음향 기반 채움말 후보 - 소리구간에서 Whisper 단어가 설명 못 하는 잔여 구간을
   ZCR 개인 기준값으로 "채움말성 목소리"인지 판별

점수/오각형 차트에는 1)+2)(텍스트로 확실히 확인된 것)만 반영한다. 3)은
"acoustic_filler_candidates"로 별도 기록만 하고 점수에서 제외 - 실사용
테스트에서 채움말 없이 녹음해도 문장 사이 숨소리가 오탐되는 문제가
반복 확인됐고, ZCR만으로는 채움말/숨소리를 안정적으로 못 가른다는 게
확인됐기 때문(근거는 /docs/filler_threshold_근거.md 참고). 완전히
버리진 않는 이유: 캘리브레이션(개인 채움말/숨소리 ZCR 측정)이 계속
의미를 갖게 하고, 정확도가 개선되면 다시 점수에 편입할 수 있도록
검출 로직과 기록은 살려둔다.

CNN(prolongation/tremor/energy)은 이미 전체 오디오를 슬라이딩 윈도우로 훑고
있으므로, 이 파일이 CNN을 직접 호출하거나 구간을 실시간으로 넘기지는 않는다.
다만 뭉쳐서 채움말 집계에서 제외된 구간(1, 2번에서)은 cnn_prolongation_candidates로
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
# 캘리브레이션 (1단계 시작 전 5초 무음 + 낭독)
# -----------------------------------------------------------------------------

CALIBRATION_SILENCE_SEC = 5.0   # 배경소음/입력음량 측정용 무음 구간
CLIPPING_AMPLITUDE = 0.99       # 이 값 이상이면 클리핑(입력 과다)으로 판단


def analyze_calibration(calibration_audio_path, whisper_model=None):
    """캘리브레이션 오디오(무음 + 낭독 + 자유발화)를 분석해서 이 사람/이
    녹음 환경의 기준값을 뽑아낸다. analyze_filler()의 calibration 인자로
    그대로 넘기면 된다 (음향 기반 후보 판별의 개인화에 쓰임)."""
    model = whisper_model or _get_default_whisper_model()

    y, sr = librosa.load(calibration_audio_path, sr=16000, mono=True)
    total_dur = len(y) / sr

    silence_end_sample = int(min(CALIBRATION_SILENCE_SEC, total_dur) * sr)
    silence_segment = y[:silence_end_sample]

    background_rms = float(librosa.feature.rms(y=silence_segment)[0].mean()) if len(silence_segment) else 0.0
    background_peak_db = float(librosa.amplitude_to_db([background_rms])[0]) if background_rms > 0 else -120.0
    background_zcr = _compute_zcr(y, sr, 0.0, silence_end_sample / sr)

    clipped_ratio = float((abs(y) >= CLIPPING_AMPLITUDE).mean())
    overall_peak_db = float(librosa.amplitude_to_db([float(abs(y).max())])[0]) if len(y) else -120.0

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
        "is_clipping": clipped_ratio > 0.001,
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


def detect_acoustic_filler_candidates(y, sr, sound_intervals, words, filler_baseline, breath_baseline):
    """Whisper가 설명하지 못한 유성 잔여 구간을 채움말 '후보'로 검출한다.
    점수에는 반영 안 되고 기록만 됨 (파일 상단 설명 참고)."""
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
                "text": "(비언어적 채움말 후보)",
                "start": leftover_start,
                "end": leftover_end,
                "type": "acoustic_candidate",
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

    점수(score/total_count/filler_rate_per_min)는 텍스트로 확실히 확인된
    채움말(고신뢰+조건부)만 반영한다. 음향 기반 후보는 별도로
    acoustic_filler_candidates에 기록되고 점수에는 안 들어간다.

    calibration: analyze_calibration()의 반환값을 넘기면, 음향 기반 후보
    판별(개인 ZCR 기준값)에 사용한다. 없으면 이 녹음 자체에서 추정."""
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

    acoustic_filler_candidates = detect_acoustic_filler_candidates(
        y, sr, sound_intervals, word_segments, filler_baseline, breath_baseline,
    )

    # 점수에 반영되는 건 이 목록뿐 - 음향 기반 후보는 안 섞는다.
    scored_events = sorted(
        conditional_events + high_confidence_events,
        key=lambda event: event["start"],
    )

    if total_duration_sec is None:
        total_duration_sec = word_segments[-1]["end"] if word_segments else len(y) / sr
    total_duration_sec = max(float(total_duration_sec), 0.001)  # 0초 파일 등 예외 방어

    filler_rate_per_min = len(scored_events) / (total_duration_sec / 60.0)
    sound_segment_count = len(sound_intervals)
    filler_ratio = len(scored_events) / sound_segment_count if sound_segment_count else 0.0

    # 잠정 점수식 - 실제 서비스 라벨 데이터로 재보정 필요.
    # 3단계 구간(0~5/5~12/12+개 분당)으로 나눠 완만하게 감점.
    if filler_rate_per_min <= 5:
        score = 100 - filler_rate_per_min * 2
    elif filler_rate_per_min <= 12:
        score = 90 - (filler_rate_per_min - 5) * 3
    else:
        score = 69 - (filler_rate_per_min - 12) * 4
    score = max(0, min(100, round(score, 1)))

    return {
        "score": score,
        "filler_rate_per_min": round(filler_rate_per_min, 2),
        "total_count": len(scored_events),
        "high_confidence_count": sum(1 for e in scored_events if e["type"] == "high"),
        "conditional_count": sum(1 for e in scored_events if e["type"] == "conditional"),
        "acoustic_count": 0,  # 점수 기준 개수 - 음향 기반은 점수에 안 들어가므로 항상 0
        "events": scored_events,
        "acoustic_filler_candidates": acoustic_filler_candidates,  # 점수 미반영, 기록용
        "cnn_prolongation_candidates": cnn_prolongation_candidates,  # 점수 미반영, 기록용
        "personal_filler_zcr": round(filler_baseline, 4) if filler_baseline is not None else None,
        "personal_breath_zcr": round(breath_baseline, 4) if breath_baseline is not None else None,
        # app.py 기존 호환 alias
        "total_duration_sec": round(total_duration_sec, 2),
        "filler_count": len(scored_events),
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
    print(f"점수(확실한 채움말 기준): {result['score']}")
    print(f"개인 채움말 ZCR: {result['personal_filler_zcr']} | 개인 숨소리 ZCR: {result['personal_breath_zcr']}")
    print(f"분당 채움말: {result['filler_rate_per_min']}개")
    print(f"총 {result['total_count']}건 (고신뢰 {result['high_confidence_count']} / "
          f"조건부 {result['conditional_count']})")
    for event in result["events"]:
        print(f"[{event['type']:>11}] {event['start']:.2f}s ~ {event['end']:.2f}s '{event['text']}'")

    if result["acoustic_filler_candidates"]:
        print(f"\n음향 기반 채움말 후보 (점수 미반영, {len(result['acoustic_filler_candidates'])}건):")
        for c in result["acoustic_filler_candidates"]:
            print(f"  {c['start']:.2f}s ~ {c['end']:.2f}s")

    if result["cnn_prolongation_candidates"]:
        print(f"\nCNN 연장 후보 (뭉쳐서 채움말 집계에서 제외됨, {len(result['cnn_prolongation_candidates'])}건):")
        for c in result["cnn_prolongation_candidates"]:
            print(f"  {c['start']:.2f}s ~ {c['end']:.2f}s '{c['text']}' (source={c['source']})")