# -*- coding: utf-8 -*-
"""
채움말(filler) 검출 로직 v3

v2에서 바뀐 점:
- "어/음/아/에/애" 같은 비언어적 간투사는 Whisper base 모델이 텍스트로
  아예 안 옮기는 경우가 많다는 게 실제 테스트로 확인됨 (STT 자체가 이런
  disfluency를 지우는 경향이 있음 -> 텍스트 매칭으로는 원천적으로 못 잡음)
- 그래서 이 그룹은 텍스트 매칭을 버리고, "소리는 나는데 Whisper 단어로
  설명이 안 되는 시간 구간"을 직접 찾는 방식(음향 기반)으로 전환
- "그/저/뭐/막" 등 단어형은 여전히 텍스트 매칭 사용 (실제 단어라 Whisper가
  인식할 가능성이 있음 - 단, 이것도 실제 데이터로 별도 검증 필요)
- 근거 자료(논문/사전 출처)는 /docs/filler_word_list_근거.md 참고
"""

import re
import librosa


# -----------------------------------------------------------------------------
# 설정값 (전부 잠정치 - 실제 데이터로 보정 예정)
# -----------------------------------------------------------------------------

MIN_GAP_SEC = 0.15         # 조건부(단어형) 채움말: 이보다 작으면 다음 단어와 이어짐 -> 연장으로 판단
SEPARATION_GAP_SEC = 0.6   # 조건부(단어형) 채움말: 이보다 크면 확실히 분리됨

REPEAT_WINDOW_SEC = 6.0    # 조건부 채움말의 "반복" 판정 시간 창
REPEAT_MIN_COUNT = 2       # 이 창 안에서 몇 번 이상 나오면 반복으로 볼지

ACOUSTIC_MIN_DUR = 0.12    # 음향 기반(고신뢰) 채움말 최소 길이 - 너무 짧으면 잡음/클릭 소리일 수 있어 제외
LIBROSA_TOP_DB = 30        # librosa 무음 판정 민감도 (기존 diagnose_top_db.py로 재검증 권장)

# 조건부(단어형) 채움말만 텍스트 매칭 대상.
CONDITIONAL_FILLERS = {"그", "저", "뭐", "막", "좀", "이제", "인제", "이렇게", "일단"}

# 고신뢰 채움말 - VERBATIM_PROMPT 적용 후 Whisper가 실제로 텍스트로 옮기기
# 시작한 게 확인됨. "그/저"와 달리 다른 뜻으로 쓰일 위험이 없는 순수 간투사라,
# 단어로 인식됐다는 사실 자체가 충분한 증거 -> gap/반복 조건 없이 바로 카운트.
HIGH_CONFIDENCE_FILLERS = {"어", "음", "아", "에", "애"}


def _normalize_word(word: str) -> str:
    """Whisper 출력에서 한글만 남기고 공백/문장부호 제거.
    조사 붙은 단어("그가", "저는")는 후보 사전과 자동으로 안 맞아서
    걸러짐 -> 다의어 오탐 1차 방어."""
    return re.sub(r"[^\uac00-\ud7a3]", "", word).strip()


# Whisper에게 "음, 어 같은 비언어적 간투사를 지우지 말고 그대로 받아써라"라고
# 유도하는 프롬프트. 우리가 실제로 필요한 건 비언어적 간투사 보존뿐이라,
# 범위를 여기로 좁힘 (반복/끊김까지 요청하면 불필요한 노이즈가 늘어날 수 있음).
# 완전한 해결책은 아니고 누락률을 줄이는 보조 수단.
VERBATIM_PROMPT = (
    "다음은 발표 음성의 축어 전사입니다. 화자가 말하는 중간에 내는 "
    "\"음\", \"어\", \"아\", \"에\" 같은 망설임 소리를 절대 생략하지 말고 "
    "들리는 그대로 표기하세요. "
    "예시: \"음... 그래서 저는\", \"어... 이 프로젝트는\", \"아 그게\". "
    "문장을 매끄럽게 다듬지 말고, 망설임 소리가 들리면 반드시 그 자리에 표기하세요."
)


def get_word_segments(audio_path_or_array, whisper_model):
    """Whisper로 STT 실행 후 단어 단위 (text, start, end) 리스트 반환."""
    result = whisper_model.transcribe(
        audio_path_or_array,
        language="ko",
        word_timestamps=True,
        initial_prompt=VERBATIM_PROMPT,
        temperature=0,  # 출력 형식이 실행마다 흔들리는 것 방지
    )
    words = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            text = _normalize_word(w.get("word", ""))
            if text:
                words.append({"text": text, "start": float(w["start"]), "end": float(w["end"])})
    words.sort(key=lambda x: x["start"])
    return words


def get_sound_intervals(y, sr):
    """librosa로 소리구간(침묵 아닌 구간)을 전부 나눈다."""
    intervals = librosa.effects.split(y, top_db=LIBROSA_TOP_DB)
    return [(s / sr, e / sr) for s, e in intervals]


def _is_voiced(y, sr, start, end, zcr_threshold=0.15):
    """구간이 '목소리(모음성 소리)'인지 '숨소리/잡음'인지 판별.
    영교차율(zero-crossing rate)이 낮으면 파형이 규칙적인 유성음
    ("음", "어" 등 모음성 채움말)일 가능성이 높고, 높으면 숨소리 같은
    잡음성 소리일 가능성이 높다. 완벽하진 않은 휴리스틱이라 오탐이
    남을 수 있음 - 실제 데이터로 zcr_threshold 재조정 필요."""
    start_i, end_i = int(start * sr), int(end * sr)
    seg = y[start_i:end_i]
    if len(seg) < int(0.02 * sr):
        return False
    zcr = librosa.feature.zero_crossing_rate(seg)[0]
    return float(zcr.mean()) < zcr_threshold


def detect_high_confidence_fillers(words):
    """음/어/아/에/애 - VERBATIM_PROMPT 덕분에 Whisper가 독립된 단어로
    인식한 경우, 조건 없이 바로 채움말로 카운트. 이 단어들이 words 목록에
    있다는 것 자체가 _subtract_words_from_interval에서 "커버된 시간"으로
    처리되므로, 음향 기반 검출(detect_acoustic_fillers)과 중복되지 않는다."""
    return [
        {"text": w["text"], "start": w["start"], "end": w["end"], "type": "high"}
        for w in words if w["text"] in HIGH_CONFIDENCE_FILLERS
    ]


def detect_conditional_fillers(words):
    """단어형 채움말(그/저/뭐/막 등)을 Whisper 텍스트에서 찾는다.
    1회 등장은 정상적인 화용 기능일 수 있으므로, 반복되거나 뒤에 긴 침묵이
    붙을 때만 채움말로 카운트. 다음 단어와 바로 이어지면 연장으로 보고 제외."""
    events = []
    recent_occurrences = {}  # {단어: [최근 등장 시각들]}

    for i, w in enumerate(words):
        text = w["text"]
        if text not in CONDITIONAL_FILLERS:
            continue

        next_gap = words[i + 1]["start"] - w["end"] if i + 1 < len(words) else None
        if next_gap is not None and next_gap < MIN_GAP_SEC:
            continue  # 다음 단어와 바로 이어짐 -> 연장으로 보고 제외

        recent_occurrences.setdefault(text, [])
        recent_occurrences[text] = [t for t in recent_occurrences[text] if w["start"] - t <= REPEAT_WINDOW_SEC]
        recent_occurrences[text].append(w["start"])

        is_repeated = len(recent_occurrences[text]) >= REPEAT_MIN_COUNT
        is_long_pause_after = next_gap is not None and next_gap >= SEPARATION_GAP_SEC

        if is_repeated or is_long_pause_after:
            events.append({"text": text, "start": w["start"], "end": w["end"], "type": "conditional"})

    return events


def _subtract_words_from_interval(interval_start, interval_end, words):
    """소리구간 하나(interval_start~interval_end)에서, 그 안에 겹치는
    Whisper 단어들이 차지하는 시간을 전부 빼고 남는 시간 구간들을 반환.
    이 "남는 시간"이 Whisper가 텍스트로 설명 못 한 소리 = 채움말 후보.

    예: 소리구간 4.48~6.56에 단어 '전공하고'(5.20~5.74), '있습니다'(5.74~6.32)만
    겹친다면 -> 남는 구간은 [4.48~5.20](앞쪽 안 잡힌 부분), [6.32~6.56](뒤쪽) 이런 식.
    """
    covering = sorted(
        [(max(w["start"], interval_start), min(w["end"], interval_end))
         for w in words if w["start"] < interval_end and w["end"] > interval_start],
        key=lambda x: x[0],
    )

    leftover = []
    cursor = interval_start
    for w_start, w_end in covering:
        if w_start > cursor:
            leftover.append((cursor, w_start))
        cursor = max(cursor, w_end)
    if cursor < interval_end:
        leftover.append((cursor, interval_end))

    return leftover


def detect_acoustic_fillers(y, sr, sound_intervals, words):
    """고신뢰 채움말(어/음/아/에/애 등 비언어적 간투사)을 텍스트 매칭 없이
    순수 음향으로 찾는다. 각 소리구간에서 Whisper 단어가 커버하는 시간을
    빼고 남는(=아무 단어로도 설명 안 되는) 부분을 채움말 후보로 본다.

    기존 v2의 "구간에 단어가 하나라도 겹치면 통째로 제외" 방식과 달리,
    여러 단어가 한 덩어리로 뭉친 소리구간이어도 그 안에 숨은 채움말을
    부분적으로 찾아낼 수 있다.

    필터 2단계:
    1) 첫 단어 시작 전 / 마지막 단어 끝난 후 구간 제외 (말 시작 전 숨소리,
       녹음 종료 시점 마이크 노이즈로 확인됨)
    2) 유성음(모음성) 판별 통과한 것만 채움말로 인정 (문장 사이 숨소리 제외 시도)

    남은 한계: 이 필터들을 거쳐도 숨소리와 "음/어"가 완전히 구분되진
    않음 - 오탐 가능성 있는 잠정 로직.
    """
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

        leftover_spans = _subtract_words_from_interval(clipped_start, clipped_end, words)
        for l_start, l_end in leftover_spans:
            dur = l_end - l_start
            if dur < ACOUSTIC_MIN_DUR:
                continue
            if not _is_voiced(y, sr, l_start, l_end):
                continue
            events.append({
                "text": "(비언어적 채움말)",
                "start": l_start,
                "end": l_end,
                "type": "acoustic",
            })
    return events


_default_whisper_model = None


def _get_default_whisper_model():
    """whisper_model 인자 없이 단독 실행될 때만 쓰는 자체 로딩."""
    global _default_whisper_model
    if _default_whisper_model is None:
        import whisper
        _default_whisper_model = whisper.load_model("base")
    return _default_whisper_model


def analyze_filler(audio_path, whisper_model=None, total_duration_sec=None):
    """app.py 진입점. 기존 함수명/시그니처 유지."""
    model = whisper_model or _get_default_whisper_model()

    word_segments = get_word_segments(audio_path, model)

    y, sr = librosa.load(audio_path, sr=16000)
    sound_intervals = get_sound_intervals(y, sr)

    conditional_events = detect_conditional_fillers(word_segments)
    high_confidence_events = detect_high_confidence_fillers(word_segments)
    acoustic_events = detect_acoustic_fillers(y, sr, sound_intervals, word_segments)
    all_events = sorted(conditional_events + high_confidence_events + acoustic_events, key=lambda e: e["start"])

    if total_duration_sec is None:
        total_duration_sec = word_segments[-1]["end"] if word_segments else 1.0

    filler_rate_per_min = len(all_events) / (total_duration_sec / 60.0)

    sound_segment_count = len(sound_intervals)
    filler_ratio = len(all_events) / sound_segment_count if sound_segment_count > 0 else 0

    # 점수 산출 (잠정 로직, 실제 데이터로 재보정 필요)
    if filler_rate_per_min <= 5:
        score = 100 - filler_rate_per_min * 2
    else:
        score = 90 - (filler_rate_per_min - 5) * 6
    score = max(0, min(100, round(score, 1)))

    return {
        # -- 새 버전 키 --
        "score": score,
        "filler_rate_per_min": round(filler_rate_per_min, 2),
        "total_count": len(all_events),
        "high_confidence_count": sum(1 for e in all_events if e["type"] == "high"),
        "conditional_count": sum(1 for e in all_events if e["type"] == "conditional"),
        "acoustic_count": sum(1 for e in all_events if e["type"] == "acoustic"),
        "events": all_events,
        # -- app.py 기존 코드 호환용 alias --
        "total_duration_sec": round(total_duration_sec, 2),
        "filler_count": len(all_events),
        "fluency_score": score,
        "sound_segment_count": sound_segment_count,
        "filler_ratio": round(filler_ratio, 4),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = analyze_filler(sys.argv[1])
        print(f"점수: {result['score']}")
        print(f"분당 채움말: {result['filler_rate_per_min']}개")
        print(f"총 {result['total_count']}건 (고신뢰 {result['high_confidence_count']} / "
              f"조건부 {result['conditional_count']} / 음향 기반 {result['acoustic_count']})")
        for e in result["events"]:
            print(f"[{e['type']:>11}] {e['start']:.2f}s ~ {e['end']:.2f}s '{e['text']}'")
    else:
        print("사용법: python analyze_filler_final.py <오디오파일경로>")