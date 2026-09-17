# -*- coding: utf-8 -*-
"""
채움말(filler) 검출 로직

구성 (3갈래로 검출 후 합침):
1) 고신뢰 채움말 (어/음/아/에/애)
   - VERBATIM_PROMPT로 Whisper를 유도해서 텍스트로 옮기게 함 (기본값으론
     Whisper가 이런 비언어적 간투사를 텍스트에서 통째로 생략하는 경우가
     많다는 게 실측으로 확인됨 -> 프롬프트로 일부 복원)
   - 프롬프트로도 복원 안 된 나머지는 음향 기반(2번)이 보완
2) 음향 기반 채움말 (1번이 텍스트로 못 건진 것들)
   - 소리는 나는데 Whisper 단어로 설명 안 되는 시간 구간을 직접 찾음
   - 단, 숨소리도 이 방식에 걸리므로 유성음 판별(ZCR) 필터를 거침 -
     고정 임계값 대신 이번 녹음 "안에서" 뽑은 개인 기준값(채움말 ZCR +
     숨소리 ZCR)으로 최근접 분류 (사람/마이크마다 다른 특성을 자동 반영)
3) 조건부 채움말 (그/저/뭐/막 등 단어형)
   - 실제 단어라 Whisper가 인식하므로 텍스트 매칭 사용
   - 판정은 librosa 소리구간 기준: 이 단어가 자기 혼자만의 소리구간을
     차지하면 분리된 채움말로 인정, 다른 단어와 같은 구간에 뭉쳐있으면
     연장/이어짐으로 보고 제외 (Whisper 자체 단어 간 gap은 부정확해서
     신뢰 안 함 - 실측으로 확인됨)

남은 한계: 숨소리와 채움말이 ZCR만으로 완전히 구분되진 않음 - 특히
채움말 기준값 확보 전(이번 녹음에 텍스트로 잡힌 어/음이 하나도 없을
때)은 예비값(고정 임계값)으로 폴백되어 정확도가 떨어짐.
근거 자료(논문/사전 출처)는 /docs/filler_word_list_근거.md 참고.
"""

import re
import librosa


# -----------------------------------------------------------------------------
# 설정값 (전부 잠정치 - 실제 데이터로 보정 예정)
# -----------------------------------------------------------------------------

# MIN_GAP_SEC / SEPARATION_GAP_SEC는 더 이상 안 씀 - Whisper 단어 간 gap이
# 부정확한 것으로 확인되어(항상 0으로 나오는 경우 다수), 조건부 채움말 판정을
# librosa 소리구간 기반으로 전환함 (detect_conditional_fillers 참고).

ACOUSTIC_MIN_DUR = 0.12    # 음향 기반(고신뢰) 채움말 최소 길이
# 근거: 서울코퍼스 기반 "담화표지 '아,어,음'의 성별과 연령별 사용 양상"(2020) -
# 한국어 '아/어/음' 평균 지속시간은 238/348/453ms, 단 최단 '어'는 19ms까지 관측됨.
# 즉 0.12초는 평균보다 훨씬 짧은 값 -> "전형적 채움말의 최소치"가 아니라
# "클릭음 등 명백한 잡음을 걸러내는 보수적 필터"로 봐야 함. 극단적으로 짧은
# 진짜 채움말(19ms대)은 이 필터에 걸러질 수 있음 - 알려진 한계로 남겨둠.
LIBROSA_TOP_DB = 30        # librosa 무음 판정 민감도
# 주의: librosa 기본값은 60dB인데 30dB로 더 공격적으로 잡아놓은 상태 ->
# 약하게 발음된 "음/어"까지 무음으로 잘려나갈 위험 있음. 아직 실측 재검증
# 안 됨 (diagnose_top_db.py로 확인 필요).

# 유성음(목소리) 판별 - 고정된 전역 임계값 대신, "이번 녹음에서 실제로 확인된
# 이 사람의 채움말 ZCR"을 기준으로 상대 비교한다 (사람/마이크/환경마다 숨소리
# 특성이 달라 고정값은 일반화가 안 됨 - 실측으로 확인됨).
# 한계 (학술 근거로도 확인됨): Rabiner & Sambur(1977) 등 고전 음성신호처리
# 연구에서도 유성음/무성음 zero-crossing 범위 자체가 겹침 (예: 10ms 프레임 기준
# 유성 0-30회, 무성 10-100회 - 10~30회 구간은 애초에 겹침). 한국어 '음/어' 대
# 호흡음의 ZCR 분포를 직접 비교한 연구는 못 찾음. 즉 ZCR 단독 판별의 한계는
# 우리 데이터만의 문제가 아니라 이 지표 자체의 알려진 한계 - 정밀도가 더
# 필요해지면 energy/periodicity/spectral flatness 등과 결합 필요 (현재는 범위 밖).
ZCR_TOLERANCE_RATIO = 1.5     # 개인 기준값의 이 배수 이내면 채움말로 인정
ZCR_FALLBACK_THRESHOLD = 0.15 # 이번 녹음에 고신뢰 채움말 샘플이 하나도 없을 때만 쓰는 예비값

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


def _compute_zcr(y, sr, start, end):
    """구간의 영교차율(zero-crossing rate) 계산. 파형이 0을 얼마나 자주
    가로지르는지 - 낮을수록 규칙적인(유성음 가능성 높은) 파형."""
    seg = y[int(start * sr):int(end * sr)]
    if len(seg) < int(0.02 * sr):
        return None
    zcr = librosa.feature.zero_crossing_rate(seg)[0]
    return float(zcr.mean())


def _get_personal_zcr_baseline(y, sr, high_confidence_events):
    """이번 녹음에서 텍스트로 확실히 잡힌 고신뢰 채움말(어/음 등)들의 ZCR
    평균을 "이 사람이 실제로 낸 채움말 소리"의 기준값으로 삼는다.
    고신뢰 샘플이 하나도 없으면 None 반환."""
    zcrs = [z for e in high_confidence_events
            if (z := _compute_zcr(y, sr, e["start"], e["end"])) is not None]
    return sum(zcrs) / len(zcrs) if zcrs else None


def _get_personal_breath_baseline(y, sr, sound_intervals, first_word_start, last_word_end):
    """말 시작 전 / 마지막 단어 끝난 후 구간(이미 숨소리·마이크 노이즈로 확인된
    부분)의 ZCR을 "이 사람의 숨소리" 기준값으로 삼는다. 채움말 기준값과
    마찬가지로 같은 녹음 안에서 뽑아내므로 개인차·환경차가 자동 반영됨."""
    zcrs = []
    for start, end in sound_intervals:
        lead = (start, min(end, first_word_start))
        trail = (max(start, last_word_end), end)
        for s, e in (lead, trail):
            if e - s >= ACOUSTIC_MIN_DUR:
                z = _compute_zcr(y, sr, s, e)
                if z is not None:
                    zcrs.append(z)
    return sum(zcrs) / len(zcrs) if zcrs else None


def _is_voiced(y, sr, start, end, filler_baseline, breath_baseline):
    """구간이 '목소리(채움말)'인지 '숨소리/잡음'인지 판별.
    - 채움말/숨소리 기준값이 둘 다 있으면: 어느 쪽에 더 가까운지로 판단(최근접 분류)
    - 채움말 기준값만 있으면: 그 값의 ZCR_TOLERANCE_RATIO배 이내인지로 판단
    - 둘 다 없으면: 고정 예비값(ZCR_FALLBACK_THRESHOLD) 사용
    """
    zcr = _compute_zcr(y, sr, start, end)
    if zcr is None:
        return False

    if filler_baseline is not None and breath_baseline is not None:
        return abs(zcr - filler_baseline) < abs(zcr - breath_baseline)
    if filler_baseline is not None:
        return zcr < filler_baseline * ZCR_TOLERANCE_RATIO
    return zcr < ZCR_FALLBACK_THRESHOLD


def detect_high_confidence_fillers(words, sound_intervals):
    """음/어/아/에/애 - VERBATIM_PROMPT 덕분에 Whisper가 독립된 단어로
    인식한 경우 채움말 후보로 본다. 단, 조건부 채움말과 마찬가지로 다른
    단어와 같은 소리구간에 뭉쳐있으면(=연장으로 이어붙여 읽은 경우) 제외한다
    (3_merged 테스트에서 이 케이스가 실제로 새는 것 확인됨)."""
    events = []
    for w in words:
        if w["text"] not in HIGH_CONFIDENCE_FILLERS:
            continue
        containing = [iv for iv in sound_intervals if iv[0] < w["end"] and iv[1] > w["start"]]
        merged_with_others = any(
            any(other is not w and other["start"] < iv[1] and other["end"] > iv[0] for other in words)
            for iv in containing
        )
        if not merged_with_others:
            events.append({"text": w["text"], "start": w["start"], "end": w["end"], "type": "high"})
    return events


def detect_conditional_fillers(words, sound_intervals):
    """단어형 채움말(그/저/뭐/막 등)을 Whisper 텍스트에서 찾는다.

    주의: Whisper 자체의 단어 간 gap(end/start 시각차)은 신뢰할 수 없음 -
    실측 결과, 단어 사이에 실제 침묵이 있어도 Whisper가 앞 단어의 끝 시각을
    다음 단어 시작 시각까지 늘려서 gap이 항상 0으로 나오는 경우가 확인됨.
    그래서 Whisper 타임스탬프 대신 librosa 소리구간을 기준으로 판단한다:
    "이 단어가 자기 혼자만의 소리구간을 차지하는가(분리됨) vs 다른 단어와
    같은 구간에 뭉쳐있는가(연장/이어짐)"."""
    events = []
    for w in words:
        if w["text"] not in CONDITIONAL_FILLERS:
            continue

        containing = [iv for iv in sound_intervals if iv[0] < w["end"] and iv[1] > w["start"]]
        merged_with_others = any(
            any(other is not w and other["start"] < iv[1] and other["end"] > iv[0] for other in words)
            for iv in containing
        )
        if not merged_with_others:
            events.append({"text": w["text"], "start": w["start"], "end": w["end"], "type": "conditional"})

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


def detect_acoustic_fillers(y, sr, sound_intervals, words, filler_baseline, breath_baseline):
    """고신뢰 채움말(어/음/아/에/애 등 비언어적 간투사)을 텍스트 매칭 없이
    순수 음향으로 찾는다. 각 소리구간에서 Whisper 단어가 커버하는 시간을
    빼고 남는(=아무 단어로도 설명 안 되는) 부분을 채움말 후보로 본다.

    기존 v2의 "구간에 단어가 하나라도 겹치면 통째로 제외" 방식과 달리,
    여러 단어가 한 덩어리로 뭉친 소리구간이어도 그 안에 숨은 채움말을
    부분적으로 찾아낼 수 있다.

    필터 2단계:
    1) 첫 단어 시작 전 / 마지막 단어 끝난 후 구간 제외 (말 시작 전 숨소리,
       녹음 종료 시점 마이크 노이즈로 확인됨)
    2) filler_baseline/breath_baseline 기준 최근접 분류 통과한 것만 인정

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
            if not _is_voiced(y, sr, l_start, l_end, filler_baseline, breath_baseline):
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

    conditional_events = detect_conditional_fillers(word_segments, sound_intervals)
    high_confidence_events = detect_high_confidence_fillers(word_segments, sound_intervals)

    # 이번 녹음 "안에서" 채움말 기준값 + 숨소리 기준값을 각각 뽑아냄.
    # 둘 다 있으면 최근접 분류, 하나만 있으면 그 기준으로 상대 비교,
    # 둘 다 없으면 고정 예비값 사용 (_is_voiced 내부 로직).
    first_word_start = word_segments[0]["start"] if word_segments else 0.0
    last_word_end = word_segments[-1]["end"] if word_segments else 0.0
    filler_baseline = _get_personal_zcr_baseline(y, sr, high_confidence_events)
    breath_baseline = _get_personal_breath_baseline(y, sr, sound_intervals, first_word_start, last_word_end)

    acoustic_events = detect_acoustic_fillers(
        y, sr, sound_intervals, word_segments, filler_baseline, breath_baseline,
    )
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
        "personal_filler_zcr": round(filler_baseline, 4) if filler_baseline is not None else None,
        "personal_breath_zcr": round(breath_baseline, 4) if breath_baseline is not None else None,
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
        print(f"개인 채움말 ZCR: {result['personal_filler_zcr']}  |  "
              f"개인 숨소리 ZCR: {result['personal_breath_zcr']}")
        print(f"분당 채움말: {result['filler_rate_per_min']}개")
        print(f"총 {result['total_count']}건 (고신뢰 {result['high_confidence_count']} / "
              f"조건부 {result['conditional_count']} / 음향 기반 {result['acoustic_count']})")
        for e in result["events"]:
            print(f"[{e['type']:>11}] {e['start']:.2f}s ~ {e['end']:.2f}s '{e['text']}'")
    else:
        print("사용법: python analyze_filler_final.py <오디오파일경로>")