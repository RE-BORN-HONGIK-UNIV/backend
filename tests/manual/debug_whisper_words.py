# -*- coding: utf-8 -*-
"""
디버그용: Whisper가 실제로 단어를 어떻게 쪼개서 반환하는지 확인.
VERBATIM_PROMPT 적용 전/후 비교도 같이 보여줌.
사용법: python debug_whisper_words.py <오디오파일경로>
"""
import sys
import librosa
sys.path.insert(0, ".")  # step1 패키지 찾기 위함 (backend 폴더에서 실행 가정)

from step1.analyze_filler_final import (
    get_word_segments, get_sound_intervals, _get_default_whisper_model, VERBATIM_PROMPT,
)

if len(sys.argv) < 2:
    print("사용법: python debug_whisper_words.py <오디오파일경로>")
    sys.exit(1)

audio_path = sys.argv[1]
model = _get_default_whisper_model()

words = get_word_segments(audio_path, model)  # VERBATIM_PROMPT 이미 내부에서 적용됨

y, sr = librosa.load(audio_path, sr=16000)
intervals = get_sound_intervals(y, sr)

print(f"총 {len(words)}개 단어 인식됨 (VERBATIM_PROMPT 적용 상태)\n")
print("-" * 60)
for w in words:
    print(f"'{w['text']}'  ({w['start']:.2f}s ~ {w['end']:.2f}s)")
print("-" * 60)

print(f"\n총 {len(intervals)}개 librosa 소리구간 (침묵 아닌 덩어리)\n")
print("-" * 60)
for start, end in intervals:
    dur = end - start
    covered_words = [w["text"] for w in words if start < w["end"] and end > w["start"]]
    flag = "  <-- 여러 단어가 한 구간에 뭉침!" if len(covered_words) >= 2 else ""
    print(f"{start:6.2f}s ~ {end:6.2f}s (길이 {dur:.2f}s)  겹치는 단어: {covered_words}{flag}")
print("-" * 60)

# 프롬프트 없이 돌렸을 때랑 비교 (진짜 프롬프트 효과가 있는지 확인용)
print("\n[비교 1] 프롬프트 없이 전체 인식 텍스트:")
result_no_prompt = model.transcribe(audio_path, language="ko")
print(result_no_prompt["text"])

print("\n[비교 2] VERBATIM_PROMPT 적용 전체 인식 텍스트:")
result_with_prompt = model.transcribe(audio_path, language="ko", initial_prompt=VERBATIM_PROMPT, temperature=0)
print(result_with_prompt["text"])