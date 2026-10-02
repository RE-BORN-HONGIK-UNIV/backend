"""
top_db 민감도 진단 스크립트

목적:
    analyze_filler / analyze_pause 둘 다 librosa.effects.split(y, top_db=30)으로
    "소리 구간"을 나누는데, 배경소음이 있으면 이 값(30)이 너무 둔감해서
    문장 사이 침묵을 못 잡고 전체가 "소리 구간 1개"로 뭉쳐버릴 수 있음.

    이 스크립트는 Whisper 없이(빠름) 여러 top_db 값으로 같은 파일을 나눠보고,
    몇 개의 소리 구간으로 쪼개지는지 비교해서 "지금 top_db=30이 맞는 값인지" 확인한다.

사용법:
    python tests/manual/diagnose_top_db.py "파일경로.m4a"
"""

import sys
import librosa

TOP_DB_CANDIDATES = [15, 20, 25, 30, 35, 40, 45]


def diagnose(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)

    print("=" * 60)
    print(f"파일: {audio_path}")
    print(f"전체 길이: {total_duration:.2f}초")
    print("=" * 60)
    print(f"{'top_db':>8} | {'소리구간 개수':>12} | 구간별 길이(초, 앞 10개만)")
    print("-" * 60)

    for top_db in TOP_DB_CANDIDATES:
        segments = librosa.effects.split(y, top_db=top_db)
        segments_sec = [(round((e - s) / sr, 2)) for s, e in segments]
        preview = segments_sec[:10]
        marker = "  ← 지금 코드 기본값" if top_db == 30 else ""
        print(f"{top_db:>8} | {len(segments):>12} | {preview}{marker}")

    print()
    print("[해석 가이드]")
    print("  - 개수가 1~2개면: 그 top_db에서는 전체가 거의 '하나의 소리'로 뭉쳐 보임 (너무 둔감)")
    print("  - 개수가 지나치게 많으면(예: 30개+): 숨소리/잡음까지 다 쪼개고 있을 가능성 (너무 민감)")
    print("  - 실제 말한 문장/단어 수와 비슷한 개수가 나오는 top_db가 현재 환경에 적절한 값")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python diagnose_top_db.py 파일경로.m4a")
        sys.exit(1)

    for path in sys.argv[1:]:
        diagnose(path)