# -*- coding: utf-8 -*-
"""
채움말 검출 테스트 - 기대값과 비교해서 결과를 보기 쉽게 출력.
사용법: python tests\\manual\\run_filler_tests.py
(backend 폴더에서 실행)
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from step1.analyze_filler_final import analyze_filler

AUDIO_DIR = r"C:\Users\seoyn\OneDrive\Desktop\filler_test"

# 파일명, 설명, 기대 개수를 한 곳에서 관리 -> 대본 바뀌면 여기만 수정하면 됨
TEST_CASES = [
    {"file": "1_baseline.m4a",     "desc": "채움말 없음 (베이스라인)",
     "expected_total": 0, "expected_high": 0, "expected_conditional": 0},
    {"file": "2_separated.m4a",    "desc": "분리된 채움말 (어 2개 + 음 2개 = 4개)",
     "expected_total": 4, "expected_high": 4, "expected_conditional": 0},
    {"file": "3_merged.m4a",       "desc": "이어진 채움말 (연장 처리로 0건이 정상)",
     "expected_total": 0, "expected_high": 0, "expected_conditional": 0},
    {"file": "4_word_type.m4a",    "desc": "단어형 채움말 (그/저/막 각 1개 = 3개)",
     "expected_total": 3, "expected_high": 0, "expected_conditional": 3},
    {"file": "5_short_normal.m4a", "desc": "짧은 정상 단어 (오탐 없어야 함)",
     "expected_total": 0, "expected_high": 0, "expected_conditional": 0},
]

TYPE_LABEL = {"high": "고신뢰", "conditional": "조건부", "acoustic": "음향기반"}


def run_one(case):
    path = os.path.join(AUDIO_DIR, case["file"])
    print(f"\n{'=' * 70}")
    print(f"  {case['file']}  —  {case['desc']}")
    print(f"{'=' * 70}")

    if not os.path.exists(path):
        print(f"  [경고] 파일을 찾을 수 없음: {path}")
        return None

    result = analyze_filler(path)
    actual_total = result["total_count"]
    actual_high = result["high_confidence_count"]
    actual_cond = result["conditional_count"]

    exp_total = case["expected_total"]
    exp_high = case["expected_high"]
    exp_cond = case["expected_conditional"]

    # 총합만 맞고 세부 타입이 안 맞는 "가짜 일치"를 걸러내기 위해 타입별로도 비교
    total_match = actual_total == exp_total
    type_match = (actual_high == exp_high) and (actual_cond == exp_cond)
    full_match = total_match and type_match

    if full_match:
        mark = "일치"
    elif total_match and not type_match:
        mark = "가짜 일치 (총합만 맞고 세부 불일치)"
    else:
        mark = "불일치"

    print(f"  기대(총/고신뢰/조건부): {exp_total}/{exp_high}/{exp_cond}  |  "
          f"실제: {actual_total}/{actual_high}/{actual_cond}  {mark}")
    print(f"  개인 채움말 ZCR: {result['personal_filler_zcr']}  |  "
          f"개인 숨소리 ZCR: {result['personal_breath_zcr']}")

    if result["events"]:
        print(f"\n  {'구간':<18} {'길이':>6}  {'타입':<8}  내용")
        print(f"  {'-' * 55}")
        for e in result["events"]:
            dur = e["end"] - e["start"]
            label = TYPE_LABEL.get(e["type"], e["type"])
            print(f"  {e['start']:5.2f}s~{e['end']:5.2f}s   {dur:4.2f}s  {label:<8}  {e['text']}")
    else:
        print("  (검출된 이벤트 없음)")

    if result.get("cnn_prolongation_candidates"):
        print(f"\n  CNN 연장 후보 (뭉쳐서 채움말 집계에서 제외됨, {len(result['cnn_prolongation_candidates'])}건):")
        for c in result["cnn_prolongation_candidates"]:
            print(f"    {c['start']:5.2f}s~{c['end']:5.2f}s  '{c['text']}'  (source={c['source']})")

    return {"file": case["file"], "expected_total": exp_total, "actual_total": actual_total, "match": full_match}


def main():
    summary = []
    for case in TEST_CASES:
        r = run_one(case)
        if r:
            summary.append(r)

    print(f"\n\n{'=' * 70}")
    print("  전체 요약")
    print(f"{'=' * 70}")
    print(f"  {'파일':<22} {'기대':>4} {'실제':>4}  결과")
    print(f"  {'-' * 55}")
    for r in summary:
        mark = "일치" if r["match"] else "불일치"
        print(f"  {r['file']:<22} {r['expected_total']:>4} {r['actual_total']:>4}  {mark}")

    matched = sum(1 for r in summary if r["match"])
    print(f"\n  {matched}/{len(summary)}개 파일이 기대값과 일치")


if __name__ == "__main__":
    main()