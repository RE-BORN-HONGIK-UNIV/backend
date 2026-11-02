"""1단계 최종 리포트 화면용 세부 값을 녹음 하나 단위로 정리한다(채움말 위치, 긴 멈춤, 연장 위치, 에너지 막대, 떨림 곡선).
점수 계산은 하지 않고, /analyze가 이미 계산한 채움말·멈춤 이벤트와 CNN 조각별 확률을 화면에 쓰기 좋은 모양으로 바꾼다.
값 이름과 모양은 프론트 lib/api/types.ts의 AnalyzeResult(filler_detail.events, pause_detail.long_pauses, *_detail)와 같다.
"""
import numpy as np

ENERGY_BARS = 10


def energy_bars(y, sr, n=ENERGY_BARS):
    """말한 구간을 n칸으로 나눠 칸마다 음절 크기(프레임 음량 상위 10%)를 앞쪽 기준 칸 대비 %로 돌려준다."""
    frame = int(0.025 * sr)
    rms = np.sqrt(np.mean(y[:len(y) // frame * frame].reshape(-1, frame) ** 2, axis=1))
    # 말의 시작·끝은 가장 큰 소리 -20dB 안쪽으로 잡는다(Whisper 단어 끝은 뒤 잡음까지 늘어나는 일이 있다)
    loud = np.flatnonzero(rms > rms.max() * 0.1) if len(rms) else []
    if len(loud) == 0 or loud[-1] - loud[0] + 1 < n:
        return []
    rms = rms[loud[0]:loud[-1] + 1]
    # 쉬는 틈이 평균을 끌어내리지 않게 칸마다 큰 소리(상위 10%)끼리 비교한다
    levels = [np.percentile(part, 90) for part in np.array_split(rms, n)]
    # 기준은 앞쪽 세 칸 중 가장 큰 칸 — 첫 칸이 말 시작 직전 잡음이면 기준이 너무 작아진다
    ref = max(levels[:3])
    if ref <= 0:
        return []
    return [round(float(v / ref * 100)) for v in levels]


def merge_windows(windows, threshold, win_sec):
    """기준값을 넘은 조각들을 겹치는 것끼리 이어 붙여 (시작초, 끝초, 최고 확률) 구간 목록으로 만든다."""
    spans = []
    for start, p in windows:
        if p < threshold:
            continue
        end = start + win_sec
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], end, max(spans[-1][2], p))
        else:
            spans.append((start, end, p))
    return spans


def prolong_events(windows, threshold, win_sec, words):
    """연장으로 잡힌 구간마다 가장 길게 발음된 단어를 연장 단어로 골라 {text, start, end}로 돌려준다."""
    events = []
    for start, end, _ in merge_windows(windows, threshold, win_sec):
        inside = [w for w in words if w["start"] < end and w["end"] > start]
        if not inside:
            events.append({"text": "", "start": round(start, 2), "end": round(end, 2)})
            continue
        # shortcut: 3초 조각 안에서 가장 긴 단어를 연장으로 본다. 글자 단위 위치가 필요해지면 정렬(forced alignment)로 바꾼다
        w = max(inside, key=lambda w: w["end"] - w["start"])
        events.append({"text": w["text"], "start": round(w["start"], 2), "end": round(w["end"], 2)})
    return events


def build_report_detail(y, sr, filler_events, pause_events, words, cnn):
    """녹음 하나의 리포트 세부 값을 만든다. cnn은 {라벨: {'windows': [(시작초, 확률)], 'threshold', 'win_sec'}}이고 모델이 없는 라벨은 None(화면은 '준비 중')."""
    pro, trem = cnn.get("prolongation"), cnn.get("tremor")
    return {
        "filler_events": [{"text": e["text"], "start": round(e["start"], 2), "end": round(e["end"], 2)} for e in filler_events],
        # 긴 멈춤 = 멈춤 점수에 쓰는 기준(1.2초 이상)과 같다
        "long_pauses": [{"start": round(e["start"], 2), "end": round(e["end"], 2), "duration": round(e["duration"], 2)}
                        for e in pause_events if e["type"] == "long_pause"],
        "prolong_detail": {"events": prolong_events(pro["windows"], pro["threshold"], pro["win_sec"], words)} if pro else None,
        "energy_detail": {"sentence_bars": energy_bars(y, sr)},
        # 떨림 확률(0~1)을 조각 순서대로. 0에 가까울수록 고르다
        "tremor_detail": {"curve": [round(p, 3) for _, p in trem["windows"]]} if trem else None,
    }
