"""
1단계 분석 응답에 붙이는 "어디서 나왔는지" 목록 (순수 함수).

최종 통합 리포트(3단계 뒤, 경량 웹 리포트 화면을 가져옴)가 "음" 같은 채움말이 어떤 말로 몇 번,
녹음의 어느 쯤에서 나왔는지와 긴 멈춤이 어디서 몇 초였는지를 보여 주려면 위치 목록이 필요하다.
analyze_filler/analyze_pause가 이미 계산하던 events를 그동안 응답에서 버리고 있었는데, 화면용으로
필요한 필드만 골라 붙인다(점수 계산은 바뀌지 않는다).

- 채움말: 점수에 반영되는 것(high/conditional)만. 음향 기반 후보(acoustic_candidate)는 점수에 안 들어가는
  기록용이라(analyze_filler.py 참고) 화면에도 안 보낸다.
- 멈춤: long_pause(LONG_PAUSE_THRESHOLD 이상)만 — 짧은 숨 고르기는 자연스러운 멈춤이라 표시하지 않는다.
- 개수 상한: 응답이 커지지 않게. 녹음 5분 기준으로도 넉넉한 값.
"""
from __future__ import annotations

MAX_EVENTS = 200
SCORED_FILLER_TYPES = ("high", "conditional")


def _num(v) -> float | None:
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return None


def filler_events(events) -> list[dict]:
    """[{text, start, end}] — 점수에 반영된 채움말만, 시간 순."""
    out = []
    for e in events or []:
        if not isinstance(e, dict) or e.get("type") not in SCORED_FILLER_TYPES:
            continue
        start, end = _num(e.get("start")), _num(e.get("end"))
        text = str(e.get("text") or "").strip()
        if start is None or end is None or not text:
            continue
        out.append({"text": text, "start": start, "end": end})
    out.sort(key=lambda x: x["start"])
    return out[:MAX_EVENTS]


def long_pause_events(events) -> list[dict]:
    """[{start, end, duration}] — 긴 멈춤만, 시간 순."""
    out = []
    for e in events or []:
        if not isinstance(e, dict) or e.get("type") != "long_pause":
            continue
        start, end, dur = _num(e.get("start")), _num(e.get("end")), _num(e.get("duration"))
        if start is None or end is None or dur is None:
            continue
        out.append({"start": start, "end": end, "duration": dur})
    out.sort(key=lambda x: x["start"])
    return out[:MAX_EVENTS]
