"""표정(blendshape) 기반 분석 — 미소 빈도 / 긴장(찡그림) 빈도.

MediaPipe FaceLandmarker가 뽑아주는 52개 ARKit 스타일 blendshape 중,
비언어적 소통 코칭에 의미 있는 신호만 골라 쓴다.
- 미소: mouthSmileLeft / mouthSmileRight (AU12, 광대근/zygomaticus major)
- 긴장: browDownLeft / browDownRight (AU4, 눈썹내림근/corrugator supercilii)

이 blendshape ↔ FACS Action Unit 대응 자체의 근거: Turrisi, R., Iacono Isidoro,
S., Bruschetta, R., Famà, F., Campisi, A., Aiello, S., Cusimano, G., Ruta, L.,
Pioggia, G., & Tartarisco, G. (2026). "Blendshape features meet action units:
a clinical mapping for enhancing facial expression analysis." Computers in
Human Behavior Reports. — 임상심리사·심리치료사 10명이 MediaPipe 52개
blendshape를 AU에 매핑해 합의 검증(88% 만장일치, 98% 과반 일치)한 논문으로,
mouthSmileLeft/Right=AU12, browDownLeft/Right=AU4, eyeSquintLeft/Right=AU7,
cheekSquintLeft/Right=AU6 대응을 명시적으로 확인해줌 — 이전까지는 이 대응을
일반적인 FACS 지식으로 추정만 했는데, 이 논문으로 대응 자체의 출처가 생김.
"""

SMILE_KEYS = ["mouthSmileLeft", "mouthSmileRight"]

# 2026-09-18 재설계: 원래는 browDown(AU4)에 eyeSquint(AU7, 눈 찡그림)까지 더해서
# 긴장 점수를 냈으나, FACS 문헌 기준으로 재검토함.
# - AU4(눈썹내림근/corrugator supercilii)는 부정 정서·스트레스의 검증된 지표:
#   Larsen, J. T., Norris, C. J., & Cacioppo, J. T. (2003). "Effects of positive
#   and negative affect on electromyographic activity over zygomaticus major and
#   corrugator supercilii." Psychophysiology, 40(5), 776-785.
#   DOI: 10.1111/1469-8986.00078.
# - AU7(눈꺼풀 조임근)은 위 문헌에서 긴장 특이적 지표로 다뤄지지 않고, 실측으로도
#   진짜 웃음(Duchenne smile, 눈가 주름) 프레임에서 함께 올라가는 걸 확인함 —
#   4번째 영상(카카오톡 촬영본)의 확실한 웃음 프레임에서 browDown≈0.004~0.006인
#   반면 eyeSquint≈0.36~0.38까지 치솟아 긴장 점수를 오염시켰음. eyeSquint를 빼고
#   browDown 단독으로 4영상(v1~v4, 총 프레임) 재검증한 결과, 기존에 있던 오탐
#   2건(v2/vlog1)이 사라지고 진탐(TP) 손실은 없었음 — 나머지 영상들도 비-긴장
#   구간의 최고점수가 전반적으로 낮아져(예: v3 0.291→0.019) 임계값과의 여유가
#   커짐. 이후 AU6(orbicularis oculi, cheekSquintLeft/Right)로 Duchenne 미소
#   판별도 시도했으나 이 MediaPipe 모델에서 cheekSquint 값이 항상 0에 가까워
#   (45프레임 실측 최댓값 0.0000048) 사실상 작동하지 않음을 확인, 미소 쪽은
#   SMILE_HIGH_CONFIDENCE_BYPASS(실측 기반 잠정치)를 그대로 유지함.
TENSION_KEYS = ["browDownLeft", "browDownRight"]
JAW_OPEN_KEY = "jawOpen"

# SMILE_THRESHOLD: 자체 라벨링(48+45프레임)으로 튜닝한 잠정치였는데, 2026-09-18에
# 독립된 임상 연구에서 비슷한 값을 찾음 — Dotzer, M., Kachel, U., Huhsmann, J.,
# Huscher, H., Raveling, N., Kugelmann, K., Blank, S., Neitzel, I., Buschermöhle,
# M., von Polier, G. G., & Radeloff, D. (2025). "Identification of smile events
# using automated facial expression recognition during the Autism Diagnostic
# Observation Schedule (ADOS-2): a proof-of-principle study." Frontiers in
# Psychiatry. DOI: 10.3389/fpsyt.2025.1497583. 이 논문은 Apple ARKit(MediaPipe와
# 같은 계열의 0~1 스케일 blendshape, 52개 중 겹치는 항목 다수)의 mouthSmile
# blendshape 단독으로 미소 이벤트를 판정하는 최적 임계값을 데이터 기반으로
# 구해서 **0.395**를 제시(사람 평가자 대비 민감도 96.43%, 특이도 96.08%,
# 카파 0.918 — n=79 테스트셋). 우리 값(0.35)과 상당히 가까워서 방향성 근거로는
# 쓸 수 있지만, 촬영 장비·모델 버전·캡처 조건이 달라 그 논문 숫자를 그대로
# 가져다 쓰지는 않음 — 임계값 변경은 팀 컨벤션대로 자체 evaluate_threshold.py
# 재검증을 거친 뒤에만 반영 (다음 액션: ACCURACY_NOTES.md 참고).
# TENSION_THRESHOLD: browDown(AU4)에 대한 동급 임상 연구는 아직 못 찾음 — 지표
# 선택(browDown=AU4)은 위 Larsen et al./Turrisi et al.로 근거가 있지만, 0.4라는
# 숫자 자체의 외부 근거는 없음. 자체 라벨링 검증만으로 유지 중 (근거자료 없는
# 잠정치, ACCURACY_NOTES.md "긴장 임계값" 참고).
#
# compute_expression_series는 baseline_tension 인자를 받아 "절대 점수" 대신
# "개인별 baseline(set_baseline.calibrate_baseline_tension) 대비 편차"로 비교하는
# 것도 가능하다 — resting face 개인차를 캘리브레이션 단계에서 상쇄하려는 목적으로
# 만들어졌으나, 4영상 재검증 결과 eyeSquint를 뺀 뒤로는 browDown 단독 신호
# 자체가 이미 개인차가 적어서 baseline까지 빼면 과보정으로 진짜 긴장(v4)의
# recall이 75%→0%로 악화됨을 확인함 — 그래서 app.py는 이 인자를 넘기지 않고
# 기본값(0.0, 절대 임계값)을 그대로 씀. 함수 자체는 남겨둠(TENSION_KEYS 구성이
# 다시 바뀌어 개인차가 커지면 재검토 여지 있음) — ACCURACY_NOTES.md "6차 검증" 참고.
SMILE_THRESHOLD = 0.35
TENSION_THRESHOLD = 0.4

# 2026-09-12 라벨링 검증(3인 48프레임)에서 발견: mouthSmile이 말하느라 입을
# 벌린 순간(발화 조음)에도 높게 반응해서 오탐(20/48)이 심했음. jawOpen이 이
# 임계값을 넘으면 그 프레임의 미소 점수를 0으로 무시하도록 게이팅해서
# 오탐을 20건→5건, 정확도 56.2%→87.5%로 줄임 (evaluate_threshold.py 재검증).
# 표본이 작아 엄밀한 최적값은 아니고, "일단 훨씬 나아짐" 수준의 잠정치.
JAW_OPEN_GATE = 0.05

# 2026-09-16 재검증(4~6번째 인물, 영상 3개·45프레임, ACCURACY_NOTES.md "2차 검증"
# 참고)에서 발견: JAW_OPEN_GATE가 "말하며 지속적으로 웃는" 화법에서 진짜 미소까지
# 대량으로 지워버려 recall이 크게 떨어지는 케이스가 있었음(영상 하나는 recall 0%,
# 다른 하나는 16.7%). mouthSmile 원본 점수가 이 값 이상이면 jawOpen 게이팅을
# 건너뛰도록 예외를 추가 — "말하기"로 인한 mouthSmile 오반응은 이 정도로 높게
# 나오는 경우가 드물다는 걸 실측으로 확인함(3영상 합산: 정확도 64.4%→80%,
# recall 15.8%→68.4%, precision 100%→81.2%). 이 값도 3영상·45프레임 기준
# 최적 구간(0.45~0.5)의 잠정치이지 확정 최적값은 아님 — 표본이 더 쌓이면 재검증.
SMILE_HIGH_CONFIDENCE_BYPASS = 0.5


def _avg(blendshapes, keys):
    vals = [blendshapes.get(k, 0.0) for k in keys]
    return sum(vals) / len(vals) if vals else 0.0


def compute_expression_series(frames, baseline_tension=0.0):
    """프레임 리스트 → 시간별 (t, smile_score, tension_score) 리스트.
    얼굴 미검출 프레임은 (t, None, None). 입을 벌린(말하는) 순간의 미소 오탐을
    줄이기 위해 jawOpen이 JAW_OPEN_GATE를 넘으면 그 프레임의 미소 점수는 0으로
    본다 — 단, 원본 미소 점수가 SMILE_HIGH_CONFIDENCE_BYPASS 이상으로 확실히
    높으면 "말하며 웃는" 진짜 미소일 가능성이 커서 게이팅을 건너뛴다.

    baseline_tension: set_baseline.calibrate_baseline_tension()으로 잡은 개인별
    캘리브레이션 구간 긴장 점수. tension_score는 raw 점수가 아니라 여기서 뺀 편차이고,
    이후 TENSION_THRESHOLD와의 비교도 이 편차 기준으로 이뤄진다. 기본값 0.0은
    캘리브레이션을 안 넘긴 호출(기존 테스트 등)을 위한 하위호환용 — 이 경우 이전과
    동일하게 절대 임계값처럼 동작한다."""
    series = []
    for f in frames:
        bs = f.get("blendshapes")
        if bs is None:
            series.append((f["t"], None, None))
            continue
        smile_raw = _avg(bs, SMILE_KEYS)
        jaw_open = bs.get(JAW_OPEN_KEY, 0.0)
        if smile_raw >= SMILE_HIGH_CONFIDENCE_BYPASS or jaw_open <= JAW_OPEN_GATE:
            smile = smile_raw
        else:
            smile = 0.0
        tension = _avg(bs, TENSION_KEYS) - baseline_tension
        series.append((f["t"], smile, tension))
    return series


def summarize_expression(series):
    """전체 구간에서 미소/긴장 임계값을 넘긴 프레임 비율."""
    valid = [(smile, tension) for _, smile, tension in series if smile is not None]
    if not valid:
        return {"smile_ratio": 0.0, "tension_ratio": 0.0, "frame_count": 0}

    smile_hits = sum(1 for smile, _ in valid if smile >= SMILE_THRESHOLD)
    tension_hits = sum(1 for _, tension in valid if tension >= TENSION_THRESHOLD)
    n = len(valid)
    return {
        "smile_ratio": round(smile_hits / n, 4),
        "tension_ratio": round(tension_hits / n, 4),
        "frame_count": n,
    }


def _merge_short_segments(segments, min_duration=0.3):
    """min_duration보다 짧은 구간은 이전 구간에 흡수 (gaze_analyzer와 동일한 방식)."""
    if not segments:
        return segments
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        too_short = (seg["end"] - seg["start"]) < min_duration
        same_type = seg["type"] == prev["type"]
        if too_short or same_type:
            prev["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    return merged


def detect_expression_segments(series, min_segment_sec=0.3):
    """시간별 (t, smile, tension) 시리즈 → 'smile'/'tension'/'neutral' 연속 구간 리스트.
    프론트에서 "이 순간 다시 보기" 용으로 쓰는 타임스탬프 — gaze의 fixation/aversion
    세그먼트와 같은 목적. 같은 프레임에서 둘 다 임계값을 넘기면 미소를 우선한다."""
    segments = []
    current_state = None
    seg_start = None
    last_t = None

    for t, smile, tension in series:
        if smile is None:
            state = "neutral"
        elif smile >= SMILE_THRESHOLD:
            state = "smile"
        elif tension >= TENSION_THRESHOLD:
            state = "tension"
        else:
            state = "neutral"

        if state != current_state:
            if current_state is not None:
                segments.append({"type": current_state, "start": seg_start, "end": t})
            current_state = state
            seg_start = t
        last_t = t

    if current_state is not None and last_t is not None:
        segments.append({"type": current_state, "start": seg_start, "end": last_t})

    return _merge_short_segments(segments, min_duration=min_segment_sec)
