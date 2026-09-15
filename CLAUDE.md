# Re-born · backend

발화 불안·사회불안을 가진 고립·은둔 청년을 위한 AI 기반 디지털 재활 웹앱의
Flask 백엔드. 3단계 파이프라인: **1단계** 음성(채움말·멈춤·CNN 불안도) →
**2단계** 표정·시선(MediaPipe blendshape 기반 규칙 판정) → **3단계** 모의 면접
(LLM 질문 생성 + TTS + STT).

## 스택

Flask + Flask-SQLAlchemy(MySQL) · torch/torchvision(CNN 추론) · whisper·librosa
(음성) · mediapipe·opencv(표정/시선) · Anthropic API(`claude-sonnet-5` 기본,
질문 생성·코칭 피드백에 사용, 환경변수로 모델 교체 가능)

## 구조

```
app.py          Flask 라우트 전체 (인증, /analyze*, /community/*, /interview/*)
step1/          음성 분석 — analyze_filler_final.py(채움말), inference_cnn_final.py(CNN),
                llm_feedback.py(코칭 피드백 생성, 순수 함수는 _user_prompt)
step2/          표정·시선 — *_analyzer.py(blink/gaze/expression), scoring.py, set_baseline.py
                ACCURACY_NOTES.md — 임계값 근거·검증 이력 (문헌 인용 + 실측 검증)
step3/          면접 질문(interview_question.py) · TTS(tts.py)
tests/          유닛테스트(pytest) + tests/labeling/(정확도 검증 하네스, 수동)
docs/           DB_DESIGN.md, TESTING.md(테스트 하네스 계층 설계)
```

## 실행

```bash
cp .env.example .env   # DATABASE_URL, SECRET_KEY 필수 — 없으면 app.py가 import 시점에 RuntimeError로 죽음
python app.py
```

`requirements.txt`가 아직 없음(레포에 미커밋 상태) — 이번 작업 전까지는 팀원 로컬 환경에
의존. 새로 만들 때 torch는 CPU-only, opencv는 headless로 가볍게 하는 게 목표.

## 테스트 — 3계층 (자세한 설계는 `docs/TESTING.md`)

1. **순수 로직 유닛테스트** (CI 연동, `tests/test_*.py`, labeling 하위 제외)
   ```bash
   pip install numpy opencv-python-headless pytest   # torch/whisper/mediapipe 불필요
   pytest tests/test_scoring.py tests/test_blink_analyzer.py tests/test_expression_analyzer.py \
          tests/test_gaze_analyzer.py tests/test_interview_question.py tests/test_llm_feedback.py -v
   ```
   대상: 임계값 비교·집계·프롬프트 조립처럼 입력→출력이 결정적인 함수. `gaze_analyzer.py`처럼
   파일 최상단에 `import cv2`가 있어도, 테스트 대상 함수 자체가 순수 수학이면 여기 포함시킨다
   (opencv-headless만 설치하면 됨, torch/whisper는 여전히 불필요).

2. **정확도 검증 하네스** (수동, CI 밖) — `tests/labeling/extract_label_candidates.py`로
   영상에서 프레임+점수 CSV 추출 → 사람이 라벨링 → `evaluate_threshold.py`로
   precision/recall/accuracy 계산. 결과·근거는 `step2/ACCURACY_NOTES.md`에 기록.
   **임계값 숫자를 바꾸는 근거는 반드시 이 하네스로 검증하고 나서 코드에 반영할 것** —
   표본이 한쪽으로 치우쳤으면(예: 특정 인물 1명이 라벨 대부분을 차지) 근거 부족으로 보류하고
   그 이유를 문서에 남긴다 (`step2/ACCURACY_NOTES.md`의 긴장 임계값 사례 참고).

3. **API/통합 테스트** — 아직 없음. 예전엔 `app.py`가 torch/whisper/mediapipe를 파일
   최상단에서 import해서 이게 구조적으로 막혀 있었는데, 2026-09 경량화 작업(배포 환경
   메모리 제한 대응)으로 전부 라우트 함수 안 lazy-import로 옮겨서(`_init_torch()`,
   `load_models()`, `analyze_gaze_blink()` 참고) 이제 Flask test client로 `/health`,
   `/api/signup` 같은 가벼운 라우트만 무거운 의존성 없이 테스트할 수 있음. 남은 건
   테스트를 실제로 쌓는 것 — `docs/TESTING.md` 참고.

## 트러블슈팅 기록

과거에 겪은 버그·오탐/미검출 이슈와 해결 과정은 별도 레포
`re-born-hongik-univ/trouble-shooting`의 `backend/`에 이슈 1건당 파일 1개로 정리돼 있음.
비슷한 증상(오탐·미검출·임계값 이상)을 다룰 땐 먼저 여기서 검색.

## 컨벤션

- 임계값·상수를 바꿀 때는 반드시 근거(문헌 인용 또는 실측 검증)를 코드 주석 또는
  `step2/ACCURACY_NOTES.md`에 남긴다 — "왜 이 숫자인지" 없이 매직넘버로 두지 않는다.
- LLM 호출(`step1/llm_feedback.py`, `step3/interview_question.py`)은 항상 실패 시
  폴백이 있어야 한다(네트워크/키/쿼터 문제로 화면 흐름이 깨지면 안 됨) — `except Exception`
  으로 감싸고 로그만 남긴 뒤 폴백 반환하는 기존 패턴을 따른다.
- 커밋 메시지는 한국어로, "무엇을"보다 "왜"(원인·근거·트레이드오프) 위주로 쓴다.
