# Re-born · backend

발화 불안·사회불안을 가진 고립·은둔 청년을 위한 AI 기반 디지털 재활 웹앱의 Flask 백엔드.
3단계 파이프라인: **1단계** 음성(채움말·멈춤·CNN 불안도) → **2단계** 표정·시선(MediaPipe
blendshape 기반 규칙 판정) → **3단계** 모의 면접(LLM 질문 생성 + TTS + STT).

## 스택

| 영역 | 기술 |
|---|---|
| 서버 | Flask 3 + Flask-SQLAlchemy(PostgreSQL, `psycopg2-binary`) + Flask-Cors |
| 인증 | PyJWT(직접 서명/검증), `werkzeug.security`(비밀번호 해시) |
| 1단계(음성) | librosa(채움말·멈춤) + torch/torchvision(CNN 불안도, CPU-only 빌드) + openai-whisper(STT) |
| 2단계(표정·시선) | mediapipe(blendshape) + opencv-python-headless |
| 3단계(면접) | Anthropic API(질문 생성, `claude-sonnet-5` 기본) + Typecast(TTS, 면접관별 음성) |
| 배포 | Render(Blueprint, `render.yaml`) — gunicorn, 무료 티어(RAM ~512MB) 기준 workers=1 |

## 실행

```bash
cp .env.example .env   # DATABASE_URL, SECRET_KEY 필수 — 없으면 app.py가 import 시점에 RuntimeError로 죽음
pip install -r requirements.txt
python app.py          # http://localhost:5000
```

torch/torchvision은 CPU-only 인덱스(`--extra-index-url .../whl/cpu`)에서 받아옴 — 배포 환경에
GPU가 없어서 GPU 빌드(수GB)를 받지 않으려는 목적. mediapipe는 `opencv-python-headless`와만
호환되니 GUI용 `opencv-python`을 따로 추가하지 말 것(둘이 같이 깔리면 충돌).

DB는 PostgreSQL(2026-09에 MySQL에서 이전 — Render가 매니지드 MySQL을 지원하지 않아서).
로컬도 `.env.example`과 동일한 `postgresql+psycopg2://` 형식으로 맞춰야 배포 환경과 같은 조건으로
테스트됨.

## 구조

```
app.py          Flask 라우트 전체 (인증, /analyze*, /community/*, /interview/*)
step1/          음성 분석 — analyze_filler_final.py(채움말), inference_cnn_final.py(CNN),
                llm_feedback.py(코칭 피드백 생성, 순수 함수는 _user_prompt)
step2/          표정·시선 — *_analyzer.py(blink/gaze/expression), scoring.py, set_baseline.py,
                landmark_face_points.py(랜드마크 인덱스 + MediaPipe 세션 초기화)
                ACCURACY_NOTES.md — 임계값 근거·검증 이력(문헌 인용 + 실측 검증)
step3/          면접 질문(interview_question.py) · TTS(tts.py, Typecast)
tests/          유닛테스트(pytest) + tests/labeling/(정확도 검증 하네스, 수동)
docs/           DB_DESIGN.md, TESTING.md(테스트 하네스 계층 설계)
render.yaml     Render Blueprint — 웹 서비스 + Postgres DB를 이 파일 하나로 생성
```

## API 엔드포인트

- **인증** — `POST /api/signup`, `POST /api/login`
- **1단계(음성)** — `POST /analyze`(업로드 → 오각형 점수), `POST /analyze/feedback`(LLM 코칭 피드백, 실패 시 템플릿 폴백), `GET /analyze/stage1/latest`(저장된 최신 결과 조회 — 3단계 난이도 산정·ai-agent용, 로그인 필요)
- **2단계(표정·시선)** — `POST /analyze/gaze-blink`(업로드 → 깜빡임·시선·표정 지표 + 하이라이트 클립), `POST /analyze/gaze-blink/live`(실시간 촬영 결과 요약 저장 — 영상 없이 숫자만, 저장 직전 기록을 `previous`로 반환), `GET /analyze/stage2/latest`(재분석 없이 최신 점수 조회 — 3단계 난이도 산정·ai-agent용, 로그인 필요)
- **3단계(면접)** — `POST /interview/next-question`(tier + 기본/꼬리질문 `mode` 기반 질문 생성, 로그인 시 최신 1·2단계 세부 점수로 질문 방식 조절 + 지난 면접의 기본 질문과 겹치지 않게, 실패 시 고정 질문 폴백), `POST /interview/tts`(질문 텍스트 → 면접관(`tier`)별 음성), `POST /interview/transcribe`(답변 STT), **면접 기록(로그인 필요, 본인만)** — `POST /interview/sessions`(시작), `POST /interview/sessions/<id>/turns`(질문 저장), `PUT /interview/sessions/<id>/turns/<turn_id>/answer`(답변 텍스트 저장), `POST /interview/sessions/<id>/complete`(완료), `GET /interview/sessions`(목록), `GET /interview/sessions/<id>`(질문·답변 상세), `DELETE /interview/sessions/<id>`(삭제)
- **커뮤니티("이야기")** — `GET/POST /community/posts`, `GET/DELETE /community/posts/<id>`, `POST /community/posts/<id>/comments`
- **기타** — `GET /health`

## 환경변수 (`.env`, `.env.example` 참고)

- `DATABASE_URL` — PostgreSQL 접속 정보. 필수(없으면 앱이 안 뜸)
- `SECRET_KEY` — JWT 서명 키, 예측 불가능한 임의 문자열. 필수
- `ANTHROPIC_API_KEY` — 없으면 `step1/llm_feedback.py`·`step3/interview_question.py`가 자동으로 폴백(템플릿/고정 질문)
- `FEEDBACK_MODEL` — 선택, 기본 `claude-sonnet-5`
- `INTERVIEW_MODEL` — 선택, 3단계 질문 생성 모델 (기본 `claude-sonnet-5`)
- `TYPECAST_API_KEY` — 3단계 면접관 음성(TTS). 없으면 음성 없이 진행(프론트가 폴백)
- `TYPECAST_VOICE_WARMUP` / `TYPECAST_VOICE_STANDARD` / `TYPECAST_VOICE_PRACTICE` — 면접관별 음성 ID. 비어 있으면 standard 음성으로 대체, `TYPECAST_MODEL`은 선택(기본 `ssfm-v30`)
- `MODEL_PATH` — 1단계 CNN 모델 가중치 파일 경로, 선택

## 테스트 — 3계층 (자세한 설계는 `docs/TESTING.md`)

1. **순수 로직 유닛테스트** (CI 연동, `tests/test_*.py`, labeling 하위 제외)
   ```bash
   pip install numpy opencv-python-headless pytest   # torch/whisper/mediapipe 불필요
   pytest tests/test_scoring.py tests/test_blink_analyzer.py tests/test_expression_analyzer.py \
          tests/test_gaze_analyzer.py tests/test_interview_question.py tests/test_llm_feedback.py -v
   ```
   대상: 임계값 비교·집계·프롬프트 조립처럼 입력→출력이 결정적인 함수.

2. **정확도 검증 하네스** (수동, CI 밖) — `tests/labeling/extract_label_candidates.py`로
   영상에서 프레임+점수 CSV 추출 → 사람이 라벨링 → `evaluate_threshold.py`로
   precision/recall/accuracy 계산. 결과·근거는 `step2/ACCURACY_NOTES.md`에 기록.

3. **API/통합 테스트** — 아직 없음. `app.py`가 torch/whisper/mediapipe를 라우트 함수 안
   lazy-import로 옮겨둬서(`_init_torch()`, `load_models()` 참고) Flask test client로 `/health`,
   `/api/signup` 같은 가벼운 라우트는 무거운 의존성 없이 테스트할 수 있음 — 아직 안 쌓였을 뿐.

push/PR 시 GitHub Actions에서 문법 검사(ruff) + 위 1번 테스트를 자동으로 확인합니다
(`.github/workflows/ci.yml`).

## 트러블슈팅 기록

과거에 겪은 버그·오탐/미검출 이슈와 해결 과정은 별도 레포
`re-born-hongik-univ/trouble-shooting`의 `backend/`에 이슈 1건당 파일 1개로 정리돼 있음.
