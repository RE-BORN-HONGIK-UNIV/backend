# Re-born · backend

발화 불안·사회불안을 가진 고립·은둔 청년을 위한 AI 면접 재활 플랫폼 리본(Re-born)의 Flask 백엔드다.
3단계 흐름: **1단계** 목소리(채움말·멈춤 규칙 + 연장·떨림·에너지 CNN) → **2단계** 표정·시선(MediaPipe
blendshape 규칙 판정) → **3단계** 실전 면접(AI 질문 생성 + TTS + STT) → 코치 노트·리포트.
1·2단계에서는 대화 상대 "하루"가 사용자의 대답에 반응하며 이야기를 이어 간다.

## 전체 흐름

```
브라우저 (React · Netlify)
  │  녹음·영상 업로드, 2단계 실시간 통화는 브라우저 MediaPipe가 계산한 숫자만 전송
  ▼
Flask 백엔드 (이 저장소 · Render)
  ├─ 1단계 목소리 분석: librosa 규칙 + CNN + Whisper
  ├─ 2단계 표정·시선 분석: MediaPipe
  ├─ 3단계 질문·코치 노트·하루 대화 → Anthropic API
  ├─ 면접관·하루 목소리 → Typecast TTS
  └─ 점수·면접 기록 → PostgreSQL
```

1·2단계 점수가 쌓이면 3단계 면접의 난이도와 질문 방식이 정해지고, 면접이 끝나면 대답 기록으로 코치 노트와 리포트를 만든다.
영상·음성 파일은 분석 뒤 바로 지우고, 서버에는 점수와 대답 글만 남긴다.

## 스택

| 영역 | 기술 |
|---|---|
| 서버 | Flask 3 + Flask-SQLAlchemy(PostgreSQL, `psycopg2-binary`) + Flask-Cors |
| 인증 | PyJWT(직접 서명·검증), `werkzeug.security`(비밀번호 해시), 게스트 계정 |
| 1단계(목소리) | librosa(채움말·멈춤) + torch(CNN 연장·떨림·에너지, CPU 빌드) + openai-whisper(STT) |
| 2단계(표정·시선) | mediapipe(blendshape) + opencv-python-headless |
| 3단계·코치·하루 | Anthropic API(질문·코치 노트·리포트·하루 대화, 기본 `claude-sonnet-5`) + Typecast(TTS) |
| 배포 | Render Blueprint(`render.yaml`) — gunicorn, workers 1·threads 2, Standard 플랜(2GB) |

## 실행

```bash
cp .env.example .env   # DATABASE_URL, SECRET_KEY 필수 — 없으면 app.py를 불러오는 순간 RuntimeError
pip install -r requirements.txt
python app.py          # http://localhost:5000
```

- torch는 CPU 전용 인덱스(`--extra-index-url .../whl/cpu`)에서 받는다. 배포 서버에 GPU가 없어 수 GB짜리 GPU 빌드를 피한다.
- mediapipe는 `opencv-python-headless`와만 맞는다. GUI용 `opencv-python`을 같이 깔면 충돌한다.
- DB는 PostgreSQL이다(2026-09 MySQL에서 이전 — Render가 매니지드 MySQL을 지원하지 않음). 로컬도 `postgresql+psycopg2://` 형식으로 맞춰야 배포와 같은 조건으로 테스트된다.
- CNN 모델 파일은 저장소에 없다. `models/<label>/model.pth` + `config.json`을 손으로 넣는다(label: `prolongation`·`tremor`·`energy`). 없는 라벨은 `pending`으로 두고 서버는 그대로 뜬다.

## 구조

```
app.py          Flask 앱 — DB 표, 모든 API 라우트
db_url.py       DATABASE_URL 드라이버를 psycopg2로 고정
llm_util.py     AI 호출 도우미(effort 설정을 SDK 버전과 상관없이 넘김)
step1/          목소리 분석 — analyze_filler(채움말)·analyze_pause(멈춤), model·audio_prep(CNN, voice-cnn에서 복사),
                energy_rule(에너지 보조 규칙), stage1_score(점수), report_detail(리포트 세부 값),
                llm_feedback(코칭 한 문단), whisper_guard(Whisper 동시 사용 막기), analyze_options(단계별 저장 여부)
step2/          표정·시선 — *_analyzer(깜빡임·시선·표정), scoring, set_baseline, landmark_face_points,
                live_payload(실시간 결과 검증), ACCURACY_NOTES.md(임계값 근거·검증 이력)
step3/          면접 — interview_question(질문 생성), anxiety(불안도 → 난이도), session_payload(기록 검증), tts(Typecast)
coach/          면접 뒤 코치 — agent(코치 노트 에이전트), report(리포트 3·4번 칸), practice(연습 힌트·횟수 제한),
                schema·safety·tools·missions(검증·위기 신호·조회 도구·일상 미션 초안)
haru/           하루 대화 — talk(대답 반응·"하루가 이렇게 이해했어요" 정리)
tests/          pytest 유닛·API 테스트, labeling/(정확도 검증 도구, 수동), manual/(수동 점검 스크립트)
docs/           설계노트(step1_·step3_·coach_·haru_·공통_), 임계값 근거(*_근거.md), REFERENCES.md, DB_DESIGN.md, TESTING.md
render.yaml     Render Blueprint — 웹 서비스 + Postgres DB
```

## API

로그인 칸: **필요** = `Authorization: Bearer <token>` 필수(게스트 토큰 포함), **선택** = 있으면 기록을 저장하거나 맞춤으로 동작, **—** = 필요 없음.
AI를 쓰는 API는 키가 없거나 호출이 실패해도 정해 둔 문장으로 답한다.

### 인증

| 메서드 | 주소 | 하는 일 | 로그인 |
|---|---|---|---|
| POST | `/api/signup` | 회원가입 | — |
| POST | `/api/login` | 로그인, 7일 토큰 발급 | — |
| POST | `/api/guest` | 게스트 계정과 30일 토큰 발급 (IP당 시간당 20개) | — |
| PUT | `/api/me/nickname` | 불러 줄 이름 저장 (20자까지) | 필요 |

### 1단계 · 목소리

| 메서드 | 주소 | 하는 일 | 로그인 |
|---|---|---|---|
| POST | `/api/stage1/calibrate` | 마이크 확인 녹음으로 개인 기준값 측정 | — |
| POST | `/analyze` | 녹음 → 목소리 5개 점수·리포트 세부 값 (`stage` 1·2면 저장) | 선택 |
| POST | `/analyze/feedback` | 분석 결과로 코칭 한 문단 생성 | — |
| GET | `/analyze/stage1/latest` | 최근 1단계 결과 | 필요 |

### 2단계 · 표정·시선

| 메서드 | 주소 | 하는 일 | 로그인 |
|---|---|---|---|
| POST | `/analyze/gaze-blink` | 업로드 영상 → 깜빡임·시선·표정 점수 + 하이라이트 클립 | 필요 |
| POST | `/analyze/gaze-blink/live` | 실시간 통화 결과(숫자) 저장 | 필요 |
| GET | `/analyze/stage2/latest` | 최근 2단계 결과 | 필요 |

### 하루 대화

| 메서드 | 주소 | 하는 일 | 로그인 |
|---|---|---|---|
| POST | `/haru/reply` | 대답에 한 문장 반응 + 다음 질문 (시간당 120회) | 필요 |
| POST | `/haru/profile` | "하루가 이렇게 이해했어요" 카드 정리 (저장 안 함) | 필요 |

### 3단계 · 면접

| 메서드 | 주소 | 하는 일 | 로그인 |
|---|---|---|---|
| GET | `/interview/anxiety` | 1·2단계 결과 → 면접 난이도 | 필요 |
| POST | `/interview/next-question` | 기본·꼬리질문 생성 | 선택 |
| POST | `/interview/tts` | 문장 → 면접관·하루 목소리 mp3 | — |
| POST | `/interview/transcribe` | 대답 영상 → 글 (Whisper) | — |
| POST | `/interview/sessions` | 면접 시작 | 필요 |
| GET | `/interview/sessions` | 내 면접 목록 (최근 20개) | 필요 |
| GET | `/interview/sessions/<id>` | 면접 하나의 질문·대답 | 필요 |
| DELETE | `/interview/sessions/<id>` | 면접 기록 삭제 | 필요 |
| POST | `/interview/sessions/<id>/turns` | 질문 저장 | 필요 |
| PUT | `/interview/sessions/<id>/turns/<turn_id>/answer` | 대답 저장 | 필요 |
| POST | `/interview/sessions/<id>/complete` | 면접 완료 기록 | 필요 |

### 코치

| 메서드 | 주소 | 하는 일 | 로그인 |
|---|---|---|---|
| POST | `/interview/sessions/<id>/coach-note` | 면접 직후 코치 노트 (면접당 한 번 생성·저장) | 필요 |
| POST | `/interview/sessions/<id>/report-coach` | 리포트의 다듬은 대답·다음 연습 계획 (면접당 한 번) | 필요 |
| POST | `/interview/practice-hint` | 맞춤 연습 힌트 (시간당 20회) | 필요 |

### 커뮤니티 · 기타

| 메서드 | 주소 | 하는 일 | 로그인 |
|---|---|---|---|
| GET | `/community/posts` | 글 목록 | — |
| POST | `/community/posts` | 글쓰기 | 필요 |
| GET | `/community/posts/<id>` | 글과 댓글 | — |
| DELETE | `/community/posts/<id>` | 글 삭제 (본인 글만) | 필요 |
| POST | `/community/posts/<id>/comments` | 댓글 쓰기 | 필요 |
| GET | `/health` | 서버·모델 상태 | — |

## 환경변수 (`.env`, `.env.example` 참고)

- `DATABASE_URL` — PostgreSQL 주소. 필수
- `SECRET_KEY` — JWT 서명 키(예측할 수 없는 긴 문자열). 필수
- `ANTHROPIC_API_KEY` — 없으면 AI를 쓰는 곳이 모두 정해 둔 문장으로 대신한다
- `FEEDBACK_MODEL` · `INTERVIEW_MODEL` · `COACH_MODEL` · `HARU_MODEL` — 선택, 기본 `claude-sonnet-5`
- `TYPECAST_API_KEY` — 면접관·하루 목소리. 없으면 프론트가 미리 녹음한 mp3나 브라우저 목소리로 대신한다
- `TYPECAST_VOICE_WARMUP` / `_STANDARD` / `_PRACTICE` / `_HARU` — 목소리 ID. 면접관은 비면 standard 목소리, 하루는 기본 목소리. `TYPECAST_MODEL`은 선택(기본 `ssfm-v30`)
- `CARE_RESOURCES` — 선택, 대답에서 위기 신호가 보였을 때 보여 줄 도움 안내(팀이 확정한 문구만). 비면 일반 돌봄 문구
- `MODELS_DIR` — 선택, CNN 모델 폴더(기본 `models`)
- `PRELOAD_MODELS` — `1`이면 서버가 켜지자마자 모델을 미리 불러 첫 사용자의 1~3분 대기를 없앤다. 메모리 2GB 이상에서만 켠다(무료 512MB에서는 켜자마자 꺼짐)

## 테스트 (설계는 `docs/TESTING.md`)

```bash
pytest -q
```

1. **순수 로직 유닛 테스트** — 임계값 비교·점수 집계·프롬프트 조립·요청 검증처럼 입력과 출력이 정해진 함수. torch·whisper·mediapipe 없이 돈다.
2. **API 테스트** — `tests/test_guest_api.py`가 임시 sqlite와 Flask test client로 게스트 흐름을 확인한다. flask가 없는 환경에서는 건너뛴다.
3. **정확도 검증 도구** (수동, CI 밖) — `tests/labeling/`으로 영상·녹음에서 후보를 뽑아 사람이 라벨링하고 precision·recall을 계산한다. 결과와 근거는 `step2/ACCURACY_NOTES.md`, `docs/*_근거.md`에 적는다.

push·PR마다 GitHub Actions(`.github/workflows/ci.yml`)가 문법 검사, ruff(`F` 규칙), 가벼운 유닛 테스트 목록을 돌린다.

## 배포

- `main`에 합치면 Render가 `render.yaml` 설정대로 자동으로 다시 배포한다(Standard 플랜, gunicorn workers 1·threads 2, timeout 300초).
- Whisper 모델은 빌드 때 미리 받아 두고, `PRELOAD_MODELS=1`로 서버가 켜지자마자 모델을 불러 둔다.
- `DATABASE_URL`(External Database URL)과 API 키는 Render 대시보드 Environment에서 직접 넣는다. 파일에는 값을 적지 않는다.

## 트러블슈팅 기록

예전에 겪은 버그·오탐·미검출과 해결 과정은 별도 저장소 `re-born-hongik-univ/trouble-shooting`의 `backend/`에 이슈 하나당 파일 하나로 정리돼 있다.
