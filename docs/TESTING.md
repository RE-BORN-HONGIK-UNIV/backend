# 테스트 하네스 설계

이 프로젝트는 성격이 다른 세 층의 테스트를 쓴다. 세 층을 섞어서 "테스트
커버리지 몇 %" 한 숫자로 말하면 안 된다 — 각 층이 답하는 질문 자체가 다르다.

## 1층 — 순수 로직 유닛테스트 (CI에 연동됨)

**질문**: "이 함수가 의도한 대로 동작하는가?" (임계값 비교, 집계, 문자열 조립처럼
입력이 정해지면 출력도 결정적인 함수)

- 위치: `tests/test_*.py` (labeling 하위 제외)
- 실행: `.github/workflows/ci.yml`의 `test-pure-logic` job — 매 push/PR마다 자동
- 의존성: numpy, opencv-python-headless(=cv2 import만 필요, 실제 영상 처리는 안 함) — torch/whisper/mediapipe 불필요
- 대상: `step2/scoring.py`, `step2/quality.py`(얼굴 검출률 품질 게이트), `blink_analyzer.py`, `expression_analyzer.py`, `gaze_analyzer.py`(`is_looking_at_camera`만 — 순수 수학), `step3/interview_question.py`(`_fallback_question`, `_user_prompt`), `step1/llm_feedback.py`(`_user_prompt`)
- **여기서 검증 못 하는 것**: 임계값 숫자 자체(0.4가 맞는 값인지)는 이 층의 책임이 아님 → 2층

## 2층 — 정확도 검증 하네스 (수동, CI 밖)

**질문**: "이 임계값/모델이 실제 데이터에서 맞는 판정을 하는가?"

- 위치: `tests/labeling/extract_label_candidates.py` (영상→프레임+점수 CSV 추출) → 사람이 직접 라벨링 → `evaluate_threshold.py` (precision/recall/accuracy 계산)
- 실행: 로컬에서 실제 영상 파일로 수동 실행. CI에 안 올라감 — mediapipe/cv2 풀버전 필요하고, "정답"이 사람 라벨이라 자동 반복 실행할 의미가 없음
- 근거/진행상황 기록: `step2/ACCURACY_NOTES.md`
- 지금 상태: 미소·긴장·시선 임계값 1차 검증 완료(48프레임·3인), 표본 확대·iris_thresh 검증은 진행 중 — 자세한 내용은 ACCURACY_NOTES.md 참고
- **주의**: 1층 테스트를 통과한다고 2층이 검증되는 게 아니다. 반대로 2층에서 임계값을 바꾸면(예: 0.4→0.3) 1층 테스트가 그 값을 하드코딩해서 검증하고 있진 않은지 확인 필요(현재 1층 테스트들은 임계값을 인자로 넘기거나 별도 상수로 검증해서 이 문제는 없음)

### step2 파이프라인 구조 회귀 테스트 (`tests/test_step2_pipeline.py`)

기존 `tests/manual_test_step2.py`(print만 찍고 사람이 눈으로 확인하던 수동
스크립트)를 pytest로 전환한 것. 실제 영상이 있어야 도는 통합 테스트라 위
2층과 같은 이유(mediapipe 풀버전 필요, 저작권상 영상을 레포에 커밋 불가)로
CI에는 안 올라간다 — `ci.yml`의 `test-pure-logic` job이 실행할 파일을
명시적으로 나열하는 방식이라 이 파일은 자동으로 제외됨.

로컬에 테스트 영상이 있을 때만 환경변수로 실행:
```
STEP2_TEST_VIDEO=/path/to/video.mp4 pytest tests/test_step2_pipeline.py -v
```
환경변수가 없으면 전부 skip. **여기서 검증하는 건 "정확도"가 아니라
"구조"**(프레임이 추출됐는가, 얼굴 검출률이 정상 범위인가, gaze segments가
영상 전체를 빈틈없이 덮는가, 점수가 0~100 범위·정의된 status 값 안에
있는가) — 임계값이 정확한지는 여전히 위 2층(라벨링 하네스)의 몫이다.

## 3층 — API 통합 테스트 (CI에 연동됨, `tests/api/`)

**질문**: "요청이 라우트→DB→응답까지 올바르게 흐르는가?"

- 위치: `tests/api/` — `conftest.py`(Flask test client + 인메모리 SQLite, 테스트마다 테이블 비움,
  실제 `/api/signup`→`/api/login`으로 JWT 발급)와 `test_auth_health.py`·`test_stage_results.py`·
  `test_gaze_blink_route.py`
- 실행: `.github/workflows/ci.yml`의 `test-api` job — 매 push/PR마다 자동. 1층과 job을 분리한 이유는
  `app.py`가 최상단에서 import하는 librosa·matplotlib 설치가 무거워서(1층을 느리게 만들지 않으려고).
  torch/whisper/mediapipe는 여전히 불필요하다.
- 로컬: `pytest tests/api` (flask 등이 없으면 이 디렉터리 전체가 skip). `app.py`가 import 시점에 DB·SECRET_KEY를
  읽으므로 conftest가 import 전에 테스트용 값으로 고정해서 `.env`의 실제 DB에 붙지 않는다.
- 다루는 것: `/health`(모델 미로딩), 가입·로그인(필수값·약관·중복·잘못된 비밀번호·무효 토큰), 1·2단계
  최신 조회(null/최신 선택/유저 격리/종합 점수), 2단계 실시간 저장(`/analyze/gaze-blink/live`: 검증 실패 시
  저장 안 함·"직전 기록" 반환), 업로드 분석(`/analyze/gaze-blink`: 인증·파일 누락·얼굴 미검출 422(저장 안 함)·
  정상 응답·두 번째 호출의 previous·500 응답에 스택트레이스 미노출)
- **가짜로 바꾼 것**: `extract_landmarks_from_video`(mediapipe), 머리 자세 추정·시선 구간 판정,
  하이라이트 ffmpeg 편집. 이 값들의 *정확도*는 2층 몫이고, 여기서는 "품질 게이트→점수 계산→저장→응답"
  흐름만 검증한다. 변이 확인으로 품질 게이트를 끄면 422 테스트 3개가 실패함을 확인했다.
- **아직 없는 것**: 커뮤니티 게시판 라우트, 1단계 `/analyze`(whisper·CNN 필요)·`/analyze/feedback`,
  3단계 `/interview/*`(세션 CRUD·코치 노트·연습 힌트). 같은 conftest를 써서 이어 쌓으면 된다.
  ※ 이 테스트를 쓰다 `/analyze/gaze-blink`가 로그인 확인보다 mediapipe import를 먼저 해서 비로그인
  요청에도 모델을 올린다는 걸 발견해 순서를 고쳤다.

## 검증 안 된 것 요약 (발표 Q&A 대비)

- 2층: 표본이 아직 작음(48프레임·3인, 라벨러 1인) — ACCURACY_NOTES.md "다음 액션" 참고
- 3층: 인증·1/2단계 결과·업로드 분석 라우트까지만 있음(위 "아직 없는 것" 참고) — 3단계 면접·커뮤니티·1단계 분석 라우트는 아직 수동 테스트(Postman 등)에 의존
- 프론트엔드: 별도 `frontend/docs`(또는 README) 참고 — 이 문서는 backend 레포 기준
