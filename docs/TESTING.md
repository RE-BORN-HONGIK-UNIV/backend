# 테스트 하네스 설계

이 프로젝트는 성격이 다른 세 층의 테스트를 쓴다. 세 층을 섞어서 "테스트
커버리지 몇 %" 한 숫자로 말하면 안 된다 — 각 층이 답하는 질문 자체가 다르다.

## 1층 — 순수 로직 유닛테스트 (CI에 연동됨)

**질문**: "이 함수가 의도한 대로 동작하는가?" (임계값 비교, 집계, 문자열 조립처럼
입력이 정해지면 출력도 결정적인 함수)

- 위치: `tests/test_*.py` (labeling 하위 제외)
- 실행: `.github/workflows/ci.yml`의 `test-pure-logic` job — 매 push/PR마다 자동
- 의존성: numpy, opencv-python-headless(=cv2 import만 필요, 실제 영상 처리는 안 함) — torch/whisper/mediapipe 불필요
- 대상: `step2/scoring.py`, `blink_analyzer.py`, `expression_analyzer.py`, `gaze_analyzer.py`(`is_looking_at_camera`만 — 순수 수학), `step3/interview_question.py`(`_fallback_question`, `_user_prompt`), `step1/llm_feedback.py`(`_user_prompt`)
- **여기서 검증 못 하는 것**: 임계값 숫자 자체(0.4가 맞는 값인지)는 이 층의 책임이 아님 → 2층

## 2층 — 정확도 검증 하네스 (수동, CI 밖)

**질문**: "이 임계값/모델이 실제 데이터에서 맞는 판정을 하는가?"

- 위치: `tests/labeling/extract_label_candidates.py` (영상→프레임+점수 CSV 추출) → 사람이 직접 라벨링 → `evaluate_threshold.py` (precision/recall/accuracy 계산)
- 실행: 로컬에서 실제 영상 파일로 수동 실행. CI에 안 올라감 — mediapipe/cv2 풀버전 필요하고, "정답"이 사람 라벨이라 자동 반복 실행할 의미가 없음
- 근거/진행상황 기록: `step2/ACCURACY_NOTES.md`
- 지금 상태: 미소·긴장·시선 임계값 1차 검증 완료(48프레임·3인), 표본 확대·iris_thresh 검증은 진행 중 — 자세한 내용은 ACCURACY_NOTES.md 참고
- **주의**: 1층 테스트를 통과한다고 2층이 검증되는 게 아니다. 반대로 2층에서 임계값을 바꾸면(예: 0.4→0.3) 1층 테스트가 그 값을 하드코딩해서 검증하고 있진 않은지 확인 필요(현재 1층 테스트들은 임계값을 인자로 넘기거나 별도 상수로 검증해서 이 문제는 없음)

## 3층 — API/통합 테스트 (아직 없음 — 구조적 결정 필요)

**질문**: "요청이 라우트→DB→응답까지 올바르게 흐르는가?" (`/api/signup`, `/community/posts` 등)

아직 안 만든 이유는 우선순위 문제가 아니라 **`app.py`의 import 구조 때문**:

```python
import torch
import torchvision.transforms as transforms
...
from step2.landmark_face_points import extract_landmarks_from_video  # mediapipe, cv2
...
import whisper
```

이게 전부 파일 최상단에 있어서, `app.test_client()`로 `/api/signup`처럼 DB만
건드리는 라우트 하나를 테스트하려 해도 **모듈 import 시점에 torch/whisper/
mediapipe가 전부 로드**된다. 선택지는 두 가지:

- **(A) CI에 무거운 의존성을 다 설치** — torch/whisper 설치만 몇 분, 매 push마다 느려짐. `requirements.txt`가 아직 없는 것도(오늘 할 일 목록) 이 판단을 더 늦추고 있음.
- **(B) 무거운 import를 라우트 함수 안으로 lazy-import 리팩터링** — `/analyze`, `/analyze/gaze-blink`처럼 실제로 ML을 쓰는 라우트만 함수 내부에서 `import torch` 하도록 옮기면, `/api/signup`·`/health`·`/community/posts` 같은 라우트는 무거운 의존성 없이 빠르게 테스트 가능. 프로덕션 동작은 안 바뀌고(Python은 import를 캐싱하므로 첫 호출 이후 비용 동일) 테스트 용이성만 좋아짐 — 일반적으로 권장되는 패턴이지만 `app.py` 구조를 건드리는 리팩터링이라 별도 작업으로 진행 필요.

**권장**: (B)를 먼저 하고 나서 3층을 얇게(인증/DB CRUD 라우트 위주) 시작하는 것.
당장은 결정만 기록해두고, 팀 논의 후 착수.

## 검증 안 된 것 요약 (발표 Q&A 대비)

- 2층: 표본이 아직 작음(48프레임·3인, 라벨러 1인) — ACCURACY_NOTES.md "다음 액션" 참고
- 3층: 전무. app.py 리팩터링 여부 결정 전까지는 수동 테스트(Postman 등)에 의존
- 프론트엔드: 별도 `frontend/docs`(또는 README) 참고 — 이 문서는 backend 레포 기준
