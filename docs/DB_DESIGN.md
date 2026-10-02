# DB 설계 기록

MySQL(`reborn_db`) 하나를 쓰고, 마이그레이션 도구 없이 `db.create_all()`로 없는
테이블만 생성한다 (Flask-SQLAlchemy, `app.py`).

## 스키마

### User
| 필드 | 타입 | 비고 |
|---|---|---|
| id | Integer, PK | |
| email | String(120), unique | |
| password_hash | String(200) | werkzeug generate_password_hash |
| name | String(80) | |
| nickname | String(80), nullable | |
| birthdate | String(10), nullable | |
| terms_agreed | Boolean | |
| created_at | DateTime | |

### Stage2Result (2026-09 추가)
2단계(표정·시선) 분석 결과를 유저별로 쌓아서, 세션 간 비교("지난번엔 ~했어요")에
쓴다. 기존엔 프론트 localStorage에만 쌓여서 기기 바뀌면 이력이 날아갔음 — 이걸
대체.

| 필드 | 타입 | 비고 |
|---|---|---|
| id | Integer, PK | |
| user_id | Integer, FK → User.id | |
| created_at | DateTime | |
| blink_rate_per_min | Float | |
| blink_status | String(20) | |
| blink_score | Float | |
| avg_fixation_sec | Float | |
| gaze_score | Float | |
| smile_ratio | Float | |
| tension_ratio | Float | |
| expression_score | Float | |
| expression_status | String(20) | |

`to_entry()` 메서드가 프론트 `Stage2Entry` shape(`at`, `blinkRatePerMin`,
`avgFixationSec`, `smileRatio`, `tensionRatio`)로 그대로 변환해줌 — 프론트/백엔드
간 타입 맞추려고 별도 변환 계층 안 둠.

### Stage1Result (2026-10 추가)
1단계(음성) 분석 결과를 유저별로 쌓는다. 용도는 두 가지 — 3단계 모의면접 난이도
산정(`GET /analyze/stage1/latest`)과 ai-agent의 세션 이력 조회. 기존엔 프론트
localStorage(`rb.stage1.score`)에만 있어서 서버가 1단계 점수를 알 방법이 없었음.

| 필드 | 타입 | 비고 |
|---|---|---|
| id | Integer, PK | |
| user_id | Integer, FK → User.id | |
| created_at | DateTime | |
| stability | Float | 음성 안정성 (tremor 기반) |
| fluency | Float | 발화 유창성 (채움말) |
| pause_ctrl | Float | 침묵 조절력 (멈춤) |
| continuity | Float | 발화 지속성 (prolongation 기반) |
| calm | Float | 발화 에너지 (energy 기반) |
| overall_score | Float | 위 5축 평균(반올림) — 프론트 `overallScore()`와 같은 공식 (`step1/stage1_score.py`) |

`to_entry()`는 camelCase(`pauseCtrl`, `overallScore` 등)로 변환. 5축 점수는 모두 0~100,
**높을수록 안정적**(불안도가 아님) — 3단계 `getTier()`에서 점수가 높을수록 실전
난이도가 되는 것도 이 방향 전제.

**저장 규칙 (`/analyze`)**: 인증은 선택 — 로그인 없이도 분석은 되지만 저장은 로그인
유저만. CNN 모델이 없는 데모 모드(`demo_mode`)의 0점 더미 값은 난이도를 왜곡하므로
저장하지 않음. 저장 실패는 로그만 남기고 분석 응답은 정상 반환(부가 기능이라 화면
흐름을 막지 않음).

**소급 불가**: 이 테이블이 생기기 전에 분석한 결과는 localStorage에만 있어서 DB로
옮길 수 없음. 기존 유저는 1단계를 한 번 더 해야 `/analyze/stage1/latest`에 값이 생김
(그 전엔 `result: null`).

## 인증 연동

- 로그인(`/api/login`) 시 JWT 발급, payload에 `user_id` 포함, 7일 만료
- `get_current_user()` — `Authorization: Bearer <token>` 헤더 검증 후 `User` 반환.
  없거나 무효하면 `None`
- **2026-09 변경**: `/analyze/gaze-blink`가 원래 인증 검사를 안 했음 (JWT는
  발급하면서 아무도 검증 안 하는 상태였음). 이제 로그인 필수로 바뀜 — DB에
  저장하려면 `user_id`가 있어야 하니까 자연스럽게 같이 고쳐짐. 프론트 `/face`는
  이미 `PrivateRoute`로 로그인 필수였어서 실사용 흐름엔 영향 없음.

## `/analyze/gaze-blink` 저장 흐름

1. 인증 확인 (없으면 401)
2. 분석 실행
3. **저장 전에** 그 유저의 최신 `Stage2Result`를 조회해 `previous`로 확보
4. 이번 결과를 `Stage2Result`로 저장
5. 응답에 `previous` 필드 포함 → 프론트가 이걸로 "지난번 대비" 비교 텍스트 생성
   (`features/face/comparison.ts`, 로직 변경 없음 — 데이터 출처만 localStorage
   → API 응답으로 바뀜)

## 비밀 관리

- `DATABASE_URL`, `SECRET_KEY`를 코드 하드코딩 → 환경변수 필수로 전환
  (`backend/.env.example` 참고). 없으면 서버가 `RuntimeError`로 즉시 죽음
  (약한 기본값으로 조용히 넘어가지 않음)
- 기존에 코드에 있던 실제 DB 비밀번호는 과거 git 커밋 히스토리에 남아있음 —
  코드에서 지운 것과 별개로 **DB 비밀번호 자체 교체 필요** (아직 안 함)

## 안 한 것 / 다음 고려사항

- 마이그레이션 도구(Alembic 등) 없음 — 테이블 추가는 `create_all()`로 되지만
  기존 컬럼 변경/삭제는 수동으로 처리해야 함
- 2단계 기록 조회용 별도 `GET` 엔드포인트(히스토리 전체 조회) 없음 — 지금은
  "직전 1개"만 응답에 포함. 필요해지면 추가
- 1단계 DB 이전은 미착수
