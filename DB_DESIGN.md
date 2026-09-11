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

**1단계는 아직 DB 테이블 없음** — 여전히 localStorage(`localProgress.ts`)로
관리. 스코프 밖이라 이번엔 안 건드림. 1단계도 다회차 기록이 필요해지면 같은
패턴(`Stage1Result` 테이블)으로 옮기면 됨.

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
