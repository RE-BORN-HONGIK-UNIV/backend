# 불안도·난이도와 에너지 규칙

## 불안도 API

`GET /interview/anxiety` (로그인 필요, `Authorization: Bearer <token>`)

```json
{
  "score": 42.3,
  "tier": "standard",
  "items": { "pause": 70, "filler": 80, "prolong": 100, "energy": 90, "tremor": 100,
             "gaze": 55, "blink": 100, "tension": 92 }
}
```

- `tier`: `warmup`(편안) / `standard`(보통) / `practice`(도전). 화면은 이 값만 쓰면 된다. 프론트 `difficulty.ts`의 `DifficultyTier`와 이름이 같다.
- `score`: 불안도 0~100, **높을수록 긴장**. 사용자에게 보여 주지 않는다.
- `items`: 계산에 쓴 항목 점수(0~100, 높을수록 안정). 1단계나 2단계 기록이 없으면 그 단계 항목이 빠지고 남은 항목으로 계산한다. 둘 다 없으면 `score`는 `null`, `tier`는 `warmup`.

## 계산식 (`step3/anxiety.py`, 판넬과 같음)

불안도 = Σ 가중치 × (100 − 항목 점수) ÷ (값이 있는 항목의 가중치 합)

| 등급 | 항목 | 저장된 값 | 가중치 | 근거 |
|---|---|---|---|---|
| 핵심 | 침묵 조절력 (긴 멈춤) | 1단계 pause_ctrl | 2 | 사회불안 환자 발표에서 상태 불안과 침묵 비율 r = .36 (Laukka et al. 2008, Teferra et al. 2022의 인용으로 확인) |
| 핵심 | 시선 피함 | 2단계 avg_fixation_sec (3초 이상 바라보면 100, 짧을수록 깎음) | 2 | 시선 피함 → 채용 의향·신뢰도 하락 (Burgoon et al. 1985), 실제 면접 녹화에서 면접관 응시 → 채용 판단 β = .40 (Frauendorfer et al. 2014) |
| 보조 | 발화 유창성 (채움말) | 1단계 fluency | 0.5 | 채움말이 많을수록 자신감·유능함 평가 하락 (Kirkland et al. 2023), 평가 불안·자의식이 'um'을 늘림 (Christenfeld & Creager 1996) |
| 보조 | 발화 에너지 (목소리 작아짐) | 1단계 calm | 0.5 | 불안할수록 평균 음량이 약간 낮음 (r ≈ −.2, Laukka et al. 2008 인용). "점점 작아짐"을 직접 잰 연구는 없음 |
| 보조 | 음성 안정성 (떨림) | 1단계 stability | 0.5 | 발표 불안 척도에 "목소리가 떨리면 끔찍할 것" 문항 (Cho et al. 2004). 음향 지표 연구는 방향이 엇갈림. 모델은 시험 통과(PR-AUC 0.77, 찍기 0.22) |
| 보조 | 발화 지속성 (말 늘이기) | 1단계 continuity | 0.5 | 직접 연구 없음. 실제 면접 영상에서 자주 보인 습관이라 낮은 비중으로만 넣음 |
| 보조 | 잦은 깜빡임 | 2단계 blink_rate_per_min (분당 30회 넘을 때만 깎음) | 0.5 | 감정 자극 중 상태 불안과 r = .42 (Tomashin et al. 2025). 말하는 상황 연구는 아님 |
| 보조 | 미간 긴장 | 2단계 100 × (1 − tension_ratio) | 0.5 | 불안과 직접 연결한 연구는 찾지 못함. 2단계 검증표의 긴장 판정을 씀 |
| 제외 | 미소 | — | 0 | 미소는 따뜻함↑(Tescari & Bangerter 2024)이지만 진지한 직무에선 덜 웃는 쪽이 유리(Ruben et al. 2015)해 방향이 엇갈림 |

- 핵심 두 항목이 전체 가중치의 약 57%다(4 ÷ 7).
- 시선·깜빡임은 화면 점수를 그대로 쓰지 않는다. 화면 점수는 인상 기준이라 5초 넘게 바라보거나 깜빡임이 적어도 깎지만, 불안 근거는 시선을 피할 때와 깜빡임이 잦을 때뿐이다. 방향이 다른 값을 섞으면 오래 잘 바라본 사람이 긴장한 것으로 계산된다.
- 목소리 5개 항목은 1단계 녹음과 2단계 통화 녹음 중 더 긴장한 쪽(낮은 점수)을 쓴다. 2단계는 상대가 보이는 상황이라 3단계 면접에 더 가깝고, 평균을 내면 그 신호가 절반으로 줄어든다. 판단이 틀려도 쉬운 난이도로 가는 방향이기도 하다. 2단계 녹음이 없으면 1단계 값만 쓴다.
- 가중치와 난이도 경계(20 · 5)는 초기값이고, 사용자 테스트로 조정한다.

### 난이도 경계 20 · 5를 정한 방법

불안도는 항목들의 "문제 정도" 평균이라, 대부분 항목이 정상(100점)인 보통 사람은 낮게 나온다. 멈춤·시선이 괜찮으면 목소리 보조 항목 4개(채움말·말 늘이기·작아짐·떨림)가 모두 최악이어도 약 29점이라, 판넬의 초기값 60 · 30으로는 거의 모두 "도전"이 된다.

2026-11-08에 CNN 세 항목(늘이기·작아짐·떨림) 점수를 "가장 높은 3초 조각 하나"에서 "기준값을 넘은 조각 비율"로 바꿨다(`step1/stage1_score.window_score`). 예전 방식은 1단계 대화처럼 긴 녹음에서 조각 하나만 넘어도 0점 가까이 떨어져, 평소처럼 말해도 불안도가 32까지 올랐다. 바뀐 점수로 다시 정했다.

정한 방법: 학습에 쓰지 않은 지인 22명의 녹음을 사람마다 이어 붙여 긴 대화를 만들고(평소 말만 / 늘이기·작아짐 녹음까지), 서버와 같은 코드로 목소리 5개 점수를 냈다. 시선은 "편하게 바라봄(100점)"과 "평균 1초만 바라보고 피함"을 가정했다.

| 경우 | 불안도 중앙값 (25~75%) | 최대 |
|---|---|---|
| 평소 말 + 바라봄 | 0 (0 ~ 2.7) | 10.2 |
| 습관 섞인 말 + 바라봄 | 10.2 (5.7 ~ 11.6) | 15.8 |
| 평소 말 + 시선 피함 | 21.9 (21.9 ~ 24.6) | 32.1 |
| 습관 섞인 말 + 시선 피함 | 32.0 (27.6 ~ 33.5) | 37.7 |

| 난이도 | 경계 | 대략 어떤 경우 |
|---|---|---|
| 편안 (warmup) | 20 이상 | 시선을 자주 피함 |
| 보통 (standard) | 5 ~ 20 | 바라보지만 습관이 꽤 있음 |
| 도전 (practice) | 5 미만 | 편하게 바라보고 습관이 적음 |

한계: 지인 녹음은 짧은 대본 문장을 이어 붙인 것이고, 지인들이 "음, 어"를 거의 쓰지 않아 채움말이 반영되지 않았으며, 시선·깜빡임·표정 값은 가정이다. 실제 사용자 기록이 쌓이면 다시 맞춘다. 경계가 틀리면 쉬운 쪽으로 실수하도록 낮게 잡았다.

### 왜 이 가중치 방식인가

| 방식 | 판단 |
|---|---|
| 같은 비중 | 교과서적 방법이다(Dawes 1979; Bobko et al. 2007). 하지만 근거가 약한 발화 지속성·미간도 멈춤과 같은 무게가 된다 |
| **근거 등급 비중 (채택)** | 근거가 비교적 강한 멈춤·시선 2, 나머지 0.5. 숫자 자체는 초기값이라고 밝힌다 |
| 논문 효과 크기를 그대로 비중으로 | 연구마다 대상·과제·장비가 달라 숫자를 섞으면 정밀한 척하게 된다 |
| 실제 데이터로 학습 | 사용자 자기 보고 긴장도(0~10)와 함께 쌓이면 v2에서 비중을 검증·조정한다 |

- 측정이 틀려 불안도가 높게 나오면 더 쉬운 난이도가 된다. 이 서비스에선 안전한 쪽의 실수다.
- 한계: 항목마다 점수가 움직이는 폭이 달라(CNN 점수는 0~100, 멈춤은 대체로 70~100) 실제 영향력이 2 : 0.5와 정확히 같지는 않다.
- 한계(시선): 시선 항목은 `avg_fixation_sec`(한 번 바라볼 때 평균 몇 초)를 쓴다. 바라본 시간의 **비율**이 아니라서, 4초 보고 20초 피하기를 반복해도 평균 4초로 "문제 없음"이 된다. 또 시선 판정 기준(머리 방향 10도)은 영상 하나로 정해 정확도 검증이 아직 없고, 실시간 통화와 영상 업로드는 계산 방식이 달라 값을 그대로 비교할 수 없다. 2단계에 "바라본 시간 비율"을 저장하게 되면 그 값으로 바꾼다(v2).

### 기존 계산식과의 차이

| | 기존 (`difficulty.ts`) | 새 (`/interview/anxiety`) |
|---|---|---|
| 재료 | 1단계 종합 점수, 2단계 종합 점수(깜빡임·시선·표정 평균) | 항목 8개를 따로 |
| 비중 | 1단계 0.6 : 2단계 0.4 (근거 없음) | 근거 등급 2 : 0.5, 미소 제외 |
| 점수 방향 | 높을수록 안정 | 높을수록 긴장 |
| 난이도 경계 | 40 미만 warmup, 70 미만 standard, 이상 practice | 20 이상 warmup, 5~20 standard, 5 미만 practice |

### 참고문헌

- Bobko, P., Roth, P. L., & Buster, M. A. (2007). The usefulness of unit weights in creating composite scores. *Organizational Research Methods*, 10(4), 689–709.
- Burgoon, J. K., Manusov, V., Mineo, P., & Hale, J. L. (1985). Effects of gaze on hiring, credibility, attraction and relational message interpretation. *Journal of Nonverbal Behavior*, 9(3), 133–146. https://doi.org/10.1007/BF01000735
- Cho, Y., Smits, J. A. J., & Telch, M. J. (2004). The Speech Anxiety Thoughts Inventory. *Behaviour Research and Therapy*, 42, 13–25.
- Christenfeld, N., & Creager, B. (1996). Anxiety, alcohol, aphasia, and ums. *Journal of Personality and Social Psychology*, 70(3), 451–460.
- Dawes, R. M. (1979). The robust beauty of improper linear models in decision making. *American Psychologist*, 34(7), 571–582.
- Frauendorfer, D., Schmid Mast, M., Nguyen, L. S., & Gatica-Perez, D. (2014). Nonverbal social sensing in action. *Journal of Nonverbal Behavior*, 38(2), 231–245. https://doi.org/10.1007/s10919-014-0173-5
- Kirkland, A., Gustafson, J., & Székely, É. (2023). Pardon my disfluency. *Proc. Interspeech 2023*, 5217–5221.
- Laukka, P., Linnman, C., Åhs, F., et al. (2008). In a nervous voice: Acoustic analysis and perception of anxiety in social phobics' speech. *Journal of Nonverbal Behavior*, 32(4), 195–214. (수치는 Teferra et al. 2022의 인용으로 확인)
- Teferra, B. G., et al. (2022). Acoustic and linguistic features of impromptu speech and their association with anxiety: Validation study. *JMIR Mental Health*, 9(7), e36828.
- Ruben, M. A., Hall, J. A., & Schmid Mast, M. (2015). Smiling in a job interview: When less is more. *The Journal of Social Psychology*, 155(2), 107–126.
- Tescari, M.-E., & Bangerter, A. (2024). Nonverbal behavior in selection interviews: Relation to communion, agency, and interview performance. *Journal of Personnel Psychology*. https://doi.org/10.1027/1866-5888/a000350
- Tomashin et al. (2025). *PLOS ONE*, 20(12), e0338262. https://pmc.ncbi.nlm.nih.gov/articles/PMC12694841/

## 에너지(작아짐) 조각 판정 규칙 (`step1/energy_rule.py`)

CNN이 작아짐이라고 한 3초 조각 중 두 경우를 빼고 인정한다. `models/energy/config.json`의 `fade_db`를 지우면 둘 다 꺼진다.

1. 같은 조각을 연장 모델이 연장으로 잡았으면 뺀다. 작아짐 없는 긴 답변의 오탐이 모두 "정상 + 연장" 조합이었다. CNN이 말 늘이기를 작아짐으로 헷갈린다.
2. 조각 안에서 앞 1/3보다 뒤 1/3이 2 dB만큼도 작아지지 않았으면 뺀다.

- 왜 조각 단위인가: 녹음 전체의 앞뒤를 비교하면 여러 문장으로 된 긴 답변에서 중간에 작아진 것을 놓친다.
- 왜 이 조합인가: 틀리게 지적하는 것은 사용자에게 해롭지만, 진짜 습관을 자주 놓치면 연습에 도움이 안 된다. 검증에서 비교한 방식 중 틀린 지적은 줄이면서 놓치는 것은 늘지 않은 것이 이 조합뿐이었다.

| 검증 (한 문장 70개 중 작아짐 12 / 이어 붙인 긴 답변 21개) | CNN만 | 조각 안 6 dB | **연장 조각 제외 + 2 dB** |
|---|---|---|---|
| 찾아냄 | 12/12 | 9/12 | **12/12** |
| 작아짐이라고 한 것 중 맞음 | 12/17 | 9/11 | **12/14** |
| 긴 답변 속 작아짐 잡음 | 13/13 | 9/13 | **13/13** |
| 작아짐 없는 긴 답변 오탐 | 5/8 | 3/8 | 4/8 |
| PR-AUC | 0.91 | 0.75 | **0.94** |

시험지(학습·검증에 안 쓴 15명, 이 방식으로 한 번 채점): PR-AUC **0.62** [0.48–0.90] (찍기 0.24, 통과), 찾아냄 13/14, 맞음 13/23, 작아짐 없는 긴 답변 오탐 11/13 → 7/13 (CNN만 대비). 녹음 전체 앞뒤 6 dB 규칙은 0.60이었다.

남은 약점: 긴 답변 오탐이 아직 절반쯤이다. 리포트 문구는 단정하지 않게 쓰고, 여러 답변에서 반복될 때 습관으로 알려 주는 쪽이 맞다.
