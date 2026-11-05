# 멈춤(pause) 검출 로직 - 임계값 근거 정리

코드(`step1/analyze_pause.py`)에 쓰인 상수들의 근거를 정리한 문서.
발표/질의응답 때 "왜 이 값을 썼는지" 답할 때 참고.

---

## MIN_PAUSE_DURATION = 0.25초

**목적**: 이보다 짧은 무음은 분석 대상에서 제외 (음절 사이 조음 전환, 짧은 호흡
등 자연스러운 리듬으로 봄).

**근거**:
- Goldman-Eisler(1968) - 250ms 미만은 조음 쉼, 이상은 망설임 쉼으로 나눈 고전적
  기준 (De Jong & Bosker 2013의 인용으로 확인). 130~250ms 쉼도 조음만으로는 설명되지
  않는다는 반론(Hieke 외 1983)이 있다.
- De Jong & Bosker(2013) - 네덜란드어 L2 화자 51명에서 하한을 20~1000ms로 바꿔 보니
  250~300ms일 때 쉼 횟수가 능숙도와 가장 강하게 관련됐다("250 ms is a good choice").

---

## LONG_PAUSE_THRESHOLD = 1.2초

**목적**: 이 이상이면 "긴 멈춤"으로 분류해 점수에 반영.

**근거**

1. **한국어 자유발화의 보통 쉼은 0.5초 안팎** - Yoo & Shin(2019)은 서울 화자 48명의
   낭독과 자유발화를 비교했다. 자유발화의 발화 안 휴지는 평균 526ms(중앙값 424ms)였고,
   2초 이상의 휴지는 비유창한 발화로 보고 분석에서 뺐다. 1.2초는 보통 쉼의 두 배가 넘는다.

2. **1초를 넘는 쉼은 자유발화에서만 따로 묶인다** - Campione & Véronis(2002)는 5개 언어
   약 5.5시간(자유발화는 인터뷰 10개)의 쉼 약 6,000개를 분석해 짧은 쉼(<200ms),
   중간 쉼(200~1000ms), 긴 쉼(>1000ms)으로 나눴다. 긴 쉼은 낭독에는 없고 자유발화에서만
   나타났다. 1.2초는 이 긴 쉼 범주 안에 있다.

3. **우리 정상 발화와 비교** - 정상 발화 무음 구간 463개(26-1학기 분석): 평균 0.342초,
   평균+1σ = 0.597초. 1.2초는 평균+1σ의 약 2배라, 짧은 숨·생각 정리는 잡지 않고 실제로
   길게 끊긴 구간만 고른다.

4. **긴 침묵은 듣는 사람의 인상에 영향을 준다 (보조 근거)**
   - Roberts & Francis(2013): 380명이 200~1200ms 침묵을 들었을 때 600ms부터 평가가
     눈에 띄게 떨어졌다(질문과 답 사이 침묵, 영어). 1.2초는 이보다 너그러운 기준이다.
   - Brosy, Bangerter & Mayor(2016): 실제 채용면접에서 질문 뒤 응답 지연이 길수록 채용
     추천 점수가 낮았다(답변 중 쉼은 다루지 않음).
   - Liu 외(2022): 영어 낭독에 넣은 쉼은 0.6초가 가장 자연스럽고 2.4초 이상은 부자연스럽게
     평가됐다. 낭독 연구라 참고만 한다.

5. **실제 면접 영상 확인** - 1.60초·1.25초·1.79초 침묵 3건을 모두 긴 멈춤으로 잡았다.
   3건뿐이라 최적값의 증거는 아니며, 서비스 녹음이 쌓이면 다시 맞춘다.

---

## MIN_RATE_DURATION_SEC = 30초

**목적**: 분당 빈도와 멈춤 비율을 낼 때 답변 길이를 최소 30초로 본다.

**이유**: 5초짜리 답에서 멈춤 한 번이 분당 12번으로 계산돼 점수가 50점대로 떨어졌다
(지인 녹음 240개 점검, 6.5초 정상 녹음의 1.79초 멈춤 한 번 → 54점). 짧은 답 하나로 크게
틀린 지적을 하지 않도록 한다. 30초는 1단계 답변 길이를 고려한 자체 설정값이다.

---

## SILENCE_TOP_DB = 30dB

**의미**: 절대 음량 기준이 아니라, 파일 내 최대 소리 대비 상대 음량 기준
(`librosa.effects.split`). 파일 내 최대 참조 레벨보다 30dB 이상 조용한
구간을 무음으로 판단. 배경 소음이 크면(캘리브레이션 값 > -40dB) 35dB로 완화.

**설정 배경**: 조용한 실내에서 한 명이 모의면접 보는 서비스 환경을 고려한
초기값. 채움말 분석(`analyze_filler.py`)과 동일한 값을 써서 Step1 규칙
기반 음향 분석 기준을 통일함.

**한계**: 큰 소리, 마이크 충격음, PC 팬 소리, 약한 발성 등에 따라 결과가
달라질 수 있음 - 보편적 정답이 아니라 실제 서비스 녹음으로 조정할 파라미터.

---

## 점수식 (`_calculate_pause_score`)

100 − (분당 긴 멈춤 수 × 2 + 긴 멈춤 시간 비율 × 100). 문헌 근거가 있는 값이 아니라
오각형 차트용 자체 설정값이며, 서비스 녹음으로 다시 맞춘다.

---

## 점검 기록 (2026-11-06)

지인 녹음 240개(대본 읽기, 정상·작아짐·연장 앞·연장 가운데 각 60개)에서 긴 멈춤이 잡힌
녹음은 정상 1개(1.79초)뿐이었다. 대본 읽기라 긴 멈춤이 거의 없어야 정상이다.

---

## 해석상 원칙

**긴 멈춤은 불안을 단독으로 진단하는 기준이 아니다.** 1.2초 이상 무음은
"보통 쉼보다 길어 발화 흐름이 일시적으로 끊겼을 가능성이 높은 구간"으로만
해석하고, 연장·에너지 변동·떨림·채움말과 결합한 발화안정성의 보조 지표로만 쓴다.

사용자 화면 문구: "긴 멈춤" / "생각 정리 시간이 길어진 구간" / "말의 흐름이
잠시 끊긴 구간" - "불안한 멈춤", "면접 역량이 낮음"처럼 심리 상태나 능력을
단정하는 표현은 쓰지 않는다.

---

## 참고문헌

1. Goldman-Eisler, F. (1968). *Psycholinguistics: Experiments in Spontaneous Speech*. Academic Press. (2차 인용)
2. De Jong, N. H., & Bosker, H. R. (2013). Choosing a threshold for silent pauses to measure second language fluency. *Proceedings of DiSS 2013*, 17–20.
3. Yoo, D., & Shin, J. (2019). A realization of pauses in utterance across speech style, gender, and generation. *말소리와 음성과학*, 11(2), 33–44. https://doi.org/10.13064/KSSS.2019.11.2.033
4. Campione, E., & Véronis, J. (2002). A large-scale multilingual study of silent pause duration. *Speech Prosody 2002*, 199–202.
5. Roberts, F., & Francis, A. L. (2013). Identifying a temporal threshold of tolerance for silent gaps after requests. *JASA*, 133(6), EL471–EL477. https://doi.org/10.1121/1.4802900
6. Brosy, J., Bangerter, A., & Mayor, E. (2016). Disfluent responses to job interview questions and what they entail. *Discourse Processes*, 53, 371–391. https://doi.org/10.1080/0163853X.2016.1150769
7. Liu, S., Nakajima, Y., Chen, L., Arndt, S., Kakizoe, M., Elliott, M. A., & Remijn, G. B. (2022). How pause duration influences impressions of English speech: Comparison between native and non-native speakers. *Frontiers in Psychology*, 13, 778018. https://doi.org/10.3389/fpsyg.2022.778018
8. librosa development team. `librosa.effects.split` documentation.

---

## 발표용 한 줄 요약

> 한국어 자유발화의 보통 쉼은 0.5초 안팎이고(Yoo & Shin 2019), 1초를 넘는 쉼은 자유발화에서만
> 따로 나타난다(Campione & Véronis 2002). 이를 근거로 0.25초 이상을 멈춤, 1.2초 이상을 긴
> 멈춤으로 본다. 긴 멈춤은 불안 진단이 아니라 발화 흐름의 보조 지표이며, 서비스 녹음으로
> 계속 맞춘다.
