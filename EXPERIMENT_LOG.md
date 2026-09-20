# EXPERIMENT_LOG.md — Mission 3 실험 기록

Mission 3에서 수행한 실험과 결과를 기록한다.

목적은 단순히 점수를 적는 것이 아니라 다음 질문에 답할 수 있도록 하는 것이다.

- 무엇을 바꿨는가?
- 왜 바꿨는가?
- 무엇은 그대로 유지했는가?
- 결과가 어떻게 달라졌는가?
- 그래서 이 방법을 채택할 것인가?

실험 후에는 가능한 한 `ADOPT`, `REJECT`, `INCONCLUSIVE` 중 하나로 결론을 남긴다.

---

## 현재 baseline

- 문제: 119 신고 대화 기반 9개 증상 다중라벨 분류
- 입력: 신고 대화 전사 텍스트
- 모델: TF-IDF + One-vs-Rest Logistic Regression
- validation: multilabel stratified split
- validation 비율: 20%
- seed: 42
- 예측 threshold: 0.5
- 핵심 평가 지표: Macro F1
- 보조 지표:
  - Micro F1
  - 증상별 Precision
  - 증상별 Recall
  - 증상별 F1
  - 증상별 Support

---

## 실험 기록

### EXP-001 — TF-IDF + Logistic Regression baseline

- 유형/상태: `TRAIN + EVAL` / baseline 구현 완료

- 목적/가설:  
  복잡한 모델을 적용하기 전에 빠르고 설명 가능한 기준 모델을 만든다.

  이후 KcBERT나 다른 개선 방법의 성능이 실제로 좋아졌는지를 판단하기 위한 기준선으로 사용한다.

---

### 데이터

Training 라벨 JSON을 사용한다.

각 JSON에서 다음 구조의 전사 텍스트를 읽는다.

`utterances[].text`

여러 발화가 존재하면 원래 순서대로 하나의 문자열로 연결한다.

---

### 모델 입력

사용:

- 전사 텍스트

사용하지 않음:

- speaker
- address
- urgencyLevel
- symptom
- 기타 메타데이터

정답 `symptom`은 모델 학습의 label로만 사용한다.

---

### 대상 라벨

총 9개 증상:

- 고열
- 구토
- 두통
- 복통
- 어지러움
- 열상
- 오심
- 전신쇠약
- 호흡곤란

---

### 데이터 점검

학습 전 다음 항목을 확인하도록 구현했다.

- 전체 sample 수
- 빈 전사 텍스트 수
- 9개 target label이 하나도 없는 sample 수
- JSON 읽기 실패 수
- 증상별 데이터 수
- 평가 대상이 아닌 symptom의 종류와 개수

---

### Train / Validation split

방법:

`MultilabelStratifiedShuffleSplit`

설정:

- validation 비율: 20%
- seed: 42

목적:

다중라벨 데이터에서 특정 증상이 train 또는 validation 한쪽에 지나치게 몰리는 현상을 줄인다.

주의:

현재 구현은 **JSON 파일 단위 split**이다.

따라서 하나의 실제 통화가 여러 JSON으로 나뉘어 있는 데이터라면 통화 단위 leakage 여부를 추가로 확인해야 한다.

---

### TF-IDF 설정

```text
analyzer = word
ngram_range = (1, 2)
min_df = 2
max_features = 50000
sublinear_tf = True
```

의미:

- 한 단어뿐 아니라 두 단어 조합까지 사용
- 지나치게 희귀한 표현은 제외
- 최대 feature는 50,000개로 제한
- 매우 자주 등장하는 단어의 영향력이 지나치게 커지는 것을 완화

---

### 분류 모델

One-vs-Rest Logistic Regression

즉 9개 증상에 대해 각각

> “이 증상이 있는가 / 없는가?”

를 판단하는 Logistic Regression 모델을 하나씩 학습한다.

설정:

```text
max_iter = 1000
class_weight = balanced
solver = liblinear
random_state = 42
```

---

### Prediction

각 증상에 대해 예측 확률을 계산한다.

기본 threshold:

```text
0.5
```

예:

```text
호흡곤란 확률 = 0.82 → 호흡곤란 있음
두통 확률 = 0.31 → 두통 없음
```

하나의 통화에 여러 증상이 동시에 0.5 이상일 수 있으므로 여러 증상을 동시에 예측할 수 있다.

---

### 평가 지표

전체:

- Precision Macro
- Recall Macro
- Precision Micro
- Recall Micro
- Macro F1
- Micro F1

증상별:

- Support
- Precision
- Recall
- F1

주요 비교 지표:

**Macro F1**

이유:

9개 증상을 각각 동일한 중요도로 평가하기 위해서다.

---

### 결과

현재 코드에는 다음 결과를 계산하는 기능이 구현되어 있다.

```text
Train samples
Validation samples
Precision (macro)
Recall (macro)
Precision (micro)
Recall (micro)
Macro F1
Micro F1
```

단, 현재 제공된 자료에는 실제 실행 후 나온 최종 Macro F1 값이 포함되어 있지 않으므로 이 문서에는 임의로 수치를 작성하지 않는다.

---

### 저장되는 산출물

#### 1. 모델

`mission3_tfidf_logreg.joblib`

포함 내용:

- TF-IDF vectorizer
- Logistic Regression classifier
- 9개 target symptom
- threshold
- seed
- model name

#### 2. 전체 평가 지표

`metrics.json`

#### 3. 증상별 평가 지표

`per_label_metrics.csv`

포함:

```text
symptom
support
precision
recall
f1
```

#### 4. Validation 예측 결과

`validation_predictions.csv`

포함:

```text
label file name
true symptom
predicted symptom
```

---

### 판정

`INCONCLUSIVE`

이유:

baseline 모델 구현과 평가 파이프라인은 완료되어 있지만, 현재 문서에 실제 실행 결과가 연결되어 있지 않기 때문에 성능 자체를 평가할 수 없다.

실제 Macro F1과 증상별 결과를 확보한 뒤 baseline 수치를 확정한다.

---

### 다음 행동

1. baseline을 실제 Training 데이터에서 실행한다.
2. `metrics.json`의 Macro F1을 기록한다.
3. `per_label_metrics.csv`에서 증상별 성능을 확인한다.
4. 성능이 낮은 증상의 `validation_predictions.csv`를 확인한다.
5. 오류 원인에 따라 다음 실험 하나를 선택한다.

---

## EXP-002 — 증상별 오류 분석

- 유형/상태: `ANALYSIS` / 실행 예정

- 목적/가설:  
  전체 Macro F1만 보는 것이 아니라 TF-IDF가 어떤 증상과 어떤 표현에서 실패하는지 확인한다.

### 확인할 내용

| 증상 | Support | Precision | Recall | F1 | 대표 오류 표현 | 추정 원인 |
|---|---:|---:|---:|---:|---|---|
| 고열 |  |  |  |  |  |  |
| 구토 |  |  |  |  |  |  |
| 두통 |  |  |  |  |  |  |
| 복통 |  |  |  |  |  |  |
| 어지러움 |  |  |  |  |  |  |
| 열상 |  |  |  |  |  |  |
| 오심 |  |  |  |  |  |  |
| 전신쇠약 |  |  |  |  |  |  |
| 호흡곤란 |  |  |  |  |  |  |

### 특히 확인할 오류

#### 1. 표현 변화

예:

```text
호흡이 곤란해요
숨이 안 쉬어져요
숨쉬기가 힘들어요
```

의미는 비슷하지만 사용 단어가 다르기 때문에 word TF-IDF에서 다른 표현으로 인식될 가능성이 있다.

#### 2. 부정 표현

예:

```text
구토는 안 했어요.
```

`구토`라는 단어가 등장했다는 이유만으로 구토를 예측하는 오류가 발생하는지 확인한다.

#### 3. 문맥 의존 표현

증상 단어 자체가 직접 등장하지 않고 대화의 전체 문맥을 이해해야 판단할 수 있는 사례를 확인한다.

---

### 결과

미실행

### 판정

`INCONCLUSIVE`

---

## EXP-003 — TF-IDF vs KcBERT 비교

- 유형/상태: `EVAL` / 예정

- 목적/가설:

KcBERT가 TF-IDF보다 문맥과 다양한 표현을 더 잘 이해하는지 확인한다.

단순히 전체 Macro F1만 비교하지 않고 두 모델의 오류 차이를 함께 분석한다.

---

### 비교 조건

가능하면 동일한 validation 데이터를 사용한다.

비교 지표:

- Macro F1
- 증상별 F1
- 증상별 Recall
- 증상별 Precision

추가 비교:

- TF-IDF만 맞힌 사례
- KcBERT만 맞힌 사례
- 두 모델 모두 맞힌 사례
- 두 모델 모두 틀린 사례

---

### 핵심 질문

두 모델의 오류가 서로 다른가?

만약 서로 다른 유형의 오류를 보인다면 이후 soft voting 등의 앙상블을 검토할 근거가 된다.

반대로 KcBERT가 대부분의 사례에서 TF-IDF를 포함해 더 좋은 결과를 낸다면 억지로 모델을 합칠 필요는 없다.

---

### 결과

미실행

### 판정

`INCONCLUSIVE`

---

## 후속 실험 후보

오류 분석 결과에 따라 **한 번에 하나만 변경한다.**

### 후보 1. Character TF-IDF

현재:

```text
word TF-IDF
```

변경:

```text
character TF-IDF
```

가설:

한국어에서는 조사·어미·띄어쓰기와 다양한 표현 때문에 문자 또는 음절 단위 특징이 표현 변화에 더 강할 수 있다.

---

### 후보 2. Threshold 조정

현재:

```text
threshold = 0.5
```

가설:

Recall이 낮은 증상은 threshold를 조금 낮추면 더 많이 탐지할 수 있다.

단, Precision 하락 여부를 함께 확인한다.

---

### 후보 3. class_weight ablation

현재:

```text
class_weight = balanced
```

비교:

```text
class_weight = None
```

목적:

현재 적용한 class weight가 실제 Macro F1 개선에 도움이 되었는지 확인한다.

---

## 실험 기록 원칙

새로운 실험에서는 다음 항목을 먼저 정한다.

```text
가설
baseline
변경 변수
고정 변수
평가 지표
validation
채택 기준
```

예를 들어 character TF-IDF를 시험한다면:

```text
가설:
character TF-IDF가 표현 변화에 더 강할 것이다.

baseline:
word TF-IDF

변경:
TF-IDF analyzer만 word → char로 변경

고정:
train/validation split
Logistic Regression
seed
threshold

평가:
Macro F1

판정:
Macro F1과 오류 분석을 기준으로 ADOPT / REJECT / INCONCLUSIVE
```

모델, split, threshold 등을 한 실험에서 동시에 변경하지 않는다.

그래야 성능이 달라졌을 때 **무엇 때문에 달라졌는지 설명할 수 있다.**
