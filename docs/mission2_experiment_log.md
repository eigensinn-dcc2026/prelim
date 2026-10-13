# Mission 2 실험 기록

> 이 문서는 Mission 2의 진행 중 실험 기록이다. 최종 모델 보고서가 아니며, 새로운 실험 결과와 결정이 검증될 때마다 누적 갱신한다.

최종 수정일: 2026-10-12

결과 표기의 우선순위는 **저장된 Notebook output → persisted artifact → source code → Git branch/commit → 설명 문서**다. 같은 ID라도 평가 집단이나 모델이 다르면 별도 결과로 기록한다. `canonical`은 다음 비교의 기준이라는 뜻이며, `final model`을 뜻하지 않는다.

## 0. 현재 실험 계약

| 항목 | 현재 계약 |
|---|---|
| 문제 | `startAt`~`endAt`으로 잘린 한 사람의 한 발화 음성에서 `speaker`를 분류 |
| 허용 입력 | 현재 분류 대상 발화의 waveform, 팀 내부 확정 정책상 그 발화 자신의 `duration = endAt - startAt` |
| 금지 입력 | `text`, 다른 annotation, 발화 순서/홀짝/index와 그 proxy, 같은 call의 다른 발화 정보, call-relative 통계, 다른 발화의 acoustic/prediction 정보, `startAt`/`endAt` 절대값 |
| 전체 제공 규모 | Training 통화 29,200개 / 발화 889,358건, 공식 검증 통화 3,640개 / 발화 111,947건 |
| 현재 internal split | Training만 사용한 고정 `call_id` GroupSplit: train 711,467 행 / 23,360 통화, validation 177,891 행 / 5,840 통화; call overlap 0 |
| 시드 | 42 |
| 평가 지표 | 공식 1차 metric Accuracy; 보조 진단은 confusion matrix, speaker별 Recall, confidence/HCE, duration, overlap, paired prediction change |
| 전처리 | 학습되는 `StandardScaler`는 해당 비교의 Training 또는 internal-training에만 fit하고 평가 집합에는 그대로 적용 |
| 공식 검증의 역할 | 초기 baseline/E-series 및 M00/M01/M04-30 확인에 사용. gradient update에는 사용하지 않음 |
| 내부 검증의 역할 | 현재 feature/model 반복 비교의 주 평가 집합. 동일 payload/split/scaler/model 계약 안에서만 Delta 계산 |
| 현재 수작업 특징 비교 기준 | **clean 59D** = clean MFCC40 + current v2 E05 2D + AF02 5D + AF04 2D + AF09 3D + AF10 4D + AF11 3D; 다음 representation 비교의 기준이며 final representation이 아님 |
| 현재 비교 기준 모델 | **59D 이중 동결 비교 기준: TabM + Tabular ResNet**. Stage 6 Stronger MLP는 복잡한 정형 모델 비교의 직접 기준으로 보존; Stage 4 M04-100은 당시 비교 기준으로 보존 |
| 현재 내부 검증 기준 결과 | TabM `0.8839795155460366`, errors `20,639`; Tabular ResNet `0.8837546587517076`, errors `20,679`. 동일 `177,891` internal-validation 행; 상세 CM·gap·paired 결과는 §4.11 |
| 현재 공식 검증 결과 | 현재 59D dual reference의 Official 결과는 없음. 과거 40D/42D Official 결과와 혼동하지 않음 |
| Stage 11 최종 상태 | **SKIP — team duplication avoidance(팀 중복 회피)**. 팀원 김광희가 시간×주파수 표현 계열을 별도로 탐색 중이므로 개인 실험 중복을 피함 |
| Stage 12 최종 상태 / 성능 | **CLOSED — resource/time feasibility(자원/시간 문제로 종료)** / **PERFORMANCE NOT EVALUATED(성능 미평가)**. Stage 12-A WavLM Base+와 Stage 12-B DPWavLM을 별도 실험으로 보존 |
| 최종 후보 / 다음 단계 | 사용자 결정으로 Stage 14의 **E_TRG (TabM + Tabular ResNet + GANDALF, 동일가중치 1/3씩)** 채택. 이는 반복 사용된 Internal Validation 기반의 현재 채택 후보이며 Official Validation·Stage 15 튜닝은 아직 수행하지 않음 |

공정 비교 gate는 동일 row/key/label, 동일 GroupSplit·seed·evaluation set, internal-train-only scaler, 비교하지 않는 model 또는 feature의 고정이다. 조건이 다른 Internal과 Official 사이에는 Delta를 계산하지 않는다.

## 1. 기준 모델

### E00 — MFCC40 + LogisticRegression

- **질문:** 현재 발화의 기본 MFCC 통계만으로 speaker 역할을 어느 정도 분류할 수 있는가?
- **입력/특징:** 8 kHz waveform에서 MFCC 20계수의 발화별 mean 20D + std 20D = 40D. text, 발화 순서, 다른 발화 정보는 사용하지 않았다.
- **모델/전처리:** `StandardScaler`를 전체 Training에만 fit한 뒤 `LogisticRegression(solver="lbfgs", max_iter=1000, class_weight=None, random_state=42)`를 Training에 fit하고 Official Validation에 평가했다.
- **평가:** 이 최초 baseline에는 현재의 fixed internal GroupSplit 결과가 보존되어 있지 않다. 아래 수치는 Official Validation 111,947 rows 결과다.

| 정확도 | 혼동행렬 | 화자 0 재현율 | 화자 1 재현율 | 정답 / 오류 |
|---:|---|---:|---:|---:|
| `0.8091060948484551` | `[[41190,12655],[8715,49387]]` | `0.7649735351` | `0.8500051633` | 90,577 / 21,370 |

- **관찰:** speaker 0 Recall이 speaker 1보다 낮고, 오류가 짧은 발화와 overlap 구간에서 더 많았다. confidence가 높은 오류도 2,086건 존재했다.
- **판단/다음 단계:** 40D MFCC만으로 놓치는 시간·스펙트럼 정보가 있는지 E-series를 통해 한 feature block씩 확인하고, 별도 error-analysis에서 취약 slice를 구체화했다.
- **근거:** `e670503:notebooks/mission2_yeonwoo_baseline.ipynb`; stronger exact reproduction은 `26d48b6:notebooks/mission2_baseline_error_analysis.ipynb`, cell `ea-cp1-gate`.

**기준 모델 → 오류 분석 → 특징 질문:** baseline error analysis가 입증한 것은 짧은 발화·overlap·speaker 0에서 오류가 더 많다는 **현상**이지, 특정 feature가 원인이나 해법이라는 결론은 아니었다. E01은 EDA에서 관찰된 speaker별 duration 차이가 MFCC40 위에 실제 증분 정보를 주는지 직접 확인했고, E02~E04는 MFCC40에 명시적으로 없는 waveform-only spectral/zero-crossing 요약을 한 번에 하나씩 검증했다. 이후 E05는 단일 후보의 Training-only 오류 보완성과 중복도를 먼저 비교한 뒤 첫 조합 질문으로 넘어갔다.

## 2. 특징 실험

E01~E05는 전체 Training에 fit하고 **Official Validation**에 평가한 역사적 LR feature 실험이다. E01/E05는 저장 실행 output, E02/E03/E04는 커밋된 Markdown 결과만 남아 있어 증거 등급이 다르다.

| ID | 질문 / 변경 | 비교 기준 | 차원 | 공식 검증 정확도 | 변화량 | 화자 0 / 1 재현율 | CM / HCE | 판단 / 다음 단계 | 근거 |
|---|---|---|---:|---:|---:|---|---|---|---|
| E01 | 발화 길이가 MFCC40에 정보를 더하는가? `duration_ms` 1D 추가 | E00 | 41 | `0.8091150276` | `+0.0000089328` | `0.765624 / 0.849420` | `[[41225,12620],[8749,49353]]`; HCE 2,078 | **DROP** — 실질 향상 없음 | feature NB cell `m2-fe-cpc-run` |
| E02 | 고주파 에너지 경계를 보완하는가? `spectral_rolloff` 1D 추가 | E00 | 41 | `0.8115090177` | `+0.0024029228` | `0.769969 / 0.850005` | `[[41459,12386],[8715,49387]]`; HCE 2,123 | **KEEP as candidate** — E05 조합으로 이동 | committed Markdown `m2-fe-e03-e04-precommit`; rendered output 없음 |
| E03 | 무성/마찰 및 전이 정보를 보완하는가? `zcr` 1D 추가 | E00 | 41 | `0.8101155011` | `+0.0010094062` | `0.769152 / 0.848078` | CM 미보존; HCE 2,102 | **HOLD** — 단독 gain 작음, E05 조합에서 재검토 | committed Markdown `m2-fe-e03-template`; rendered output 없음 |
| E04 | spectral 중심 위치가 도움이 되는가? `spectral_centroid` 1D 추가 | E00 | 41 | `0.8104996114` | `+0.0013935166` | `0.765865 / 0.851864` | CM 미보존; HCE 2,102 | **HOLD** — class trade-off, canonical에 미포함 | committed Markdown `m2-fe-e04-template`; rendered output 없음 |
| E05 | E02와 E03의 상보성이 있는가? `spectral_rolloff + zcr` 2D 추가 | E00 | 42 | `0.8131169214` | `+0.0040108266` | `0.774389 / 0.849007` | `[[41697,12148],[8773,49329]]`; HCE 2,158 | **KEEP** — model screen의 42D representation으로 승격 | feature NB cell `m2-fe-cpf-run` |

**선정 근거 — 보존된 근거만 요약:**

- **E01:** EDA의 speaker별 duration 차이가 허용된 현재 발화 duration만으로 추가 signal이 되는지 확인했다. 전체 gain이 사실상 0이고 짧은 구간도 개선되지 않아 DROP했다.
- **E02:** Training-only 후보 비교에서 가장 큰 단일 gain을 보인 `spectral_rolloff`로 MFCC40에 없는 고주파 에너지 경계를 보완하려 했다. Official에서도 같은 방향의 modest gain이 재현되어 조합 후보로 유지했지만 HCE 증가는 위험으로 남겼다.
- **E03:** Training-only 순위에서 다음 후보로 사전 확정된 `zcr`로 거칠기·잡음성/zero-crossing 정보를 보완하려 했다. 단독 gain은 작았지만 rolloff와의 오류 보완성을 별도로 확인하기 위해 HOLD했다.
- **E04:** Training-only 분석에서 class separation이 크고 MFCC 중복도가 중간 수준이던 `spectral_centroid`로 평균 spectral 위치를 보완하려 했다. 소폭 gain과 class trade-off만 확인되어 canonical에는 넣지 않았다.
- **E05:** Checkpoint H에서 rolloff가 가장 큰 net correction을 보였고, zcr가 centroid보다 rolloff와의 feature correlation·error overlap이 낮아 첫 조합으로 사전 확정됐다. E02 대비 추가 gain은 확인됐지만 speaker 1 Recall/HCE trade-off를 남긴 채 42D model-screen reference로만 승격했다.

E05는 E02보다 `+0.0016079037` Accuracy, 180 correct rows 증가했다. 다만 HCE는 E00보다 72건 늘었으므로 Accuracy 향상만으로 오류 품질이 전부 좋아졌다고 해석하지 않는다. E02~E04의 숫자는 저장 실행 output이 없는 역사 기록이므로 재현성 동결 결과가 아니다.

## 3. 모델 실험

M00~M04의 internal screen은 동일한 E05 42D payload, fixed GroupSplit, seed 42, internal-train-only scaler를 사용해 **model만 변경**했다. family 선택 이유도 결과 확인 전에 고정했다.

- **M00 LR:** 특징 효과와 비교할 수 있는 단순한 선형 비교 기준.
- **M01 HGB:** 순차 부스팅 트리가 42D 특징 사이의 비선형 상호작용을 포착하는지 확인.
- **M02 ExtraTrees:** 무작위 독립 트리의 평균화가 순차 부스팅과 다른 상호작용을 포착하는지 확인.
- **M03 QDA:** 화자별 가우시안 공분산 가정과 이차 결정 경계가 맞는지 비교.
- **M04 MLP:** 하나의 ReLU 은닉층이 연속형 음향 특징의 분산된 비선형 상호작용을 포착하는지 확인.

| ID | 모델 / 변경 | 내부 검증 정확도 | CM | 화자 0 / 1 재현율 | HCE | 판단 |
|---|---|---:|---|---|---:|---|
| M00 | LogisticRegression | `0.8173319625` | `[[66321,19030],[13465,79075]]` | `0.777038 / 0.854495` | 3,383 | 모델 비교 기준 |
| M01 | HistGradientBoosting | `0.8418750811` | `[[68036,17315],[10814,81726]]` | `0.797132 / 0.883142` | 1,965 | **KEEP** 당시 tree candidate |
| M02 | ExtraTrees | `0.8199965150` | 표에는 미보존; 건수로 재구성 가능 | `0.783166 / 0.853966` | 50 | **DROP** — M01보다 낮음 |
| M03 | QDA | `0.7973140860` | 표에는 미보존; 건수로 재구성 가능 | `0.828543 / 0.768511` | 13,846 | **DROP** — Accuracy/Recall 균형/HCE 악화 |
| M04-30 | MLP `(64,)`, `max_iter=30` | `0.8579523424` | `[[68806,16545],[8724,83816]]` | `0.806153 / 0.905727` | 2,769 | **KEEP as candidate**; convergence warning |
| M04-60 | 같은 MLP, `max_iter=60` | `0.8587730689` | `[[70327,15024],[10099,82441]]` | `0.823974 / 0.890869` | 2,768 | **HOLD**; +0.000821 vs 30, fit provenance 불완전, convergence warning |
| M04-100 | 같은 MLP, `max_iter=100` | `0.8595319610` | `[[69613,15738],[9250,83290]]` | `0.815608 / 0.900043` | 2,900 | **KEEP / 42D internal reference**; convergence warning |

M04 공통 설정은 `activation="relu"`, `solver="adam"`, `alpha=0.0001`, `batch_size=1024`, constant learning rate `0.001`, `shuffle=True`, `tol=0.0001`, `early_stopping=False`, seed 42다. 30/60/100 모두 max-iteration에 도달했으므로 수렴 완료를 주장하지 않는다.

**왜 M04-100을 이후 AF 비교에 고정했는가:** 동일 42D·split·scaler 조건의 family screen에서 M04-30이 가장 높은 Internal Accuracy를 기록했고, 저장된 Official 비교에서도 M01보다 높은 Accuracy와 두 speaker Recall 개선을 보였다. 이후 다른 설정은 유지하고 `max_iter`만 30→60→100으로 늘린 제한 비교에서 M04-100이 42D Internal reference를 `0.859532`까지 갱신했다. 따라서 acoustic block의 효과만 분리하려고 이 모델을 고정했으며, convergence warning과 미완료 Official 검증 때문에 final model로 보지 않는다.

같은 42D representation의 저장된 **Official Validation** 결과도 별도로 존재한다.

| ID | 공식 검증 정확도 | 화자 0 / 1 재현율 | CM | HCE | 근거 상태 |
|---|---:|---|---|---:|---|
| M00 | `0.8131169214` | `0.774389 / 0.849007` | `[[41697,12148],[8773,49329]]` | 2,158 | 저장된 출력 |
| M01 | `0.8372890743` | `0.792534 / 0.878765` | `[[42674,11171],[7044,51058]]` | 1,313 | 저장된 출력 |
| M04-30 | `0.8529482702` | 출력에 저장됨 | 표시된 혼동행렬 미보존; 행 수로 재구성 가능 | 1,839 | 저장 출력, 별도 결과 파일 없음 |

Official 수치는 full Training fit 결과이고 internal 수치와 평가 집합이 다르므로 서로 Delta를 계산하지 않는다. M04-60/100 및 clean 52D에는 Official 결과가 없다.

**근거:** `notebooks/mission2_model_experiments.ipynb`, cells `checkpoint-i-run-code`, `checkpoint-m-code`, `checkpoint-n-code`, `checkpoint-o-code`, `0c522aec`; `src/dcc_data/mission2_model_experiments.py`.

## 4. 음향 특징 확장

아래 AF01~AF12와 Stage 1~2 score는 당시 cache를 사용한 **historical Training/internal-validation** 결과다. 같은 M04-100, fixed split/seed, internal-train-only scaler를 유지했지만 current v2 E05를 포함한 clean chain과 입력 값 3개가 다르므로 current clean 59D와 성능 Delta를 직접 귀속하지 않는다. 별도의 실행 시간/비용 artifact는 없어 계산비용에 대한 정량 결론은 기록하지 않는다.

AF 후보는 generic feature 확장이 아니라 직전 residual과 representation에서 **빠진 정보 축**을 먼저 적은 뒤 선택했다. 47D 이후에는 overlap/noise/HCE와 짧은 발화, 49D 이후에는 지속되는 overlap/HCE, 재현 가능한 52D 이후에는 speaker 0의 짧은 overlap 및 low-information/crop-boundary 취약점을 다음 질문으로 삼았다. overlap·call 정보는 이 가설을 평가하는 diagnostic일 뿐 model feature로 쓰지 않았다.

### 4.1 음향 특징 묶음 정의

| ID | 특징 묶음 | 추가 차원 | 의미 |
|---|---|---:|---|
| AF01 | 에너지 변화 | 3 | log-RMS mean/std 및 p90-p10 범위 |
| AF02 | 음높이/유성성 | 5 | F0 semitone median/IQR/slope, voiced fraction, periodicity mean |
| AF03 | 정규화된 스펙트럼 변화량 | 3 | frame 간 정규화 spectral 변화량 |
| AF04 | CPP | 2 | 켑스트럼 봉우리 돌출도 중앙값/IQR |
| AF05 | Delta-MFCC 시간 변화 | 2 | MFCC의 시간 변화 요약 |
| AF06 | 포먼트 후보 | — | 후보 단계에서 retired; 구현·실행 결과 없음 |
| AF07 | MFCC 교차 상관 | 2 | MFCC band 간 동조/상관 요약 |
| AF08 | 경쟁하는 음높이 | 2 | 복수 pitch 후보 경쟁 정도 |
| AF09 | 다중 대역 진폭 포락선 일관성 | 3 | 주파수 대역별 amplitude envelope 동조성 |
| AF10 | 음성 활동 경계 | 4 | 발화 시작/끝 경계의 activity 패턴 |
| AF11 | 스펙트럼 엔트로피 변화 | 3 | spectral entropy의 수준과 시간 변화 |
| AF12 | 발화 구간별 MFCC 대비 | 3 | 발화 구간 간 MFCC contrast |

**선정 근거 — source/Notebook에 명시적으로 보존된 의도:**

- **AF01:** E05에 없는 explicit log-RMS contour로 역할별 energy variability를 보려 했다. recording gain shortcut 위험을 함께 사전 기록했다.
- **AF02:** E05에 없는 F0 contour·voicing/periodicity로 prosodic 정보를 보완하려 했다. short/noisy/unvoiced 구간의 pitch failure 위험도 함께 검사했다.
- **AF03:** static rolloff/ZCR가 버리는 frame-to-frame spectral-shape 변화를 보려 했다. transient noise와 짧은 발화 민감성을 위험으로 뒀다.
- **AF04:** AF02가 직접 표현하지 않는 voice quality로 남은 overlap/noisy/HCE를 겨냥했다. overlap을 해결한다고 가정하지 않았다.
- **AF05:** MFCC mean/std가 버리는 frame order와 1차 trajectory dynamics로 반복된 0.5~1.0 s 약점을 겨냥했다. 초기 47D 단계에서는 fixed-duration 결과가 가설을 지지하지 않아 DROP했지만, 이후 59D reference에서 취약 slice 개선 여부만 보는 Stage 2 후보로 한 번 재검증했다.
- **AF06:** formant 후보는 telephone distortion, content dependence, tracking failure, MFCC redundancy 위험 때문에 screening 전에 RETIRED됐다.
- **AF07:** 49D의 marginal MFCC moments에 없는 cross-coefficient covariance가 overlap/mixed structure를 구분하는지 확인하려 했다.
- **AF08:** AF02가 저장하지 않는 competing pitch-lag ambiguity가 voiced overlap/HCE와 연결되는지 확인하려 했다. 이를 overlap detector로 간주하지 않았다.
- **AF09:** 49D에 없는 cross-band envelope coherence로 single-source와 mixed/noisy frame의 시간적 결합 차이를 보려 했다.
- **AF10:** short/cropped/low-information 발화에서 activity density·run·edge asymmetry가 global moments의 빈틈을 보완하는지 확인했다.
- **AF11:** overlap/clipped/low-information fragment의 spectral diffuseness와 entropy variability를 확인했다. broadband noise가 같은 신호를 만들 수 있다는 반증 가능성을 유지했다.
- **AF12:** global MFCC moments가 버리는 early/middle/late 위치 정보를 현재 발화 내부 contrast로만 복원해 boundary-clipped asymmetry를 확인했다.

AF01~03, AF04~05, AF07~09, AF10~12는 각각 변경 불가능한 로컬 Training 캐시로 보존돼 있다. 캐시는 특징 재추출 방지에는 도움이 되지만 모델 점수 자체를 저장한 독립 결과 파일은 아니다.

### 4.2 AF01~AF03 — 42D 기준

| ID | 차원 | 내부 검증 정확도 | 42D 대비 변화 | 화자 0 / 1 재현율 | CM | HCE | 판단 |
|---|---:|---:|---:|---|---|---:|---|
| AF01 | 45 | `0.8571484790` | `-0.0023834820` | 저장된 진단 | `[[74156,11195],[14217,78323]]` | 2,814 | **DROP** |
| AF02 | 47 | `0.8662664219` | `+0.0067344609` | `0.822662 / 0.906484` | `[[70215,15136],[8654,83886]]` | 2,580 | **KEEP**; 47D reference로 승격 |
| AF03 | 45 | `0.8538205979` | `-0.0057113631` | 저장된 진단 | `[[74429,10922],[15082,77458]]` | 2,781 | **DROP** |

AF02만 양의 gain과 두 class의 usable Recall을 보여 다음 reference에 포함됐다. Evidence: acoustic NB cells `afe-screening-gate`, `afe-diagnostics`.

### 4.3 AF04~AF06 — 47D 이후

- **AF04:** AF02 47D에 CPP 2D를 더한 49D 결과 `0.8686274182`, CM `[[70576,14775],[8595,83945]]`, Recall `0.826891 / 0.907121`, HCE 2,707이 저장되어 이후 reference로 사용됐다. 다만 원래 Round 2의 AF04-vs-47D 단독 screening cell은 output이 없어 당시 승격 과정은 **partially verified**다.
- **AF05:** source/Markdown에는 **DROP**으로 남아 있으나 standalone Accuracy·CM output은 보존되지 않았다. score는 `unresolved`로 둔다.
- **AF06:** formant 후보는 실행 전에 **RETIRED**됐다. 완료된 실험으로 세지 않는다.

근거: 음향 Notebook `afe-next-candidate-diagnostics`; 미실행된 원래 셀 `afe-round2-cache`, `afe-round2-prepare`, `afe-round2-screening`, `afe-round2-diagnostics`.

### 4.4 AF07~AF09 — 49D 기준

| ID | 차원 | 내부 검증 정확도 | 49D 대비 변화 | 화자 0 / 1 재현율 | CM | HCE | 판단 |
|---|---:|---:|---:|---|---|---:|---|
| AF07 | 51 | `0.8691277242` | `+0.0005003060` | `0.816686 / 0.917495` | `[[69705,15646],[7635,84905]]` | 2,774 | **HOLD**, 이후 52D+AF07에서 DROP |
| AF08 | 51 | `0.8684025608` | `-0.0002248574` | `0.815433 / 0.917257` | `[[69598,15753],[7657,84883]]` | 2,835 | **DROP** |
| AF09 historical | 52 | `0.8689703245` | `+0.0003429063` | `0.830992 / 0.903998` | `[[70926,14425],[8884,83656]]` | 2,595 | historical non-reproduced; 승격 근거로 사용 금지 |
| AF09 canonical refit (old cache) | 52 | `0.8695324665` | `+0.0009050483` | `0.832632 / 0.903566` | `[[71066,14285],[8924,83616]]` | 2,590 | **KEEP / guarded 52D base**; 당시 canonical, 현재 기준 아님 |

AF09의 당시 canonical refit은 exact feature order와 cache/split/scaler/result contract를 통과했다. 이 `0.869532` 결과는 과거 캐시로 정확히 재현됐지만 현재 clean 52D 기준은 아니다. 최초 `0.868970`은 old cache로도 재현되지 않아 원인을 별도 미해결로 둔다.

AF07을 canonical 52D에 다시 더한 54D 결과는 Accuracy `0.8696392730` (`+0.0001068065`, +19 correct)였지만, speaker 0 Recall이 `-0.012361`, speaker 1 Recall이 `+0.011606`, HCE가 2,590→2,647로 변했다. Notebook 결정은 **DROP / 52D 유지**다.

MFCC 축소 ablation도 canonical을 대체하지 못했다.

| 표현 | 차원 | 정확도 | CM | HCE | 판단 |
|---|---:|---:|---|---:|---|
| MFCC13 mean + MFCC13 std + selected AF blocks | 38 | `0.863793` | `[[72173,13178],[11052,81488]]` | 2,655 | DROP |
| MFCC20 mean + MFCC13 std + selected AF blocks | 45 | `0.868717` | `[[70749,14602],[8752,83788]]` | 2,707 | DROP |
| MFCC13 mean + MFCC20 std + selected AF blocks | 45 | `0.863175` | `[[73335,12016],[12324,80216]]` | 2,621 | DROP |

근거: 음향 Notebook 셀 `afe-next-candidate-diagnostics`, `afe-52d-reproduction`, `afe-52d-54d-comparison`, `afe-mfcc-ablation-core-comparison`.

### 4.5 AF10~AF12 — 52D 기준, 독립 학습 완료

결합 cache shape는 `(889358, 12)` (`record_key`, `call_id`, 10 feature columns)이며 strict-load만 허용됐다. 3개 후보는 같은 52D prefix, split, scaler 계약 아래 **각각 독립 fit 완료**됐다.

| ID | 차원 | 내부 검증 정확도 | 52D 대비 변화 | 화자 0 / 1 재현율 | CM | 정답 수 변화 | HCE | 현재 판단 |
|---|---:|---:|---:|---|---|---:|---:|---|
| AF10 | 56 | `0.8730739610` | `+0.0035414945` | `0.831836 / 0.911109` | `[[70998,14353],[8226,84314]]` | +630 | 2,486 (-104) | **HOLD** — 최고 point estimate이나 자동 승격 안 함 |
| AF11 | 55 | `0.8707466931` | `+0.0012142266` | `0.835889 / 0.902896` | `[[71344,14007],[8986,83554]]` | +216 | 2,670 (+80) | **HOLD** |
| AF12 | 55 | `0.8701114730` | `+0.0005790065` | `0.837916 / 0.899805` | `[[71517,13834],[9272,83268]]` | +103 | 2,433 (-157) | **HOLD** |

AF10은 overlap `+0.005256`, no-overlap `+0.002185`였지만 speaker 0 Recall은 `-0.000797`이고 speaker 1 Recall은 `+0.007543`이다. AF11/AF12도 class trade-off가 있다. Notebook이 `automatic_promotion=False`로 중단했으므로 **single-block screening 시점의 canonical은 52D 그대로**였다.

근거: 음향 Notebook 셀 `afe-residual-round-cache`, `afe-residual-round-prepare`, `afe-residual-round-screen`, `afe-residual-round-summary`.

### 4.6 AF10~AF12 제한된 조합 비교 — 과거 캐시 결과

single-block 결과를 바꾸지 않고 기존 52D와 immutable combined cache를 사용해 사전 고정된 4개 조합만 비교했다. 52D prefix, fixed split/seed, internal-train-only scaler, M04-100을 유지했으며 자동 승격은 비활성화됐다.

| ID | 차원 | 내부 검증 정확도 | 52D 대비 변화 | 화자 0 / 1 재현율 | CM | 정답 수 변화 | HCE | 현재 상태 |
|---|---:|---:|---:|---|---|---:|---:|---|
| AF10_AF11_COMBINATION (old cache) | 59 | `0.874558` | `+0.005026` | `0.836792 / 0.909391` | `[[71421,13930],[8385,84155]]` | +894 | 2,377 (-213) | 당시 handcrafted reference; 현재 eGeMAPS 비교 기준 아님 |
| AF10_AF12_COMBINATION | 59 | `0.872574` | `+0.003041` | `0.835807 / 0.906484` | `[[71337,14014],[8654,83886]]` | +541 | 2,339 (-251) | completed; decision pending |
| AF11_AF12_COMBINATION | 58 | `0.872107` | `+0.002575` | `0.826329 / 0.914329` | `[[70528,14823],[7928,84612]]` | +458 | 2,346 (-244) | completed; decision pending |
| AF10_AF11_AF12_COMBINATION | 62 | `0.876042` | `+0.006510` | `0.829317 / 0.919138` | `[[70783,14568],[7483,85057]]` | +1,158 | 2,386 (-204) | completed; decision pending |

**Residual 질문에 대한 관찰:** AF10+12는 핵심 `speaker 0 × 0.5~1 s × overlap` Accuracy를 `0.675652 → 0.696841`로 가장 크게 높였고, AF10+11도 두 speaker Recall과 두 focus slice를 함께 개선했다. 반면 전체 Accuracy가 가장 높은 3-block 조합은 speaker 0 Recall이 52D보다 `-0.003316`, 핵심 0.5~1 s overlap slice도 `-0.006143`이었다. 따라서 조합은 상보성 가능성을 보여 줬지만, 최고 Accuracy만으로 canonical을 바꾸지 않고 사용자 판단을 기다린다.

근거: 음향 Notebook 셀 `afe-residual-combination-prepare`, `afe-residual-combination-screen`, `afe-residual-combination-summary`, `afe-residual-combination-diagnostics`.

### 4.7 과거 단계별 진단 — 과거 캐시 59D, AF05 재검증, 수작업 특징 탐색 종료

#### Stage 1 — 59D 일반화 점검

당시 AF10+AF11 59D를 후속 비교용 handcrafted reference로 고정했다. 아래 Train Accuracy와 gap은 **old-cache 결과**이며 clean 59D의 Train Accuracy·gap으로 재사용하지 않는다. M04-100은 `max_iter=100`에서도 미수렴했으며 final model이 아니다.

| 학습 정확도 | 내부 검증 정확도 | 학습−검증 격차 | CM | 화자 0 재현율 | 화자 1 재현율 |
|---:|---:|---:|---|---:|---:|
| `0.8766112834467376` | `0.8745580158636468` | `0.0020532675830907676` | `[[71421,13930],[8385,84155]]` | `0.8367916017387025` | `0.9093905338232116` |

관찰된 gap은 크지 않았지만 이것만으로 model, optimization, representation 중 어느 bottleneck이 지배적인지는 확정하지 않는다. 특히 M04-100의 미수렴 상태 때문에 작은 gap을 optimization 완료의 근거로 사용하지 않는다.

#### Stage 2 — 59D + AF05 Delta-MFCC 재검증

AF05가 global score보다 `speaker 0 × 0.5~1.0 s × overlap` 취약 구간에 선택적으로 유효한지 확인하기 위해, 59D에 AF05 2D만 추가한 61D를 같은 Internal 조건에서 비교했다.

| 표현 | 차원 | Val Accuracy | 변화량 | 화자 0 재현율 변화 | 화자 1 재현율 변화 | 오답→정답 | 정답→오답 | 순변화 | HCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 59D + AF05 | 61 | `0.874738` | `+0.000180` | `+0.014681` | `-0.013194` | 4,419 | 4,387 | +32 | 2,377 → 2,305 (-72) |

- 핵심 slice Accuracy: `speaker 0 × 0.5~1.0 s × overlap` `0.690948 → 0.758149`.
- **Decision:** 특정 취약 구간과 HCE에는 개선이 있었지만 전체 net gain이 +32로 매우 작고 class Recall trade-off가 커서 reference에는 채택하지 않는다.
- **상태:** AF05는 **HOLD candidate**다. 이 결정과 Delta는 당시 old-cache 59D 기준이며 clean chain에서 재평가한 결과가 아니다.

#### Stage 3 — 수작업 특징 확장 종료

추가 handcrafted feature 조합 탐색은 종료한다. 다음 단계는 동일한 공정 비교 계약 아래 openSMILE eGeMAPSv02 Functionals 88D를 **clean 59D reference**와 비교하는 것이다.

근거: 사용자 제공 Stage 1–3 결과 기록 (2026-09-21). 이 문서 갱신에서는 실험이나 Notebook을 새로 실행하지 않았다.

### 4.8 현재 clean Stage 4 비교 기준 — 사용자 확인 결과

현재 raw Training manifest에서 만든 versioned chain `stage4_clean_v1_090422f783f0d911`의 provenance fingerprint는 `49d65a35d1e98c280f98e45cbed864ef8bac36268ccffbe81aa0985923c90de3`이다. 사용자 보고에 따르면 fixed call_id split, seed 42, internal-train-only scaler, M04-100 조건에서 B4를 두 번 실행해 동일 결과를 얻었다. 이는 **Training 내부 validation**이며 Official Validation 결과나 final model 승격이 아니다.

| 현재 clean 표현 | 차원 | 내부 검증 정확도 | CM | 화자 0 / 1 재현율 | 정답 / 오류 |
|---|---:|---:|---|---|---:|
| AF09 52D | 52 | `0.869594` | `[[71215,14136],[9062,83478]]` | `0.834378 / 0.902075` | `154693 / 23198` |
| AF10_AF11_COMBINATION | 59 | **`0.873760`** | `[[72265,13086],[9371,83169]]` | `0.846680 / 0.898736` | `155434 / 22457` |

앞으로 Stage 4의 질문은 “같은 Training 행·fixed call_id split·seed 42·M04-100·train-only scaler·평가 집합에서 표준 acoustic representation인 eGeMAPS 88D가 **clean 59D**보다 유용한가?”이다. Accuracy가 주 metric이고 CM, speaker별 Recall, Train−Val gap, paired error 변화는 보조 진단이다. historical `0.874558`은 이 비교의 기준이 아니다.

### 4.9 Stage 4 — eGeMAPSv02 88D와 clean 59D의 비교 완료

저장 근거 파일 `data/processed/mission2_standard_representation_comparison/clean59d_egemaps88d_20261004T190153_4dbd7177`와 Notebook 출력을 대조했다. 지표와 혼동행렬, 짝지은 정답 전환/오답 전환 및 주요 발화 길이 구간의 값이 일치한다. 결과는 Training 내부 고정 GroupSplit(711,467/177,891행, 통화 23,360/5,840개, 통화 중복 0), 시드 42, 동일 M04-100, 표현별 내부 학습 전용 StandardScaler로 계산됐다. 공식 검증은 사용하지 않았다. eGeMAPS 캐시 출처 지문은 `49d65a35d1e98c280f98e45cbed864ef8bac36268ccffbe81aa0985923c90de3`이며 짧은 음성의 패딩 행 세 개가 메타데이터에 기록돼 있다.

| 지표 | Clean AF10+AF11 59D | eGeMAPSv02 88D | 88D − 59D |
|---|---:|---:|---:|
| 학습 정확도 | `0.8757876332` | `0.8666473638` | `-0.0091402693` |
| 내부 검증 정확도 | `0.8737597742` | `0.8639897465` | `-0.0097700277` |
| 일반화 격차 | `0.0020278589` | `0.0026576173` | `+0.0006297584` |
| 화자 0 재현율 | `0.8466801795` | `0.8236458858` | `-0.0230342937` |
| 화자 1 재현율 | `0.8987356819` | `0.9011994813` | `+0.0024637994` |
| 혼동행렬 | `[[72265,13086],[9371,83169]]` | `[[70299,15052],[9143,83397]]` | — |
| 오류 수 | `22,457` | `24,195` | `+1,738` |
| 높은 신뢰도의 오류(HCE, 기준 `0.90`) | `2,378` | `2,733` | `+355` |
| HCE FP / FN | `1,197 / 1,181` | `1,414 / 1,319` | `+217 / +138` |

행 단위 짝지은 비교: 두 모델 모두 정답 `146,653`, 두 모델 모두 오답 `15,414`, 59D 오답→88D 정답 `7,043`, 59D 정답→88D 오답 `8,781`, 순변화 `-1,738`. 전체 정확도 하락은 짝지은 오류의 순변화와 일치한다.

발화 길이별 구간에서는 결과가 엇갈렸다. speaker 1의 `0.5–1 s` 정확도는 `0.831356`에서 `0.868084`로 상승했고 (`n=23,606`), speaker 0의 `0.5–1 s` 정확도는 `0.816924`에서 `0.758332`로 하락했다 (`n=19,593`). speaker 0의 중첩 발화 구간에서는 `0.5–1 s` 정확도가 `0.745236→0.678661`로 (`n=7,976`), `1–2 s` 정확도가 `0.777022→0.755937`로 바뀌었다 (`n=13,517`). 이러한 구간별 차이는 보완적 활용을 후속 검증할 가능성을 남기지만, 그 활용이 유효하다고 입증한 것은 아니다.

**판단:** 이 고정 MLP 비교에서는 eGeMAPSv02 88D를 단독 대체 표현으로 채택하지 않는다 (DROP). 내부 검증 정확도가 `0.0097700277` 하락했고 오류는 1,738건, 높은 신뢰도의 오류는 355건 증가했다. 관측된 학습−검증 성능 차이는 작으므로 (`0.002028`, `0.002658`) 이 실행만으로 큰 일반화 격차가 있다고 볼 근거는 부족하다. 두 학습 모두 `max_iter=100`에 도달해 수렴 경고를 남겼다. MLP 최적화 문제는 미해결이며 모델 병목도 확정하지 않는다. 같은 실행에서 비교한 clean 59D/88D 결과는 과거 캐시의 59D 점수 `0.8745580159`와 구분하며, 위 역사 기록은 그대로 보존한다.

근거: 저장 실행의 `comparison/summary.json`, `comparison/comparison.csv`, `comparison/paired_slices.csv`, `comparison/validation_predictions.parquet`와 완료된 비교 Notebook의 출력.

### 4.10 Stage 6 — clean 59D 모델 계열 비교

저장 run `data/processed/mission2_model_family_screen/20261005T113752Z_8ab60cd0`의 `comparison.csv`를 확인했다. Stage 4 M04-100과 구분되는 MLP300 reference 및 후보 결과이며, historical old-cache `0.874558`을 기준으로 사용하지 않는다. clean 59D, Training 내부 fixed call_id split, seed 42, internal-train-only StandardScaler, Accuracy, threshold `0.5`를 유지했고 Official Validation은 사용하지 않았다.

| 후보 | 학습 정확도 | 내부 검증 정확도 | 학습−검증 격차 | 오류 수 | HCE | Delta vs MLP300 |
|---|---:|---:|---:|---:|---:|---:|
| MLP300 reference | `0.8767251327` | `0.8744624517` | `0.0022626810` | `22,332` | `2,505` | `0` |
| LightGBM | `0.8613653198` | `0.8597062246` | `0.0016590952` | `24,957` | `2,015` | `-0.0147562271` |
| XGBoost hist | `0.8637083660` | `0.8590991113` | `0.0046092547` | `25,065` | `1,861` | `-0.0153633405` |
| CatBoost | `0.8529067406` | `0.8535844984` | `-0.0006777578` | `26,046` | `1,702` | `-0.0208779534` |
| Numerical-embedding MLP | `0.9005449304` | `0.8749009225` | `0.0256440079` | `22,254` | `3,510` | `+0.0004384707` |
| **Stronger MLP** | `0.8817766671` | **`0.8760364493`** | `0.0057402178` | **`22,052`** | `2,616` | `+0.0015739976` |

**복잡 정형 모델 비교의 직접 기준:** Stronger MLP (`MLPClassifier`, hidden `(128,)`, `max_iter=300`, seed 42). Recall speaker 0/1은 `0.8327729025 / 0.9159390534`, CM은 `[[71078,14273],[7779,84761]]`, FP/FN은 `14,273 / 7,779`다. 다음 complex run의 contract에 저장된 Stage 6 원본 파일 SHA-256 10개가 로컬 원본과 일치했고, Stronger MLP prediction을 후속 후보와 같은 row/key/label 순서로 대조했다. refit은 수행하지 않았다.

관측된 Stage 6 최고 Accuracy는 Stronger MLP였다. 세 GBDT 후보는 MLP300보다 낮았고 numerical-embedding MLP의 개선은 78개 정답에 그쳤으며 gap과 HCE가 증가했다. 이는 이 first-screen recipe에서의 결과이며 모든 classifier family의 우열을 확정하지 않는다.

### 4.11 Stage 6 이후 — 복잡한 심층 정형 모델 비교 검증 결과

**실행:** `data/processed/mission2_complex_tabular_screen/20261006T031510Z_a9a01a2d`. 로컬에 해당 run이 없어 [동일 Google Drive run](https://drive.google.com/drive/folders/1WDSriGQDQ3k0xoUQsdEIwWSicETEbic9)의 저장 artifact를 읽어 검증했다. `comparison.csv`, 후보별 `metrics.json`·`config.json`·`history.csv`·`predictions.parquet`, `contract.json`, `environment.json`, `paired_comparison.csv`, `paired_slices.csv`, `analysis_metadata.json`을 대조했다. 아래 internal-validation 지표·CM·HCE·paired counts는 저장 prediction에서 재계산했으며 제공 수치와 일치한다. Training Accuracy·gap·시간은 저장 metrics와 comparison의 일치를 확인했다. checkpoint 재학습·재추론은 수행하지 않았고, reload 검증은 Colab에서 저장한 `EXACT_PASS` 기록을 확인했다.

**비교 계약:** clean 59D 지문 `49d65a35d1e98c280f98e45cbed864ef8bac36268ccffbe81aa0985923c90de3`, float32 및 동일 특징 순서, 내부 학습 `711,467`행 / 통화 `23,360`개, 내부 검증 `177,891`행 / 통화 `5,840`개, 통화 중복 `0`. 고정 call_id 분할, 시드 `42`, 내부 학습 전용 스케일러 학습, 검증에는 변환만 적용, 임계값 `0.5`, 정확도, 공식 검증 미사용을 확인했다. Stage 6와 전처리 계약 및 평가 행/키/정답이 일치한다. 환경 간 변환된 행렬의 바이트 해시는 참고 정보이며 필수 통과 기준이 아니다. 원본 데이터의 전송 시점 바이트 검증은 기록돼 있지만 과거 구축 시점의 내용 증명은 `not_collected`로 유지된다.

후보별 first-screen config는 결과 확인 전에 고정했다. 공통 batch `256`, 최대 `100` epochs, patience `10`, internal Validation Accuracy의 earliest best checkpoint 선택, tie 시 이전 best 유지, scheduler 없음이다. 저장 `config.json`은 run 시작의 `contract.json` config와 동일하다.

| 후보 | 고정 구조 / 패키지 | AdamW 학습률 / 가중치 감쇠 | 설정 ID |
|---|---|---|---|
| TabM | 수치 임베딩 없음, 블록 3개, 너비 512, 드롭아웃 0.1, k=32; `tabm==0.0.3` | `0.002 / 0.0003` | `aa9852781208772c2c43139e6454d96a1f67cda9abae1303900c792ae2e21142` |
| FT-Transformer | 블록 3개, 너비 192, 헤드 8개; `rtdl-revisiting-models==0.0.2` | `0.0001 / 0.00001` | `c50f39d91b55b2a147f20a96caf4000bb0b6f11c4fbfe5ea8b95e503eb75a28b` |
| Tabular ResNet | 블록 2개, 너비 192, 은닉 너비 배수 2, 드롭아웃 0.15/0; 동일 rtdl 패키지 | `0.0003 / 0.00001` | `2af3a7c55dfdacf6ebc8557e1b481e230f7d51f69dde5d89562d3fe6b1c647a6` |

Stage 6 sklearn MLP는 loss-based stopping을 사용했고 이번 후보는 validation-Accuracy checkpoint를 선택한다. 따라서 **실용적인 model/pipeline 및 training recipe 비교**이며 개선 전부를 순수 architecture 효과로 귀속하지 않는다. 저장 실행 환경은 Python `3.13.15`, NumPy `2.1.3`, sklearn `1.6.1`, Torch `2.11.0+cu130`, CUDA `13.0`, Tesla T4다.

#### 세 후보 결과

| 지표 | TabM | FT-Transformer | Tabular ResNet |
|---|---:|---:|---:|
| 최적 epoch / 실행 epoch 수 | `52 / 62` | `24 / 34` | `24 / 34` |
| 학습 정확도 | `0.8878205173` | `0.8931995440` | `0.9001794883` |
| 내부 검증 정확도 | **`0.8839795155`** | `0.8821975255` | **`0.8837546588`** |
| 학습−검증 격차 | `0.0038410018` | `0.0110020186` | `0.0164248295` |
| 화자 0 재현율 | `0.8345537838` | `0.8307225457` | **`0.8447001207`** |
| 화자 1 재현율 | `0.9295655933` | `0.9296736546` | `0.9197752323` |
| 균형 정확도 | `0.8820596885` | `0.8801981002` | `0.8822376765` |
| 혼동행렬 | `[[71230,14121],[6518,86022]]` | `[[70903,14448],[6508,86032]]` | `[[72096,13255],[7424,85116]]` |
| FP / FN | `14,121 / 6,518` | `14,448 / 6,508` | `13,255 / 7,424` |
| 오류 수 | **`20,639`** | `20,956` | **`20,679`** |
| 높은 신뢰도의 오류(HCE, 신뢰도 ≥ `0.90`) | **`1,629`** | `3,024` | `2,693` |
| 평균 신뢰도 | `0.8696148662` | `0.8953233685` | `0.8927482435` |
| 학습 시간(초) | `2810.449` | `5585.018` | `748.222` |
| 예측 시간(초) | `14.861` | `58.534` | `6.692` |
| 체크포인트 재로드 (저장된 Colab 기록) | `EXACT_PASS` | `EXACT_PASS` | `EXACT_PASS` |
| Stage 6 Stronger MLP 대비 변화 | `+0.007943066260` | `+0.006161076165` | `+0.007718209465` |
| 퍼센트포인트 변화 | `+0.7943 pp` | `+0.6161 pp` | `+0.7718 pp` |
| Stronger MLP 대비 오류 감소 | `1,413` | `1,096` | `1,373` |

표시값은 반올림했고 persisted exact Accuracy는 TabM `0.8839795155460366`, FT-Transformer `0.8821975254509784`, ResNet `0.8837546587517076`이다. 저장 `predict_seconds`는 train+validation probability 산출 합계이며 validation-only latency가 아니다. 같은 run의 계측값에서 ResNet의 학습·예측 비용이 가장 낮고 FT-Transformer가 가장 높았다.

FT-Transformer도 Stronger MLP보다 개선됐지만 다른 두 후보보다 overall Accuracy가 낮고 비용이 가장 높았다. 따라서 현재 dual frozen reference에는 포함하지 않으며 실험 결과는 보존한다.

#### Stronger MLP 대비 짝지은 / 화자별 비교

| Paired category (`n=177,891`) | TabM | Tabular ResNet |
|---|---:|---:|
| 두 모델 모두 정답 | `153,314` | `152,153` |
| 두 모델 모두 오답 | `18,114` | `16,993` |
| Stronger MLP 정답 / 후보 오답 | `2,525` | `3,686` |
| Stronger MLP 오답 / 후보 정답 | `3,938` | `5,059` |
| 정답 수 순증가 | **`+1,413`** | **`+1,373`** |

| 화자 | n | TabM 정답→오답 / 오답→정답 / 순변화 | ResNet 정답→오답 / 오답→정답 / 순변화 |
|---|---:|---:|---:|
| speaker 0 | `85,351` | `1,479 / 1,631 / +152` | `1,633 / 2,651 / +1,018` |
| speaker 1 | `92,540` | `1,046 / 2,307 / +1,261` | `2,053 / 2,408 / +355` |

TabM 개선 1,413개 중 1,261개는 speaker 1의 net correction이며 speaker 0 개선은 152개다. speaker 0의 큰 개선으로 과장하지 않는다. ResNet은 speaker 0에서 1,018개, speaker 1에서 355개를 개선했다. TabM보다 speaker 0 Recall이 높지만 speaker 1 Recall은 낮고 Train−Validation gap은 크다.

Confidence slice도 저장 결과와 prediction이 일치한다. 후보 confidence `≥0.9` / `<0.9` 집단에서 Stronger MLP 대비 TabM net correction은 `+29 / +1,384` (`n=111,425 / 66,466`), ResNet은 `+397 / +976` (`n=122,429 / 55,462`)다. 후보별 confidence로 집단을 정하므로 두 모델의 해당 slice는 동일 row 집단이 아니며 slice 간 직접 우열을 계산하지 않는다.

#### TabM 대 Tabular ResNet — 직접 짝지은 비교

| 분류 | 건수 |
|---|---:|
| 두 모델 모두 정답 | `153,809` |
| 두 모델 모두 오답 | `17,236` |
| TabM 정답 / ResNet 오답 | `3,443` |
| TabM 오답 / ResNet 정답 | `3,403` |
| TabM 정답 수 순우위 | **`+40`** |

두 저장 prediction을 동일 row/key/label 순서로 직접 대조했다. `153,809 + 17,236 + 3,443 + 3,403 = 177,891`, `3,443 - 3,403 = 40`이다. 정오가 달라지는 `6,846` rows는 전체의 `3.8484%`이며 Accuracy 차이는 `40 / 177,891 = 0.0002248568`, 약 `0.0225 pp`다.

관측 최고점은 TabM이지만 두 모델의 overall Accuracy는 매우 근접하다. 정오가 달라지는 행이 있어 두 모델의 decision behavior가 완전히 같지는 않으며, TabM은 상대적으로 speaker 1, ResNet은 speaker 0에 강하다. 두 모델 모두 틀린 `17,236`개 공통 residual은 representation bottleneck 가능성을 검토할 근거 중 하나지만 이를 확정하지 않는다.

**미분석/현재 artifact 미확인:** `analysis_metadata.json`의 `tracking_used=false`와 duration/overlap slice 부재를 확인했다. 이번 run의 duration, overlap/no-overlap, speaker 0 × 0.5–1s × overlap, speaker 0 × 1–2s × overlap 결과는 없다. §4.9 및 historical residual 분석의 duration/overlap 결과를 이번 TabM/ResNet에 연결하지 않는다. active/silence/low-information mask도 metadata에 missing으로 기록돼 있다.

### 4.12 Stage 10 결정 — 59D 이중 동결 비교 기준

**Final model: NOT SELECTED.** 59D classifier 탐색에서 확보한 서로 다른 강점을 가진 두 reference를 다음 representation 계열 실험의 비교 기준으로 보존한다.

| 동결 비교 기준 | 역할과 관측 근거 | 내부 검증 정확도 |
|---|---|---:|
| **A — TabM** | Accuracy-oriented 59D reference. 현재 관측 최고 Accuracy, 후보 중 낮은 gap·HCE, speaker 1 쪽 강점 | **`0.8839795155`** |
| **B — Tabular ResNet** | speaker-0-oriented / complementary 59D reference. TabM과 근접한 Accuracy, 더 높은 speaker 0 Recall `0.8447001207`, 다른 paired error behavior, 상대적으로 낮은 계측 비용 | **`0.8837546588`** |

Stage 6 Stronger MLP 기준보다 complex deep-tabular recipe에서 약 `+0.6~0.8 pp`의 incremental value가 확인됐다. 따라서 이전 시점의 “classifier가 충분하다”는 가정은 수정한다. 다만 architecture·capacity와 training recipe가 함께 달라졌으므로 순수 architecture 효과는 분리하지 못했다.

세 후보는 약 `0.882~0.884` 구간에 모였고 가장 강한 두 모델도 overall Accuracy가 거의 동률이며 상당한 공통 오류가 남았다. 59D 위의 잔여 오류 중 상당 부분이 classifier 교체만으로 쉽게 해결되지 않을 가능성이 있어 시간×주파수 representation을 시험할 근거가 된다. 단일 seed/fixed split의 결과이므로 `0.884`를 59D의 절대 ceiling으로 보거나 representation bottleneck이 확정됐다고 판단하지 않는다.

**현재 결정:** 59D classifier search를 추가 모델 탐색 없이 일단 중단하고, TabM + Tabular ResNet 이중 비교 기준를 보존한 채 **Stage 11 시간×주파수 표현 계열 조사**으로 이동한다. 어느 reference도 final model로 선정하지 않는다. tuning을 영구 금지하는 결정은 아니며 Stage 11/12 결과와 최종 비교에서 필요성이 생기면 bounded revisit 후보로 남긴다.

**후속 상태 — 2026-10-09:** 위 Stage 10 결정은 당시 기록으로 보존한다. 이후 Stage 11은 팀원 김광희의 별도 탐색과 개인 실험 중복을 피하기 위해 `SKIP — team duplication avoidance`로 정리했고, Stage 12-A/B는 성능 평가 없이 계산 가능성 문제로 종료했다. 현재 reference는 TabM + Tabular ResNet, final model은 `NOT SELECTED`이며 다음은 후보 미확정의 **Stage 13 — clean 59D에서 아직 시험하지 않은 모델 계열 재검토**다.

## 5. 기준 표현의 변경 이력

아래 Stage 4 reference 표기는 당시 representation 비교의 역할을 보존한 것이다. 현재 classifier 비교 기준은 §4.12의 TabM + Tabular ResNet 이중 동결 비교 기준다.

| 단계 | 정확한 구성 | 차원 | 승격 근거 / 당시 점수 | 다음 비교 기준 역할 |
|---|---|---:|---|---|
| E00 | MFCC20 means + MFCC20 stds | 40 | LR Official `0.809106` baseline | E01~E05 reference |
| E05 | E00 + spectral rolloff + ZCR | 42 | LR Official `0.813117`; 이후 M04-100 internal `0.859532` | model screen 및 AF01~03 reference |
| AF02 | E05 + F0 semitone median/IQR/slope + voiced fraction + periodicity mean | 47 | M04-100 internal `0.866266` | AF04/AF05 reference |
| AF04 | 47D + CPP median/IQR | 49 | 저장된 49D internal `0.868627`; 원래 promotion output은 불완전 | AF07~09 reference |
| AF09 (old cache) | 49D + 다중 대역 진폭 포락선 일관성 3D | 52 | 당시 검증 기준을 통과한 재학습 내부 결과 `0.8695324665`; 과거 캐시로 정확히 재현 | AF10~12 과거 비교 및 잔여 오류 분석 기준 |
| AF10+AF11 (old cache) | 52D + activity/boundary 4D + entropy dynamics 3D | 59 | 당시 Internal Val `0.8745580158636468`; Stage 1 Train−Val gap `0.0020532675830907676`; 과거 캐시로 정확히 재현 | historical handcrafted reference; 현재 비교 기준 아님 |
| AF09 (clean chain) | clean MFCC40 + current E05 + AF02 + AF04 + AF09 | 52 | Internal Val `0.869594`; CM `[[71215,14136],[9062,83478]]` | current reproducible 52D prefix |
| AF10+AF11 (clean chain) | clean 52D + AF10 4D + AF11 3D | 59 | Internal Val `0.873760`; CM `[[72265,13086],[9371,83169]]` | **current Stage 4 eGeMAPS comparison reference**; final 아님 |

**승격 판단의 흐름:** 40D는 exact baseline 수치와 오류 ledger를 만드는 출발점이었다. 42D는 rolloff+zcr의 Training-only 보완성이 Official point estimate와 speaker 0 Recall 개선으로 이어져 model-family 비교의 feature를 고정했다. 47D는 42D 대비 Accuracy뿐 아니라 두 speaker Recall 개선과 HCE 감소가 함께 관찰되어 다음 reference가 됐다. 49D는 저장 결과에서 두 speaker Recall이 소폭 개선되어 후속 비교에 쓰였지만, HCE와 최초 promotion output 부재를 provenance caveat로 유지한다. 52D는 historical score를 승격 근거로 쓰지 않고 guarded refit의 feature order·split·scaler·CM 재현을 통과한 결과만 canonical로 삼았으며, 54D AF07 추가가 Recall/HCE trade-off로 기각된 뒤에도 기준을 유지했다.

old-cache 52D는 당시 guarded reproducibility base로 역사에 유지한다. 현재 Stage 4 비교에는 별도 clean chain의 AF10+AF11 59D를 사용한다. 과거 point estimate와 clean score는 서로 다른 저장 입력에서 나온 값이므로 historical 수치를 clean 기준으로 대체하거나 두 결과의 차이를 새로운 feature 효과로 해석하지 않는다.

## 6. 오류 분석

### 기준 모델 오류 분석

전체 Official Validation 111,947 rows를 1:1 join해 분석했다. row-level ledger는 `SAVE_ANALYSIS_ARTIFACTS=False`여서 별도 파일로 보존되지 않고 Notebook output만 남았다.

| 구간 / 진단 | 관측 결과 |
|---|---|
| total errors | 21,370; mean confidence 전체 `0.802446`, errors `0.687204` |
| high-confidence errors | confidence ≥0.90: 2,086 |
| duration: very short | n=5,786, Accuracy `0.790010` |
| duration: short | n=22,545, Accuracy `0.755068` |
| duration: medium | n=55,659, Accuracy `0.797787` |
| duration: long | n=22,384, Accuracy `0.871024` |
| duration: very long | n=5,573, Accuracy `0.911897` |
| 중첩 | n=48,863, Accuracy `0.765426` |
| no overlap | n=63,084, Accuracy `0.842940` |
| acoustic-quality sample | 600 sampled 행만 검사: normal 551 / `0.840290`, suspicious 49 / `0.816327` |
| manual listening | 12개 baseline example 청취 완료; unclear/silence/noise/nonverbal, 짧고 audio-only로 어려운 사례 반복 |

**관찰과 가설 구분:** 짧은 발화와 overlap에서 Accuracy가 낮다는 것은 관찰이다. 청취 표본이 보여 준 low-information/cropping/noise가 전체 오류의 원인이라는 결론은 표본 대표성이 없어 확정하지 않는다.

근거: `26d48b6:notebooks/mission2_baseline_error_analysis.ipynb`. 원본 데이터 탐색은 Training/Validation 각각 발화 889,358/111,947건, 심각한 문제 0, 중복 구간 경고 3개 그룹/6행, 1,200행 음향 표본의 추출 실패 0을 기록했다. 다만 Step 3에 `KeyboardInterrupt`가 남아 있어 커밋 상태 그대로 위에서 아래로 재현한 실행본은 아니다 (`6e4c7d6:notebooks/mission2_raw_data_eda.ipynb`).

### 52D 기준 모델 잔여 오류 분석 — 과거 검증된 기준

아래는 historical `0.868970` object가 아니라 canonical 52D `0.8695324665` result에서 다시 만든 ledger다.

| 구간 / 진단 | 관측 결과 |
|---|---|
| overall | correct 154,682 / errors 23,209; FP 14,285 / FN 8,924 |
| 재현율 | speaker 0 `0.832632`; speaker 1 `0.903566` |
| 중첩 | n=78,581, Accuracy `0.827846` |
| no overlap | n=99,310, Accuracy `0.902517` |
| duration <0.5 s | n=14,076, Accuracy `0.863527` |
| duration 0.5~1 s | n=43,199, Accuracy `0.825598` |
| duration 1~2 s | n=58,320, Accuracy `0.847154` |
| duration 2~5 s | n=51,015, Accuracy `0.915221` |
| duration ≥5 s | n=11,281, Accuracy `0.954348` |
| primary weak slice | speaker 0 × 0.5~1 s × overlap: 2,587 / 7,976 errors (`32.43%`) |
| second weak slice | speaker 0 × 1~2 s × overlap: 3,145 / 13,517 errors (`23.27%`) |
| high-confidence errors | 2,590 = FP 1,223 + FN 1,367; 67.9% overlap; 1,723/2,590이 0.5~2 s |
| low confidence | confidence <0.60 Accuracy `0.549152` |
| call concentration | 5,439 / 5,840 통화에 error; top-1 share `0.001379`, top-20 share `0.01857` — 소수 call 집중 아님 |
| manual listening | deterministic 10-sample 청취 완료; short/cropped/low-information/difficult audio 반복, overlap은 항상 청각적으로 뚜렷하지 않음, 1건은 label/segment alignment 점검 후보 |

청취 결과는 원인 후보를 만드는 질적 증거이며 전체 residual의 원인을 확정하지 않는다. 가장 재현 가능한 정량 취약점은 **speaker 0의 짧은 overlap 발화**다.

**잔여 오류 → 음향 후보 연결:** 이 약점을 해결하기 위해 금지된 overlap/call-relative 정보를 입력하지 않고, 현재 발화 waveform 안에서만 검증 가능한 서로 다른 설명축을 골랐다. AF10은 짧거나 잘린 crop의 activity occupancy/boundary, AF11은 overlap·low-information 구간의 spectral diffuseness, AF12는 global MFCC moments가 놓치는 발화 내부 temporal asymmetry를 각각 검증한다. 세 block은 같은 52D/M04-100 계약에서 독립 비교되므로, 결과 차이는 해당 block 추가 효과로 해석할 수 있지만 인과 원인으로 단정하지 않는다.

## 7. 실패 / 제외 / 보류 실험

| 상태 | Experiments | 이유 |
|---|---|---|
| DROP | E01 | Accuracy 변화가 사실상 0 |
| HOLD | E03, E04 | 소폭 gain과 class/HCE trade-off; 단독 승격 없음 |
| DROP | M02, M03 | M01/M04보다 낮거나 Recall/HCE 균형 악화 |
| HOLD | M04-60 | gain 작음, fit provenance 불완전, convergence warning |
| DROP | AF01, AF03, AF08 | 해당 reference 대비 Accuracy 하락 |
| PARTIALLY VERIFIED | AF04 | 49D 결과는 있으나 원래 standalone screening/promotion output 없음 |
| DROP / unresolved score | AF05 | 결정은 남았으나 standalone score output 없음 |
| HOLD | AF05 on 59D (61D) | 핵심 취약 slice와 HCE는 개선됐지만 전체 +32 net correction과 class Recall trade-off로 reference 미채택 |
| RETIRED / not run | AF06 | 구현·평가 전 후보 폐기 |
| HOLD→DROP | AF07 | 49D에서 소폭 gain, 52D 조합은 class trade-off/HCE 악화로 미승격 |
| HISTORICAL QUARANTINE | AF09 `0.868970` | 현재 code/cache로 재현되지 않음 |
| DROP | MFCC reduction 3종 | 모두 canonical 52D보다 낮음 |
| HOLD | AF10, AF11, AF12 | independent fit 완료, 자동 승격 금지; class/HCE trade-off 검토 필요 |
| COMPLETED / decision pending | AF10~12 combinations 4종 | 모두 같은 52D 계약에서 실행됐으나 자동 승격은 비활성화됨 |

## 8. 검증 전략

1. **초기 단계:** E00~E05와 M00/M01/M04-30은 full Training fit 후 Official Validation을 사용했다. 이는 저장된 역사 결과이며 규정상 허용된 model/feature 선택 용도다.
2. **현재 단계:** 반복적인 acoustic feature 비교는 Training 안의 fixed call-group split으로 전환했다. Official을 매 실험마다 반복 확인하는 대신 동일 평가 집합에서 paired diagnostics를 만들고, 같은 call 파생 utterance가 split을 넘지 않도록 했다.
3. **preprocessing:** baseline/Official에서는 scaler를 full Training에만, internal에서는 711,467 internal-training rows에만 fit한다.
4. **직접 비교 가능:** 같은 42D 내부 모델 비교끼리, 같은 M04-100 계약의 AF 비교끼리, 그리고 동일 clean 59D 전처리·평가 집합의 Stage 6/복잡 정형 모델 비교끼리 변화량을 해석한다. 마지막 비교는 학습 절차 차이가 있어 순수 모델 구조 효과로 귀속하지 않는다.
5. **직접 비교 금지:** Official과 internal, LR과 MLP가 함께 바뀐 결과, historical non-reproduced object와 canonical refit 사이의 성능 원인 귀속.
6. **규정 점검:** handcrafted reference를 구성한 canonical/AF block은 현재 발화 waveform에서만 계산한다. overlap·call concentration은 진단용이며 model input이 아니다. duration 외의 `startAt`/`endAt` 절대값 및 call-relative 정보는 feature에 포함하지 않는다.
7. **과거 단계별 비교와 현재 다음 단계:** 당시 Stage 1은 과거 캐시 59D의 학습/내부 검증 격차를 진단했고, Stage 2는 같은 과거 캐시 59D/M04-100 기준에 AF05만 추가한 61D를 비교했다. Stage 4의 clean 59D/eGeMAPS 비교와 Stage 6 이후 분류기 비교를 마쳤으며 현재는 59D 이중 동결 비교 기준을 보존한다. Stage 11은 팀 중복 회피로 SKIP, Stage 12는 성능 미평가 상태에서 자원/시간 실행 가능성 문제로 종료했다. 다음은 **Stage 13 — clean 59D에서 아직 시험하지 않은 모델 계열 재검토**이며 모델 후보는 미확정이다.

## 9. 재현성 / 출처 검증 문제

### 9.1 AF09 과거 최초 결과 `0.868970` 대 과거 캐시 기준 `0.869532`

| 항목 | 과거 최초 결과 | 과거 캐시 기준 재학습 |
|---|---|---|
| 정확도 | `0.8689703245245685` | `0.8695324665103912` |
| CM | `[[70926,14425],[8884,83656]]` | `[[71066,14285],[8924,83616]]` |
| 상태 | `historical_non_reproduced_result` | 과거 캐시로 정확히 재현; current clean reference 아님 |

2026-09-19 smoke는 기존 immutable cache만 strict-load했고, train `(711467,52)` / validation `(177891,52)`, WAV read 0, cache build 0, Official use false를 확인했다. 첫 helper 호출은 `refit_stale_or_missing`, 같은 process의 두 번째 호출은 동일 result object를 `reused_canonical`로 반환했다.

승인 상수 `0.869532466510391`과 CM에서 계산되는 exact ratio `0.8695324665103912`의 1-ULP 차이 때문에 최초 exact-float gate가 거부된 문제는 machine-epsilon Accuracy rounding만 허용하도록 관리됐다. CM과 recomputed prediction 검사는 exact하게 유지한다.

**상태:** old-cache canonical 52D `0.869532`의 재현은 확인됐지만 최초 `0.868970`은 같은 old cache로도 재현되지 않았다. 최초 결과의 divergence 원인은 **unresolved**이며, 현재 reference selection/delta/promotion에 사용하지 않는다.

### 9.2 과거 캐시와 clean 캐시의 제한된 감사 — 사용자 확인 결과

B5에서 MFCC40, AF02, AF04, AF09, AF10/AF11의 저장된 model-input 값은 old/clean exact 동일했고, current v2 E05 candidate의 `zcr`만 **3개 값**이 달랐다. 3개 행 모두 fixed split의 **internal-train**에 있다. 사용자가 보고한 old/clean 값은 다음과 같다(표시값은 반올림됨).

| Row | Historical zcr | Clean zcr | Absolute difference |
|---:|---:|---:|---:|
| 1 | `0.012207` | `0.169014` | `0.156807` |
| 2 | `0.010254` | `0.113924` | `0.103670` |
| 3 | `0.013672` | `0.164557` | `0.150885` |

이 3개 값은 validation 행 자체가 아니라 model 학습에 들어간 행에 있다. 학습 입력이 달라지면 MLP의 업데이트 경로가 달라져 여러 validation 예측이 바뀔 수 있다. B6에서 old cache만 읽어 같은 fixed split, train-only scaler, M04-100으로 52D/59D를 다시 조립·fit했을 때 **canonical 52D `0.8695324665103912`와 59D `0.8745580158636468`의 Accuracy·CM이 exact 재현**됐다고 사용자가 확인했다. 반면 최초 AF09 `0.8689703245245685`는 재현되지 않았다.

따라서 저장된 feature 입력의 차이와 old-cache canonical/59D score 재현은 확인됐다. 당시 raw WAV 내용 identity, prediction 및 model checkpoint가 없으므로 과거 모든 실행 원인을 완전히 복원했다고 주장하지 않는다. 최초 AF09 divergence는 미해결로 기록하고 **historical forensic은 여기서 종료**한다.

### 9.3 남은 출처 증명 한계

- E02/E03/E04는 committed Markdown 숫자만 있고 rendered output/result artifact가 없다.
- 기준 모델 오류 분석의 행 단위 오류 기록은 메모리에만 있었고 저장 근거 파일은 없다.
- Raw EDA Step 3에는 `KeyboardInterrupt`가 있으며 이후 cell output은 역사적 실행 증거이지 clean run 증거가 아니다.
- feature Notebook의 일부 source switch와 저장 output state가 불일치한다. E01/E05는 역사적 output evidence로는 쓰되 재현성 freeze로 부르지 않는다.
- M04-60은 metrics와 current-scaler identity output은 있으나 이를 생성한 fit cell이 저장돼 있지 않아 `current-session-only / incomplete provenance`다.
- AF04의 49D result는 저장됐지만 최초 47D 대비 standalone screening/promotion output은 없다.
- 위 historical score들은 대체로 Notebook output에만 있고 독립 JSON/CSV model-result artifact나 checkpoint는 없다. acoustic parquet/metadata는 feature cache이지 score artifact가 아니다. Stage 4 및 Stage 6 이후의 저장 result artifact는 별도 증거로 확인했으며 이로써 과거 artifact 부재가 해소된 것은 아니다.
- historical cache의 raw WAV/JSON 내용 hash 및 부모 provenance가 없어 당시 raw content identity는 `UNVERIFIABLE`이다. 앞으로는 raw content identity(비용이 크면 별도 audit), source manifest schema/version·inventory signature, stable record_key/order signature, config fingerprint, feature names/order, row count·split, parent artifact hash, builder/code contract version, 모델 split·scaler·seed·config·prediction fingerprint를 함께 보존한다.
- clean 특징 구성의 엄격한 검증/로드에 약 4분 30초가 걸렸다고 사용자가 보고했다. 재추출 없이 `REUSING_EXISTING_CLEAN_CHAIN`을 확인했으며 성능 최적화는 별도 개선 과제다.
- 최근 targeted pytest는 사용자 보고 기준 `92 passed, 40 errors`다. 40건은 assertion 실패가 아니라 Windows 기본 pytest temp 경로의 `tmp_path` 생성 중 `PermissionError [WinError 5]`로 발생했으므로 test logic failure로 분류하지 않는다. workspace 내부의 별도 `--basetemp`로 재실행 확인이 필요하며 이번 문서 갱신에서는 pytest를 실행하지 않았다.
- 현재 `docs/`, Mission 2 Notebook/source/test와 local cache는 Git-tracked reproducibility package로 동결되어 있지 않다. 이 사실은 결과를 무효화하지 않지만 다른 checkout의 즉시 재현을 보장하지 않는다.

## 10. 현재 상태

- **현재 수작업 특징 비교 기준:** clean AF10+AF11 59D = clean AF09 52D + speech activity boundary 4D + spectral entropy dynamics 3D. fingerprint `49d65a35d1e98c280f98e45cbed864ef8bac36268ccffbe81aa0985923c90de3`의 versioned chain을 사용하며 final representation이 아니다.
- **59D 이중 동결 비교 기준:** TabM 내부 검증 정확도 `0.8839795155` / ResNet `0.8837546588`. 동일 clean 59D, 고정 분할·시드 42·스케일러·임계값 0.5를 유지한 후속 표현 비교 기준이다. Stage 6 Stronger MLP `0.8760364493`은 직접 비교 기준, Stage 4 M04-100 MLP `(64,)`는 당시의 미수렴 기준으로 보존한다.
- **현재 채택 후보:** Stage 14 `E_TRG` (TabM + Tabular ResNet + GANDALF, 각 1/3)를 사용자 결정으로 채택했다. 이 선택은 Internal Validation 결과에 기반한다. Official Validation과 Stage 15 튜닝은 아직 하지 않았으며, 이를 통과한 대회 제출 최종 모델로 표현하지 않는다.
- **현재 clean B4:** 52D 내부 검증 정확도 `0.869594`, 혼동행렬 `[[71215,14136],[9062,83478]]`; 59D 내부 검증 정확도 `0.873760`, 혼동행렬 `[[72265,13086],[9371,83169]]`. 같은 B4를 2회 실행해 동일 결과를 얻었다는 사용자 보고다.
- **과거 Stage 1:** 과거 캐시 59D 학습 정확도 `0.8766112834467376`, 내부 검증 정확도 `0.8745580158636468`, 격차 `0.0020532675830907676`. 현재 clean 학습 정확도·격차로 재사용하지 않는다.
- **일반화 해석:** Stage 4 same-run clean 59D gap은 `0.0020278589` (§4.9), Stage 6 Stronger MLP는 `0.0057402178`, complex TabM/FT/ResNet은 `0.0038410018 / 0.0110020186 / 0.0164248295`다. Stage 4 비교만으로 optimization/representation을 구분하지 못했던 한계를 유지하며 historical gap을 current clean 결과로 재사용하지 않는다. 단일 seed의 gap과 공통 residual만으로 bottleneck을 확정하지 않는다.
- **현재 표현의 공식 검증 결과:** 없음. 과거 40D/42D Official 결과를 59D 점수로 재사용하지 않는다.
- **과거 특징 검증:** old-cache 61D(59D+AF05)는 Val `0.874738`, old-cache delta `+0.000180`, net +32였다. AF05는 HOLD이며 clean 59D에 대한 결과로 재해석하지 않는다.
- **과거 재현성 기준:** old-cache guarded 52D `0.8695324665103912`와 59D `0.8745580158636468`은 exact 재현된 과거 결과로 보존한다. 최초 52D `0.8689703245245685`는 미재현·원인 미해결이다.
- **수작업 특징 Stage 3:** 추가 handcrafted feature 조합 탐색은 종료했다.
- **Stage 4 비교:** 같은 실행의 M04-100 비교에서 eGeMAPSv02 88D를 단독 대체 표현으로 채택하지 않는다 (DROP). 내부 정확도는 `0.8639897465`, clean 59D는 `0.8737597742`이며 짝지은 정답 수의 순변화는 `-1,738`이다. 전체 비교와 근거 파일의 식별 정보는 §4.9에 보존한다.
- **Stage 10 결정 / 현재 비교 기준:** 추가 59D classifier 탐색을 일단 중단하고 **TabM + Tabular ResNet 이중 동결 비교 기준**를 보존한다. 당시 Stage 10의 수치·설명·결정은 §4.12에 그대로 남긴다.
- **Stage 11:** **SKIP — team duplication avoidance**. 팀원 김광희가 시간×주파수 표현 계열을 별도로 탐색 중이므로 개인 실험 중복을 피한다. 성능 저하에 따른 DROP이나 계산 불가로 종료한 것이 아니다.
- **Stage 12:** **CLOSED — resource/time feasibility(자원/시간 문제로 종료)**. Stage 12-A WavLM Base+는 기존 `45.3700 h` projection과 배치별 안정성 한계, Stage 12-B DPWavLM은 안정성을 만족하는 batch=1 경로의 실제 4,096행 처리량 측정와 전체 약 `63.87 h` projection을 근거로 현재 개인 로드맵에서 종료한다.
- **Stage 12 성능:** **PERFORMANCE NOT EVALUATED**. A/B 모두 전체 임베딩 추출·전체 임베딩 기반 분류기 학습·내부 검증 성능 평가·TabM/Tabular ResNet과 짝지은 비교·공식 검증·인코더 미세 조정을 진행하지 않았다. 표현 성능 열세로 해석하지 않는다.
- **Stage 13 상태:** **first-screen 완료** — EBM, Spline-GAM, RFF + Logistic SGD, DCNv2, TabNet, GANDALF를 clean 59D에서 고정 설정으로 실행했다. 여섯 후보의 Accuracy는 모두 TabM 및 Tabular ResNet보다 낮았고, GANDALF가 신규 후보 중 가장 높았다. 두 reference도 final model로 선정하지 않았다. 상세 결과·오류 분석은 §4.13.
- **복잡한 정형 모델 실행 검증 / 미확인 사항:** 저장 예측에서 정확도·혼동행렬·HCE·짝지은 비교/화자별/신뢰도 결과를 재계산해 일치 확인. 직접 짝지은 공통 오류 `17,236`, TabM 정답 수 순우위 `40`이다. 발화 길이/중첩은 **미분석/현재 추적 근거 파일 미확인**이며 과거 구간 결과와 섞지 않는다. 재로드 `EXACT_PASS`는 저장된 Colab 기록이며 이번 문서 작업에서 모델 학습/추론은 실행하지 않았다.
- **Stage 13 데이터·prediction 검증:** clean 59D 지문 `49d65a35d1e9…`, train/validation `711,467 / 177,891`, seed `42`, `call_id` overlap `0`, threshold `0.5`, Official Validation 미사용. exact record identity로 붙인 tracking은 오류 분석 전용이다. 6개 저장 prediction 모두 validation `177,891`행으로 재정렬되고 저장 지표를 재계산했으며, 6개 checkpoint reload gate가 모두 `EXACT_PASS`다. TabNet의 초기 in-memory prediction과 저장본 불일치는 남아 있으며 §4.13에 범위를 구분해 기록한다.
- **Stage 13 계획 변경:** 로드맵 Stage 13의 원래 후보 수 제한은 `1–2개`다. 이번 실행은 사용자의 명시적 요청으로 6개 family로 확장됐다. Colab Checkpoint D에서 여섯 설정이 첫 결과 전에 고정됐고 결과 기반 튜닝은 없었다. 변경 전의 별도 2개 후보 목록·승인 기록은 현재 저장소/Colab 출력에서 찾지 못했다.
- **보류 / 미해결:** AF05 61D, model/optimization/representation bottleneck 구분, AF04 promotion provenance, 최초 AF09 `0.868970` divergence 원인, M04-60 fit provenance. Stage 15 튜닝과 Official Validation도 현재 미실행이다.

현재 사용자가 채택한 모델 후보는 Stage 14의 149D 입력 3모델 동일가중치 앙상블 `E_TRG`다. clean 59D dual reference와 Stage 13 first-screen 결과는 비교 이력으로 보존한다. 이번 채택 근거는 재사용된 Internal Validation이므로 독립 일반화 성능을 뜻하지 않으며, Official Validation과 Stage 15 튜닝은 수행하지 않았다.

## 11. 전체 실험 요약표

`N/A`는 비교 조건이 달라 Delta를 계산하지 않았거나 실행 결과가 없다는 뜻이다.

| ID | 비교 기준 | 변경 | 차원 | 모델 | 평가 집합 | 정확도 | 변화량 | 판단 | 근거 |
|---|---|---|---:|---|---|---:|---:|---|---|
| E00 | — | MFCC40 baseline | 40 | LR | 공식 검증 | `0.809106` | N/A | baseline | baseline/error-analysis NB output |
| E01 | E00 | + duration | 41 | LR | 공식 검증 | `0.809115` | `+0.000009` | DROP | `m2-fe-cpc-run` |
| E02 | E00 | + rolloff | 41 | LR | 공식 검증 | `0.811509` | `+0.002403` | KEEP candidate | 커밋된 Markdown만 존재 |
| E03 | E00 | + ZCR | 41 | LR | 공식 검증 | `0.810116` | `+0.001009` | HOLD | 커밋된 Markdown만 존재 |
| E04 | E00 | + centroid | 41 | LR | 공식 검증 | `0.810500` | `+0.001394` | HOLD | 커밋된 Markdown만 존재 |
| E05 | E00 | + rolloff + ZCR | 42 | LR | 공식 검증 | `0.813117` | `+0.004011` | KEEP / 42D | `m2-fe-cpf-run` |
| M00 | E05 42D | LR | 42 | LR | 내부 검증 | `0.817332` | N/A | 비교 기준 | `checkpoint-m-code` |
| M01 | E05 42D | HGB | 42 | HGB | 내부 검증 | `0.841875` | `+0.024543` vs M00 | KEEP candidate | 동일 근거 |
| M02 | E05 42D | ExtraTrees | 42 | ExtraTrees | 내부 검증 | `0.819997` | `-0.021879` vs M01 | DROP | 동일 근거 |
| M03 | E05 42D | QDA | 42 | QDA | 내부 검증 | `0.797314` | `-0.044561` vs M01 | DROP | 동일 근거 |
| M04-30 | E05 42D | MLP max_iter 30 | 42 | MLP | 내부 검증 | `0.857952` | `+0.016077` vs M01 | KEEP candidate | 동일 근거 |
| M04-60 | M04-30 | max_iter 60 | 42 | MLP | 내부 검증 | `0.858773` | `+0.000821` | HOLD | `checkpoint-o-code`; incomplete fit provenance |
| M04-100 | M04-30 | max_iter 100 | 42 | MLP | 내부 검증 | `0.859532` | `+0.001580` | KEEP / AF reference | `0c522aec` |
| M00 | E05 42D | LR | 42 | LR | 공식 검증 | `0.813117` | N/A | historical Official | `checkpoint-i-run-code` |
| M01 | E05 42D | HGB | 42 | HGB | 공식 검증 | `0.837289` | N/A | historical Official | 동일 근거 |
| M04-30 | E05 42D | MLP max_iter 30 | 42 | MLP | 공식 검증 | `0.852948` | N/A | historical Official | `checkpoint-n-code` |
| AF01 | E05 42D | + Energy Dynamics | 45 | M04-100 | 내부 검증 | `0.857148` | `-0.002383` | DROP | `afe-diagnostics` |
| AF02 | E05 42D | + Pitch/Voicing | 47 | M04-100 | 내부 검증 | `0.866266` | `+0.006734` | KEEP / 47D | 동일 근거 |
| AF03 | E05 42D | + Spectral Flux | 45 | M04-100 | 내부 검증 | `0.853821` | `-0.005711` | DROP | 동일 근거 |
| AF04 | AF02 47D | + CPP | 49 | M04-100 | 내부 검증 | `0.868627` | N/A | KEEP(유지), 일부만 검증 | `afe-next-candidate-diagnostics` |
| AF05 | AF02 47D | + Delta-MFCC | 49 | M04-100 | 내부 검증 | N/A | N/A | historical DROP; score unresolved | 소스/Markdown만 존재 |
| AF06 | — | 포먼트 후보 | — | — | — | N/A | N/A | RETIRED / not run | 소스/Markdown |
| AF07 | AF04 49D | + MFCC Cross-correlation | 51 | M04-100 | 내부 검증 | `0.869128` | `+0.000500` | HOLD | `afe-next-candidate-diagnostics` |
| AF08 | AF04 49D | + Competing Pitch | 51 | M04-100 | 내부 검증 | `0.868403` | `-0.000225` | DROP | 동일 근거 |
| AF09 historical | AF04 49D | + Envelope Coherence | 52 | M04-100 | 내부 검증 | `0.868970` | `+0.000343` | QUARANTINE | 동일 근거 |
| AF09 canonical (old cache) | AF04 49D | + Envelope Coherence, guarded refit | 52 | M04-100 | 내부 검증 | `0.869532` | `+0.000905` vs old 49D | 과거 결과, 정확히 재현 | `afe-52d-reproduction`; user-reported B6 |
| AF09 clean | clean 49D prefix | + AF09 | 52 | M04-100 | 내부 검증 | `0.869594` | N/A | 현재 clean 52D 기본 구성 | 사용자 보고 B4 2회 실행 |
| AF07 on 52D | AF09 52D | + AF07 | 54 | M04-100 | 내부 검증 | `0.869639` | `+0.000107` | DROP | `afe-52d-54d-comparison` |
| AF10 | AF09 52D | + Boundary | 56 | M04-100 | 내부 검증 | `0.873074` | `+0.003541` | HOLD | `afe-residual-round-summary` |
| AF11 | AF09 52D | + Entropy Dynamics | 55 | M04-100 | 내부 검증 | `0.870747` | `+0.001214` | HOLD | 동일 근거 |
| AF12 | AF09 52D | + Temporal MFCC Contrast | 55 | M04-100 | 내부 검증 | `0.870111` | `+0.000579` | HOLD | 동일 근거 |
| AF10_AF11_COMBINATION (old cache) | AF09 old-cache 52D | + AF10 + AF11 | 59 | M04-100 | 내부 검증 | `0.874558` | `+0.005026` vs old 52D | 과거 결과, 정확히 재현 | `afe-residual-combination-summary`; user-reported B6 |
| AF10_AF11_COMBINATION (clean) | AF09 clean 52D | + AF10 + AF11 | 59 | M04-100 | 내부 검증 | **`0.873760`** | `+0.004165` vs clean 52D (CM 계산) | **현재 Stage 4 비교 기준** | 사용자 보고 B4 2회; clean 특징 구성 지문은 §4.8 |
| eGeMAPSv02 Functionals | clean AF10+AF11 59D | representation replacement | 88 | M04-100 | 내부 검증 | `0.8639897465` | `-0.0097700277`; `+1,738` errors | **DROP(단독 대체 제외)**; 일부 구간의 변화는 보완적 활용의 별도 검증 후보 | saved Stage 4 run `clean59d_egemaps88d_20261004T190153_4dbd7177`; §4.9 |
| AF10_AF12_COMBINATION | AF09 52D | + AF10 + AF12 | 59 | M04-100 | 내부 검증 | `0.872574` | `+0.003041` | 판단 대기 | 동일 근거 |
| AF11_AF12_COMBINATION | AF09 52D | + AF11 + AF12 | 58 | M04-100 | 내부 검증 | `0.872107` | `+0.002575` | 판단 대기 | 동일 근거 |
| AF10_AF11_AF12_COMBINATION | AF09 52D | + AF10 + AF11 + AF12 | 62 | M04-100 | 내부 검증 | `0.876042` | `+0.006510` | 판단 대기 | 동일 근거 |
| AF05 (old-cache 59D recheck) | AF10+AF11 old-cache 59D | + Delta-MFCC 2D | 61 | M04-100 | 내부 검증 | `0.874738` | `+0.000180` vs old 59D | HOLD / reference 미채택 | 사용자 제공 Stage 2 기록 |
| Complex TabM | Stage 6 Stronger MLP | 고정된 최초 비교 모델/학습 절차 | 59 | TabM | Internal (`177,891`) | **`0.8839795155`** | `+0.007943066260`; `-1,413` errors | **Frozen reference A**; final 미선정 | run `20261006T031510Z_a9a01a2d`; prediction/metrics/config/history; §4.11–4.12 |
| Complex FT-Transformer | Stage 6 Stronger MLP | 고정된 최초 비교 모델/학습 절차 | 59 | FT-Transformer | Internal (`177,891`) | `0.8821975255` | `+0.006161076165`; `-1,096` errors | 결과 보존; dual reference 미포함 | same complex run; §4.11 |
| Complex Tabular ResNet | Stage 6 Stronger MLP | 고정된 최초 비교 모델/학습 절차 | 59 | Tabular ResNet | Internal (`177,891`) | **`0.8837546588`** | `+0.007718209465`; `-1,373` errors | **Frozen reference B**; final 미선정 | same complex run; prediction/metrics/config/history; §4.11–4.12 |
| Stage13-EBM | TabM | EBM additive first-screen | 59 | EBM | Internal (`177,891`) | `0.839604` | `-0.044375` | 결과 보존; 두 reference보다 낮음 | Stage13 run `20261009T051303Z_7d149a2b`; §4.13 |
| Stage13-Spline-GAM | TabM | train-quantile cubic spline + logistic | 59 | Spline-GAM | Internal (`177,891`) | `0.851836` | `-0.032143` | 결과 보존; 두 reference보다 낮음 | same Stage13 run; §4.13 |
| Stage13-RFF-SGD | TabM | fixed-seed RFF + batched Logistic SGD | 59 | RFF + Logistic SGD | Internal (`177,891`) | `0.831453` | `-0.052527` | 결과 보존; 두 reference보다 낮음 | same Stage13 run; §4.13 |
| Stage13-DCNv2 | TabM | cross network + parallel MLP | 59 | DCNv2 | Internal (`177,891`) | `0.877830` | `-0.006150` | 결과 보존; 두 reference보다 낮음 | same Stage13 run; §4.13 |
| Stage13-TabNet | TabM | sequential feature-attention steps | 59 | TabNet | Internal (`177,891`) | `0.877509` | `-0.006470` | 결과 보존; 두 reference보다 낮음; 최초 prediction mismatch 미해결 | same Stage13 run; §4.13 |
| Stage13-GANDALF | TabM | gated feature interactions | 59 | GANDALF | Internal (`177,891`) | `0.881748` | `-0.002232` | 신규 후보 최고 point estimate; 두 reference보다 낮음 | same Stage13 run; §4.13 |
| Stage14-TabM-F01-F02 | Stage14 same-run member | Clean59D + F01 80D + F02 10D | 149 | TabM | Internal (`177,891`) | `0.886430` | N/A | 앙상블 멤버 | Drive run `20261010T012333Z_8de41728_stage14_joint_f01_f02_attempt_02`; §4.14 |
| Stage14-ResNet-F01-F02 | Stage14 same-run member | Clean59D + F01 80D + F02 10D | 149 | Tabular ResNet | Internal (`177,891`) | `0.885492` | N/A | 앙상블 멤버 | same Stage14 run; §4.14 |
| Stage14-GANDALF-F01-F02 | Stage14 same-run member | Clean59D + F01 80D + F02 10D | 149 | GANDALF | Internal (`177,891`) | `0.885846` | N/A | 앙상블 멤버 | same Stage14 run; §4.14 |
| Stage14-E_TR | Stage14 members | TabM + ResNet, each 1/2 | 149/member | fixed equal-weight ensemble | Internal (`177,891`) | `0.887217` | `+0.0787 pp` vs best member | candidate; below E_TRG | same Stage14 run; §4.14 |
| Stage14-E_TG | Stage14 members | TabM + GANDALF, each 1/2 | 149/member | fixed equal-weight ensemble | Internal (`177,891`) | `0.888151` | `+0.1720 pp` vs best member | candidate; below E_TRG | same Stage14 run; §4.14 |
| Stage14-E_RG | Stage14 members | ResNet + GANDALF, each 1/2 | 149/member | fixed equal-weight ensemble | Internal (`177,891`) | `0.888167` | `+0.2322 pp` vs best member | strongest 2-member candidate; E_TRG higher by `0.0315 pp` | same Stage14 run; §4.14 |
| Stage14-E_TRG | Stage14 members | TabM + ResNet + GANDALF, each 1/3 | 149/member | fixed equal-weight ensemble | Internal (`177,891`) | **`0.888482`** | `+0.2052 pp` vs best single member; `+0.0315 pp` vs E_RG | **사용자 채택 후보**; Official/Stage15 미실행 | same Stage14 run; §4.14 |

## 근거 파일 목록

- 규정: `docs/dcc_rules.md`
- 기준 모델: `e670503:notebooks/mission2_yeonwoo_baseline.ipynb`
- 원본 데이터 탐색: `6e4c7d6:notebooks/mission2_raw_data_eda.ipynb`
- 기준 모델 오류 분석: `26d48b6:notebooks/mission2_baseline_error_analysis.ipynb`
- E계열 특징 실험: `860acea:notebooks/mission2_feature_experiments.ipynb`
- 모델 실험: `notebooks/mission2_model_experiments.ipynb`, `src/dcc_data/mission2_model_experiments.py`
- 음향 실험: `notebooks/mission2_acoustic_feature_expansion.ipynb`, `src/dcc_data/mission2_acoustic_feature_expansion.py`
- 로컬 불변 음향 캐시 메타데이터: `data/processed/mission2_acoustic_feature_expansion/training_acoustic_expansion_2b612b68d75e.json`, `training_acoustic_expansion_a58da347bd3e.json`, `training_acoustic_expansion_4c5be191a568.json`, `training_acoustic_expansion_ff08df8e3b5b.json`
- Stage 6 clean 59D 모델 계열 비교: `data/processed/mission2_model_family_screen/20261005T113752Z_8ab60cd0`; `comparison.csv`, `contract.json`, scaler/environment, `candidates/stronger_mlp/{config.json,metrics.json,validation_predictions.parquet}`; §4.10
- 복잡한 심층 정형 모델 실행: [Drive run `20261006T031510Z_a9a01a2d`](https://drive.google.com/drive/folders/1WDSriGQDQ3k0xoUQsdEIwWSicETEbic9), [comparison.csv](https://drive.google.com/file/d/1IluaSIF6xHOkh3In_Ktm-JcUS5Ejluh4/view), [paired_comparison.csv](https://drive.google.com/file/d/1vpQtKxnEaJmY68a9blO53OEpkLc_fcWn/view), [paired_slices.csv](https://drive.google.com/file/d/1urw6Rn0Yl_W0lfHXlcVgobVlk_kjIj4L/view), candidate config/metrics/history/predictions; §4.11–4.12
- 복잡 정형 모델 실행의 누락된 추적 근거: [analysis_metadata.json](https://drive.google.com/file/d/1S-OrHQAdDIgMRi_BL_4MvXPyP3D7CJS5/view), `tracking_used=false`; duration/overlap 미분석. checkpoint reload는 후보 metrics의 saved `EXACT_PASS`로 확인했으며 이번 문서 작업에서 재실행하지 않음

- Stage 12-A 기존 실행 Notebook: `notebooks/mission2_raw_waveform_pretrained_speech_colab.ipynb` (로컬 실제 파일 확인). 아래 WavLM 역사 기록의 수치·근거 파일 경로는 보존한다. 이 Notebook을 DPWavLM 실행본으로 간주하지 않는다.
- Stage 12-B 확인된 Drive 근거 파일 디렉터리: [run `dpwavlm_sp075_20261008T152007Z_4a2fee71`](https://drive.google.com/drive/folders/1dR6Z8Q1XwObtFt76UnYhxUXnpeUFptrl); 저장 보고서에 명시된 Colab 경로는 `/content/drive/MyDrive/prelim/data/processed/mission2_dpwavlm_stage12b/dpwavlm_sp075_20261008T152007Z_4a2fee71`. 로컬에 같은 실행 디렉터리가 있는 것으로 주장하지 않는다.
- Stage 12-B 데이터/모델 동일성: 위 실행의 [data_identity.json](https://drive.google.com/file/d/1-0ZfAz6iNtc4rg1RNEqUDYeR5cpBNCmr/view), [model_load_preflight.json](https://drive.google.com/file/d/1-PPAJ8yF5tYLOh_VhCrHHmjuE7MWkpOu/view), [fresh_row_plan.json](https://drive.google.com/file/d/1CgLwt40CobsFds_8D2MnhWev9nhZi_6s/view). 고정 분할·clean-59D/데이터 동일성·실측 매개변수 수·기존 WavLM 행 제외 근거를 읽어 확인했다.
- Stage 12-B 안정성: 위 run의 [batch1_fp32_tolerance.json](https://drive.google.com/file/d/1zcPJrWsNf9jpUTjz5zcyh-CGmVJKnv5Y/view), [batching_stability.json](https://drive.google.com/file/d/1-vF971yYIpJV6YIxcgDyyyl0qRiu9zY4/view), `canonical_batch1_stability_7e7f8526/`의 [canonical_batch1_stability.json](https://drive.google.com/file/d/1GthU-2POnEvg3_plLt3v64zq_2isbK_J/view). 허용오차 사전 고정, 길이 혼합 batch=8 FAIL(실패), 기준 batch=1 PASS(통과)를 저장 보고서에서 확인했다.
- Stage 12-B 실제 처리량 측정: 위 run의 `fresh_benchmark_4096_batch1_ed94bfca/` 아래 [benchmark_4096_report.json](https://drive.google.com/file/d/1_8eTK7ddqRJDMf_3Y6_uwKksFx38UYq-/view), `benchmark_manifest.json`, [embedding_shards/](https://drive.google.com/drive/folders/15ONpzh2k3NUBgo7c2jNe3GdaF4Ad8Kpo). 처리량 측정 보고서를 읽어 4,096행·16개 분할 파일·저장된 재로드/동일성 PASS(통과)를 확인했다. 이번 문서 갱신에서 분할 파일 재추론·재검증은 실행하지 않았다.
- Stage 12-B 전체 실행 시간/저장 용량 예상값: 위 run의 [full_889358_runtime_storage_projection_ee102fb6.json](https://drive.google.com/file/d/1Wgb5UQ3Y6tHb6BH-GWlBbGj531O7SNvu/view). `PROJECTED_NOT_EXECUTED`; 분류기 학습/평가 시간은 `NOT_MEASURED_NOT_RUN`, 예상값은 `null`이다. 별도 DPWavLM 실행 Notebook 경로는 확인되지 않아 추측해 추가하지 않는다.
- Stage 13 실행 Notebook: [mission2_stage13_untested_model_family_first_screen.ipynb](https://colab.research.google.com/drive/1aW0NTaCbnNv7-QAUm3lUZIO3Ci4XdIIJ). 저장된 Checkpoint B/C/D와 모델별 Checkpoint E–J, 통합 분석 Checkpoint K–M output을 대조했다. run root: `/content/drive/MyDrive/prelim/data/processed/mission2_stage13_untested_model_family_first_screen/20261009T051303Z_7d149a2b`.
- Stage 13 공통·통합 artifact: run root의 `contract.json`, `environment.json`, `run_summary.json`, `reference_metrics.json`, `model_artifact_audit.json`, `implementation_patch_recovery.json`, `analysis_patch_recovery.json`; 통합 결과 `eight_model_comparison.csv`/`.parquet`, `paired_error_comparison.csv`, `common_reference_error_corrections.csv`, `class_recall_comparison.csv`, `slice_comparison.csv`.
- Stage 13 행 단위 검토·후보 artifact: `row_level_error_ledger.parquet`, `misclassified_sample_review.parquet`, `high_confidence_error_cases.parquet`, `model_unique_correction_cases.parquet`, `all_model_common_errors.parquet`, `validated_tracking.parquet`; 각 후보 폴더에는 고정 config, checkpoint, `predictions.parquet`, `metrics.json`, status/history와 해당하는 복구 근거가 저장돼 있다. TabNet 상세 근거는 `tabnet/failure.json`, `tabnet/attempt_02/failure.json`, `tabnet/checkpoint_recovery_evidence.json`; 전체 reload·row audit는 `model_artifact_audit.json`이다. 파일은 Colab output에서 확인했으며 이번 문서 작업에서 다시 읽거나 재추론하지 않았다.

### Stage 11 — 최종 SKIP 사유 (2026-10-09 상태 갱신)

**상태:** `SKIP — team duplication avoidance`.

팀원 김광희가 시간×주파수 표현 계열을 별도로 탐색 중이므로 개인 실험 중복을 피하기 위해 SKIP한다. 이것은 팀 내 탐색 분담에 따른 최종 상태이며 성능이 나빠서 DROP하거나 계산 불가로 종료한 결정이 아니다. 기존 준비·실행 이력이 있더라도 그 결과를 이 최종 SKIP 사유로 치환하지 않는다. 이 사유는 사용자가 확인한 팀 운영 결정이며 공식 대회 규정이 아니다.

### Stage 12-A — WavLM Base+ 현재 상태 (기존 기록 보존)

**상태:** `RESOURCE/TIME-BUDGET INFEASIBLE`
**성능:** `PERFORMANCE NOT EVALUATED`.

- **질문·선정 근거:** 현재 발화 파형의 사전학습 음성 표현이 clean 59D의 수작업 요약이 놓치는 정보를 보완할 가능성을 확인하되, 전체 추출 전에 계산 가능성과 표현 안정성을 검증한다. WavLM의 선정 근거는 기존 논문/모델 카드의 화자 식별·준언어적 정보 전이 동기이며 Mission 2 정확도 향상을 미리 가정한 것이 아니다.
- **정확한 모델/버전:** `microsoft/wavlm-base-plus` / `4c66d4806a428f2e922ccfa1a962776e232d487b`. 아래 기존 결과·G4 재실행·최적화 기록은 모두 Stage 12-A의 누적 이력이며 수치·판단·근거 경로를 보존한다.
- **검증된 기존 전체 경로 예상 시간:** 내부 학습 + 내부 검증 임베딩 대상 전체에 대한 G4 실행의 약 `45.3700 h` 예상 시간을 보존한다. 실제 전체 추출 시간이나 성능 결과가 아니다.
- **추가 배치 최적화의 한계:** 빠른 예열/캐시 결과는 동일 행의 반복 읽기와 통제되지 않은 캐시 상태를 포함해 통제된 전체 경로 처리량 측정이 아니었다. 후속 패딩/배치별 표현 안정성 문제도 확인되어 안정성을 만족하는 최적화 기준 경로를 확보하지 못했다. 이 빠른 결과로 기존 계산 기준 판단을 대체하지 않는다.
- **최종 판단:** 자원/시간 예산 제약으로 종료하며 WavLM 정확도가 나빴다고 해석하지 않는다. Stage 12-B의 DPWavLM 처리량 측정/예상값과 합치지 않는다.

### Stage 12 기존 결과 — 원시 파형 기반 사전학습 음성 표현

**상태:** `RESOURCE/TIME-BUDGET INFEASIBLE`
**성능:** `NOT EVALUATED` — 계산 가능성 기준을 통과하지 못했다 (FAIL). 전체 학습/검증 임베딩 추출과 분류기 평가는 시작하지 않았다.
**실행 근거 파일:** `data/processed/mission2_raw_waveform_pretrained_speech/stage12_wavlm_d54a03f1f0_871fb4d8c3_a13766fc` (Colab: `/content/drive/MyDrive/prelim/data/processed/mission2_raw_waveform_pretrained_speech/stage12_wavlm_d54a03f1f0_871fb4d8c3_a13766fc`)

- **인코더 후보:** `microsoft/wavlm-base-plus@4c66d4806a428f2e922ccfa1a962776e232d487b`, 매개변수 94,381,936개, Microsoft UniSpeech의 CC-BY-SA-3.0 라이선스. 논문·모델 카드의 화자 식별 및 준언어적 표현 전이 근거로 선택했으며 Mission 2 정확도를 예측해 선택한 것이 아니다. 영어로 사전학습된 체크포인트라는 전이 한계가 있다. 후보 출처와 고정 설정은 해당 실행의 `candidate_research.md`와 `config.json`에 저장돼 있다.
- **데이터/분할:** 저장된 Stage 11 대회 제공 Training 전송 증명과 현재 `data/raw_original/대학부 데이터` ZIP을 재사용했다. 다운로드·압축 해제·재구성·새 원본 바이트 해시는 수행하지 않았다. clean 59D 지문 `49d65a35d1e98c280f98e45cbed864ef8bac36268ccffbe81aa0985923c90de3`, 고정 시드 `42`, 학습 711,467행 / 내부 검증 177,891행, 통화 23,360 / 5,840개, 통화 중복 `0`. CP1 행 동일성은 PASS였다. CP3/CP4는 Training 행만 사용했고 사전 점검의 검증 행 접근/학습은 `0`이었다.
- **동결 표현/분류기 계약:** 8→16 kHz 결정적 sinc 재표본화, 동결된 `eval()` WavLM, 마지막 은닉층의 유효 프레임 마스크 평균, float32 768D 임베딩, 인코더 최소 입력 길이를 위한 오른쪽 0 채우기만 허용한다. 학습 집합에만 맞춘 `StandardScaler`와 하나의 5-epoch `SGDClassifier(loss=log_loss, penalty=l2, alpha=0.0001, average=True, seed=42)`를 사용하며 판정 임계값은 `0.5`다.
- **CP3 사전 점검:** Training 3행에서 PASS. 배치 크기별 안정성의 최대 절대 차이는 `0.00069505` (허용오차 `0.002`), NaN/Inf `0`, 건너뛴 행 `0`, 검증 접근/학습/기울기 갱신 `0`. 짧은 행 1개에는 최소 수용영역에 맞춘 결정적 패딩을 적용했다.
- **CP4 실제 처리량 측정 (T4, CUDA 13.0, AMP 사용):** 발화 3,000건 / 음성 길이 5,948.177초 / 경과 시간 639.359초. 처리량 `4.692203`행/초, 음성 `9.303350`초/초, 최대 GPU 메모리 `643,854,336`바이트, CPU RSS 3.197→3.698 GB. 분할 파일 쓰기는 `10,215,510`바이트, 속도 `30.684 MiB/s`. 메모리 부족 조정 `0`, 건너뛴 행 `0`. 발화 길이별 처리량(행/초): `<0.5s` `0.952286`, `0.5–1s` `3.752905`, `1–2s` `9.136020`, `2–5s` `12.668319`, `>=5s` `9.706472`.
- **CP5 예상 시간/계산 기준:** 혼합 표본 기반 예상 시간 `52.684431`시간, 발화 길이별 비중 가중 예상 시간 `52.678306`시간, 분류기/평가/재로드/근거 파일 처리 예상 시간 `113.120`초. 두 추정의 차이는 `22.051`초 (약 `0.0061`시간). 사전 정의한 18–22시간 추가 확인 구간에 해당하지 않았다. 보수적인 전체 경로 예상 시간 `52.684431`시간이 20시간 제한을 넘었으므로 계산 기준은 `FAIL`이다.
- **판단:** `RESOURCE/TIME-BUDGET INFEASIBLE` (자원/시간 예산상 실행 불가), `PERFORMANCE NOT EVALUATED` (성능 미평가). Stage 12 검증 성능이나 짝지은 비교 결과는 없으며 사전학습 음성 표현의 성능 열세를 보여 주는 결과가 아니다. Stage 13은 시작하지 않았다.

### Stage 12 G4 계산 가능성 기준 재실행 — 2026-10-08

**상태:** `RESOURCE/TIME-BUDGET INFEASIBLE`
**성능:** `PERFORMANCE NOT EVALUATED`
**실행 근거 파일:** `data/processed/mission2_raw_waveform_pretrained_speech/stage12_wavlm_d54a03f1f0_871fb4d8c3_3d88f294` (Colab: `/content/drive/MyDrive/prelim/data/processed/mission2_raw_waveform_pretrained_speech/stage12_wavlm_d54a03f1f0_871fb4d8c3_3d88f294`)

- **인코더:** `microsoft/wavlm-base-plus@4c66d4806a428f2e922ccfa1a962776e232d487b` (매개변수 94,381,936개, CC-BY-SA-3.0). 동결된 `eval()` 추론, 결정적 8→16 kHz 재표본화, 마지막 은닉층의 유효 프레임 마스크 평균을 사용한다. 추론은 AMP float16, 임베딩은 float32이며 ASR/CTC/text는 사용하지 않는다.
- **데이터/분할:** 검증된 대회 제공 Training ZIP의 출처 정보를 재사용했으며 다운로드·압축 해제·재구성·전체 원본 바이트 재해시는 하지 않았다. clean 59D 지문 `49d65a35d1e98c280f98e45cbed864ef8bac36268ccffbe81aa0985923c90de3`, 데이터 동일성 지문 `871fb4d8c323579a4398adc4119c8ec3583b218545183b7578d40ef9c16e4b82`, 시드 `42`, 내부 학습 발화 711,467건 / 통화 23,360개, 내부 검증 177,891 / 5,840, 통화 중복 `0`.
- **G4 실행 환경:** NVIDIA RTX PRO 6000 Blackwell Server Edition, CUDA 13.0, Python 3.13.15, PyTorch/Torchaudio 2.11.0+cu130, Transformers 5.18.0. 모델 로드 4.3385초와 CUDA 예열 3.9277초는 별도로 기록하고 정상 처리 상태의 처리량에서 제외했다.
- **CP3 사전 점검:** 내부 학습 3행에서 PASS. `(3, 768)`의 유한한 float32 임베딩, 배치 크기별 안정성 최대 절대 차이 `0.000496` (허용오차 `0.002`), 건너뛴 행 `0`. 인코더는 동결/평가 모드다.
- **CP4 실제 처리량 측정:** 내부 학습 발화 3,000건 / 음성 길이 5,948.177초 / 전체 경로 550.658초 (WAV 입출력·잘라내기·재표본화·배치/패딩·인코더·풀링·전송·분할 파일 쓰기 포함). 처리량 `5.4480`행/초, 음성 `10.8019`초/초. 배치 크기 8 (관측 1–8), AMP 사용, 최대 GPU 메모리 643,584,000바이트, CPU RSS 3,079,704,576→3,785,093,120바이트, 건너뛴 행 `0`. 캐시 쓰기는 10,215,510바이트 / 1.3288초 (`7.331 MiB/s`).
- **실측 발화 길이별 처리량 (행/초):** `<0.5s` `1.0216`, `0.5–1s` `3.7777`, `1–2s` `13.2835`, `2–5s` `20.2303`, `>=5s` `23.2724`. 측정 표본 비중은 고정 Training 비중과 일치했다 (7.93%, 24.26%, 33.10%, 28.36%, 6.35%).
- **예상 시간:** 대표 혼합 표본 기반 `163,331.998`초 (`45.3700`시간), 발화 길이별 비중 가중 `163,305.565`초 (`45.3627`시간), 차이 `26.433`초. 분류기/평가/근거 파일 처리 예산을 포함한 보수적 전체 경로는 `45.3700`시간이며 모델 로드/예열 `8.2662`초는 별도로 측정했다. 두 예상 시간 모두 고정 20시간 기준을 넘었으며, 기준 부근에서의 사전 정의된 추가 확인은 적용 대상이 아니었다.
- **기준/판단:** `FAIL`, `RESOURCE/TIME-BUDGET INFEASIBLE`, `PERFORMANCE NOT EVALUATED`. 전체 학습/검증 임베딩 추출과 분류기 학습/평가는 시작하지 않았고 성능 지표나 짝지은 비교 결과도 없다. 자원/시간 문제의 결과이며 표현 성능이 더 낮다는 증거가 아니다. Stage 13은 시작하지 않았다.

### Stage 12 보충 검증 — batch-64 처리량 — 2026-10-08

**목적:** 일회성 처리량 민감도 측정이다. 기존 CP5 계산 기준은 `45.3700`시간으로 `FAIL`을 유지한다.

- **표본:** 기존 batch-8 실행과 같은 내부 학습 3,000행. 행 위치 SHA-256 `0d633d718b78de6cff16578ac97881c149c9b862b48e9bbc84abf4c3d0c2eb0f`이며 표본 정렬이 정확히 일치함을 확인했다.
- **요청/관측 배치:** 요청 `64`, 관측 최소/최대 `1` / `10`. 메모리 부족 조정 `0`, 건너뛴 행 `0`. 정확히 같은 길이끼리 묶는 방식으로 인해 이 표본의 실제 최대 배치는 10이었다.
- **실측:** 전체 경로 `33.899`초, `88.4991`행/초, 음성 `175.4694`초/초, 최대 GPU 메모리 `643,584,000`바이트, CPU RSS `3,766,890,496`→`4,151,275,520`바이트.
- **보충 예상 시간:** 혼합 표본 기반 `2.8144`시간, 발화 길이별 비중 가중 `2.8146`시간, 보수적 값 `2.8146`시간. batch 8 대비 관측 처리량 비율은 `16.24x`였으나 동일 행을 다시 읽었고 파일 시스템/캐시 상태를 통제하지 않았으며 실제 배치 크기는 64에 도달하지 않았다. 따라서 배치 크기만의 효과를 분리한 결과가 아니며 이 예상 시간은 기존 CP5 기준을 대체하지 않는다.
- **근거 파일:** `data/processed/mission2_raw_waveform_pretrained_speech/stage12_wavlm_d54a03f1f0_871fb4d8c3_3d88f294/engineering_batch64_check_20261008/batch_size_sensitivity.json`. 전체 학습/검증 임베딩 추출과 분류기 평가는 시작하지 않았으며 Stage 13도 시작하지 않았다.

### Stage 12 보충 검증 — 발화 길이 구간별 배치의 표현 동등성 — 2026-10-08

**목적:** 새 처리량 측정 전에 발화 길이 구간별 배치 경로가 기존 표현을 보존하는지 검증한다. 기존 Stage 12 계산 판단은 45.3700시간의 RESOURCE/TIME-BUDGET INFEASIBLE을 유지한다.

- **데이터:** 5개 발화 길이 구간의 새로운 내부 학습 15행. 검증 집합 접근, 분류기/스케일러 학습은 없었고 건너뛴 행도 없었다.
- **원시 파형 오른쪽 패딩:** 5개 발화 길이 구간 내 표현 동등성 검증에 실패했다. 최대 절대 임베딩 차이 1.690806은 사전 허용오차 0.002를 넘었다. 근거: /content/drive/MyDrive/prelim/data/processed/mission2_raw_waveform_pretrained_speech/stage12_wavlm_d54a03f1f0_871fb4d8c3_3d88f294/engineering_batch64_check_20261008/duration_bucket_padding_20261008T035716Z_263b5217/duration_bucket_padding_equivalence.json.
- **표현 보존 후보:** GroupNorm 통계를 유지하기 위해 동결된 WavLM 특징 추출기를 파형별로 실행한 뒤 각 길이 구간 내 추출 프레임 시퀀스에만 패딩하고 공식 은닉 프레임 마스크를 사용했다. 구간 내 최대 절대 차이는 0.00843456, 15개 전체를 패딩한 배치의 최대 차이는 0.01971160이었다. 모두 0.002를 넘어 사전 점검에 실패했다. 근거: /content/drive/MyDrive/prelim/data/processed/mission2_raw_waveform_pretrained_speech/stage12_wavlm_d54a03f1f0_871fb4d8c3_3d88f294/engineering_batch64_check_20261008/feature_stage_batch_preflight_20261008T040524Z_c4456a69/feature_stage_batch_preflight.json (SHA-256 14ef77f96f7c5a69d340f9721cec2392117b17174e218776d0235f7a0f7fbb16).
- **판단:** 새로운 5,000–10,000행 처리량 측정은 시작하지 않았고 새 실행 시간 예상값이나 계산 기준 판단도 생성하지 않았다. 이전의 동일 행 batch-64 결과는 보충 기록이며 CP5를 대체하지 않는다. 전체 임베딩 추출/분류기 평가는 미시작이고 Stage 13도 시작하지 않았다.
### Stage 12 보충 검증 — 동일 프레임 수 배치의 표현 동등성 — 2026-10-08

**목적:** 원시 파형 패딩과 가변 은닉 프레임 패딩 실패 후 마지막으로 표현 보존 소규모 배치를 검사했다. 동결된 특징 추출기는 발화별로 실행하고 유효 프레임 길이가 같은 Transformer 인코더 입력만 배치로 묶었으며 은닉 프레임 패딩은 하지 않았다.

- **표본:** 이전 15행 동등성 진단에서 선택한 내부 학습 8행 (유효 프레임 수 24, 49, 99, 249가 같은 4쌍). 검증 접근/학습은 없었고 건너뛴 행도 없었다.
- **결과:** 최대 절대 임베딩 차이 0.00239822가 사전 허용오차 0.002를 넘어 실패했다. 각 쌍의 최대 차이는 0.00218387, 0.00239822, 0.00102359, 0.00080337이었다.
- **판단:** 새로운 처리량 측정이나 계산 예상은 수행하지 않았다. 기존 CP5는 45.3700시간의 RESOURCE/TIME-BUDGET INFEASIBLE을 유지한다. 전체 추출/분류기 평가는 미시작이며 Stage 13도 시작하지 않았다.
- **근거 파일:** /content/drive/MyDrive/prelim/data/processed/mission2_raw_waveform_pretrained_speech/stage12_wavlm_d54a03f1f0_871fb4d8c3_3d88f294/engineering_batch64_check_20261008/same_frame_batch_preflight_20261008T040752Z_3ebbc60a/same_frame_batch_preflight.json (SHA-256 114ec217e99fce2f4d9e710038001668fe3b4e700c8ef085739aff529a088193).

### Stage 12 보충 검증 — 기준 패딩 경로의 안정성 — 2026-10-08

**목적:** 결정적인 표본 길이 패딩과 더 큰 동질 배치가 새로운 Stage 12 처리량 측정에 적합한지 다시 평가한다. 구현 측면의 실행 가능성 점검이며 패딩 선택에 정확도를 사용하지 않았다.

- **체크포인트/데이터:** 동결된 `microsoft/wavlm-base-plus@4c66d4806a428f2e922ccfa1a962776e232d487b`, 내부 Training만 사용, 데이터 동일성 `871fb4d8c323579a4398adc4119c8ec3583b218545183b7578d40ef9c16e4b82`, clean 59D 기준 지문 `49d65a35d1e98c280f98e45cbed864ef8bac36268ccffbe81aa0985923c90de3`, 시드 42. 고정 분할은 학습 711,467행 / 통화 23,360개, 검증 177,891행 / 통화 5,840개, 통화 중복 0을 유지한다. 표본은 발화 길이 구간별 새로운 Training 발화 192건 (32, 32, 64, 32, 32)이며 이전 측정/진단 3,015행을 제외했다. 검증 행 읽기 0, Official Validation 미사용.
- **구현 정정:** 첫 안정성 셀은 선택 표본이 하나의 2의 거듭제곱 패딩 목표 길이를 공유한다고 잘못 가정해 결과 생성 전에 멈췄다. 정정 실행은 후보별 결정적 목표 길이에 따라 순전파 배치를 묶었다. 새 버전의 실행을 저장했으며 이전 기준 판단이나 진단 근거 파일을 덮어쓰지 않았다.
- **사전 고정 합격 기준:** 코사인 유사도 최솟값 >= 0.999, 중앙값 >= 0.9999, 상대 L2 중앙값 <= 0.005, p99 <= 0.01, 일관된 평균 이동 비율 <= 0.001. 유한한 값만 허용하고 건너뛴 행은 0이어야 한다.
- **후보 실측 결과:** 250 ms 단위 4,000표본 배수 패딩은 18/38 조건에서 실패했다. 전체 상대 L2 중앙값/p99는 0.002375/0.015217, 일관된 평균 이동 비율 최댓값은 0.001472였다. 다음 2의 거듭제곱 길이 패딩은 15/36 조건에서 실패했고 상대 L2 중앙값/p99는 0.002067/0.011335, 일관된 평균 이동 비율 최댓값은 0.000909였다. 두 후보 모두 사전 고정한 배치 대 단독 추론 표현 안정성 기준을 통과하지 못했다.
- **해석 한계:** 저장 조건들은 각각 패딩 규칙별 단독 추론 기준과 비교한다. 원래 배치와 반복 배치, 원래 배치와 함께 묶인 행의 순서를 바꾼 배치 사이의 직접 짝지은 차이는 계산하지 않았다. 반복/순서 불변성을 독립적으로 확인했다고 주장하지 않는다. 배치 대 단독 추론의 실패만으로 작업을 중단했다.
- **중단:** Drive와 임시 작업 공간의 입출력 비교, 새로운 5k–10k 처리량 측정, 새 실행 시간 예상 또는 전체 추출은 시작하지 않았다. 성능은 `NOT EVALUATED`를 유지한다. 이전 45.3700시간 계산 기준은 역사 측정 기록으로 보존하며 처리량을 측정하지 않은 이번 패딩 후보의 예상값으로 쓰지 않는다. Stage 13은 시작하지 않았다.
- **근거 파일:** `/content/drive/MyDrive/prelim/data/processed/mission2_raw_waveform_pretrained_speech/stage12_wavlm_canonical_reassessment_20261008T043852Z_fac003a8` — `stability_config.json`, `stability_sample_identity.json`, `batch_stability_report.json`, `candidate_comparison.json`, `preflight_stop_report_v2.json` (기준 경로 종료 기록), `unit_test_report_v2.json`. 원래 `preflight_stop_report.json`은 보존했고, v2는 이전 행 수 필드의 잘못된 이름을 정정한다.
- **오프라인 합성 보조 함수 검사:** 최소 오른쪽 패딩/유효 길이, 패딩 값, 유효 프레임 마스크 평균, 길이 0 거부에서 PASS. Colab Drive 사본에 테스트 파일이 없어 pytest 모음은 `NOT RUN`이었다. pytest는 테스트 없음 (종료 코드 4)을 보고했으며 의존성은 설치하지 않았다.

### Stage 12-B — DPWavLM 경량 동결 음성 표현

**기록일:** 2026-10-09
**상태:** `RESOURCE/TIME-BUDGET INFEASIBLE`
**성능:** `PERFORMANCE NOT EVALUATED`.

**실험 목적·가설·근거:** WavLM Base+보다 작은 사전학습 음성 인코더로 원시 파형 기반 사전학습 표현이라는 Stage 12의 질문을 유지하면서 계산 가능성을 다시 확인한다. 약 23.59M 매개변수의 경량 인코더라면 전체 추출 부담을 줄일 수 있다는 가설을 검증하되, 매개변수 수가 작다는 사실만으로 전체 처리 경로가 빠르거나 정확도가 개선된다고 가정하지 않는다. 먼저 FP32 반복 재현성으로 안정성 기준을 고정하고, 이를 만족하는 기준 추론 경로의 실제 처리량으로 전체 규모 실행 여부를 판단한다. 분류기 성능 비교는 이 실행 가능성 기준을 통과했을 때의 후속 질문이다.

**정확한 후보 / 입력 계약:** DPWavLM, 체크포인트 `DPWavLM-sp0.75.pth`, 공개 모델의 예상 크기는 약 `23.59M` 매개변수다. 저장된 `model_load_preflight.json`의 실제 매개변수 수는 `23,586,325`로 예상 개수와 일치한다. 인코더는 동결·`eval()`·FP32이며 현재 분류 대상 발화의 파형만 사용한다. 공식 `extract_features(lengths)`의 마지막 은닉 표현에서 유효 프레임을 평균 풀링하고 float32 CPU 임베딩으로 저장한다. 전사 텍스트·다른 발화 정보·공식 검증은 사용하지 않았고 인코더 미세 조정도 하지 않았다. 저장 환경은 NVIDIA RTX PRO 6000 Blackwell Server Edition, CUDA `13.0`, Python `3.13.15`, Torch/Torchaudio `2.11.0+cu130`이다.

**데이터 / 고정 분할:** 대회 제공 Training의 기존 clean-59D 출처 증명과 고정 `call_id` 분할을 재사용했다. 아래 내부 검증은 Training에서 분리한 집합이며 공식 검증이 아니다. 안정성 검사와 처리량 측정에는 내부 학습 표본만 사용했다.

| 항목 | 고정 값 |
|---|---|
| 내부 학습 | `711,467` 발화 / `23,360` 통화 |
| 내부 검증 | `177,891` 발화 / `5,840` 통화 |
| 전체 대상 | `889,358` 발화 |
| 통화 중복 / 시드 | `0` / `42` |
| clean-59D 지문 | `49d65a35d1e98c280f98e45cbed864ef8bac36268ccffbe81aa0985923c90de3` |
| 데이터 동일성 | `871fb4d8c323579a4398adc4119c8ec3583b218545183b7578d40ef9c16e4b82` |
| 공식 검증 | 사용 안 함 |

**안정성 — 사전 기준 고정 → 길이 혼합 배치 FAIL(실패) → 기준 batch=1 PASS(통과):**

- **사전 기준:** DPWavLM FP32 batch=1 반복 재현성을 내부 학습 발화 64건에서 먼저 측정했고 최대 절대 차이는 `0`이었다. `max(20 × batch=1 repeatability max-abs delta, 1e-5 × median per-row embedding RMS)`로 절대 허용오차를 `9.277814626693727 × 10^-7` (약 **`9.28 × 10^-7`**)로 고정했다. 여러 발화를 묶은 배치의 결과를 보기 전에 고정했으며 Stage 12-A WavLM의 허용오차를 복사하지 않았다.
- **길이 혼합 batch=8:** 발화 128건을 단독 추론 기준과 비교한 결과 **128/128 모두 허용오차 초과**, 안정성 **FAIL(실패)**. 저장된 최대 절대 차이는 `1.4044610261917114`, 유효 프레임 수 불일치는 `0`이다. 프레임 수가 같아도 임베딩 동등성이 성립하지 않았다.
- **해석·경로 선택:** 공식 WavLM 파생 전처리부의 시간축 GroupNorm과 원시 파형 오른쪽 패딩의 상호작용이 확인됐다. 유효 프레임 풀링만으로 패딩이 전처리부 통계에 미친 영향을 제거할 수 없으므로 길이가 다른 발화를 묶는 배치 처리를 기준 추론 경로로 사용하지 않는다.
- **기준 batch=1:** 각 발화를 독립 추론하고 행 사이 길이를 맞추는 패딩을 사용하지 않는다. 인코더 최소 수용영역 미만 입력에 대한 결정적 오른쪽 패딩 규칙은 별도로 유지한다. 128건 안정성 재검증에서 **최대 차이 `0`, 프레임 불일치 `0`, 허용오차 초과 `0/128`**, **PASS(통과)**. 이 경로를 고정한 뒤 새로운 표본으로 처리량을 측정했다.

**새 표본 4,096행의 처리량 — 실제 측정:** 이전 WavLM 처리량 측정/사전 점검 행을 제외한 새로운 내부 학습 표본에서 기준 batch=1 전체 경로를 측정했다. `fresh_row_plan.json`은 기존 WavLM 처리량 측정 3,000건·진단 212건·안정성 검사 192건의 행 ID 제외를 기록한다. 기존 사전 점검의 정확한 ID가 저장되지 않은 부분은 알려진 재표본화 길이에 해당하는 상위 집합 704건을 보수적으로 제외했다. 따라서 미보존 ID를 직접 대조했다고 주장하지 않는다. 공식 검증은 사용하지 않았다.

| 실제 측정 항목 | DPWavLM 결과 |
|---|---:|
| 행 수 / 기준 배치 크기 | `4,096` / `1` |
| 추출 + 분할 파일 쓰기 경과 시간 | `1,058.89 sec` (정확한 값 `1058.8876171309985`) |
| 전체 경로 처리량 | `3.868 rows/sec` (정확한 값 `3.868210312155601`) |
| 저장된 분할 파일 수 / 파일당 행 수 | `16` / `256` |
| 분할 파일 재로드 / 동일성 검증 | **PASS(통과)**, 정확히 재로드한 `4,096` 행 |

경과 시간에는 WAV/ZIP 읽기·디코딩·잘라내기·재표본화, 인코더 순전파, 유효 프레임 풀링·CPU 전송, 임베딩 분할 파일 쓰기가 들어간다. 모델 로드/예열과 별도 분할 파일 재로드/검증 시간은 정상 처리 상태의 처리량에서 분리했다. 저장된 측정 상태는 `PASS_4096_END_TO_END_AND_RELOAD`이며 처리량·저장 데이터의 동일성 확인 결과다. 분류기 정확도 또는 전체 추출 완료를 뜻하지 않는다. 이 수치는 DPWavLM의 실제 처리량 측정이며 WavLM의 기존 `45.3700 h` 예상 시간을 대체하지 않는다.

**전체 889,358행의 실행 시간 / 저장 용량 — 예상값, 전체 실행은 미수행:** 새 표본 처리량을 행 수 / 분할 파일 수에 비례해 외삽했다. 다음 값은 저장된 예상값 JSON과 일치하며 상태는 `PROJECTED_NOT_EXECUTED`다.

| 예상 항목 | 예상 값 |
|---|---:|
| 전체 대상 행 수 | `889,358` |
| 추출 + 분할 파일 쓰기 | `229,914.6 sec` / 약 **`63.865 h`** |
| 모델 로드/예열 + 분할 파일 재로드/검증 포함, 분류기 제외 소계 | 약 **`63.868 h`** (약 **`63.87 h`**) |
| 분할 파일당 행 수 / 예상 파일 수 | `256` / `3,475` |
| 임베딩 NPZ 예상 바이트 수 | `3,031,119,663 bytes` |
| 임베딩 NPZ 예상 저장 용량 | 약 **`2.823 GiB`** |
| 분류기 학습/평가 시간 | **미측정·미실행**, 예상값 `null` |

분류기 시간은 측정하지 않았으므로 임의 추정치를 추가하지 않는다. NPZ 예상 용량은 임베딩 분할 파일 저장량이며 체크포인트, 전체 규모 목록과 최종 보고서의 추가 처리 비용은 포함하지 않는다. 시간 외삽은 이번 4,096건의 ZIP 접근·디코딩된 음성 캐시·Drive 입출력 조건에 근거하며 전체 실행의 캐시 적중/입출력 편차는 미측정이다.

**미실행 / 없는 성능 결과:** 전체 `889,358`행의 임베딩 추출, 분류기 학습, 내부 검증 성능 평가, 학습 정확도, 내부 검증 정확도, 혼동행렬, 화자 0/1 재현율, 높은 신뢰도의 오류(HCE), TabM과의 짝지은 비교, Tabular ResNet과의 짝지은 비교, 공식 검증, 인코더 미세 조정. 따라서 **`PERFORMANCE NOT EVALUATED`(성능 미평가)**를 유지한다. 안정성/처리량 측정 통과를 실제 데이터의 분류 성능 통과로 해석하지 않는다.

**최종 판단:** 안정성을 만족한 기준 batch=1 경로에서 실제 4,096행의 처리량을 측정했고, 전체 889,358행의 추출 예상 시간이 약 63.87시간으로 계산되었다. 따라서 현재 실행 환경과 실험 계획에서는 전체 추출을 진행하지 않고 Stage 12-B를 종료한다. 상태는 **`RESOURCE/TIME-BUDGET INFEASIBLE`(자원/시간 예산상 실행 불가) / `PERFORMANCE NOT EVALUATED`(성능 미평가)**다. 저장된 예상값 JSON의 `time_pass_fail_judgment`는 `null`이며, 이 종료 결정은 해당 예상값을 근거로 사용자가 확정한 판단을 문서화한 것이다.

**해석 한계:** DPWavLM 표현의 성능 열세를 확인한 실험이 아니다. `63.87 h > WavLM 45.37 h`만으로 DPWavLM 모델 자체가 WavLM보다 느리다고 일반화하지 않는다. 두 실험은 배치 처리·구현·입출력 계약이 다르므로 말할 수 있는 결론은 **현재 검증된 기준 처리 경로에서는 DPWavLM도 전체 규모 추출이 현실적이지 않았다**는 것이다.

**근거:** 근거 파일 목록의 확인된 Stage 12-B 실행, 허용오차·배치/기준 경로 안정성 보고서, `fresh_benchmark_4096_batch1_ed94bfca/benchmark_4096_report.json`, `full_889358_runtime_storage_projection_ee102fb6.json`. 이번 문서 갱신에서는 저장 보고서를 읽고 수치·범위를 대조했으며 실험·학습·추론·분할 파일 재로드를 재실행하지 않았다.

### Stage 12 전체 결론 / 다음 단계 — 2026-10-09

| 실험 | 검증된 실행 가능성 근거 | 최종 상태 / 성능 |
|---|---|---|
| Stage 12-A — WavLM Base+ | 기존 전체 경로 예상 시간 약 `45.3700 h`; 예열/캐시 최적화 결과는 통제된 처리량 측정이 아니며 안정성을 만족하는 최적화 기준 경로 미확보 | `RESOURCE/TIME-BUDGET INFEASIBLE` / `PERFORMANCE NOT EVALUATED` |
| Stage 12-B — DPWavLM 경량 모델 시도 | 안정성을 만족하는 batch=1 경로 PASS(통과), 실제 4,096행 처리량 측정, 전체 예상 시간 약 `63.87 h` (분류기 제외) | `RESOURCE/TIME-BUDGET INFEASIBLE` / `PERFORMANCE NOT EVALUATED` |

원시 파형 기반 사전학습 음성 표현 계열은 **성능 실패가 아니라 계산 가능성 문제로 현재 개인 로드맵에서 종료**한다. Stage 12 전체 상태는 **`CLOSED — resource/time feasibility`(자원/시간 실행 가능성 문제로 종료)**, 성능은 **`NOT EVALUATED`(미평가)**다. 최종 모델은 **`NOT SELECTED`(미선정)**, 현재 비교 기준은 **TabM + Tabular ResNet**을 유지한다. 다음은 **Stage 13 — clean 59D에서 아직 시험하지 않은 모델 계열 재검토**이며 모델 후보는 아직 확정하지 않았다. Stage 13의 실험이나 결과를 이 기록에서 미리 생성하지 않는다.

### Stage 13 — Untested Model Family First-Screen — 2026-10-09

**상태:** `COMPLETED — 6/6 candidate fits, prediction artifacts, metrics, and checkpoint reload audits passed`. 이는 실행·저장 검증 완료를 뜻하며 최종 모델 선정을 뜻하지 않는다. 실행은 해당 사용자 계정의 저장 Colab Notebook과 Drive run output에서 확인했고, Colab Notebook header는 `모든 변경사항이 저장됨`을 표시했다. run ID는 `20261009T051303Z_7d149a2b`다. 아래 Stage 12 결론과 §4.12/§8의 “Stage 13 미시작” 문구는 Stage 13 run보다 앞선 상태 기록으로 그대로 보존하며, 현재 상태는 이 항목과 §0/§10을 따른다.

#### 계획 변경과 사전 고정

로드맵 v2의 Stage 13은 결과 확인 전에 `1–2개` 후보를 고정하도록 적혀 있다 (`reference/mission2_yeonwoo_experiment_roadmap.md`, Stage 13). 이번 6개 후보는 사용자의 명시적 요청으로 확장됐다. Colab Checkpoint D에는 EBM, Spline-GAM, RFF + Logistic SGD, DCNv2, TabNet, GANDALF의 6개 설정이 표시됐고, 그 출력은 EBM 첫 학습 결과보다 앞에 있다. 저장된 Notebook도 각 후보별 one-shot 설정과 결과 기반 변경 금지를 설명한다. 따라서 **여섯 후보를 결과 전에 고정하고 실행한 점은 확인**했다. 다만 변경 전의 별도 1–2개 shortlist/config 문서는 현재 repo/Notebook output에서 찾지 못했다. 기존 roadmap 제한을 이번에 넓힌 주체·범위는 사용자 요청이며 성능을 본 뒤 후보 수를 늘린 것으로 기록하지 않는다. 다만 동일 internal Validation에서 검토한 후보 수가 늘어났으므로 1–2개 first-screen보다 validation multiplicity가 커졌다. 후보·설정 사전 고정은 결과 기반 tuning을 막지만, 이 다중 비교의 선택 편향 가능성까지 제거하지는 않는다.

#### 공통 계약 및 검증

- 기존 clean 59D cache 재사용: provenance fingerprint `49d65a35d1e98c280f98e45cbed864ef8bac36268ccffbe81aa0985923c90de3`; 기존 feature를 재추출하거나 waveform을 다시 처리하지 않았다.
- fixed internal split: train `711,467`행 / validation `177,891`행, seed `42`, `call_id` overlap `0`, training-only scaler, threshold `0.5`. Notebook output에서 Official Validation 사용 `False`를 확인했다.
- frozen TabM/Tabular ResNet은 기존 run `20261006T031510Z_a9a01a2d`의 저장 prediction/metrics를 재사용했다. Stage 13에서 재학습하지 않았다.
- duration/overlap은 `record_key`로 validation prediction과 tracking을 일대일 결합한 뒤 `call_id`·label을 대조한 분석 annotation이다. tracking `889,358`행, duration 범위 `0.009–24.172 s`, overlap `389,529`행이다. 어떤 후보의 feature matrix에도 넣지 않았다.
- 저장 artifact audit은 여섯 prediction 각각에 대해 validation 행 수 `177,891`, prediction에서 재계산한 Accuracy, config ID, checkpoint reload 상태를 기록한다. Notebook output은 여섯 prediction이 fixed validation row identity에 재정렬됐고 지표가 저장 예측에서 재계산됐다고 보고한다. 전체 후보 reload는 `EXACT_PASS`다.

#### 8개 모델 통합 성능 비교

Accuracy가 공식 1차 지표다. Train−Val gap은 같은 후보의 training Accuracy에서 internal-validation Accuracy를 뺀 값이며 아래는 percentage point다. 혼동행렬은 `[[TN, FP], [FN, TP]]`, HCE는 validation에서 confidence `≥0.90`인 오분류 수다.

| 모델 | Accuracy | Train−Val gap | Recall 0 / 1 | 혼동행렬 | HCE ≥0.90 | runtime (s) | reload |
|---|---:|---:|---|---|---:|---:|---|
| TabM (frozen) | **88.3980%** | +0.384 pp | 0.834554 / 0.929566 | `[[71230,14121],[6518,86022]]` | 1,629 | 2,825.3 | `EXACT_PASS` |
| Tabular ResNet (frozen) | 88.3755% | +1.642 pp | **0.844700 / 0.919775** | `[[72096,13255],[7424,85116]]` | 2,693 | 754.9 | `EXACT_PASS` |
| GANDALF | 88.1748% | +0.210 pp | 0.840142 / 0.920121 | `[[71707,13644],[7392,85148]]` | 2,204 | ~483.2 | `EXACT_PASS` |
| DCNv2 | 87.7830% | +0.169 pp | 0.828461 / 0.923363 | `[[70710,14641],[7092,85448]]` | 2,240 | 13.7 | `EXACT_PASS` |
| TabNet | 87.7509% | +0.075 pp | 0.825017 / 0.925924 | `[[70416,14935],[6855,85685]]` | 2,491 | ~330.4 | `EXACT_PASS` |
| Spline-GAM | 85.1836% | −0.149 pp | 0.817530 / 0.883477 | `[[69777,15574],[10783,81757]]` | 3,090 | 303.8 | `EXACT_PASS` |
| EBM | 83.9604% | −0.164 pp | 0.791051 / 0.884385 | `[[67517,17834],[10699,81841]]` | 1,961 | 151.6 | `EXACT_PASS` |
| RFF + Logistic SGD | 83.1453% | −0.072 pp | 0.784502 / 0.874757 | `[[66958,18393],[11590,80950]]` | 2,915 | 34.8 | `EXACT_PASS` |

| 후보 | 무엇을 하는가 / 왜 비교했나 | 관찰된 결과 |
|---|---|---|
| EBM | feature별 비선형 효과를 더하는 boosting 모델이다. 상호작용 없이 각 feature의 단독 효과만으로 reference 오류를 얼마나 다르게 고치는지 봤다. | Accuracy `83.9604%`; TabM 대비 순정답 `−7,894`; 두 reference 공통 오류 `2,892`개 교정. |
| Spline-GAM | Training quantile에서 feature별 cubic spline 곡선을 만들고 이를 logistic model로 더한다. EBM과 다른 부드러운 additive 함수를 시험했다. | Accuracy `85.1836%`; 공통 오류 `3,005`개를 가장 많이 교정했지만 reference 대비 순정답은 `−5,718`. |
| RFF + Logistic SGD | Random Fourier Features로 RBF 유사도를 근사한 뒤 batch SGD로 logistic model을 학습한다. 큰 표본에서 비선형 근사 학습의 빠른 기준선을 봤다. | 측정 runtime `34.8 s`로 짧았지만 Accuracy `83.1453%`, TabM 대비 순정답 `−9,344`; 공통 오류 `2,690`개 교정. |
| DCNv2 | Cross layer와 MLP를 함께 써 feature 조합을 직접 학습한다. 명시적인 교차 학습이 reference와 다른 오류를 줄이는지 확인했다. | Accuracy `87.7830%`; TabM 대비 순정답 `−1,094`; 공통 오류 `1,098`개 교정. |
| TabNet | 여러 decision step에서 feature attention mask를 선택한다. 단계적 feature 선택이 다른 오류 패턴을 만드는지 비교했다. | Accuracy `87.7509%`; speaker 1 Recall `0.925924`; TabM 대비 순정답 `−1,151`; 공통 오류 `1,028`개 교정. 예측 일치 미해결은 아래에 별도 기록했다. |
| GANDALF | Gate로 feature 조합을 조절하는 tabular model이다. 학습된 gating이 59D feature 관계를 다르게 사용할 수 있는지 봤다. | Accuracy `88.1748%`로 신규 후보 중 가장 높고 TabM에 가장 가까웠지만, TabM 대비 순정답은 `−397`; 공통 오류 `795`개 교정. |

**표 해석:** 신규 후보 중 GANDALF가 가장 높은 Accuracy `88.1748%`를 기록했지만 TabM보다 `0.2232 pp`, ResNet보다 `0.2007 pp` 낮다. DCNv2와 TabNet은 `87.8%` 근처였으며 EBM/Spline-GAM/RFF는 더 낮았다. 두 reference 중 TabM은 Accuracy와 HCE가 더 좋았고, ResNet은 speaker 0 Recall이 더 높았다. 후보별 class recall, FP/FN 및 HCE가 달라 Accuracy 단독 순위가 오류의 종류까지 설명하지는 않는다. 낮은 Train−Val gap은 training Accuracy도 낮은 후보에서 관찰된 값이며 그 자체를 일반화 우위나 구조적 원인으로 해석할 수 없다.

#### Frozen reference와 paired error 분석

각 쌍의 `reference만 정답 / 후보만 정답`은 같은 177,891개 validation 행을 직접 비교한 건수다. 순교정은 후보만 정답에서 reference만 정답을 뺀 값이다. 마지막 열은 TabM과 ResNet이 **둘 다 틀린 17,236행** 중 그 후보가 정답으로 바꾼 행이다.

| 후보 | TabM만 정답 / 후보만 정답 | 순교정 vs TabM | ResNet만 정답 / 후보만 정답 | 순교정 vs ResNet | 공통 reference 오류 교정 |
|---|---:|---:|---:|---:|---:|
| EBM | 11,582 / 3,688 | −7,894 | 13,328 / 5,474 | −7,854 | 2,892 |
| Spline-GAM | 9,718 / 4,000 | −5,718 | 11,094 / 5,416 | −5,678 | **3,005** |
| RFF + Logistic SGD | 13,123 / 3,779 | −9,344 | 14,288 / 4,984 | −9,304 | 2,690 |
| DCNv2 | 3,250 / 2,156 | −1,094 | 4,602 / 3,548 | −1,054 | 1,098 |
| TabNet | 3,101 / 1,950 | −1,151 | 4,743 / 3,632 | −1,111 | 1,028 |
| GANDALF | 2,369 / 1,972 | −397 | 3,595 / 3,238 | −357 | 795 |

Spline-GAM은 공통 reference 오류를 여섯 후보 중 가장 많이 고쳤지만 reference가 맞힌 행을 더 많이 놓쳐 전체 정답 수는 줄었다. GANDALF는 두 reference와의 정오 전환 수가 비교적 작아 가장 가까운 결과였지만 순정답 수는 두 reference 모두보다 적다. 따라서 공통 residual의 일부 교정은 확인됐지만, 이 first-screen에서 reference를 대체할 성능 개선은 관찰되지 않았다.

8개 전부 틀린 행은 `11,534`, 적어도 한 모델이 틀린 review 대상은 `44,229`다. 다른 7개 모델이 틀리고 해당 모델만 맞힌 고유 정답은 Tabular ResNet `872`, RFF `750`, Spline-GAM `731`, EBM `413`, DCNv2 `192`, TabNet `169`, GANDALF `155`, TabM `54`행으로 합계 `3,336`이다. 각 모델이 단독으로 맞힌 row의 실제 예와 8개 공통 오류는 아래 review artifact에 `record_key`, `call_id`, label, prediction/confidence 및 사람이 청취 검토할 `audio_path`와 연결해 보존했다.

#### Duration / overlap / class 취약 구간

각 duration/overlap 값은 정확한 validation row identity를 확인한 사후 분석이다. 모든 모델에서 표에 표시한 duration 구간 중 `0.5–1 s`가 가장 낮았고, overlap 구간 Accuracy가 no-overlap보다 낮았다.

| 모델 | Duration 0.5–1 s Accuracy | No-overlap Accuracy | Overlap Accuracy | no-overlap 대비 overlap 하락 |
|---|---:|---:|---:|---:|
| TabM | 84.1617% | 91.4349% | 84.5599% | 6.875 pp |
| Tabular ResNet | 84.1570% | 91.3262% | 84.6464% | 6.680 pp |
| GANDALF | 83.8607% | 91.1711% | 84.3881% | 6.783 pp |
| DCNv2 | 83.4117% | 90.8831% | 83.8651% | 7.018 pp |
| TabNet | 83.4279% | 90.8086% | 83.8867% | 6.922 pp |
| Spline-GAM | 80.3352% | 88.4755% | 81.0234% | 7.452 pp |
| EBM | 78.4856% | 87.1634% | 79.9124% | 7.251 pp |
| RFF + Logistic SGD | 77.4925% | 86.3428% | 79.1044% | 7.238 pp |

speaker 0 Recall은 여덟 모델 모두 speaker 1 Recall보다 낮다. ResNet은 overall speaker 0 Recall `0.844700`으로 가장 높고, TabM은 speaker 1 Recall `0.929566`으로 가장 높다. 새 후보에서는 TabNet speaker 1 Recall `0.925924`, GANDALF speaker 0 Recall `0.840142`가 상대적으로 reference에 가까웠다. 짧은 발화·overlap은 특정 feature 원인이나 인과관계를 증명하지 않으며, 표본 조건이 다를 수 있는 slice 정확도는 `slice_comparison.csv`와 `n`을 함께 봐야 한다.

#### Confidence / high-confidence error 분석

HCE는 모든 validation 행 중 confidence `≥0.90`인 오분류 수다. 아래 `n`과 slice Accuracy는 **각 모델 자신의 확률**로 만든 `0.9–1.0` 집단이므로 모델 간 동일 sample 집합 비교가 아니다.

| 모델 | confidence 0.9–1.0 n | 그 구간 Accuracy | HCE | 해당 구간 중 HCE 비율 |
|---|---:|---:|---:|---:|
| TabM | 111,425 | 98.5380% | 1,629 | 1.4620% |
| Tabular ResNet | 122,429 | 97.8004% | 2,693 | 2.1996% |
| GANDALF | 116,131 | 98.1021% | 2,204 | 1.8979% |
| DCNv2 | 114,100 | 98.0368% | 2,240 | 1.9632% |
| TabNet | 115,626 | 97.8456% | 2,491 | 2.1544% |
| Spline-GAM | 94,372 | 96.7257% | 3,090 | 3.2743% |
| EBM | 70,646 | 97.2242% | 1,961 | 2.7758% |
| RFF + Logistic SGD | 81,352 | 96.4168% | 2,915 | 3.5832% |

TabM은 총 HCE 수와 고신뢰도 구간 내 HCE 비율이 가장 낮았다. EBM은 HCE 수가 ResNet보다 적지만 해당 모델의 고신뢰도 표본 수 자체가 작고 비율은 높다. 그러므로 HCE 절대 건수만으로 confidence 품질을 순위화하지 않으며 집단 크기와 정확도를 같이 본다.

#### 실행 복구와 미해결 문제

- **EBM:** 모델 checkpoint, prediction, metrics가 저장된 뒤 `status.json` 생성 과정이 `FileExistsError`로 실패 상태가 됐다. 저장 파일과 지표를 확인해 최종 audit에 `SUCCESS`로 반영했다. 원래 failure 기록은 보존됐다.
- **DCNv2:** 첫 시도는 생성자에서 문자열을 module처럼 처리해 `'str' object has no attribute '_apply'` 오류로 끝났다. frozen config ID를 바꾸지 않고 구현 경로를 복구해 재학습·평가·reload가 통과했다.
- **GANDALF:** 첫 시도는 설치된 `TrainerConfig`가 `enable_progress_bar` 인자를 받지 않아 실패했다. 같은 고정 설정으로 재시도해 checkpoint를 만들었고, 뒤이은 저장 지표 구성 오류는 checkpoint 재학습 없이 복구했다. 최종 reload audit은 통과했다.
- **TabNet — 미해결:** 첫 번째 in-memory prediction과 reload prediction이 달라 최초 reload gate가 실패했다. `attempt_02`에서도 `atol=1e-7` 조건의 비교가 실패했다. 별도 저장 checkpoint recovery에서는 같은 checkpoint를 두 번 reload한 확률이 완전히 같아 `max_abs_probability_delta=0.0`, `EXACT_PASS`가 됐다. 이는 **저장 checkpoint의 반복 reload 일치**를 검증한다. 최초 in-memory 예측과 저장/reload 결과의 불일치 원인은 해결되지 않았고, 최초 vector가 현재 prediction artifact와 동일하다는 독립 증거도 없다. 최종 표는 row-aligned 저장 prediction과 checkpoint-reload 검증 결과를 사용한다.
- **runtime:** EBM, Spline-GAM, RFF, DCNv2의 runtime은 저장 측정값이다. TabNet의 약 `330.4 s`와 GANDALF의 약 `483.2 s`는 fit 시간이 각각 약 `300 s`, `452 s`로 사후 복원된 근사치이며 prediction+reload 구간만 재측정됐다. 두 모델 runtime은 다른 네 후보와 정밀도가 다르다.
- **analysis repair:** 비교 artifact 최종화 과정의 pandas categorical 처리 오류는 분석 helper 복구 뒤 저장 prediction에서 다시 계산했다. 모델 fit은 다시 실행하지 않았다. `analysis_patch_recovery.json`과 final audit을 함께 보존했다.

#### 가설·근거 결과 정리 상태 점검

기존 §1–§8에는 E00–E05의 질문과 feature 선택 근거, M00–M04의 family별 질문, AF01–AF12의 사전 목적·반증 가능성, 기준 오류 관찰에서 잔여 feature 가설로 이어지는 흐름, Stage 4/6의 비교 질문과 판정 근거가 정리돼 있다. Stage 10–12의 결정·실행 가능성 가설·실측/미실행 구분은 이 추가 기록에서도 수정하지 않았다. 이번 Stage 13의 모델별 목적·고정 설정 전 결과 및 관찰/가능한 이유 구분은 Colab Checkpoint E–M에 있고, 통합 수치와 artifact audit은 §4.13과 위 Evidence Map에 추가했다.

따라서 가설과 결과의 **기록 구조는 대체로 갖춰져 있지만, 모든 과거 결과가 동등한 재현 증거로 완결된 것은 아니다.** 기존 §9.3의 한계와 이번 Stage 13 unresolved를 그대로 유지한다. 대표적으로 E02–E04는 저장된 rendered output이 없고 committed Markdown만 있다; 기준 오류 분석 row-level ledger는 파일로 저장되지 않았다; AF04의 promotion 과정은 일부 output이 빠져 partially verified이며 AF05 standalone score, M04-60 fit provenance, 최초 AF09 `0.868970` 원인은 unresolved다. historical raw cache의 원본 content identity도 완전히 입증되지 않았다. 이 항목들은 관측 결과와 미해결 원인을 구분해 기록하며 가설을 확인된 원인으로 승격하지 않는다.

**종합 관찰과 범위:** 여섯 신규 후보 중 GANDALF가 두 frozen reference에 가장 가까웠지만 두 reference보다 Accuracy가 낮았고, 다른 후보도 overall 정답 수를 늘리지 못했다. 각 후보가 공통 reference residual 및 전체 8개 모델의 일부 오류를 고친 사례는 확인됐지만, paired regression이 더 많아 reference 대체 개선은 확인되지 않았다. 구조와 전체 training recipe가 함께 달라졌으므로 이 결과는 동일 clean-59D 계약 아래의 first-screen pipeline 비교다. 특정 구조만의 인과 효과, 59D 성능 ceiling 또는 Official Validation 성능을 입증하지 않는다. 이 Stage 13 결과 작성 당시에는 Stage 14·ensemble·final selection을 수행하지 않은 상태였다. 이후 Stage 14 실행과 현재 후보 선택은 §4.14에 기록한다.

### Stage 14 — F01+F02 통합 입력과 3모델 앙상블 채택 (2026-10-11 실행, 2026-10-12 사용자 결정)

**상태와 결정:** Drive의 run `20261010T012333Z_8de41728_stage14_joint_f01_f02_attempt_02`에서 세 멤버와 네 고정 가중치 앙상블의 artifact 검증이 완료됐고, 저장된 보고서의 상태는 `ENSEMBLES_COMPLETE`다. 사용자는 내부 비교 결과를 검토한 뒤 세 멤버를 각각 1/3로 평균하는 `E_TRG`를 채택했다. Stage 15 하이퍼파라미터 튜닝과 Official Validation은 이번 단계에서 실행하지 않았다.

**비교 계약:** 세 멤버 모두 Clean 59D + F01 80D + F02 10D = 149D 입력을 사용했다. 기존 고정 Internal Split(Train 711,467행/23,360통화, Validation 177,891행/5,840통화), seed 42, `call_id` 그룹 분리(교집합 0), Train-only `StandardScaler`, threshold 0.5를 유지했다. Official Validation은 이 run에서 사용하지 않았다. Stage14 Colab에서 관련 검증 테스트 `8 passed in 4.06s`였고, 세 체크포인트 reload prediction 일치 audit도 통과했다. 저장 보고서: Drive `STAGE14_REPORT_KO.md`; 결과 디렉터리는 위 run 경로다.

| 후보 | Validation Accuracy | Recall 0 | Recall 1 | HCE |
|---|---:|---:|---:|---:|
| TabM + F01 + F02 | `0.8864304546` | `0.8391465829` | `0.9300410633` | 1,526 |
| ResNet + F01 + F02 | `0.8854916775` | `0.8432707291` | `0.9244326778` | 2,305 |
| GANDALF + F01 + F02 | `0.8858458269` | `0.8391348666` | `0.9289280311` | 2,949 |
| E_TR (TabM + ResNet) | `0.8872174534` | `0.8426849129` | `0.9282904690` | 1,664 |
| E_TG (TabM + GANDALF) | `0.8881506091` | `0.8411500744` | **`0.9314998919`** | 1,841 |
| E_RG (ResNet + GANDALF) | `0.8881674733` | **`0.8433644597`** | `0.9294899503` | 2,202 |
| **E_TRG (동일가중치 1/3씩)** | **`0.8884822729`** | `0.8434464740` | `0.9300194510` | 1,797 |

E_TRG는 네 앙상블 중 Accuracy가 가장 높고, 가장 높은 단일 멤버 TabM 대비 `+0.0020518182`(약 `+0.2052` percentage point, 순 정답 약 365건)였다. 최고 2모델 후보 E_RG보다 차이는 `+0.0003147995`(약 `+0.0315` percentage point, 56건)에 그친다. E_TRG의 speaker 0 Recall은 `0.8434465`, speaker 1 Recall은 `0.9300195`다. HCE 1,797건은 TabM 단독 1,526건보다 271건 많으므로 Accuracy 최고가 모든 오류 지표의 우위를 뜻하지 않는다.

E_TRG와 각 멤버의 paired 결과는 TabM 대비 교정 1,974/회귀 1,609(순 +365, 변경 3,583행), ResNet 대비 교정 2,150/회귀 1,618(순 +532, 변경 3,768행), GANDALF 대비 교정 2,070/회귀 1,601(순 +469, 변경 3,671행)이다. 멤버 간 확률 상관은 약 `0.978`–`0.984`로 높았다. `call_id` 5,840개를 단위로 한 2,000회 bootstrap에서 E_TRG Accuracy의 95% 구간은 `[0.8865753, 0.8905042]`였다. 이 구간은 후보 간 paired 차이의 신뢰구간이 아니다.

취약 slice로 speaker 0의 0.5–1초 구간 Accuracy는 `0.767060`(19,593행), 같은 길이의 overlap 구간은 `0.679037`(7,976행), speaker 0의 1–2초 overlap은 `0.784494`(13,517행)였다. 채택은 전체 Accuracy point estimate 우선의 선택이며, 해당 구간 문제가 해소됐다는 뜻은 아니다.

**59D와의 해석 경계:** Stage 10/13의 59D 수치는 별도 학습 모델의 역사적 기준으로 계속 보존한다. E_TRG는 149D 입력 세 모델을 평균한 후보이므로, 그 59D 단일 모델 수치와의 차이를 F01/F02만의 순수 증분 효과로 해석하지 않는다. Stage14의 네 앙상블 및 세 멤버 선택은 같은 Internal Validation에서 이뤄졌으며, Stages 10–13을 포함한 과거 실험에서도 해당 split이 반복 사용됐다. 따라서 선택 편향 가능성과 낙관적 성능 추정 위험이 있다. Official Validation은 Stage14에서 열지 않았고, 본 기록은 Stage15/Official 결과를 주장하지 않는다.

**실행 설정·비용 근거:** Stage14 run 설정에서 TabM은 LR `0.002`, 최대 100 epoch/patience 10/batch 256, weight decay `0.0003`, 3 blocks/width 512/dropout `0.1` (fit `1,969.63s`); ResNet은 LR `0.0003`, 최대 100 epoch/patience 10/batch 256, weight decay `1e-5`, 2 blocks/width 192/dropout1 `0.15`/dropout2 `0` (fit `533.52s`); GANDALF는 LR `0.001`, 최대 20 epoch/patience 5/batch 4,096, 6 stages/dropout `0.1`/sparsity `0.3` (fit `341.39s`)였다. 개별 checkpoint, scaler, prediction, metrics, 설정과 해시는 Drive의 run 내 `members/` 및 `ensembles/`에 보존돼 있다. Git에는 Stage14 notebook과 실행에 필요한 source/test 및 transfer-identity/fixed-split dependency 변경을 push했고 데이터와 학습 artifact는 포함하지 않았다.

#### Stage 14 근거 및 후속 경계

- 구현/실행: `notebooks/mission2_stage14_joint_f01_f02_ensemble_colab.ipynb`; Colab L4/CUDA 실행 기록과 Drive 저장 산출물.
- 저장 보고서: `STAGE14_REPORT_KO.md` (Drive); Stage14 run 경로는 위와 같다.
- 코드 push: remote branch `codex/mission2-stage14-ensemble`, commit `41c221f16` (Stage14 구현).
- 사용자 선택 기록: E_TRG 3-member equal-weight를 Stage14 채택 후보로 선택. Official Validation 또는 Stage15 튜닝은 실행하지 않았다. 현재 채택은 내부 검증 재사용에 따른 불확실성을 가진다.

### Stage 15 — Final Full-Training Refit + Official Validation (2026-10-12 구현, 2026-10-13 실행 완료)

**결정과 고정 계약:** 추가 튜닝 없이 Stage 14의 TabM + Tabular ResNet + GANDALF를 유지하고, Clean 59D + F01 80D + F02 10D의 149D 입력, seed 42, E_TRG 확률 단순 평균 1:1:1, threshold 0.5를 고정한다. Training 전체 889,358행/29,200통화에서 scaler를 fit하고, Stage 14 저장 근거에서 정한 고정 학습 길이(TabM 68 epoch, ResNet 16 epoch, GANDALF 17 epoch)를 사용하도록 구현했다. Official Validation은 학습·scaler·checkpoint 선택에 사용하지 않는 최종 평가 split이다.

**구현 및 사전 검증:** 새 실행 모듈 `src/dcc_data/mission2_final_fulltrain_refit.py`, 계약 테스트 `tests/test_mission2_final_fulltrain_refit.py`, Colab Notebook `notebooks/mission2_final_fulltrain_official_validation.ipynb`, source-only bundle 생성 스크립트 `scripts/package_mission2_final_refit_code_bundle.py`를 추가했다. bundle SHA-256은 `e6e25dba621d43f76e5ac6e91c27f62338a7256342cddde34ac279bd53af7e53`이며 원본 데이터는 포함하지 않는다. 계약 테스트 9개와 Notebook JSON/Python 구문 검사는 통과했다. 로컬 Validation 원천으로 만든 text-free 추적 manifest는 111,947행/3,640통화이며 기존 manifest와 row identity, label, timestamp, crop 경계가 모두 일치했다. 이 로컬 manifest 정합성은 Colab 최종 모델 평가 결과가 아니다.

**실행 및 복구 경과:** 첫 Colab Drive mount 시도는 `credential propagation was unsuccessful`로 preflight 전에 중단됐다. 이후 본 실행에서 Official Validation 3,640통화의 E05·AF·F01/F02 추출이 각각 `3,640/3,640`까지 완료됐다. 첫 refit run은 TabM 68 epoch와 ResNet 16 epoch를 완료했으나 GANDALF가 17 대신 18 epoch로 기록되어 고정 epoch 계약에 따라 `FAILED_STOP`됐다. 복구 run `mission2_final_fulltrain_official_validation_20261012T151108Z_a1606771`은 기존 Validation feature cache(SHA-256 `6aaefb7c8bc5548fac07e8ae95a3ecfe5698158166dcebb34368cd374b412666`)를 재사용하고, TabM/ResNet checkpoint를 복사 후 재검증했으며 GANDALF만 17 epoch 다시 학습했다. 복구 preflight 계약 테스트는 `10 passed in 1.23s`였다. 따라서 복구 때 feature 추출과 앞의 두 모델 학습은 반복되지 않았다. TabM·ResNet·GANDALF fit 시간은 각각 `2,119.841s`, `389.427s`, `357.305s`다.

**Official Validation 결과:** 최종 run의 manifest는 `source_split=Validation`, 111,947행/3,640통화이며 Training과 `call_id` 및 record key 교집합은 모두 0이다. 149D feature schema fingerprint는 `0e9834296216cf39b5715e9e80df20b56fba2ee70d2ffa84dfed90179ace1626`이다. Official Validation은 scaler fit, gradient, loss, early stopping, checkpoint 선택에 사용되지 않았고, 고정 threshold `0.5`에서 저장된 예측을 평가했다. 혼동행렬 순서는 `TN/FP/FN/TP`다.

| 모델 | Accuracy | Balanced Accuracy | Recall 0 | Recall 1 | HCE | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|
| TabM | `0.880560` | `0.878692` | `0.829585` | `0.927799` | 1,189 | 44,669 / 9,176 / 4,195 / 53,907 |
| Tabular ResNet | `0.880854` | `0.879486` | `0.843514` | `0.915459` | 1,780 | 45,419 / 8,426 / 4,912 / 53,190 |
| GANDALF | `0.881864` | `0.880256` | `0.837979` | `0.922533` | 1,938 | 45,121 / 8,724 / 4,501 / 53,601 |
| **E_TRG (각 1/3 확률 평균)** | **`0.884919`** | **`0.883339`** | **`0.841787`** | **`0.924891`** | **1,301** | **45,326 / 8,519 / 4,364 / 53,738** |

HCE는 각 모델의 confidence가 `0.90` 이상인 오분류 건수다.

E_TRG는 세 단독 멤버 중 가장 높은 Accuracy인 GANDALF보다 0.003055(0.3055 percentage point) 높았다. 단, Official 결과는 이 고정 후보의 최종 평가이며 가중치·threshold를 탐색하지 않았다. E_TRG와 각 멤버의 paired 비교에서 (앙상블만 맞음/멤버만 맞음/공통 오답/순 교정/예측 변경 행)은 TabM `1,785/1,297/11,586/+488/3,082`, ResNet `1,483/1,028/11,855/+455/2,511`, GANDALF `1,532/1,190/11,693/+342/2,722`였다.

| Official Validation slice | 행 수 | TabM | ResNet | GANDALF | E_TRG |
|---|---:|---:|---:|---:|---:|
| 전체 | 111,947 | `0.880560` | `0.880854` | `0.881864` | **`0.884919`** |
| overlap | 48,863 | `0.841762` | `0.842805` | `0.844136` | **`0.847390`** |
| no-overlap | 63,084 | `0.910611` | `0.910326` | `0.911087` | **`0.913988`** |
| 발화 길이 0.5–1초 | 27,392 | `0.837288` | `0.837215` | `0.838347` | **`0.843056`** |

**최종 artifact audit 및 경계:** run audit 상태는 `FINAL_VERIFIED`; 세 checkpoint reload은 `EXACT_PASS`, 저장 prediction alignment와 재계산 metric은 `PASS`, 혼동행렬 count 합도 111,947행과 일치한다. run artifact manifest/SHA 점검은 `FINAL_RUN_ARTIFACTS_AND_SHA256: PASS`였다. 결과와 증거 파일은 [Drive run 폴더](https://drive.google.com/drive/folders/1ITmKuvHSx6LcIdm31HJ2T3GoYAJkOHqf)의 `final_audit.json`, `official_validation_comparison.csv`, `official_validation_paired_errors.csv`, `official_validation_duration_overlap_slices.csv`, `ensemble_predictions.parquet`, `artifact_manifest.json`에 있다. 독립적인 Drive 재다운로드 후 SHA 검증은 아직 남아 있다.

복구 run provenance의 실행 bundle SHA-256은 `e8496dec85e2528d6e63f011740f9a3417f5fbea7c936e6d185303fb0e7dd5b0`, 실행 module SHA-256은 `c76cb7524e39ef939238fde87f2528b4ea0c72d724550269d34b7f1859404fd7`이다. 현재 열린 Colab Notebook에는 원격/다른 탭 업데이트로 자동 저장할 수 없다는 경고가 표시된다. 화면의 셀에 입력된 bundle SHA(`02dc4983998ea91f46395874f68d0093feb8caa74652a623e569160b02d3b61e`)는 실행 provenance의 bundle SHA와 다르므로, 현재 Notebook 저장 상태가 이 실행본과 동일한지는 확인되지 않았다. 본 결과의 기준은 run별 Drive audit와 manifest다.

공용 `src/dcc_inference/mission2.py`와 `scripts/mission2_inference.py`는 수정하지 않았다. 최종 audit도 `SUBMISSION INFERENCE NOT READY`로 판정했다. 149D Stage 15 모델을 공용 42D inference 경로에 통합하고 공식 sample-output parity를 확인하는 일, 그리고 artifact 재다운로드 SHA 확인은 미완료다.

#### Stage 15 후속 — 저장 예측 기반 E_TRG 오분류 분석

분석 대상은 복구 run `mission2_final_fulltrain_official_validation_20261012T151108Z_a1606771`의 `FINAL_VERIFIED` Official Validation 예측과 text-free manifest다. 111,947행의 저장된 예측을 고정 threshold `0.5`로 재확인했으며, 재학습·재추출·threshold/모델 튜닝·음성 청취·transcript 사용은 하지 않았다. 이 내용은 기존 Official Validation 성능 결과와 별개의 기술적 오류 분석이다.

E_TRG는 12,883건(11.508%)을 오분류했다: FP 8,519, FN 4,364, HCE(confidence ≥ `0.90`) 1,301건. 혼동행렬은 `[[45,326, 8,519], [4,364, 53,738]]`(행=true 0/1, 열=prediction 0/1)이며, speaker 0/1 Recall은 각각 `0.841787`/`0.924891`이다.

| 구간 | 행 수 | 오류 | 오류율 |
|---|---:|---:|---:|
| overlap | 48,863 | 7,457 | 15.2610% |
| no-overlap | 63,084 | 5,426 | 8.6012% |
| 발화 길이 `<0.5초` | 8,954 | 986 | 11.0118% |
| 발화 길이 `0.5–1초` | 27,392 | 4,299 | 15.6944% |
| 발화 길이 `1–2초` | 36,858 | 5,101 | 13.8396% |
| 발화 길이 `2–5초` | 31,637 | 2,279 | 7.2036% |
| 발화 길이 `≥5초` | 7,106 | 218 | 3.0678% |

confidence `[0.50,0.60)` 구간은 12,253행 중 5,507건(44.9441%)이 오답이었고, confidence `≥0.90` 구간은 75,016행 중 1,301건(1.7343%)이 오답이었다. 세 멤버 투표는 FP 중 7,040/8,519건에서 모두 speaker 1, FN 중 2,871/4,364건에서 모두 speaker 0이었다. 오류행에서 멤버 예측 불일치율은 FP `17.3612%`, FN `34.2117%`였다. 모두 고정 예측을 설명하는 기술 통계이며 원인이나 개선 효과를 입증하지 않는다.

집계 CSV/JSON과 오류행 CSV/Parquet는 [Drive misclassification_analysis 폴더](https://drive.google.com/drive/folders/1Gz53pNofawO_1ImcFiwPe9_OhbqBWFmV)에 보존했다. 원본 학습 Colab Notebook(Drive ID `1IZTQcwMnaCCsSm-mzSWoRuglvC15F7AN`)과 저장 output이 없는 로컬 단일 notebook `notebooks/mission2_final_fulltrain_official_validation.ipynb`에는 분석 코드를 source-only로 추가했다. 이 notebook 셀은 새로 실행하지 않았으며, Git에 오류 개별 행이나 실행 output은 넣지 않는다.
