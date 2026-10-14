# M3 승인 실험 준비 — 2026-10-12 KST

사용자 최신 지시: **GPU 실행 보류, 코드·검증만 준비**. Colab 계정은
`daulkim2001@gmail.com`. A100 실행은 승인되지 않았고 실제 GPU 학습을 시작하지 않았다.
이 문서는 실험 설계와 구현 설명이며 새 성능 결과를 주장하지 않는다.

## 실행 파일

- `notebooks/M3_approved_experiments_20261012.ipynb`: 소스 내장 Colab 노트북.
  `RUN_GPU=False`, `RUN_PREFLIGHT=False`, 다운로드도 기본 비활성화.
- `scripts/train_mission3_approved.py`: 단계별 학습/데이터 검증/사람 주석 양식.
- `src/dcc_data/mission3_approved_core.py`: 손실, 누수 검사, 근거 상태 모델, 교차 적합 재검토.
- `scripts/inference_mission3_approved.py`: 오프라인 실험 모델 추론 및 CPU 시간 측정.
- `scripts/summarize_mission3_approved.py`: 실제 결과가 존재할 때만 비교/paired bootstrap.
- `outputs/m3_approved/approved_source.zip`: 소스·테스트만 포함. 비공개 데이터 없음.

기존 `outputs/m3_context_gpu/input.tar.gz`를 재사용한다. 입력은 Training 전사/9개
라벨, baseline.pt, 내부 검증 확률이다. 공식 Validation은 포함되지 않는다.
학습 분할은 기존 23,408 / 내부 검증 5,792를 파일명으로 복원하고 라벨 정렬을 검사한다.
학습/검증의 파일명·통화 ID·정규화 전사 해시 중복이 있으면 중단한다.
전사문 이외의 메타데이터는 인코더에 전달하지 않는다. 판정은 항상 sigmoid >= 0.5.

## 1. soft-F1 집계 범위

| 비교 | 고정하는 것 | 바꾸는 것 |
|---|---|---|
| micro8 / global16 | 유효 배치16, 동일 기존 체크포인트, seed, 학습률, 샘플 순서 | F1 집계 8 / 16 |
| micro8 / global64 | 유효 배치64, 동일 기존 체크포인트, seed, 학습률, 샘플 순서 | F1 집계 8 / 64 |

기존 ROBERTA-003에서 1 epoch 연속 학습, 인코더 LR 1e-6. 작은 마지막 배치는 실제
샘플 수로 가중한다. global은 여러 microbatch의 **연산 그래프를 유지한 logits**를
모아 F1을 계산한다. 각 microbatch별 F1에 gradient accumulation만 적용한 방식과
다르다. 모든 대조군에 동일한 gradient checkpointing을 사용한다.

GPU 메모리와 단계별 시간은 아직 미측정. 각 배치 크기 안의 비교가 집계 방식의 효과다.
배치16과 배치64 사이에는 optimizer step 수도 달라지므로 이를 집계 효과만으로 해석하지 않는다.
seed42 pilot 이후 유망한 쌍만 seed43/44로 반복한다.

## 2. 사람 근거 감독

`annotations` 단계는 Training 300통화를 고정 난수로 뽑아 `texts.csv`와
`template.jsonl`을 생성한다. 실제 증상 정답은 검토 화면에서 숨긴다. 각 통화의
9개 증상을 `positive/negative/unknown`, 정규화 전사문 내 `[start,end)` 문자 구간으로
검토한다. reviewer와 provenance=human이 필요하다. 정규식/AI 제안은 사람 주석이 아니다.

추가 수동 주석 허용 여부는 저장된 규정만으로 명시적으로 확정되지 않아,
`--human-annotation-rules-confirmed`와 검토 완료 JSONL 없이는 이 단계가 실행되지 않는다.
현재는 사람이 작성한 주석이 없으며, 생성한 양식을 학습 정답으로 쓰지 않는다.

모델: 증상별 token attention → 근거 표현/3상태 → 기존 CLS 예측에 residual 추가.
동일 구조의 보조 감독 없는 대조군과 비교한다. 인코더 LR1e-6, 새 모듈 LR1e-4,
보조 손실 가중치0.1, 1 epoch. 주석이 없는 라벨에는 보조 손실을 주지 않는다.
긍정·부정 근거가 입력 잘림으로 사라졌으면 그 상태 감독도 생략한다.
근거 미확인을 강제 음성 판정으로 바꾸지 않는다. Attention은 모델의 기여도이며
사람이 검증한 설명이라고 표시하지 않는다. 사람의 설명 정확성 평가와 가림 실험은 별도 필요하다.

## 3. OOF 증류

기존 29,200건 전체의 ROB5F OOF 교사는 재사용하지 않는다. 이 교사들은 학생 평가용
5,792건 일부를 학습했을 수 있어 새 학생의 검증에 누수가 생긴다.

대신 **23,408건 안에서 새로운 multilabel stratified 3-fold**를 만든다.
교사는 공개 사전학습 `klue/roberta-large@28d9112`에서 fold마다 새로 시작한다.
기존 task-finetuned checkpoint를 OOF 교사의 초기값으로 사용하는 것도 금지한다.
각 fold의 학습 부분으로 char TF-IDF(2~5gram)+LR을 함께 학습하고
고정 `0.6*RoBERTa + 0.4*LR`을 교사 신호로 쓴다. 기존 최고 앙상블과는 다른 새 레시피다.
교사 neural 학습은 3 epochs, LR1e-5, micro8/effective16, 기존 weighted BCE+soft-F1.

학생 `beomi/KcELECTRA-base-v2022@8728557`을 3 epochs, LR2e-5로 학습한다.
hard-label 대조군과 `0.5*기존 손실 + 0.5*교사 확률 BCE`를 비교한다.
각 학생 학습행에는 자신을 학습하지 않은 교사의 확률만 사용한다.
교사 모델 revision, split, 파일 해시, 확률 파일 해시와 행 순서를 저장·검사한다.
각 교사의 TF-IDF/LR도 `linear.joblib`로 보존하며 추론 전에 neural/linear 해시를 검사한다.
teacher seed는42 고정, 학생 seed만42/43/44로 반복한다.

성공 목표는 동일 외부 holdout에서 해당 교사 대비 Macro F1 하락 <=0.003,
같은 CPU 조건의 실제 추론 시간 <=교사의 절반이다. 예측 정확도와 속도 모두 아직 미측정.
교사가 기존 최고 기준보다 낮으면 단순히 작은 모델을 만들었다고 최종 채택하지 않는다.

## 4. 선택적 512토큰 재검토

기존 checkpoint에서 256·512 각각 같은 1 epoch를 추가 학습해 길이 차이와
추가 학습 효과를 분리한다. 모든 통화는256으로 예측하고, 본문254토큰 초과 통화 중
최소 `abs(p-0.5)`가 전역 재검토 margin 이내인 통화만512로 다시 예측한다.
재검토 통화는512 확률로 대체하고 최종 증상 threshold는0.5로 유지한다.

내부 검증에서 5-fold 교차 적합: 다른4fold로 margin 0/.025/.05/.1/.2/.5 중 하나를
선택하고 남은 fold를 평가한다. 항상512 대비 F1 하락 <=0.003인 후보 중 재검토가
가장 적은 것을 선택한다(해당 후보가 없으면 최고F1). 같은 수의 긴 통화를 무작위로
재검토한 대조군도 제공한다. 이 통계는 내부 탐색이며 공식 점수로 표시하지 않는다.
실제 CPU 전체 지연 시간은 별도로 측정한다. 재검토 비율을 속도 향상 실측값으로 바꾸지 않는다.

CPU 교사 전체는 `inference_mission3_approved.py --teacher-dir run/teachers`로 측정한다.
학생은 `--ckpt-path .../model.pt`, 두 경우 모두 같은 `--label-dir`, `--limit`, `--threads`를
사용한다. `total_seconds_including_load`는 시작 시 로딩까지 포함한 시간이다. 교사에는
무결성 검사, 학생에는1회 warm-up이 포함되므로 장기 상주 서버 지연으로 일반화하지 않는다.

## 향후 실행

```powershell
# GPU 없이 데이터 검증 및 검토 양식 생성
python scripts/train_mission3_approved.py preflight --bundle outputs/m3_approved/input --out outputs/m3_approved/preflight
python scripts/train_mission3_approved.py annotations --bundle outputs/m3_approved/input --out outputs/m3_approved/preflight
```

GPU는 재승인 후 Colab에서만 실행한다. 예:

```text
python scripts/train_mission3_approved.py loss --bundle /content/m3_approved/input --out /content/m3_approved/run --execute --max-hours 3
```

`teachers`, `student`, `evidence`, `cascade`는 같은 CLI의 단계 이름이다.
`--execute`가 없거나 Colab CUDA가 아니면 학습을 거부한다.
동일 out 폴더의 누적 시간 제한은 최대3시간이며, 전체 단계가3시간 내 완료된다는 뜻이 아니다.
budget.json이 중단되면 승인 없이 한도를 늘리거나 새 폴더로 우회하지 않는다.
완료 epoch는 optimizer/scheduler/scaler/RNG와 함께 resume.pt에 저장한다.
코드·데이터·분할·설정이 달라지면 기존 결과 재사용을 거부한다.

## 검증과 채택

CPU 테스트는 미분값의 수학적 일치, 불완전 배치, 누수 차단, 사람 주석 검증,
잘린 근거 처리, padding 마스킹, 실제 작은 모델 학습/오프라인 복원 등을 확인한다.
작은 무작위 모델의 테스트 수치는 대회 성능이 아니다.

유망 pilot만 3seed 반복, 기존 앙상블과 paired bootstrap 및 Micro F1을 함께 비교한다.
큰 개선 목표 +0.02는 예상치가 아니다. 후보별 기여를 대조군과 분리하고 선택 후의
낙관성을 기록한다. 공식 평가 및 실제 제출 모델 교체는 이 키트가 자동 수행하지 않는다.
실제 학습 로그, 파일 크기와 SHA-256을 확인한 뒤에만 체크포인트를 보존 완료로 표시한다.
