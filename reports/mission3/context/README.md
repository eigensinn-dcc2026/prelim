# M3 문맥 개선 실험

구현은 A(유사 양성·음성 통화 대조), B(질문과 뒤 두 발화의 근거 attention), CLS 보정 대조군을 비교한다.
목표 공식 Macro F1은 0.67이며, 실제 학습 결과와 목표는 구분한다.
현재 제출 후보 ROBERTA-003 공식 0.6433과 기존 최고 앙상블 공식 0.6510을 유지한다.
내부 검증의 기준 앙상블은 `0.45*r3 + 0.15*k12 + 0.40*p0`이다.

## 입력과 고정 설정

- 이전 예측 NPZ의 파일 이름으로 Training 23,408 / 내부 검증 5,792를 복원한다. 세 NPZ의 이름·정답 일치를 검사한다.
- `utterances[].text`만 정규화해 연결하고 앞 256토큰을 사용한다. B의 경계는 fast tokenizer의 문자 offset으로 복원한다.
- 화자 번호, 증상 정답, 연령, 위치 등의 JSON 필드는 모델 입력에 들어가지 않는다. 파일 이름은 검색 제외에만 사용한다.
- 9개 증상 순서와 `sigmoid >= 0.5`, 빈 예측 `[]`, CSV 스키마를 그대로 유지한다.
- 고정 인코더: 차원 128, dropout 0.1, AdamW lr 0.001, batch 128, 10 epoch. 기존 가중 BCE + soft-F1을 fp32로 계산한다.
- 추가 학습: seed 42·43·44, 인코더 lr 1e-6, 새 모듈 lr 1e-4, 1 epoch. 기존 학습과 같은 micro batch 8 / effective batch 16.
- A: 양성 8 / 음성 8 / 전체 32, temperature 0.1. 학습 통화·같은 통화 ID·같은 전체 정규화 전사 SHA256을 제외한다. 유효 양성·음성이 없으면 기존 logits로 복귀한다.
- B: 각 발화와 뒤 두 발화의 평균 contextual representation을 연결해 128차원으로 투영하고, 증상별 query로 attention을 계산한다. CLS와 결합한 residual을 학습 게이트로 기존 logits에 더한다.

## 실험·선택·채택

처음에는 control/A/B를 seed 42로 학습한다. CLS 보정 대조군, 원래 RoBERTa, 기존 최고 앙상블을 모두 넘는 방법 중 높은 Macro F1을 선택한다. 차이 <0.001이면 B를 선택한다. 둘 다 실패하면 반복/추가/최종 학습을 중단한다.

2026-10-10 첫 실험은 두 방법 모두 실패했다. 이후 사용자가 추가 학습을 요청해, **별도 확장 실험**으로 control/A/B 각각의 epoch-10 head에서 seed 42 인코더 1 epoch를 학습한다. 기존 분할·학습률·임계값과 총 누적 실행 예산은 유지한다. 이 단일 seed 결과만으로는 채택하지 않으며, 기존 3개 seed 검증 및 공식 평가 기준을 유지한다.

선택 방법, CLS 대조군, 구성 요소 제거 모델을 같은 seed 3개로 비교한다. A 제거 모델은 검색 정보를 0으로 만들고 같은 크기의 head를 학습한다. B 제거 모델은 뒤 발화 두 슬롯을 0으로 만든다. **B 제거 모델에도 RoBERTa 전체 문맥과 CLS는 남아 있으므로, 명시적인 답변 채널의 기여를 비교하는 실험이다.**

고정 인코더 및 추가 학습 결과를 모두 기록한다. 구성 요소 기여는 별도 보고하며, 사전 지정한 채택 기준은 다음과 같다: 최고 기존 앙상블 대비 평균 +0.003 이상, 최소 2개 seed 개선, 통화 단위 paired bootstrap 95% CI 하한 >0, 평균 Micro F1 하락 ≤0.005. 중복 통화 ID가 있으면 함께 재표집한다. bootstrap 1,000회, RNG seed 42.

통과하면 전체 Training 29,200건, 대표 seed 42로 설정을 고정해 학습한다. 공식 Validation은 별도 명령에서 동일 후보를 한 번 평가한다. 중단된 동일 후보 추론만 shard로 재개할 수 있다. 공식 Macro F1 ≥0.6540인 경우에만 패키지를 만든다. 기존 제출 폴더를 자동 덮어쓰지 않는다.

## 실행

```bash
python scripts/train_mission3_context.py \
  --train-label-dir /data/Training/labels \
  --base-ckpt /checkpoints/m3_roberta003_seed42.pt \
  --reference-probs-dir /reports/mission3/ensemble/probs \
  --out outputs/m3_context --device cuda --phase all --max-hours 10
```

`--phase pilot`은 첫 비교까지 실행한다. `--phase repeat`은 반복·구성 요소 제거·추가 학습·채택 판단까지 실행한다.
같은 명령으로 완료된 feature shard와 head epoch, encoder epoch를 재사용한다. 데이터·체크포인트·설정·표현 정밀도가 바뀌면 다른 출력 경로를 사용해야 한다.
누적 wall-clock 제한은 기본 10시간이며 초과 시 `budget_reached_resume_pending`으로 저장한다. GPU 요금에는 입력 전달·대기 시간도 포함되므로 Colab 연결 시간을 별도로 관리한다.

작은 CPU 점검은 `--smoke-rows 16 --phase pilot --device cpu`를 추가한다. 이 모드는 head 2 epoch, bootstrap 100회이며, **채택이 차단된다. 점수를 실제 모델 성능으로 사용하지 않는다.**

실패한 동결 실험에서 이어가는 사용자 요청 확장 명령:

```bash
python scripts/finetune_mission3_context_pilot.py \
  --train-label-dir /content/m3_context/training \
  --base-ckpt /content/m3_context/baseline.pt \
  --reference-probs-dir /content/m3_context/reference \
  --run-dir /content/m3_context/run \
  --out /content/m3_context/run_fine_pilot --device cuda --max-hours 9
```

완료된 전체 데이터 동결 실험과 epoch-10 head, feature cache의 해시가 일치해야 한다. 동결 출력은 보존하며, `run/budget.json`을 공유해 실행 예산을 초기화하지 않는다. 완료된 인코더 epoch는 재개 시 재사용한다.

```bash
python scripts/evaluate_mission3_context.py --ckpt outputs/m3_context/final/candidate.pt \
  --official-label-dir /data/Validation/labels --out outputs/m3_context/official --device cuda
python scripts/package_mission3_context.py --ckpt outputs/m3_context/final/candidate.pt \
  --official-result outputs/m3_context/official/official_metrics.json --out outputs/m3_context/package
```

공식 평가 registry는 `reports/mission3/context/official_once.json` 하나를 공유한다. 다른 후보로 이를 초기화해 점수를 튜닝하지 않는다.

## 산출물과 검증

2026-10-11 실측: 동결 A/B Macro 0.6426/0.6428, 인코더 추가 학습 A/B 0.6408/0.6405로 기존 앙상블 0.6539를 넘지 못했다. 현재 제출 후보를 유지한다. 원본 ZIP의 지표·로그·holdout 예측을 검증해 보존했다. 세 실험 가중치는 이전 GPU 세션 종료로 삭제됐다. 자세한 결과와 보존 한계는 `validation.md`에 기록했다.

- `notebooks/M3_context_experiment.ipynb`: 설정·실행 로그·비교·반복·채택 판단. 기본값은 저장된 결과 읽기이며 실제 학습에는 `RUN_TRAINING=True`가 필요하다.
- 출력 폴더의 `run.json`, `history.json`, `holdout.npz`, `fine_holdout.npz`, `metrics.json`, `report.md`, `comparison.png`.
- `methods.png`, `methods.svg`, `slide_text.md`: PPT용 방법 설명. 수치 그림은 실제 실행 결과에서 생성한다.
- `measured_comparison.png/svg`: 동결·추가 학습 비교. `measured_diagnostics.png`는 추가 학습의 증상별 지표를 사용한다.
- `fine_metrics.json`, `fine_report.md`, `fine_execution.log`, `fine_export.log`, `fine_checkpoint_manifest.json`: 검증한 원본 ZIP의 전체 집계 결과와 export 기록. 가중치는 포함되지 않았다.
- `submission/mission3_context`: 별도 오프라인 패키지 템플릿. 새 체크포인트를 포함한 실제 패키지는 공식 통과 후 생성한다.
- 테스트: 정규화·256토큰 경계, 메타데이터 무시, 검색 제외/빈 이웃, residual 초기값, 답변 제거, 저장/복원, 재개 설정 보호, bootstrap, 공식 평가 1회 제한, standalone CSV와 SHA256.

GPU 전달 묶음은 `scripts/stage_mission3_context.py`로 생성한다. 전체 Training의 정규화 텍스트·9개 라벨, 내부 기준 예측, 기준 모델과 코드만 포함한다. 공식 Validation과 오디오는 포함하지 않는다.

## 해석 한계

앞 256토큰 밖의 답변은 읽지 않는다. 내부 질문/부정/정규식 미검출 진단은 규칙으로 정의한 부분집합이며 사람의 정확한 의미 주석을 대신하지 않는다. 검색 labels는 표현 공간에 노이즈를 추가할 수 있고, 답변 제거 실험은 전체 문맥까지 제거하지 않는다. 채택 기준을 실패하면 개선을 주장하지 않고 기존 모델을 유지한다.

기반 연구: [Retrieval-Augmented Multi-Label Text Classification](https://arxiv.org/abs/2305.13058), [Label attention, IJCAI 2020](https://www.ijcai.org/proceedings/2020/461).
