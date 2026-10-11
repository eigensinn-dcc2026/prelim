# M3-KCBERT-CORR-001 — 증상 간 관계 보조 학습 실험 (A안)

## 목적과 비교 기준

- **가설**: 증상 동시출현(PLCP 유사 과제)과 일부 증상 조건부 예측(CLCP 유사 과제)을 함께 학습하면 특히 오심·두통 등 증상의 Macro F1이 올라간다.
- **Baseline**: `scripts/lab/m3-kcbert-012.py` (KcBERT + weighted BCE + soft macro-F1).
- **변경**: Training에서만 9개 증상 쌍 36개의 동시 양성 여부와, 무작위로 가린 증상 예측을 **보조 손실**로 추가한다. 각 보조 손실 가중치 0.2로 사전 고정. 이는 ACL 논문의 정확한 복제 구현이 아닌 **PLCP/CLCP 아이디어를 간소화한 파일럿**이다.
- **고정**: 입력은 원래 `utterances[].text` 순서, 서울 Training만 사용, 동일 80:20 multilabel stratified 분리(seed 42), KcBERT revision `0f2f3f8`, 앞 256토큰, pos_weight, LR, batch, 3 epoch, 학습 seed 42/43/44. 추가 정답은 **보조 학습 시에만** 사용한다.
- **평가**: 같은 holdout에서 증상별 F1, Macro/Micro F1, FP/FN. 예측은 원래 **9개 출력**만 사용하고 모든 증상에 `>=0.5` 고정. 추가 후처리 없음. 공식 Validation은 이 실험에서 사용하지 않는다.
- **채택 기준**: KCBERT-012의 동일 분할·3-seed 평균 Macro F1보다 **+0.01 이상** 개선하는지 확인하되, seed별 편차도 함께 제시한다. 그보다 작은 차이는 채택 근거로 과장하지 않는다.

## 코드

`scripts/lab/m3-kcbert-corr-001.py`

보조 헤드는 학습 그래디언트에만 영향을 주며 추론에서는 정답/조건부 라벨을 입력으로 쓰지 않는다. **체크포인트에는 보조 헤드 가중치가 추가되므로 이 파일을 바로 최종 제출용 inference.py에 연결하지 않는다.** 성능이 검증된 뒤 별도의 규정 적합 추론 경로를 준비해야 한다.

## 실행

저장소 루트에서 실행하며 기존 lab 환경(`PYTHONPATH=src`, `DCC_RUN_ROOT`, `LAB_WORK`, 데이터 Drive 연결, CUDA 환경)을 사용한다.

```bash
python scripts/lab/m3-kcbert-corr-001.py --smoke

export PYTHONPATH=src
export LAB_EXP_ID=M3-KCBERT-CORR-001
export LAB_OUT=/path/to/experiment-output
export LAB_WORK=/path/to/private-working-dir
export DCC_RUN_ROOT=/path/to/run-root
python scripts/lab/m3-kcbert-corr-001.py
```

`--smoke`는 다운로드·학습 없이 보조 손실, 마스킹, 그래디언트 등의 작은 검사를 실행한다. 실제 학습에는 GPU, KcBERT 체크포인트와 주최 측 제공 데이터가 필요하다.

## 진행 상태

- 코드 작성: 완료
- `--smoke` 실제 실행: **미실행** (공유 GitHub 코드 검토 기준)
- Training 전체 3-seed 학습: **미실행**
- 성능 결과 및 ADOPT/REJECT/INCONCLUSIVE: **미정**

제출일(10/14) 전에 검증이 불가능하면 예선 제출 모델에 섞지 않고 본선용 실험으로 보존한다.
