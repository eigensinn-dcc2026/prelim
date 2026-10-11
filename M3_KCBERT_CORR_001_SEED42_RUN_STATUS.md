# M3-KCBERT-CORR-001 seed 42 실행 상태

- 점검일: 2026-10-11 (KST)
- 상태: BLOCKED_BEFORE_TRAINING
- 판정: INCONCLUSIVE (학습 미실행, 성능 판단 불가)
- 기준: PR #155 head `f7bade86268d8aec2f1b956ad50c13b8d010c88c`
- 노트북: `notebooks/mission3/M3_KCBERT_CORR_001_COLAB.ipynb`
- 작업 브랜치: `exp/m3-kcbert-corr-001-seed42-colab`

## 실제 수행 및 차단 원인

Colab 홈페이지에 접근했으나 원격 브라우저가 로그아웃 상태였다. Google 로그인 링크에서 `502 Bad Gateway` / `[Errno 111] Connection refused`가 표시되었으며, 한 번 새로고침해도 동일했다. 따라서 로그인, Drive 연결, T4 할당, 노트북 셀 실행, 공식 Training 데이터 접근 및 학습은 수행되지 않았다. 이는 관찰된 연결 오류이며 학습 코드 오류나 봇 차단으로 판정하지 않는다.

PR의 학습 스크립트를 별도 로컬 환경에서 `--smoke`로 실행하여 종료 코드 0을 확인했다. 단, 해당 환경에 torch가 없어 tensor loss/gradient 검사는 명시적으로 SKIP되었다. 따라서 전체 smoke 통과, Colab 실행 성공 또는 GPU 검증으로 해석하면 안 된다. 실행 출력의 포괄적인 `smoke OK` 문구보다 이 제한이 우선한다.

## 다음 실행의 고정 조건

- Google Colab T4 GPU, training seed 42, 3 epochs.
- 공식 `Training/TL_서울_구급.zip`만 사용. Training 내부 multilabel stratified 80:20, split seed 42.
- 공식 Validation을 평가나 튜닝에 사용하지 않는다. 기존 경로 탐색기는 Validation 디렉터리 존재 여부만 확인한다.
- KcBERT front-256, LR 2e-5, batch 16, weighted BCE + soft F1, PLCP/CLCP 유사 보조 손실 각 0.2, threshold >=0.5.
- 결과 경로: 개인 Drive `dcc_runs_me/lab/M3-KCBERT-CORR-001-seed42`.
- 계획서와 노트북에 기록된 KCBERT-012 seed42 기준 Macro F1은 0.6337이다. 이번 시도에서 원본 baseline 결과 파일이나 split 일치 여부를 대조하지는 못했다. 실제 비교 전 원본 지표와 동일 데이터/분할을 확인한다.

## 결과

| 항목 | 값 |
|---|---|
| 새 모델 Macro F1 | 미측정 |
| KCBERT-012 대비 차이 | 계산 불가 |
| 학습 metrics.json / checkpoint | 이번 시도에서 생성 안 됨 |
| T4 GPU 할당 | 수행 안 됨 |
| 학습 중 오류 및 수정 | 학습 미시작으로 확인 불가 |

로그인 연결 문제가 해결된 실행 환경에서 노트북 ①~⑦을 실행하고, 실제 metrics.json과 학습 로그를 확인한 후 이 기록을 갱신한다. seed42 하나로 채택을 확정하지 않는다. 공식 데이터/전사/개별 예측/체크포인트는 GitHub에 업로드하지 않는다.

이번 커밋은 상태 문서만 추가한다. 기존 제출 모델, main, PR #155 원본 브랜치와 학습 코드는 변경하지 않았으며 병합하지 않았다.
