# 준비·검증 기록 — 2026-10-12 KST

현재 상태: **코드·실행 키트 준비, GPU 학습 보류**.
사용자가 A100 실행 제안에 `GPU 실행 보류 — 코드·검증만 준비`를 선택했다.
이후 작업 재개 요청도 GPU 실행 재승인으로 해석하지 않았다.

## 코드 검증

- 신규 실험 테스트: **17 passed**, 51.88초.
  실행: `python -m pytest tests/test_mission3_approved.py -q --disable-warnings --basetemp outputs/m3_approved/tests_workspace_final --junitxml outputs/m3_approved/test_results.xml`
- 기존 BERT 추론 회귀: **6 passed**. 앞선 통합 실행에서 신규14개와 함께 **20 passed**,151.21초.
  신규 테스트 확장 후17개를 다시 실행했다. 중복 실행을 별도 테스트 수로 합산하지 않는다.
- 첫 통합 실행의 실패2개는 sandbox 임시 폴더의 `os.replace` 접근 거부였다.
  임시 경로를 쓰기 허용된 workspace의 새 테스트 폴더로 바꿔 통과했다.
  파일 저장을 비원자 방식으로 바꾸거나 실패를 숨기지 않았다.
- Colab 노트북: nbformat schema, 모든 Python 셀 AST, 기본 RUN_GPU/RUN_PREFLIGHT=False 확인.
- 소스 ZIP: CRC와 각 파일 SHA-256 일치 확인. 비공개 전사·라벨·모델 가중치 없음.

검증한 주요 동작:

1. global soft-F1의 파라미터 미분이 전체 배치 직접 계산과 일치.
2. micro 방식은 기존 학습 코드와 미분까지 일치하며 마지막 불완전 배치도 정확히 가중.
3. 파일명·통화 ID·전사 해시 중복 및 OOF 외부 검증셋 유입 차단.
4. 전체 task-finetuned 모델을 OOF 교사 초기값으로 지정하면 거부.
5. 사람 근거 주석의 출처/범위 검사, 잘린 근거에 대한 잘못된 보조 학습 차단.
6. 근거 모델 초기 residual은 원래 예측 보존, padding 무시, 저장/복원 일치.
7. 작은 모델의 실제 학습 → export → 오프라인 예측 일치.
8. epoch 중단 후 optimizer/scheduler/RNG를 복원한 결과가 연속 실행과 일치.
9. 추론에서 symptom·speaker·triage 변경이 예측에 영향을 주지 않음.
10. 교사 neural/TF-IDF 구성요소 오프라인 추론 및 해시 불일치 거부.
11. 재검토 기준은 자신의 평가 fold 라벨을 바꿔도 영향을 받지 않음.
12. `--execute` 없이 GPU 학습 거부, 결과가 없으면 준비 상태로 보고.

작은 무작위 초기화 모델의 CPU 검사는 구현 검증이다. **M3 성능 개선 실험 결과가 아니다.**

## 데이터 확인

기존 private input.tar.gz에서 Training29,200건, reference, baseline.pt만 로컬로 추출했다.
baseline.pt는1,347,790,829바이트이며 bundle에 기록된 SHA-256과 대조한다.
실제 데이터 사전 점검에서23,408/5,792 분할 및 정답 정렬, 파일명/통화ID/전사 중복
차단 검사를 통과했다. 기존 앙상블 내부 Macro F1은 **0.6539275340833469**로 재현했다.
공식 Validation은 읽지 않았다.

최종 소스 fingerprint를 포함한 점검 결과와 사람 검토 양식은
`outputs/m3_approved/validated`에 보존한다. 이전 점검 파일은 `preflight`에 남긴다.
양식은 Training만 포함하며 미검토 상태다. 사람 주석이 완료됐다는 뜻이 아니다.

## Colab 상태와 남은 단계

지정 계정 `daulkim2001@gmail.com`으로 전환했으나 A100 전환은 취소했고
GPU 학습은 시작하지 않았다. 이후 빈 CPU 런타임은 Colab에서 자동 연결 종료되었다.
Chrome의 `Allow access to file URLs` 권한 부재로 파일 업로드가 실패했다.
권한을 임의로 넓히거나 우회하지 않았으며, 완성 노트북은 로컬 파일로 제공한다.
온라인에 생성된 같은 이름의 준비용 빈 노트북은 완성 키트가 아니다.

남은 실행 조건:

- GPU 사용 재승인 및 Colab에 로컬 노트북/입력 파일 업로드.
- 각 단계의 실제 학습·지표·메모리/시간 측정, 유망 후보의3seed 반복.
- 근거 감독은 수동 주석 허용 범위 확인 및 사람이 검토한 주석 확보.
- 최종 후보 선정, 공식 평가 및 제출 패키지 동결은 미실행.

새 Macro F1, 성능 향상, CPU2배 가속을 주장할 근거는 아직 없다.
