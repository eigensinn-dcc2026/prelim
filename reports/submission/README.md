# 제출 형식 정리 — 2026-10-10

공식 제출 단위는 `mission1/`, `mission2/`, `mission3/` 세 폴더와 결과 정리 PPT 한 개입니다. 실제 파일은 저장소의 `submission/`에 있으며, 가중치는 로컬에 포함하고 Git에서 제외합니다.

| 항목 | 준비한 내용 | 검증 결과 |
|---|---|---|
| M1 | 공식 CLI, 564D 모델 코드, LR checkpoint, WavLM config/processor/weights, 고정 requirements, 학습 코드와 실제 GPU 기록 ipynb | 허용 필드만 남긴 실제 Validation 2통화에서 독립 CLI 실행. 이전 공식 평가 CSV와 일치 |
| M2 | 공식 CLI, 기존 CNN 추론 모듈, 의존 모듈, 단일 ensemble packing 코드, requirements, 참고 재학습 ipynb | 테스트용 임의 3모델 가중치로 67발화 출력, 형식·행수·0/1·fallback 0 확인. 최종 가중치 성능 검증 아님 |
| M3 | 검증한 RoBERTa seed 42 checkpoint, 공식 CLI, 모델 코드, 고정 requirements, 실제 GPU seed별 학습 기록 ipynb | 허용 필드만 남긴 실제 Validation 2통화에서 독립 CLI 실행. 기존 3,640통화 전체 검증 CSV와 일치 |
| PPT | 19장 공식 템플릿 유지, M1 97.47%, M3 채택, M2 가중치 미확보 상태 반영 | 표/그래프 원본 Excel 연결 보존, 별도 구조·레이아웃 검사 및 화면 점검 |

## 남은 필수 자료

1. M2-RERUN-001의 최종 3-seed CNN 가중치 또는 각 seed 가중치. 기록상 Drive `m2_cache/m2_rerun_from_raw/rerun_2026-10-06b/`에 있으나 현재 접근 가능한 자료에서 발견하지 못했습니다.
2. 위 M2 재학습에 대응하는 실행 출력이 보존된 원본 노트북. `mission2/training_reference.ipynb`는 최종 실행 로그로 확인하지 못한 참고본입니다.

이 자료를 확보한 뒤 M2 가중치 전체 SHA256과 공식 Validation 점수 재현을 확인해야 최종 제출본이 됩니다. 지연우 공유 comparison 폴더의 Training 내부 검증 결과는 이 자료를 대신하지 않습니다.

M1·M3 `training.ipynb`는 실제 사용한 학습 코드와 완료 작업 JSON의 학습 기록을 markdown으로 보존한 정리본입니다. 이번에 학습을 다시 실행한 ipynb로 표시하지 않았고, code cell outputs를 만들어 넣지 않았습니다. 세 미션 모두 새 가상환경에 패키지를 처음 설치해 보는 검증은 수행하지 않았습니다. 이번 독립 실행은 기존 설치 환경에서 저장소 PYTHONPATH 없이 수행했습니다(M1 checkpoint와 동일한 sklearn 1.6.1만 로컬 dependency 폴더에서 제공).

제출 압축파일은 미완료 M2를 포함하므로 `DRAFT`로 표시합니다. 원본 데이터, 테스트용 임의 가중치, 생성 CSV, 캐시, Git 파일은 넣지 않습니다. GitHub 업로드나 주최 측 제출은 수행하지 않았습니다.
