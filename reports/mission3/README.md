# Mission 3 KcBERT 계열 근거 파일

이 폴더는 Mission 3 KcBERT 계열의 결정 및 실험 로그에 연결되는 근거 파일을 보관합니다.

| 파일 | 용도 | 근거 수준 |
|---|---|---|
| `m3_kcbert_baseline_train.log` | 최초 KcBERT 3 epoch 학습 및 내부 검증 결과 | 직접 실행 로그 |
| `m3_chunk_validation_metrics.json` | 전체 토큰 청크의 공식 Validation 지표 | 기계 판독 JSON |
| `m3_turn_chunk_validation_metrics.json` | 발화 턴 청크의 공식 Validation 지표 | 기계 판독 JSON |
| `m3_weighted_diagnosis_sources.json` | 강한 가중치 모델 결과를 정리한 출처 기록 | 화면 결과 전사·추가 확인 필요 |
| `m3_threshold_comparison.json` | 임계값 변경 전후 증상별 지표 | 선택 임계값 배열 미보존 |
| `m3_error_analysis_summary.json` | 공식 Validation 오답의 전사 단서 분류 요약 | 가공 요약·상세 전사 미포함 || `m3_hardnegative_validation_summary.json` | hard-negative 모델의 공식 Validation 지표와 이전 실행 비교 | 보고용 집계·원본 실행 로그 추가 확인 필요 |
| `m3_role_official_validation_metrics.json` | 규정에 맞지 않는 역할 토큰 과거 실행의 공식 Validation 집계 | 실행 화면 전사·원본 JSON 없음·모델 선택 근거 사용 금지 |

> 2026-09-28 오류분석의 식별자는 `M3-ERR-PR87-001`로 구분합니다. 원래 기록에 사용된 `M3-ERR-001`은 `main`의 2026-10-09~10 오류분석 ID와 겹치므로, 원본 식별자는 JSON의 `original_experiment_id`에 보존했습니다.

가중치 및 임계값 관련 파일은 실험을 수행했다는 근거로 보존하지만, 누락된 원본 결과와 설정을 확인하기 전까지 최종 성능 근거로 사용하지 않습니다. 특히 M3-ROLE-001은 `speaker` annotation을 사용해 Mission 3 입력 규정을 위반하고 임계값 0.5 고정 여부도 확인할 수 없으므로 기각된 참고 기록입니다.

체크포인트는 용량 때문에 Git에 포함하지 않습니다. 필요할 경우 외부 저장 위치와 SHA-256을 별도로 기록합니다.
