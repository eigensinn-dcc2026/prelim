# 검증 기록 — 2026-10-11

## 완료

| 검증 | 결과 |
|---|---|
| 새 초기 실험 + 기존 문맥 모델 + RoBERTa 추론 회귀 | 32 passed, 152.78초 |
| 추가 학습 재개·최종 후보 보호 | 1 passed, 29.67초 |
| v2 공식 평가 1회 제한·실제 standalone CSV 실행 | 1 passed, 71.41초 |
| 근거 규칙 수정 후 새 전체 테스트 | 13 passed, 111.71초 |
| 소스 포함 Colab 노트북 | nbformat schema·전체 code-cell AST 통과 |
| source ZIP | CRC 통과, manifest/guards 전이 의존성 포함 |
| 변경 파일 공백 검사 | git diff --check 통과 |
| 실제 Training29200건 길이·기존 미탐 진단 | 완료, 415.09초; 증상 라벨별 어휘 위치 진단 |
| 진단 그림 | 생성 후 시각 확인 완료 |

테스트 사이에는 중복이 있으며 통과 횟수를 더해 독립 테스트 개수로 표시하지 않는다.
모델 학습·추론 검증은 작은 합성 BERT를 사용했다. 이 테스트의 F1은 대회 성능이 아니다.
실제 데이터에서는 기존 점수와 길이/어휘 위치만 확인했다.

```powershell
python -m pytest tests/test_mission3_next.py tests/test_mission3_context.py tests/test_inference_mission3_bert.py -q
python -m pytest tests/test_mission3_next.py -q
python scripts/build_mission3_next_colab.py
python scripts/report_mission3_next_diagnostics.py --audit outputs/m3_next/audit_local_v2 --out reports/mission3/next
```

입력·분할 지문은 기존 실험과 같다:

- 원본 체크포인트 SHA-256: `bbc0bd7a5f320df78bc20e59fc98827c6d9bb9367d23ac45367d7c2ff9a56e7e`
- rows SHA-256: `d2b1c79e9a5f9781d9c9bcefd70873f50c9afafca64e2fc888d62ccbea659ca6`
- split SHA-256: `c168831a3008f8fecaa43f0d319143d428d8f8f0fbdb42a79e672ebffb2db267`

처음 관찰한 Training 근거 후보에서 범위를 잘못 적용한 부정 오류를 수정했다.
“기력이 없”, 다른 사건의 “없고”, 담화 표현 “아니”, 부탁 표현 “주세요”,
다른 질문 이후의 “네”를 회귀 사례로 추가했다. 이 검토는 약한 감독 후보의
오류 점검이며 수동 근거 정답 데이터셋 구축 완료를 뜻하지 않는다.
진단 그림은 잘림과 미탐의 인과관계를 주장하지 않는다.

## 아직 실행하지 않은 작업

새 원본 RoBERTa 손실3종의 전체 GPU 학습, 입력 모델 전체 학습,
3seed gate, 전체 Training 재학습, 새 공식 평가, 실제 제출 패키지 동결은 미실행이다.
구현은 해당 단계를 조건부로 연결했으나 실제 데이터에서 성공 여부는 확인 전이다.
소스·설정이 바뀌면 재개를 거부하며, smoke는 채택할 수 없다.
최종 제출 패키지는 기준을 통과한 실제 가중치와 공식 결과가 있어야 생성된다.

## Colab 장애와 인계

동일 사용자 노트북의 계정 `daulkim2001@gmail.com`을 확인했다.
파일 탐색을 열면서 새 A100 세션에 연결됐다. 파일 목록에는 기본 `sample_data`만
있었고 업로드 도구는 “Allow access to file URLs” 권한 오류로 실패했다.
이후 브라우저 제어 연결도 중단돼 노트북 편집이나 업로드를 완료하지 못했다.
새 데이터·가중치·학습 결과를 해당 런타임에 저장하지 않았다.

빈 런타임 해제는 Colab이 세션 파일을 영구 삭제한다고 안내하므로 실행 직전
승인을 요청했다. 승인 응답 없이 삭제를 수행하지 않았다. 마지막으로 확인한
세션은 연결 상태였다. 현재 사용량·런타임 상태는 브라우저 연결 복구 후 확인해야 한다.

새 노트북에는 소스를 포함했으므로 `outputs/m3_context_gpu/input.tar.gz`만
세션 저장소에 올리면 된다. 기존 과거 노트북의 “모두 실행”을 사용하면 실패한
v1 실험을 재실행하므로, 새 노트북을 실행해야 한다.
