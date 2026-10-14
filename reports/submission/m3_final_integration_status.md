# M3 최종 제출 반영 상태

확인 시점: 2026-10-12 00시대 KST. 상태: **학습 결과 대기, 제출 폴더 교체 미완료**.

작업 브랜치: `final/model-selection` (`wt-final-model-selection`).
학습/앙상블 추론 코드: `wt-m3-kit`, commit `7a55d9f0cd4ab8e2e9ab39cf3be157da4f8718ca`.

## 확보한 증거와 준비

- [M3-FINAL-002 Drive 폴더](https://drive.google.com/drive/folders/1BPkX1zOzuNu1V8-khFB3mIg-amIuzDid)에서 최종 학습이 진행 중임을 확인했다. L4, Training 29,200건, seeds 42/43/44, 각 3 epoch.
- seed 42 가중치는 Drive에 저장됐다. 크기 1,347,798,651 bytes, 기록 SHA256 `7280f4f594441ab5c43d828e53d76712b492bfad9bdd7f0d43d28242bf553308`. 이는 sidecar의 기록이며 로컬 바이너리를 직접 해시 검증한 결과가 아니다.
- seed 42 sidecar를 `outputs/m3_final_download/m3_roberta003_full_seed42.pt.json`에 확보했다. history 3개, rows 29200, epochs 3을 확인했다. epoch당 약 970초였다.
- seed 43/44, 최종 앙상블, 실행 완료 노트북은 마지막 확인 당시 아직 없었다. 남은 두 seed 학습만 약 97분이며 평가·내보내기·다운로드·패키징 시간이 추가된다.
- 앙상블 추론 코드를 `outputs/m3_final_candidate/mission3`에 준비했다. **이 폴더에는 최종 가중치가 없으므로 제출할 수 없다.**
- 기존 kit의 의미 있는 앙상블 테스트 4개 통과: TF-IDF 확률 일치, 안전한 checkpoint 로딩과 확률 평균, 가중치 합, CLI 추론. `4 passed in 235.84s`.
- `scripts/accept_m3_final.py`를 추가했다. 완성 산출물·실행 노트북·해시·전체 Validation 점수 재계산·오프라인 실제 추론을 검사하고, `--activate`일 때 기존 폴더를 백업한 뒤 교체한다. 문법 검사와 산출물 누락 거부 확인을 완료했다. 완성 가중치가 없어서 이 도구의 전체 승인 경로는 아직 실행하지 못했다.

## 남은 문제

1. **최종 학습이 끝나지 않았다.** 실행 완료 노트북을 미리 만들거나 기존 80% 학습 가중치를 최종 학습본으로 표시하지 않는다.
2. Drive connector는 seed 42 다운로드를 HTTP 413으로 거부했다: 1.35GB가 도구의 256MiB 제한을 초과한다. Chrome의 실제 Drive 다운로드도 시도했으나 다운로드 제어 오류가 발생했고 로컬 파일 확보를 확인하지 못했다. 다운로드 목록 페이지는 브라우저 URL 정책으로 접근이 거부돼 추가 UI 확인을 중단했다. 완성 바이너리는 인증된 정상 다운로드 경로로 로컬에 확보해야 한다.
3. 최신 `dcc2026-prelim/docs/decisions.md`는 OOF 기준 RoBERTa:TF-IDF **0.5:0.5를 기본**으로 기록하고 최종 결과로 확정한다고 한다. 진행 중인 kit는 **0.55:0.45**를 고정 사용한다. 최종 반영 전에 결정과 패키지 가중치를 일치시켜야 한다. Validation 점수를 보고 가중치를 탐색하지 않는다. 승인 도구의 `--expected-bert-weight`는 이 불일치를 감지한다.
4. 제출 폴더와 팀 ZIP은 아직 교체하지 않았다. 기존 submission/mission3 checkpoint SHA256은 `bbc0bd7a5f320df78bc20e59fc98827c6d9bb9367d23ac45367d7c2ff9a56e7e`다.

## 반영 순서

완료된 Drive export 전체를 `outputs/m3_final_download`에 내려받는다. 최종 설정과 export 가중치를 일치시킨 뒤 승인 도구를 실행한다. 예를 들어 0.5로 확정했다면 해당 비율로 export/저장된 평가 산출물을 먼저 준비한 후:

```powershell
python scripts/accept_m3_final.py --artifacts outputs/m3_final_download --expected-bert-weight 0.5 --activate
```

통과한 경우 submission/mission3의 final_validation.json에서 활성화 여부·해시·오프라인 증거를 확인한다. 이어서 전체 제출 manifest와 팀 ZIP을 다시 생성하고 preflight를 수행한다. 이 명령은 학습이나 재가중치, 팀 ZIP 재생성을 수행하지 않는다.
