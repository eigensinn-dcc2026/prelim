# scripts/colab — Colab 실행 스크립트 (실행 담당 `DCC 실행(colab)`)

이 폴더는 실행 담당이 Colab(및 공유 Linux 박스)에서 돌린 러너 스크립트를 리포에 반입한 것이다.
이전에는 Google Drive(`MyDrive/dcc_runs/...`)와 공유 박스에만 있었다.

- Colab은 이 파일들을 리포에서 직접 읽지 않고 **Drive 사본**(`/content/drive/MyDrive/dcc_runs/...`)을 `bash`로 실행한다.
  미션 코드는 Drive의 고정 zip `MyDrive/dcc2026-prelim-main-4bfd2c4.zip`(main `4bfd2c4`)을 풀어서 쓴다.
- **앞으로는 GitHub가 원본(source of truth)이다.** Drive에는 아래 sha256과 같은 사본만 둔다.
  스크립트를 고치려면 먼저 이 폴더에서 PR로 고치고, 머지된 파일을 Drive에 그대로 복사한 뒤 sha256이 같은지 확인하고 실행한다.
- 미션 모듈 코드(`src/`, 기존 `scripts/*.py`, `notebooks/`)는 바꾸지 않는다. 데이터, 체크포인트, `.npy/.tar/.pt/.pkl/.joblib`, 로그는 넣지 않았다.
- 파일명의 `_v2`, `_v3`는 실제로 돌린 수정본이다. 이력을 남기려고 이전 버전도 같이 넣었다.

## 파일 목록

"Colab 실행" 열은 2026-10-06 10:00 KST 기준, Drive의 `STATUS.txt` 기록으로 판단했다.

| 파일 | 용도 | 원본 Drive 경로 | sha256 | Colab 실행 |
|---|---|---|---|---|
| `m3_p1/m3_p1_train.py` | M3 P1 KcBERT(front-256, 9 sigmoid) 학습 v1 (best ckpt는 로컬에만 저장) | `MyDrive/dcc_runs/m3_p1/m3_p1_train.py` | `a2c8741a783445aac402009d28629760b65aeb6ac9e6b5cee7b8c2429545318b` | 예 (attempt1, 10-06 04:05 KST 시작, ep3 도중 ~04:30 런타임 끊김) |
| `m3_p1/run_p1.sh` | v1 학습 런처 | `MyDrive/dcc_runs/m3_p1/run_p1.sh` | `663af50d9af7e116e8b3736174e3810197f72084d8dbb30aacae1b5925036613` | 예 (attempt1) |
| `m3_p1/m3_p1_train_v2.py` | v2: BEST 갱신 때마다 best ckpt를 Drive로 복사 | `MyDrive/dcc_runs/m3_p1/m3_p1_train_v2.py` | `ac8c969052d8577666a4a727c43ccf6a019c17e364f895facca4953b272ee1ce` | 기록 없음 (STATUS에 attempt2 RESTART 없음) |
| `m3_p1/run_p1_v2.sh` | v2 런처 (코드 zip 자동 해제, attempt1 로그 보존) | `MyDrive/dcc_runs/m3_p1/run_p1_v2.sh` | `f37f2bfcd92f26abd3699df61d6a2a346a901bad12acf67285617e6bc5d827e0` | 기록 없음 |
| `m3_p1/m3_p1_train_v3.py` | v3: 매 epoch ckpt와 best ckpt를 Drive로 저장, epoch_history.json 기록 | `MyDrive/dcc_runs/m3_p1/m3_p1_train_v3.py` | `f00a6ffad198742979e1fe48e0202cec2f341b163d7d9cffe9496f87db7cb17f` | 아직 아님 (STATUS에 START v3 없음) |
| `m3_p1/run_p1_v3.sh` | v3 런처 (T4 확인, zip/학습 스크립트 sha256 출력, 노트북 셀에서 foreground 실행) | `MyDrive/dcc_runs/m3_p1/run_p1_v3.sh` | `44a9cfe4f251a06fdac081cd0421c2bce9676e6eef5b8956ec683ee98bf7347c` | 아직 아님 |
| `m3_p1/m3_p1_train_v4.py` | v4: 경로만 colab_env 호환 탐색 (하이퍼파라미터는 v3와 동일). colab_env가 없으면 내장 fallback | 없음 (리포에서 추가, Drive 실행 전) | `34cfaa65c95237cf83ee8f3f824073691878aef5cf4982da27882086d788a84d` | 아직 아님 |
| `m3_p1/run_p1_v4.sh` | v4 런처 (스크립트 위치 기준, DCC_RUN_ROOT, zip 또는 git 출처 출력, pip freeze) | 없음 | `84cad2e6f7a14096013d9973071bf7914eb6e48319e533a5a417603c430c8c15` | 아직 아님 |
| `m3_p1/m3_p1_train_v5.py` | v5: P0와 같은 내부 Val 분할, TL_서울_구급만, 예측 CSV (하이퍼파라미터는 v4/v3와 동일) | 없음 (리포에서 추가, Drive 실행 전) | `e02782320bc54ce2db5fc0aa7c6851a71d67c539c5a7159b2da666c780e9f2b7` | 아직 아님 |
| `m3_p1/run_p1_v5.sh` | v5 런처 (기본 zip `dcc2026-prelim-main-6cc1181.zip`, 최상위 폴더 자동 감지, sha256/comment/파일 목록) | 없음 | `94bbcc277d7b64b9b24c3432770815d4c7919a00136c5b1730dc87ce1d58369d` | 아직 아님 |
| `m2_l2_spec/run_l2_spec.sh` | M2 L2 스펙트로그램 재빌드 (Validation → verify → pack → Drive, 이어서 Training) | `MyDrive/dcc_runs/m2_l2_spec/run_l2_spec.sh` | `8e4834fa4ba2726468b52f0830952d6e8607651ded20b781e4f62728da2d5ab0` | 예 (10-06 07:30 KST 시작, Validation 완료, 래퍼는 09:20 KST에 중단) |
| `m2_l2_spec/finish_train.sh` | Training 빌드 마무리 (30분마다 .npy를 train_parts tar로 백업, 빌드 종료 뒤 verify/pack/Drive 복사) | `MyDrive/dcc_runs/m2_l2_spec/finish_train.sh` | `9acbfc1c61b15f30538019e42bcb4172ba9581b9161f4458f0be8d57a65447bf` | 예 (10-06 09:28 KST FINISHER_START, 진행 중) |
| `m3_p0/run_m3_p0_nbconvert.sh` | M3 P0 재실행 노트북을 같은 VM에서 nbconvert로 실행, error 셀/verdict를 STATUS에 기록 | `MyDrive/dcc_runs/run_m3_p0_nbconvert.sh` | `d76fa7a88c9e431ced0f30ed7acfc4bbdf5b40258a599d03232be573dc525117` | 예 (10-06 08:47 KST 시작, END 기록 없음) |
| `m3_p0/mission3_p0_rerun_exec.ipynb` | 위 러너가 실행하는 P0 재실행 노트북 (TF-IDF+LogReg, 80:20 strat, seed 42, 출력 없음) | `MyDrive/dcc_runs/mission3_p0_rerun_exec.ipynb` | `4c37c8449524cb63186a74814fb78d1cc394bafff06c8a92161289d36a0fa687` | 예 (위 러너로) |
| `m3_p0/run_m3_p0_nbconvert_v2.sh` | v2 러너 (v2 노트북 실행) | `MyDrive/dcc_runs/run_m3_p0_nbconvert_v2.sh` | `de2c1671bfe5a6e09734b54a79958e694e9ebe2eff1d812466a2b72da88941dd` | 예 (10-06 08:49 KST 시작, END 기록 없음) |
| `m3_p0/mission3_p0_rerun_exec_v2.ipynb` | v2 노트북: zip sha256, 파일 목록, 스크립트 sha256 출력 추가 | `MyDrive/dcc_runs/mission3_p0_rerun_exec_v2.ipynb` | `39a0d360641d5ec7e2ea9e668f2610198136c43ab13394b0e23abf41207573d0` | 예 (v2 러너로) |
| `m3_p0/run_m3_p0_nbconvert_v3.sh` | v3 러너: Training 서울 라벨(29,200 JSON)을 /content로 먼저 복사(Drive FUSE 경합 회피)한 뒤 nbconvert | `MyDrive/dcc_runs/run_m3_p0_nbconvert_v3.sh` | `8ae2e711af9bbd0e6f97a8f49fae2a219c6c8bc57ab31356eda4be74155c5bed` | 기록 없음 (STATUS에 START v3 없음) |
| `m3_p0/mission3_p0_rerun_exec_v3.ipynb` | v3 노트북: LABEL_DIR을 /content 로컬 사본으로 지정 | `MyDrive/dcc_runs/mission3_p0_rerun_exec_v3.ipynb` | `20a07c77f60cbe8f72bbb0eb604825d584821e89f457e99c38d8e485b73d5d97` | 기록 없음 |
| `m2_cpu_timing/make_random_ckpts.py` | 시간 측정 전용 랜덤 초기화 SpeakerCNNLarge 3개 ckpt 생성 (학습 안 함) | 없음 (공유 박스 `/workspace/m2_cpu_timing/`) | `4a30c2f4ac20b0c80f0d4f6560941500ceac0898677d225693399ea0d9a95fdb` | 아니오 (공유 Linux 박스 CPU에서 실행) |
| `m2_cpu_timing/stage_profile.py` | mission2_inference.py와 같은 함수로 단계별(import/ckpt/WAV/mel/CNN×3) 시간 측정 | 없음 (공유 박스) | `c064477885c29576f53056bda1f2bb6dc966c3221b32e3a2006d37785357d2dd` | 아니오 (공유 박스) |
| `m2_cpu_timing/timing_report.md` | M2 L2 3-seed 앙상블 CPU 추론 시간 측정 보고서 (10-06 09:35–09:46 KST) | 없음 (공유 박스) | `5f2e78778f054b53fe167c11f8935fd8603bcfa4ce5d853e3e3570ed128b22ff` | 해당 없음 (보고서) |

## 참고

- `m2_cpu_timing/`의 두 스크립트에는 측정 당시 박스 경로(`/workspace/m2_cpu_timing/...`)가 하드코딩돼 있다. 기록 보존용으로 원본 그대로 두었다.
- `m3_p1/m3_p1_train.py`와 `m3_p1/run_p1.sh`는 실제로 돌린 Drive 버전이다. 박스 사본은 이후 v2 내용으로 덮여 있어서 쓰지 않았다.
- 실행 기록(`STATUS.txt`, `train.log`, 빌드 로그)과 산출물(ckpt, .npy, .tar, parquet, manifest CSV)은 Drive `MyDrive/dcc_runs/` 아래에만 있다.
