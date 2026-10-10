# Mission 3 최종 학습 키트 (도서관 PC, RTX 5060 Ti 16GB)

M3-ROBERTA-003 설정(klue/roberta-large + soft-F1)으로 **Training 100%**를 학습합니다. 결과물은 세 가지입니다.
1. 제출용 가중치 `.pt` (seed별, 인터넷 없이 추론 가능)
2. 공식 Validation 점수 (평가에만 사용)
3. **학습 로그가 담긴 실행된 ipynb** (필수 제출물)

| 파일 | 역할 |
|---|---|
| `m3_final_train.ipynb` | 실행할 노트북. 이 노트북을 실행하고 저장한 것이 학습 로그 제출물입니다 |
| `train_final.py` | 학습, 내보내기, 채점 코드 (노트북이 `%run`으로 부름) |
| `requirements-train.txt` | 패키지 버전 (torch는 따로 설치) |

예상 시간은 seed당 30~40분입니다(T4 78분의 2~2.5배 속도로 가정). 정확한 값은 3단계 속도 확인으로 정합니다.

## 0. 준비물 (Google Drive `MyDrive/대학부 데이터 압축해제/대학부 데이터/`)
- `Training/TL_서울_구급.zip` (54MB): Training 라벨
- `Validation/VL_서울_구급.zip` (6.7MB): 공식 Validation 라벨 (채점에만 사용)

음성(`TS_*`, `VS_*`)은 필요 없습니다. 두 zip을 `C:\dcc_data\Training\`, `C:\dcc_data\Validation\`에 풀면, 노트북의 기본 경로 `C:\dcc_data\Training\TL_서울_구급`와 `C:\dcc_data\Validation\VL_서울_구급`가 그대로 맞습니다.

## 1. 설치 (PowerShell, 관리자 권한 불필요)
```powershell
# Python 3.11~3.12가 없으면 python.org에서 "Install for current user"로 설치
git clone https://github.com/eigensinn-dcc2026/prelim.git dcc; cd dcc
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50 시리즈: cu128 이상 필수
pip install -r scripts/m3_final/requirements-train.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); torch.zeros(1,device='cuda').add_(1); print('GPU OK')"
```
- `GPU OK`가 나오지 않으면 "no kernel image" 오류일 가능성이 큽니다. `pip uninstall torch` 후 `cu129`나 `cu130` 인덱스로 다시 설치합니다.
- `Activate.ps1`이 막히면 `Set-ExecutionPolicy -Scope Process Bypass`를 먼저 실행합니다.

## 2. 경로 확인
노트북 1번 셀의 기본값이 0단계 경로와 같으면 고칠 것이 없습니다. 다르면 아래 환경 변수로 바꿉니다(노트북을 고치지 않아도 됨).
```powershell
$env:M3_TRAIN_DIR = "C:\dcc_data\Training\TL_서울_구급"
$env:M3_VAL_DIR   = "C:\dcc_data\Validation\VL_서울_구급"
$env:M3_SEEDS     = "42 43 44"     # 시간이 부족하면 "42"
$env:M3_RUN_QUICK = "1"            # 512행 1 epoch 속도 확인을 먼저 실행
```
모델(`klue/roberta-large` revision `28d9112`, 1.35GB)은 처음 실행할 때 `model_cache/`에 자동으로 받습니다.

## 3. 실행
**Jupyter가 되면:** `pip install notebook` 후 `jupyter notebook`을 실행하고 `scripts/m3_final/m3_final_train.ipynb`의 셀을 위에서부터 실행합니다.

**Jupyter UI가 안 되면 (헤드리스):**
```powershell
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 scripts/m3_final/m3_final_train.ipynb
```
실행이 끝나면 같은 파일에 셀 출력(로그)이 저장됩니다.

**권장 순서:**
1. 먼저 `M3_RUN_QUICK=1`, `M3_SEEDS=42`로 한 번 실행합니다. 3번 셀의 epoch 시간으로 전체 시간을 추정합니다.
2. 그다음 `M3_RUN_QUICK=0`, `M3_SEEDS=42 43 44`로 본 실행을 합니다.

노트북 없이 스크립트만 돌릴 때(로그 ipynb는 남지 않음):
```powershell
python scripts/m3_final/train_final.py --train_dir C:\dcc_data\Training\TL_서울_구급 --out runs/m3_final --seeds 42 43 44 --val_dir C:\dcc_data\Validation\VL_서울_구급
```

## 4. 결과물 (`runs/m3_final/`)
- `m3_roberta003_full_seed<s>.pt`: 제출용 가중치. `submission/mission3/inference.py --ckpt_path`에 그대로 씁니다.
- `*.pt.json`: 크기, SHA256, 학습 기록
- `seed_<s>/val_score.json`: 공식 Val 점수와 추론 시간 (계산 효율 보고용)
- `seed_<s>/history.json`: epoch별 loss와 시간
- 노트북 5번 셀: seed별 점수와 seed 확률 평균의 점수

## 5. 끝나면 꼭
1. `.pt`, `.pt.json`, `seed_*/*.json`, **실행된 `m3_final_train.ipynb`**를 Drive나 USB로 옮깁니다.
2. 도서관 PC에서 `C:\dcc_data`, `model_cache`, `runs`, `.venv`, 클론한 폴더를 지웁니다(대회 데이터 반출 방지).

## 규정
- 입력은 `utterances[].text`만 씁니다.
- 판정은 `확률 ≥ 0.5`이고 후처리는 없습니다.
- 공식 Validation은 학습, 가중치 계산, 선택에 쓰지 않습니다(채점 전용).
- 9개 외 증상과 다른 라벨 필드는 쓰지 않습니다.
