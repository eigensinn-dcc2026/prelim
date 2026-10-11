# ④ 복구: 공식 Training의 압축 해제된 JSON 폴더 사용
import os
from pathlib import Path

DATA_ROOT = Path("/content/drive/MyDrive/대학부 데이터")
TRAIN_DIR = DATA_ROOT / "Training/2.라벨링데이터/TL_서울_구급"
assert TRAIN_DIR.is_dir(), "Training 폴더를 읽을 수 없습니다."
assert next(TRAIN_DIR.glob("*.json"), None), "JSON 파일이 없습니다."

source = SCRIPT_SOURCE
start = source.index("    from dcc_data.colab_env import find_data_root\n", source.index("def prepare_rows"))
end = source.index("    rows = []\n", start)
source = source[:start] + '    training_dir = Path(os.environ["M3_TRAIN_DIR"])\n' + source[end:]
source = source.replace(
    '(extracted / "TL_서울_구급").rglob("*.json")',
    'training_dir.rglob("*.json")',
)
compile(source, str(SCRIPT_PATH), "exec")
SCRIPT_PATH.write_text(source, encoding="utf-8")
os.environ["M3_TRAIN_DIR"] = str(TRAIN_DIR)

PERSONAL_RUN_ROOT = Path("/content/drive/MyDrive/dcc_runs_me")
LAB_OUT = PERSONAL_RUN_ROOT / "lab/M3-KCBERT-CORR-001-seed42"
LAB_WORK = Path("/content/m3_kcbert_corr_work")
PERSONAL_RUN_ROOT.mkdir(parents=True, exist_ok=True)
LAB_WORK.mkdir(parents=True, exist_ok=True)
os.environ.update(
    DCC_DATA_ROOT=str(DATA_ROOT),
    DCC_RUN_ROOT=str(PERSONAL_RUN_ROOT),
    LAB_OUT=str(LAB_OUT),
    LAB_WORK=str(LAB_WORK),
    LAB_EXP_ID="M3-KCBERT-CORR-001",
)
print("④ 복구 완료. 다음은 ⑤ 작은 자동검사입니다.")
