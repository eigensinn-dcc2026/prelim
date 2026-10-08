"""Generator for notebooks/mission2_l2_retrain.ipynb (M2 L2 3-seed retrain, Colab GPU; PR #47 review).

Usage (from repo root): python notebooks/gen/make_l2_nb.py [notebook.ipynb]
Default: notebooks/mission2_l2_retrain.ipynb -- rewritten in place (keeps its notebook metadata), then validated
(JSON loads, nbformat 4, every code cell compiles, outputs empty).
"""
import json, sys
from pathlib import Path

NB_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "notebooks/mission2_l2_retrain.ipynb")
old = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# Mission 2 L2 재학습 노트북 (제출용 · 코드 = Drive 코드 zip, commit은 0번 셀에서 출력)

- 과제: 발화 1개(startAt~endAt 음성만)로 화자 분류 — 신고자(1) / 119대원(0), 지표 Accuracy
- 모델: `SpeakerCNNLarge` × torch-seed 1/2/3, 소프트보팅 앙상블 1파일 (`--ckpt_path` 1개 규칙)
  - **우선순위 1**: 3-seed 모두 `EPOCHS`까지 학습 → 3개 앙상블
  - **우선순위 2**: 시간이 모자라면 끝난 seed만(`SEEDS_DONE`) — 2개면 2개 앙상블, 1개면 그 단일 ckpt를 제출 ckpt로
- 데이터: **Training(서울)만 학습**, call_id 80/20 분할(`--seed 42`)로 에폭 선택. 공식 Validation은 **평가만**(backprop·에폭 선택 금지)
- 라벨: 제공된 speaker 라벨 그대로(`verify_mission2_raw_labels.py`로 대조). text 사용·라벨 수정·통화 단위 정보 사용 금지(#41, 운영진 10-01/10-05)
- 입력: 대상 발화 구간 음성의 스펙트로그램만. 발화 순서·통화 단위 정보는 모델 입력에 쓰지 않음 (슬라이스는 분석용)
- 팀 규칙: 0번 셀에서 Drive의 코드 zip을 `/content`로 복사·압축 해제하고 zip sha256 · commit · `scripts/`·`src/` 파일 목록을 출력한 뒤 그 코드로만 실행. 이번 실행 zip은 `MyDrive/dcc2026-prelim-main-6cc1181.zip`이고, sha256이 `CODE_ZIP_SHA256_EXPECTED`와 다르면 그 셀에서 중단
- 경로: `dcc_data.colab_env`(`find_run_root`/`find_data_root`)로 자동 결정 — 코드 zip에 없으면 같은 기본값(`MyDrive/dcc_runs`)
- 출력 폴더: `MyDrive/dcc_runs/m2_l2_rerun_6cc1181/` (이번 실행 전용). 이전 실행 체크포인트가 이미 있으면 재사용하지 않고 중단. 이 폴더에서 끊긴 에폭은 기존 `--resume`으로 이어감
- 데이터는 Drive에서 **큰 파일로 통째로 복사**(작은 파일 수천 개를 Drive 마운트에서 읽지 않음)하고 파일마다 sha256 출력. 기본(`RUN_BUILD=False`)은 미리 만든 스펙트로그램(`MyDrive/dcc_runs/m2_l2_spec/`) 사용. `TRAIN_PACKED_SHA256`이 비어 있거나 Drive 파일과 다르면 학습을 시작하지 않음
- ckpt는 `OUT`(Drive)에 **에폭마다** 갱신되고, 에폭별 학습 로그·JSON은 이어 붙이기만 하므로 런타임이 끊겨도 에폭 기록이 남음
- AMP: A안 — `--amp`(float16 autocast + GradScaler) 유지, 가중치는 fp32. fp32 1에폭 기준값은 없음 (`FP32_EPOCH1_REF = None`)
- **채택 기준**: 공식 Val ≥ **87.26%** (L1 보험 제출본 84.26% + 3%p) **그리고** `SEEDS_DONE` 모든 seed의 에폭 로그 완비. 이전 91.55% ±0.3%p는 재현 참고용
""")

md("## 0. 코드 고정 (팀 규칙: Drive 코드 zip → /content 압축 해제 → sha256·commit·파일 목록 출력)")
code(r'''
from pathlib import Path
import os, sys, re, glob, json, hashlib, shutil, subprocess, time, zipfile

try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    print("Colab 밖에서 실행 중 — /content/drive 가 이미 있어야 함")

CODE_ZIP = Path("/content/drive/MyDrive/dcc2026-prelim-main-6cc1181.zip")   # git archive of 6cc1181
CODE_ZIP_SHA256_EXPECTED = "f1526b1df25fe9c9a5c985e01521df244955ceea5af256604000b4d717c92407"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def copy_big(src, dst_dir, expected=None):
    """Drive의 큰 파일 1개를 로컬로 통째로 복사하고 sha256을 출력·대조한다.
    expected가 있으면 그 값과, Drive에 '<이름>.sha256' 파일이 같이 있으면 그 값과도 맞아야 한다."""
    src, dst_dir = Path(src), Path(dst_dir)
    assert src.is_file(), f"Drive에 파일 없음: {src}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not (dst.exists() and dst.stat().st_size == src.stat().st_size):
        t0 = time.time(); shutil.copyfile(src, dst)
        print(f"copied {src} -> {dst} ({dst.stat().st_size / 1e9:.3f} GB, {time.time() - t0:.0f}s)")
    h = sha256(dst)
    print(f"  sha256 {h}  {dst.name}")
    if expected:
        assert h == expected, f"{dst.name} sha256 불일치: {h} != 기대값 {expected}"
        print("  = 기대값 일치")
    side = src.with_name(src.name + ".sha256")
    if side.exists():
        want = side.read_text(encoding="utf-8").split()[0]
        assert h == want, f"{dst.name} sha256 불일치: {h} != {side.name} {want}"
        print(f"  = {side.name} 일치")
    return dst, h

# 1) zip을 /content로 복사 + sha256. 기대값과 다르면 여기서 중단 (압축 해제·학습으로 가지 않음)
LOCAL_ZIP, CODE_ZIP_SHA256 = copy_big(CODE_ZIP, "/content", CODE_ZIP_SHA256_EXPECTED)
if CODE_ZIP_SHA256 != CODE_ZIP_SHA256_EXPECTED:
    raise RuntimeError(
        f"code zip sha256 불일치로 중단: computed {CODE_ZIP_SHA256} != expected {CODE_ZIP_SHA256_EXPECTED}"
    )

# 2) commit: git archive가 zip comment에 넣는 커밋 해시 → 없으면 zip 이름의 해시 (하드코딩 금지)
with zipfile.ZipFile(LOCAL_ZIP) as z:
    ZIP_COMMENT = z.comment.decode("ascii", "replace").strip()
    UNZ = Path("/content/_code_unzip"); shutil.rmtree(UNZ, ignore_errors=True)
    z.extractall(UNZ)
m_name = re.search(r"-([0-9a-f]{7,40})\.zip$", CODE_ZIP.name)
if re.fullmatch(r"[0-9a-f]{40}", ZIP_COMMENT):
    COMMIT_FULL, COMMIT_SOURCE = ZIP_COMMENT, "zip comment (git archive)"
    if m_name:
        assert COMMIT_FULL.startswith(m_name.group(1)), f"zip 이름 {m_name.group(1)} ≠ zip comment {COMMIT_FULL}"
elif m_name:
    COMMIT_FULL, COMMIT_SOURCE = m_name.group(1), "zip 이름 (zip comment 없음)"
else:
    raise RuntimeError(f"commit을 알 수 없음: zip comment {ZIP_COMMENT!r}, 이름 {CODE_ZIP.name}")
COMMIT = COMMIT_FULL[:7]

# 3) 압축 해제 → REPO (최상위 폴더가 정확히 하나면 그 폴더. 6cc1181 git archive는 dcc2026-prelim-main-6cc1181/)
top = list(UNZ.iterdir())
src_root = top[0] if len(top) == 1 and top[0].is_dir() else UNZ
print(f"zip top-level: {src_root.name if src_root != UNZ else '(zip root)'} ({len(top)} entries)")
REPO = Path(f"/content/dcc2026-prelim-main-{COMMIT}")
shutil.rmtree(REPO, ignore_errors=True)
shutil.move(str(src_root), str(REPO)); shutil.rmtree(UNZ, ignore_errors=True)
for need in ["scripts/train_mission2_cnn.py", "scripts/pack_mission2_l2_ensemble.py",
             "scripts/evaluate_mission2_l2_on_official_validation.py", "scripts/mission2_inference.py", "requirements.txt"]:
    assert (REPO / need).is_file(), f"코드 zip에 없음: {need}"

print(f"\ncode zip   : {CODE_ZIP} ({LOCAL_ZIP.stat().st_size:,} bytes)")
print(f"zip sha256 : {CODE_ZIP_SHA256}  (expected {CODE_ZIP_SHA256_EXPECTED})")
print(f"zip comment: {ZIP_COMMENT!r}")
print(f"commit     : {COMMIT_FULL}  ← {COMMIT_SOURCE}")
print(f"REPO       : {REPO}")
for sub in ["scripts", "src"]:
    files = sorted(str(p.relative_to(REPO)) for p in (REPO / sub).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    print(f"\n[{sub}/] {len(files)}개")
    for f in files: print("  ", f)

os.chdir(REPO); sys.path.insert(0, str(REPO / "src"))
''')

md("## 0-1. 설정 · 실행 헬퍼")
code(r'''
# ---- 경로 (팀원 공통: Drive 바로가기만 있으면 경로 수정 불필요 — dcc_data.colab_env, docs/team_colab_setup.md) ----
try:
    from dcc_data.colab_env import find_data_root, find_run_root   # DCC_RUN_ROOT > Colab MyDrive/dcc_runs
except ImportError:   # 코드 zip이 colab_env.py 이전 커밋(예: 4bfd2c4)이면 같은 기본값으로 대체
    find_data_root = None
    def find_run_root():
        return Path("/content/drive/MyDrive/dcc_runs")
DATA_ROOT_FALLBACK = Path("/content/drive/.shortcut-targets-by-id/1Ezl13Cpmt2lGgl-FqISYJWfop8omii6z/대학부 데이터")
RUN_ROOT   = find_run_root()
SPEC_DRIVE = RUN_ROOT / "m2_l2_spec"                             # 미리 만든 스펙트로그램 (run_l2_spec.sh / finish_train.sh 결과, STATUS.txt)
WORK = Path("/content/m2_work")                                  # 큰 파일 복사본·압축 해제한 개별 .npy (로컬 디스크만, Drive 금지)
OUT  = RUN_ROOT / "m2_l2_rerun_6cc1181"                          # 이번 Colab 실행 전용 (MyDrive/dcc_runs/m2_l2_rerun_6cc1181)
_out_norm = str(OUT).replace("\\", "/").rstrip("/")
if not _out_norm.endswith("MyDrive/dcc_runs/m2_l2_rerun_6cc1181"):
    raise RuntimeError(f"이 실행의 OUT은 MyDrive/dcc_runs/m2_l2_rerun_6cc1181 이어야 함 (현재 {OUT})")
if COMMIT != "6cc1181":
    raise RuntimeError(f"코드 zip commit {COMMIT_FULL} 이 6cc1181 이 아님 — 이 폴더에 학습하지 않음")
WORK.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run_log.txt"                                        # 모든 명령 출력 (이어 붙이기)

# ---- 미리 만든 스펙트로그램 (SPEC_DRIVE 안의 파일 이름) ----
# Training 묶음 파일 이름은 아래 두 개로 고정. sha256만 TRAIN_PACKED_SHA256 한 줄에 채우면 된다.
TRAIN_PACKED_NAME   = "m2_spectrograms_packed.npy"            # float16 묶음 (~7 GB)
TRAIN_INDEX_NAME    = "m2_spectrogram_index_packed.parquet"   # 묶음 인덱스 (frame_offset, label_source=verified_against_provided_json)
TRAIN_PACKED_GLOB   = "*packed*.npy"
TRAIN_INDEX_GLOB    = "*index*packed*.parquet"
# ← 실행 전 이 한 줄만 채운다. 값이 비어 있거나 Drive 파일 sha256과 다르면 학습을 시작하지 않는다.
TRAIN_PACKED_SHA256 = {"m2_spectrograms_packed.npy": "69a770c8100f6851c2f6a642a801dc1e0dd0a14691bc3b37f047ebf2ba4712b8", "m2_spectrogram_index_packed.parquet": "6afe3caacbf5916e291009849d4ff27d32ead3699b80c1a23ec038c4eda79aad"}

VAL_PACKED_NAME       = "m2_spectrograms_validation_packed.npy"
VAL_PACKED_SHA256     = "1fb278a7b9bd3db674b834685eaaf9f23cf57ee830fd30156eac18567991b613"
VAL_INDEX_NAME        = "m2_spectrogram_index_validation.parquet"         # npy_path 열 → 평가 스크립트(개별 .npy)용
VAL_PACKED_INDEX_NAME = "m2_spectrogram_index_validation_packed.parquet"  # frame_offset 열 → 묶음 파일 대조용
VAL_TAR_NAME          = "m2_spec_val_files.tar"                           # 1.88 GB, 개별 .npy 111,947개
# Drive m2_spec_val_files.tar.sha256 / STATUS.txt VAL_TAR_DONE (2026-10-06 08:41 KST) 값. 새로 tar를 만들면 갱신
VAL_TAR_SHA256        = "fd32b64974b80c9a6ee4108b9c13af0b569476cbc29cdfe052ed58aa63c28285"

# ---- 원본에서 다시 만들기 (RUN_BUILD=True) ----
# TODO: 원본 zip(Training/Validation 원천·라벨)이 Drive에 있으면 경로를 넣는다 → 큰 파일로 복사 후 /content에서 압축 해제.
RAW_ZIPS = []
ALLOW_DRIVE_SMALL_FILE_BUILD = False   # RAW_ZIPS 없이 Drive 폴더의 wav/json 수만 개를 직접 읽는 것 (느림·비권장, 명시적으로 켤 때만)

# ---- 학습 ----
SEEDS      = [1, 2, 3]   # 학습할 seed (우선순위 1: 3개 모두)
SEEDS_DONE = [1, 2, 3]   # EPOCHS를 끝까지 마치고 에폭 로그가 완비된 seed만. 우선순위 2: 끝난 seed만 남김 (예: [1, 2] 또는 [1])
EPOCHS     = 30
CHUNK      = 1           # --resume 1회당 에폭 수 = 1 → --model-output(OUT, Drive)의 ckpt가 에폭마다 갱신됨 (#43: 난수 상태까지 이어받음)
USE_AMP    = True        # A안: --amp (float16 autocast + GradScaler). 가중치는 fp32로 유지
RUN_BUILD  = False       # False: SPEC_DRIVE의 미리 만든 스펙트로그램 사용 / True: 원본 → 스펙트로그램 → 라벨 대조 → 묶기
RUN_TRAIN  = True
RUN_EVAL   = True
DEVICE     = "cuda"

# fp32 1에폭 로그가 없어 기준을 둘 수 없음. 기록된 것은 시드별 최종 held-out best 뿐 (0.9133/0.9134).
FP32_EPOCH1_REF = None
AMP_SUSPECT_HELD_OUT_BELOW = 0.903  # 최종 held-out best가 이 값 미만이면 AMP 의심 경고 (중단하지 않음)
AMP_SUSPECT_FLAGS = []

def train_packed_sha_expected(name):
    """딕셔너리 값이 비어 있으면 None. copy_big은 None이면 sha 단정을 건너뛰고, 학습 시작 게이트가 빈 값으로 중단한다."""
    if not isinstance(TRAIN_PACKED_SHA256, dict):
        return None
    val = TRAIN_PACKED_SHA256.get(name)
    if val is None:
        return None
    val = str(val).strip()
    return val or None

def _fsync_write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text); f.flush(); os.fsync(f.fileno())

# 이 폴더에 이전 실행 체크포인트가 있으면 재사용하지 않고 중단한다.
# 이 노트북이 이 폴더에 찍어 둔 run_identity / 에폭 로그가 있을 때만 에폭 단위 --resume 을 허용한다.
RUN_IDENTITY_PATH = OUT / "run_identity.json"
THIS_RUN = {
    "folder": "m2_l2_rerun_6cc1181",
    "commit": COMMIT_FULL,
    "code_zip": CODE_ZIP.name,
    "code_zip_sha256": CODE_ZIP_SHA256,
    "amp": USE_AMP,
}
def _existing_out_ckpts():
    return sorted(
        p for p in OUT.glob("m2_cnn_large_raw_*.pt")
        if p.is_file() and not p.name.endswith(".prev.pt")
    )
def _this_run_owns_out():
    if RUN_IDENTITY_PATH.is_file():
        try:
            prev = json.loads(RUN_IDENTITY_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prev = None
        if isinstance(prev, dict) and all(prev.get(k) == THIS_RUN[k] for k in ("folder", "commit", "code_zip_sha256", "amp")):
            return True
    for p in sorted(OUT.glob("train_ts*_epochs.jsonl")):
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        if '"run_folder": "m2_l2_rerun_6cc1181"' in text and CODE_ZIP_SHA256 in text:
            return True
    return False
def guard_fresh_output_folder():
    """이전 실행 체크포인트가 있으면 중단. 이 실행이 이 폴더에 남긴 표시가 있으면 에폭 단위 resume."""
    ckpts_here = _existing_out_ckpts()
    if ckpts_here and not _this_run_owns_out():
        raise RuntimeError(
            f"{OUT} 에 이전 실행 체크포인트가 있어 재사용하지 않고 중단: {[p.name for p in ckpts_here]}. "
            "이 폴더는 m2_l2_rerun_6cc1181 전용이다. 체크포인트를 옮긴 뒤 다시 실행. "
            "이 폴더에서 끊긴 에폭을 이어 학습하는 것은 이 실행의 run_identity.json 또는 에폭 로그가 있을 때만 한다."
        )
    _fsync_write(RUN_IDENTITY_PATH, json.dumps(THIS_RUN, ensure_ascii=False, indent=2) + "\n")
    print("fresh OUT" if not ckpts_here else "이 폴더에서 에폭 단위 resume", OUT, [p.name for p in ckpts_here])
    return ckpts_here

guard_fresh_output_folder()

CHILD_ENV = {**os.environ, "PYTHONUNBUFFERED": "1", "TQDM_MININTERVAL": "60"}

def _append(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text); f.flush(); os.fsync(f.fileno())

def run(cmd, also_log=None, on_line=None):
    """출력을 화면 + Drive 로그(LOG, also_log)에 줄마다 바로 남김 (Popen 스트리밍, PYTHONUNBUFFERED)."""
    cmd = [str(c) for c in cmd]
    line = "$ " + " ".join(cmd)
    print(line, flush=True)
    logs = [LOG] + ([also_log] if also_log else [])
    files = [open(p, "a", encoding="utf-8") for p in logs]
    try:
        for f in files: f.write(f"\n[{time.strftime('%F %T')}] {line}\n"); f.flush()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=CHILD_ENV)
        for l in p.stdout:
            print(l, end="", flush=True)
            for f in files: f.write(l); f.flush()
            if on_line: on_line(l)
        rc = p.wait()
        for f in files: f.write(f"[{time.strftime('%F %T')}] exit {rc}\n"); f.flush(); os.fsync(f.fileno())
    finally:
        for f in files: f.close()
    if rc != 0:
        raise RuntimeError(f"exit {rc}: {line}")

_append(LOG, f"\n[{time.strftime('%F %T')}] === M2 L2 rerun session · commit {COMMIT_FULL} ({COMMIT_SOURCE}) · "
             f"code zip {CODE_ZIP.name} sha256 {CODE_ZIP_SHA256} · AMP {USE_AMP} · RUN_BUILD {RUN_BUILD} ===\n")
print("OUT =", OUT)
''')

md("## 1. 환경 기록")
code(r'''
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
import torch, numpy as np, pandas as pd
print("commit", COMMIT_FULL, "| torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")

def query_nvidia_smi():
    """GPU 이름·드라이버 버전. 실패해도 예외를 올리지 않고 error 문자열로 남긴다."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as exc:
        return {"gpu_name": None, "driver_version": None, "error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"nvidia-smi exit {proc.returncode}"
        return {"gpu_name": None, "driver_version": None, "error": err}
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if not lines:
        return {"gpu_name": None, "driver_version": None, "error": "nvidia-smi produced no output"}
    gpu_name, sep, driver = lines[0].rpartition(",")
    if not sep:
        return {"gpu_name": None, "driver_version": None, "error": f"nvidia-smi parse failed: {lines[0]!r}"}
    return {"gpu_name": gpu_name.strip(), "driver_version": driver.strip(), "error": None}

def write_env_json():
    smi = query_nvidia_smi()
    env = {
        "commit": COMMIT_FULL,
        "commit_source": COMMIT_SOURCE,
        "code_zip": CODE_ZIP.name,
        "code_zip_sha256": CODE_ZIP_SHA256,
        "torch": torch.__version__,
        "torch.version.cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "nvidia_smi_gpu_name": smi["gpu_name"],
        "nvidia_smi_driver_version": smi["driver_version"],
        "nvidia_smi_error": smi["error"],
    }
    path = OUT / "env.json"
    path.write_text(json.dumps(env, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("env.json", path)
    print(json.dumps(env, ensure_ascii=False, indent=2))
    return env

write_env_json()
''')

md("""
## 2. 데이터 — 기본(`RUN_BUILD=False`): Drive의 미리 만든 스펙트로그램을 큰 파일로 복사

- Training 묶음 `.npy` + 인덱스 parquet, Val 묶음 `.npy` + 인덱스 parquet 2개, Val 개별 파일 tar를 `/content`로 통째로 복사하고 sha256 출력
- 평가 스크립트는 발화별 `.npy`를 읽으므로 Val tar는 `/content`에 풀어서 씀 (Drive 마운트에서 작은 파일을 읽지 않음)
""")
code(r'''
DATA_SHAS = {}

def pick(name, pattern):
    """이름이 있고 파일이 있으면 그것, 아니면 glob(validation·.sha256 제외)으로 정확히 1개."""
    if name and (SPEC_DRIVE / name).is_file():
        return SPEC_DRIVE / name
    cands = [Path(x) for x in sorted(glob.glob(str(SPEC_DRIVE / pattern)))
             if "validation" not in Path(x).name and not x.endswith(".sha256")]
    assert len(cands) == 1, f"{SPEC_DRIVE}/{pattern}: 정확히 1개여야 함 (이름 {name!r} 없음) → {len(cands)}개 {cands}"
    print(f"[TODO 확인] {name!r} 없음 → glob {pattern} 으로 {cands[0].name} 사용")
    return cands[0]

if not RUN_BUILD:
    status = (SPEC_DRIVE / "STATUS.txt").read_text(encoding="utf-8") if (SPEC_DRIVE / "STATUS.txt").exists() else ""
    print("--- SPEC_DRIVE/STATUS.txt (요약) ---")
    for l in status.splitlines():
        if re.search(r"VERIFY|VAL_DONE|VAL_TAR_DONE|TRAIN_BUILD_DONE|TRAIN_PACK_DONE|ALL_DONE|FAILED", l): print(" ", l)
    if "ALL_DONE" not in status:
        print("경고: STATUS.txt에 ALL_DONE 없음 — Training 묶음이 아직 끝나지 않았을 수 있음")

    train_packed_src = pick(TRAIN_PACKED_NAME, TRAIN_PACKED_GLOB)
    train_index_src = pick(TRAIN_INDEX_NAME, TRAIN_INDEX_GLOB)
    assert train_packed_src.name == TRAIN_PACKED_NAME, train_packed_src
    assert train_index_src.name == TRAIN_INDEX_NAME, train_index_src
    PACKED,     DATA_SHAS["train_packed"]     = copy_big(train_packed_src, WORK, train_packed_sha_expected(TRAIN_PACKED_NAME))
    PACKED_IDX, DATA_SHAS["train_index"]      = copy_big(train_index_src, WORK, train_packed_sha_expected(TRAIN_INDEX_NAME))
    VAL_PACKED, DATA_SHAS["val_packed"]       = copy_big(SPEC_DRIVE / VAL_PACKED_NAME, WORK, VAL_PACKED_SHA256)
    VA_IDX,     DATA_SHAS["val_index"]        = copy_big(SPEC_DRIVE / VAL_INDEX_NAME, WORK)
    VA_PACKED_IDX, DATA_SHAS["val_packed_index"] = copy_big(SPEC_DRIVE / VAL_PACKED_INDEX_NAME, WORK)
    VAL_TAR,    DATA_SHAS["val_files_tar"]    = copy_big(SPEC_DRIVE / VAL_TAR_NAME, WORK, VAL_TAR_SHA256)

    # Val 개별 .npy: tar를 로컬에서 풀고, 인덱스의 npy_path가 가리키는 기준 폴더를 찾음
    VA_X = WORK / "spec_val_files"
    if not (VA_X / ".extracted").exists():
        shutil.rmtree(VA_X, ignore_errors=True); VA_X.mkdir(parents=True)
        run(["tar", "-xf", VAL_TAR, "-C", VA_X])
        (VA_X / ".extracted").write_text(DATA_SHAS["val_files_tar"])
    first = pd.read_parquet(VA_IDX)["npy_path"].iloc[0]
    roots = [VA_X] + [d for d in VA_X.iterdir() if d.is_dir()] + [d for d in VA_X.glob("*/*") if d.is_dir()]
    VA_DIR = next((r for r in roots if (r / first).is_file()), None)
    assert VA_DIR is not None, f"tar 안에서 {first} 를 찾지 못함 (풀린 위치 {VA_X})"
    print("Val 개별 .npy 기준 폴더:", VA_DIR)
''')

md("## 2-B. (선택) 원본에서 다시 만들기 — `RUN_BUILD=True` (원본 zip을 큰 파일로 복사 → 스펙트로그램 → 제공 라벨 대조 → 묶기)")
code(r'''
def label_root(manifest_csv, split, folder_re, data_root):
    # manifest의 json_path에서 제공 라벨 폴더(TL_서울_구급 / VL_서울_구급)를 찾음 (run_l2_spec.sh와 같은 방식)
    import csv, unicodedata
    from pathlib import PurePosixPath
    dirs = set()
    for r in csv.DictReader(open(manifest_csv, encoding="utf-8")):
        if r["split"] != split: continue
        parts = PurePosixPath(r["json_path"]).parts
        for i, p in enumerate(parts):
            if re.fullmatch(folder_re, unicodedata.normalize("NFC", p)):
                dirs.add("/".join(parts[:i + 1])); break
    assert dirs, f"{split} 라벨 폴더({folder_re})를 manifest에서 찾지 못함"
    return Path(data_root) / os.path.commonpath(sorted(dirs))

if RUN_BUILD:
    if RAW_ZIPS:
        BUILD_ROOT = WORK / "raw"
        for z in RAW_ZIPS:   # 큰 파일 1개씩 복사 → 로컬에서 압축 해제 (TODO: 한글 파일명 인코딩이 깨지면 unzip -O cp949)
            zp, DATA_SHAS[f"raw:{Path(z).name}"] = copy_big(z, WORK / "raw_zips")
            run(["unzip", "-q", "-o", zp, "-d", BUILD_ROOT])
        MANIFEST = WORK / "dcc_manifest_v1.csv"
        run([sys.executable, "scripts/build_manifest.py", "--data-root", BUILD_ROOT, "--output", MANIFEST, "--overwrite"])
    else:
        assert ALLOW_DRIVE_SMALL_FILE_BUILD, ("RAW_ZIPS가 비어 있음: 원본 zip 경로를 넣거나(권장), Drive 폴더를 직접 읽으려면 "
                                              "ALLOW_DRIVE_SMALL_FILE_BUILD=True (작은 파일 수만 개를 Drive에서 읽음, 느림)")
        BUILD_ROOT = find_data_root() if find_data_root else DATA_ROOT_FALLBACK   # DCC_DATA_ROOT > Drive '대학부 데이터' 바로가기 자동 탐색
        MANIFEST, DATA_SHAS["manifest"] = copy_big(RUN_ROOT / "manifests/dcc_manifest_v1.csv", WORK)
    os.environ["DCC_DATA_ROOT"] = str(BUILD_ROOT); CHILD_ENV["DCC_DATA_ROOT"] = str(BUILD_ROOT)
    n_rows = len(pd.read_csv(MANIFEST)); print("manifest rows", n_rows); assert n_rows == 32840, n_rows

    # 2-B-1. 스펙트로그램 (mel 40 / hop 160 / FFT 512 / 8kHz, 발화 구간만) — 개별 .npy는 로컬(WORK)에만
    VA_DIR, VA_IDX = WORK / "spec_val",   WORK / "m2_spectrogram_index_validation.parquet"
    TR_DIR, TR_IDX = WORK / "spec_train", WORK / "m2_spectrogram_index_full.parquet"
    for split, d, idx in [("validation", VA_DIR, VA_IDX), ("training", TR_DIR, TR_IDX)]:
        run([sys.executable, "scripts/build_mission2_spectrograms.py", "--manifest", MANIFEST, "--data-root", BUILD_ROOT,
             "--split", split, "--n-calls", "-1", "--seed", "42", "--workers", os.cpu_count(),
             "--output-dir", d, "--index-output", idx, "--failures-log", OUT / f"spec_{split}_failures.jsonl"])

    # 2-B-2. 제공 라벨 대조 (라벨 수정 없음, 새로 만든 인덱스는 불일치 0행이 정상)
    run([sys.executable, "scripts/verify_mission2_raw_labels.py", "--index", VA_IDX,
         "--label-root", label_root(MANIFEST, "Validation", r"VL_서울_구급", BUILD_ROOT)])
    TR_IDX_PROVIDED = WORK / "m2_spectrogram_index_full_provided_labels.parquet"
    run([sys.executable, "scripts/verify_mission2_raw_labels.py", "--index", TR_IDX,
         "--label-root", label_root(MANIFEST, "Training", r"TL_서울_구급", BUILD_ROOT), "--output", TR_IDX_PROVIDED])

    # 2-B-3. 묶기 (float16 1파일 ~7GB) → 로컬에 쓰고, Drive에는 큰 파일로 사본
    PACKED, PACKED_IDX = WORK / "m2_spectrograms_packed.npy", WORK / "m2_spectrogram_index_packed.parquet"
    run([sys.executable, "scripts/pack_mission2_spectrograms.py", "--index", TR_IDX_PROVIDED,
         "--spectrogram-dir", TR_DIR, "--packed-output", PACKED, "--index-output", PACKED_IDX])
    VAL_PACKED = None
    for p in [PACKED, PACKED_IDX, VA_IDX]:
        h = sha256(p); DATA_SHAS[f"built:{p.name}"] = h; print(f"sha256 {h}  {p.name}")
        (OUT / "spec_build").mkdir(exist_ok=True); shutil.copyfile(p, OUT / "spec_build" / p.name)
''')

md("## 2-C. 데이터 점검 (두 경로 공통)")
code(r'''
from dcc_data.mission2_spectrogram import assert_training_spectrogram_index
tr, va = pd.read_parquet(PACKED_IDX), pd.read_parquet(VA_IDX)
prov = assert_training_spectrogram_index(tr)          # Training 출처·제공 라벨 표시가 없으면 멈춤 (학습 스크립트도 다시 확인)
print("train index provenance:", prov)
print("train 발화", len(tr), "통화", tr.call_id.nunique(), "| val 발화", len(va), "통화", va.call_id.nunique())
# 기대: train 약 889,358발화 / 29,200통화, val 111,947발화 / 3,640통화
assert set(va["split"].astype(str)) == {"Validation"}, va["split"].unique()
assert not (set(tr.call_id) & set(va.call_id)), "Training 학습 인덱스에 공식 Validation 통화가 섞임"
assert len(va) == 111947 and va.call_id.nunique() == 3640, (len(va), va.call_id.nunique())
missing = [r for r in va["npy_path"] if not (VA_DIR / r).is_file()]
assert not missing, f"Val .npy {len(missing)}개 없음, 예: {missing[:3]}"
print(f"Val 개별 .npy {len(va):,}개 모두 있음:", VA_DIR)

if VAL_PACKED is not None:   # 묶음 파일과 개별 파일이 같은 빌드인지 표본 대조 (float16 반올림 허용)
    vp = pd.read_parquet(VA_PACKED_IDX); assert len(vp) == len(va)
    arr = np.load(VAL_PACKED, mmap_mode="r")
    sample = vp[["call_id", "utt_idx", "frame_offset", "n_frames"]].sample(200, random_state=0).merge(
        va[["call_id", "utt_idx", "npy_path"]], on=["call_id", "utt_idx"])
    assert len(sample) == 200
    for r in sample.itertuples():
        a = np.asarray(arr[r.frame_offset:r.frame_offset + r.n_frames]).T.astype(np.float32)
        b = np.load(VA_DIR / r.npy_path)
        assert a.shape == b.shape and np.allclose(a, b, rtol=1e-3, atol=0.1), (r.call_id, r.utt_idx)
    print("Val 묶음 ↔ 개별 .npy 표본 200개 일치")
print("DATA_SHAS =", json.dumps(DATA_SHAS, indent=1))
''')

md("""
## 3. 시드별 학습 — ckpt는 에폭마다 Drive에 갱신 · 에폭 로그는 이어 붙이기만

- `CHUNK=1`: `train_mission2_cnn.py`를 1에폭씩 `--resume`으로 부르고 `--model-output`이 `OUT`(Drive)이므로, **에폭이 끝날 때마다 Drive의 ckpt가 갱신**됨
- 호출 직전 ckpt를 `*.prev.pt`로 복사 (쓰는 도중 끊겨 깨지면 이걸로 되돌림)
- seed별로 Drive에 이어 붙이기만 하는 기록 3개 (덮어쓰지 않음):
  `train_ts{s}_log.txt`(학습 출력 전체) · `train_ts{s}_epochs.jsonl`(에폭별 train_loss/val_acc/lr/시간) · `train_ts{s}_ckpt_history.jsonl`(호출마다 ckpt 에폭 수·best·sha256)
- 에폭 선택은 Training 내부 held-out만 사용
- AMP(A안): seed 1의 1에폭에서 loss/metrics에 NaN/inf가 없고 loss가 감소해야 다음 에폭·시드로 진행. 에폭당 train_loss가 하나이고 fp32 1에폭 로그가 없어, 감소는 `train_loss < ln(2)`(균등 2-class 초기 CE)로 본다. 게이트 결과는 `amp_epoch1_gate.json`에 train_loss, ln2, held_out_acc(트레이너 에폭 로그의 val_acc), pass로 남긴다. 학습이 끝난 뒤 held-out best < 0.903이면 AMP 의심 경고만 하고 중단하지 않음
""")
code(r'''
def ckpt(s):          return OUT / f"m2_cnn_large_raw_ts{s}.pt"
def ckpt_prev(s):     return OUT / f"m2_cnn_large_raw_ts{s}.prev.pt"
def train_log(s):     return OUT / f"train_ts{s}_log.txt"
def epoch_jsonl(s):   return OUT / f"train_ts{s}_epochs.jsonl"
def ckpt_hist(s):     return OUT / f"train_ts{s}_ckpt_history.jsonl"

def load_ckpt(s):
    try:
        return torch.load(ckpt(s), map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"{ckpt(s)} 를 읽지 못함 ({e}). 쓰는 도중 끊겼다면 {ckpt_prev(s).name} 를 {ckpt(s).name} 로 복사 후 다시 실행") from e

def done_epochs(s):
    if not ckpt(s).exists(): return 0
    c = load_ckpt(s)
    amp = c.get("training_options", {}).get("amp")
    assert amp == USE_AMP, f"seed {s} ckpt는 amp={amp}인데 USE_AMP={USE_AMP} — 섞어서 이어 학습하지 않음 (ckpt를 옮기고 처음부터)"
    return int(c["total_epochs_trained"])   # train_mission2_cnn.py가 저장하는 키

EPOCH_RE = re.compile(r"^epoch\s+(\d+)\s+train_loss=(\S+)\s+val_acc=(\S+)\s+lr=(\S+)\s+time=(\d+)s")
def epoch_logger(seed):
    def on_line(l):
        m = EPOCH_RE.match(l.strip())
        if m:
            rec = {"seed": seed, "epoch": int(m[1]), "train_loss": float(m[2]), "val_acc_internal": float(m[3]),
                   "lr": float(m[4]), "epoch_time_s": int(m[5]), "amp": USE_AMP, "commit": COMMIT,
                   "run_folder": "m2_l2_rerun_6cc1181", "code_zip_sha256": CODE_ZIP_SHA256,
                   "logged_at": time.strftime("%F %T")}
            _append(epoch_jsonl(seed), json.dumps(rec) + "\n")
    return on_line

def epoch_log_status(s):
    # 마지막 fresh_start 이후 기록만 봄. 같은 에폭이 두 번이면(저장 전에 끊겨 다시 돈 경우) 마지막 기록이 ckpt에 반영된 것
    recs = [json.loads(l) for l in open(epoch_jsonl(s), encoding="utf-8")] if epoch_jsonl(s).exists() else []
    starts = [i for i, r in enumerate(recs) if r.get("event") == "fresh_start"]
    recs = [r for r in recs[(starts[-1] + 1 if starts else 0):] if "epoch" in r]
    by_epoch = {r["epoch"]: r for r in recs}
    miss = sorted(set(range(1, EPOCHS + 1)) - set(by_epoch))
    return {"complete": not miss, "missing": miss, "n_records": len(recs), "by_epoch": by_epoch}

def train(seed, n_epochs, resume):
    if resume:
        shutil.copyfile(ckpt(seed), ckpt_prev(seed))
    else:
        _append(epoch_jsonl(seed), json.dumps({"event": "fresh_start", "seed": seed, "amp": USE_AMP, "commit": COMMIT,
                                               "run_folder": "m2_l2_rerun_6cc1181", "code_zip_sha256": CODE_ZIP_SHA256,
                                               "at": time.strftime("%F %T")}) + "\n")
    cmd = [sys.executable, "scripts/train_mission2_cnn.py",
           "--index", PACKED_IDX, "--packed-spectrograms", PACKED,
           "--model", "large", "--seed", "42", "--torch-seed", seed, "--test-size", "0.2",
           "--epochs", n_epochs, "--batch-size", "64", "--lr", "1e-3", "--lr-patience", "2",
           "--device", DEVICE, "--num-workers", "2", "--model-output", ckpt(seed)]
    if USE_AMP: cmd.append("--amp")
    if resume:  cmd.append("--resume")
    run(cmd, also_log=train_log(seed), on_line=epoch_logger(seed))
    c = load_ckpt(seed)
    rec = {"seed": seed, "total_epochs_trained": int(c["total_epochs_trained"]), "best_epoch": c["best_epoch"],
           "best_val_acc_internal": c["best_val_acc"], "sha256": sha256(ckpt(seed)), "saved_at": time.strftime("%F %T")}
    _append(ckpt_hist(seed), json.dumps(rec) + "\n")
    print(f"[Drive ckpt 갱신] seed {seed}: {rec['total_epochs_trained']}/{EPOCHS} 에폭 → {ckpt(seed)}")

def require_train_packed_sha256_before_training():
    """TRAIN_PACKED_SHA256 이 비어 있거나 Drive에서 복사한 파일 sha256과 다르면 학습을 시작하지 않는다."""
    if not RUN_TRAIN or RUN_BUILD:
        return
    if not isinstance(TRAIN_PACKED_SHA256, dict):
        raise RuntimeError("TRAIN_PACKED_SHA256 가 딕셔너리가 아니어서 학습을 시작하지 않음")
    got = {
        TRAIN_PACKED_NAME: DATA_SHAS.get("train_packed"),
        TRAIN_INDEX_NAME: DATA_SHAS.get("train_index"),
    }
    for name in (TRAIN_PACKED_NAME, TRAIN_INDEX_NAME):
        expected = TRAIN_PACKED_SHA256.get(name, "")
        expected = "" if expected is None else str(expected).strip()
        actual = got.get(name)
        if expected == "":
            raise RuntimeError(
                f"TRAIN_PACKED_SHA256[{name!r}] 가 비어 있어 학습을 시작하지 않음. "
                f"복사된 Drive 파일 sha256={actual}. TRAIN_PACKED_SHA256 이 한 줄만 채운 뒤 다시 실행"
            )
        if actual != expected:
            raise RuntimeError(
                f"{name} sha256이 Drive 파일과 달라 학습을 시작하지 않음: "
                f"computed {actual} != TRAIN_PACKED_SHA256 {expected}"
            )
    print("TRAIN_PACKED_SHA256 확인: Training 묶음 두 파일 모두 기대값과 일치")

def _json_number(value):
    """Finite numbers stay numbers. NaN/inf and anything else become None so the gate JSON is strict."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return int(value) if isinstance(value, int) else float(value)

def assert_amp_seed1_epoch1():
    """A안 seed-1 1에폭 게이트. 실패하면 이후 에폭·시드로 진행하지 않는다.

    트레이너는 에폭당 train_loss와 val_acc(내부 held-out)를 한 줄로 찍는다. fp32 1에폭 로그는 없다.
    균등 2-class 초기 cross-entropy는 ln(2)이므로, 1에폭 평균 train_loss < ln(2)이면 그 에폭 동안 loss가 감소한 것이다.
    판정은 OUT/amp_epoch1_gate.json 에 train_loss, ln2, held_out_acc, pass 로 남긴다.
    held_out_acc 는 에폭 로그의 val_acc_internal (트레이너가 찍는 val_acc)이다.
    """
    if not USE_AMP:
        return
    e1 = epoch_log_status(1)["by_epoch"].get(1)
    init_ce = float(np.log(2.0))
    print("AMP seed1 1에폭:", e1, "| FP32_EPOCH1_REF:", FP32_EPOCH1_REF)
    train_loss = _json_number(e1.get("train_loss")) if e1 else None
    # Trainer epoch line: val_acc=<internal held-out>. The notebook stores that as val_acc_internal.
    held_out_acc = _json_number(e1.get("val_acc_internal")) if e1 else None
    lr = _json_number(e1.get("lr")) if e1 else None
    reason = None
    if e1 is None:
        reason = "AMP seed1 1에폭 로그가 없음"
    elif train_loss is None or held_out_acc is None or lr is None or not (0.0 <= held_out_acc <= 1.0):
        reason = f"loss/metrics NaN/inf (또는 val_acc 범위 밖): {e1}"
    elif not (train_loss < init_ce):
        reason = f"train_loss={train_loss} 가 초기 손실 ln(2)={init_ce:.6f}보다 낮지 않음 (loss가 감소하지 않음)"
    passed = reason is None
    verdict = {
        "seed": 1,
        "epoch": 1,
        "train_loss": train_loss,
        "ln2": init_ce,
        "held_out_acc": held_out_acc,
        "pass": passed,
        "fail_reason": reason,
    }
    if held_out_acc is None:
        verdict["held_out_acc_note"] = (
            "epoch-1 로그에 유한한 held-out accuracy가 없음. "
            "트레이너 에폭 줄의 val_acc 를 val_acc_internal 로 읽는다."
        )
        verdict["epoch1_log_available"] = (
            {k: _json_number(v) if isinstance(v, (int, float)) else v for k, v in e1.items()} if e1 else {}
        )
    gate_path = OUT / "amp_epoch1_gate.json"
    gate_path.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    line = (f"AMP seed1 1에폭 게이트 {'통과' if passed else '실패'}: "
            f"train_loss={train_loss} ln2={init_ce:.6f} held_out_acc={held_out_acc} pass={passed}")
    print(line)
    _append(LOG, f"[{time.strftime('%F %T')}] {line} | {gate_path.name}\n")
    if not passed:
        raise RuntimeError(f"AMP seed1 1에폭 게이트 실패 → 이후 에폭·시드로 진행하지 않음: {reason}")

# 3-1. AMP 점검: seed 1을 1에폭만 돌리고, 실패하면 여기서 멈춘다 (다음 셀의 나머지 에폭·시드로 가지 않음)
if RUN_TRAIN:
    require_train_packed_sha256_before_training()
    if USE_AMP and done_epochs(1) < 1:
        train(1, 1, resume=False)
    if USE_AMP:
        assert_amp_seed1_epoch1()
''')
code(r'''
# 3-2. 전체 학습: 1에폭씩 --resume (런타임이 끊기면 0번·0-1·1·2·2-C 셀 실행 후 이 셀만 다시 실행 → 이 폴더의 Drive ckpt에서 이어감)
if RUN_TRAIN:
    require_train_packed_sha256_before_training()
    if USE_AMP and done_epochs(1) < 1:
        train(1, 1, resume=False)
    if USE_AMP:
        assert_amp_seed1_epoch1()   # 실패하면 이후 에폭·시드로 진행하지 않음
    suspect = []
    for s in SEEDS:
        while (e := done_epochs(s)) < EPOCHS:
            train(s, min(CHUNK, EPOCHS - e), resume=e > 0)
        st = epoch_log_status(s)
        print(f"seed {s} 완료 {EPOCHS}에폭 · sha256 {sha256(ckpt(s))} · 에폭 로그 "
              + ("완비" if st["complete"] else f"누락 {st['missing']}"))
        best = float(load_ckpt(s)["best_val_acc"])
        if USE_AMP and best < AMP_SUSPECT_HELD_OUT_BELOW:
            msg = (f"WARNING AMP_SUSPECT seed {s}: held-out best {best:.4f} < {AMP_SUSPECT_HELD_OUT_BELOW} "
                   f"(기록된 fp32 최종 held-out best는 0.9133/0.9134). AMP가 의심됨. 학습은 중단하지 않음.")
            print(msg)
            _append(LOG, f"[{time.strftime('%F %T')}] {msg}\n")
            suspect.append({"flag": "AMP_SUSPECT", "seed": s, "held_out_best": best,
                            "threshold": AMP_SUSPECT_HELD_OUT_BELOW, "stopped": False})
    AMP_SUSPECT_FLAGS[:] = suspect
    json.dump(suspect, open(OUT / "amp_suspect_flags.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("AMP_SUSPECT flags:", suspect if suspect else "없음")
''')

md("""
## 3-B. 저장 체크포인트 dtype (가중치는 fp32)

각 seed 체크포인트를 `map_location='cpu'`로 읽는다. AMP autocast는 연산만 float16이고, 저장된 모델 state dict의 부동소수 텐서는 전부 `torch.float32`여야 한다.
BatchNorm `num_batches_tracked`만 정수 버퍼(가중치가 아님)라 예외로 둔다.
""")
code(r'''
def _iter_tensors(obj, prefix):
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_tensors(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _iter_tensors(v, f"{prefix}[{i}]")

def assert_ckpt_weights_fp32(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    named = [(key, blob[key]) for key in ("best_model_state_dict", "last_model_state_dict", "ensemble_model_state_dicts") if key in blob]
    assert named, f"{path} 에 모델 state dict 없음"
    n_fp32 = 0
    for key, sd in named:
        for name, tensor in _iter_tensors(sd, key):
            if tensor.dtype.is_floating_point:
                assert tensor.dtype == torch.float32, f"{path} {name} dtype={tensor.dtype} (torch.float32 여야 함)"
                n_fp32 += 1
            else:
                assert name.endswith("num_batches_tracked"), (
                    f"{path} {name} dtype={tensor.dtype} 는 fp32 가중치가 아님"
                )
    assert n_fp32 > 0, f"{path} state dict에 부동소수 텐서가 없음"
    print(f"fp32 OK {Path(path).name}: {n_fp32} float32 tensors")

for s in SEEDS:
    assert ckpt(s).is_file(), ckpt(s)
    assert_ckpt_weights_fp32(ckpt(s))
''')

md("## 4. `SEEDS_DONE` 확인 · 제출 ckpt (2개 이상 → 앙상블 1파일, 1개 → 그 단일 ckpt)")
code(r'''
from dcc_data.mission2_checkpoint import resolve_l2_checkpoint_metadata, require_l2_inference_config, l2_member_state_dicts

assert SEEDS_DONE and len(set(SEEDS_DONE)) == len(SEEDS_DONE), SEEDS_DONE
SEEDS_DONE = sorted(SEEDS_DONE)
DONE_INFO = {}
for s in SEEDS_DONE:
    c = load_ckpt(s)
    assert int(c["total_epochs_trained"]) == EPOCHS, \
        f"seed {s}: ckpt는 {c['total_epochs_trained']}에폭 (EPOCHS={EPOCHS}) — 학습을 마치거나 SEEDS_DONE에서 뺄 것"
    st = epoch_log_status(s)
    DONE_INFO[s] = {"epochs_trained": int(c["total_epochs_trained"]), "best_epoch": c["best_epoch"],
                    "best_val_acc_internal": c["best_val_acc"], "amp": c.get("training_options", {}).get("amp"),
                    "epoch_log_complete": st["complete"], "epoch_log_missing": st["missing"], "sha256": sha256(ckpt(s))}
    print(f"seed {s}: {DONE_INFO[s]}")

PATH_LABEL = "우선순위 1 (3-seed 재학습본)" if len(SEEDS_DONE) == 3 else "우선순위 2 (완료 seed만)"
if len(SEEDS_DONE) >= 2:   # pack_mission2_l2_ensemble.py는 2개 이상만 받음
    SUBMIT = OUT / f"m2_cnn_large_raw_ensemble_ts{''.join(map(str, SEEDS_DONE))}.pt"
    run([sys.executable, "scripts/pack_mission2_l2_ensemble.py"] +
        sum([["--ckpt-path", ckpt(s)] for s in SEEDS_DONE], []) + ["--output", SUBMIT])
    SUBMIT_KIND = f"{len(SEEDS_DONE)}-seed 소프트보팅 앙상블"
else:                      # 1개: 묶지 않고 그 ckpt가 제출 ckpt
    SUBMIT = ckpt(SEEDS_DONE[0])
    SUBMIT_KIND = "단일 ckpt (1개라 앙상블 묶기 생략)"

# 제출 스크립트(mission2_inference.py)와 같은 로더로 확인: 설정 일치·멤버 수
_c = torch.load(SUBMIT, map_location="cpu", weights_only=False)
_meta, _ = resolve_l2_checkpoint_metadata(_c, SUBMIT); require_l2_inference_config(_meta)
assert len(l2_member_state_dicts(_c, _meta)) == len(SEEDS_DONE)

SHAS = {f"ts{s}": DONE_INFO[s]["sha256"] for s in SEEDS_DONE} | {"submission": sha256(SUBMIT)}
json.dump({"commit": COMMIT_FULL, "amp": USE_AMP, "path": PATH_LABEL, "seeds_done": SEEDS_DONE,
           "submission_kind": SUBMIT_KIND, "submission_ckpt": str(SUBMIT), "sha256": SHAS},
          open(OUT / "ckpt_sha256.json", "w"), indent=2, ensure_ascii=False)
print(PATH_LABEL, "|", SUBMIT_KIND, "|", SUBMIT); SHAS
''')

md("""
## 5. 공식 Validation (평가만)

- (a) `evaluate_mission2_l2_on_official_validation.py`: 완료 seed별 단독 평가 (배치 패딩, 팀 기존 JSON 형식). 이 스크립트는 앙상블 파일을 받지 않음
- (b) **제출 방식 점수** (판정에 사용): 제출 스크립트 `mission2_inference.py`와 같은 처리 — 같은 로더(설정 확인·멤버 가중치), 발화 하나씩 300프레임 패딩, softmax 평균 ≥ 0.5 → 신고자. 입력은 Val 개별 `.npy`(원본 음성에서 같은 함수로 만든 스펙트로그램)라 Drive에서 음성을 다시 읽지 않음. 완료 seed 각각과 제출 ckpt를 채점
""")
code(r'''
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from dcc_data.mission2_cnn import SpeakerCNNLarge
sys.path.insert(0, str(REPO / "scripts"))
from mission2_inference import pad_spectrogram_to_min_frames   # 제출 스크립트의 발화별 패딩

@torch.no_grad()
def score_submission_path(path, name):
    c = torch.load(path, map_location="cpu", weights_only=False)
    meta, _ = resolve_l2_checkpoint_metadata(c, path); require_l2_inference_config(meta)
    models = []
    for sd in l2_member_state_dicts(c, meta):
        m = SpeakerCNNLarge(); m.load_state_dict(sd); models.append(m.eval().to(DEVICE))
    preds = np.empty(len(va), dtype=np.int64); t0 = time.time()
    for i, rel in enumerate(va["npy_path"]):
        x = torch.from_numpy(pad_spectrogram_to_min_frames(np.load(VA_DIR / rel))).unsqueeze(0).unsqueeze(0).to(DEVICE)
        if len(models) == 1:
            lg = models[0](x); preds[i] = int((lg[0, 1] >= lg[0, 0]).item())
        else:
            p = sum(F.softmax(m(x), dim=1) for m in models) / len(models); preds[i] = int((p[0, 1] >= p[0, 0]).item())
        if (i + 1) % 20000 == 0: print(f"  {name}: {i + 1}/{len(va)} ({time.time() - t0:.0f}s)", flush=True)
    y = va["speaker"].to_numpy(); cm = confusion_matrix(y, preds, labels=[0, 1])
    res = {"name": name, "checkpoint": str(path), "sha256": sha256(path), "n_models": len(models),
           "n_utterances": int(len(va)), "n_calls": int(va.call_id.nunique()), "accuracy": float((preds == y).mean()),
           "recall_daewon_0": float(cm[0, 0] / cm[0].sum()), "recall_singoja_1": float(cm[1, 1] / cm[1].sum()),
           "confusion_matrix": cm.tolist(), "method": "submission-equivalent (per-utterance pad 300, softmax mean)"}
    va[["call_id", "utt_idx", "speaker"]].assign(predicted=preds).to_csv(OUT / f"val_sub_{name}_pred.csv", index=False)
    json.dump(res, open(OUT / f"val_sub_{name}.json", "w"), indent=2, ensure_ascii=False)
    _append(LOG, f"[{time.strftime('%F %T')}] official Val (submission path) {name}: {res['accuracy']:.6f} sha256 {res['sha256']}\n")
    print(f"{name}: 공식 Val(제출 방식) {res['accuracy']:.4%}  recall0 {res['recall_daewon_0']:.4f} recall1 {res['recall_singoja_1']:.4f}")
    return res

RES_SCRIPT, RES_SUB = {}, {}
if RUN_EVAL:
    for s in SEEDS_DONE:   # (a) 단독 평가 스크립트
        log, pred = OUT / f"val_ts{s}.json", OUT / f"val_ts{s}_pred.csv"
        run([sys.executable, "scripts/evaluate_mission2_l2_on_official_validation.py", "--ckpt-path", ckpt(s),
             "--index", VA_IDX, "--spectrogram-dir", VA_DIR, "--model", "large", "--device", DEVICE,
             "--log-output", log, "--predictions-output", pred])
        RES_SCRIPT[f"ts{s}"] = json.load(open(log))
    for s in SEEDS_DONE:   # (b) 제출 방식: 완료 seed 각각 + 제출 ckpt
        RES_SUB[f"ts{s}"] = score_submission_path(ckpt(s), f"ts{s}")
    RES_SUB["submission"] = score_submission_path(SUBMIT, "submission")
    for k, v in RES_SCRIPT.items(): print(f"[평가 스크립트] {k}: {v.get('accuracy'):.4%}")
''')

md("## 6. 슬라이스 (분석용 · 모델 입력 아님)")
code(r'''
pred = pd.read_csv(OUT / "val_sub_submission_pred.csv")
df = pred.merge(va.drop(columns=["speaker"]), on=["call_id", "utt_idx"], how="left")
df["dur"] = df["duration_sec"]
df["ok"] = df["speaker"] == df["predicted"]
slices = {"all": df, "<0.5s": df[df.dur < 0.5], "0.5-1s": df[(df.dur >= 0.5) & (df.dur < 1)], "<1s": df[df.dur < 1]}

# 겹침 슬라이스: Val 인덱스에 발화 시작·끝 열이 있을 때만 (분석용, 통화 안 다른 발화와의 겹침)
s_col = next((c for c in ["startAt", "start", "start_sec"] if c in df), None)
e_col = next((c for c in ["endAt", "end", "end_sec"] if c in df), None)
if s_col and e_col:
    def overlap_ratio(g):
        st, en = g[s_col].to_numpy(), g[e_col].to_numpy(); r = np.zeros(len(g))
        for i in range(len(g)):
            ov = np.clip(np.minimum(en[i], en) - np.maximum(st[i], st), 0, None); ov[i] = 0
            r[i] = ov.max() / max(en[i] - st[i], 1e-9)
        return pd.Series(r, index=g.index)
    df["ov"] = df.groupby("call_id", group_keys=False).apply(overlap_ratio)
    p90 = df["ov"].quantile(0.9)
    slices[f"overlap>=p90({p90:.2f})"] = df[df.ov >= p90]
else:
    print("Val 인덱스에 startAt/endAt 열이 없어 겹침 슬라이스는 생략 (길이 슬라이스만)")
rows = []
for k, d in slices.items():
    rows.append({"slice": k, "n": len(d), "acc": d.ok.mean(),
                 "recall_0": d[d.speaker == 0].ok.mean(), "recall_1": d[d.speaker == 1].ok.mean()})
SL = pd.DataFrame(rows); SL.to_csv(OUT / "val_submission_slices.csv", index=False); SL
''')

md("""
## 7. 판정 · 환경 고정

- **채택 기준**: 제출 ckpt의 공식 Val(제출 방식)이 **0.8726 이상**(L1 보험 제출본 84.26% + 3%p) **그리고** `SEEDS_DONE` 모든 seed의 에폭별 로그가 완비되면 채택한다.
- 경로 표시: `SEEDS_DONE` 3개 = **우선순위 1 (3-seed 재학습본)**, 그보다 적으면 = **우선순위 2 (완료 seed만)**.
- 이전 L2 앙상블 91.55% ±0.3%p(0.9125–0.9185)는 **재현 범위 안/밖 참고용**이며 채택 기준이 아니다.
""")
code(r'''
acc = RES_SUB["submission"]["accuracy"]
L1_VAL_ACC      = 0.8426   # M2-L1-9k 공식 Val (보험 제출본, PR #46)
MIN_GAIN        = 0.03     # 채택: L1보다 +3%p 이상 + 에폭별 로그 완비
ADOPT_THRESHOLD = 0.8726   # = L1_VAL_ACC + MIN_GAIN (부동소수 오차 없이 고정)
REF_ACC, REF_LO, REF_HI = 0.9155, 0.9125, 0.9185   # 이전 L2 앙상블 공식 Val ±0.3%p (재현 참고용, 채택 기준 아님)

logs_complete = all(DONE_INFO[s]["epoch_log_complete"] for s in SEEDS_DONE)
acc_ok = acc >= ADOPT_THRESHOLD
adopt = acc_ok and logs_complete
reproduced = REF_LO <= acc <= REF_HI
if adopt:
    verdict = f"채택 후보 · {PATH_LABEL} (L1 대비 +{(acc - L1_VAL_ACC) * 100:.2f}%p)"
else:
    why = ([] if acc_ok else [f"공식 Val {acc:.2%} < 87.26%"]) + \
          ([] if logs_complete else ["에폭 로그 누락: " + ", ".join(f"ts{s} {DONE_INFO[s]['epoch_log_missing']}" for s in SEEDS_DONE if not DONE_INFO[s]["epoch_log_complete"])])
    verdict = f"채택 보류 — L1 보험 제출본 유지, 로그 공유 ({'; '.join(why)})"
print(f"{PATH_LABEL} · {SUBMIT_KIND} · seeds {SEEDS_DONE}")
print(f"제출 ckpt 공식 Val {acc:.4%} (기준 ≥ 87.26%) · 에폭 로그 {'완비' if logs_complete else '누락'} → {verdict}")
print(f"재현 참고: 이전 91.55% ±0.3%p (0.9125–0.9185) {'재현 범위 안' if reproduced else '재현 범위 밖'} ({(acc - REF_ACC) * 100:+.2f}%p)")

SUMMARY = {
    "commit": COMMIT_FULL, "commit_source": COMMIT_SOURCE, "code_zip": CODE_ZIP.name, "code_zip_sha256": CODE_ZIP_SHA256,
    "amp": USE_AMP, "fp32_epoch1_ref": FP32_EPOCH1_REF, "amp_suspect_flags": AMP_SUSPECT_FLAGS,
    "amp_epoch1_gate": json.load(open(OUT / "amp_epoch1_gate.json", encoding="utf-8"))
                       if (OUT / "amp_epoch1_gate.json").is_file() else None,
    "epochs": EPOCHS, "path": PATH_LABEL, "seeds_done": SEEDS_DONE,
    "per_seed": {f"ts{s}": {**DONE_INFO[s],
                            "official_val_acc_submission_path": RES_SUB.get(f"ts{s}", {}).get("accuracy"),
                            "official_val_acc_eval_script": RES_SCRIPT.get(f"ts{s}", {}).get("accuracy")} for s in SEEDS_DONE},
    "submission": {"kind": SUBMIT_KIND, "ckpt": str(SUBMIT), "sha256": SHAS["submission"], "accuracy": acc,
                   "recall_daewon_0": RES_SUB["submission"]["recall_daewon_0"],
                   "recall_singoja_1": RES_SUB["submission"]["recall_singoja_1"],
                   "confusion_matrix": RES_SUB["submission"]["confusion_matrix"]},
    "adoption": {"rule": "official Val >= 0.8726 (L1 0.8426 + 0.03) AND epoch logs complete for every seed in seeds_done",
                 "threshold": ADOPT_THRESHOLD, "l1_val_acc": L1_VAL_ACC, "min_gain": MIN_GAIN,
                 "acc_ok": acc_ok, "epoch_logs_complete": logs_complete, "adopt": adopt},
    "reproduction_reference": {"ref_acc": REF_ACC, "range": [REF_LO, REF_HI], "within": reproduced,
                               "label": "재현 범위 안" if reproduced else "재현 범위 밖", "note": "참고용, 채택 기준 아님"},
    "sha256": {"ckpt": SHAS, "data": DATA_SHAS, "code_zip": CODE_ZIP_SHA256},
    "verdict": verdict, "written_at": time.strftime("%F %T"),
}
json.dump(SUMMARY, open(OUT / "m2_l2_rerun_summary.json", "w"), indent=2, ensure_ascii=False)
_append(LOG, f"[{time.strftime('%F %T')}] VERDICT {verdict} | acc {acc:.6f} | {PATH_LABEL} | seeds {SEEDS_DONE}\n")
write_env_json()
run([sys.executable, "-m", "pip", "freeze"])  # Drive 로그에 버전 기록
subprocess.run(f"{sys.executable} -m pip freeze > '{OUT}/pip_freeze.txt'", shell=True)
''')

nb = {"cells": cells, "metadata": old["metadata"], "nbformat": 4, "nbformat_minor": old.get("nbformat_minor", 5)}
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")

# validate
nb2 = json.loads(NB_PATH.read_text(encoding="utf-8"))
assert nb2["nbformat"] == 4
for i, c in enumerate(nb2["cells"]):
    if c["cell_type"] == "code":
        assert c["outputs"] == [] and c["execution_count"] is None
        compile("".join(c["source"]), f"cell{i}", "exec")
print(f"OK: {len(nb2['cells'])} cells, {sum(c['cell_type'] == 'code' for c in nb2['cells'])} code cells compile, outputs empty")
