#!/bin/bash
# M3 P1 v5 (2026-10-06): P0 internal-Val split; hyperparameters identical to v4/v3.
# Default code zip: dcc2026-prelim-main-6cc1181.zip (one top folder dcc2026-prelim-main-6cc1181/).
#   bash scripts/colab/m3_p1/run_p1_v5.sh
#   bash /content/drive/MyDrive/dcc_runs/m3_p1/run_p1_v5.sh
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TRAIN_SRC="$HERE/m3_p1_train_v5.py"
ZIP_NAME=dcc2026-prelim-main-6cc1181.zip
[ -f "$TRAIN_SRC" ] || { echo "train script not found: $TRAIN_SRC"; exit 1; }

RUN_ROOT=${DCC_RUN_ROOT:-/content/drive/MyDrive/dcc_runs}
D=$RUN_ROOT/m3_p1
mkdir -p "$D" || { echo "OUT is not writable: $D. Set DCC_RUN_ROOT to a writable folder in your own Drive."; exit 1; }
probe="$D/.write_probe"
if ! printf ok > "$probe" 2>/dev/null; then
  echo "OUT is not writable: $D. Set DCC_RUN_ROOT to a writable folder in your own Drive."
  exit 1
fi
rm -f "$probe" || { echo "OUT is not writable: $D (probe delete failed). Set DCC_RUN_ROOT to a writable folder in your own Drive."; exit 1; }
export M3P1_OUT="$D"

nvidia-smi -L || { echo "NO GPU - set runtime to T4 first"; exit 1; }

# Lookup order: $M3P1_ZIP -> script folder -> $RUN_ROOT/code/ -> MyDrive root.
ZIP=""
if [ -n "${M3P1_ZIP:-}" ] && [ -f "$M3P1_ZIP" ]; then ZIP="$M3P1_ZIP"; fi
if [ -z "$ZIP" ] && [ -f "$HERE/$ZIP_NAME" ]; then ZIP="$HERE/$ZIP_NAME"; fi
if [ -z "$ZIP" ] && [ -f "$RUN_ROOT/code/$ZIP_NAME" ]; then ZIP="$RUN_ROOT/code/$ZIP_NAME"; fi
if [ -z "$ZIP" ] && [ -f "/content/drive/MyDrive/$ZIP_NAME" ]; then ZIP="/content/drive/MyDrive/$ZIP_NAME"; fi

if [ -n "$ZIP" ]; then
  actual=$(sha256sum "$ZIP" | cut -d' ' -f1)
  echo "code zip: $ZIP"
  echo "code zip sha256: $actual"
  expected=""
  if [ -n "${M3P1_ZIP_SHA256:-}" ]; then expected="$M3P1_ZIP_SHA256"
  elif [ -f "$ZIP.sha256" ]; then expected=$(awk 'NF { print $1; exit }' "$ZIP.sha256")
  fi
  if [ -n "$expected" ]; then
    if [ "$actual" != "$expected" ]; then
      echo "ERROR: zip sha256 mismatch got=$actual expected=$expected"
      exit 1
    fi
    echo "zip sha256 matches expected"
  fi
  echo "code zip comment:"
  unzip -z "$ZIP" || true
  python3 - "$ZIP" << 'PY'
import sys, zipfile
path = sys.argv[1]
with zipfile.ZipFile(path) as zf:
    comment = zf.comment.decode("utf-8", "replace")
    print("zip comment:", comment if comment else "(none)")
    names = sorted(n for n in zf.namelist() if not n.endswith("/"))
    print("zip files:", len(names))
    for name in names:
        print(" ", name)
PY
  if [ -d /content ]; then
    UNZ=$(mktemp -d /content/m3p1_code.XXXXXX)
  else
    UNZ=$(mktemp -d "${TMPDIR:-/tmp}/m3p1_code.XXXXXX")
  fi
  unzip -q -o "$ZIP" -d "$UNZ" || { echo "unzip failed: $ZIP"; exit 1; }
  # Same rule as notebooks/gen/make_l2_nb.py: one top-level directory becomes the repo root.
  REPO_DIR=$(python3 - "$UNZ" << 'PY'
import pathlib, sys
unz = pathlib.Path(sys.argv[1])
top = list(unz.iterdir())
src = top[0] if len(top) == 1 and top[0].is_dir() else unz
print(src)
PY
)
  echo "code repo (detected top folder): $REPO_DIR"
  export M3P1_REPO="$REPO_DIR"
  comment=$(unzip -z "$ZIP" | tail -n +2 || true)
  hash=$(printf '%s\n' "$comment" | grep -oE '\b[0-9a-fA-F]{40}\b' | head -n 1 || true)
  if [ -z "$hash" ]; then
    hash=$(printf '%s\n' "$comment" | grep -oE '\b[0-9a-fA-F]{7,40}\b' | head -n 1 || true)
  fi
  if [ -n "$hash" ]; then
    export M3P1_COMMIT="$hash"
    echo "M3P1_COMMIT=$hash (from zip comment)"
  else
    echo "zip comment has no commit hash; training script keeps M3P1_COMMIT or default 6cc1181"
  fi
elif rev=$(git -C "$HERE" rev-parse HEAD 2>/dev/null); then
  top=$(git -C "$HERE" rev-parse --show-toplevel)
  echo "code git commit: $rev"
  echo "code repo: $top"
  export M3P1_REPO="$top"
  export M3P1_COMMIT="$rev"
else
  echo "ERROR: code provenance must be printed (team rule). No code zip found (checked M3P1_ZIP, $HERE/$ZIP_NAME, $RUN_ROOT/code/$ZIP_NAME, /content/drive/MyDrive/$ZIP_NAME) and $HERE is not inside a git checkout."
  exit 1
fi

if [ -d /content ]; then mkdir -p /content/m3_p1; fi
for f in train.log; do [ -f "$D/$f" ] && cp "$D/$f" "$D/${f%.log}_before_v5_$(date +%H%M%S).log"; done
echo "$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S') KST START v5 host=$(hostname) gpu=$(nvidia-smi -L | head -1)" >> "$D/STATUS.txt"
echo "train script sha256: $(sha256sum "$TRAIN_SRC" | cut -d' ' -f1)"
cd /content/m3_p1 2>/dev/null || true
pip install -q iterative-stratification 2>&1 | tail -2
python -u "$TRAIN_SRC" 2>&1 | tee "$D/train.log"
rc=$?
echo "EXIT=$rc"
pip freeze > "$D/pip_freeze.txt"
{
  echo "GPU=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
  python - << 'PY'
def ver(name):
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:
        return "missing (%s)" % type(exc).__name__
try:
    import torch
    torch_ver = torch.__version__
    cuda = getattr(torch.version, "cuda", None)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
except Exception as exc:
    torch_ver = "missing (%s)" % type(exc).__name__
    cuda = None
    gpu = "none"
print("torch=%s transformers=%s CUDA=%s GPU=%s" % (torch_ver, ver("transformers"), cuda, gpu))
PY
} | tee -a "$D/train.log"
tail -5 "$D/STATUS.txt"
exit "$rc"
