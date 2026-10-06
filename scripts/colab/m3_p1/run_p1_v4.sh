#!/bin/bash
# M3 P1 v4 (2026-10-06): path handling via colab_env-compatible lookup; hyperparameters identical to v3.
# Works from a repo checkout or a Drive folder:
#   bash scripts/colab/m3_p1/run_p1_v4.sh
#   bash /content/drive/MyDrive/dcc_runs/m3_p1/run_p1_v4.sh
# 3 epochs from scratch; every-epoch ckpt + best ckpt -> $DCC_RUN_ROOT/m3_p1 (default MyDrive/dcc_runs/m3_p1).
# Foreground: output streams in the cell and is tee'd to train.log.
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TRAIN_SRC="$HERE/m3_p1_train_v4.py"
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

ZIP=""
if [ -n "${M3P1_ZIP:-}" ] && [ -f "$M3P1_ZIP" ]; then ZIP="$M3P1_ZIP"; fi
if [ -z "$ZIP" ] && [ -f "$HERE/dcc2026-prelim-main-4bfd2c4.zip" ]; then ZIP="$HERE/dcc2026-prelim-main-4bfd2c4.zip"; fi
if [ -z "$ZIP" ] && [ -f "$RUN_ROOT/code/dcc2026-prelim-main-4bfd2c4.zip" ]; then ZIP="$RUN_ROOT/code/dcc2026-prelim-main-4bfd2c4.zip"; fi
if [ -z "$ZIP" ] && [ -f /content/drive/MyDrive/dcc2026-prelim-main-4bfd2c4.zip ]; then ZIP=/content/drive/MyDrive/dcc2026-prelim-main-4bfd2c4.zip; fi

if [ -n "$ZIP" ]; then
  if [ ! -d /content/dcc2026-prelim-main ]; then
    [ -d /content ] || { echo "ERROR: /content is missing, cannot unzip the code zip (Colab VM disk)."; exit 1; }
    (cd /content && unzip -q -o "$ZIP") || { echo "unzip failed: $ZIP"; exit 1; }
  fi
  echo "code zip: $ZIP"
  echo "code zip sha256: $(sha256sum "$ZIP" | cut -d' ' -f1)"
  echo "code zip comment:"
  unzip -z "$ZIP" || true
  comment=$(unzip -z "$ZIP" | tail -n +2 || true)
  hash=$(printf '%s\n' "$comment" | grep -oE '\b[0-9a-fA-F]{40}\b' | head -n 1 || true)
  if [ -z "$hash" ]; then
    hash=$(printf '%s\n' "$comment" | grep -oE '\b[0-9a-fA-F]{7,40}\b' | head -n 1 || true)
  fi
  export M3P1_REPO=/content/dcc2026-prelim-main
  if [ -n "$hash" ]; then
    export M3P1_COMMIT="$hash"
    echo "M3P1_COMMIT=$hash (from zip comment)"
  else
    echo "zip comment has no commit hash; training script keeps M3P1_COMMIT or default 4bfd2c4"
  fi
elif rev=$(git -C "$HERE" rev-parse HEAD 2>/dev/null); then
  top=$(git -C "$HERE" rev-parse --show-toplevel)
  echo "code git commit: $rev"
  echo "code repo: $top"
  export M3P1_REPO="$top"
  export M3P1_COMMIT="$rev"
else
  echo "ERROR: code provenance must be printed (team rule). No code zip found (checked M3P1_ZIP, $HERE/dcc2026-prelim-main-4bfd2c4.zip, $RUN_ROOT/code/dcc2026-prelim-main-4bfd2c4.zip, /content/drive/MyDrive/dcc2026-prelim-main-4bfd2c4.zip) and $HERE is not inside a git checkout."
  exit 1
fi

mkdir -p /content/m3_p1
for f in train.log; do [ -f "$D/$f" ] && cp "$D/$f" "$D/${f%.log}_before_v4_$(date +%H%M%S).log"; done
echo "$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S') KST START v4 host=$(hostname) gpu=$(nvidia-smi -L | head -1)" >> "$D/STATUS.txt"
echo "train script sha256: $(sha256sum "$TRAIN_SRC" | cut -d' ' -f1)"
cd /content/m3_p1
pip install -q iterative-stratification 2>&1 | tail -2
python -u "$TRAIN_SRC" 2>&1 | tee "$D/train.log"
rc=$?
echo "EXIT=$rc"
pip freeze > "$D/pip_freeze.txt"
python - <<'PY' | tee -a "$D/train.log"
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
except Exception as exc:
    torch_ver = "missing (%s)" % type(exc).__name__
    cuda = None
print("torch=%s transformers=%s CUDA=%s" % (torch_ver, ver("transformers"), cuda))
PY
tail -5 "$D/STATUS.txt"
exit "$rc"
