#!/bin/bash
# M3 P1 v3 (2026-10-06): run in a Colab notebook cell (T4) for live output:  !bash /content/drive/MyDrive/dcc_runs/m3_p1/run_p1_v3.sh
# 3 epochs from scratch; every-epoch ckpt + best ckpt -> Drive dcc_runs/m3_p1/; foreground (output streams in the cell and is tee'd to train.log).
set -o pipefail
D=/content/drive/MyDrive/dcc_runs/m3_p1
ZIP=/content/drive/MyDrive/dcc2026-prelim-main-4bfd2c4.zip
nvidia-smi -L || { echo "NO GPU - set runtime to T4 first"; exit 1; }
if [ ! -d /content/dcc2026-prelim-main ]; then
  [ -f "$ZIP" ] || { echo "code zip not found: $ZIP"; exit 1; }
  (cd /content && unzip -q -o "$ZIP")
fi
echo "code zip sha256: $(sha256sum "$ZIP" | cut -d' ' -f1)"
mkdir -p /content/m3_p1
for f in train.log; do [ -f "$D/$f" ] && cp "$D/$f" "$D/${f%.log}_before_v3_$(date +%H%M%S).log"; done
echo "$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S') KST START v3 host=$(hostname) gpu=$(nvidia-smi -L | head -1)" >> "$D/STATUS.txt"
cp "$D/m3_p1_train_v3.py" /content/m3_p1/m3_p1_train.py
echo "train script sha256: $(sha256sum /content/m3_p1/m3_p1_train.py | cut -d' ' -f1)"
cd /content/m3_p1
pip install -q iterative-stratification 2>&1 | tail -2
python -u /content/m3_p1/m3_p1_train.py 2>&1 | tee "$D/train.log"
echo "EXIT=$?"; tail -5 "$D/STATUS.txt"
