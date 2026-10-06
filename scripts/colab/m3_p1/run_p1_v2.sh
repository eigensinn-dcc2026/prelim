#!/bin/bash
# launched from Colab terminal: bash /content/drive/MyDrive/dcc_runs/m3_p1/run_p1_v2.sh
# attempt2 (2026-10-06): runtime disconnected ~04:30 KST during ep3; best ckpt now copied to Drive on every BEST.
D=/content/drive/MyDrive/dcc_runs/m3_p1
if [ ! -d /content/dcc2026-prelim-main ] && [ -f /content/drive/MyDrive/dcc2026-prelim-main-4bfd2c4.zip ]; then
  (cd /content && unzip -q -o /content/drive/MyDrive/dcc2026-prelim-main-4bfd2c4.zip)
fi
mkdir -p /content/m3_p1
[ -f "$D/train.log" ] && [ ! -f "$D/train_attempt1.log" ] && cp "$D/train.log" "$D/train_attempt1.log"
echo "$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S') KST RESTART attempt2 (runtime disconnected ~04:30 KST in ep3; attempt1 local ckpt lost)" >> "$D/STATUS.txt"
cp "$D/m3_p1_train_v2.py" /content/m3_p1/m3_p1_train.py
cd /content/m3_p1
pip install -q iterative-stratification > /content/m3_p1/pip.log 2>&1
nohup python /content/m3_p1/m3_p1_train.py > "$D/train.log" 2>&1 &
echo "P1_PID=$!"
