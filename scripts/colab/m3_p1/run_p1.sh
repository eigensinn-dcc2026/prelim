#!/bin/bash
# launched from Colab terminal: bash /content/drive/MyDrive/dcc_runs/m3_p1/run_p1.sh
D=/content/drive/MyDrive/dcc_runs/m3_p1
mkdir -p /content/m3_p1
cp "$D/m3_p1_train.py" /content/m3_p1/
cd /content/m3_p1
pip install -q iterative-stratification > /content/m3_p1/pip.log 2>&1
nohup python /content/m3_p1/m3_p1_train.py > "$D/train.log" 2>&1 &
echo "P1_PID=$!"
