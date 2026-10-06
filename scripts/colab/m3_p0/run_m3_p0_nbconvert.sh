#!/bin/bash
# M3 P0 rerun via nbconvert in the same Colab VM (single-session limit)
OUT=/content/drive/MyDrive/dcc_runs/m3_p0_rerun
mkdir -p "$OUT"
ts(){ date '+%Y-%m-%d %H:%M:%S KST'; }
export TZ=Asia/Seoul
echo "$(ts) START m3_p0 nbconvert (nproc=$(nproc))" | tee -a "$OUT/STATUS.txt"
cp /content/drive/MyDrive/dcc_runs/mission3_p0_rerun_exec.ipynb /content/m3p0_in.ipynb
cd /content
jupyter nbconvert --to notebook --execute --allow-errors \
  --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
  /content/m3p0_in.ipynb --output /content/m3p0_out.ipynb
rc=$?
cp /content/m3p0_out.ipynb "$OUT/mission3_p0_rerun_executed.ipynb"
python3 - <<'PY' | tee -a "$OUT/STATUS.txt"
import json
nb=json.load(open('/content/m3p0_out.ipynb'))
errs=[i for i,c in enumerate(nb['cells']) if c['cell_type']=='code' and any(o.get('output_type')=='error' for o in c.get('outputs',[]))]
print('error_cells=',errs)
try:
    v=json.load(open('/content/drive/MyDrive/dcc_runs/m3_p0_rerun/verdict.json')); print('verdict=',json.dumps(v,ensure_ascii=False))
except Exception as e: print('no verdict:',e)
PY
echo "$(ts) END m3_p0 rc=$rc" | tee -a "$OUT/STATUS.txt"
