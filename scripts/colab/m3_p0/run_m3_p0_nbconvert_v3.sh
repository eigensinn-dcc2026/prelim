#!/bin/bash
# M3 P0 rerun v3: copy Seoul Training labels to /content first (avoid Drive FUSE contention), then nbconvert
export TZ=Asia/Seoul
OUT=/content/drive/MyDrive/dcc_runs/m3_p0_rerun
DATA_ROOT="/content/drive/.shortcut-targets-by-id/1Ezl13Cpmt2lGgl-FqISYJWfop8omii6z/대학부 데이터"
mkdir -p "$OUT"
ts(){ date '+%Y-%m-%d %H:%M:%S KST'; }
log(){ echo "$(ts) $*" | tee -a "$OUT/STATUS.txt"; }
log "START m3_p0 v3 (nproc=$(nproc))"
SRC=$(find "$DATA_ROOT/Training" -maxdepth 4 -type d -name 'TL_서울_구급' 2>/dev/null | head -n 5)
log "label src candidates: $(echo "$SRC" | tr '\n' '|')"
N=$(echo "$SRC" | grep -c .)
if [ "$N" != "1" ]; then log "ERROR label src count=$N"; exit 1; fi
mkdir -p /content/m3_labels
if [ ! -d /content/m3_labels/TL_서울_구급 ]; then
  log "COPY_START $SRC -> /content/m3_labels/"
  cp -r "$SRC" /content/m3_labels/ || { log "ERROR copy failed"; exit 1; }
fi
CNT=$(find /content/m3_labels/TL_서울_구급 -name '*.json' | wc -l)
log "COPY_DONE json=$CNT (expected 29200) size=$(du -sh /content/m3_labels | cut -f1)"
[ "$CNT" = "29200" ] || { log "ERROR json count mismatch"; exit 1; }
cp /content/drive/MyDrive/dcc_runs/mission3_p0_rerun_exec_v3.ipynb /content/m3p0_in.ipynb
cd /content
log "NBCONVERT_START"
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
log "END m3_p0 rc=$rc"
