#!/usr/bin/env bash
# Finisher for M2 L2 Training spectrogram build (wrapper run_l2_spec.sh was killed 2026-10-06 09:20 KST; python build kept running).
# 1) While build runs: every 30 min, tar newly written .npy files into Drive train_parts/ (intermediate backup, per 총괄 rule).
# 2) When build exits: run the remaining steps of run_l2_spec.sh (verify -> pack -> Drive copy -> sha256 -> ALL_DONE).
# Launch: nohup bash /content/drive/MyDrive/dcc_runs/m2_l2_spec/finish_train.sh > /content/finish_train.log 2>&1 &
set -uo pipefail
OUT=/content/drive/MyDrive/dcc_runs/m2_l2_spec
STATUS="$OUT/STATUS.txt"
PARTS="$OUT/train_parts"; mkdir -p "$PARTS"
SRC=/content/m2_spec_train
st(){ echo "$(TZ=Asia/Seoul date '+%F %T KST') $*" | tee -a "$STATUS"; }
fail(){ st "FAILED: $*"; exit 1; }
DR="/content/drive/.shortcut-targets-by-id/1Ezl13Cpmt2lGgl-FqISYJWfop8omii6z/대학부 데이터"
export DCC_DATA_ROOT="$DR"
cd /content/dcc2026-prelim-main || fail "repo dir missing"
MAN=data/manifests/dcc_manifest_v1.csv
BPID=$(pgrep -of 'build_mission2_spectrograms.py --manifest' || true)
st "FINISHER_START build_pid=${BPID:-none} files_now=$(find $SRC -name '*.npy' | wc -l)"
MARK=/content/.train_part_mark
n=$(ls "$PARTS"/part_*.tar 2>/dev/null | wc -l)
backup(){
  local new=/content/.train_part_new; touch "$new"
  local lst=/content/.train_part_list
  if [ -f "$MARK" ]; then (cd "$SRC" && find . -name '*.npy' -newer "$MARK" ! -newer "$new") > "$lst"
  else (cd "$SRC" && find . -name '*.npy' ! -newer "$new") > "$lst"; fi
  local cnt=$(wc -l < "$lst")
  [ "$cnt" -gt 0 ] || { mv -f "$new" "$MARK"; return; }
  local dfree=$(df -B1G --output=avail /content/drive | tail -1 | tr -d ' ')
  if [ "${dfree:-0}" -lt 12 ]; then st "TRAIN_PART skipped (Drive free ${dfree}G < 12G, keep room for pack)"; return; fi
  n=$((n+1)); local p=$(printf 'part_%03d.tar' $n)
  if tar -cf "/content/$p" -C "$SRC" -T "$lst" && cp -f "/content/$p" "$PARTS/$p"; then
    st "TRAIN_PART $p files=$cnt size=$(du -h /content/$p | cut -f1) total_files=$(find $SRC -name '*.npy' | wc -l)"
    rm -f "/content/$p"; mv -f "$new" "$MARK"
  else st "TRAIN_PART $p FAILED (will retry next round)"; rm -f "/content/$p"; n=$((n-1)); fi
}
if [ -n "${BPID:-}" ]; then
  while kill -0 "$BPID" 2>/dev/null; do
    for i in $(seq 1 30); do kill -0 "$BPID" 2>/dev/null || break; sleep 60; done
    kill -0 "$BPID" 2>/dev/null && backup
  done
fi
backup
grep -qE 'Spectrograms saved' "$OUT/train_build.log" || fail "train build ended without 'Spectrograms saved': $(tail -3 "$OUT/train_build.log" | tr '\n' ' ')"
[ -f "$OUT/m2_spectrogram_index_full.parquet" ] || fail "train index parquet missing"
NC=$(python -c "import pandas as pd;d=pd.read_parquet('$OUT/m2_spectrogram_index_full.parquet');print(d.call_id.nunique(),len(d))")
st "TRAIN_BUILD_DONE calls,utts=$NC $(grep -E 'Spectrograms saved|Failures|Elapsed' "$OUT/train_build.log" | tr '\n' ' ')"
TLR=$(python - "$MAN" "$DR" <<'PY'
import csv,sys,unicodedata,os,re
from pathlib import PurePosixPath
dirs=set()
for r in csv.DictReader(open(sys.argv[1],encoding="utf-8")):
    if r["split"]!="Training": continue
    parts=PurePosixPath(r["json_path"]).parts
    for i,p in enumerate(parts):
        if re.fullmatch(r"TL_서울_구급",unicodedata.normalize("NFC",p)): dirs.add("/".join(parts[:i+1])); break
dirs=sorted(dirs)
print(os.path.join(sys.argv[2],os.path.commonpath(dirs)) if dirs else "")
PY
)
st "TRAIN label root: $TLR"
python scripts/verify_mission2_raw_labels.py --index "$OUT/m2_spectrogram_index_full.parquet" --label-root "$TLR" \
  --output "$OUT/m2_spectrogram_index_full_provided_labels.parquet" > "$OUT/train_verify.log" 2>&1 || fail "train verify: $(tail -3 "$OUT/train_verify.log" | tr '\n' ' ')"
LS=$(python -c "import pandas as pd;d=pd.read_parquet('$OUT/m2_spectrogram_index_full_provided_labels.parquet');print(len(d),d.label_source.unique().tolist())")
st "TRAIN_VERIFY_DONE rows,label_source=$LS $(grep -E 'differs from the provided|restored' "$OUT/train_verify.log" | tr '\n' ' ')"
mkdir -p /content/m2_l2_pack
python scripts/pack_mission2_spectrograms.py --index "$OUT/m2_spectrogram_index_full_provided_labels.parquet" --spectrogram-dir "$SRC" \
  --packed-output /content/m2_l2_pack/m2_spectrograms_packed.npy --index-output "$OUT/m2_spectrogram_index_packed.parquet" > "$OUT/train_pack.log" 2>&1 || fail "train pack: $(tail -3 "$OUT/train_pack.log" | tr '\n' ' ')"
st "TRAIN_PACK_DONE $(grep -E '^Packed' "$OUT/train_pack.log" | head -1); copying to Drive"
cp -f /content/m2_l2_pack/m2_spectrograms_packed.npy "$OUT/" || fail "train pack copy to Drive"
TS=$(sha256sum /content/m2_l2_pack/m2_spectrograms_packed.npy | cut -d' ' -f1)
DS=$(sha256sum "$OUT/m2_spectrograms_packed.npy" | cut -d' ' -f1)
echo "$TS  m2_spectrograms_packed.npy" > "$OUT/m2_spectrograms_packed.npy.sha256"
[ "$TS" = "$DS" ] || fail "Drive copy sha mismatch local=$TS drive=$DS"
st "ALL_DONE train_pack size=$(du -h "$OUT/m2_spectrograms_packed.npy" | cut -f1) sha256=$TS"
