#!/usr/bin/env bash
# M2 L2 spectrogram rebuild (CPU runtime). Launch: nohup bash /content/drive/MyDrive/dcc_runs/m2_l2_spec/run_l2_spec.sh > /content/m2_l2_spec_launcher.log 2>&1 &
# Order: Validation (3,640 calls) build -> verify -> pack -> Drive ; then Training (all Seoul, 29,200) build -> verify(--output) -> pack -> Drive.
# Individual .npy only under /content. STATUS / index / failures-log / logs written straight to Drive.
set -uo pipefail
OUT=/content/drive/MyDrive/dcc_runs/m2_l2_spec
mkdir -p "$OUT"
STATUS="$OUT/STATUS.txt"
st(){ echo "$(TZ=Asia/Seoul date '+%F %T KST') $*" | tee -a "$STATUS"; }
fail(){ st "FAILED: $*"; exit 1; }
st "START m2_l2_spec (host=$(hostname) nproc=$(nproc) gpu=$(nvidia-smi -L 2>/dev/null | head -1 || echo none))"
export TQDM_MININTERVAL=60
DR="/content/drive/.shortcut-targets-by-id/1Ezl13Cpmt2lGgl-FqISYJWfop8omii6z/대학부 데이터"
[ -d "$DR" ] || DR=$(ls -d /content/drive/.shortcut-targets-by-id/1Ezl13Cpmt2lGgl-FqISYJWfop8omii6z/* 2>/dev/null | head -1)
[ -d "$DR" ] || fail "DATA_ROOT not found"
export DCC_DATA_ROOT="$DR"
st "DATA_ROOT=$DR ($(ls "$DR" | tr '\n' ' '))"
# disk checks
DFREE=$(df -B1G --output=avail /content/drive | tail -1 | tr -d ' ')
LFREE=$(df -B1G --output=avail /content | tail -1 | tr -d ' ')
st "free space: Drive(df)=${DFREE}G local=${LFREE}G"
[ "${DFREE:-0}" -ge 10 ] || fail "Drive free ${DFREE}G < 10G (need ~8G for packs)"
[ "${LFREE:-0}" -ge 30 ] || fail "local free ${LFREE}G < 30G (train npy ~14G + pack 7G)"
# code
ZIP=/content/drive/MyDrive/dcc2026-prelim-main-4bfd2c4.zip
DEST=/content/dcc2026-prelim-main
if [ ! -d "$DEST" ]; then
  [ -f "$ZIP" ] || fail "zip not found $ZIP"
  rm -rf /content/_unz && mkdir -p /content/_unz && unzip -q -o "$ZIP" -d /content/_unz || fail "unzip"
  n=$(ls -1 /content/_unz | wc -l)
  if [ "$n" = 1 ] && [ -d "/content/_unz/$(ls -1 /content/_unz)" ]; then mv "/content/_unz/$(ls -1 /content/_unz)" "$DEST"; else mv /content/_unz "$DEST"; fi
fi
[ -f "$DEST/scripts/build_mission2_spectrograms.py" ] || fail "repo layout unexpected in $DEST"
cd "$DEST"
pip install -q librosa pyarrow >> "$OUT/pip.log" 2>&1
# manifest (reuse Drive copy if present, else rebuild from DATA_ROOT and save to Drive)
MAN=data/manifests/dcc_manifest_v1.csv
MAN_DRIVE=/content/drive/MyDrive/dcc_runs/manifests/dcc_manifest_v1.csv
mkdir -p data/manifests /content/drive/MyDrive/dcc_runs/manifests
if [ -f "$MAN_DRIVE" ]; then cp -f "$MAN_DRIVE" "$MAN"; st "manifest from Drive copy";
else
  st "manifest rebuild from DATA_ROOT (rglob over Drive, may take a while)"
  python scripts/build_manifest.py --data-root "$DR" --output "$MAN" --overwrite > "$OUT/manifest_build.log" 2>&1 || fail "manifest build: $(tail -3 "$OUT/manifest_build.log" | tr '\n' ' ')"
  cp -f "$MAN" "$MAN_DRIVE"
fi
ROWS=$(( $(wc -l < "$MAN") - 1 ))
st "manifest rows=$ROWS (expected 32840) $(tail -2 "$OUT/manifest_build.log" 2>/dev/null | tr '\n' ' ')"
W=$(nproc)
# ---------- Validation ----------
if grep -q "VAL_DONE" "$STATUS"; then st "VAL already done, skip"; else
st "VAL_BUILD_START workers=$W"
python scripts/build_mission2_spectrograms.py --manifest "$MAN" --data-root "$DR" --split validation --n-calls -1 --seed 42 --workers "$W" \
  --output-dir /content/m2_spec_val --index-output "$OUT/m2_spectrogram_index_validation.parquet" \
  --failures-log "$OUT/m2_spectrogram_build_failures_validation.jsonl" > "$OUT/val_build.log" 2>&1 || fail "val build: $(tail -3 "$OUT/val_build.log" | tr '\n' ' ')"
st "VAL_BUILD_DONE $(grep -E 'Spectrograms saved|Failures|Elapsed' "$OUT/val_build.log" | tr '\n' ' ')"
VLR=$(python - "$MAN" "$DR" <<'PY'
import csv,sys,unicodedata,os,re
from pathlib import PurePosixPath
dirs=set()
for r in csv.DictReader(open(sys.argv[1],encoding="utf-8")):
    if r["split"]!="Validation": continue
    parts=PurePosixPath(r["json_path"]).parts
    for i,p in enumerate(parts):
        if re.fullmatch(r"VL_서울_구급",unicodedata.normalize("NFC",p)): dirs.add("/".join(parts[:i+1])); break
dirs=sorted(dirs)
print(os.path.join(sys.argv[2],os.path.commonpath(dirs)) if dirs else "")
PY
)
st "VAL label root: $VLR"
python scripts/verify_mission2_raw_labels.py --index "$OUT/m2_spectrogram_index_validation.parquet" --label-root "$VLR" > "$OUT/val_verify.log" 2>&1
VRC=$?
st "VAL_VERIFY exit=$VRC $(grep -E 'rows checked|differs|OK:|CANNOT' "$OUT/val_verify.log" | tr '\n' ' ')"
[ "$VRC" = 0 ] || fail "val verify exit $VRC (see val_verify.log)"
mkdir -p /content/m2_l2_pack
python scripts/pack_mission2_spectrograms.py --index "$OUT/m2_spectrogram_index_validation.parquet" --spectrogram-dir /content/m2_spec_val \
  --packed-output /content/m2_l2_pack/m2_spectrograms_validation_packed.npy --index-output "$OUT/m2_spectrogram_index_validation_packed.parquet" > "$OUT/val_pack.log" 2>&1 || fail "val pack: $(tail -3 "$OUT/val_pack.log" | tr '\n' ' ')"
cp -f /content/m2_l2_pack/m2_spectrograms_validation_packed.npy "$OUT/" || fail "val pack copy to Drive"
VS=$(sha256sum /content/m2_l2_pack/m2_spectrograms_validation_packed.npy | cut -d' ' -f1)
echo "$VS  m2_spectrograms_validation_packed.npy" > "$OUT/m2_spectrograms_validation_packed.npy.sha256"
st "VAL_DONE $(grep -E '^Packed' "$OUT/val_pack.log" | head -1) size=$(du -h "$OUT/m2_spectrograms_validation_packed.npy" | cut -f1) sha256=$VS"
fi
# ---------- Training ----------
st "TRAIN_BUILD_START workers=$W (all Seoul Training, --n-calls -1)"
python scripts/build_mission2_spectrograms.py --manifest "$MAN" --data-root "$DR" --split training --n-calls -1 --seed 42 --workers "$W" \
  --output-dir /content/m2_spec_train --index-output "$OUT/m2_spectrogram_index_full.parquet" \
  --failures-log "$OUT/m2_spectrogram_build_failures_full.jsonl" > "$OUT/train_build.log" 2>&1 || fail "train build: $(tail -3 "$OUT/train_build.log" | tr '\n' ' ')"
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
python scripts/pack_mission2_spectrograms.py --index "$OUT/m2_spectrogram_index_full_provided_labels.parquet" --spectrogram-dir /content/m2_spec_train \
  --packed-output /content/m2_l2_pack/m2_spectrograms_packed.npy --index-output "$OUT/m2_spectrogram_index_packed.parquet" > "$OUT/train_pack.log" 2>&1 || fail "train pack: $(tail -3 "$OUT/train_pack.log" | tr '\n' ' ')"
st "TRAIN_PACK_DONE $(grep -E '^Packed' "$OUT/train_pack.log" | head -1); copying to Drive"
cp -f /content/m2_l2_pack/m2_spectrograms_packed.npy "$OUT/" || fail "train pack copy to Drive"
TS=$(sha256sum /content/m2_l2_pack/m2_spectrograms_packed.npy | cut -d' ' -f1)
echo "$TS  m2_spectrograms_packed.npy" > "$OUT/m2_spectrograms_packed.npy.sha256"
st "ALL_DONE train_pack size=$(du -h "$OUT/m2_spectrograms_packed.npy" | cut -f1) sha256=$TS"
