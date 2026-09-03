#!/usr/bin/env python3
"""Mission 2 inference entry point (contest-mandated CLI contract).

    python scripts/mission2_inference.py \\
        --audio_dir {wav folder} --label_dir {json folder} \\
        --ckpt_path {checkpoint file} --output ./outputs/mission2.csv

audio_dir/label_dir are paired by filename stem with no Training/Validation
subfolder assumption, since the grader hands us a flat folder (this is why
dcc_data.manifest.build_manifest -- which requires exactly one Training or
Validation path component -- is not used here; see
dcc_data.mission2_features.pair_by_stem).

Row order follows each label file's own utterance order, and startAt/endAt
are copied through unmodified, per the contest's output format
([audio file name], [startAt], [endAt], [speaker]).

If feature extraction or prediction fails for a given utterance, the row is
still emitted (falling back to the training-set majority class) so the
output row count always equals the input utterance count -- the failure is
logged to stderr with the stem/utt_idx instead of silently dropping a row.

Development note: this currently lives under scripts/ rather than as a
top-level inference.py because this repository holds all three missions and
the final submission packaging (one inference.py per mission, at whatever
path each submission needs) has not been decided yet -- see the checkpoint
format note in dcc_data.mission2_train.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dcc_data.mission2_features import (  # noqa: E402
    FEATURE_COLUMNS,
    N_FFT,
    pair_by_stem,
    read_call_waveform,
    slice_and_pad,
    utterance_feature_vector,
    utterance_frame_range,
)
from dcc_data.mission2_train import load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model, meta = load_model(args.ckpt_path)
    feature_columns = tuple(meta["feature_columns"])
    if feature_columns != FEATURE_COLUMNS:
        raise ValueError(
            "checkpoint feature_columns do not match the current feature extractor "
            "(dcc_data.mission2_features.FEATURE_COLUMNS) -- retrain or update the checkpoint"
        )
    fallback_speaker = int(meta.get("majority_class", 1))

    pairs = pair_by_stem(args.audio_dir, args.label_dir)
    if not pairs:
        print(f"[warn] no matching audio/label stem pairs under {args.audio_dir} / {args.label_dir}", file=sys.stderr)

    rows: list[dict] = []
    failure_count = 0

    for stem, wav_path, json_path in pairs:
        with json_path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        utterances = document.get("utterances", [])

        waveform = None
        sample_rate = None
        try:
            waveform, sample_rate = read_call_waveform(wav_path)
        except Exception as exc:  # noqa: BLE001 - one bad call must not stop the whole run
            print(f"[warn] failed to read {wav_path}: {type(exc).__name__}: {exc}", file=sys.stderr)

        for utt_idx, utterance in enumerate(utterances):
            start_at = utterance.get("startAt")
            end_at = utterance.get("endAt")
            speaker = fallback_speaker

            if waveform is not None:
                try:
                    start_frame, end_frame = utterance_frame_range(start_at, end_at, sample_rate, waveform.shape[0])
                    segment = slice_and_pad(waveform, start_frame, end_frame, N_FFT)
                    duration_sec = (float(end_at) - float(start_at)) / 1000.0
                    vector = utterance_feature_vector(segment, sample_rate, duration_sec)
                    speaker = int(model.predict(vector.reshape(1, -1).astype(np.float64))[0])
                except Exception as exc:  # noqa: BLE001 - fall back to majority class, log, keep the row
                    failure_count += 1
                    print(
                        f"[warn] feature/predict failed for stem={stem} utt_idx={utt_idx}: "
                        f"{type(exc).__name__}: {exc} -- falling back to majority class",
                        file=sys.stderr,
                    )
            else:
                failure_count += 1

            rows.append(
                {
                    "audio_file_name": wav_path.name,
                    "startAt": start_at,
                    "endAt": end_at,
                    "speaker": speaker,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["audio_file_name", "startAt", "endAt", "speaker"])
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_name, args.output)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise

    print(f"Wrote {len(rows)} rows to {args.output} ({failure_count} fell back to majority class)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
