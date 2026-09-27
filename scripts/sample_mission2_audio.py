#!/usr/bin/env python3
"""Pull a handful of real audio clips out for manual listening (mentor advice:
"녹음 채널/환경 차이 가설은 표본을 까서 직접 들어보라").

Reads the CSV produced by scripts/analyze_mission2_errors.py plus the
utterance index parquet, samples a few utterances per group, crops the
matching audio out of the original WAVs (one open per call, reusing
dcc_data.mission2_features), and writes each as a small standalone .wav file
you can play in Colab with:

    from IPython.display import Audio, display
    display(Audio(path))

Default groups (each answers a different question):
  - 대원_정답_안겹침 / 신고자_정답_안겹침: clean, non-overlapping, correctly
    classified examples of each role -- what does a "typical" dispatcher vs
    caller clip actually sound like? (the channel/environment hypothesis)
  - 대원_오답 / 신고자_오답: misclassified examples -- what do the mistakes
    actually sound like? (noisy? overlapping? just an atypical voice?)

Usage:
    python scripts/sample_mission2_audio.py \\
        --error-analysis data/m2_error_analysis.csv \\
        --index data/m2_utterances_train.parquet \\
        --data-root "$DATA_ROOT" \\
        --output-dir /content/listen_samples \\
        --n-per-group 3 --seed 42
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dcc_data.mission2_analysis import select_listening_samples  # noqa: E402
from dcc_data.mission2_features import read_call_waveform, utterance_frame_range  # noqa: E402

SPEAKER_LABELS = {0: "대원", 1: "신고자"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--error-analysis", type=Path, required=True, help="output of analyze_mission2_errors.py")
    parser.add_argument("--index", type=Path, required=True, help="m2_utterances_train.parquet")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("listen_samples"))
    parser.add_argument("--n-per-group", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_wav_segment(path: Path, segment: np.ndarray, sample_rate: int) -> None:
    pcm16 = (np.clip(segment, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def main() -> int:
    args = parse_args()
    error_df = pd.read_csv(args.error_analysis)
    index_df = pd.read_parquet(args.index)

    merged = error_df.merge(
        index_df[["call_id", "utt_idx", "wav_path", "startAt", "endAt"]],
        on=["call_id", "utt_idx"],
        how="left",
    )
    missing_path = merged["wav_path"].isna().sum()
    if missing_path:
        print(f"[warn] {missing_path} row(s) had no matching wav_path in --index; dropped", file=sys.stderr)
        merged = merged.dropna(subset=["wav_path"])

    groups = select_listening_samples(merged, n_per_group=args.n_per_group, seed=args.seed)
    if len(groups) < 4:
        missing = {"대원_정답_안겹침", "신고자_정답_안겹침", "대원_오답", "신고자_오답"} - set(groups)
        print(f"[warn] no rows available for group(s): {sorted(missing)}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    # Open each call's WAV only once even if it supplies clips to multiple groups.
    picks = []
    for group_name, rows in groups.items():
        for _, row in rows.iterrows():
            picks.append((group_name, row))

    waveform_cache: dict[str, tuple[np.ndarray, int]] = {}
    for group_name, row in picks:
        wav_path = args.data_root / row["wav_path"]
        cache_key = row["wav_path"]
        if cache_key not in waveform_cache:
            waveform_cache[cache_key] = read_call_waveform(wav_path)
        waveform, sample_rate = waveform_cache[cache_key]

        start_frame, end_frame = utterance_frame_range(
            row["startAt"], row["endAt"], sample_rate, waveform.shape[0]
        )
        segment = waveform[start_frame:end_frame]

        speaker_label = SPEAKER_LABELS.get(int(row["speaker"]), str(row["speaker"]))
        out_name = f"{group_name}__{row['call_id']}_utt{int(row['utt_idx'])}.wav"
        out_path = args.output_dir / out_name
        write_wav_segment(out_path, segment, sample_rate)

        manifest_rows.append(
            {
                "file": out_name,
                "group": group_name,
                "실제": speaker_label,
                "예측": SPEAKER_LABELS.get(int(row["predicted"]), str(row["predicted"])),
                "정답여부": bool(row["correct"]),
                "duration_sec": round(float(row["duration_sec"]), 2),
                "겹침": bool(row["overlaps_other"]),
                "확신도": round(float(row["confidence"]), 3) if "confidence" in row and not pd.isna(row["confidence"]) else None,
                "call_id": row["call_id"],
                "utt_idx": int(row["utt_idx"]),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = args.output_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print(f"{len(manifest_rows)}개 클립을 {args.output_dir}에 저장했습니다.")
    print(f"목록: {manifest_path}")
    print()
    print(manifest_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
