#!/usr/bin/env python3
"""Refine the quiet-candidate pool with a real voice-activity measure.

rms_mean (a single mean over the whole utterance) can't tell "genuinely
silent/noise-only the whole way through" apart from "a real, if quiet,
utterance". This computes, per utterance, the fraction of the segment that
librosa's energy-based voice-activity split considers non-silent
(``librosa.effects.split``, top_db relative to that segment's own peak) --
a much more direct signal than a single aggregate.

Only touches the utterances already flagged by
scripts/find_mission2_quiet_utterances.py (quiet_candidates.csv), so it
re-reads ~6.2k call WAVs instead of the full 29.2k-call dataset.

Usage:
    ../.venv/Scripts/python.exe scripts/compute_mission2_silence_ratio.py \\
        --data-root /c/dcc_data_local_full
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

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dcc_data.mission2_features import read_call_waveform, utterance_frame_range  # noqa: E402

SPEAKER_LABELS = {0: "대원", 1: "신고자"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quiet-candidates", type=Path, default=Path("reports/mission2/quiet_samples/quiet_candidates.csv"))
    parser.add_argument("--index", type=Path, default=Path("data/local_cache/m2_utterances_train_local_full.parquet"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--top-db", type=float, default=30.0, help="librosa.effects.split threshold below segment peak")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/mission2/quiet_samples"))
    parser.add_argument("--n-per-speaker", type=int, default=10, help="how many top-silence_ratio clips per speaker to export")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_wav_segment(path: Path, segment: np.ndarray, sample_rate: int) -> None:
    pcm16 = (np.clip(segment, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def silence_ratio(segment: np.ndarray, top_db: float) -> float:
    if segment.size == 0 or np.max(np.abs(segment)) == 0:
        return 1.0
    intervals = librosa.effects.split(segment, top_db=top_db)
    voiced_frames = int(np.sum(intervals[:, 1] - intervals[:, 0])) if len(intervals) else 0
    return 1.0 - (voiced_frames / len(segment))


def main() -> int:
    args = parse_args()

    quiet_df = pd.read_csv(args.quiet_candidates, dtype={"call_id": str})
    index_df = pd.read_parquet(args.index)

    merged = quiet_df.merge(
        index_df[["call_id", "utt_idx", "wav_path", "startAt", "endAt"]],
        on=["call_id", "utt_idx"],
        how="left",
    )
    missing = merged["wav_path"].isna().sum()
    if missing:
        print(f"[warn] {missing} row(s) had no matching wav_path, dropped", file=sys.stderr)
        merged = merged.dropna(subset=["wav_path"])

    unique_calls = merged["wav_path"].nunique()
    print(f"처리할 발화: {len(merged)}개, 열어야 할 통화 wav: {unique_calls}개")

    # Sort so each call's utterances are contiguous -- lets us hold only the
    # current call's waveform in memory (single-entry cache) instead of all
    # ~6.2k call WAVs at once.
    merged = merged.sort_values(["wav_path", "utt_idx"]).reset_index(drop=True)

    ratios = np.empty(len(merged), dtype=np.float64)
    current_wav_path = None
    current_waveform = None
    current_sample_rate = None
    calls_done = 0
    for pos, row in enumerate(merged.itertuples(index=False)):
        if row.wav_path != current_wav_path:
            current_waveform, current_sample_rate = read_call_waveform(args.data_root / row.wav_path)
            current_wav_path = row.wav_path
            calls_done += 1
            if calls_done % 500 == 0:
                print(f"  ...{calls_done}/{unique_calls} 통화 처리")

        start_frame, end_frame = utterance_frame_range(
            row.startAt, row.endAt, current_sample_rate, current_waveform.shape[0]
        )
        segment = current_waveform[start_frame:end_frame]
        ratios[pos] = silence_ratio(segment, args.top_db)

    merged["silence_ratio"] = ratios

    args.output_dir.mkdir(parents=True, exist_ok=True)
    refined_csv = args.output_dir / "quiet_candidates_with_silence_ratio.csv"
    merged.drop(columns=["wav_path", "startAt", "endAt"]).to_csv(refined_csv, index=False)

    print()
    print("silence_ratio 분포 (조용한 후보 16,527개 중):")
    print(merged["silence_ratio"].describe())
    print()

    high_silence = merged[merged["silence_ratio"] >= 0.8]
    print(f"silence_ratio >= 0.8 (사실상 80% 이상 무음/노이즈): {len(high_silence)}개")
    by_speaker = high_silence.groupby("speaker").size()
    for speaker, count in by_speaker.items():
        label = SPEAKER_LABELS.get(int(speaker), str(speaker))
        pool_count = (merged["speaker"] == speaker).sum()
        print(f"  {label}({speaker}): {count}개 (조용한 후보 중 {label}의 {count / pool_count * 100:.1f}%)")

    rng = np.random.default_rng(args.seed)
    picks = []
    for speaker in (0, 1):
        pool = high_silence[high_silence["speaker"] == speaker]
        n = min(args.n_per_speaker, len(pool))
        if n == 0:
            continue
        chosen_idx = rng.choice(pool.index.to_numpy(), size=n, replace=False)
        for row_idx in chosen_idx:
            picks.append(merged.loc[row_idx])

    clips_dir = args.output_dir / "clips_round2"
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    clip_cache: dict[str, tuple[np.ndarray, int]] = {}
    for row in picks:
        wav_path = args.data_root / row["wav_path"]
        cache_key = row["wav_path"]
        if cache_key not in clip_cache:
            clip_cache[cache_key] = read_call_waveform(wav_path)
        waveform, sample_rate = clip_cache[cache_key]
        start_frame, end_frame = utterance_frame_range(row["startAt"], row["endAt"], sample_rate, waveform.shape[0])
        segment = waveform[start_frame:end_frame]

        speaker_label = SPEAKER_LABELS.get(int(row["speaker"]), str(row["speaker"]))
        out_name = f"{speaker_label}__{row['call_id']}_utt{int(row['utt_idx'])}.wav"
        write_wav_segment(clips_dir / out_name, segment, sample_rate)
        manifest_rows.append(
            {
                "file": out_name,
                "화자": speaker_label,
                "duration_sec": round(float(row["duration_sec"]), 3),
                "silence_ratio": round(float(row["silence_ratio"]), 3),
                "rms_mean": round(float(row["rms_mean"]), 6),
                "call_id": row["call_id"],
                "utt_idx": int(row["utt_idx"]),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = args.output_dir / "clips_round2_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print()
    print(f"전체 결과(silence_ratio 포함): {refined_csv}")
    print(f"2차 청취용 클립 {len(manifest_rows)}개: {clips_dir}")
    print(f"클립 목록: {manifest_path}")
    print()
    print(manifest_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
