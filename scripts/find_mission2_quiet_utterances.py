#!/usr/bin/env python3
"""Find near-silent / noise-only Mission 2 utterances and pull audio clips
to listen to (mentoring question: "사람 목소리가 거의 없는 데이터는 어떻게
해야 할까?").

Uses the already-cached 53-dim feature parquet (rms_mean is the low-energy
signal) joined against the utterance index (for wav_path/startAt/endAt), no
re-extraction needed. Flags the bottom --quiet-percentile of rms_mean as
"quiet candidates", reports how big that bucket is and how it splits by
speaker/duration, then crops a random sample of the actual audio out to
--output-dir so they can be listened to directly.

Usage (from prelim-l1-compare):
    ../.venv/Scripts/python.exe scripts/find_mission2_quiet_utterances.py \\
        --data-root /c/dcc_data_local_full \\
        --quiet-percentile 2.0 --n-per-speaker 8
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

from dcc_data.mission2_analysis import DURATION_LABELS, bucket_duration  # noqa: E402
from dcc_data.mission2_features import read_call_waveform, utterance_frame_range  # noqa: E402

SPEAKER_LABELS = {0: "대원", 1: "신고자"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=Path("data/local_cache/m2_features_train_local_full.parquet"))
    parser.add_argument("--index", type=Path, default=Path("data/local_cache/m2_utterances_train_local_full.parquet"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--quiet-percentile", type=float, default=2.0, help="bottom N%% of rms_mean flagged as quiet")
    parser.add_argument("--n-per-speaker", type=int, default=8, help="how many quiet clips per speaker class to export")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/mission2/quiet_samples"))
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

    features_df = pd.read_parquet(args.features)
    index_df = pd.read_parquet(args.index)

    merged = features_df.merge(
        index_df[["call_id", "utt_idx", "wav_path", "startAt", "endAt"]],
        on=["call_id", "utt_idx"],
        how="inner",
    )
    dropped = len(features_df) - len(merged)
    if dropped:
        print(f"[warn] {dropped} feature row(s) had no matching index entry, dropped", file=sys.stderr)

    threshold = float(np.percentile(merged["rms_mean"], args.quiet_percentile))
    merged["is_quiet"] = merged["rms_mean"] <= threshold
    merged["duration_bucket"] = bucket_duration(merged["duration_sec"])

    quiet_df = merged[merged["is_quiet"]].copy()

    print(f"전체 발화: {len(merged)}개, rms_mean 하위 {args.quiet_percentile}% 임계값: {threshold:.6f}")
    print(f"조용한 후보 발화: {len(quiet_df)}개 ({len(quiet_df) / len(merged) * 100:.2f}%)")
    print()

    print("speaker별 분포 (조용한 후보 중):")
    by_speaker = quiet_df.groupby("speaker").size()
    for speaker, count in by_speaker.items():
        overall = (merged["speaker"] == speaker).sum()
        label = SPEAKER_LABELS.get(int(speaker), str(speaker))
        print(f"  {label}({speaker}): {count}개 (전체 {label} 발화의 {count / overall * 100:.2f}%)")
    print()

    print("발화 길이 구간별 분포 (조용한 후보 중):")
    by_duration = quiet_df.groupby("duration_bucket", observed=True).size()
    for label in DURATION_LABELS:
        if label in by_duration.index:
            print(f"  {label}: {by_duration[label]}개")
    print()

    summary_path = args.output_dir / "summary.md"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        f"# Mission 2 조용한/노이즈 후보 발화 요약\n",
        f"전체 발화: {len(merged)}개\n",
        f"rms_mean 하위 {args.quiet_percentile}% 임계값: {threshold:.6f}\n",
        f"조용한 후보: {len(quiet_df)}개 ({len(quiet_df) / len(merged) * 100:.2f}%)\n",
        "\n## speaker별 분포\n",
    ]
    for speaker, count in by_speaker.items():
        overall = (merged["speaker"] == speaker).sum()
        label = SPEAKER_LABELS.get(int(speaker), str(speaker))
        summary_lines.append(f"- {label}({speaker}): {count}개 (전체 {label} 발화의 {count / overall * 100:.2f}%)\n")
    summary_lines.append("\n## 발화 길이 구간별 분포\n")
    for label in DURATION_LABELS:
        if label in by_duration.index:
            summary_lines.append(f"- {label}: {by_duration[label]}개\n")
    summary_path.write_text("".join(summary_lines), encoding="utf-8")

    quiet_csv_path = args.output_dir / "quiet_candidates.csv"
    quiet_df[
        ["call_id", "utt_idx", "speaker", "duration_sec", "rms_mean", "rms_std", "zcr_mean", "spectral_flatness_mean"]
    ].to_csv(quiet_csv_path, index=False)

    rng = np.random.default_rng(args.seed)
    picks = []
    for speaker in (0, 1):
        pool = quiet_df[quiet_df["speaker"] == speaker]
        n = min(args.n_per_speaker, len(pool))
        if n == 0:
            continue
        chosen_idx = rng.choice(pool.index.to_numpy(), size=n, replace=False)
        for row_idx in chosen_idx:
            picks.append(pool.loc[row_idx])

    clips_dir = args.output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    waveform_cache: dict[str, tuple[np.ndarray, int]] = {}
    manifest_rows = []
    for row in picks:
        wav_path = args.data_root / row["wav_path"]
        cache_key = row["wav_path"]
        if cache_key not in waveform_cache:
            waveform_cache[cache_key] = read_call_waveform(wav_path)
        waveform, sample_rate = waveform_cache[cache_key]

        start_frame, end_frame = utterance_frame_range(row["startAt"], row["endAt"], sample_rate, waveform.shape[0])
        segment = waveform[start_frame:end_frame]

        speaker_label = SPEAKER_LABELS.get(int(row["speaker"]), str(row["speaker"]))
        out_name = f"{speaker_label}__{row['call_id']}_utt{int(row['utt_idx'])}.wav"
        out_path = clips_dir / out_name
        write_wav_segment(out_path, segment, sample_rate)

        manifest_rows.append(
            {
                "file": out_name,
                "화자": speaker_label,
                "duration_sec": round(float(row["duration_sec"]), 3),
                "rms_mean": round(float(row["rms_mean"]), 6),
                "zcr_mean": round(float(row["zcr_mean"]), 4),
                "spectral_flatness_mean": round(float(row["spectral_flatness_mean"]), 4),
                "call_id": row["call_id"],
                "utt_idx": int(row["utt_idx"]),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = args.output_dir / "clips_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print(f"조용한 후보 전체 목록: {quiet_csv_path} ({len(quiet_df)}행)")
    print(f"요약: {summary_path}")
    print(f"청취용 클립 {len(manifest_rows)}개: {clips_dir}")
    print(f"클립 목록: {manifest_path}")
    print()
    print(manifest_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
