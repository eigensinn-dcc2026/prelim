#!/usr/bin/env python3
"""Pack a spectrogram cache (one .npy per utterance) into one float16 array for fast training.

scripts/build_mission2_spectrograms.py writes one small .npy per utterance -- ~890k files for the full
Seoul Training set. Opening every one of them every epoch is slow, and at float32 the set (~14 GB) is larger
than a Colab runtime's RAM, so it can't stay cached. This writes:

* ``--packed-output``: every spectrogram transposed to (n_frames, n_mels), stacked along the frame axis, as
  float16 (~7 GB) -- read with np.load(mmap_mode="r") by dcc_data.mission2_cnn.PackedSpectrogramDataset;
* ``--index-output``: the input index, sorted by (call_id, utt_idx) so the result doesn't depend on the order a
  parallel build finished in, plus a ``frame_offset`` column (where each row's frames start). Every other
  column -- labels and the split/settings/label_source provenance train_mission2_cnn.py checks -- is copied
  unchanged.

Every file's shape is checked against its index row (n_mels and n_frames) before it is packed.

Usage:
    python scripts/pack_mission2_spectrograms.py \\
        --index data/m2_spectrogram_index.parquet --spectrogram-dir data/m2_spectrograms \\
        --packed-output data/m2_spectrograms_packed.npy --index-output data/m2_spectrogram_index_packed.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PACKED_DTYPE = np.float16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--spectrogram-dir", type=Path, required=True)
    parser.add_argument("--packed-output", type=Path, required=True)
    parser.add_argument("--index-output", type=Path, required=True)
    return parser.parse_args()


def pack_spectrograms(index_df: pd.DataFrame, spectrogram_dir: Path, packed_output: Path) -> pd.DataFrame:
    """Write the packed array and return the sorted index with ``frame_offset`` added."""

    if "frame_offset" in index_df.columns:
        raise ValueError("index already has a frame_offset column -- it is already a packed index")
    n_mels_values = index_df["n_mels"].unique()
    if len(n_mels_values) != 1:
        raise ValueError(f"index mixes several n_mels values: {sorted(n_mels_values)}")
    n_mels = int(n_mels_values[0])

    packed_df = index_df.sort_values(["call_id", "utt_idx"], kind="stable").reset_index(drop=True)
    n_frames = packed_df["n_frames"].to_numpy(dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(n_frames)[:-1]]).astype(np.int64)
    packed_df["frame_offset"] = offsets

    packed_output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = packed_output.with_name(packed_output.name + ".tmp.npy")
    try:
        _write_packed(packed_df, spectrogram_dir, tmp_output, n_mels)
        tmp_output.replace(packed_output)
    except BaseException:
        tmp_output.unlink(missing_ok=True)
        raise
    return packed_df


def _write_packed(packed_df: pd.DataFrame, spectrogram_dir: Path, output: Path, n_mels: int) -> None:
    n_frames = packed_df["n_frames"].to_numpy(dtype=np.int64)
    offsets = packed_df["frame_offset"].to_numpy(dtype=np.int64)
    packed = np.lib.format.open_memmap(output, mode="w+", dtype=PACKED_DTYPE, shape=(int(n_frames.sum()), n_mels))
    try:
        for i, npy_path in enumerate(packed_df["npy_path"]):
            spec = np.load(spectrogram_dir / npy_path)
            if spec.shape != (n_mels, n_frames[i]):
                raise ValueError(
                    f"{npy_path}: shape {spec.shape} does not match its index row (n_mels={n_mels}, "
                    f"n_frames={n_frames[i]})"
                )
            packed[offsets[i] : offsets[i] + n_frames[i]] = spec.T
            if (i + 1) % 100_000 == 0:
                print(f"  packed {i + 1}/{len(packed_df)}", flush=True)
        packed.flush()
    finally:
        del packed  # closes the file (Windows can't delete or rename a file that is still mapped)


def main() -> int:
    args = parse_args()
    start = time.monotonic()
    index_df = pd.read_parquet(args.index)
    packed_df = pack_spectrograms(index_df, args.spectrogram_dir, args.packed_output)
    args.index_output.parent.mkdir(parents=True, exist_ok=True)
    packed_df.to_parquet(args.index_output, index=False)
    total_frames = int(packed_df["n_frames"].sum())
    print(f"Packed {len(packed_df)} spectrograms ({total_frames} frames) -> {args.packed_output}")
    print(f"Packed index -> {args.index_output}")
    print(f"Elapsed: {time.monotonic() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
