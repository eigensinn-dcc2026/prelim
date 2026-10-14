#!/usr/bin/env python3
"""발화별 스펙트로그램 .npy 들을 float16 배열 하나로 묶어 학습을 빠르게 한다.

scripts/build_mission2_spectrograms.py 는 발화마다 작은 .npy 를 하나씩 쓴다(서울 Training 전체 약 89만 개). 에폭마다 그 파일을 모두 여는 것은
느리고, float32 로는 약 14GB 라 Colab 메모리에 계속 올려 둘 수 없다. 그래서 다음 두 가지를 쓴다.

* ``--packed-output``: 모든 스펙트로그램을 (n_frames, n_mels) 로 돌려 프레임 축으로 이어 붙인 float16 배열(약 7GB).
  dcc_data.mission2_cnn.PackedSpectrogramDataset 이 np.load(mmap_mode="r") 로 읽는다.
* ``--index-output``: 입력 목록을 (call_id, utt_idx) 순으로 정렬하고(병렬로 만든 순서에 결과가 좌우되지 않게) 각 발화의 시작 위치
  ``frame_offset`` 열을 더한 것. 라벨과 출처 정보(분할·설정·label_source, train_mission2_cnn.py 가 확인함) 등 다른 열은 그대로 옮긴다.

묶기 전에 파일마다 모양(n_mels, n_frames)이 목록과 맞는지 확인한다.

사용법 (CNN 학습 노트북의 실제 명령과 같음):
    python scripts/pack_mission2_spectrograms.py \
        --index /content/m2_spectrogram_index.parquet --spectrogram-dir /content/m2_specs \
        --packed-output /content/m2_spectrograms_packed.npy --index-output /content/m2_spectrogram_index_packed.parquet
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
    """묶은 배열을 쓰고, ``frame_offset`` 을 더한 정렬된 목록을 돌려준다."""

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
        del packed  # 파일을 닫는다 (Windows 는 메모리에 매핑된 파일을 지우거나 이름을 바꿀 수 없다)


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
