#!/usr/bin/env python3
"""CNN 앙상블의 발화별 P(신고자)를 계산한다 (scripts/mission2_inference.py 가 0/1 을 정할 때 쓰는 바로 그 숫자).

mission2_inference.py 는 0/1 판정만 쓴다. CNN 을 다른 모델과 섞으려면 확률이 필요하므로, 발화마다 똑같은 과정(그 발화의 소리 구간,
log-mel 스펙트로그램, 300프레임까지 채우기, 앙상블 멤버 softmax 평균)을 거쳐 평균 P(신고자)를 쓴다.

두 가지 방식:
* ``--label_dir``: 폴더 한 쌍의 모든 발화 (채점 명령과 같은 방식, 공식 Validation 에 사용).
* ``--rows-csv``: ``audio file name,startAt,endAt`` 열이 있는 CSV 에 적힌 발화만 (다른 모델을 잰 떼어 둔 Training 발화 20,000개).
  이때 ``--audio_dir`` 아래를 하위 폴더까지 뒤져 wav 를 찾는다.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dcc_data.mission2_checkpoint import (  # noqa: E402
    l2_member_state_dicts,
    require_l2_inference_config,
    resolve_l2_checkpoint_metadata,
)
from dcc_data.mission2_cnn import SpeakerCNNLarge  # noqa: E402
from dcc_data.mission2_features import N_FFT, read_call_waveform, slice_and_pad, utterance_frame_range  # noqa: E402
from dcc_data.mission2_spectrogram import mel_spectrogram  # noqa: E402
from mission2_inference import _index_by_stem, count_label_utterances, pad_spectrogram_to_min_frames, pair_by_stem_strict  # noqa: E402


def load_ensemble(ckpt_path: Path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    metadata, _ = resolve_l2_checkpoint_metadata(checkpoint, ckpt_path)
    require_l2_inference_config(metadata)
    models = []
    for state_dict in l2_member_state_dicts(checkpoint, metadata):
        model = SpeakerCNNLarge()
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    return models, int(metadata["sample_rate"])


@torch.no_grad()
def p1_of(models, segment: np.ndarray, sample_rate: int) -> float:
    spectrogram = pad_spectrogram_to_min_frames(mel_spectrogram(segment, sample_rate))
    batch = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0)
    probabilities = sum(torch.softmax(model(batch), dim=1) for model in models) / len(models)
    return float(probabilities[0, 1])


def utterance_segment(waveform: np.ndarray, sample_rate: int, start_at, end_at) -> np.ndarray:
    start_frame, end_frame = utterance_frame_range(start_at, end_at, sample_rate, waveform.shape[0])
    return slice_and_pad(waveform, start_frame, end_frame, N_FFT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, default=None)
    parser.add_argument("--rows-csv", type=Path, default=None)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.label_dir is None) == (args.rows_csv is None):
        parser.error("give exactly one of --label_dir / --rows-csv")

    models, expected_sample_rate = load_ensemble(args.ckpt_path)
    started = time.time()
    rows: list[tuple] = []

    if args.label_dir is not None:
        import json

        pairs = pair_by_stem_strict(args.audio_dir, args.label_dir)
        total = count_label_utterances(json_path for _, _, json_path in pairs)
        for number, (stem, wav_path, json_path) in enumerate(pairs, start=1):
            waveform, sample_rate = read_call_waveform(wav_path)
            if sample_rate != expected_sample_rate:
                raise SystemExit(f"{wav_path}: {sample_rate} Hz, checkpoint expects {expected_sample_rate} Hz")
            with json_path.open("r", encoding="utf-8") as stream:
                utterances = json.load(stream).get("utterances", [])
            for utterance in utterances:
                segment = utterance_segment(waveform, sample_rate, utterance["startAt"], utterance["endAt"])
                rows.append((wav_path.name, utterance["startAt"], utterance["endAt"], p1_of(models, segment, sample_rate)))
            if number % 100 == 0:
                print(f"  {number}/{len(pairs)} calls, {len(rows)}/{total} utterances, {(time.time() - started) / 60:.1f} min",
                      flush=True)
        if len(rows) != total:
            raise SystemExit(f"wrote {len(rows)} rows for {total} labelled utterances")
    else:
        with args.rows_csv.open("r", encoding="utf-8", newline="") as stream:
            wanted = list(csv.DictReader(stream))
        wavs = _index_by_stem(args.audio_dir, "*.wav")
        by_wav: dict[str, list[dict]] = {}
        for row in wanted:
            by_wav.setdefault(row["audio file name"], []).append(row)
        for number, (name, group) in enumerate(sorted(by_wav.items()), start=1):
            paths = wavs.get(Path(name).stem)
            if not paths:
                raise SystemExit(f"{name} not found under {args.audio_dir}")
            waveform, sample_rate = read_call_waveform(paths[0])
            for row in group:
                start_at, end_at = float(row["startAt"]), float(row["endAt"])
                segment = utterance_segment(waveform, sample_rate, start_at, end_at)
                rows.append((name, row["startAt"], row["endAt"], p1_of(models, segment, sample_rate)))
            if number % 50 == 0:
                print(f"  {number}/{len(by_wav)} calls, {len(rows)}/{len(wanted)} utterances, {(time.time() - started) / 60:.1f} min",
                      flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["audio file name", "startAt", "endAt", "p1_cnn"])
        for name, start_at, end_at, p1 in rows:
            writer.writerow([name, start_at, end_at, f"{p1:.8f}"])
    print(f"wrote {len(rows)} rows to {args.output} in {(time.time() - started) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
