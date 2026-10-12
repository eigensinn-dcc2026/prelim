#!/usr/bin/env python3
"""Export per-utterance CNN ensemble P(신고자) on the official Validation set.

The preprocessing is the submission inference path (pad to 300 frames, log-mel,
softmax of each packed seed). This script does not modify that code. Each
utterance is scored from its own startAt/endAt span. Call-level results are
cached under outputs/m2_offval/parts so a rerun skips finished calls.

The final parquet and summary are written only when the 0.5-threshold accuracy
is exactly 102,419 / 111,947.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_ROOT = REPOSITORY_ROOT / "submission" / "mission2"
for _root in (SUBMISSION_ROOT / "src", SUBMISSION_ROOT / "scripts"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import numpy as np  # noqa: E402

from dcc_data.mission2_checkpoint import (  # noqa: E402
    l2_member_state_dicts,
    require_l2_inference_config,
    resolve_l2_checkpoint_metadata,
)
from dcc_data.mission2_features import N_FFT, read_call_waveform, slice_and_pad, utterance_frame_range  # noqa: E402
from mission2_inference import pad_spectrogram_to_min_frames  # noqa: E402

EXPECTED_SHA256 = "19ea482a80a2ceb2ea5b87981be0a381a11e01fd576b3ba6860ee5d7144e3246"
EXPECTED_CALLS = 3640
EXPECTED_UTTERANCES = 111947
EXPECTED_CORRECT = 102419
DEFAULT_CKPT = Path(
    r"C:\Users\sooyah\Desktop\아주대\dcc\01_prelim\package_rehearsal3\mission2\checkpoints\m2_cnn_large_rerun_ensemble.pt"
)

_MODELS = None
_EXPECTED_SR = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validation_pairs(repo: Path) -> list[tuple[str, str, Path, Path]]:
    raw = repo / "data" / "raw"
    if not raw.is_dir():
        sibling = repo.parent / "dcc2026-prelim" / "data" / "raw"
        if sibling.is_dir():
            raw = sibling
        else:
            raise SystemExit(f"official data not found at {repo / 'data' / 'raw'}")
    parts = sorted(path for path in raw.iterdir() if path.is_dir() and path.name[-3:] in {"001", "002", "003", "004"})
    if len(parts) != 4:
        raise SystemExit(f"expected 4 Validation zip parts, found {len(parts)} under {raw}")
    wavs: dict[str, tuple[str, Path]] = {}
    jsons: dict[str, tuple[str, Path]] = {}
    for part in parts:
        base = part / "대학부 데이터" / "Validation"
        wav_root = base / "1.원천데이터" / "VS_서울_구급"
        json_root = base / "2.라벨링데이터" / "VL_서울_구급"
        if not wav_root.is_dir() or not json_root.is_dir():
            raise SystemExit(f"missing Validation audio or labels under {part.name}")
        for path in wav_root.glob("*.wav"):
            if path.stem in wavs:
                raise SystemExit(f"duplicate wav stem {path.stem}")
            wavs[path.stem] = (part.name, path)
        for path in json_root.glob("*.json"):
            if path.stem in jsons:
                raise SystemExit(f"duplicate label stem {path.stem}")
            jsons[path.stem] = (part.name, path)
    if set(wavs) != set(jsons):
        raise SystemExit(f"wav/json stems differ: wavs={len(wavs)} jsons={len(jsons)}")
    if len(wavs) != EXPECTED_CALLS:
        raise SystemExit(f"expected {EXPECTED_CALLS} calls, found {len(wavs)}")
    return [(stem, wavs[stem][0], wavs[stem][1], jsons[stem][1]) for stem in sorted(wavs)]


def _init_worker(ckpt_path: str) -> None:
    import torch
    from dcc_data.mission2_cnn import SpeakerCNNLarge

    torch.set_num_threads(1)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    metadata, _source = resolve_l2_checkpoint_metadata(checkpoint, ckpt_path)
    require_l2_inference_config(metadata)
    models = []
    for state_dict in l2_member_state_dicts(checkpoint, metadata):
        model = SpeakerCNNLarge()
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    if len(models) != 3:
        raise RuntimeError(f"expected 3 CNN members, found {len(models)}")
    global _MODELS, _EXPECTED_SR
    _MODELS = models
    _EXPECTED_SR = int(metadata["sample_rate"])


def _member_probabilities(segment: np.ndarray, sample_rate: int) -> list[float]:
    import torch
    import torch.nn.functional as functional
    from dcc_data.mission2_spectrogram import mel_spectrogram

    if sample_rate != _EXPECTED_SR:
        raise RuntimeError(f"audio is {sample_rate} Hz but the checkpoint was trained at {_EXPECTED_SR} Hz")
    spectrogram = pad_spectrogram_to_min_frames(mel_spectrogram(segment, sample_rate))
    batch = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0)
    with torch.inference_mode():
        probabilities = [functional.softmax(model(batch), dim=1)[0, 1] for model in _MODELS]
        return [float(probability.item()) for probability in probabilities]


def _part_path(parts_dir: Path, call_id: str) -> Path:
    return parts_dir / f"{hashlib.sha1(call_id.encode('utf-8')).hexdigest()}.json"


def _score_call(task: tuple[str, str, str, str, str]) -> str:
    call_id, part_name, wav_path, json_path, part_file = task
    destination = Path(part_file)
    if destination.exists():
        return call_id
    with open(json_path, "r", encoding="utf-8") as stream:
        utterances = json.load(stream).get("utterances", [])
    waveform, sample_rate = read_call_waveform(Path(wav_path))
    rows = []
    for utt_idx, utterance in enumerate(utterances):
        start_at = utterance["startAt"]
        end_at = utterance["endAt"]
        start_frame, end_frame = utterance_frame_range(start_at, end_at, sample_rate, waveform.shape[0])
        segment = slice_and_pad(waveform, start_frame, end_frame, N_FFT)
        probabilities = _member_probabilities(segment, sample_rate)
        if len(probabilities) != 3:
            raise RuntimeError(f"{call_id} utterance {utt_idx} did not return 3 member probabilities")
        rows.append(
            [
                utt_idx,
                start_at,
                end_at,
                (float(end_at) - float(start_at)) / 1000.0,
                int(utterance["speaker"]),
                probabilities[0],
                probabilities[1],
                probabilities[2],
            ]
        )
    payload = {
        "file": Path(wav_path).name,
        "call_id": call_id,
        "part": part_name,
        "rows": rows,
    }
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, destination)
    return call_id


def _load_rows(parts_dir: Path, pairs: list[tuple[str, str, Path, Path]]) -> list[dict]:
    records = []
    part_calls: dict[str, int] = {}
    for call_id, part_name, wav_path, _json_path in pairs:
        payload = json.loads(_part_path(parts_dir, call_id).read_text(encoding="utf-8"))
        if payload["call_id"] != call_id or payload["file"] != wav_path.name:
            raise SystemExit(f"cached part does not match {call_id}")
        part_calls[part_name] = part_calls.get(part_name, 0) + 1
        for utt_idx, start_at, end_at, duration_sec, label, p1, p2, p3 in payload["rows"]:
            mean = (float(p1) + float(p2) + float(p3)) / 3.0
            records.append(
                {
                    "file": payload["file"],
                    "call_id": call_id,
                    "utt_idx": int(utt_idx),
                    "startAt": start_at,
                    "endAt": end_at,
                    "duration_sec": float(duration_sec),
                    "label": int(label),
                    "p_cnn_s1": float(p1),
                    "p_cnn_s2": float(p2),
                    "p_cnn_s3": float(p3),
                    "p_cnn_mean": mean,
                    "_part": part_name,
                }
            )
    return records, part_calls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "outputs" / "m2_offval")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    digest = sha256_file(args.ckpt)
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"checkpoint sha256 is {digest}, expected {EXPECTED_SHA256}")
    pairs = validation_pairs(REPOSITORY_ROOT)
    parts_dir = args.output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for call_id, part_name, wav_path, json_path in pairs:
        part_file = _part_path(parts_dir, call_id)
        if not part_file.exists():
            pending.append((call_id, part_name, str(wav_path), str(json_path), str(part_file)))
    print(f"calls={len(pairs)} pending={len(pending)} workers={args.workers}", flush=True)
    started = time.perf_counter()
    if pending:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with context.Pool(args.workers, initializer=_init_worker, initargs=(str(args.ckpt),)) as pool:
            done = 0
            for _call_id in pool.imap_unordered(_score_call, pending, chunksize=1):
                done += 1
                if done % 50 == 0 or done == len(pending):
                    elapsed = time.perf_counter() - started
                    print(f"scored {done}/{len(pending)} in {elapsed:.1f}s", flush=True)
    elapsed = time.perf_counter() - started

    records, part_calls = _load_rows(parts_dir, pairs)
    if len(records) != EXPECTED_UTTERANCES:
        raise SystemExit(f"expected {EXPECTED_UTTERANCES} utterances, built {len(records)} -- not writing results")
    import pandas as pd

    frame = pd.DataFrame(records).sort_values(["file", "utt_idx"], kind="mergesort").reset_index(drop=True)
    predicted = (frame["p_cnn_mean"] >= 0.5).astype(int)
    labels = frame["label"].astype(int)
    correct = int((predicted == labels).sum())
    matrix = [[int(((labels == 0) & (predicted == 0)).sum()), int(((labels == 0) & (predicted == 1)).sum())],
              [int(((labels == 1) & (predicted == 0)).sum()), int(((labels == 1) & (predicted == 1)).sum())]]
    recall = {
        "대원": matrix[0][0] / max(matrix[0][0] + matrix[0][1], 1),
        "신고자": matrix[1][1] / max(matrix[1][0] + matrix[1][1], 1),
    }
    print(
        f"correct={correct}/{len(frame)} accuracy={correct / len(frame):.6f} "
        f"recall={recall} matrix={matrix} seconds={elapsed:.1f}",
        flush=True,
    )
    if correct != EXPECTED_CORRECT:
        raise SystemExit(
            f"accuracy {correct}/{len(frame)} is not {EXPECTED_CORRECT}/{EXPECTED_UTTERANCES}; "
            "leaving part files in place and not writing the parquet or summary"
        )
    columns = [
        "file", "call_id", "utt_idx", "startAt", "endAt", "duration_sec", "label",
        "p_cnn_s1", "p_cnn_s2", "p_cnn_s3", "p_cnn_mean",
    ]
    parquet_path = args.output_dir / "cnn_offval_probs.parquet"
    summary_path = args.output_dir / "cnn_offval_summary.json"
    frame[columns].to_parquet(parquet_path, index=False)
    summary = {
        "accuracy": correct / EXPECTED_UTTERANCES,
        "correct": correct,
        "utterances": EXPECTED_UTTERANCES,
        "threshold": 0.5,
        "recall": recall,
        "confusion_matrix": {"labels": [0, 1], "matrix": matrix},
        "seconds": elapsed,
        "checkpoint_sha256": digest,
        "calls_per_part": part_calls,
        "parquet": str(parquet_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {parquet_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
