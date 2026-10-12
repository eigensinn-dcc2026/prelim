#!/usr/bin/env python3
"""Mission 2 inference entry point (contest-mandated CLI contract).

    python scripts/mission2_inference.py \\
        --audio_dir {wav folder} --label_dir {json folder} \\
        --ckpt_path {checkpoint file} --output ./outputs/mission2.csv

audio_dir/label_dir are paired by filename stem with no Training/Validation
subfolder assumption, since the grader hands us a flat folder (this is why
dcc_data.manifest.build_manifest -- which requires exactly one Training or
Validation path component -- is not used here; see pair_by_stem_strict below,
which is dcc_data.mission2_features.pair_by_stem plus the checks a submission
needs).

The run fails, without writing a CSV, if a WAV has no label JSON (or the
reverse), if a stem appears twice, or if the finished output does not have one
row per labelled utterance -- a partial CSV must never look like a successful
run.

Row order follows each label file's own utterance order, and startAt/endAt
are copied through unmodified, per the contest's output format
([audio file name], [startAt], [endAt], [speaker]).

If feature extraction or prediction fails for a given utterance, the row is
still emitted (falling back to the majority class) so the output row count
always equals the input utterance count -- the failure is logged to stderr
with the stem/utt_idx instead of silently dropping a row.

Supports both models this repo has produced, selected by ``--ckpt_path``'s
file extension -- no separate flag needed, so the CLI contract above is
unchanged either way:
- L1 (GBM over acoustic-statistic features): any non-torch extension, e.g.
  ``.pkl`` (``dcc_data.mission2_train.save_model``'s pickle format).
- L2 (CNN over a log-mel spectrogram, ``dcc_data.mission2_cnn.SpeakerCNNLarge``):
  ``.pt``/``.pth``/``.ckpt``, matching ``scripts/train_mission2_cnn.py``'s
  ``--model large`` checkpoint format. The spectrogram
  is built here with ``dcc_data.mission2_spectrogram``'s current module-level
  defaults (``N_MELS``, and ``mission2_features.HOP_LENGTH``) -- these must
  match whatever the checkpoint was actually trained with, since a mismatch
  would not raise (``SpeakerCNNLarge`` accepts any spectrogram size via
  ``AdaptiveAvgPool2d``), it would just quietly degrade predictions. So the
  checkpoint's recorded training config (``checkpoint["metadata"]``, written by
  the trainer; ``dcc_data.mission2_checkpoint``) is checked against them first,
  and the run fails if the model kind, mel settings or training split differ,
  if the audio's sample rate isn't the one it was trained on, or if the
  checkpoint was not trained on the provided speaker labels as they are (the
  organizers' 2026-10-01 ruling: no label correction on the basis of ``text``).
  A checkpoint with no recorded config is refused too. The earlier
  ``m2_cnn_large.pt`` (M2-VAL-003) was trained on corrected labels, so it is
  deliberately not accepted any more.

An L2 ``--ckpt_path`` may also be an *ensemble* checkpoint built by
``scripts/pack_mission2_l2_ensemble.py`` (several same-architecture, same-mel-
settings models that only differ by ``--torch-seed``, soft-voted together --
see M2-ENS-003: +0.50pp over the best single seed on the full official
Validation set). Still exactly one ``--ckpt_path`` file, so the CLI contract
is unchanged; the ensemble's member weights are bundled inside that one file
under ``checkpoint["ensemble_model_state_dicts"]`` instead of a single
``"best_model_state_dict"``, and ``dcc_data.mission2_checkpoint.
ENSEMBLE_STATE_DICTS_KEY`` is the key the loader checks for.

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
from typing import Callable

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dcc_data.mission2_features import (  # noqa: E402
    FEATURE_COLUMNS,
    N_FFT,
    read_call_waveform,
    slice_and_pad,
    utterance_feature_vector,
    utterance_frame_range,
)
from dcc_data.mission2_checkpoint import (  # noqa: E402
    CheckpointConfigError,
    l2_member_state_dicts,
    require_l2_inference_config,
    resolve_l2_checkpoint_metadata,
)
from dcc_data.mission2_train import load_model  # noqa: E402

# torch.save's default format (a zip archive) -- these three are what this
# repo's own CNN scripts (train_mission2_cnn.py, evaluate_mission2_l2_on_
# official_validation.py) already use for L2 checkpoints; anything else is
# treated as L1's pickle format via dcc_data.mission2_train.load_model.
CNN_CHECKPOINT_SUFFIXES = frozenset({".pt", ".pth", ".ckpt"})

PredictFn = Callable[[np.ndarray, int, float], int]

# The CNN was trained on batches padded (with SPEC_FLOOR_DB, see mission2_cnn.collate_batch) to each
# batch's longest utterance, so it almost never saw an unpadded short utterance; feeding one at
# inference costs ~1.1pp (M2-VAL-004). Padding every utterance up to a fixed length -- using only its own
# spectrogram, never another utterance -- recovers that on the Training-internal held-out calls
# (89.58% -> 90.84% at 300; 200-500 is a plateau). 300 frames = 6 s at this repo's 8 kHz / hop 160.
L2_MIN_FRAMES = 300


def pad_spectrogram_to_min_frames(spectrogram: np.ndarray, min_frames: int = L2_MIN_FRAMES) -> np.ndarray:
    from dcc_data.mission2_spectrogram import SPEC_FLOOR_DB

    n_mels, n_frames = spectrogram.shape
    if n_frames >= min_frames:
        return spectrogram
    padding = np.full((n_mels, min_frames - n_frames), SPEC_FLOOR_DB, dtype=spectrogram.dtype)
    return np.concatenate([spectrogram, padding], axis=1)


def _speaker_from_logits(logits) -> int:
    """The contest's decision rule for one utterance's two logits: class 1 iff P(speaker=1) >= 0.5.

    For two logits that is ``logit1 >= logit0``. ``argmax`` differs only at an exact tie, where it returns
    class 0 -- so use this instead of ``logits.argmax``.
    """

    return int((logits[0, 1] >= logits[0, 0]).item())


def _index_by_stem(root: Path, pattern: str) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for path in sorted(Path(root).rglob(pattern)):
        found.setdefault(path.stem, []).append(path)
    return found


def _preview(items, limit: int = 5) -> str:
    items = list(items)
    shown = ", ".join(str(item) for item in items[:limit])
    return f"{shown}, ... (+{len(items) - limit} more)" if len(items) > limit else shown


def pair_by_stem_strict(audio_dir: Path, label_dir: Path) -> list[tuple[str, Path, Path]]:
    """What dcc_data.mission2_features.pair_by_stem returns, or a ValueError naming every problem.

    pair_by_stem keeps only the stems present in *both* folders and lets a repeated stem silently
    overwrite another, which is fine for building training data but not for a submission: a WAV without
    its label JSON (or the reverse) would just vanish from the CSV and the run would still exit 0.
    """

    wavs = _index_by_stem(audio_dir, "*.wav")
    jsons = _index_by_stem(label_dir, "*.json")

    problems = []
    if not wavs and not jsons:
        problems.append(f"no .wav files under {audio_dir} and no .json files under {label_dir}")
    duplicated_wavs = sorted(stem for stem, paths in wavs.items() if len(paths) > 1)
    if duplicated_wavs:
        problems.append(f"{len(duplicated_wavs)} WAV stem(s) appear more than once: {_preview(duplicated_wavs)}")
    duplicated_jsons = sorted(stem for stem, paths in jsons.items() if len(paths) > 1)
    if duplicated_jsons:
        problems.append(f"{len(duplicated_jsons)} label JSON stem(s) appear more than once: {_preview(duplicated_jsons)}")
    wav_only = sorted(set(wavs) - set(jsons))
    if wav_only:
        problems.append(f"{len(wav_only)} WAV stem(s) have no label JSON: {_preview(wav_only)}")
    json_only = sorted(set(jsons) - set(wavs))
    if json_only:
        problems.append(f"{len(json_only)} label JSON stem(s) have no WAV: {_preview(json_only)}")
    if problems:
        raise ValueError(
            f"audio/label inputs do not line up ({len(wavs)} WAV stems, {len(jsons)} label stems) -- "
            "refusing to write a partial CSV:\n  - " + "\n  - ".join(problems)
        )
    return [(stem, wavs[stem][0], jsons[stem][0]) for stem in sorted(wavs)]


def count_label_utterances(json_paths) -> int:
    total = 0
    for json_path in json_paths:
        with Path(json_path).open("r", encoding="utf-8") as stream:
            total += len(json.load(stream).get("utterances", []))
    return total


def verify_output_complete(*, expected_calls: int, calls_seen: int, expected_rows: int, rows: list) -> None:
    if calls_seen != expected_calls or len(rows) != expected_rows:
        raise ValueError(
            f"output is incomplete: processed {calls_seen} of {expected_calls} calls and produced "
            f"{len(rows)} rows for {expected_rows} labelled utterances -- refusing to write a partial CSV"
        )


def _load_l1_predictor(ckpt_path: Path) -> tuple[PredictFn, int]:
    model, meta = load_model(ckpt_path)
    feature_columns = tuple(meta["feature_columns"])
    if feature_columns != FEATURE_COLUMNS:
        raise ValueError(
            "checkpoint feature_columns do not match the current feature extractor "
            "(dcc_data.mission2_features.FEATURE_COLUMNS) -- retrain or update the checkpoint"
        )
    fallback_speaker = int(meta.get("majority_class", 1))

    def predict(segment: np.ndarray, sample_rate: int, duration_sec: float) -> int:
        vector = utterance_feature_vector(segment, sample_rate, duration_sec)
        return int(model.predict(vector.reshape(1, -1).astype(np.float64))[0])

    return predict, fallback_speaker


def _load_l2_predictor(ckpt_path: Path) -> tuple[PredictFn, int]:
    import torch
    import torch.nn.functional as torch_functional

    from dcc_data.mission2_cnn import SpeakerCNNLarge
    from dcc_data.mission2_spectrogram import mel_spectrogram

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    # A SpeakerCNNLarge-shaped state_dict loads without error whatever it was trained as, so check what
    # the checkpoint says it was trained with before trusting it (see dcc_data.mission2_checkpoint).
    metadata, _source = resolve_l2_checkpoint_metadata(checkpoint, ckpt_path)
    require_l2_inference_config(metadata)
    expected_sample_rate = int(metadata["sample_rate"])

    # An ensemble checkpoint (scripts/pack_mission2_l2_ensemble.py) bundles several members' weights under
    # one key instead of a single "best_model_state_dict" -- everything else about it (one --ckpt_path
    # file, same mel settings) is unchanged, so this is the only branch point.
    models = []
    for state_dict in l2_member_state_dicts(checkpoint, metadata):
        model = SpeakerCNNLarge()
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    # No stored majority_class for L2 checkpoints (train_mission2_cnn.py
    # doesn't compute one) -- 1 matches load_model's own fallback-of-a-
    # fallback default above, so both backends agree when nothing better is
    # known. This only fires if the call's audio couldn't even be read.
    fallback_speaker = 1

    @torch.no_grad()
    def predict(segment: np.ndarray, sample_rate: int, duration_sec: float) -> int:
        if sample_rate != expected_sample_rate:
            raise CheckpointConfigError(
                f"audio is {sample_rate} Hz but this checkpoint was trained on {expected_sample_rate} Hz audio"
            )
        spectrogram = pad_spectrogram_to_min_frames(mel_spectrogram(segment, sample_rate))
        batch = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, n_frames)
        if len(models) == 1:
            # Bit-for-bit the single-model path this always was -- comparing softmax outputs instead of
            # raw logits would be mathematically equivalent (softmax is order-preserving) but there is no
            # reason to touch the already-tested tie-breaking behavior when there is nothing to average.
            return _speaker_from_logits(models[0](batch))
        average_probs = sum(torch_functional.softmax(model(batch), dim=1) for model in models) / len(models)
        return int((average_probs[0, 1] >= average_probs[0, 0]).item())

    return predict, fallback_speaker


def load_predictor(ckpt_path: Path) -> tuple[PredictFn, int]:
    """Return (predict_fn, fallback_speaker) for whichever model ``ckpt_path`` is."""

    if ckpt_path.suffix.lower() in CNN_CHECKPOINT_SUFFIXES:
        return _load_l2_predictor(ckpt_path)
    return _load_l1_predictor(ckpt_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predict, fallback_speaker = load_predictor(args.ckpt_path)

    pairs = pair_by_stem_strict(args.audio_dir, args.label_dir)
    expected_rows = count_label_utterances(json_path for _, _, json_path in pairs)

    rows: list[dict] = []
    failure_count = 0
    calls_seen = 0

    for stem, wav_path, json_path in pairs:
        calls_seen += 1
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
                    speaker = predict(segment, sample_rate, duration_sec)
                except CheckpointConfigError:
                    raise  # a checkpoint/input mismatch is a setup error, never one bad utterance to paper over
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

    verify_output_complete(expected_calls=len(pairs), calls_seen=calls_seen, expected_rows=expected_rows, rows=rows)

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
