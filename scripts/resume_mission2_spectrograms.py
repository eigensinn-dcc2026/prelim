#!/usr/bin/env python3
"""Resume a Mission 2 spectrogram cache without recomputing calls that already finished.

``scripts/build_mission2_spectrograms.py`` writes one ``<output_dir>/<call_id>/<utt_idx>.npy``
per utterance and only writes the index parquet and failures log after every sampled call.
A crashed run can leave hundreds of thousands of ``.npy`` files and no index.
``scripts/pack_mission2_spectrograms.py`` needs that index. This script rebuilds it.

Same arguments as the builder (including the per-split default paths), plus
``--expected-utterances``. Sampling uses the same ``select_seoul_training_rows`` /
``select_seoul_validation_rows`` and ``sample_calls(rows, n_calls, seed)`` the builder
uses, so the call list matches a builder run in the same interpreter.

``sample_calls`` is deterministic. When ``n_calls <= 0`` (including the crashed run's
``--n-calls -1``) or ``n_calls >= len(rows)`` it returns every row it was given, in
that order, and does not call ``random``. The rows it is given are already filtered
by ``select_seoul_training_rows`` or ``select_seoul_validation_rows``, so ``-1`` means
every eligible call, not every row of the 32,840-row manifest. Otherwise it returns
``random.Random(seed).sample(rows, n_calls)``, which is repeatable for a given
CPython version and the same input order. Commit ``4bfd2c4``
(the crashed Colab build) and current main have identical blobs for
``scripts/build_mission2_spectrograms.py``, ``src/dcc_data/mission2_spectrogram.py``,
``src/dcc_data/mission2_features.py``, and ``src/dcc_data/manifest.py`` (and for
``audio.py``, ``mission2.py``, and ``guards.py``, which that path calls). Nothing that
changes which calls are sampled, which utterances exist, or the ``.npy`` contents
differs, so this script calls the current functions rather than a private fork.

``build_utterance_records`` reads only the label JSON (``startAt``, ``endAt``, ``speaker``,
``_id``). It does not open audio. A call is reused only when every utterance ``.npy``
exists, ``np.load(mmap_mode="r")`` succeeds, and the array is 2-D ``float32`` with
``shape[0] == n_mels`` and at least one frame. Reused index rows copy the builder's
columns. ``n_frames`` comes from ``shape[1]``. ``read_call_waveform`` does not resample:
it returns the PCM as stored and the WAV header rate (``wave.getframerate`` via
``read_wav_info`` / ``read_wav_frames``). Reused rows take ``sample_rate`` from
``read_wav_info`` (header only, no PCM decode), which is the integer a fresh build
records. ``soundfile.info`` reads that same header rate on these PCM files; it is not
a direct project dependency, so the value stored is the builder's own header read.

Any missing, truncated, unreadable, or wrong-shaped file recomputes the whole call
with ``_process_one_call_spectrograms``, which overwrites that call's files. Utterances
the original run failed (no ``.npy``, and no failures log, because that log is written
only at the end) are retried this way. Pass the same ``--hop-length`` and ``--n-mels``
as the crashed build. ``n_mels`` is checked against ``shape[0]``; ``hop_length`` is
stamped from the argument and is not re-derived from the array.

Index order matches a single-worker builder: ``sample_calls`` order, and within a call
the label-JSON utterance order (``utt_idx`` ascending). With ``--workers`` greater than
1 the work runs in a process pool, then the rows are written back in that same order.
``pack_spectrograms`` stable-sorts by ``(call_id, utt_idx)`` before it assigns
``frame_offset`` and writes the packed array, so the packed index and packed ``.npy``
depend on the row contents, not on this order. The official index parquet, failures
jsonl, and ``<index-output>.resume_summary.json`` are written only after every
sampled call is done. A crash and a rerun re-check the ``.npy`` files. They do not
read ``<index-output>.partial.parquet``.

Write every output and log to local disk, not a Google Drive mount. The crashed
Colab process wrote stderr and the failures log straight to Drive, and the mount
write failed. Defaults match the builder's relative ``data/`` paths, which are local
only when the working directory is local. On Colab, pass ``/content/...`` for
``--index-output``, ``--failures-log``, and ``--progress-log``. The partial index
and the resume summary are siblings of ``--index-output``, so that path has to be
local too. ``--output-dir`` for the crashed run is the existing local cache
``/content/m2_spec_train``.

``--progress-log`` appends one flushed line every ``--progress-every`` finished
calls (default 500) and on the last call: UTC timestamp plus calls reused, calls
recomputed, utterances indexed, and failures. ``--partial-index-every N`` atomically
writes the index rows collected so far (sample order, finished calls only) to
``<index-output>.partial.parquet`` via a temp file in the same directory and
``os.replace``. That checkpoint is a bonus. Correctness does not depend on it.

Usage (same arguments as the crashed 4bfd2c4 build, plus the utterance check and
local index/log paths)::

    python scripts/resume_mission2_spectrograms.py \\
        --manifest data/manifests/dcc_manifest_v1.csv \\
        --data-root <data> \\
        --split training \\
        --n-calls -1 --seed 42 --workers 2 \\
        --output-dir /content/m2_spec_train \\
        --index-output /content/m2_spectrogram_index.parquet \\
        --failures-log /content/m2_spectrogram_build_failures.jsonl \\
        --expected-utterances 889358 \\
        --progress-log /content/m2_spectrogram_resume_progress.log \\
        --progress-every 500 \\
        --partial-index-every 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np  # noqa: E402

from dcc_data.audio import read_wav_info  # noqa: E402
from dcc_data.manifest import ManifestRow, load_manifest  # noqa: E402
from dcc_data.mission2_features import (  # noqa: E402
    HOP_LENGTH,
    N_FFT,
    build_utterance_records,
    sample_calls,
    select_seoul_training_rows,
    select_seoul_validation_rows,
)
from dcc_data.mission2_spectrogram import (  # noqa: E402
    LABEL_SOURCE_PROVIDED,
    N_MELS,
    _process_one_call_spectrograms,
    _spectrogram_path,
)

SELECT_ROWS = {
    "training": select_seoul_training_rows,
    "validation": select_seoul_validation_rows,
}

# Kept in step with scripts/build_mission2_spectrograms.py. The two splits must not share a default.
DEFAULT_CACHE_PATHS = {
    "training": {
        "output_dir": Path("data/m2_spectrograms"),
        "index_output": Path("data/m2_spectrogram_index.parquet"),
        "failures_log": Path("data/m2_spectrogram_build_failures.jsonl"),
    },
    "validation": {
        "output_dir": Path("data/m2_spectrograms_validation"),
        "index_output": Path("data/m2_spectrogram_index_validation.parquet"),
        "failures_log": Path("data/m2_spectrogram_build_failures_validation.jsonl"),
    },
}


def select_and_sample(rows: list[ManifestRow], split: str, n_calls: int, seed: int) -> list[ManifestRow]:
    """Same call list as ``build_seoul_*_spectrograms`` for this split.

    See the module docstring for why ``sample_calls`` is deterministic, including
    the ``n_calls <= 0`` full-list path the Colab build uses.
    """

    return sample_calls(SELECT_ROWS[split](rows), n_calls, seed)


def npy_n_frames_if_reusable(path: Path, n_mels: int) -> int | None:
    """Frame count when ``path`` is a finished spectrogram, else ``None``.

    Opens the array with ``np.load(mmap_mode="r")`` and does not decode audio.
    Reusable means a non-empty 2-D ``float32`` array whose first axis is ``n_mels``.
    """

    if not path.is_file():
        return None
    try:
        spec = np.load(path, mmap_mode="r")
    except Exception:  # noqa: BLE001 - a truncated or foreign file is recomputed, not fatal
        return None
    try:
        if getattr(spec, "ndim", None) != 2 or spec.dtype != np.dtype(np.float32):
            return None
        if spec.shape[0] != n_mels or spec.shape[1] <= 0:
            return None
        return int(spec.shape[1])
    finally:
        mmap = getattr(spec, "_mmap", None)
        if mmap is not None:
            mmap.close()
        del spec


def _index_row(
    row: ManifestRow,
    record,
    path: Path,
    output_dir: Path,
    n_frames: int,
    *,
    hop_length: int,
    n_mels: int,
    sample_rate: int,
) -> dict:
    # Same keys, in the same order, as _process_one_call_spectrograms.
    return {
        "call_id": record.call_id,
        "utt_idx": record.utt_idx,
        "speaker": record.speaker,
        "duration_sec": record.duration_sec,
        "n_frames": n_frames,
        "npy_path": str(path.relative_to(output_dir)),
        "split": row.split,
        "n_mels": n_mels,
        "hop_length": hop_length,
        "n_fft": N_FFT,
        "sample_rate": int(sample_rate),
        "label_source": LABEL_SOURCE_PROVIDED,
    }


def reuse_index_rows(
    row: ManifestRow,
    records: list,
    data_root: Path,
    output_dir: Path,
    hop_length: int,
    n_mels: int,
) -> list[dict] | None:
    """Index rows for one call when every utterance file is reusable, else ``None``.

    Does not decode PCM. ``sample_rate`` is the WAV header rate from ``read_wav_info``.
    A call with no utterances is reusable without opening the WAV: the builder would
    also emit no rows (an audio-open failure with an empty utterance list logs nothing).
    """

    output_dir = Path(output_dir)
    frames: list[tuple] = []
    for record in records:
        path = _spectrogram_path(output_dir, record.call_id, record.utt_idx)
        n_frames = npy_n_frames_if_reusable(path, n_mels)
        if n_frames is None:
            return None
        frames.append((record, path, n_frames))
    if not records:
        return []
    try:
        sample_rate = int(read_wav_info(Path(data_root) / row.wav_path).sample_rate)
    except Exception:  # noqa: BLE001 - header unreadable: recompute, which logs the builder's failure
        return None
    return [
        _index_row(
            row, record, path, output_dir, n_frames, hop_length=hop_length, n_mels=n_mels, sample_rate=sample_rate
        )
        for record, path, n_frames in frames
    ]


def _resume_one_call(
    row: ManifestRow,
    records: list | None,
    data_root: Path,
    output_dir: Path,
    hop_length: int,
    n_mels: int,
) -> tuple[list[dict], list[dict], str]:
    """Reuse one call's files or recompute the whole call.

    Top-level so a process-pool worker can pickle it. ``records is None`` means the
    label JSON could not be read while counting utterances; the builder's per-call
    routine logs that failure. Returns ``(index_rows, failures, "reused"|"recomputed")``.
    """

    if records is not None:
        reused = reuse_index_rows(row, records, data_root, output_dir, hop_length, n_mels)
        if reused is not None:
            return reused, [], "reused"
    index_rows, failures = _process_one_call_spectrograms(
        row, Path(data_root), Path(output_dir), hop_length, n_mels
    )
    return index_rows, failures, "recomputed"


def _load_records(sampled: list[ManifestRow], data_root: Path, expected_utterances: int | None):
    """Read every sampled call's label JSON. No spectrogram work happens here."""

    prepared: list[tuple[ManifestRow, list | None]] = []
    total_utterances = 0
    total_calls = len(sampled)
    for index, row in enumerate(sampled, start=1):
        try:
            records = build_utterance_records(row, data_root)
        except Exception as exc:  # noqa: BLE001
            if expected_utterances is not None:
                raise RuntimeError(
                    f"label read failed for stem {row.stem!r} before the --expected-utterances check"
                ) from exc
            prepared.append((row, None))
        else:
            total_utterances += len(records)
            prepared.append((row, records))
        if index == 1 or index == total_calls or index % 2000 == 0:
            print(f"labels {index}/{total_calls} utterances_so_far={total_utterances}", flush=True)
    print(f"Utterances in sample: {total_utterances}", flush=True)
    if expected_utterances is not None and total_utterances != expected_utterances:
        raise AssertionError(
            f"utterance count {total_utterances} != --expected-utterances {expected_utterances} "
            f"({total_calls} sampled calls)"
        )
    return prepared, total_utterances


def resume_summary_path(index_output: Path) -> Path:
    return Path(str(index_output) + ".resume_summary.json")


def partial_index_path(index_output: Path) -> Path:
    """Bonus checkpoint. Resume never reads this path."""

    return Path(str(index_output) + ".partial.parquet")


def _progress_line(done: int, total: int, reused: int, recomputed: int, utterances: int, failures: int) -> str:
    stamp = datetime.now(timezone.utc).isoformat()
    return (
        f"{stamp} done={done} total={total} reused={reused} recomputed={recomputed} "
        f"utterances_indexed={utterances} failures={failures}"
    )


def _append_progress(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()


def _collected_rows(results: list) -> tuple[list[dict], int]:
    """Index rows of finished calls, in sample order, plus how many failures they logged."""

    rows: list[dict] = []
    failure_count = 0
    for item in results:
        if item is None:
            continue
        index_rows, failures, _status = item
        rows.extend(index_rows)
        failure_count += len(failures)
    return rows, failure_count


def _write_partial_index(index_rows: list[dict], index_output: Path) -> None:
    """Atomically replace ``<index-output>.partial.parquet`` with the rows collected so far."""

    import pandas as pd

    partial = partial_index_path(index_output)
    partial.parent.mkdir(parents=True, exist_ok=True)
    temporary = partial.with_name(partial.name + ".tmp")
    try:
        pd.DataFrame(index_rows).to_parquet(temporary, index=False)
        temporary.replace(partial)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _resume_all(
    prepared,
    data_root: Path,
    output_dir: Path,
    hop_length: int,
    n_mels: int,
    workers: int,
    bar,
    *,
    progress_every: int,
    progress_log: Path | None,
    partial_index_every: int | None,
    index_output: Path,
):
    total = len(prepared)
    reused = 0
    recomputed = 0
    results: list = [None] * total

    def account(status: str) -> None:
        nonlocal reused, recomputed
        if status == "reused":
            reused += 1
        else:
            recomputed += 1

    def after_call(done: int) -> None:
        if bar is not None:
            bar.update(1)
        log_now = done == total or done % progress_every == 0
        partial_now = partial_index_every is not None and (done == total or done % partial_index_every == 0)
        if not log_now and not partial_now:
            return
        collected, failure_count = _collected_rows(results)
        if log_now:
            line = _progress_line(done, total, reused, recomputed, len(collected), failure_count)
            print(line, flush=True)
            if progress_log is not None:
                _append_progress(progress_log, line)
        if partial_now:
            _write_partial_index(collected, index_output)

    if workers <= 1:
        for index, (row, records) in enumerate(prepared, start=1):
            results[index - 1] = _resume_one_call(row, records, data_root, output_dir, hop_length, n_mels)
            account(results[index - 1][2])
            after_call(index)
        return results

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_resume_one_call, row, records, data_root, output_dir, hop_length, n_mels): index
            for index, (row, records) in enumerate(prepared)
        }
        done = 0
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            done += 1
            account(results[index][2])
            after_call(done)
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=list(SELECT_ROWS),
        default="training",
        help="training (default) or validation -- the official Validation folder, eval-only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"default per split: {DEFAULT_CACHE_PATHS['training']['output_dir']} (training) / "
        f"{DEFAULT_CACHE_PATHS['validation']['output_dir']} (validation). "
        "Point this at the directory that already holds the crashed run's .npy files.",
    )
    parser.add_argument(
        "--n-calls",
        type=int,
        default=3000,
        help="Passed through to sample_calls after the split's select_*_rows filter. "
        "-1 (and any value <= 0) keeps every eligible call in manifest order. "
        "The crashed Colab build used -1 on the 32840-row manifest.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--index-output",
        type=Path,
        default=None,
        help=f"default per split: {DEFAULT_CACHE_PATHS['training']['index_output']} (training) / "
        f"{DEFAULT_CACHE_PATHS['validation']['index_output']} (validation). "
        "Local disk only. On Colab use a /content path; do not use a Google Drive mount. "
        "The resume summary and partial index are written beside this path.",
    )
    parser.add_argument(
        "--failures-log",
        type=Path,
        default=None,
        help=f"default per split: {DEFAULT_CACHE_PATHS['training']['failures_log']} (training) / "
        f"{DEFAULT_CACHE_PATHS['validation']['failures_log']} (validation). "
        "Local disk only. The crashed run lost this log on a Drive mount write.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--hop-length",
        type=int,
        default=HOP_LENGTH,
        help=f"STFT hop length in samples (default {HOP_LENGTH}, same as the builder). "
        "Must match the hop used for the .npy files already in --output-dir.",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=N_MELS,
        help=f"number of mel frequency bands (default {N_MELS}). A file whose first axis "
        "is not this value is treated as not reusable.",
    )
    parser.add_argument(
        "--expected-utterances",
        type=int,
        default=None,
        help="If set, require the sampled calls' label utterances to total this number "
        "before any spectrogram is read or written (the Colab Training run uses 889358).",
    )
    parser.add_argument(
        "--progress-log",
        type=Path,
        default=None,
        help="Append a flushed progress line every --progress-every calls and on the last call. "
        "Optional. Local disk only.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Finished calls between progress lines (default 500). Also controls the stdout line.",
    )
    parser.add_argument(
        "--partial-index-every",
        type=int,
        default=None,
        help="Every N finished calls, atomically write rows collected so far to "
        "<index-output>.partial.parquet. Optional bonus checkpoint. Resume never reads it; "
        "reuse is always re-derived from the .npy files. Local disk only.",
    )
    args = parser.parse_args(argv)
    if args.progress_every <= 0:
        parser.error("--progress-every must be a positive integer")
    if args.partial_index_every is not None and args.partial_index_every <= 0:
        parser.error("--partial-index-every must be a positive integer")
    for name, default in DEFAULT_CACHE_PATHS[args.split].items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    return args


def _jsonable_args(args: argparse.Namespace) -> dict:
    payload = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def main() -> int:
    args = parse_args()
    rows = load_manifest(args.manifest, data_root=args.data_root)
    sampled = select_and_sample(rows, args.split, args.n_calls, args.seed)
    print(f"Sampled calls: {len(sampled)}", flush=True)
    print(f"Index output (local disk): {args.index_output}", flush=True)
    print(f"Failures log (local disk): {args.failures_log}", flush=True)
    if args.progress_log is not None:
        print(f"Progress log (local disk): {args.progress_log}", flush=True)

    # Label JSON only. The utterance-count check finishes before any .npy read or write.
    prepared, _total_utterances = _load_records(sampled, args.data_root, args.expected_utterances)

    try:
        from tqdm import tqdm

        bar = tqdm(total=len(prepared), unit="call")
    except ImportError:
        bar = None

    start = time.monotonic()
    try:
        results = _resume_all(
            prepared,
            args.data_root,
            args.output_dir,
            args.hop_length,
            args.n_mels,
            args.workers,
            bar,
            progress_every=args.progress_every,
            progress_log=args.progress_log,
            partial_index_every=args.partial_index_every,
            index_output=args.index_output,
        )
    finally:
        if bar is not None:
            bar.close()

    index_rows: list[dict] = []
    failures: list[dict] = []
    calls_reused = 0
    calls_recomputed = 0
    for call_index_rows, call_failures, status in results:
        index_rows.extend(call_index_rows)
        failures.extend(call_failures)
        if status == "reused":
            calls_reused += 1
        else:
            calls_recomputed += 1

    import pandas as pd

    args.index_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(index_rows).to_parquet(args.index_output, index=False)

    args.failures_log.parent.mkdir(parents=True, exist_ok=True)
    with args.failures_log.open("w", encoding="utf-8") as stream:
        for failure in failures:
            stream.write(json.dumps(failure, ensure_ascii=False) + "\n")

    summary = {
        "calls_reused": calls_reused,
        "calls_recomputed": calls_recomputed,
        "utterances_indexed": len(index_rows),
        "failures": len(failures),
        "sampled_calls": len(sampled),
        "utterances_in_sample": _total_utterances,
        "args": _jsonable_args(args),
    }
    summary_path = resume_summary_path(args.index_output)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    elapsed = time.monotonic() - start
    print(f"Calls reused: {calls_reused}")
    print(f"Calls recomputed: {calls_recomputed}")
    print(f"Utterances indexed: {len(index_rows)}")
    print(f"Failures: {len(failures)} (see {args.failures_log})")
    print(f"Index parquet: {args.index_output}")
    print(f"Resume summary: {summary_path}")
    if args.partial_index_every is not None:
        print(f"Partial index: {partial_index_path(args.index_output)}")
    print(f"Spectrogram dir: {args.output_dir}")
    print(f"Elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
