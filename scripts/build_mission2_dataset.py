#!/usr/bin/env python3
"""Build the Mission 2 utterance-level index and acoustic-feature cache (L1).

Reuses the repository's manifest/guards/audio primitives; only Training +
Seoul rows are used (dcc_data.guards.ensure_training_rows keeps Validation
data out of this path entirely).

Usage (from the repository root; run scripts/build_manifest.py first if
--manifest does not exist yet):

    python scripts/build_mission2_dataset.py \\
        --manifest data/manifests/dcc_manifest_v1.csv \\
        --data-root data/raw \\
        --n-calls 3000 --seed 42

Colab: mount Drive, point --data-root at the extracted dataset root (or a
local copy under /content/ for speed), and run scripts/build_manifest.py
first to produce the --manifest CSV.

Outputs:
  --index-output    data/m2_utterances_train.parquet
                     (call_id, wav_stem, wav_path, json_path, utt_idx,
                      startAt, endAt, duration_sec, speaker -- one row per
                      utterance in the sampled calls; utt_idx exists only to
                      restore row order later and must never be used as a
                      model feature)
  --features-output data/m2_features_train.parquet
                     (call_id, utt_idx, speaker, + the 53 feature columns --
                      only utterances whose feature extraction succeeded)
  --failures-log     data/m2_build_failures.jsonl
                     (one JSON object per skipped utterance/call, with the
                      call_id/utt_idx/reason -- nothing is dropped silently)

Partial --index-output/--features-output are written every --flush-every
calls so a Colab disconnect does not lose all progress.

Known limitation: a call whose label JSON parses fine but has an empty
`utterances: []` list produces zero rows in every sink (index/features/
failures), so --resume can't tell it apart from "never attempted" and
re-reads its audio on every resumed run. Harmless (just wasted work, not
data loss) and not expected to occur in this dataset -- every real call
here has at least one utterance -- so it's left unhandled rather than
threading an explicit "attempted calls" marker through every layer for a
case that shouldn't come up.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dcc_data.manifest import load_manifest  # noqa: E402
from dcc_data.mission2_features import (  # noqa: E402
    FEATURE_COLUMNS,
    UtteranceRecord,
    build_seoul_training_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--n-calls", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index-output", type=Path, default=Path("data/m2_utterances_train.parquet"))
    parser.add_argument("--features-output", type=Path, default=Path("data/m2_features_train.parquet"))
    parser.add_argument("--failures-log", type=Path, default=Path("data/m2_build_failures.jsonl"))
    parser.add_argument(
        "--flush-every", type=int, default=200, help="write partial outputs every N calls processed (0 disables)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="process calls in parallel across this many worker processes (1 = sequential, the default)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "load --index-output/--features-output if they already exist and skip calls already "
            "present in them (matched by wav_path), appending only the remainder -- for continuing "
            "after an interrupted run (e.g. a Colab disconnect) instead of starting over from zero. "
            "Only meaningful with --n-calls 0 (process everything); combining --resume with a "
            "positive --n-calls does not reproduce what a single uninterrupted run would have sampled."
        ),
    )
    return parser.parse_args()


def _load_existing_index(index_output: Path) -> list[UtteranceRecord]:
    import pandas as pd

    df = pd.read_parquet(index_output)
    return [
        UtteranceRecord(
            call_id=row.call_id,
            wav_stem=row.wav_stem,
            wav_path=row.wav_path,
            json_path=row.json_path,
            utt_idx=int(row.utt_idx),
            start_at=float(row.startAt),
            end_at=float(row.endAt),
            duration_sec=float(row.duration_sec),
            speaker=int(row.speaker),
        )
        for row in df.itertuples(index=False)
    ]


def _load_existing_features(features_output: Path) -> list[dict]:
    import pandas as pd

    return pd.read_parquet(features_output).to_dict("records")


def _load_existing_failures(failures_log: Path) -> list[dict]:
    if not failures_log.exists():
        return []
    with failures_log.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_failures(failures, failures_log: Path) -> None:
    failures_log.parent.mkdir(parents=True, exist_ok=True)
    with failures_log.open("w", encoding="utf-8") as stream:
        for failure in failures:
            stream.write(json.dumps(failure, ensure_ascii=False) + "\n")


def _preserve_stale_file(path: Path) -> Path:
    """Rename ``path`` out of the way without clobbering an existing backup.

    ``int(time.time())`` collides across two runs started in the same
    wall-clock second (e.g. an automated retry), which raised an uncaught
    ``FileExistsError`` on Windows instead of the intended graceful
    preserve-and-restart. Nanosecond resolution plus a fallback counter makes
    a real collision essentially impossible.
    """

    suffix = time.time_ns()
    backup = path.with_name(f"{path.stem}.stale-{suffix}{path.suffix}")
    while backup.exists():
        suffix += 1
        backup = path.with_name(f"{path.stem}.stale-{suffix}{path.suffix}")
    path.rename(backup)
    return backup


def _write_outputs(index_records, feature_rows, index_output: Path, features_output: Path) -> None:
    import pandas as pd

    index_output.parent.mkdir(parents=True, exist_ok=True)
    features_output.parent.mkdir(parents=True, exist_ok=True)

    index_df = pd.DataFrame(
        [
            {
                "call_id": r.call_id,
                "wav_stem": r.wav_stem,
                "wav_path": r.wav_path,
                "json_path": r.json_path,
                "utt_idx": r.utt_idx,
                "startAt": r.start_at,
                "endAt": r.end_at,
                "duration_sec": r.duration_sec,
                "speaker": r.speaker,
            }
            for r in index_records
        ]
    )
    index_df.to_parquet(index_output, index=False)

    features_df = pd.DataFrame(feature_rows, columns=["call_id", "utt_idx", "speaker", *FEATURE_COLUMNS])
    features_df.to_parquet(features_output, index=False)


def main() -> int:
    args = parse_args()
    rows = load_manifest(args.manifest, data_root=args.data_root)

    index_records: list = []
    feature_rows: list = []
    failures: list = []

    if args.resume:
        if args.index_output.exists() and args.features_output.exists():
            index_records = _load_existing_index(args.index_output)
            feature_rows = _load_existing_features(args.features_output)
            failures = _load_existing_failures(args.failures_log)

            # A call only counts as "done" if every utterance indexed for it
            # already has a resolved outcome (a feature row, or a logged
            # failure) -- being present in the index alone is NOT enough.
            # Index rows for a call are written before that call's audio is
            # opened/features are extracted (see _process_one_call), so if the
            # process crashes between those two steps (e.g. a worker OOM/
            # BrokenProcessPool), the call ends up indexed but with zero
            # feature rows and zero failure entries. The old check here
            # (`{record.wav_path for record in index_records}`) treated that
            # call as done and would skip it forever on every future
            # --resume, silently losing it with no error and no log entry.
            # int(...) casts: values loaded back from parquet/JSON can come
            # back as numpy/pandas integer types, which normally hash equal
            # to plain Python ints -- but the two sides of this comparison
            # are built independently, so cast explicitly rather than rely on
            # that.
            resolved_keys = {(row["call_id"], int(row["utt_idx"])) for row in feature_rows}
            resolved_keys.update(
                (failure["call_id"], int(failure["utt_idx"]))
                for failure in failures
                if failure.get("utt_idx") is not None  # stage="index" failures
                # never produced index_records in the first place, so they
                # have no counterpart to resolve here.
            )

            indexed_keys_by_wav_path: dict[str, set[tuple[str, int]]] = {}
            # call_id -> wav_path, built with an explicit uniqueness check.
            # call_id comes from the label JSON's `_id` field, which nothing
            # in the pipeline enforces as globally unique (only (split, stem)
            # is validated) -- matching feature_rows/failures for cleanup by
            # call_id alone would, on an id collision, silently delete an
            # unrelated and fully-complete call's data. Fail loudly instead;
            # in practice call_id looks like a Mongo ObjectId and this should
            # never actually fire.
            wav_path_by_call_id: dict[str, str] = {}
            for record in index_records:
                indexed_keys_by_wav_path.setdefault(record.wav_path, set()).add(
                    (record.call_id, int(record.utt_idx))
                )
                existing = wav_path_by_call_id.get(record.call_id)
                if existing is not None and existing != record.wav_path:
                    raise ValueError(
                        f"call_id {record.call_id!r} maps to two different wav_path values "
                        f"({existing!r} and {record.wav_path!r}) in {args.index_output} -- "
                        "refusing to --resume with ambiguous call_id data."
                    )
                wav_path_by_call_id[record.call_id] = record.wav_path

            incomplete_wav_paths = {
                wav_path
                for wav_path, keys in indexed_keys_by_wav_path.items()
                if not keys <= resolved_keys
            }
            done_wav_paths = set(indexed_keys_by_wav_path) - incomplete_wav_paths

            before = len(rows)
            rows = [row for row in rows if row.wav_path not in done_wav_paths]

            # Every wav_path about to be reprocessed -- whether it was
            # "incomplete" above, or never made it into the index at all
            # last time (e.g. it failed at the index/JSON-parsing stage) --
            # must have its old index/feature/failure entries dropped first.
            # Otherwise reprocessing appends on top of stale rows (silent
            # duplicates), and a call that fails identically on every
            # --resume attempt (a genuinely malformed JSON, say) would
            # otherwise accumulate one duplicate failures.jsonl entry per
            # attempt forever. Matched by wav_path throughout (not call_id)
            # for the same collision-safety reason as above -- failure
            # dicts always carry "wav_path", regardless of stage.
            reprocess_wav_paths = {row.wav_path for row in rows}
            index_records = [r for r in index_records if r.wav_path not in reprocess_wav_paths]
            feature_rows = [
                r for r in feature_rows if wav_path_by_call_id.get(r["call_id"]) not in reprocess_wav_paths
            ]
            failures = [f for f in failures if f.get("wav_path") not in reprocess_wav_paths]

            message = (
                f"--resume: 완전히 끝난 통화 {before - len(rows)}개 건너뜀 "
                f"(발화 {len(feature_rows)}개 기존 결과 유지)"
            )
            if incomplete_wav_paths:
                message += f", 인덱스엔 있었지만 미완료였던 통화 {len(incomplete_wav_paths)}개는 처음부터 재처리"
            message += f", 남은 통화 {len(rows)}개 처리 예정"
            print(message)
        else:
            # --resume needs BOTH files to reconstruct prior progress. If only
            # one exists (e.g. a crash landed between the two writes inside
            # _write_outputs), the old code silently fell through to "start
            # fresh" and then, at the end of main(), unconditionally
            # overwrote that surviving file with the fresh (much smaller)
            # run's results -- permanently discarding whatever progress it
            # held, with no warning. Rename it out of the way instead so nothing
            # is lost.
            preserved = []
            for path in (args.index_output, args.features_output):
                if path.exists():
                    backup = _preserve_stale_file(path)
                    preserved.append(str(backup))
            if preserved:
                print(
                    "--resume 지정됐지만 index/features 중 한쪽 파일만 존재해 이어할 수 없습니다 "
                    "(중단된 실행의 부분 결과로 추정). 덮어써서 잃어버리지 않도록 보존했습니다: "
                    + ", ".join(preserved)
                    + " -- 처음부터 새로 시작합니다."
                )
            else:
                print("--resume 지정됐지만 기존 출력 파일이 없어 새로 시작합니다.")

    try:
        from tqdm import tqdm

        bar = tqdm(unit="call")
    except ImportError:
        bar = None

    last_flush_call = 0

    def on_call_done(done: int, total: int) -> None:
        nonlocal last_flush_call
        if bar is not None:
            if bar.total is None:
                bar.total = total
            bar.update(done - bar.n)
        if args.flush_every > 0 and done - last_flush_call >= args.flush_every:
            _write_outputs(index_records, feature_rows, args.index_output, args.features_output)
            # failures_log used to be written only once at the very end of
            # main(), unlike index/features which flush here -- a crash
            # after this point but before the final write lost every
            # in-memory failure recorded since the last flush, even though
            # the index/feature progress for the same period survived.
            _write_failures(failures, args.failures_log)
            last_flush_call = done

    start = time.monotonic()
    build_seoul_training_dataset(
        rows,
        args.data_root,
        n_calls=args.n_calls,
        seed=args.seed,
        index_sink=index_records,
        feature_sink=feature_rows,
        failure_sink=failures,
        on_call_done=on_call_done,
        max_workers=args.workers,
    )
    if bar is not None:
        bar.close()

    _write_outputs(index_records, feature_rows, args.index_output, args.features_output)
    _write_failures(failures, args.failures_log)

    elapsed = time.monotonic() - start
    print(f"Index rows (utterances): {len(index_records)}")
    print(f"Feature rows (utterances with successful extraction): {len(feature_rows)}")
    print(f"Failures: {len(failures)} (see {args.failures_log})")
    print(f"Index parquet: {args.index_output}")
    print(f"Features parquet: {args.features_output}")
    print(f"Elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
