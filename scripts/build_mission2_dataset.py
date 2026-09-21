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
import os
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
            "Requires --n-calls 0 (process everything); enforced, not just advisory -- with a "
            "positive --n-calls, incomplete calls the sample doesn't happen to reselect would be "
            "deleted (their stale rows purged for reprocessing) but never actually reprocessed."
        ),
    )
    args = parser.parse_args()
    if args.resume and args.n_calls != 0:
        # --resume deletes every not-yet-fully-resolved call's prior rows
        # before re-sampling (see main()), then build_seoul_training_dataset
        # samples only --n-calls of the remaining rows. With a positive
        # --n-calls, any incomplete call NOT selected by that sample is
        # deleted but never reprocessed -- real, reviewer-reproduced data
        # loss, not just "doesn't reproduce a single uninterrupted run" as
        # the old help text implied. Fail fast instead of allowing it
        # (the default --n-calls is 3000, so this previously fired silently
        # on any --resume invocation that forgot to override it).
        parser.error("--resume requires --n-calls 0 (process everything) -- see --resume's help for why")
    return args


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


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so a crash mid-write can never leave a
    truncated/corrupted file at ``path`` itself.

    Writing directly to the target (the original code's behavior) can leave
    a 0-byte or partial file there if the process dies mid-write; --resume's
    ``exists()`` check can't tell that apart from a complete file and would
    crash trying to parse it (reviewer-reproduced by corrupting the features
    parquet). Writing to a temp file in the same directory first, then
    ``os.replace`` (atomic on both POSIX and Windows for same-filesystem
    renames), means ``path`` is always either the previous complete version
    or the new complete version, never something in between.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def _write_failures(failures, failures_log: Path) -> None:
    _atomic_write_text(failures_log, "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in failures))


def _check_no_call_id_collision(index_records, *, context: str) -> dict[str, str]:
    """Return call_id -> wav_path, raising if any call_id maps to two.

    call_id is the label JSON's `_id` field; nothing in the pipeline
    enforces it as globally unique (only (split, stem) is validated).
    Matching rows across sinks by call_id alone would, on a collision,
    silently merge/delete an unrelated call's data. Fail loudly instead --
    in practice call_id looks like a Mongo ObjectId and this should never
    actually fire.
    """

    wav_path_by_call_id: dict[str, str] = {}
    for record in index_records:
        existing = wav_path_by_call_id.get(record.call_id)
        if existing is not None and existing != record.wav_path:
            raise ValueError(
                f"call_id {record.call_id!r} maps to two different wav_path values "
                f"({existing!r} and {record.wav_path!r}) {context} -- refusing to continue "
                "with ambiguous call_id data."
            )
        wav_path_by_call_id[record.call_id] = record.wav_path
    return wav_path_by_call_id


def _validate_loaded_state(index_records, feature_rows, failures, *, index_output: Path) -> None:
    """Refuse to --resume from a corrupt/inconsistent prior run rather than
    silently misclassifying it as complete.

    A plain "is this utterance's key present somewhere" subset check (the
    first version of this fix) still calls a call "done" if e.g. a feature
    row is accidentally duplicated, or a feature and a failure both exist
    for the same utterance, or a failure references an utterance that was
    never indexed. All of those are real states a partial/interrupted prior
    run's parquet files can end up in -- reviewer-reproduced by duplicating
    one feature row and observing the call get skipped, duplicate row and
    all, on --resume.
    """

    index_keys = [(r.call_id, int(r.utt_idx)) for r in index_records]
    if len(index_keys) != len(set(index_keys)):
        raise ValueError(f"duplicate (call_id, utt_idx) rows found in {index_output} -- refusing to --resume.")
    index_key_set = set(index_keys)

    feature_keys = [(row["call_id"], int(row["utt_idx"])) for row in feature_rows]
    if len(feature_keys) != len(set(feature_keys)):
        raise ValueError("duplicate (call_id, utt_idx) rows found in the features parquet -- refusing to --resume.")
    feature_key_set = set(feature_keys)

    utt_failures = [f for f in failures if f.get("utt_idx") is not None]
    failure_keys = [(f["call_id"], int(f["utt_idx"])) for f in utt_failures]
    if len(failure_keys) != len(set(failure_keys)):
        raise ValueError("duplicate (call_id, utt_idx) entries found in the failures log -- refusing to --resume.")
    failure_key_set = set(failure_keys)

    overlap = feature_key_set & failure_key_set
    if overlap:
        raise ValueError(
            f"{len(overlap)} utterance(s) have both a feature row and a failure entry "
            f"(e.g. {next(iter(overlap))!r}) -- refusing to --resume with contradictory state."
        )

    orphans = (feature_key_set | failure_key_set) - index_key_set
    if orphans:
        raise ValueError(
            f"{len(orphans)} feature/failure entries (e.g. {next(iter(orphans))!r}) reference an "
            "utterance that was never indexed -- refusing to --resume with orphaned data."
        )

    wav_path_by_key = {(r.call_id, int(r.utt_idx)): r.wav_path for r in index_records}
    for failure in utt_failures:
        key = (failure["call_id"], int(failure["utt_idx"]))
        recorded_wav_path = failure.get("wav_path")
        if key in wav_path_by_key and recorded_wav_path is not None and recorded_wav_path != wav_path_by_key[key]:
            raise ValueError(
                f"failure entry for {key!r} records wav_path {recorded_wav_path!r}, but the index has "
                f"{wav_path_by_key[key]!r} for the same key -- refusing to --resume with inconsistent data."
            )


def _preserve_and_restart_fresh(args: argparse.Namespace, reason: str) -> None:
    """Back up whatever output files exist, then let --resume fall through
    to a from-scratch run.

    Used both when the on-disk state can't be reconstructed at all (only
    one of index/features exists) and when it exists but fails to load or
    fails ``_validate_loaded_state`` (e.g. a parquet corrupted by a crash
    mid-write predating the atomic-write fix, or by something outside this
    script's control). Either way, the old behavior -- crash uncaught, or
    silently overwrite the survivor at the end of the run -- is wrong;
    preserving every existing file under a `.stale-*` name and starting
    over is always safe, just possibly wasteful of prior work.
    """

    preserved = []
    for path in (args.index_output, args.features_output, args.failures_log):
        if path.exists():
            preserved.append(str(_preserve_stale_file(path)))
    print(f"--resume: {reason} -- 처음부터 새로 시작합니다." + (f" 보존된 파일: {', '.join(preserved)}" if preserved else ""))


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


def _atomic_write_parquet(df, path: Path) -> None:
    """Same rationale as ``_atomic_write_text``, for parquet output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def _write_outputs(index_records, feature_rows, index_output: Path, features_output: Path) -> None:
    import pandas as pd

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
    _atomic_write_parquet(index_df, index_output)

    features_df = pd.DataFrame(feature_rows, columns=["call_id", "utt_idx", "speaker", *FEATURE_COLUMNS])
    _atomic_write_parquet(features_df, features_output)


def main() -> int:
    args = parse_args()
    rows = load_manifest(args.manifest, data_root=args.data_root)

    index_records: list = []
    feature_rows: list = []
    failures: list = []

    resumed = False
    if args.resume:
        # Captured once, up front: the preserve-and-restart fallback below
        # renames index/features/failures away on a read failure, which
        # would otherwise make a live `.exists()` re-check after the fact
        # see "only one file exists" and redundantly (harmlessly, but
        # confusingly) attempt to preserve an already-empty state again.
        both_existed = args.index_output.exists() and args.features_output.exists()
        if both_existed:
            try:
                index_records = _load_existing_index(args.index_output)
                feature_rows = _load_existing_features(args.features_output)
                failures = _load_existing_failures(args.failures_log)
            except Exception as exc:  # noqa: BLE001 -- deliberately broad: a corrupt
                # parquet can raise anything from pyarrow/pandas (reviewer-
                # reproduced by truncating the features file). A file that
                # can't even be *read* is unrecoverable -- fall back to
                # preserve-and-restart, same as "only one file exists".
                index_records, feature_rows, failures = [], [], []
                _preserve_and_restart_fresh(
                    args, f"기존 출력을 읽는 데 실패했습니다({type(exc).__name__}: {exc})"
                )
            else:
                # Unlike a read failure, a *validation* failure (duplicate
                # rows, orphan features, contradictory feature+failure for
                # the same utterance, ...) means the files parsed fine but
                # are logically inconsistent -- deliberately NOT caught
                # alongside the read failures above. Silently discarding
                # (possibly hours of) prior work over one bad row would be
                # its own kind of data loss; this should be a loud, blocking
                # error a human looks at, not an auto-recovery.
                _validate_loaded_state(index_records, feature_rows, failures, index_output=args.index_output)
                resumed = True

        if resumed:
            # A call only counts as "done" if every utterance indexed for it
            # already has a resolved outcome (a feature row, or a logged
            # failure) -- being present in the index alone is NOT enough.
            # _process_one_call returns index/feature/failure rows for a
            # call together, as one tuple, so a mid-call worker crash
            # (OOM/BrokenProcessPool) actually drops all three for that call
            # from the parent sinks, not just features -- an earlier version
            # of this comment claimed otherwise; that was wrong (caught in
            # PR review). The real mechanism for "indexed but unresolved" is:
            # every utterance in a call fails at the *feature* stage (not
            # audio_open), which still returns (index_records, [], failures)
            # normally -- but failures_log used to be written only once at
            # the very end of main() while index/features flushed
            # periodically (see _write_failures's call sites below, now
            # fixed to flush together). A run interrupted after such a
            # periodic flush but before the final write left exactly this
            # state on disk: indexed, zero features, and failures.jsonl
            # missing that call (or empty, if it was the only failure so
            # far) -- matching the real 2,170/29,200-call pattern this fix
            # was written against. The old completeness check here
            # (`{record.wav_path for record in index_records}`) treated any
            # indexed call as done regardless, and would skip it forever on
            # every future --resume, silently losing it with no error and no
            # log entry.
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

            # call_id -> wav_path map, collision-checked (see
            # _check_no_call_id_collision) -- used below to match
            # feature_rows (which only carry call_id) back to a wav_path.
            wav_path_by_call_id = _check_no_call_id_collision(index_records, context=f"in {args.index_output}")

            indexed_keys_by_wav_path: dict[str, set[tuple[str, int]]] = {}
            for record in index_records:
                indexed_keys_by_wav_path.setdefault(record.wav_path, set()).add(
                    (record.call_id, int(record.utt_idx))
                )

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
        elif not both_existed:
            # --resume needs BOTH files to reconstruct prior progress. If only
            # one exists (e.g. a crash landed between the two writes inside
            # _write_outputs -- still possible in principle even with the
            # atomic-write fix, since the two files are written one after
            # the other, not as a single transaction), the old code silently
            # fell through to "start fresh" and then, at the end of main(),
            # unconditionally overwrote that surviving file with the fresh
            # (much smaller) run's results. Same fallback as the load-failure
            # path above (a validation failure, by contrast, does NOT reach
            # here -- it's raised, not caught, see the `else` branch above).
            reason = (
                "index/features 중 한쪽 파일만 존재해 이어할 수 없습니다(중단된 실행의 부분 결과로 추정)"
                if (args.index_output.exists() or args.features_output.exists())
                else "기존 출력 파일이 없습니다"
            )
            _preserve_and_restart_fresh(args, reason)

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

    # Re-check after processing, not just on the initially-loaded index: a
    # newly (re)processed call's JSON `_id` could collide with an
    # already-"done" call's call_id that was loaded from disk and never
    # touched this run (reviewer-reproduced) -- the earlier check inside the
    # --resume branch only ever saw the pre-processing state.
    _check_no_call_id_collision(index_records, context="after this run's processing")

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
