"""Mission 2 baseline error-analysis helpers (post-hoc, metadata-only).

These operate on the already-computed index/feature parquets and a
prediction array -- no audio is re-read here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DURATION_BUCKETS = [0.0, 0.5, 1.0, 2.0, 4.0, float("inf")]
DURATION_LABELS = ["<0.5s", "0.5-1s", "1-2s", "2-4s", ">=4s"]

# The dispatcher's near-universal fixed opening line. Used only to *find*
# candidate mislabeled calls for human review -- never fed to the model
# (mission2_features never touches "text" as a feature; this is a one-time
# data-quality audit, a text-based stand-in for actually listening to the
# call, see reports/mission2/m2_label_swap_audit.md).
DISPATCHER_GREETING_MARKERS = ("119입니다", "119 입니다", "119예요", "119 상황실")


def compute_overlap_flags(index_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (call_id, utt_idx) with an ``overlaps_other`` bool.

    Two utterances overlap if their [startAt, endAt) ranges intersect. This
    only needs the index parquet (pure metadata) -- no audio touched. The
    team EDA found 37.1% of utterances overlap another utterance in the same
    call; this flag lets baseline accuracy be checked against that split.
    """

    rows = []
    for call_id, group in index_df.groupby("call_id"):
        starts = group["startAt"].to_numpy(dtype=float)
        ends = group["endAt"].to_numpy(dtype=float)
        n = len(group)
        if n > 1:
            overlap_matrix = (starts[:, None] < ends[None, :]) & (starts[None, :] < ends[:, None])
            np.fill_diagonal(overlap_matrix, False)
            flags = overlap_matrix.any(axis=1)
        else:
            flags = np.zeros(n, dtype=bool)
        rows.append(
            pd.DataFrame(
                {"call_id": call_id, "utt_idx": group["utt_idx"].to_numpy(), "overlaps_other": flags}
            )
        )
    return pd.concat(rows, ignore_index=True)


def bucket_duration(duration_sec: pd.Series) -> pd.Series:
    return pd.cut(duration_sec, bins=DURATION_BUCKETS, labels=DURATION_LABELS, right=False)


def select_listening_samples(
    df: pd.DataFrame, *, n_per_group: int, seed: int
) -> dict[str, pd.DataFrame]:
    """Sample a few rows per (role, correctness) group for manual listening.

    ``df`` must have ``speaker`` (0/1), ``correct`` (bool-like) and
    ``overlaps_other`` (bool-like) columns -- the shape produced by
    scripts/analyze_mission2_errors.py. The two "정답_안겹침" groups
    deliberately exclude overlapping utterances so a listener hears each
    role's typical sound uncontaminated by the other speaker's voice
    bleeding into the crop; the "오답" groups are not filtered by overlap
    since misclassified rows are scarcer.
    """

    clean = ~df["overlaps_other"].astype(bool)
    correct = df["correct"].astype(bool)

    definitions = {
        "대원_정답_안겹침": (df["speaker"] == 0) & correct & clean,
        "신고자_정답_안겹침": (df["speaker"] == 1) & correct & clean,
        "대원_오답": (df["speaker"] == 0) & ~correct,
        "신고자_오답": (df["speaker"] == 1) & ~correct,
    }
    groups: dict[str, pd.DataFrame] = {}
    for name, mask in definitions.items():
        pool = df[mask]
        n = min(n_per_group, len(pool))
        if n > 0:
            groups[name] = pool.sample(n=n, random_state=seed)
    return groups


def find_label_swap_candidates(
    calls: list[tuple[str, list[dict]]], *, exclude_call_ids: frozenset[str] = frozenset()
) -> list[dict]:
    """Flag calls whose speaker labels look reversed for the *whole* call.

    ``calls`` is (call_id, utterances) pairs, each utterances a list of the
    JSON's own utterance dicts (needs "text" and "speaker"). Looks at the
    first 3 utterances for a DISPATCHER_GREETING_MARKERS phrase and flags the
    call if that utterance's speaker isn't 0 (the established convention:
    대원=0) -- 대원's opening line is one of the few places in a call where
    the *content* alone (not tone/audio) unambiguously identifies the role,
    which is exactly why it works as a cheap, no-audio-needed first pass.

    High precision on the calls it does flag, not full recall: it misses any
    call whose dispatcher skipped this exact phrasing, and it will also flag
    a small number of false positives (a 119-to-119 inter-region transfer
    call, where the *other* party also introduces themselves with "119입니다"
    while functionally acting as the reporter for this call) -- both classes
    of error, plus the calls this already confirmed, are catalogued in
    reports/mission2/m2_label_swap_audit.md. Every flagged call still needs a
    human read/listen before any bulk relabeling; this only narrows a
    dataset-wide search down to a short list.
    """

    candidates: list[dict] = []
    for call_id, utterances in calls:
        if call_id in exclude_call_ids:
            continue
        anchor = None
        for utterance in utterances[:3]:
            text = utterance.get("text", "")
            if any(marker in text for marker in DISPATCHER_GREETING_MARKERS):
                anchor = utterance
                break
        if anchor is not None and anchor.get("speaker") != 0:
            candidates.append(
                {
                    "call_id": call_id,
                    "anchor_speaker": anchor.get("speaker"),
                    "anchor_text": anchor.get("text"),
                    "n_utterances": len(utterances),
                }
            )
    return candidates
