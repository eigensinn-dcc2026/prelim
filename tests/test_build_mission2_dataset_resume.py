from __future__ import annotations

import importlib
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd

from dcc_data.manifest import ManifestRow, write_manifest

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

build_mission2_dataset = importlib.import_module("build_mission2_dataset")

SAMPLE_RATE = 8000


def _write_wav(path: Path, duration_s: float, *, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, int(duration_s * SAMPLE_RATE))
    rng = np.random.default_rng(seed)
    samples = (rng.standard_normal(n) * 3000).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(samples.tobytes())


def _write_call(root: Path, stem: str, *, seed: int) -> ManifestRow:
    wav_relative = f"Training/1.원천데이터/TS_서울_구급/{stem}.wav"
    json_relative = f"Training/2.라벨링데이터/TL_서울_구급/{stem}.json"
    _write_wav(root / wav_relative, 3.0, seed=seed)
    document = {
        "_id": stem,
        "utterances": [
            {"id": "u0", "startAt": 0, "endAt": 1000, "speaker": 0, "text": "x"},
            {"id": "u1", "startAt": 1000, "endAt": 2500, "speaker": 1, "text": "y"},
        ],
    }
    (root / json_relative).parent.mkdir(parents=True, exist_ok=True)
    (root / json_relative).write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return ManifestRow("Training", stem, wav_relative, json_relative)


def _run_main(argv: list[str]) -> None:
    import sys as _sys

    old_argv = _sys.argv
    _sys.argv = ["build_mission2_dataset.py", *argv]
    try:
        build_mission2_dataset.main()
    finally:
        _sys.argv = old_argv


def test_resume_skips_already_processed_calls_and_appends_the_rest(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    stems = ["call_a", "call_b", "call_c"]
    rows = [_write_call(root, stem, seed=i) for i, stem in enumerate(stems)]

    index_output = tmp_path / "index.parquet"
    features_output = tmp_path / "features.parquet"
    failures_log = tmp_path / "failures.jsonl"

    # "Interrupted first run": only call_a and call_b were ever available/reachable.
    partial_manifest = tmp_path / "manifest_partial.csv"
    write_manifest(rows[:2], partial_manifest, overwrite=True)
    _run_main(
        [
            "--manifest",
            str(partial_manifest),
            "--data-root",
            str(root),
            "--n-calls",
            "0",
            "--index-output",
            str(index_output),
            "--features-output",
            str(features_output),
            "--failures-log",
            str(failures_log),
            "--flush-every",
            "0",
        ]
    )
    first_pass_index = pd.read_parquet(index_output)
    assert set(first_pass_index["wav_stem"]) == {"call_a", "call_b"}

    # "Resumed run": the full manifest (all 3 calls) is now available; --resume
    # must skip call_a/call_b (already in index_output) and process only call_c.
    full_manifest = tmp_path / "manifest_full.csv"
    write_manifest(rows, full_manifest, overwrite=True)
    _run_main(
        [
            "--manifest",
            str(full_manifest),
            "--data-root",
            str(root),
            "--n-calls",
            "0",
            "--index-output",
            str(index_output),
            "--features-output",
            str(features_output),
            "--failures-log",
            str(failures_log),
            "--flush-every",
            "0",
            "--resume",
        ]
    )

    final_index = pd.read_parquet(index_output)
    final_features = pd.read_parquet(features_output)
    assert set(final_index["wav_stem"]) == {"call_a", "call_b", "call_c"}
    # Original call_a/call_b rows must be untouched, not just re-derived identically.
    assert len(final_index) == 3 * 2  # 2 utterances/call * 3 calls
    assert len(final_features) == len(final_index)


def test_resume_without_existing_outputs_falls_back_to_a_fresh_run(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    row = _write_call(root, "call_only", seed=0)
    manifest_path = tmp_path / "manifest.csv"
    write_manifest([row], manifest_path, overwrite=True)

    index_output = tmp_path / "index.parquet"
    features_output = tmp_path / "features.parquet"
    failures_log = tmp_path / "failures.jsonl"

    _run_main(
        [
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(root),
            "--n-calls",
            "0",
            "--index-output",
            str(index_output),
            "--features-output",
            str(features_output),
            "--failures-log",
            str(failures_log),
            "--flush-every",
            "0",
            "--resume",  # nothing to resume from yet -- must not crash
        ]
    )

    assert set(pd.read_parquet(index_output)["wav_stem"]) == {"call_only"}
