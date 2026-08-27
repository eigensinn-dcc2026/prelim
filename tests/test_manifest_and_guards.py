from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from dcc_data.guards import (
    DuplicateIdentifierError,
    ManifestPairingError,
    SplitContaminationError,
    ensure_split,
    ensure_training_batch,
)
from dcc_data.manifest import ManifestRow, build_manifest, load_manifest, write_manifest


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def add_pair(root: Path, wav_relative: str, json_relative: str) -> None:
    touch(root / wav_relative)
    touch(root / json_relative)


def test_manifest_pairs_globally_across_shards_and_supports_012_layout(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    add_pair(
        root,
        "chunk-001/대학부 데이터/Training/1.원천데이터/TS_서울_구급/a.wav",
        "chunk-006/대학부 데이터/Training/2.라벨링데이터/TL_서울_구급/a.json",
    )
    add_pair(
        root,
        "chunk-012/Training/1.원천데이터/TS_서울_구급/b.wav",
        "chunk-012/Training/2.라벨링데이터/TL_서울_구급/b.json",
    )
    add_pair(
        root,
        "chunk-001/대학부 데이터/Validation/1.원천데이터/VS_서울_구급/c.wav",
        "chunk-002/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급/c.json",
    )

    rows = build_manifest(root)

    assert [(row.split, row.stem) for row in rows] == [
        ("Training", "a"),
        ("Training", "b"),
        ("Validation", "c"),
    ]
    assert "chunk-001" in rows[0].wav_path
    assert "chunk-006" in rows[0].json_path

    output = tmp_path / "manifests" / "manifest.csv"
    write_manifest(rows, output)
    assert load_manifest(output, data_root=root) == rows
    with pytest.raises(FileExistsError):
        write_manifest(rows, output)


def test_manifest_rejects_unpaired_and_duplicate_files(tmp_path: Path) -> None:
    unpaired = tmp_path / "unpaired"
    touch(unpaired / "chunk/Training/audio/lonely.wav")
    with pytest.raises(ManifestPairingError):
        build_manifest(unpaired)

    duplicate = tmp_path / "duplicate"
    add_pair(
        duplicate,
        "one/Training/audio/a.wav",
        "one/Training/label/a.json",
    )
    touch(duplicate / "two/Training/audio/a.wav")
    with pytest.raises(DuplicateIdentifierError):
        build_manifest(duplicate)


def test_split_guards_fail_on_mixed_rows_and_batches() -> None:
    training = ManifestRow(
        "Training",
        "a",
        "chunk/Training/audio/a.wav",
        "chunk/Training/label/a.json",
    )
    validation = replace(
        training,
        split="Validation",
        stem="b",
        wav_path="chunk/Validation/audio/b.wav",
        json_path="chunk/Validation/label/b.json",
    )

    assert ensure_split([training], "Training") == [training]
    with pytest.raises(SplitContaminationError):
        ensure_split([training, validation], "Training")
    ensure_training_batch({"split": ["Training", "Training"]})
    with pytest.raises(SplitContaminationError):
        ensure_training_batch({"split": ["Training", "Validation"]})
