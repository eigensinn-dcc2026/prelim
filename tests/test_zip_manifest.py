from __future__ import annotations

import zipfile
from pathlib import Path

from dcc_data.guards import TRAINING, VALIDATION
from dcc_data.zip_manifest import extract_all_entries, extract_calls, index_zip_entries, select_calls


def _make_zip(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)


def test_index_zip_entries_pairs_wav_and_json_across_separate_zip_chunks(tmp_path: Path) -> None:
    zip1 = tmp_path / "chunk1.zip"
    zip2 = tmp_path / "chunk2.zip"
    _make_zip(zip1, {"Training/1.원천데이터/TS_서울_구급/a.wav": b"wav-bytes"})
    _make_zip(zip2, {"Training/2.라벨링데이터/TL_서울_구급/a.json": b"{}"})

    entries = index_zip_entries([zip1, zip2])

    assert len(entries) == 1
    assert entries[0].split == TRAINING
    assert entries[0].stem == "a"
    assert entries[0].wav.zip_path == zip1
    assert entries[0].json.zip_path == zip2


def test_index_zip_entries_drops_unpaired_files_and_non_split_entries(tmp_path: Path) -> None:
    zip1 = tmp_path / "chunk.zip"
    _make_zip(
        zip1,
        {
            "Training/1.원천데이터/TS_서울_구급/lonely.wav": b"x",  # no matching json -> dropped
            "README.txt": b"not data",  # no Training/Validation component -> skipped, not raised
            "Validation/1.원천데이터/VS_서울_구급/b.wav": b"x",
            "Validation/2.라벨링데이터/VL_서울_구급/b.json": b"{}",
        },
    )

    entries = index_zip_entries([zip1])

    assert [(e.split, e.stem) for e in entries] == [(VALIDATION, "b")]


def test_select_calls_filters_seoul_only_for_training_but_not_validation(tmp_path: Path) -> None:
    zip1 = tmp_path / "chunk.zip"
    _make_zip(
        zip1,
        {
            "Training/1.원천데이터/TS_서울_구급/seoul1.wav": b"x",
            "Training/2.라벨링데이터/TL_서울_구급/seoul1.json": b"{}",
            "Training/1.원천데이터/TS_경기_구급/gyeonggi1.wav": b"x",
            "Training/2.라벨링데이터/TL_경기_구급/gyeonggi1.json": b"{}",
        },
    )
    entries = index_zip_entries([zip1])

    seoul_only = select_calls(entries, split=TRAINING, seoul_only=True, n_calls=10, seed=42)
    assert [e.stem for e in seoul_only] == ["seoul1"]

    everyone = select_calls(entries, split=TRAINING, seoul_only=False, n_calls=10, seed=42)
    assert {e.stem for e in everyone} == {"seoul1", "gyeonggi1"}


def test_extract_calls_writes_exactly_the_selected_files_at_their_zip_relative_path(tmp_path: Path) -> None:
    zip1 = tmp_path / "chunk.zip"
    _make_zip(
        zip1,
        {
            "Training/1.원천데이터/TS_서울_구급/a.wav": b"wav-a",
            "Training/2.라벨링데이터/TL_서울_구급/a.json": b"{}",
            "Training/1.원천데이터/TS_서울_구급/b.wav": b"wav-b",
            "Training/2.라벨링데이터/TL_서울_구급/b.json": b"{}",
        },
    )
    entries = index_zip_entries([zip1])
    only_a = [entry for entry in entries if entry.stem == "a"]

    output_dir = tmp_path / "out"
    extract_calls(only_a, output_dir)

    extracted = sorted(p.relative_to(output_dir).as_posix() for p in output_dir.rglob("*") if p.is_file())
    assert extracted == [
        "Training/1.원천데이터/TS_서울_구급/a.wav",
        "Training/2.라벨링데이터/TL_서울_구급/a.json",
    ]
    assert (output_dir / "Training/1.원천데이터/TS_서울_구급/a.wav").read_bytes() == b"wav-a"


def test_extract_calls_skips_already_extracted_calls_and_reports_only_new_ones(tmp_path: Path) -> None:
    # Reproduces resuming after an interrupted (e.g. Colab-disconnected)
    # extraction: re-running the same selection must not re-extract "a" and
    # must report only "b" as newly extracted.
    zip1 = tmp_path / "chunk.zip"
    _make_zip(
        zip1,
        {
            "Training/1.원천데이터/TS_서울_구급/a.wav": b"wav-a",
            "Training/2.라벨링데이터/TL_서울_구급/a.json": b"{}",
            "Training/1.원천데이터/TS_서울_구급/b.wav": b"wav-b",
            "Training/2.라벨링데이터/TL_서울_구급/b.json": b"{}",
        },
    )
    entries = index_zip_entries([zip1])
    output_dir = tmp_path / "out"

    first_pass = extract_calls([e for e in entries if e.stem == "a"], output_dir)
    assert first_pass == 1

    second_pass = extract_calls(entries, output_dir)  # both "a" and "b" requested this time

    assert second_pass == 1  # only "b" was actually extracted
    assert (output_dir / "Training/1.원천데이터/TS_서울_구급/a.wav").read_bytes() == b"wav-a"
    assert (output_dir / "Training/1.원천데이터/TS_서울_구급/b.wav").read_bytes() == b"wav-b"


def test_extract_all_entries_extracts_a_call_even_when_its_partner_is_outside_any_zip(tmp_path: Path) -> None:
    # Reproduces the real bug: call "a"'s wav lives in a zip, but its json
    # was already extracted loose (e.g. a chunk unzipped by hand before this
    # tool existed) -- index_zip_entries()/select_calls() never see both
    # halves together and would silently drop "a" entirely. extract_all_entries
    # must extract "a"'s wav on its own merit (it never tries to pair).
    zip1 = tmp_path / "chunk.zip"
    _make_zip(
        zip1,
        {
            "Training/1.원천데이터/TS_서울_구급/a.wav": b"wav-a",
            "Training/1.원천데이터/TS_서울_구급/b.wav": b"wav-b",
            "Training/2.라벨링데이터/TL_서울_구급/b.json": b"{}",
        },
    )
    output_dir = tmp_path / "out"
    # "a"'s json already sitting loose in output_dir, from outside any zip.
    loose_json = output_dir / "Training/2.라벨링데이터/TL_서울_구급/a.json"
    loose_json.parent.mkdir(parents=True, exist_ok=True)
    loose_json.write_text("{}", encoding="utf-8")

    extracted = extract_all_entries([zip1], output_dir, split=TRAINING, seoul_only=True)

    assert extracted == 3  # a.wav + b.wav + b.json, all extracted on their own merit
    assert (output_dir / "Training/1.원천데이터/TS_서울_구급/a.wav").read_bytes() == b"wav-a"
    assert (output_dir / "Training/1.원천데이터/TS_서울_구급/b.wav").read_bytes() == b"wav-b"
    assert (output_dir / "Training/2.라벨링데이터/TL_서울_구급/b.json").is_file()
    assert loose_json.exists()  # untouched -- now a.wav+a.json can both be found under output_dir


def test_extract_all_entries_filters_by_split_and_seoul_region(tmp_path: Path) -> None:
    zip1 = tmp_path / "chunk.zip"
    _make_zip(
        zip1,
        {
            "Training/1.원천데이터/TS_서울_구급/seoul1.wav": b"x",
            "Training/1.원천데이터/TS_경기_구급/gyeonggi1.wav": b"x",
            "Validation/1.원천데이터/VS_서울_구급/v1.wav": b"x",
        },
    )
    output_dir = tmp_path / "out"

    extracted = extract_all_entries([zip1], output_dir, split=TRAINING, seoul_only=True)

    assert extracted == 1
    found = sorted(p.relative_to(output_dir).as_posix() for p in output_dir.rglob("*") if p.is_file())
    assert found == ["Training/1.원천데이터/TS_서울_구급/seoul1.wav"]


def test_extract_all_entries_skips_already_present_files(tmp_path: Path) -> None:
    zip1 = tmp_path / "chunk.zip"
    _make_zip(zip1, {"Training/1.원천데이터/TS_서울_구급/a.wav": b"wav-a"})
    output_dir = tmp_path / "out"

    first = extract_all_entries([zip1], output_dir, split=TRAINING, seoul_only=True)
    second = extract_all_entries([zip1], output_dir, split=TRAINING, seoul_only=True)

    assert first == 1
    assert second == 0


def test_extract_calls_skip_existing_false_forces_re_extraction(tmp_path: Path) -> None:
    zip1 = tmp_path / "chunk.zip"
    _make_zip(
        zip1,
        {
            "Training/1.원천데이터/TS_서울_구급/a.wav": b"wav-a",
            "Training/2.라벨링데이터/TL_서울_구급/a.json": b"{}",
        },
    )
    entries = index_zip_entries([zip1])
    output_dir = tmp_path / "out"

    extract_calls(entries, output_dir)
    second_pass = extract_calls(entries, output_dir, skip_existing=False)

    assert second_pass == 1
