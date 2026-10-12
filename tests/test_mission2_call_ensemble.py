"""CPU tests for the Mission 2 call-subset ensemble gate. Synthetic ids only."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "scripts"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from dcc_data.mission2_call_ensemble import (  # noqa: E402
    full_fold_assignments,
    gate_assignments,
    gate_decision,
    load_call_id_list,
    mcnemar_exact,
    resolve_call_id_split,
    shuffle_call_ids,
    soft_vote,
)

prepare_mission2_call_ensemble = importlib.import_module("prepare_mission2_call_ensemble")

CALL_IDS = [f"c{i:05d}" for i in range(9000)]


def test_shuffle_call_ids_is_deterministic_pcg64() -> None:
    ids = [f"c{i:04d}" for i in range(50)]
    first = shuffle_call_ids(ids, 42)
    assert first == shuffle_call_ids(list(reversed(ids)) + ids, 42)
    assert first != shuffle_call_ids(ids, 43)

    unique = sorted(set(ids))
    order = np.arange(len(unique))
    np.random.Generator(np.random.PCG64(42)).shuffle(order)
    assert first == [unique[int(i)] for i in order]


def test_gate_sizes_and_holdout_are_disjoint() -> None:
    assigned = gate_assignments(CALL_IDS, seed=42)
    assert len(assigned["holdout"]) == 1500
    assert len(assigned["source"]) == 6000
    assert len(assigned["arm_a_train"]) == 6000
    assert len(assigned["arm_a_val"]) == 1200
    assert len(assigned["b0_train"]) == 4800
    assert len(assigned["b0_val"]) == 1200
    assert len(assigned["b1_train"]) == 4800
    assert len(assigned["b1_val"]) == 1200
    holdout = set(assigned["holdout"])
    for name in ("source", "arm_a_train", "arm_a_val", "b0_train", "b0_val", "b1_train", "b1_val"):
        assert holdout.isdisjoint(assigned[name])
    assert set(assigned["arm_a_train"]) == set(assigned["source"])
    assert set(assigned["b0_train"]).isdisjoint(assigned["b0_val"])
    assert set(assigned["b1_train"]).isdisjoint(assigned["b1_val"])
    with pytest.raises(ValueError, match="8700"):
        gate_assignments(CALL_IDS[:8699], seed=42)


def test_b0_train_is_source_without_its_first_1200_in_shuffle_order() -> None:
    assigned = gate_assignments(CALL_IDS, seed=42)
    assert assigned["b0_val"] == assigned["source"][:1200]
    assert assigned["b0_train"] == assigned["source"][1200:]
    assert assigned["arm_a_train"] == assigned["source"]


def test_full_folds_put_each_call_in_exactly_one_val() -> None:
    assigned = full_fold_assignments(CALL_IDS, seed=42, n_folds=3)
    folds = assigned["folds"]
    assert [len(fold) for fold in folds] == [3000, 3000, 3000]
    assert [call_id for fold in folds for call_id in fold] == shuffle_call_ids(CALL_IDS, 42)
    assert folds[0] != gate_assignments(CALL_IDS, seed=42)["holdout"]
    seen: list[str] = []
    for member in assigned["members"]:
        assert member["torch_seed"] == 1
        assert member["val"] == folds[member["fold"]]
        others = [fold for index, fold in enumerate(folds) if index != member["fold"]]
        assert member["train"] == [call_id for fold in others for call_id in fold]
        assert set(member["train"]).isdisjoint(member["val"])
        seen.extend(member["val"])
    assert len(seen) == len(set(seen)) == len(CALL_IDS)


def test_soft_vote_assigns_class_1_at_mean_one_half() -> None:
    assert soft_vote([[0.5]]).tolist() == [1]
    assert soft_vote([[0.4, 0.2], [0.6, 0.8]]).tolist() == [1, 1]
    assert soft_vote([[0.49], [0.49]]).tolist() == [0]


def test_mcnemar_exact_matches_a_known_two_by_two_table() -> None:
    # Correctness table: n11=3, n10=1, n01=9, n00=2. Discordant counts are 1 and 9.
    correct_a = [True] * 3 + [True] + [False] * 9 + [False] * 2
    correct_b = [True] * 3 + [False] + [True] * 9 + [False] * 2
    n_a_only, n_b_only, p_value = mcnemar_exact(correct_a, correct_b)
    assert (n_a_only, n_b_only) == (1, 9)
    import math

    n, k = 10, 1
    observed = math.comb(n, k)
    total = sum(math.comb(n, i) for i in range(n + 1) if math.comb(n, i) <= observed)
    assert p_value == pytest.approx(total / 2**n)
    assert mcnemar_exact([True, False], [True, False]) == (0, 0, 1.0)


def test_gate_decision_requires_both_frozen_thresholds() -> None:
    assert gate_decision(0.90, 0.903, 0.049) is True
    assert gate_decision(0.90, 0.903, 0.05) is False
    assert gate_decision(0.90, 0.902, 0.01) is False
    assert gate_decision(0.90, 0.90, 0.0) is False


def test_overlapping_train_and_val_call_ids_are_rejected() -> None:
    frame = pd.DataFrame({"call_id": ["a", "a", "b"], "speaker": [0, 1, 1]})
    with pytest.raises(ValueError, match="overlap"):
        resolve_call_id_split(frame, ["a", "b"], ["b"])
    with pytest.raises(ValueError, match="not in the index"):
        resolve_call_id_split(frame, ["a"], ["missing"])
    with pytest.raises(ValueError, match="empty"):
        resolve_call_id_split(frame, [], ["a"])


def _write_index(tmp_path: Path, call_ids: list[str]) -> Path:
    rows = [
        {
            "call_id": call_id,
            "utt_idx": 0,
            "speaker": 0,
            "n_frames": 4,
            "npy_path": f"{call_id}/0.npy",
            "split": "Training",
            "n_mels": 40,
            "hop_length": 160,
            "n_fft": 512,
            "sample_rate": 8000,
            "label_source": "provided_json",
        }
        for call_id in call_ids
    ]
    path = tmp_path / "index.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_prepare_gate_writes_sorted_lists_and_hashes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    index_path = _write_index(tmp_path, CALL_IDS[:8700])
    out_dir = tmp_path / "gate"
    monkeypatch.setattr(
        sys, "argv",
        ["prepare_mission2_call_ensemble.py", "--index", str(index_path), "--out-dir", str(out_dir), "--mode", "gate"],
    )
    assert prepare_mission2_call_ensemble.main() == 0
    holdout_text = (out_dir / "holdout.txt").read_text(encoding="utf-8")
    holdout_lines = holdout_text.splitlines()
    assert holdout_lines == sorted(holdout_lines)
    assert len(holdout_lines) == 1500
    payload = json.loads((out_dir / "assignments.json").read_text(encoding="utf-8"))
    assert payload["lists"]["b0_train"]["count"] == 4800
    assert payload["lists"]["holdout"]["sha256"] == hashlib.sha256(holdout_text.encode("utf-8")).hexdigest()
    assert payload["roles"]["arm_a_seed1"]["torch_seed"] == 1
    assert payload["roles"]["arm_a_seed2"]["torch_seed"] == 2
    assert payload["roles"]["b0"]["torch_seed"] == 1


def test_train_script_rejects_overlapping_call_ids(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    train_mission2_cnn = importlib.import_module("train_mission2_cnn")
    index_path = _write_index(tmp_path, ["a", "b"])
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "val.txt"
    train_path.write_text("a\nb\n", encoding="utf-8")
    val_path.write_text("b\n", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv",
        [
            "train_mission2_cnn.py", "--index", str(index_path), "--spectrogram-dir", str(tmp_path / "specs"),
            "--train-call-ids", str(train_path), "--val-call-ids", str(val_path),
            "--model-output", str(tmp_path / "out" / "model.pt"), "--epochs", "1",
        ],
    )
    with pytest.raises(ValueError, match="overlap"):
        train_mission2_cnn.main()
    assert not (tmp_path / "out").exists()


def test_train_script_rejects_one_call_id_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    train_mission2_cnn = importlib.import_module("train_mission2_cnn")
    monkeypatch.setattr(
        sys, "argv",
        ["train_mission2_cnn.py", "--index", str(tmp_path / "missing.parquet"), "--train-call-ids", str(tmp_path / "train.txt")],
    )
    with pytest.raises(SystemExit):
        train_mission2_cnn.main()


def test_explicit_call_ids_skip_make_group_split(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    train_mission2_cnn = importlib.import_module("train_mission2_cnn")
    index_path = _write_index(tmp_path, ["a", "b"])
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "val.txt"
    train_path.write_text("a\n", encoding="utf-8")
    val_path.write_text("b\n", encoding="utf-8")

    def _unexpected_split(*_args, **_kwargs):
        raise AssertionError("make_group_split should not run")

    def _stop_before_training(*_args, **_kwargs):
        raise RuntimeError("reached training")

    monkeypatch.setattr(train_mission2_cnn, "make_group_split", _unexpected_split)
    monkeypatch.setattr(train_mission2_cnn, "train_with_best_checkpoint", _stop_before_training)
    monkeypatch.setattr(
        sys, "argv",
        [
            "train_mission2_cnn.py", "--index", str(index_path), "--spectrogram-dir", str(tmp_path / "specs"),
            "--train-call-ids", str(train_path), "--val-call-ids", str(val_path),
            "--model-output", str(tmp_path / "out" / "model.pt"), "--epochs", "1", "--batch-size", "1",
        ],
    )
    with pytest.raises(RuntimeError, match="reached training"):
        train_mission2_cnn.main()
    assert load_call_id_list(train_path) == ["a"]
