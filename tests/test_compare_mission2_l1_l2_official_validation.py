from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_mission2_l1_l2_official_validation.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("compare_mission2_l1_l2_official_validation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = _load_script()

KEYS = [("call-a", 0), ("call-a", 1), ("call-b", 0), ("call-b", 1)]
SPEAKERS = [0, 1, 1, 0]


def _key_columns(keys=KEYS) -> dict:
    return {"call_id": [k[0] for k in keys], "utt_idx": [k[1] for k in keys]}


def _l1() -> pd.DataFrame:
    return pd.DataFrame({**_key_columns(), "speaker": SPEAKERS, "l1_pred": [0, 0, 1, 0]})


def _l2() -> pd.DataFrame:
    return pd.DataFrame({**_key_columns(), "speaker": SPEAKERS, "l2_pred": [0, 1, 1, 1]})


def _utt_index() -> pd.DataFrame:
    return pd.DataFrame({**_key_columns(), "duration_sec": [1.0, 1.5, 3.0, 0.4], "speaker": SPEAKERS})


def _overlap() -> pd.DataFrame:
    return pd.DataFrame({**_key_columns(), "overlaps_other": [True, True, False, False]})


def test_build_comparison_frame_joins_exactly_and_keeps_l1_row_order() -> None:
    l1 = _l1().iloc[::-1].reset_index(drop=True)

    merged = compare.build_comparison_frame(l1, _l2(), _utt_index(), _overlap())

    assert list(merged.columns) == [
        "call_id", "utt_idx", "speaker", "l1_pred", "l2_pred", "duration_sec", "overlaps_other",
    ]
    assert merged[["call_id", "utt_idx"]].values.tolist() == l1[["call_id", "utt_idx"]].values.tolist()
    assert len(merged) == 4
    row = merged[(merged["call_id"] == "call-a") & (merged["utt_idx"] == 1)].iloc[0]
    assert (row["l1_pred"], row["l2_pred"], row["duration_sec"], row["overlaps_other"]) == (0, 1, 1.5, True)


@pytest.mark.parametrize("name", ["L1", "L2", "utterance index", "overlap flags"])
def test_build_comparison_frame_rejects_duplicate_keys_in_any_input(name: str) -> None:
    inputs = {"L1": _l1(), "L2": _l2(), "utterance index": _utt_index(), "overlap flags": _overlap()}
    inputs[name] = pd.concat([inputs[name], inputs[name].iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match=f"{name}.*duplicate"):
        compare.build_comparison_frame(inputs["L1"], inputs["L2"], inputs["utterance index"], inputs["overlap flags"])


def test_build_comparison_frame_rejects_utterance_missing_from_l2() -> None:
    l2 = _l2().iloc[:-1]

    with pytest.raises(ValueError, match="L2.*1 missing.*0 extra"):
        compare.build_comparison_frame(_l1(), l2, _utt_index(), _overlap())


def test_build_comparison_frame_rejects_extra_utterance_in_l2() -> None:
    extra = pd.DataFrame({"call_id": ["call-c"], "utt_idx": [0], "speaker": [1], "l2_pred": [1]})
    l2 = pd.concat([_l2(), extra], ignore_index=True)

    with pytest.raises(ValueError, match="L2.*0 missing.*1 extra"):
        compare.build_comparison_frame(_l1(), l2, _utt_index(), _overlap())


def test_build_comparison_frame_rejects_utterance_index_that_does_not_cover_every_utterance() -> None:
    with pytest.raises(ValueError, match="utterance index.*1 missing"):
        compare.build_comparison_frame(_l1(), _l2(), _utt_index().iloc[:-1], _overlap())


def test_build_comparison_frame_rejects_missing_overlap_flags() -> None:
    with pytest.raises(ValueError, match="overlap flags.*1 missing"):
        compare.build_comparison_frame(_l1(), _l2(), _utt_index(), _overlap().iloc[1:])


def test_build_comparison_frame_rejects_l2_ground_truth_that_disagrees_with_l1() -> None:
    l2 = _l2()
    l2.loc[2, "speaker"] = 0

    with pytest.raises(ValueError, match=r"L2 speaker.*1 of 4"):
        compare.build_comparison_frame(_l1(), l2, _utt_index(), _overlap())


def test_build_comparison_frame_rejects_utterance_index_ground_truth_that_disagrees_with_l1() -> None:
    utt_index = _utt_index()
    utt_index.loc[[0, 3], "speaker"] = [1, 1]

    with pytest.raises(ValueError, match=r"utterance index speaker.*2 of 4"):
        compare.build_comparison_frame(_l1(), _l2(), utt_index, _overlap())


def _saved_result(tmp_path: Path, **overrides) -> Path:
    payload = {"n_utterances": 4, "confusion_matrix": [[1, 1], [0, 2]], **overrides}
    path = tmp_path / "saved_result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_check_against_saved_result_accepts_matching_counts(tmp_path: Path) -> None:
    merged = compare.build_comparison_frame(_l1(), _l2(), _utt_index(), _overlap())

    compare.check_against_saved_result(merged, "l2_pred", _saved_result(tmp_path), "L2")


def test_check_against_saved_result_rejects_different_confusion_matrix(tmp_path: Path) -> None:
    merged = compare.build_comparison_frame(_l1(), _l2(), _utt_index(), _overlap())

    with pytest.raises(ValueError, match="L2.*confusion_matrix"):
        compare.check_against_saved_result(
            merged, "l2_pred", _saved_result(tmp_path, confusion_matrix=[[2, 0], [0, 2]]), "L2"
        )


def test_check_against_saved_result_rejects_different_utterance_count(tmp_path: Path) -> None:
    merged = compare.build_comparison_frame(_l1(), _l2(), _utt_index(), _overlap())

    with pytest.raises(ValueError, match="L2.*n_utterances"):
        compare.check_against_saved_result(merged, "l2_pred", _saved_result(tmp_path, n_utterances=5), "L2")


class _StubL1Model:
    def predict(self, features) -> np.ndarray:
        return np.array([0, 0, 1, 0])[: len(features)]


def _write_main_inputs(
    tmp_path: Path, *, l2_frame: pd.DataFrame, subset_size: bool = True, explicit_outputs: bool = True
) -> list[str]:
    """argv for main() on the 4-utterance / 2-call synthetic inputs. ``subset_size`` adds the flags that
    declare that size (without them main() expects the official Validation size); ``explicit_outputs``
    adds explicit --output/--summary-output paths (without them main() would use the official ones)."""

    features = pd.DataFrame({**_key_columns(), "speaker": SPEAKERS})
    for column in compare.FEATURE_COLUMNS:
        features[column] = 0.0
    features.to_parquet(tmp_path / "l1_features.parquet")

    utt_index = _utt_index()
    utt_index["start_at"] = [0.0, 0.5, 0.0, 4.0]
    utt_index["end_at"] = [1.0, 2.0, 3.0, 4.4]
    utt_index.to_parquet(tmp_path / "utterance_index.parquet")

    l2_frame.rename(columns={"l2_pred": "predicted"}).to_csv(tmp_path / "l2_predictions.csv", index=False)

    argv = [
        "compare_mission2_l1_l2_official_validation.py",
        "--l1-ckpt", str(tmp_path / "unused.pkl"),
        "--l1-features", str(tmp_path / "l1_features.parquet"),
        "--l2-predictions-csv", str(tmp_path / "l2_predictions.csv"),
        "--utterance-index", str(tmp_path / "utterance_index.parquet"),
    ]
    if subset_size:
        argv += ["--expected-utterances", "4", "--expected-calls", "2"]
    if explicit_outputs:
        argv += [
            "--output", str(tmp_path / "out" / "comparison.csv"),
            "--summary-output", str(tmp_path / "out" / "comparison.json"),
        ]
    return argv


@pytest.fixture
def stub_l1_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        compare, "load_model", lambda path: (_StubL1Model(), {"feature_columns": list(compare.FEATURE_COLUMNS)})
    )


def test_main_writes_outputs_when_inputs_line_up_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_l1_checkpoint: None
) -> None:
    monkeypatch.setattr(sys, "argv", _write_main_inputs(tmp_path, l2_frame=_l2()))

    assert compare.main() == 0

    summary = json.loads((tmp_path / "out" / "comparison.json").read_text(encoding="utf-8"))
    assert summary["n_utterances"] == 4
    assert summary["l1_accuracy"] == 0.75
    assert summary["l2_accuracy"] == 0.75
    assert len(pd.read_csv(tmp_path / "out" / "comparison.csv")) == 4


def test_main_saves_nothing_when_l2_is_missing_an_utterance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_l1_checkpoint: None
) -> None:
    monkeypatch.setattr(sys, "argv", _write_main_inputs(tmp_path, l2_frame=_l2().iloc[:-1]))

    with pytest.raises(ValueError, match="L2"):
        compare.main()

    assert not (tmp_path / "out" / "comparison.csv").exists()
    assert not (tmp_path / "out" / "comparison.json").exists()


def test_official_validation_size_constants_match_the_saved_official_results() -> None:
    root = Path(__file__).resolve().parents[1] / "reports" / "mission2"
    for name in ("m2_l1_official_validation_result_postfix2.json", "m2_l2_official_validation_result_colab_gpu.json"):
        saved = json.loads((root / name).read_text(encoding="utf-8"))
        assert saved["n_utterances"] == compare.OFFICIAL_VALIDATION_UTTERANCES == 111_947
        assert saved["n_calls"] == compare.OFFICIAL_VALIDATION_CALLS == 3_640


def _joined() -> pd.DataFrame:
    return compare.build_comparison_frame(_l1(), _l2(), _utt_index(), _overlap())


def test_verify_expected_size_accepts_exactly_the_expected_utterances_and_calls() -> None:
    compare.verify_expected_size(_joined(), 4, 2)


@pytest.mark.parametrize(
    ("expected_utterances", "expected_calls"),
    [(5, 2), (3, 2), (4, 3), (4, 1)],
)
def test_verify_expected_size_refuses_any_other_size(expected_utterances: int, expected_calls: int) -> None:
    with pytest.raises(ValueError, match="joined result has 4 utterances from 2 calls") as excinfo:
        compare.verify_expected_size(_joined(), expected_utterances, expected_calls)

    assert "111947" in str(excinfo.value) and "3640" in str(excinfo.value)


def test_main_refuses_inputs_that_agree_with_each_other_but_are_not_the_whole_official_validation_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_l1_checkpoint: None
) -> None:
    # every input covers the same 4 utterances of 2 calls: consistent, but nowhere near 111,947 / 3,640
    monkeypatch.setattr(sys, "argv", _write_main_inputs(tmp_path, l2_frame=_l2(), subset_size=False))

    with pytest.raises(ValueError, match="expected 111947 utterances from 3640 calls"):
        compare.main()

    assert not (tmp_path / "out" / "comparison.csv").exists()
    assert not (tmp_path / "out" / "comparison.json").exists()


def test_a_subset_comparison_must_not_be_written_over_the_official_comparison_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_l1_checkpoint: None
) -> None:
    argv = _write_main_inputs(tmp_path, l2_frame=_l2(), subset_size=True, explicit_outputs=False)
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as excinfo:
        compare.main()

    assert excinfo.value.code == 2  # argparse usage error, raised before any prediction runs


def test_a_subset_comparison_with_one_official_path_left_at_its_default_is_also_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_l1_checkpoint: None
) -> None:
    argv = _write_main_inputs(tmp_path, l2_frame=_l2(), subset_size=True, explicit_outputs=False)
    argv += ["--output", str(tmp_path / "out" / "comparison.csv")]  # --summary-output stays the official default
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit):
        compare.main()


@pytest.mark.parametrize(
    ("which", "column", "value"),
    [
        ("l1", "l1_pred", None),
        ("l1", "speaker", None),
        ("l2", "l2_pred", None),
        ("l2", "speaker", None),
        ("index", "duration_sec", None),
        ("index", "speaker", None),
        ("overlap", "overlaps_other", None),
    ],
)
def test_build_comparison_frame_refuses_a_missing_required_value(which: str, column: str, value) -> None:
    frames = {"l1": _l1(), "l2": _l2(), "index": _utt_index(), "overlap": _overlap()}
    frames[which] = frames[which].astype({column: "object"})
    frames[which].loc[1, column] = value

    with pytest.raises(ValueError, match=r"missing value\(s\)"):
        compare.build_comparison_frame(frames["l1"], frames["l2"], frames["index"], frames["overlap"])


@pytest.mark.parametrize(
    ("which", "column", "expected_name"),
    [
        ("l1", "l1_pred", "l1_pred"),
        ("l1", "speaker", "speaker"),
        ("l2", "l2_pred", "l2_pred"),
        ("l2", "speaker", "l2_speaker"),
        ("index", "speaker", "index_speaker"),
    ],
)
def test_build_comparison_frame_refuses_a_speaker_or_prediction_that_is_not_0_or_1(
    which: str, column: str, expected_name: str
) -> None:
    frames = {"l1": _l1(), "l2": _l2(), "index": _utt_index(), "overlap": _overlap()}
    frames[which].loc[1, column] = 2

    with pytest.raises(ValueError, match=rf"{expected_name} must be 0 or 1, found \[2\]"):
        compare.build_comparison_frame(frames["l1"], frames["l2"], frames["index"], frames["overlap"])


def test_build_comparison_frame_refuses_predictions_that_are_labels_not_numbers() -> None:
    l1 = _l1().astype({"l1_pred": "object"})
    l1.loc[0, "l1_pred"] = "대원"

    with pytest.raises(ValueError, match="l1_pred must be 0 or 1"):
        compare.build_comparison_frame(l1, _l2(), _utt_index(), _overlap())


def test_main_saves_nothing_when_saved_official_result_does_not_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_l1_checkpoint: None
) -> None:
    argv = _write_main_inputs(tmp_path, l2_frame=_l2())
    wrong_l1_result = _saved_result(tmp_path, confusion_matrix=[[2, 0], [0, 2]])
    monkeypatch.setattr(sys, "argv", [*argv, "--l1-result-json", str(wrong_l1_result)])

    with pytest.raises(ValueError, match="L1.*confusion_matrix"):
        compare.main()

    assert not (tmp_path / "out" / "comparison.csv").exists()
    assert not (tmp_path / "out" / "comparison.json").exists()


# ---------------------------------------------------------------------------
# The local CNN path scores one model: a packed seed ensemble (PR #32) is refused with an explanation
# ---------------------------------------------------------------------------


def test_single_model_state_dict_returns_the_weights_of_a_training_checkpoint() -> None:
    checkpoint = {"best_model_state_dict": {"w": 1}, "last_model_state_dict": {"w": 2}}

    assert compare.single_model_state_dict(checkpoint, Path("c.pt")) == {"w": 1}


def test_single_model_state_dict_still_reads_the_older_checkpoint_format() -> None:
    assert compare.single_model_state_dict({"model_state_dict": {"w": 3}}, Path("old.pt")) == {"w": 3}


def test_single_model_state_dict_refuses_a_packed_ensemble_and_says_what_to_do() -> None:
    packed = {compare.PACKED_ENSEMBLE_KEY: [{"w": 1}, {"w": 2}, {"w": 3}], "metadata": {}}

    with pytest.raises(ValueError, match=r"packed seed ensemble \(3 models\).*--l2-predictions-csv"):
        compare.single_model_state_dict(packed, Path("ensemble.pt"))


def test_single_model_state_dict_refuses_a_file_without_weights() -> None:
    with pytest.raises(ValueError, match="holds no model weights"):
        compare.single_model_state_dict({"metadata": {}}, Path("empty.pt"))


def test_the_packed_ensemble_key_is_the_one_the_packing_code_writes() -> None:
    checkpoint_module = Path(__file__).resolve().parents[1] / "src" / "dcc_data" / "mission2_checkpoint.py"
    if not checkpoint_module.exists():  # that module comes with PR #32; this PR alone does not have it yet
        pytest.skip("src/dcc_data/mission2_checkpoint.py is not in this branch")

    assert f'ENSEMBLE_STATE_DICTS_KEY = "{compare.PACKED_ENSEMBLE_KEY}"' in checkpoint_module.read_text(encoding="utf-8")


def test_main_stops_before_any_cnn_import_when_the_l2_checkpoint_is_a_packed_ensemble(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_l1_checkpoint: None
) -> None:
    torch = pytest.importorskip("torch")
    argv = _write_main_inputs(tmp_path, l2_frame=_l2())
    csv_flag = argv.index("--l2-predictions-csv")
    del argv[csv_flag:csv_flag + 2]
    ensemble = tmp_path / "ensemble.pt"
    torch.save({compare.PACKED_ENSEMBLE_KEY: [{"w": 1}, {"w": 2}], "metadata": {}}, ensemble)
    argv += ["--l2-ckpt", str(ensemble), "--l2-index", str(tmp_path / "unused_index.parquet"),
             "--l2-spectrogram-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(ValueError, match="packed seed ensemble"):
        compare.main()

    assert not (tmp_path / "out" / "comparison.csv").exists()
    assert not (tmp_path / "out" / "comparison.json").exists()
