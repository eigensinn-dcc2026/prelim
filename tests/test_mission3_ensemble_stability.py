import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

spec = importlib.util.spec_from_file_location(
    "m3_stability", Path(__file__).resolve().parents[1] / "scripts/analyze_mission3_ensemble_stability.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_metrics_and_paired_bootstrap():
    rng = np.random.default_rng(12)
    gold = rng.random((40, 9)) > .7
    pred = rng.random((4, 40, 9)) > .6
    pred[-1] = pred[0]  # identical models must have exact zero paired deltas
    result = module.analyse(gold, pred, iterations=100, seed=7)
    for i, name in enumerate(module.MODELS):
        row = result["models"][name]
        assert row["macro_f1"] == pytest.approx(f1_score(gold, pred[i], average="macro", zero_division=0))
        assert row["micro_f1"] == pytest.approx(f1_score(gold, pred[i], average="micro", zero_division=0))
    paired = result["paired_comparisons"]["r3"]
    assert paired["macro_f1_delta_ci95"] == [0, 0]
    assert all(row["corrected_errors"] == row["introduced_errors"] == 0 for row in paired["per_symptom"].values())
    assert result == module.analyse(gold, pred, iterations=100, seed=7)


def dumps(folder, names=None, gold=None, probs=None):
    for tag in module.TAGS:
        np.savez(folder / f"{tag}_holdout.npz",
                 names=np.array(["a", "b"]) if names is None else names,
                 gold=np.zeros((2, 9)) if gold is None else gold,
                 probs=np.full((2, 9), .5) if probs is None else probs)


def test_threshold_and_reject_misalignment(tmp_path):
    dumps(tmp_path)
    names, gold, pred, sources = module.load_split(tmp_path, "holdout")
    assert pred.all()  # exactly .5 is positive, including the weighted average
    assert len(sources) == 3
    np.savez(tmp_path / "k12_holdout.npz", names=names[::-1], gold=gold, probs=np.ones((2, 9)))
    with pytest.raises(ValueError, match="mismatch"):
        module.load_split(tmp_path, "holdout")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -.1, 1.1])
def test_reject_invalid_probabilities(tmp_path, bad):
    probs = np.full((2, 9), .5)
    probs[0, 0] = bad
    dumps(tmp_path, probs=probs)
    with pytest.raises(ValueError, match="invalid probabilities"):
        module.load_split(tmp_path, "holdout")


def test_zero_support_and_empty_predictions():
    result = module.analyse(np.zeros((4, 9), dtype=bool), np.zeros((4, 4, 9), dtype=bool), 100)
    assert result["models"]["ensemble"]["macro_f1_ci95"] == [0, 0]
    assert result["models"]["ensemble"]["empty_predictions"] == 4
