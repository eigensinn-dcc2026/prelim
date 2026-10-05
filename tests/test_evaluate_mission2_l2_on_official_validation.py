"""scripts/evaluate_mission2_l2_on_official_validation.py scores ONE model; a packed seed ensemble is refused."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "scripts"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

evaluate_script = importlib.import_module("evaluate_mission2_l2_on_official_validation")


def test_a_packed_ensemble_is_refused_with_a_pointer_to_the_submission_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "index.parquet"
    pd.DataFrame({"call_id": ["a"], "utt_idx": [0], "speaker": [0], "n_frames": [5], "npy_path": ["a/0.npy"]}).to_parquet(index_path)
    ensemble = tmp_path / "ensemble.pt"
    torch.save({"ensemble_model_state_dicts": [{}, {}], "metadata": {}}, ensemble)
    monkeypatch.setattr(
        sys, "argv",
        ["evaluate_mission2_l2_on_official_validation.py", "--ckpt-path", str(ensemble), "--index", str(index_path),
         "--spectrogram-dir", str(tmp_path), "--log-output", str(tmp_path / "out.json")],
    )

    with pytest.raises(SystemExit, match="packed seed ensemble.*mission2_inference"):
        evaluate_script.main()

    assert not (tmp_path / "out.json").exists()
