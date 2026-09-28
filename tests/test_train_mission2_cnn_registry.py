"""scripts/train_mission2_cnn.py picks a model, its Dataset and its collate_fn
from three parallel dicts keyed by --model. A key missing from one of them, or
a collate_fn whose batch shape the model's forward() doesn't accept, only shows
up mid-run on Colab -- so check every registered --model end to end on a tiny
CPU batch here.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for _root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

train_mission2_cnn = importlib.import_module("train_mission2_cnn")

from dcc_data.mission2_spectrogram import N_MELS  # noqa: E402


def _tiny_index(tmp_path) -> pd.DataFrame:
    rows = []
    for i, n_frames in enumerate([3, 10, 25]):
        spec = np.random.default_rng(i).uniform(-40, 0, size=(N_MELS, n_frames)).astype(np.float32)
        npy_path = f"call{i}/0.npy"
        (tmp_path / npy_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(tmp_path / npy_path, spec)
        rows.append(
            {"call_id": f"call{i}", "utt_idx": 0, "speaker": i % 2, "duration_sec": 1.0 + i, "n_frames": n_frames, "npy_path": npy_path}
        )
    return pd.DataFrame(rows)


def test_the_three_model_registries_have_the_same_keys() -> None:
    keys = set(train_mission2_cnn.MODEL_CLASSES)
    assert set(train_mission2_cnn.DATASET_CLASSES) == keys
    assert set(train_mission2_cnn.COLLATE_FNS) == keys


@pytest.mark.parametrize("model_name", ["small", "large", "large-duration", "resnet", "large-masked"])
def test_every_registered_model_accepts_a_batch_from_its_own_dataset_and_collate(model_name, tmp_path) -> None:
    dataset = train_mission2_cnn.DATASET_CLASSES[model_name](_tiny_index(tmp_path), tmp_path)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=3, shuffle=False, collate_fn=train_mission2_cnn.COLLATE_FNS[model_name]
    )
    model = train_mission2_cnn.MODEL_CLASSES[model_name]()
    model.eval()

    *inputs, labels = next(iter(loader))
    with torch.no_grad():
        logits = model(*inputs)

    assert logits.shape == (3, 2)
    assert labels.shape == (3,)
