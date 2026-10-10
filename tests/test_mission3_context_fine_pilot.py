"""The failed-pilot extension must use complete, unchanged frozen runs."""
import importlib.util
from pathlib import Path

import pytest
import numpy as np
from test_mission3_context import tiny

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fine_pilot", ROOT / "scripts/finetune_mission3_context_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
export_spec = importlib.util.spec_from_file_location("export_pilot", ROOT / "scripts/export_mission3_context_pilot.py")
exporter = importlib.util.module_from_spec(export_spec)
export_spec.loader.exec_module(exporter)


def test_source_requires_completed_full_identical_experiment():
    source = {"smoke": False, "checkpoint": "original"}
    summary = {"smoke": False, "source": source, "status": "both_pilots_failed"}
    pilot.validate_source(source, dict(source), summary)
    with pytest.raises(ValueError, match="changed"):
        pilot.validate_source(source, dict(source, checkpoint="other"), summary)
    with pytest.raises(ValueError, match="smoke"):
        pilot.validate_source(source, source, dict(summary, smoke=True))
    with pytest.raises(ValueError, match="completed"):
        pilot.validate_source(source, source, dict(summary, status="pilot"))


def test_head_rejects_wrong_seed_method_or_incomplete_epoch():
    state = {"specification": {"signature": "original", "method": "memory", "seed": 42,
                                "recipe": pilot.exp.RECIPE}, "epoch": 10}
    pilot.validate_head(state, "original", "memory")
    for changed in ({"epoch": 9}, {"specification": dict(state["specification"], seed=43)},
                    {"specification": dict(state["specification"], method="control")}):
        with pytest.raises(ValueError, match="epoch-10 seed-42"):
            pilot.validate_head(dict(state, **changed), "original", "memory")


@pytest.mark.parametrize("method", ["control", "memory", "evidence"])
def test_experimental_export_roundtrip_preserves_predictions_and_bank(tiny, tmp_path, method):
    model, tokenizer, payload = tiny
    torch, context = pilot.torch, pilot.context
    rows = [dict(name=f"{i}.json", call=str(i), text_hash=str(i), gold=[i % 2] * 9,
                 doc={"utterances": [{"text": "머리"}, {"text": "배"}]}) for i in range(4)]
    train = [0, 1, 2]
    model.head = context.ResidualHead(model.base.classifier.in_features, method, context.DEFAULTS)
    if method == "memory":
        model.bank = context.MemoryBank(torch.randn(3, model.base.classifier.in_features),
                                       [rows[i]["gold"] for i in train],
                                       [rows[i]["call"] for i in train], [rows[i]["text_hash"] for i in train])
    else:
        model.bank = None
    model.eval()
    docs, names = [row["doc"] for row in rows], [row["name"] for row in rows]
    expected = context.probabilities(model, tokenizer, docs, names, batch_size=2)
    state = dict(specification=dict(signature="tiny", seed=42, recipe=pilot.exp.RECIPE, method=method),
                 epoch=1, model=model.state_dict())
    exporter.restore_state(model, state, method, rows, train, "tiny")
    checkpoint = tmp_path / "experimental.pt"
    torch.save(context.export_payload(model, payload, dict(experimental=True, training_rows=3,
                                                         internal_gate=dict(passed=False))), checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored, restored_tokenizer, _, _ = context.load_payload(saved, "cpu")
    actual = context.probabilities(restored, restored_tokenizer, docs, names, batch_size=2)
    np.testing.assert_array_equal(expected, actual)
    assert saved["source"]["internal_gate"]["passed"] is False
    if method == "memory":
        assert restored.bank.calls == [rows[i]["call"] for i in train]
        assert restored.bank.hashes == [rows[i]["text_hash"] for i in train]
        state["model"]["bank.labels"] = torch.ones_like(state["model"]["bank.labels"])
        with pytest.raises(ValueError, match="memory labels"):
            exporter.restore_state(model, state, method, rows, train, "tiny")
