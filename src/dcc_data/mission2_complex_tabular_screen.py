"""Three official neural architectures on the existing clean 59D contract.

Importing never installs packages, reads competition data, or fits models.
Cross-environment scaled-matrix hashes are recorded, not used as hard gates.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from time import perf_counter
import uuid

import numpy as np
import pandas as pd

from . import mission2_model_family_screen as screen
from . import mission2_standard_representation_comparison as diagnostics


PACKAGE_VERSIONS = {"tabm": "0.0.3", "rtdl-revisiting-models": "0.0.2",
                    "rtdl-num-embeddings": "0.0.12"}
CANDIDATES = ("tabm", "ft_transformer", "tabular_resnet")
STAGE6_REFERENCES = ("mlp_reference", "stronger_mlp")
RECIPE_NOTE = ("Practical model/pipeline-level first-screen, not pure architecture effect. "
               "Stage 6 sklearn MLP used loss-based stopping; these models select the "
               "earliest best internal Validation Accuracy checkpoint with patience 10.")


def get_config(candidate: str) -> dict:
    """Exactly one first-screen recipe per model, fixed before seeing results."""
    common = dict(seed=42, threshold=0.5, batch_size=256, max_epochs=100,
                  patience=10, optimizer="AdamW", adam_betas=[0.9, 0.999],
                  adam_eps=1e-8, dtype="float32", checkpoint_metric="validation_accuracy",
                  tie_policy="keep_earlier_best", scheduler="none")
    configs = {
        "tabm": dict(
            model_family="TabM", implementation="tabm.TabM.make", package="tabm",
            architecture=dict(n_num_features=59, cat_cardinalities=[], num_embeddings=None,
                              d_out=1, n_blocks=3, d_block=512, dropout=0.1,
                              activation="ReLU", k=32, arch_type="tabm",
                              start_scaling_init="random-signs"),
            learning_rate=0.002, weight_decay=0.0003,
            source="https://github.com/yandex-research/tabm",
            rationale="Official no-embedding TabM defaults and default AdamW. No data-derived bins or extra feature engineering."),
        "ft_transformer": dict(
            model_family="FT-Transformer", implementation="rtdl_revisiting_models.FTTransformer",
            package="rtdl-revisiting-models",
            architecture=dict(n_cont_features=59, cat_cardinalities=[], d_out=1,
                              n_blocks=3, d_block=192, attention_n_heads=8,
                              attention_dropout=0.2, ffn_d_hidden=None,
                              ffn_d_hidden_multiplier=4 / 3, ffn_dropout=0.1,
                              residual_dropout=0.0, _is_default=True),
            learning_rate=0.0001, weight_decay=0.00001,
            source="https://github.com/yandex-research/rtdl-revisiting-models/tree/main/package",
            rationale="Official 3-block default and optimizer parameter groups; feature tokenization is part of this architecture."),
        "tabular_resnet": dict(
            model_family="Tabular ResNet", implementation="rtdl_revisiting_models.ResNet",
            package="rtdl-revisiting-models",
            architecture=dict(d_in=59, d_out=1, n_blocks=2, d_block=192,
                              d_hidden=None, d_hidden_multiplier=2.0,
                              dropout1=0.15, dropout2=0.0),
            learning_rate=0.0003, weight_decay=0.00001,
            source="https://github.com/yandex-research/rtdl-revisiting-models/tree/main/package",
            rationale="Official README ResNet architecture; AdamW lr=3e-4 and weight_decay=1e-5 follow the official end-to-end example."),
    }
    config = {"candidate": candidate, **common, **deepcopy(configs[candidate]),
              "package_version": PACKAGE_VERSIONS[configs[candidate]["package"]],
              "recipe_note": RECIPE_NOTE,
              "budget_rationale": "Batch 256 limits GPU memory; 100 epochs is a cap, patience 10 is shared. No sweep or post-result changes."}
    config["config_id"] = screen.models.config_fingerprint(config)
    return config


def make_model(candidate, config, device):
    """Call authors' APIs directly; do not substitute a home-built architecture."""
    if candidate == "tabm":
        from tabm import TabM
        model = TabM.make(**config["architecture"])
    else:
        from rtdl_revisiting_models import FTTransformer, ResNet
        factory = FTTransformer if candidate == "ft_transformer" else ResNet
        model = factory(**config["architecture"])
    import torch
    return model.to(device=device, dtype=torch.float32)


def _logits(candidate, model, batch):
    import torch
    values = model(batch, None) if candidate == "ft_transformer" else model(batch)
    expected = (len(batch), 32, 1) if candidate == "tabm" else (len(batch), 1)
    if tuple(values.shape) != expected or not torch.isfinite(values).all():
        raise ValueError("Official model output shape differs or contains NaN/Inf")
    return values.squeeze(-1)


def check_environment(*, device="cuda") -> dict:
    """Check the actual runtime and official API using synthetic rows only."""
    import torch
    for package, expected in PACKAGE_VERSIONS.items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"Package/API STOP: require {package}=={expected}")
    torch_version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if not (2, 6) <= torch_version < (3, 0):
        raise RuntimeError("Require PyTorch >=2.6,<3; keep the installed CUDA build")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError("Requested device unavailable; no CPU fallback")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
        torch.manual_seed(42)
        for candidate in CANDIDATES:
            model = make_model(candidate, get_config(candidate), device)
            model.eval()
            with torch.inference_mode():
                _logits(candidate, model, torch.zeros(4, 59, device=device, dtype=torch.float32))
            del model
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in
                         [*PACKAGE_VERSIONS, "torch", "numpy", "pandas", "scikit-learn", "librosa", "pyarrow"]},
            "device": device, "backend": f"torch:{device}", "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "torch_threads": torch.get_num_threads(), "official_api_probe": "PASS"}


def _preprocessing_contract(context):
    result = deepcopy(context.contract)
    # Exact transformed bytes can differ across NumPy/sklearn versions. Source
    # identity, split and scaler statistics remain mandatory comparison gates.
    for name in ("train_matrix", "validation_matrix"):
        result["identity"].pop(name, None)
    result.pop("raw_source_verification", None)
    return result


def load_stage6(context, stage6_root):
    """Checkpoint C: align original Stage 6 references without refitting them."""
    root = Path(stage6_root).resolve()
    saved = screen._read_json(root / "contract.json")
    current = _preprocessing_contract(context)
    expected = deepcopy(saved)
    for name in ("train_matrix", "validation_matrix"):
        expected["identity"].pop(name, None)
    expected.pop("raw_source_verification", None)
    if current != expected:
        raise ValueError("Stage 6 preprocessing/source/split/label contract differs")
    if (current["provenance_fingerprint"] != screen.CLEAN_FINGERPRINT
            or current["seed"] != 42 or current["threshold"] != 0.5
            or current["official_validation_used"] is not False
            or current["scaler_fit_scope"] != "internal_train_only"
            or current["validation_preprocessing"] != "transform_only"
            or any(current["split"][k] != v for k, v in screen.models.EXPECTED_INTERNAL_SPLIT.items())):
        raise ValueError("Frozen clean 59D comparison contract differs")
    scaler = screen._read_json(root / "scaler.json")
    if (scaler["n_samples_seen"] != len(context.prepared.train_idx)
            or scaler["fit_scope"] != "internal_train_only"
            or scaler["feature_names"] != current["feature_names"]):
        raise ValueError("Stage 6 scaler provenance differs")
    for name in ("mean", "scale", "var"):
        if not np.array_equal(getattr(context.prepared.scaler, name + "_"), scaler[name]):
            raise ValueError(f"Stage 6 scaler {name} differs")
    comparison = pd.read_csv(root / "comparison.csv", float_precision="round_trip")
    if not comparison.candidate.is_unique or set(comparison.candidate) != set(screen.CANDIDATES):
        raise ValueError("Stage 6 comparison candidate identity differs")
    frames = {}
    hashes = {name: diagnostics.file_sha256(root / name)
              for name in ("comparison.csv", "contract.json", "scaler.json", "environment.json")}
    for candidate in STAGE6_REFERENCES:
        directory = root / "candidates" / candidate
        if screen._read_json(directory / "contract.json") != saved:
            raise ValueError("Stage 6 candidate contract differs")
        metrics = screen._read_json(directory / "metrics.json")
        frame = pd.read_parquet(directory / "validation_predictions.parquet")
        _check_prediction_alignment(frame, context)
        observed, regenerated = screen.evaluate_candidate(
            frame.y_true, frame.probability_speaker1, frame.record_key, frame.call_id)
        # Compare task metrics exactly. mean_confidence is descriptive and its
        # reduction may round differently between NumPy versions.
        if not frame.equals(regenerated) or any(metrics.get(k) != v for k, v in observed.items()
                                                if k != "mean_confidence"):
            raise ValueError("Stage 6 prediction/metric reconstruction differs")
        row = comparison.loc[comparison.candidate == candidate].iloc[0]
        for key in ("train_accuracy", "validation_accuracy", "train_val_gap", "errors", "hce",
                    "balanced_accuracy", "recall_0", "recall_1", "fp", "fn", "config_id", "status"):
            if row[key] != metrics[key]:
                raise ValueError(f"Stage 6 comparison/metrics differ: {candidate}/{key}")
        if metrics["status"] != "SUCCESS":
            raise ValueError("Successful Stage 6 references required")
        frames[candidate] = frame
        for name in ("metrics.json", "config.json", "validation_predictions.parquet"):
            relative = f"candidates/{candidate}/{name}"
            hashes[relative] = diagnostics.file_sha256(root / relative)
    return {"root": str(root), "comparison": comparison, "predictions": frames,
            "hashes": hashes, "saved_matrix_hashes": {k: saved["identity"][k]
                                                     for k in ("train_matrix", "validation_matrix")}}


def _check_prediction_alignment(frame, context):
    p = context.prepared
    for name, values in (("record_key", p.evaluation_record_keys),
                         ("call_id", context.validation_calls), ("y_true", p.y_evaluation)):
        if not np.array_equal(frame[name].to_numpy(), values):
            raise ValueError(f"Prediction row alignment differs: {name}")
    if not frame.record_key.is_unique:
        raise ValueError("Prediction keys duplicated")


def create_run(repo_root, context, references, environment):
    """Checkpoint D only: reserve a new run, freeze configs, snapshot references."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    root = Path(repo_root).resolve() / "data/processed/mission2_complex_tabular_screen" / run_id
    root.mkdir(parents=True, exist_ok=False)
    configs = {c: get_config(c) for c in CANDIDATES}
    screen._write_json_new(root / "contract.json", {
        "schema_version": "mission2_complex_tabular_screen_v1",
        "preprocessing": _preprocessing_contract(context), "configs": configs,
        "scaled_matrix_hashes": {k: context.contract["identity"][k]
                                 for k in ("train_matrix", "validation_matrix")},
        "matrix_hash_policy": "informational_only_across_environments",
        "stage6": {k: references[k] for k in ("root", "hashes", "saved_matrix_hashes")},
        "raw_source_verification": context.contract.get("raw_source_verification"),
        "recipe_note": RECIPE_NOTE,
    })
    screen._write_json_new(root / "environment.json", environment)
    scaler = context.prepared.scaler
    screen._write_json_new(root / "scaler.json", {
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "var": scaler.var_.tolist(),
        "n_samples_seen": int(scaler.n_samples_seen_), "fit_scope": "internal_train_only",
        "feature_names": list(context.prepared.feature_names)})
    snapshot = root / "stage6_reference"
    snapshot.mkdir()
    with (snapshot / "comparison.csv").open("x", encoding="utf-8", newline="") as stream:
        references["comparison"].to_csv(stream, index=False)
    for candidate, frame in references["predictions"].items():
        with (snapshot / f"{candidate}_predictions.parquet").open("xb") as stream:
            frame.to_parquet(stream, index=False)
    # Freeze prepared arrays against accidental notebook edits, without repeating
    # full-array hashing before each independent model cell.
    p = context.prepared
    for values in (p.X_train_scaled, p.X_evaluation_scaled, p.y_train, p.y_evaluation,
                   p.training_record_keys, p.evaluation_record_keys, p.train_idx, p.validation_idx,
                   context.train_calls, context.validation_calls):
        values.setflags(write=False)
    return root


@contextmanager
def _deterministic_training(device, seed):
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                torch.get_default_dtype())
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
            torch.manual_seed(seed)
            torch.set_default_dtype(torch.float32)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous[2:4]
        torch.set_default_dtype(previous[4])


def predict_probability(candidate, model, X, *, device, batch_size):
    """Inference never updates gradients or ResNet BatchNorm running statistics."""
    import torch
    model.eval()
    probability = np.empty(len(X), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            batch = torch.from_numpy(np.array(X[start:end], dtype=np.float32, copy=True)).to(device)
            values = torch.sigmoid(_logits(candidate, model, batch))
            # TabM training uses separate member losses; inference averages
            # member probabilities, never member logits (authors' contract).
            if candidate == "tabm":
                values = values.mean(dim=1)
            probability[start:end] = values.cpu().numpy()
    return probability


def fit_network(candidate, config, X_train, y_train, X_validation, y_validation, *, device):
    """Train on internal train only; validation selects the earliest best epoch.

    Kept independent of competition I/O so a small synthetic smoke can exercise
    the real training path. Production run_candidate accepts only frozen configs.
    """
    import torch
    for X, y in ((X_train, y_train), (X_validation, y_validation)):
        if (X.dtype != np.float32 or X.shape != (len(y), 59) or not np.isfinite(X).all()
                or not set(np.unique(y)).issubset({0, 1}) or len(y) < 2):
            raise ValueError("Expected finite float32 59D rows and binary labels")
    started = perf_counter()
    with _deterministic_training(device, config["seed"]):
        model = make_model(candidate, config, device)
        parameters = model.make_parameter_groups() if candidate == "ft_transformer" else model.parameters()
        optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"],
                                      betas=tuple(config["adam_betas"]), eps=config["adam_eps"])
        rng = np.random.default_rng(config["seed"])
        history, best_state, best_correct, best_epoch, stale = [], None, -1, 0, 0
        for epoch in range(1, config["max_epochs"] + 1):
            model.train()
            indices = rng.permutation(len(y_train))
            batches = [indices[start:start + config["batch_size"]]
                       for start in range(0, len(indices), config["batch_size"])]
            # BatchNorm requires >=2 train rows. Merge a singleton tail rather
            # than dropping a row or feeding validation rows to its statistics.
            if len(batches[-1]) == 1 and len(batches) > 1:
                tail = batches.pop()
                batches[-1] = np.concatenate([batches[-1], tail])
            total_loss = 0.0
            for indices_batch in batches:
                batch = torch.from_numpy(X_train[indices_batch].copy()).to(device)
                truth = torch.tensor(y_train[indices_batch], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                values = _logits(candidate, model, batch)
                if candidate == "tabm":
                    truth = truth[:, None].expand_as(values)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(values, truth)
                if not torch.isfinite(loss):
                    raise ValueError("NaN/Inf training loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError("NaN/Inf gradient")
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(indices_batch)
            probabilities = predict_probability(candidate, model, X_validation, device=device,
                                                batch_size=config["batch_size"])
            correct = int(((probabilities >= 0.5) == y_validation).sum())
            if correct > best_correct:
                best_correct, best_epoch, stale = correct, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            history.append({"epoch": epoch, "train_loss": total_loss / len(y_train),
                            "validation_accuracy": correct / len(y_validation),
                            "validation_correct": correct, "best_epoch": best_epoch})
            print(f"{candidate} epoch={epoch} loss={history[-1]['train_loss']:.6f} "
                  f"val_accuracy={correct / len(y_validation):.6f} best_epoch={best_epoch}", flush=True)
            if stale >= config["patience"]:
                break
        model.load_state_dict(best_state, strict=True)
        model.eval()
        if device == "cuda":
            torch.cuda.synchronize()
    return model, history, {"best_epoch": best_epoch, "epochs_run": len(history),
                            "best_validation_correct": best_correct,
                            "fit_seconds": perf_counter() - started}


def save_reload_checkpoint(candidate, model, config, directory, X_validation, *, device):
    """Save once, then demand exact same-environment validation probabilities."""
    import torch
    before = predict_probability(candidate, model, X_validation, device=device,
                                 batch_size=config["batch_size"])
    path = Path(directory) / "checkpoint.pt"
    with path.open("xb") as stream:
        torch.save({"candidate": candidate, "config": config,
                    "feature_names": list(screen.clean_59d_feature_names()),
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, stream)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["config"] != config or checkpoint["feature_names"] != list(screen.clean_59d_feature_names()):
        raise ValueError("Checkpoint config/feature contract differs")
    # Release model GPU storage before building the reloaded model.
    model.to("cpu")
    reloaded = make_model(candidate, checkpoint["config"], device)
    reloaded.load_state_dict(checkpoint["state_dict"], strict=True)
    after = predict_probability(candidate, reloaded, X_validation, device=device,
                                batch_size=config["batch_size"])
    if not np.array_equal(before, after):
        raise ValueError("Checkpoint reload prediction mismatch; stop subsequent fits")
    return reloaded, after


def run_candidate(candidate, context, run_root):
    """One independent E/F/G cell; errors propagate and block later fits."""
    root = Path(run_root).resolve()
    if root.parent.name != "mission2_complex_tabular_screen":
        raise ValueError("Refuse writes outside the new complex-screen artifact root")
    contract = screen._read_json(root / "contract.json")
    config = get_config(candidate)
    if (_preprocessing_contract(context) != contract["preprocessing"]
            or config != contract["configs"][candidate]):
        raise ValueError("Frozen preprocessing/config changed")
    environment = screen._read_json(root / "environment.json")
    device = environment["device"]
    for name in CANDIDATES:
        previous = root / name
        if previous.exists() and (not (previous / "metrics.json").is_file()
                                  or screen._read_json(previous / "metrics.json")["status"] != "SUCCESS"):
            raise RuntimeError("Earlier candidate incomplete/failed; stop this run, do not retry")
    directory = root / candidate
    directory.mkdir(exist_ok=False)
    screen._write_json_new(directory / "config.json", config)
    p = context.prepared
    model = None
    try:
        model, history, fit = fit_network(candidate, config, p.X_train_scaled, p.y_train,
                                         p.X_evaluation_scaled, p.y_evaluation, device=device)
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False, mode="x")
        started = perf_counter()
        train_probability = predict_probability(candidate, model, p.X_train_scaled, device=device,
                                                batch_size=config["batch_size"])
        validation_probability = predict_probability(candidate, model, p.X_evaluation_scaled,
                                                     device=device, batch_size=config["batch_size"])
        predict_seconds = perf_counter() - started
        model, restored = save_reload_checkpoint(candidate, model, config, directory,
                                                 p.X_evaluation_scaled, device=device)
        if not np.array_equal(restored, validation_probability):
            raise ValueError("Reloaded prediction mismatch")
        metrics, frame = screen.evaluate_candidate(p.y_evaluation, validation_probability,
                                                   p.evaluation_record_keys, context.validation_calls)
        _check_prediction_alignment(frame, context)
        train_accuracy = float(((train_probability >= 0.5) == p.y_train).mean())
        metrics.update(candidate=candidate, model_family=config["model_family"],
                       implementation=config["implementation"], package_version=config["package_version"],
                       device=device, backend=environment["backend"], config_id=config["config_id"],
                       seed=42, train_accuracy=train_accuracy,
                       train_val_gap=train_accuracy - metrics["validation_accuracy"],
                       predict_seconds=predict_seconds, **fit, checkpoint_reload="EXACT_PASS",
                       recipe_note=RECIPE_NOTE, status="SUCCESS")
        with (directory / "predictions.parquet").open("xb") as stream:
            frame.to_parquet(stream, index=False)
        screen._write_json_new(directory / "metrics.json", metrics)
        return metrics
    except Exception as error:
        screen._write_json_new(directory / "metrics.json", {
            "candidate": candidate, "status": "FAILED", "reason": f"{type(error).__name__}: {error}"})
        raise
    finally:
        del model
        gc.collect()
        if device == "cuda":
            __import__("torch").cuda.empty_cache()


def save_comparison(run_root):
    """Checkpoint H: completed candidates plus two original Stage 6 baselines."""
    root = Path(run_root)
    stage6 = pd.read_csv(root / "stage6_reference/comparison.csv", float_precision="round_trip")
    reference_accuracy = float(stage6.loc[stage6.candidate == "mlp_reference", "validation_accuracy"].iloc[0])
    best_accuracy = float(stage6.loc[stage6.candidate == "stronger_mlp", "validation_accuracy"].iloc[0])
    rows = stage6.loc[stage6.candidate.isin(STAGE6_REFERENCES)].to_dict("records")
    for row in rows:
        row.update(source="stage6", training_recipe="Stage 6 sklearn loss-based stopping",
                   recipe_note=RECIPE_NOTE)
    for candidate in CANDIDATES:
        metrics = screen._read_json(root / candidate / "metrics.json")
        if metrics["status"] != "SUCCESS":
            raise RuntimeError("Comparison requires three successful candidates")
        rows.append({**metrics, "source": "complex_screen", "training_recipe": "validation_accuracy_early_stopping"})
    for row in rows:
        row.update(delta_vs_stage6_best=row["validation_accuracy"] - best_accuracy,
                   delta_vs_mlp_reference=row["validation_accuracy"] - reference_accuracy)
    columns = ["candidate", "model_family", "implementation", "device", "backend", "config_id",
               "train_accuracy", "validation_accuracy", "delta_vs_stage6_best", "delta_vs_mlp_reference",
               "balanced_accuracy", "recall_0", "recall_1", "fp", "fn", "errors", "train_val_gap",
               "hce", "fit_seconds", "predict_seconds", "status", "source", "training_recipe", "recipe_note"]
    result = pd.DataFrame(rows).reindex(columns=columns)
    result.to_csv(root / "comparison.csv", index=False, mode="x")
    return result


def save_paired_analysis(run_root, candidates, tracking=None):
    """Checkpoint I: selected models only; aligned diagnostics never enter X."""
    root = Path(run_root)
    if not candidates or len(set(candidates)) != len(candidates) or not set(candidates).issubset(CANDIDATES):
        raise ValueError("Select distinct complex candidates after reading the overall comparison")
    reference = pd.read_parquet(root / "stage6_reference/stronger_mlp_predictions.parquet")
    if tracking is not None:
        for name in ("record_key", "call_id", "y_true"):
            if not np.array_equal(tracking[name].to_numpy(), reference[name].to_numpy()):
                raise ValueError("Analysis metadata row alignment differs; no slice inference")
        if tracking.overlap.isna().any() or not tracking.overlap.isin([True, False]).all():
            raise ValueError("Analysis overlap metadata must be explicit Boolean values")
    paired_rows, slice_rows, call_rows = [], [], []
    for candidate in candidates:
        frame = pd.read_parquet(root / candidate / "predictions.parquet")
        changes = screen.paired_comparison(reference, frame)
        paired_rows.append({"candidate": candidate, "reference": "stage6_stronger_mlp", **changes})
        masks = {"all": np.ones(len(frame), dtype=bool), "speaker0": frame.y_true.to_numpy() == 0,
                 "speaker1": frame.y_true.to_numpy() == 1,
                 "confidence_ge_0.9": frame.confidence.to_numpy() >= 0.9,
                 "confidence_lt_0.9": frame.confidence.to_numpy() < 0.9}
        if tracking is not None:
            masks.update(diagnostics.slice_masks(tracking))
            # Existing, explicit analysis masks may be supplied. No new silence
            # or low-information cutoff is invented or learned from labels.
            for name in ("active", "silence", "low_information"):
                if name in tracking:
                    if tracking[name].isna().any() or not tracking[name].isin([True, False]).all():
                        raise ValueError(f"Invalid analysis mask: {name}")
                    masks[name] = tracking[name].to_numpy(dtype=bool)
        first, second = reference.correct.to_numpy(), frame.correct.to_numpy()
        for name, mask in masks.items():
            n = int(mask.sum())
            lost, gained = int((mask & first & ~second).sum()), int((mask & ~first & second).sum())
            slice_rows.append({"candidate": candidate, "slice": name, "n": n,
                               "reference_accuracy": float(first[mask].mean()) if n else None,
                               "candidate_accuracy": float(second[mask].mean()) if n else None,
                               "both_correct": int((mask & first & second).sum()),
                               "both_wrong": int((mask & ~first & ~second).sum()),
                               "regression": lost, "correction": gained, "net_correction": gained - lost,
                               "reference_hce": int((mask & ~first & (reference.confidence.to_numpy() >= .9)).sum()),
                               "candidate_hce": int((mask & ~second & (frame.confidence.to_numpy() >= .9)).sum())})
        concentration = pd.DataFrame({"call_id": frame.call_id, "reference_errors": ~first,
                                      "candidate_errors": ~second}).groupby("call_id", sort=False).sum().reset_index()
        concentration.insert(0, "candidate", candidate)
        call_rows.extend(concentration.to_dict("records"))
    paired = pd.DataFrame(paired_rows)
    slices = pd.DataFrame(slice_rows)
    paired.to_csv(root / "paired_comparison.csv", index=False, mode="x")
    slices.to_csv(root / "paired_slices.csv", index=False, mode="x")
    pd.DataFrame(call_rows).to_csv(root / "error_concentration_by_call.csv", index=False, mode="x")
    screen._write_json_new(root / "analysis_metadata.json", {
        "selected_candidates": list(candidates), "reference": "stage6_stronger_mlp",
        "tracking_used": tracking is not None,
        "missing_analysis_masks": [name for name in ("active", "silence", "low_information")
                                   if tracking is None or name not in tracking],
        "role": "diagnostics only; never model inputs"})
    return paired, slices
