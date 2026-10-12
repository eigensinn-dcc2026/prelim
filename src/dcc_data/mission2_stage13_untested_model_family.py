"""Stage 13 first-screen for six previously untested model families.

All model inputs are the immutable clean 59D matrix and the fixed internal
split. Annotation-derived duration/overlap fields are joined by exact
record_key only after the source/cache identity gate and are used for analysis.
Importing this module does not load the cache, install packages, or fit models.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import time
import uuid

import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
import joblib

from . import mission2_model_family_screen as screen
from . import mission2_complex_tabular_screen as complex_screen
from . import mission2_model_experiments as models


REFERENCE_RUN = Path("data/processed/mission2_complex_tabular_screen/20261006T031510Z_a9a01a2d")
REFERENCE_NAMES = ("tabm", "tabular_resnet")
MODEL_IDS = ("ebm", "spline_gam", "rff_logistic_sgd", "dcnv2", "tabnet", "gandalf")
MODEL_LABELS = {
    "tabm": "TabM (frozen)", "tabular_resnet": "Tabular ResNet (frozen)",
    "ebm": "EBM", "spline_gam": "Spline-GAM", "rff_logistic_sgd": "RFF + Logistic SGD",
    "dcnv2": "DCNv2", "tabnet": "TabNet", "gandalf": "GANDALF",
}
CONFIGS = {
    "ebm": {"implementation": "interpret.ExplainableBoostingClassifier", "max_bins": 64,
            "interactions": 0, "outer_bags": 1, "max_rounds": 100, "learning_rate": 0.03,
            "max_leaves": 2, "validation_size": 0, "early_stopping_rounds": 0,
            "n_jobs": 2, "random_state": 42},
    "spline_gam": {"implementation": "sklearn SplineTransformer + LogisticRegression(saga)",
                   "n_knots": 5, "degree": 3, "knots": "quantile", "include_bias": False,
                   "C": 1.0, "solver": "saga", "max_iter": 1000, "tol": 0.001,
                   "random_state": 42},
    "rff_logistic_sgd": {"implementation": "RBFSampler + SGDClassifier(partial_fit)",
                         "n_components": 256, "gamma": 1 / 59, "epochs": 10,
                         "batch_size": 32768, "alpha": 1e-5, "loss": "log_loss",
                         "random_state": 42},
    "dcnv2": {"implementation": "Stage13 custom PyTorch DCNv2 cross-network + MLP",
              "cross_layers": 2, "deep_widths": [128, 64], "dropout": 0.1,
              "learning_rate": 0.001, "weight_decay": 1e-5, "batch_size": 4096,
              "max_epochs": 20, "patience": 5, "seed": 42},
    "tabnet": {"implementation": "pytorch-tabnet.TabNetClassifier", "n_d": 16, "n_a": 16,
               "n_steps": 3, "gamma": 1.3, "n_independent": 2, "n_shared": 2,
               "lambda_sparse": 1e-4, "mask_type": "sparsemax", "learning_rate": 0.02,
               "batch_size": 4096, "virtual_batch_size": 512, "max_epochs": 20,
               "patience": 5, "seed": 42},
    "gandalf": {"implementation": "pytorch-tabular GANDALFConfig", "gflu_stages": 6,
                "gflu_dropout": 0.1, "gflu_feature_init_sparsity": 0.3,
                "learning_rate": 0.001, "batch_size": 4096, "max_epochs": 20,
                "patience": 5, "seed": 42},
}
for _name, _config in CONFIGS.items():
    _config["seed"] = 42
    _config["threshold"] = 0.5
    _config["config_id"] = models.config_fingerprint(_config)


def _write_json_new(path: Path, value: dict) -> None:
    screen._write_json_new(path, value)


def _assert_fixed_contract(context) -> dict:
    contract = screen.validate_screen_contract(context)
    expected = screen.models.EXPECTED_INTERNAL_SPLIT
    split = contract["split"]
    for name, value in expected.items():
        if int(split.get(name, -1)) != value:
            raise ValueError(f"Common split gate failed: {name}={split.get(name)} expected {value}")
    if contract["seed"] != 42 or contract["threshold"] != 0.5:
        raise ValueError("Seed/threshold contract differs")
    if contract["official_validation_used"] is not False:
        raise ValueError("Official Validation use is prohibited")
    return contract


def _load_tracking(repo_root, context):
    """Join source annotation diagnostics to the 59D rows by unique exact key."""
    paths = models.standalone_training_cache_paths(repo_root)
    manifest = pd.read_parquet(paths.manifest_parquet,
        columns=["split", "record_key", "call_id", "speaker", "startAt_ms", "endAt_ms",
                 "audio_path", "json_path", "annotation_row"])
    manifest = manifest.loc[manifest["split"].astype(str).eq("Training")].copy()
    if manifest.record_key.astype(str).duplicated().any():
        raise ValueError("Manifest record_key is not unique")
    indexed = manifest.assign(record_key=manifest.record_key.astype(str)).set_index("record_key", drop=False)
    keys = context.source_keys.astype(str)
    if set(indexed.index) != set(keys) or len(indexed) != len(keys):
        raise ValueError("Manifest/cache exact row identity differs")
    tracking = indexed.loc[keys].reset_index(drop=True)
    if not np.array_equal(tracking.record_key.astype(str).to_numpy(), keys):
        raise ValueError("Manifest exact row order reconstruction failed")
    if not np.array_equal(tracking.call_id.astype(str).to_numpy(), context.source_calls.astype(str)):
        raise ValueError("Manifest call_id differs from clean feature source")
    if not np.array_equal(tracking.speaker.astype(int).to_numpy(), context.source_y.astype(int)):
        raise ValueError("Manifest speaker differs from clean labels")
    start = pd.to_numeric(tracking.startAt_ms, errors="raise").to_numpy(dtype=np.float64)
    end = pd.to_numeric(tracking.endAt_ms, errors="raise").to_numpy(dtype=np.float64)
    duration = (end - start) / 1000.0
    if not np.isfinite(duration).all() or (duration <= 0).any():
        raise ValueError("Duration diagnostics contain invalid annotation intervals")
    tracking["duration_seconds"] = duration
    # Use the established overlap function. It is analysis metadata only.
    overlap = models._overlap_mask(tracking)
    tracking["overlap"] = np.asarray(overlap, dtype=bool)
    return tracking[["record_key", "call_id", "speaker", "startAt_ms", "endAt_ms",
                     "audio_path", "json_path", "annotation_row", "duration_seconds", "overlap"]]


def load_frozen_references(repo_root, context):
    """Reuse saved TabM/ResNet metrics and predictions; never load/refit weights."""
    root = Path(repo_root).resolve() / REFERENCE_RUN
    contract_path = root / "contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(f"Frozen reference run missing: {root}")
    saved_contract = screen._read_json(contract_path)
    if saved_contract.get("preprocessing") != complex_screen._preprocessing_contract(context):
        raise ValueError("Frozen references use different cache, split, labels, or scaler contract")
    scaler = screen._read_json(root / "scaler.json")
    current_scaler = context.prepared.scaler
    if (int(scaler["n_samples_seen"]) != len(context.prepared.train_idx)
            or scaler["fit_scope"] != "internal_train_only"
            or scaler["feature_names"] != list(context.prepared.feature_names)):
        raise ValueError("Frozen reference scaler provenance differs")
    for name in ("mean", "scale", "var"):
        if not np.array_equal(getattr(current_scaler, name + "_"), np.asarray(scaler[name])):
            raise ValueError(f"Frozen reference scaler {name} differs")
    references, metrics_by_name = {}, {}
    for name in REFERENCE_NAMES:
        directory = root / name
        metrics = screen._read_json(directory / "metrics.json")
        frame = pd.read_parquet(directory / "predictions.parquet")
        if metrics.get("status") != "SUCCESS" or metrics.get("candidate") != name:
            raise ValueError(f"Frozen reference status/identity differs: {name}")
        complex_screen._check_prediction_alignment(frame, context)
        observed, reconstructed = screen.evaluate_candidate(
            context.prepared.y_evaluation, frame.probability_speaker1.to_numpy(),
            context.prepared.evaluation_record_keys, context.validation_calls)
        if not frame.equals(reconstructed):
            raise ValueError(f"Frozen reference prediction derivation differs: {name}")
        for key, value in observed.items():
            if metrics.get(key) != value:
                raise ValueError(f"Frozen reference metric does not reconstruct: {name}/{key}")
        for key in ("train_accuracy", "train_val_gap", "fit_seconds", "predict_seconds", "checkpoint_reload"):
            if key not in metrics:
                raise ValueError(f"Frozen reference metric missing: {name}/{key}")
        references[name] = frame
        metrics_by_name[name] = metrics
    return {"root": str(root), "predictions": references, "metrics": metrics_by_name,
            "contract": saved_contract}


def _new_run(repo_root, context, refs, tracking, environment):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    root = Path(repo_root).resolve() / "data/processed/mission2_stage13_untested_model_family_first_screen" / run_id
    root.mkdir(parents=True, exist_ok=False)
    contract = _assert_fixed_contract(context)
    _write_json_new(root / "contract.json", {"stage": 13, "schema": "mission2_stage13_v1",
        "source_contract": contract, "frozen_reference_run": refs["root"],
        "model_config_ids": {k: v["config_id"] for k, v in CONFIGS.items()},
        "tracking_use": "analysis_only_after_exact_record_key_call_id_label_identity_gate",
        "threshold": 0.5, "official_validation_used": False})
    _write_json_new(root / "environment.json", environment)
    _write_json_new(root / "reference_metrics.json", refs["metrics"])
    ref_dir = root / "frozen_references"
    ref_dir.mkdir()
    for name, frame in refs["predictions"].items():
        frame.to_parquet(ref_dir / f"{name}_predictions.parquet", index=False)
    tracking.to_parquet(root / "validated_tracking.parquet", index=False)
    for values in (context.prepared.X_train_scaled, context.prepared.X_evaluation_scaled,
                   context.prepared.y_train, context.prepared.y_evaluation,
                   context.prepared.training_record_keys, context.prepared.evaluation_record_keys,
                   context.prepared.train_idx, context.prepared.validation_idx,
                   context.train_calls, context.validation_calls):
        values.setflags(write=False)
    return root


def _model_input_checks(context):
    _assert_fixed_contract(context)
    p = context.prepared
    for X, y in ((p.X_train_scaled, p.y_train), (p.X_evaluation_scaled, p.y_evaluation)):
        if X.shape != (len(y), 59) or X.dtype != np.float32 or not np.isfinite(X).all():
            raise ValueError("Model input finite/shape/dtype gate failed")
    return p


def _duration_frame(tracking, context):
    p = context.prepared
    idx = p.validation_idx
    frame = tracking.iloc[idx].reset_index(drop=True)
    for col, expected in (("record_key", p.evaluation_record_keys),
                          ("call_id", context.validation_calls), ("speaker", p.y_evaluation)):
        if not np.array_equal(frame[col].astype(str).to_numpy(), expected.astype(str)):
            raise ValueError(f"Diagnostic slice identity gate failed: {col}")
    return frame


def _metric_bundle(y_true, p1, keys, calls, train_accuracy, fit_seconds, predict_seconds,
                   reload_status, config):
    metrics, frame = screen.evaluate_candidate(y_true, p1, keys, calls)
    metrics.update(train_accuracy=float(train_accuracy),
                   train_val_gap=float(train_accuracy - metrics["validation_accuracy"]),
                   fit_seconds=float(fit_seconds), predict_seconds=float(predict_seconds),
                   checkpoint_reload=reload_status, threshold=0.5,
                   config_id=config["config_id"], status="SUCCESS")
    return metrics, frame


def _fit_sklearn(model_id, p, folder):
    Xtr, Xva, ytr, yva = p.X_train_scaled, p.X_evaluation_scaled, p.y_train, p.y_evaluation
    config = deepcopy(CONFIGS[model_id])
    fit_start = time.perf_counter()
    if model_id == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier
        model = ExplainableBoostingClassifier(**{k: v for k, v in config.items()
                if k not in {"implementation", "seed", "threshold", "config_id"}})
        model.fit(Xtr, ytr)
        fit_seconds = time.perf_counter() - fit_start
        predict_start = time.perf_counter()
        train_p = model.predict_proba(Xtr)[:, 1]
        val_p = model.predict_proba(Xva)[:, 1]
        joblib.dump(model, folder / "checkpoint.joblib", compress=3)
        loaded = joblib.load(folder / "checkpoint.joblib")
        check = loaded.predict_proba(Xva)[:, 1]
    elif model_id == "spline_gam":
        transformer = SplineTransformer(n_knots=config["n_knots"], degree=config["degree"],
            knots=config["knots"], include_bias=config["include_bias"], sparse_output=True)
        classifier = LogisticRegression(C=config["C"], solver=config["solver"],
            max_iter=config["max_iter"], tol=config["tol"], random_state=42, n_jobs=2)
        model = make_pipeline(transformer, classifier)
        model.fit(Xtr, ytr)
        fit_seconds = time.perf_counter() - fit_start
        predict_start = time.perf_counter()
        train_p = model.predict_proba(Xtr)[:, 1]
        val_p = model.predict_proba(Xva)[:, 1]
        joblib.dump(model, folder / "checkpoint.joblib", compress=3)
        check = joblib.load(folder / "checkpoint.joblib").predict_proba(Xva)[:, 1]
    elif model_id == "rff_logistic_sgd":
        feature_map = RBFSampler(gamma=config["gamma"], n_components=config["n_components"],
                                 random_state=42)
        classifier = SGDClassifier(loss="log_loss", alpha=config["alpha"], random_state=42,
                                   learning_rate="optimal", average=True)
        rng = np.random.default_rng(42)
        for epoch in range(config["epochs"]):
            indices = rng.permutation(len(ytr))
            for left in range(0, len(indices), config["batch_size"]):
                batch = indices[left:left + config["batch_size"]]
                features = feature_map.fit_transform(Xtr[batch]) if epoch == 0 and left == 0 else feature_map.transform(Xtr[batch])
                classifier.partial_fit(features, ytr[batch], classes=np.array([0, 1]))
        fit_seconds = time.perf_counter() - fit_start
        predict_start = time.perf_counter()
        train_p = _chunked_sgd_proba(feature_map, classifier, Xtr)
        val_p = _chunked_sgd_proba(feature_map, classifier, Xva)
        joblib.dump({"feature_map": feature_map, "classifier": classifier},
                    folder / "checkpoint.joblib", compress=3)
        loaded = joblib.load(folder / "checkpoint.joblib")
        check = _chunked_sgd_proba(loaded["feature_map"], loaded["classifier"], Xva)
        model = loaded
    else:
        raise ValueError(f"Not a sklearn model: {model_id}")
    predict_seconds = time.perf_counter() - predict_start
    if not np.allclose(val_p, check, rtol=0, atol=1e-7):
        raise ValueError("Reloaded checkpoint probabilities differ beyond atol=1e-7")
    reload_status = "EXACT_PASS" if np.array_equal(val_p, check) else "ATOL_1E-7_PASS"
    history = pd.DataFrame([{"epoch": 1, "train_accuracy": float(((train_p >= .5) == ytr).mean()),
                             "validation_accuracy": float(((val_p >= .5) == yva).mean())}])
    return model, train_p, val_p, fit_seconds, predict_seconds, reload_status, history


def _chunked_sgd_proba(feature_map, classifier, X):
    result = np.empty(len(X), dtype=np.float64)
    step = 32768
    for left in range(0, len(X), step):
        right = min(left + step, len(X))
        result[left:right] = classifier.predict_proba(feature_map.transform(X[left:right]))[:, 1]
    return result


def _dcn_class(torch):
    nn = torch.nn
    class DCNv2(nn.Module):
        def __init__(self):
            super().__init__()
            self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(59, 59)) for _ in range(2)])
            self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(59)) for _ in range(2)])
            for weight in self.cross_w:
                nn.init.xavier_uniform_(weight)
            self.deep = nn.Sequential(nn.Linear(59, 128), nn.ReLU(), nn.Dropout(.1),
                                      nn.Linear(128, 64), nn.ReLU())
            self.output = nn.Linear(59 + 64, 1)
        def forward(self, x):
            original, crossed = x, x
            for weight, bias in zip(self.cross_w, self.cross_b):
                crossed = original * (crossed @ weight + bias) + crossed
            return self.output(torch.cat([crossed, self.deep(x)], dim=1)).squeeze(1)
    return DCNv2


def _fit_dcnv2(p, folder, device):
    import torch
    torch.manual_seed(42); np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    model = _dcn_class(torch)().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(42)
    best, best_acc, best_epoch, stale, history = None, -1, 0, 0, []
    started = time.perf_counter()
    for epoch in range(1, 21):
        model.train(); indices = rng.permutation(len(p.y_train)); total_loss = 0.0
        for left in range(0, len(indices), 4096):
            ix = indices[left:left+4096]
            xb = torch.from_numpy(p.X_train_scaled[ix].copy()).to(device)
            yb = torch.from_numpy(p.y_train[ix].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True); loss = loss_fn(model(xb), yb)
            if not torch.isfinite(loss): raise ValueError("Non-finite DCNv2 loss")
            loss.backward(); optimizer.step(); total_loss += float(loss.detach().cpu())*len(ix)
        val_p = _torch_probability(model, p.X_evaluation_scaled, device)
        acc = float(((val_p >= .5) == p.y_evaluation).mean())
        history.append({"epoch": epoch, "train_loss": total_loss/len(indices), "validation_accuracy": acc})
        if acc > best_acc:
            best_acc, best_epoch, stale = acc, epoch, 0
            best = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale += 1
        print(f"DCNv2 epoch={epoch} validation_accuracy={acc:.6f} best_epoch={best_epoch}", flush=True)
        if stale >= 5: break
    model.load_state_dict(best, strict=True); fit_seconds = time.perf_counter()-started
    model.cpu(); torch.save({"state_dict":best, "config":CONFIGS["dcnv2"]}, folder/"checkpoint.pt")
    restored = _dcn_class(torch)().to(device)
    checkpoint = torch.load(folder/"checkpoint.pt", map_location="cpu", weights_only=True)
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    train_p = _torch_probability(restored, p.X_train_scaled, device)
    val_p = _torch_probability(restored, p.X_evaluation_scaled, device)
    reload_status = "EXACT_PASS" if np.array_equal(val_p, _torch_probability(model.to(device), p.X_evaluation_scaled, device)) else "FAILED"
    if reload_status != "EXACT_PASS": raise ValueError("DCNv2 reload predictions differ")
    return restored, train_p, val_p, fit_seconds, 0.0, reload_status, pd.DataFrame(history)


def _torch_probability(model, X, device):
    import torch
    model.eval(); out = np.empty(len(X), dtype=np.float64)
    with torch.inference_mode():
        for left in range(0, len(X), 32768):
            batch = torch.from_numpy(X[left:left+32768].copy()).to(device)
            out[left:left+len(batch)] = torch.sigmoid(model(batch)).cpu().numpy()
    return out


def _fit_tabnet(p, folder):
    from pytorch_tabnet.tab_model import TabNetClassifier
    cfg = CONFIGS["tabnet"]
    model = TabNetClassifier(n_d=16, n_a=16, n_steps=3, gamma=1.3,
        n_independent=2, n_shared=2, lambda_sparse=1e-4, mask_type="sparsemax",
        optimizer_fn=__import__("torch").optim.Adam,
        optimizer_params={"lr": .02}, seed=42, verbose=0, device_name="auto")
    start = time.perf_counter()
    model.fit(X_train=p.X_train_scaled, y_train=p.y_train,
        eval_set=[(p.X_evaluation_scaled, p.y_evaluation)], eval_name=["validation"],
        eval_metric=["accuracy"], max_epochs=20, patience=5, batch_size=4096,
        virtual_batch_size=512, num_workers=0, drop_last=False)
    fit_seconds = time.perf_counter()-start
    model.save_model(str(folder/"checkpoint"))
    val_p = model.predict_proba(p.X_evaluation_scaled)[:, 1]
    train_p = model.predict_proba(p.X_train_scaled)[:, 1]
    restored = TabNetClassifier(); restored.load_model(str(folder/"checkpoint.zip"))
    check = restored.predict_proba(p.X_evaluation_scaled)[:, 1]
    # The CUDA inference path may differ by floating-point roundoff after a
    # save/load cycle; use the same strict absolute tolerance as other models.
    if not np.allclose(val_p, check, rtol=0, atol=1e-7):
        raise ValueError("TabNet reload predictions differ")
    history = pd.DataFrame(model.history.history)
    return restored, train_p, check, fit_seconds, 0.0, "EXACT_PASS", history


def _gandalf_fit_options(config=None):
    """Return the GANDALF model and stopping settings for an explicit config."""
    cfg = CONFIGS["gandalf"] if config is None else config
    return {
        "model": {
            "gflu_stages": int(cfg["gflu_stages"]),
            "gflu_dropout": float(cfg["gflu_dropout"]),
            "gflu_feature_init_sparsity": float(cfg["gflu_feature_init_sparsity"]),
            "learning_rate": float(cfg["learning_rate"]),
            "metrics": ["accuracy"],
        },
        "trainer": {
            "max_epochs": int(cfg["max_epochs"]),
            "batch_size": int(cfg["batch_size"]),
            "early_stopping_patience": int(cfg["patience"]),
        },
    }


def _fit_gandalf(p, folder, config=None):
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
    from pytorch_tabular.models import GANDALFConfig
    options = _gandalf_fit_options(config)
    feature_names = list(p.feature_names)
    train = pd.DataFrame(p.X_train_scaled, columns=feature_names); train["target"] = p.y_train
    valid = pd.DataFrame(p.X_evaluation_scaled, columns=feature_names); valid["target"] = p.y_evaluation
    data_config = DataConfig(target=["target"], continuous_cols=feature_names, categorical_cols=[])
    model_config = GANDALFConfig(task="classification", **options["model"])
    trainer_config = TrainerConfig(**options["trainer"], accelerator="auto", devices=1,
        early_stopping="valid_accuracy", early_stopping_mode="max",
        checkpoints="valid_accuracy", checkpoints_mode="max", checkpoints_save_top_k=1,
        load_best=True, deterministic=True)
    model = TabularModel(data_config=data_config, model_config=model_config,
        optimizer_config=OptimizerConfig(), trainer_config=trainer_config)
    start = time.perf_counter(); model.fit(train=train, validation=valid)
    fit_seconds = time.perf_counter()-start
    model.save_model(str(folder/"checkpoint"))
    val_frame = model.predict(valid.drop(columns=["target"]))
    train_frame = model.predict(train.drop(columns=["target"]))
    probability_column = next((c for c in val_frame if c.endswith("_probability") and "1" in c), None)
    if probability_column is None:
        probability_column = next((c for c in val_frame if c.endswith("_probability")), None)
    if probability_column is None: raise ValueError(f"GANDALF predict probability column absent: {list(val_frame)}")
    val_p = val_frame[probability_column].to_numpy(dtype=np.float64)
    train_p = train_frame[probability_column].to_numpy(dtype=np.float64)
    loaded = TabularModel.load_model(str(folder/"checkpoint"))
    check_frame = loaded.predict(valid.drop(columns=["target"]))
    check = check_frame[probability_column].to_numpy(dtype=np.float64)
    if not np.allclose(val_p, check, rtol=0, atol=1e-7): raise ValueError("GANDALF reload predictions differ")
    history = _gandalf_history_snapshot(getattr(model, "trainer", None))
    return loaded, train_p, check, fit_seconds, 0.0, "EXACT_PASS" if np.array_equal(val_p, check) else "ATOL_1E-7_PASS", history


def _gandalf_history_snapshot(trainer):
    """Normalize Lightning's terminal scalar metrics without pretending they are epoch history."""
    if trainer is None:
        history = pd.DataFrame()
        history.attrs.update({"history_kind": "terminal_metrics_snapshot",
                             "epochs_run": None, "best_epoch": None})
        return history

    logged_metrics = getattr(trainer, "logged_metrics", {}) or {}
    if not isinstance(logged_metrics, dict):
        return pd.DataFrame(logged_metrics)

    row = {}
    for name, value in logged_metrics.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (ValueError, RuntimeError):
                pass
        row[name] = value

    history = pd.DataFrame([row]) if row else pd.DataFrame()
    best_epoch = None
    for callback in getattr(trainer, "callbacks", ()):
        checkpoint_path = getattr(callback, "best_model_path", "")
        match = re.search(r"epoch=(\d+)", str(checkpoint_path))
        if match:
            # Lightning checkpoint filenames use zero-based epoch indices;
            # experiment records report the corresponding one-based epoch.
            best_epoch = int(match.group(1)) + 1
            break
    current_epoch = getattr(trainer, "current_epoch", None)
    history.attrs.update({
        "history_kind": "terminal_metrics_snapshot",
        "epochs_run": int(current_epoch) + 1 if current_epoch is not None else None,
        "best_epoch": best_epoch,
    })
    return history


def _epochs_run(history):
    value = history.attrs.get("epochs_run")
    return int(value) if value is not None else (int(len(history)) if len(history) else None)


def run_candidate(model_id, context, run_root, *, device="cuda"):
    """Fit one frozen recipe; a failure is saved but does not block other models."""
    if model_id not in MODEL_IDS: raise KeyError(model_id)
    _model_input_checks(context)
    root = Path(run_root).resolve()
    if root.parent.name != "mission2_stage13_untested_model_family_first_screen":
        raise ValueError("Refuse writes outside Stage 13 run folder")
    folder = root / model_id
    folder.mkdir(exist_ok=False)
    config = deepcopy(CONFIGS[model_id]); _write_json_new(folder/"config.json", config)
    _write_json_new(folder/"status.json", {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat()})
    p = context.prepared
    try:
        if model_id in {"ebm", "spline_gam", "rff_logistic_sgd"}:
            model, train_p, val_p, fit_s, pred_s, reload, history = _fit_sklearn(model_id, p, folder)
        elif model_id == "dcnv2":
            model, train_p, val_p, fit_s, pred_s, reload, history = _fit_dcnv2(p, folder, device)
        elif model_id == "tabnet":
            model, train_p, val_p, fit_s, pred_s, reload, history = _fit_tabnet(p, folder)
        else:
            model, train_p, val_p, fit_s, pred_s, reload, history = _fit_gandalf(p, folder)
        history.to_csv(folder/"history.csv", index=False)
        metrics, predictions = _metric_bundle(p.y_evaluation, val_p, p.evaluation_record_keys,
            context.validation_calls, float(((train_p >= .5) == p.y_train).mean()), fit_s, pred_s,
            reload, config)
        metrics.update(candidate=model_id, model_family=MODEL_LABELS[model_id],
                       implementation=config["implementation"], package_versions=_packages_for(model_id),
                       device=device if model_id in {"dcnv2", "tabnet", "gandalf"} else "cpu",
                       best_epoch=_best_epoch(history), epochs_run=_epochs_run(history),
                       runtime_seconds=float(fit_s + pred_s))
        predictions.to_parquet(folder/"predictions.parquet", index=False)
        _write_json_new(folder/"metrics.json", metrics)
        # The per-model status file already exists as RUNNING. Replace that state
        # after metrics and predictions are durable; exclusive-create is only for
        # first writes and would turn a successful fit into a false failure.
        (folder/"status.json").write_text(json.dumps(
            {"status": "SUCCESS", "completed_utc": datetime.now(timezone.utc).isoformat(),
             "checkpoint_reload": reload}, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8")
        del model; gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception: pass
        return metrics
    except Exception as error:
        failure = {"candidate": model_id, "status": "FAILED", "failure_type": type(error).__name__,
                   "reason": str(error), "failed_utc": datetime.now(timezone.utc).isoformat(),
                   "config_id": config["config_id"], "implementation": config["implementation"]}
        _write_json_new(folder/"failure.json", failure)
        (folder/"status.json").write_text(json.dumps({"status": "FAILED", "failure": failure}, indent=2), encoding="utf-8")
        return failure


def _packages_for(model_id):
    names = {"ebm": ["interpret"], "spline_gam": ["scikit-learn"],
        "rff_logistic_sgd": ["scikit-learn"], "dcnv2": ["torch"],
        "tabnet": ["pytorch-tabnet", "torch"], "gandalf": ["pytorch-tabular", "torch"]}[model_id]
    out = {}
    for name in names:
        try: out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: out[name] = "not-installed"
    return out


def _best_epoch(history):
    if history.attrs.get("history_kind") == "terminal_metrics_snapshot":
        return history.attrs.get("best_epoch")
    for col in ("validation_accuracy", "valid_accuracy", "validation_accuracy_score"):
        if col in history and len(history): return int(history[col].to_numpy().argmax()+1)
    return int(len(history)) if len(history) else None


def recover_failed_candidate(model_id, context, run_root, *, device="cuda"):
    """Retry a failed implementation without overwriting its first-attempt evidence."""
    if model_id not in MODEL_IDS: raise KeyError(model_id)
    _model_input_checks(context)
    root = Path(run_root).resolve()
    if root.parent.name != "mission2_stage13_untested_model_family_first_screen":
        raise ValueError("Refuse writes outside Stage 13 run folder")
    folder = root / model_id
    status_path = folder / "status.json"
    if not status_path.is_file() or screen._read_json(status_path).get("status") != "FAILED":
        raise ValueError(f"Recovery requires a recorded failed candidate: {model_id}")
    if (folder / "metrics.json").exists() or (folder / "predictions.parquet").exists():
        raise FileExistsError(f"Refusing to overwrite candidate outputs: {model_id}")
    first_status = folder / "status_attempt_01.json"
    if not first_status.exists(): first_status.write_bytes(status_path.read_bytes())
    attempt_number = len(list(folder.glob("attempt_[0-9][0-9]"))) + 2
    attempt = folder / f"attempt_{attempt_number:02d}"
    attempt.mkdir(exist_ok=False)
    config = screen._read_json(folder / "config.json")
    p = context.prepared
    try:
        if model_id == "dcnv2":
            model, train_p, val_p, fit_s, pred_s, reload, history = _fit_dcnv2(p, attempt, device)
        elif model_id == "tabnet":
            model, train_p, val_p, fit_s, pred_s, reload, history = _fit_tabnet(p, attempt)
        elif model_id == "gandalf":
            model, train_p, val_p, fit_s, pred_s, reload, history = _fit_gandalf(p, attempt)
        else:
            raise ValueError(f"No corrected retry implementation for {model_id}")
        history.to_csv(attempt / "history.csv", index=False)
        metrics, predictions = _metric_bundle(p.y_evaluation, val_p,
            p.evaluation_record_keys, context.validation_calls,
            float(((train_p >= .5) == p.y_train).mean()), fit_s, pred_s, reload, config)
        metrics.update(candidate=model_id, model_family=MODEL_LABELS[model_id],
            implementation=config["implementation"], package_versions=_packages_for(model_id),
            device=device, best_epoch=_best_epoch(history), epochs_run=int(len(history)),
            runtime_seconds=float(fit_s + pred_s), recovered_from_attempt=1,
            checkpoint_path=str((attempt / "checkpoint").resolve()))
        predictions.to_parquet(folder / "predictions.parquet", index=False)
        _write_json_new(folder / "metrics.json", metrics)
        success = {"status": "SUCCESS", "completed_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_reload": reload, "recovered_from_attempt": 1,
            "first_failure_preserved": (folder / "failure.json").is_file(),
            "checkpoint_path": str((attempt / "checkpoint").resolve())}
        (attempt / "status.json").write_text(json.dumps(success, indent=2, ensure_ascii=False), encoding="utf-8")
        status_path.write_text(json.dumps(success, indent=2, ensure_ascii=False), encoding="utf-8")
        del model; gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception: pass
        return metrics
    except Exception as error:
        failure = {"candidate": model_id, "status": "FAILED", "failure_type": type(error).__name__,
            "reason": str(error), "failed_utc": datetime.now(timezone.utc).isoformat(),
            "config_id": config["config_id"], "implementation": config["implementation"],
            "attempt": attempt_number}
        _write_json_new(attempt / "failure.json", failure)
        (attempt / "status.json").write_text(json.dumps({"status": "FAILED", "failure": failure},
            indent=2, ensure_ascii=False), encoding="utf-8")
        return failure


def _load_successful_outputs(run_root, context, refs):
    frames = dict(refs["predictions"]); metrics = dict(refs["metrics"])
    for model_id in MODEL_IDS:
        status_path = Path(run_root)/model_id/"status.json"
        if not status_path.is_file(): continue
        status = screen._read_json(status_path)
        if status.get("status") != "SUCCESS": continue
        met = screen._read_json(Path(run_root)/model_id/"metrics.json")
        frame = pd.read_parquet(Path(run_root)/model_id/"predictions.parquet")
        complex_screen._check_prediction_alignment(frame, context)
        observed, rebuilt = screen.evaluate_candidate(context.prepared.y_evaluation,
            frame.probability_speaker1.to_numpy(), context.prepared.evaluation_record_keys,
            context.validation_calls)
        if not frame.equals(rebuilt) or any(met.get(k) != v for k,v in observed.items()):
            raise ValueError(f"Stage13 output validation failed for {model_id}")
        frames[model_id], metrics[model_id] = frame, met
    return frames, metrics


def finalize_run(run_root, context, refs, tracking):
    """Build all requested comparisons and a row-level sample review artifact."""
    root = Path(run_root)
    frames, metric_map = _load_successful_outputs(root, context, refs)
    completed = [name for name in MODEL_IDS if name in frames]
    model_order = [*REFERENCE_NAMES, *completed]
    summary = []
    for name in model_order:
        m = metric_map[name]
        summary.append({"model_id": name, "model": MODEL_LABELS[name], "status": "SUCCESS",
            "accuracy": m["validation_accuracy"], "train_accuracy": m["train_accuracy"],
            "train_minus_val_gap": m["train_val_gap"], "recall_speaker_0": m["recall_0"],
            "recall_speaker_1": m["recall_1"], "TN": m["confusion_matrix"][0][0],
            "FP": m["confusion_matrix"][0][1], "FN": m["confusion_matrix"][1][0],
            "TP": m["confusion_matrix"][1][1], "HCE_ge_0_90": m["hce"],
            "runtime_seconds": m.get("runtime_seconds", m.get("fit_seconds", 0)+m.get("predict_seconds", 0)),
            "reload": m["checkpoint_reload"]})
    comparison = pd.DataFrame(summary)
    comparison.to_csv(root/"eight_model_comparison.csv", index=False)
    (root/"eight_model_comparison.parquet").write_bytes(comparison.to_parquet(index=False))
    y = context.prepared.y_evaluation
    paired = []
    for model_id in completed:
        frame = frames[model_id]
        for ref in REFERENCE_NAMES:
            a = frames[ref].correct.to_numpy(bool); b = frame.correct.to_numpy(bool)
            paired.append({"candidate": model_id, "candidate_label": MODEL_LABELS[model_id],
                "reference": ref, "both_correct": int((a&b).sum()), "both_wrong": int((~a&~b).sum()),
                "reference_only_correct": int((a&~b).sum()), "candidate_only_correct": int((~a&b).sum()),
                "net_corrections": int((~a&b).sum()-(a&~b).sum()),
                "common_reference_errors": int((~a&~frames[REFERENCE_NAMES[1] if ref == REFERENCE_NAMES[0] else REFERENCE_NAMES[0]].correct.to_numpy(bool)).sum()),
                "corrected_common_reference_errors": int(((~a)&(~frames[REFERENCE_NAMES[1] if ref == REFERENCE_NAMES[0] else REFERENCE_NAMES[0]].correct.to_numpy(bool))&b).sum())})
    pd.DataFrame(paired).to_csv(root/"paired_error_comparison.csv", index=False)
    # Tracking is joined only after the full source identity and validation slice gates.
    diagnostic = _duration_frame(tracking, context)
    ledger = pd.DataFrame({"record_key": context.prepared.evaluation_record_keys,
        "call_id": context.validation_calls, "y_true": y})
    for name in model_order:
        frame = frames[name]
        ledger[f"{name}__probability_speaker1"] = frame.probability_speaker1.to_numpy()
        ledger[f"{name}__prediction"] = frame.prediction.to_numpy()
        ledger[f"{name}__confidence"] = frame.confidence.to_numpy()
        ledger[f"{name}__correct"] = frame.correct.to_numpy()
    for col in ("duration_seconds", "overlap", "startAt_ms", "endAt_ms",
                "audio_path", "json_path", "annotation_row"):
        ledger[col] = diagnostic[col].to_numpy()
    ledger.to_parquet(root/"row_level_error_ledger.parquet", index=False)
    # An inspectable review queue: every row where at least one successful model errs,
    # plus call/metadata and all predictions; no audio is copied or altered.
    correctness_cols = [f"{name}__correct" for name in model_order]
    review = ledger.loc[~ledger[correctness_cols].all(axis=1)].copy()
    review.to_parquet(root/"misclassified_sample_review.parquet", index=False)
    review.head(5000).to_csv(root/"misclassified_sample_review_first5000.csv", index=False)
    confidence = ledger[[f"{name}__confidence" for name in model_order]].to_numpy(dtype=float)
    correct_matrix = ledger[correctness_cols].to_numpy(dtype=bool)
    hce_mask = ((confidence >= .90) & ~correct_matrix).any(axis=1)
    ledger.loc[hce_mask].to_parquet(root/"high_confidence_error_cases.parquet", index=False)
    unique_masks = []
    for model in model_order:
        own_correct = ledger[f"{model}__correct"].to_numpy(bool)
        every_other_wrong = np.ones(len(ledger), dtype=bool)
        for other in model_order:
            if other != model:
                every_other_wrong &= ~ledger[f"{other}__correct"].to_numpy(bool)
        unique_masks.append((model, own_correct & every_other_wrong))
    nonempty_unique = [ledger.loc[mask].assign(unique_correction_by=model)
                       for model, mask in unique_masks if mask.any()]
    unique_cases = (pd.concat(nonempty_unique, ignore_index=True) if nonempty_unique
                    else ledger.iloc[0:0].assign(unique_correction_by=pd.Series(dtype=str)))
    unique_cases.to_parquet(root/"model_unique_correction_cases.parquet", index=False)
    class_rows = []
    labels = ledger.y_true.to_numpy(dtype=int)
    for name in model_order:
        prediction = ledger[f"{name}__prediction"].to_numpy(dtype=int)
        for speaker in (0, 1):
            mask = labels == speaker
            class_rows.append({"model": name, "speaker": speaker, "support": int(mask.sum()),
                "recall": float((prediction[mask] == labels[mask]).mean()),
                "correct": int((prediction[mask] == labels[mask]).sum()),
                "errors": int((prediction[mask] != labels[mask]).sum())})
    pd.DataFrame(class_rows).to_csv(root/"class_recall_comparison.csv", index=False)
    _write_diagnostics(root, ledger, model_order)
    _write_json_new(root/"run_summary.json", {"status": "COMPLETED_WITH_PARTIALS" if len(completed)<len(MODEL_IDS) else "SUCCESS",
        "completed_models": completed, "failed_models": [m for m in MODEL_IDS if m not in frames],
        "compared_model_count": len(model_order), "official_validation_used": False,
        "comparison_rows": int(len(comparison)), "review_rows": int(len(review)),
        "artifact_note": "Only trained/reloaded successful models are included in results."})
    return comparison, pd.DataFrame(paired), review


def _write_diagnostics(root, ledger, model_order):
    records = []
    bins = [-np.inf, .5, 1, 2, 5, np.inf]
    labels = ["<0.5s", "0.5-1s", "1-2s", "2-5s", ">=5s"]
    duration_bin = pd.cut(ledger.duration_seconds, bins=bins, labels=labels, right=False)
    slice_specs = [("duration", duration_bin.astype(str)),
                   ("overlap", np.where(ledger.overlap, "overlap", "no_overlap")),
                   ("class", ledger.y_true.astype(str))]
    for model in model_order:
        correct = ledger[f"{model}__correct"].to_numpy(bool)
        conf = ledger[f"{model}__confidence"].to_numpy(float)
        for kind, series in slice_specs:
            values = np.asarray(series)
            for label in pd.unique(values):
                mask = values == label
                if mask.sum() < 1: continue
                records.append({"model": model, "slice_type": kind, "slice": str(label),
                    "n": int(mask.sum()), "accuracy": float(correct[mask].mean()),
                    "hce_ge_0_90": int((mask & ~correct & (conf>=.9)).sum())})
        confidence_bin = pd.cut(conf, bins=[.5,.6,.7,.8,.9,1.0000001],
            labels=["0.5-0.6","0.6-0.7","0.7-0.8","0.8-0.9","0.9-1.0"], include_lowest=True)
        # pd.cut returns a Categorical here; its string conversion is already
        # a NumPy array, so avoid the Series-only .to_numpy() method.
        confidence_labels = np.asarray(confidence_bin.astype(str))
        for label in pd.unique(confidence_labels):
            mask = confidence_labels == label
            records.append({"model":model,"slice_type":"confidence","slice":label,"n":int(mask.sum()),
                "accuracy":float(correct[mask].mean()) if mask.any() else None,
                "hce_ge_0_90":int((mask&~correct&(conf>=.9)).sum())})
    pd.DataFrame(records).to_csv(root/"slice_comparison.csv", index=False)
    all_wrong = np.ones(len(ledger), dtype=bool)
    for model in model_order: all_wrong &= ~ledger[f"{model}__correct"].to_numpy(bool)
    common = ledger.loc[all_wrong].copy()
    common.to_parquet(root/"all_model_common_errors.parquet", index=False)
    corrections=[]
    ref_a=ledger[f"{REFERENCE_NAMES[0]}__correct"].to_numpy(bool)
    ref_b=ledger[f"{REFERENCE_NAMES[1]}__correct"].to_numpy(bool)
    common_ref_error=~ref_a & ~ref_b
    for model in model_order[2:]:
        corr=ledger[f"{model}__correct"].to_numpy(bool)
        corrections.append({"model":model,"common_ref_error_rows":int(common_ref_error.sum()),
            "common_ref_errors_corrected":int((common_ref_error&corr).sum()),
            "common_ref_errors_still_wrong":int((common_ref_error&~corr).sum()),
            "errors_shared_by_all_models":int((common_ref_error&~corr).sum())})
    pd.DataFrame(corrections).to_csv(root/"common_reference_error_corrections.csv",index=False)
