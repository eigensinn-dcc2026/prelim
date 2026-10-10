"""M1-HEAD-001: cached 564D features, fixed Training holdout, RBF-SVM.

sklearn batch fit has no optimizer/epoch resume API. Atomic CV-fold results
and fitted head checkpoints resume completed stages; interrupted fits restart.
No feature extraction, pretrained model loading, or official Validation access.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import tempfile
import time
import unicodedata
import zipfile

EXP_ID = "M1-HEAD-001"
SEED = 42
CS = (0.3, 1.0, 3.0)
CONFIG = dict(exp_id=EXP_ID, seed=SEED, source="M1-XVEC-001", schema=1,
              candidates=list(CS), folds=5, expected_rows=29200,
              train_rows=23360, holdout_rows=5840)


def baseline_module():
    spec = importlib.util.spec_from_file_location("m1_xvec_cache", Path(__file__).with_name("m1-xvec-001.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = baseline_module()
atomic = BASE.atomic
write_json = BASE.write_json


def load_cache(source):
    """Validate all feature chunks and split before touching the dataset."""
    split_path = source / "split.json"
    paths = sorted((source / "features").glob("chunk_*.npz"))
    if not split_path.is_file() or not paths:
        raise ValueError(f"Required baseline split/features missing: {source}")
    import numpy as np
    split = json.loads(split_path.read_text())
    arrays, durations, cursor = [], [], 0
    for path in paths:
        match = re.fullmatch(r"chunk_(\d+)_(\d+)\.npz", path.name)
        if match is None:
            raise ValueError(f"Invalid chunk filename: {path}")
        start, stop = map(int, match.groups())
        if start != cursor or stop <= start:
            raise ValueError(f"Noncontiguous feature chunks: {path}")
        with np.load(path, allow_pickle=False) as chunk:
            a, x, d = chunk["acoustic"], chunk["xvector"], chunk["durations"]
            if a.shape != (stop-start, 52) or x.shape != (stop-start, 512) or d.shape != (stop-start,):
                raise ValueError(f"Invalid feature dimensions: {path}")
            if not all(np.isfinite(v).all() for v in (a, x, d)) or (d < 0).any():
                raise ValueError(f"Invalid feature values: {path}")
            arrays.append(np.concatenate([a, x], axis=1))
            durations.append(d.copy())
        cursor = stop
    if cursor != 29200:
        raise ValueError(f"Expected 29,200 rows, got {cursor}")
    indices = []
    for name, size in (("train", 23360), ("holdout", 5840)):
        raw = split.get(name)
        if not isinstance(raw, list) or len(raw) != size or any(type(i) is not int for i in raw):
            raise ValueError(f"Invalid {name} split")
        indices.append(np.asarray(raw, dtype=int))
    if not np.array_equal(np.sort(np.concatenate(indices)), np.arange(cursor)):
        raise ValueError("Split must partition every call exactly once")
    if not isinstance(split.get("rows_sha256"), str):
        raise ValueError("Baseline row identity missing")
    digest = hashlib.sha256(split_path.read_bytes())
    for path in paths:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return np.concatenate(arrays), np.concatenate(durations), split, indices, digest.hexdigest()


def training_metadata(work):
    """Reuse baseline ordering/identity; copy labels only, inspect audio zip index."""
    from dcc_data.colab_env import find_data_root
    from dcc_data.mission1 import _read_gender
    from dcc_data.mission1_train import encode_gender
    source = find_data_root() / "Training"
    labels, members = BASE.unpack(source / "TL_서울_구급.zip", work)

    def index(names, suffix):
        result = {}
        for name in names:
            path = Path(name)
            if path.suffix.lower() != suffix:
                continue
            key = unicodedata.normalize("NFC", path.stem).casefold()
            if key in result:
                raise ValueError(f"Duplicate call stem: {key}")
            result[key] = path
        return result

    jsons = index(members, ".json")
    with zipfile.ZipFile(source / "TS_서울_구급.zip") as archive:
        wavs = index((m.filename for m in archive.infolist() if not m.is_dir()), ".wav")
    if jsons.keys() != wavs.keys() or len(jsons) != 29200:
        raise ValueError("Expected 29,200 paired Training calls")
    calls, genders, paths, files = [], [], [], []
    for key in sorted(jsons):
        path = labels / jsons[key]
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        calls.append(str(doc.get("call_id") or doc.get("_id") or key))
        genders.append(_read_gender(doc, path))
        paths.append((str(Path("TS_서울_구급") / wavs[key]), str(path.relative_to(work))))
        files.append(path.stem)
    if len(set(calls)) != len(calls):
        raise ValueError("Duplicate call_id")
    identity = dict(calls=calls, genders=genders, paths=paths)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return encode_gender(genders), calls, files, digest


def svm(c):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return make_pipeline(StandardScaler(), SVC(C=c, kernel="rbf", gamma="scale", class_weight=None))


def choose_c(records):
    # Exact ties favor the smaller C, independent of candidate iteration order.
    return min(CS, key=lambda c: (-records[str(c)]["mean"], c))


def run_cv(out, x, y):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(x, y))
    records = {}
    for c in CS:
        results = []
        for fold, (train, val) in enumerate(folds):
            path = out / "cv" / f"C_{c}_fold_{fold}.json"
            if path.exists():
                result = json.loads(path.read_text())
            else:
                began = time.perf_counter()
                model = svm(c)
                model.fit(x[train], y[train])
                result = dict(C=c, fold=fold, accuracy=float(np.mean(model.predict(x[val]) == y[val])),
                              seconds=time.perf_counter()-began)
                write_json(path, result)
                print(f"CV C={c}, fold={fold}: {result['accuracy']:.6f}", flush=True)
            results.append(result)
        scores = [r["accuracy"] for r in results]
        records[str(c)] = dict(folds=results, mean=float(np.mean(scores)), std=float(np.std(scores)), std_ddof=0)
    selected = choose_c(records)
    write_json(out / "cv_results.json", dict(candidates=records, selected_C=selected))
    return selected, records


def fit_head(out, name, factory, x, y):
    import joblib
    checkpoint = out / f"{name}.ckpt"
    if checkpoint.exists():
        model = joblib.load(checkpoint)
        return model, float(model.dcc_fit_seconds_)
    model = factory()
    began = time.perf_counter()
    model.fit(x, y)
    model.dcc_fit_seconds_ = time.perf_counter()-began
    atomic(checkpoint, lambda p: joblib.dump(model, p))
    return model, float(model.dcc_fit_seconds_)


def diagnostics(y, predictions, durations):
    import numpy as np
    from scipy.stats import binomtest
    b, e = predictions["b"] == y, predictions["e"] == y
    both_right, b_only, e_only, both_wrong = [int(np.sum(mask)) for mask in
                                            (b & e, b & ~e, ~b & e, ~b & ~e)]
    bins = {}
    for name, mask in (("<=20s", durations <= 20), ("20-40s", (durations > 20) & (durations <= 40)),
                       (">40s", durations > 40)):
        bins[name] = dict(rows=int(mask.sum()), b_accuracy=float(b[mask].mean()) if mask.any() else None,
                         e_accuracy=float(e[mask].mean()) if mask.any() else None)
    return dict(paired_mcnemar=dict(table=[[both_right, b_only], [e_only, both_wrong]],
                    rows=["b_correct", "b_wrong"], columns=["e_correct", "e_wrong"],
                    exact_two_sided_p=float(binomtest(e_only, b_only+e_only, .5).pvalue) if b_only+e_only else 1.),
                error_overlap=dict(both_wrong=both_wrong, b_only_wrong=e_only, e_only_wrong=b_only),
                duration_bins=bins)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=os.environ.get("LAB_OUT"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    if args.out is None:
        parser.error("--out or LAB_OUT is required")
    if os.environ.get("LAB_EXP_ID", EXP_ID) != EXP_ID:
        raise ValueError("LAB_EXP_ID mismatch")
    source = Path(os.environ["DCC_RUN_ROOT"]) / "lab" / "M1-XVEC-001"
    x, durations, split, (train, holdout), fingerprint = load_cache(source)
    import numpy as np
    import torch
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from dcc_data.mission1_train import _build_lr_pipeline, compute_metrics
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    out = args.out
    config = dict(CONFIG, cache_sha256=fingerprint)
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text()) != config:
        raise ValueError("Output configuration/cache mismatch; use a fresh LAB_OUT")
    write_json(out / "config.json", config)
    if (out / "metrics.json").exists() and (out / "best.ckpt").exists():
        print("Experiment already complete", flush=True)
        return
    work = Path(os.environ["LAB_WORK"]) / EXP_ID.lower()
    work.mkdir(parents=True, exist_ok=True)
    y, calls, files, digest = training_metadata(work)
    if digest != split["rows_sha256"]:
        raise ValueError("Training metadata does not match baseline cached row identity")
    expected = train_test_split(np.arange(len(y)), test_size=.2, random_state=SEED, stratify=y)
    if not all(np.array_equal(a, b) for a, b in zip((train, holdout), expected)):
        raise ValueError("Baseline split differs from fixed seed-42 stratified split")
    if {calls[i] for i in train} & {calls[i] for i in holdout}:
        raise ValueError("call_id leakage")
    write_json(out / "split.json", split)
    selected, cv = run_cv(out, x[train], y[train])
    factories = dict(b=lambda: _build_lr_pipeline(SEED), e=lambda: svm(selected),
                     f=lambda: HistGradientBoostingClassifier(random_state=SEED),
                     g=lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256,),
                         early_stopping=True, validation_fraction=.1, random_state=SEED)))
    scores, predictions, columns = {}, {}, []
    for name, factory in factories.items():
        model, fit_seconds = fit_head(out, name, factory, x[train], y[train])
        path = out / f"{name}_holdout.npz"
        timing = out / f"{name}_metrics.json"
        if path.exists() and timing.exists():
            with np.load(path, allow_pickle=False) as saved:
                pred, values = saved["predictions"].copy(), saved["scores"].copy()
            report = json.loads(timing.read_text())
        else:
            began = time.perf_counter()
            pred = model.predict(x[holdout])
            predict_seconds = time.perf_counter()-began
            values = model.decision_function(x[holdout]) if name == "e" else model.predict_proba(x[holdout])[:, 1]
            report = dict(compute_metrics(y[holdout], pred), fit_seconds=fit_seconds,
                          holdout_predict_seconds=predict_seconds)
            BASE.save_arrays(path, predictions=pred, scores=values)
            write_json(timing, report)
        predictions[name], scores[name] = pred, report
        columns.append(values)
        if name == "e":
            atomic(out / "best.ckpt", lambda p: joblib.dump(model, p))
        print(f"Head ({name}): {report['accuracy']:.6f}", flush=True)
    seed_out = out / "seed_42"
    atomic(seed_out / "val_files.txt", lambda p: p.write_text("\n".join(files[i] for i in holdout)+"\n", encoding="utf-8"))
    def save_probs(path):
        with path.open("wb") as stream:
            np.save(stream, np.column_stack(columns), allow_pickle=False)
    atomic(seed_out / "holdout_probs.npy", save_probs)
    write_json(seed_out / "holdout_probs_columns.json", ["b_LR_predict_proba_M", "e_SVM_decision_function",
                                                        "f_HGB_predict_proba_M", "g_MLP_predict_proba_M"])
    BASE.save_arrays(seed_out / "holdout_targets.npz", targets=y[holdout], indices=holdout)
    details = diagnostics(y[holdout], predictions, durations[holdout])
    gain = scores["e"]["accuracy"] - scores["b"]["accuracy"]
    reproduced = abs(scores["b"]["accuracy"] - .9738) <= .002
    write_json(out / "metrics.json", dict(exp_id=EXP_ID,
        primary_metric=dict(name="accuracy", value=scores["e"]["accuracy"], split="internal_holdout"),
        metrics=dict(heads=scores, cv=cv, selected_C=selected, accuracy_gain= gain,
                     baseline_reference=.9738, baseline_reproduced=reproduced,
                     success_criteria_met=reproduced and gain >= .004 and details["paired_mcnemar"]["exact_two_sided_p"] < .05,
                     **details), seed=SEED, train_rows=len(train), eval_rows=len(holdout), official_val=None,
        ckpt="best.ckpt", notes="Fixed primary head (e); (f)/(g) diagnostic only. "
        "No official Validation. Interrupted sklearn fits restart; completed folds/heads resume. "
        + ("LR reproduced." if reproduced else "LR reproduction failed: outside 0.9738 +/- 0.002.")))


def smoke():
    """No fitting, downloads, or dataset access."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_json(root / "state.json", {"done": False})
        write_json(root / "state.json", {"done": True})
        assert json.loads((root / "state.json").read_text()) == {"done": True}
        assert choose_c({str(c): {"mean": .9} for c in CS}) == .3
        assert choose_c({str(c): {"mean": c / 10} for c in CS}) == 3.
        try:
            load_cache(root)
        except ValueError:
            pass
        else:
            raise AssertionError("Missing cache accepted")
    print("Smoke OK: atomic resume writes, C selection/tie, missing-cache fail-fast; no training")


if __name__ == "__main__":
    main()
