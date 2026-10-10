"""M1-ECAPA-001: Training-only fixed LR comparison; no fine-tuning.

Feature chunks and fitted heads are atomic resumable stages. sklearn LBFGS
exposes no epoch/optimizer checkpoint: an interrupted fit restarts that head,
while completed heads are retained. No torch training or epoch loop is used.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import time
import zipfile

# Keep the baseline's row ordering, zip staging, acoustics, x-vector and LR exact.
_spec = importlib.util.spec_from_file_location("m1_xvec_baseline", Path(__file__).with_name("m1-xvec-001.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
atomic, write_json, save_arrays = base.atomic, base.write_json, base.save_arrays
EXP_ID = "M1-ECAPA-001"
MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
# https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/commit/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286
REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
SEED = 42
CONFIG = dict(exp_id=EXP_ID, seed=SEED, model=MODEL_ID, revision=REVISION,
              baseline=base.CONFIG, speechbrain="1.0.3", chunk_size=2000,
              primary_head="d", probability_class="M", schema=1)
HEADS = ["a", "b", "c", "d"]


def prepare_model(torch):
    from huggingface_hub import snapshot_download
    from speechbrain.inference.classifiers import EncoderClassifier
    root = Path(os.environ["DCC_RUN_ROOT"]) / "models" / "spkrec-ecapa-voxceleb"
    marker = root / "dcc_revision.json"
    identity = dict(repo_id=MODEL_ID, revision=REVISION)
    required = ["hyperparams.yaml", "embedding_model.ckpt", "mean_var_norm_emb.ckpt",
                "classifier.ckpt", "label_encoder.txt"]
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError(f"ECAPA revision mismatch: {root}")
    if not marker.exists() or not all((root / name).is_file() for name in required):
        snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=str(root),
                          allow_patterns=required)
    if not all((root / name).is_file() for name in required):
        raise ValueError("Incomplete ECAPA snapshot")
    # Upstream YAML has explicit HF paths: override them before collection.
    # Optional custom.py lookup also uses this local directory; every loadable is local.
    model = EncoderClassifier.from_hparams(
        source=str(root), savedir=str(root),
        overrides={"pretrained_path": str(root)},
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"})
    model.eval()
    with torch.inference_mode():
        check = model.encode_batch(torch.zeros(1, 8000, device=model.device))
    if check.numel() != 192 or not torch.isfinite(check).all():
        raise ValueError("Expected finite 192D ECAPA output")
    write_json(marker, identity)
    return model


def caller_durations(rows, work, np):
    """Recompute exact crop lengths from local WAV headers and allowed intervals."""
    from dcc_data.audio import read_wav_info
    from dcc_data.mission1 import _parse_caller_intervals
    from dcc_data.mission2 import milliseconds_to_frame
    result = []
    for row in rows:
        path = work / row.json_path
        intervals = _parse_caller_intervals(json.loads(path.read_text(encoding="utf-8")), path)
        info = read_wav_info(work / row.wav_path)
        if info.channels != 1:
            raise ValueError(f"Expected mono: {row.stem}")
        frames = 0
        for start, end in intervals:
            first = milliseconds_to_frame(start, info.sample_rate)
            last = milliseconds_to_frame(end, info.sample_rate)
            if not 0 <= first < last <= info.frame_count:
                raise ValueError(f"Invalid caller crop: {row.stem}")
            frames += last - first
        result.append(frames / info.sample_rate)
    return np.asarray(result)


def read_baseline(root, split, durations, np):
    """All-or-nothing reuse, with full row identity, split and duration checks."""
    try:
        if json.loads((root / "config.json").read_text()) != base.CONFIG:
            return None
        if json.loads((root / "split.json").read_text()) != split:
            return None
        acoustics, vectors = [], []
        expected = []
        for start in range(0, len(durations), 2000):
            stop = min(len(durations), start + 2000)
            path = root / "features" / f"chunk_{start:05d}_{stop:05d}.npz"
            expected.append(path.name)
            with np.load(path, allow_pickle=False) as chunk:
                a, x, d = chunk["acoustic"], chunk["xvector"], chunk["durations"]
                if (a.shape != (stop-start, 52) or x.shape != (stop-start, 512)
                        or d.shape != (stop-start,) or not np.isfinite(a).all()
                        or not np.isfinite(x).all() or not np.isfinite(d).all()
                        or not np.allclose(d, durations[start:stop], atol=1e-3, rtol=0)):
                    return None
                acoustics.append(a.copy())
                vectors.append(x.copy())
        if sorted(p.name for p in (root / "features").glob("chunk_*.npz")) != sorted(expected):
            return None
        return np.concatenate(acoustics), np.concatenate(vectors)
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def baseline_features(out, work, rows, split, durations, torch, np):
    local = out / "baseline"
    cached = read_baseline(local, split, durations, np)
    if cached is not None:
        return *cached, json.loads((local / "provenance.json").read_text())["source"]
    source = Path(os.environ["DCC_RUN_ROOT"]) / "lab" / base.EXP_ID
    cached = read_baseline(source, split, durations, np)
    if cached is not None:
        # Persist validated arrays here, so the experiment no longer depends on baseline artifacts.
        for start in range(0, len(rows), 2000):
            stop = min(len(rows), start + 2000)
            save_arrays(local / "features" / f"chunk_{start:05d}_{stop:05d}.npz",
                        acoustic=cached[0][start:stop], xvector=cached[1][start:stop],
                        durations=durations[start:stop], elapsed=np.asarray(0.))
        provenance = "M1-XVEC-001"
    else:
        print("Baseline reuse rejected/unavailable; recomputing with M1-XVEC-001", flush=True)
        # Retain only valid partial chunks when resuming a baseline recomputation.
        for start in range(0, len(rows), 2000):
            stop = min(len(rows), start + 2000)
            path = local / "features" / f"chunk_{start:05d}_{stop:05d}.npz"
            if not path.exists():
                continue
            try:
                with np.load(path, allow_pickle=False) as chunk:
                    valid = (chunk["acoustic"].shape == (stop-start, 52)
                             and chunk["xvector"].shape == (stop-start, 512)
                             and chunk["durations"].shape == (stop-start,)
                             and np.isfinite(chunk["acoustic"]).all()
                             and np.isfinite(chunk["xvector"]).all()
                             and np.allclose(chunk["durations"], durations[start:stop], atol=1e-3, rtol=0)
                             and np.isfinite(chunk["elapsed"]).all())
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                valid = False
            if not valid:
                path.unlink()
        processor, model = base.prepare_model()
        a, x, d, _ = base.extract(local, work, rows, processor, model, torch, np)
        if not np.allclose(d, durations, atol=1e-3, rtol=0):
            raise ValueError("Recomputed baseline duration mismatch")
        cached = a, x
        provenance = "recomputed"
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(local / "provenance.json", dict(source=provenance))
    write_json(local / "config.json", base.CONFIG)
    write_json(local / "split.json", split)
    return *cached, provenance


def extract_ecapa(out, work, rows, durations, model, torch, np):
    import librosa
    from dcc_data.mission1 import extract_caller_segments, concatenate_caller_waveforms
    arrays, timings, total = [], [], 0.
    for start in range(0, len(rows), 2000):
        stop = min(len(rows), start + 2000)
        path = out / "features" / f"chunk_{start:05d}_{stop:05d}.npz"
        if not path.exists():
            vectors, seconds = [], []
            began = time.perf_counter()
            for i in range(start, stop):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                tick = time.perf_counter()
                segments = extract_caller_segments(rows[i], work)
                waveform = concatenate_caller_waveforms(segments)
                sr = segments[0].sample_rate
                if waveform.shape[0] != 1 or abs(waveform.shape[1]/sr - durations[i]) > 1e-3:
                    raise ValueError(f"Caller waveform mismatch: {rows[i].stem}")
                prefix = waveform[0, :20*sr]
                if sr != 16000:
                    prefix = librosa.resample(prefix, orig_sr=sr, target_sr=16000)
                prefix = np.asarray(prefix[:320000], dtype=np.float32)
                prefix = np.pad(prefix, (0, max(0, 8000-len(prefix))))
                with torch.inference_mode():
                    embedding = model.encode_batch(torch.from_numpy(prefix).unsqueeze(0).to(model.device))
                    embedding = torch.nn.functional.normalize(embedding.reshape(-1), dim=0)
                vectors.append(embedding.cpu().numpy())
                seconds.append(time.perf_counter()-tick)
            save_arrays(path, ecapa=np.asarray(vectors), durations=durations[start:stop],
                        seconds=np.asarray(seconds), elapsed=np.asarray(time.perf_counter()-began))
            print(f"Completed ECAPA chunk {start}:{stop}", flush=True)
        with np.load(path, allow_pickle=False) as chunk:
            e, d, t = chunk["ecapa"], chunk["durations"], chunk["seconds"]
            if (e.shape != (stop-start, 192) or d.shape != (stop-start,)
                    or t.shape != (stop-start,) or not np.isfinite(e).all()
                    or not np.isfinite(t).all() or np.any(t < 0)
                    or not np.allclose(d, durations[start:stop], atol=1e-3, rtol=0)
                    or not np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-4)):
                raise ValueError(f"Invalid completed ECAPA chunk: {path}")
            arrays.append(e.copy())
            timings.append(t.copy())
            total += float(chunk["elapsed"])
    return np.concatenate(arrays), np.concatenate(timings), total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=os.environ.get("LAB_OUT"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Fail on model download/load before touching the dataset.
    model = prepare_model(torch)
    import joblib
    from scipy.stats import binomtest
    from sklearn.model_selection import train_test_split
    from dcc_data.mission1_train import _build_lr_pipeline, compute_metrics, encode_gender
    if args.out is None:
        parser.error("--out or LAB_OUT required")
    if os.environ.get("LAB_EXP_ID", EXP_ID) != EXP_ID:
        raise ValueError("LAB_EXP_ID mismatch")
    out = args.out
    work = Path(os.environ["LAB_WORK"]) / EXP_ID.lower()
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text()) != CONFIG:
        raise ValueError("Output configuration mismatch")
    write_json(out / "config.json", CONFIG)
    if (out / "metrics.json").exists() and (out / "best.ckpt").exists():
        print("Experiment already complete", flush=True)
        return
    rows, genders, calls = base.prepare_rows(work)
    identity = dict(calls=calls, genders=genders, paths=[(r.wav_path, r.json_path) for r in rows])
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    y = encode_gender(genders)
    train, holdout = train_test_split(np.arange(len(rows)), test_size=.2, random_state=SEED, stratify=y)
    split = dict(rows_sha256=digest, train=train.tolist(), holdout=holdout.tolist())
    if (out / "split.json").exists() and json.loads((out / "split.json").read_text()) != split:
        raise ValueError("Cached manifest/split mismatch")
    write_json(out / "split.json", split)
    assert len(holdout) == 5840 and not ({calls[i] for i in train} & {calls[i] for i in holdout})
    durations = caller_durations(rows, work, np)
    e, timing, elapsed = extract_ecapa(out, work, rows, durations, model, torch, np)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    a, x, provenance = baseline_features(out, work, rows, split, durations, torch, np)
    scores, predictions, probabilities = {}, {}, []
    seed_out = out / "seed_42"
    seed_out.mkdir(exist_ok=True)
    features_by_head = [("a", a), ("b", e), ("c", np.concatenate([a, x], axis=1)),
                        ("d", np.concatenate([a, x, e], axis=1)), ("xvector_diagnostic", x)]
    for name, features in features_by_head:
        path = out / ("best.ckpt" if name == "d" else name + ".ckpt")
        if path.exists():
            pipeline = joblib.load(path)
        else:
            pipeline = _build_lr_pipeline(SEED)
            pipeline.fit(features[train], y[train])
            atomic(path, lambda p: joblib.dump(pipeline, p))
        predictions[name] = pipeline.predict(features[holdout])
        scores[name] = compute_metrics(y[holdout], predictions[name])
        if name in HEADS:
            probabilities.append(pipeline.predict_proba(features[holdout])[:, list(pipeline.classes_).index(1)])
        print(f"Completed LR head {name}", flush=True)
    def save_probs(p):
        with p.open("wb") as stream:
            np.save(stream, np.column_stack(probabilities), allow_pickle=False)
    atomic(seed_out / "holdout_probs.npy", save_probs)
    atomic(seed_out / "val_files.txt", lambda p: p.write_text(
        "".join(rows[i].stem + "\n" for i in holdout), encoding="utf-8"))
    write_json(seed_out / "probability_columns.json", dict(heads=HEADS, class_label="M", class_index=1))
    save_arrays(seed_out / "holdout_predictions.npz", indices=holdout, targets=y[holdout], **predictions)
    c, d = predictions["c"] == y[holdout], predictions["d"] == y[holdout]
    both_right = int(np.sum(c & d))
    regressed, improved = int(np.sum(c & ~d)), int(np.sum(~c & d))
    both_wrong = int(np.sum(~c & ~d))
    pvalue = float(binomtest(improved, improved+regressed, .5).pvalue) if improved+regressed else 1.
    xwrong, ewrong = predictions["xvector_diagnostic"] != y[holdout], predictions["b"] != y[holdout]
    bins = {}
    dh = durations[holdout]
    for name, mask in [("le_20s", dh <= 20), ("gt_20_le_40s", (dh > 20) & (dh <= 40)), ("gt_40s", dh > 40)]:
        bins[name] = dict(rows=int(mask.sum()), c_accuracy=float(c[mask].mean()) if mask.any() else None,
                         d_accuracy=float(d[mask].mean()) if mask.any() else None)
    gain = scores["d"]["accuracy"] - scores["c"]["accuracy"]
    reproduced = abs(scores["c"]["accuracy"] - .9738) <= .002 + 1e-12
    write_json(out / "metrics.json", dict(
        exp_id=EXP_ID, primary_metric=dict(name="accuracy", value=scores["d"]["accuracy"], split="internal_holdout"),
        metrics=dict(heads=scores, head_features=dict(a="52D", b="ECAPA 192D", c="52D+x-vector", d="52D+x-vector+ECAPA"),
                     paired_mcnemar=dict(table=[[both_right, regressed], [improved, both_wrong]],
                                         rows="c correct/incorrect", columns="d correct/incorrect", exact_two_sided_p=pvalue),
                     embedding_error_overlap=dict(both_wrong=int(np.sum(xwrong & ewrong)),
                         xvector_only_wrong=int(np.sum(xwrong & ~ewrong)), ecapa_only_wrong=int(np.sum(~xwrong & ewrong)),
                         both_correct=int(np.sum(~xwrong & ~ewrong))),
                     caller_duration_bins=bins, accuracy_gain= gain, baseline_reference=.9738,
                     baseline_reproduced=reproduced, reproduction_failed=not reproduced,
                     success_criteria_met=bool(reproduced and gain >= .004 and pvalue < .05),
                     ecapa_seconds_per_call=float(timing.mean()), ecapa_seconds_p50=float(np.median(timing)),
                     ecapa_seconds_p95=float(np.quantile(timing, .95)), ecapa_extraction_seconds=elapsed,
                     timing_scope="local waveform read, concatenation, resampling, ECAPA encode and CPU transfer; excludes initial model load",
                     baseline_feature_source=provenance),
        seed=SEED, train_rows=len(train), eval_rows=len(holdout), official_val=None, ckpt="best.ckpt",
        notes="Pre-fixed head d; Training only. LR interrupted fit restarts only that head; no epoch/optimizer API. "
              + ("Baseline reproduced." if reproduced else "M1-XVEC-001 reproduction failed (>0.2 percentage points).")))


def smoke():
    """No downloads, data access or fitting; exercise atomic persistence."""
    base.smoke()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_json(root / "config.json", CONFIG)
        assert json.loads((root / "config.json").read_text()) == CONFIG
        assert len(REVISION) == 40 and HEADS == ["a", "b", "c", "d"]
    print("ECAPA smoke OK: fixed model identity and head order; no training")


if __name__ == "__main__":
    main()
