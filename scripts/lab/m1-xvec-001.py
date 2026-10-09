"""M1-XVEC-001: fixed 52D/x-vector/combined LR, Training-only 80:20.

No fine-tuning. sklearn LBFGS has no resumable epoch/optimizer API: each
head is an atomic stage; interruption during fit restarts only that head.
Feature chunks and completed heads survive runtime loss in LAB_OUT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
import unicodedata
import zipfile

EXP_ID = "M1-XVEC-001"
MODEL_ID = "microsoft/wavlm-base-plus-sv"
# Public immutable HF revision: https://huggingface.co/microsoft/wavlm-base-plus-sv/tree/1bfd64eca136543feb28c5ffaf05381c6af33121
REVISION = "1bfd64eca136543feb28c5ffaf05381c6af33121"
SEED = 42
CONFIG = dict(exp_id=EXP_ID, model=MODEL_ID, revision=REVISION, seed=SEED,
              chunk_size=2000, expected_calls=29200, sample_rate=16000,
              max_seconds=20, min_seconds=.5, holdout_ratio=.2, schema=1)


def atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        writer(Path(name))
        with open(name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path, value):
    atomic(path, lambda p: p.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                                 allow_nan=False), encoding="utf-8"))


def prepare_model():
    from huggingface_hub import snapshot_download
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    root = Path(os.environ["DCC_RUN_ROOT"]) / "models" / "wavlm-base-plus-sv"
    marker = root / "dcc_revision.json"
    identity = dict(repo_id=MODEL_ID, revision=REVISION)
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError(f"Model revision mismatch: {root}")
    else:
        snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=str(root),
                          allow_patterns=["config.json", "preprocessor_config.json",
                                          "pytorch_model.bin", "model.safetensors"])
    processor = Wav2Vec2FeatureExtractor.from_pretrained(root, local_files_only=True)
    model = WavLMForXVector.from_pretrained(root, local_files_only=True)
    if model.config.xvector_output_dim != 512:
        raise ValueError("Expected a 512-dimensional x-vector")
    write_json(marker, identity)
    return processor, model


def unpack(source, work):
    """Copy one specified archive; enumerate members only on local disk."""
    local = work / source.name
    if not local.exists():
        atomic(local, lambda p: shutil.copyfile(source, p))
    destination = work / source.stem
    marker = destination / ".complete"
    with zipfile.ZipFile(local) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        for member in members:
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe zip path: {member.filename}")
        if not marker.exists():
            destination.mkdir(parents=True, exist_ok=True)
            archive.extractall(destination)
            atomic(marker, lambda p: p.write_text("complete"))
    return destination, [m.filename for m in members]


def prepare_rows(work):
    from dcc_data.colab_env import find_data_root
    from dcc_data.manifest import ManifestRow
    from dcc_data.mission1 import _read_gender
    source = find_data_root() / "Training"
    labels, label_members = unpack(source / "TL_서울_구급.zip", work)
    audio, audio_members = unpack(source / "TS_서울_구급.zip", work)
    def index(root, members, suffix):
        found = {}
        for name in members:
            path = Path(name)
            if path.suffix.lower() != suffix:
                continue
            key = unicodedata.normalize("NFC", path.stem).casefold()
            if key in found:
                raise ValueError(f"Duplicate call stem: {key}")
            found[key] = root / path
        return found
    jsons = index(labels, label_members, ".json")
    wavs = index(audio, audio_members, ".wav")
    if jsons.keys() != wavs.keys() or len(jsons) != CONFIG["expected_calls"]:
        raise ValueError(f"Expected 29,200 paired calls; JSON={len(jsons)}, WAV={len(wavs)}")
    rows, genders, calls = [], [], []
    for key in sorted(jsons):
        path = jsons[key]
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        gender = _read_gender(doc, path)
        if gender not in {"F", "M"}:
            raise ValueError(f"Invalid gender: {path}")
        calls.append(str(doc.get("call_id") or doc.get("_id") or key))
        genders.append(gender)
        rows.append(ManifestRow("Training", path.stem,
                                str(wavs[key].relative_to(work)), str(path.relative_to(work))))
    if len(set(calls)) != len(calls):
        raise ValueError("Multiple samples for the same call_id")
    return rows, genders, calls


def save_arrays(path, **arrays):
    import numpy as np
    def writer(p):
        with p.open("wb") as stream:
            np.savez(stream, **arrays)
    atomic(path, writer)


def extract(out, work, rows, processor, model, torch, np):
    import librosa
    from dcc_data.mission1 import extract_caller_segments, concatenate_caller_waveforms
    from dcc_data.mission2_features import acoustic_feature_vector
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    files = []
    for start in range(0, len(rows), CONFIG["chunk_size"]):
        stop = min(len(rows), start + CONFIG["chunk_size"])
        path = out / "features" / f"chunk_{start:05d}_{stop:05d}.npz"
        files.append(path)
        if path.exists():
            with np.load(path, allow_pickle=False) as chunk:
                if chunk["acoustic"].shape != (stop-start, 52) or chunk["xvector"].shape != (stop-start, 512):
                    raise ValueError(f"Invalid completed chunk: {path}")
            continue
        acoustic, xvectors, durations = [], [], []
        began = time.perf_counter()
        for row in rows[start:stop]:
            segments = extract_caller_segments(row, work)
            waveform = concatenate_caller_waveforms(segments)
            if waveform.ndim != 2 or waveform.shape[0] != 1:
                raise ValueError(f"Expected mono audio: {row.stem}")
            sr = segments[0].sample_rate
            acoustic.append(acoustic_feature_vector(waveform[0], sr))
            durations.append(waveform.shape[1] / sr)
            # Resample only the prefix needed for the x-vector, avoiding long-call work.
            prefix = waveform[0, :20 * sr]
            if sr != 16000:
                prefix = librosa.resample(prefix, orig_sr=sr, target_sr=16000)
            prefix = np.asarray(prefix[:320000], dtype=np.float32)
            prefix = np.pad(prefix, (0, max(0, 8000 - len(prefix))))
            inputs = processor(prefix, sampling_rate=16000, return_tensors="pt",
                               return_attention_mask=True)
            with torch.inference_mode():
                embedding = model(**{k: v.to(device) for k, v in inputs.items()}).embeddings
                embedding = torch.nn.functional.normalize(embedding, dim=-1)
            xvectors.append(embedding[0].cpu().numpy())
        if device == "cuda":
            torch.cuda.synchronize()
        a, x = np.asarray(acoustic), np.asarray(xvectors)
        if a.shape != (stop-start, 52) or x.shape != (stop-start, 512) or not np.isfinite(a).all() or not np.isfinite(x).all():
            raise ValueError("Invalid extracted features")
        save_arrays(path, acoustic=a, xvector=x, durations=np.asarray(durations),
                    elapsed=np.asarray(time.perf_counter()-began))
        print(f"Completed feature chunk {start}:{stop}", flush=True)
    chunks = [np.load(path, allow_pickle=False) for path in files]
    try:
        return (np.concatenate([c["acoustic"] for c in chunks]),
                np.concatenate([c["xvector"] for c in chunks]),
                np.concatenate([c["durations"] for c in chunks]),
                sum(float(c["elapsed"]) for c in chunks))
    finally:
        for chunk in chunks:
            chunk.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=os.environ.get("LAB_OUT"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    # Fail before data copying/extraction on missing or incompatible local weights.
    processor, model = prepare_model()
    import numpy as np
    import torch
    import joblib
    from scipy.stats import binomtest
    from sklearn.model_selection import train_test_split
    from dcc_data.mission1_train import _build_lr_pipeline, compute_metrics, encode_gender
    if args.out is None:
        parser.error("--out or LAB_OUT is required")
    if os.environ.get("LAB_EXP_ID", EXP_ID) != EXP_ID:
        raise ValueError("LAB_EXP_ID mismatch")
    out = args.out
    work = Path(os.environ["LAB_WORK"]) / EXP_ID.lower()
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text()) != CONFIG:
        raise ValueError("Output configuration mismatch")
    write_json(out / "config.json", CONFIG)
    if (out / "metrics.json").exists() and (out / "best.ckpt").exists():
        print("Experiment already complete", flush=True)
        return
    rows, genders, calls = prepare_rows(work)
    identity = dict(calls=calls, genders=genders, paths=[(r.wav_path, r.json_path) for r in rows])
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    y = encode_gender(genders)
    train, holdout = train_test_split(np.arange(len(rows)), test_size=.2,
                                     random_state=SEED, stratify=y)
    split = dict(rows_sha256=digest, train=train.tolist(), holdout=holdout.tolist())
    if (out / "split.json").exists() and json.loads((out / "split.json").read_text()) != split:
        raise ValueError("Cached rows/split mismatch")
    write_json(out / "split.json", split)
    assert not ({calls[i] for i in train} & {calls[i] for i in holdout})
    a, x, durations, elapsed = extract(out, work, rows, processor, model, torch, np)
    del model
    scores, predictions = {}, {}
    for name, features in [("acoustic_52d", a), ("xvector_512d", x),
                           ("combined_564d", np.concatenate([a, x], axis=1))]:
        path = out / ("best.ckpt" if name == "combined_564d" else name + ".ckpt")
        if path.exists():
            pipeline = joblib.load(path)
        else:
            pipeline = _build_lr_pipeline(SEED)
            pipeline.fit(features[train], y[train])
            atomic(path, lambda p: joblib.dump(pipeline, p))
        predictions[name] = pipeline.predict(features[holdout])
        scores[name] = compute_metrics(y[holdout], predictions[name])
        write_json(out / (name + "_metrics.json"), scores[name])
    baseline = predictions["acoustic_52d"] == y[holdout]
    combined = predictions["combined_564d"] == y[holdout]
    improved = int(np.sum(~baseline & combined))
    regressed = int(np.sum(baseline & ~combined))
    pvalue = float(binomtest(improved, improved + regressed, .5).pvalue) if improved + regressed else 1.
    save_arrays(out / "holdout_predictions.npz", indices=holdout, targets=y[holdout], **predictions)
    quantiles = dict(zip(["min", "p25", "median", "p75", "p95", "max"],
                         np.quantile(durations, [0, .25, .5, .75, .95, 1]).tolist()))
    gain = scores["combined_564d"]["accuracy"] - scores["acoustic_52d"]["accuracy"]
    write_json(out / "metrics.json", dict(
        exp_id=EXP_ID, primary_metric=dict(name="accuracy", value=scores["combined_564d"]["accuracy"],
                                          split="internal_holdout"),
        metrics=dict(heads=scores, paired_mcnemar=dict(exact_two_sided_p=pvalue,
                     baseline_wrong_combined_right=improved, baseline_right_combined_wrong=regressed),
                     accuracy_gain= gain, extraction_calls_per_second=len(rows)/elapsed,
                     extraction_seconds=elapsed, caller_duration_seconds=quantiles,
                     caller_duration_mean_seconds=float(durations.mean()),
                     padded_calls=int(np.sum(durations < .5)), truncated_calls=int(np.sum(durations > 20))),
        seed=SEED, train_rows=len(train), eval_rows=len(holdout), official_val=None,
        ckpt="best.ckpt", notes="Fixed combined head; no official Validation. LR interrupted fit restarts that head."))


def smoke():
    """Dependency-free, no model download, no training, no official data access."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "TL_서울_구급.zip"
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("TL_서울_구급/example.json", '{"gender":"F"}')
        extracted, members = unpack(source, root / "work")
        assert json.loads((extracted / members[0]).read_text())["gender"] == "F"
        assert unpack(source, root / "work")[1] == members
        target = root / "atomic.json"
        write_json(target, CONFIG)
        assert json.loads(target.read_text()) == CONFIG
        write_json(target, {"resumed": True})
        assert json.loads(target.read_text()) == {"resumed": True}
        unsafe = root / "unsafe.zip"
        with zipfile.ZipFile(unsafe, "w") as archive:
            archive.writestr("../escape", "bad")
        try:
            unpack(unsafe, root / "work")
        except ValueError:
            pass
        else:
            raise AssertionError("Unsafe zip accepted")
    print("Smoke OK: atomic writes, archive staging/resume, path traversal guard; no training")


if __name__ == "__main__":
    main()
