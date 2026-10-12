"""M1-XVEC-002: 고정된 52D / 첫 창 / 평균 창 결합 로지스틱 회귀, Training만 사용한 80:20 분할.

미세조정은 하지 않는다. sklearn LBFGS에는 중간부터 이어 학습하는 기능(에폭·옵티마이저 상태)이 없으므로
각 헤드는 더 쪼갤 수 없는 한 단계이다. 학습 도중 끊기면 그 헤드만 처음부터 다시 학습한다.
특징 청크와 학습이 끝난 헤드는 런타임이 끊겨도 LAB_OUT에 남는다.
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

EXP_ID = "M1-XVEC-002"
MODEL_ID = "microsoft/wavlm-base-plus-sv"
# 공개된 고정(변경 불가) Hugging Face 리비전: https://huggingface.co/microsoft/wavlm-base-plus-sv/tree/1bfd64eca136543feb28c5ffaf05381c6af33121
REVISION = "1bfd64eca136543feb28c5ffaf05381c6af33121"
SEED = 42
CONFIG = dict(exp_id=EXP_ID, model=MODEL_ID, revision=REVISION, seed=SEED,
              chunk_size=2000, expected_calls=29200, sample_rate=16000,
              max_seconds=80, window_seconds=20, max_windows=4, tail_seconds=3,
              min_seconds=.5, holdout_ratio=.2, schema=2)


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
    """지정한 압축 파일 하나만 복사하고, 내용 목록은 로컬 디스크에서만 확인한다."""
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


def window_ranges(length, sample_rate=16000):
    """첫 번째 창은 항상 포함하고, 뒤쪽 창은 길이가 3초 이상일 때만 포함한다."""
    width = 20 * sample_rate
    result = [(0, min(length, width))]
    for start in range(width, min(length, 4 * width), width):
        stop = min(length, start + width)
        if stop - start >= 3 * sample_rate:
            result.append((start, stop))
    return result


def average_windows(vectors, lengths, np):
    weights = np.asarray(lengths, dtype=np.float64)
    # 길이가 0인 구간은 0으로 채운 첫 번째 임베딩 하나만 그대로 쓴다.
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    mean = np.average(vectors, axis=0, weights=weights)
    norm = np.linalg.norm(mean)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid averaged x-vector")
    return (mean / norm).astype(np.float32)


def validate_chunk(chunk, count, np):
    shapes = dict(acoustic=(count, 52), window_xvectors=(count, 4, 512),
                  window_lengths=(count, 4), durations=(count,),
                  window_counts=(count,), xvector_seconds=(count,))
    for key, shape in shapes.items():
        if chunk[key].shape != shape or not np.isfinite(chunk[key]).all():
            raise ValueError(f"Invalid chunk field: {key}")
    counts = chunk["window_counts"]
    if not np.isin(counts, [1, 2, 3, 4]).all():
        raise ValueError("Invalid window counts")
    for i, count in enumerate(counts):
        lengths = chunk["window_lengths"][i]
        vectors = chunk["window_xvectors"][i]
        if (lengths[:count] < 0).any() or (lengths[:count] > 20).any():
            raise ValueError("Invalid window lengths")
        if (lengths[1:count] < 3).any() or lengths[count:].any() or vectors[count:].any():
            raise ValueError("Invalid tail or padding")
        if not np.allclose(np.linalg.norm(vectors[:count], axis=1), 1, atol=1e-4):
            raise ValueError("Unnormalized window embeddings")


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
                validate_chunk(chunk, stop-start, np)
            continue
        acoustic, vectors, lengths, durations, counts, timings = [], [], [], [], [], []
        for row in rows[start:stop]:
            segments = extract_caller_segments(row, work)
            waveform = concatenate_caller_waveforms(segments)
            if waveform.ndim != 2 or waveform.shape[0] != 1:
                raise ValueError(f"Expected mono audio: {row.stem}")
            sr = segments[0].sample_rate
            # 신고자 전체 파형에서 52D를 다시 계산한다. 캐시를 믿고 가져다 쓰지 않는다.
            acoustic.append(acoustic_feature_vector(waveform[0], sr))
            durations.append(waveform.shape[1] / sr)
            if device == "cuda":
                torch.cuda.synchronize()
            began = time.perf_counter()
            call_vectors = np.zeros((4, 512), dtype=np.float32)
            call_lengths = np.zeros(4, dtype=np.float64)
            ranges = window_ranges(waveform.shape[1], sr)
            for index, (left, right) in enumerate(ranges):
                prefix = waveform[0, left:right]
                if sr != 16000:
                    prefix = librosa.resample(prefix, orig_sr=sr, target_sr=16000)
                prefix = np.asarray(prefix[:320000], dtype=np.float32)
                prefix = np.pad(prefix, (0, max(0, 8000-len(prefix))))
                inputs = processor(prefix, sampling_rate=16000, return_tensors="pt",
                                   return_attention_mask=True)
                with torch.inference_mode():
                    embedding = model(**{k: v.to(device) for k, v in inputs.items()}).embeddings
                    embedding = torch.nn.functional.normalize(embedding, dim=-1)
                call_vectors[index] = embedding[0].cpu().numpy()
                call_lengths[index] = (right-left) / sr
            if device == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter()-began)
            vectors.append(call_vectors)
            lengths.append(call_lengths)
            counts.append(len(ranges))
        arrays = dict(acoustic=np.asarray(acoustic), window_xvectors=np.asarray(vectors),
                      window_lengths=np.asarray(lengths), durations=np.asarray(durations),
                      window_counts=np.asarray(counts), xvector_seconds=np.asarray(timings))
        validate_chunk(arrays, stop-start, np)
        save_arrays(path, **arrays)
        print(f"Completed feature chunk {start}:{stop}", flush=True)
    result = {}
    for path in files:
        with np.load(path, allow_pickle=False) as chunk:
            for key in chunk.files:
                result.setdefault(key, []).append(chunk[key])
    return {key: np.concatenate(value) for key, value in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=os.environ.get("LAB_OUT"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    # 로컬 가중치가 없거나 맞지 않으면 데이터를 복사하거나 추출하기 전에 바로 실패시킨다.
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
    features = extract(out, work, rows, processor, model, torch, np)
    del model
    a = features["acoustic"]
    first = features["window_xvectors"][:, 0]
    mean = np.stack([average_windows(v[:n], w[:n], np) for v, w, n in zip(
        features["window_xvectors"], features["window_lengths"], features["window_counts"])])
    scores, predictions, probabilities = {}, {}, []
    for name, matrix in [("a_acoustic_52d", a),
                         ("b_first_combined_564d", np.concatenate([a, first], axis=1)),
                         ("c_mean_combined_564d", np.concatenate([a, mean], axis=1))]:
        path = out / ("best.ckpt" if name.startswith("c_") else name + ".ckpt")
        if path.exists():
            pipeline = joblib.load(path)
        else:
            pipeline = _build_lr_pipeline(SEED)
            pipeline.fit(matrix[train], y[train])
            atomic(path, lambda p: joblib.dump(pipeline, p))
        predictions[name] = pipeline.predict(matrix[holdout])
        # 이진 확률은 P(M)이고, 열은 (a), (b), (c) 순서이다.
        male = encode_gender(["M"])[0]
        column = list(pipeline.classes_).index(male)
        probabilities.append(pipeline.predict_proba(matrix[holdout])[:, column])
        scores[name] = compute_metrics(y[holdout], predictions[name])
        write_json(out / (name + "_metrics.json"), scores[name])
    seed_out = out / "seed_42"
    atomic(seed_out / "holdout_probs.npy", lambda p: save_npy(p, np.column_stack(probabilities), np))
    atomic(seed_out / "val_files.txt", lambda p: p.write_text(
        "".join(rows[i].stem + "\n" for i in holdout), encoding="utf-8"))
    write_json(seed_out / "probability_columns.json", dict(
        columns=list(scores), probability_of="M", row_identifier="manifest stem"))
    b_name, c_name = "b_first_combined_564d", "c_mean_combined_564d"
    baseline = predictions[b_name] == y[holdout]
    combined = predictions[c_name] == y[holdout]
    improved = int(np.sum(~baseline & combined))
    regressed = int(np.sum(baseline & ~combined))
    pvalue = float(binomtest(improved, improved + regressed, .5).pvalue) if improved + regressed else 1.
    save_arrays(out / "holdout_predictions.npz", indices=holdout, targets=y[holdout], **predictions)
    duration = features["durations"][holdout]
    bins = {}
    for label, mask in [("le_20s", duration <= 20),
                        ("gt_20_le_40s", (duration > 20) & (duration <= 40)),
                        ("gt_40s", duration > 40)]:
        bins[label] = dict(rows=int(mask.sum()),
                          b_accuracy=float(baseline[mask].mean()) if mask.any() else None,
                          c_accuracy=float(combined[mask].mean()) if mask.any() else None)
    gain = scores[c_name]["accuracy"] - scores[b_name]["accuracy"]
    reproduced = abs(scores[b_name]["accuracy"] - .9738) <= .002 + 1e-12
    timings = features["xvector_seconds"]
    write_json(out / "metrics.json", dict(
        exp_id=EXP_ID, primary_metric=dict(name="accuracy", value=scores[c_name]["accuracy"],
                                          split="internal_holdout"),
        metrics=dict(heads=scores, paired_mcnemar=dict(
            table_rows="b wrong/right", table_columns="c wrong/right",
            table=[[int(np.sum(~baseline & ~combined)), improved],
                   [regressed, int(np.sum(baseline & combined))]], exact_two_sided_p=pvalue),
            accuracy_gain= gain, success=bool(gain >= .004 - 1e-12 and pvalue < .05),
            baseline_reproduction=dict(reference_accuracy=.9738, tolerance=.002, passed=reproduced),
            duration_bin_accuracy=bins,
            windows_per_call={str(n): int(np.sum(features["window_counts"] == n)) for n in range(1, 5)},
            xvector_extraction_seconds_per_call=dict(mean=float(timings.mean()),
                p50=float(np.median(timings)), p95=float(np.quantile(timings, .95)),
                total=float(timings.sum()), includes="resampling, padding, processor, model, CPU transfer; excludes audio I/O and 52D")),
        seed=SEED, train_rows=len(train), eval_rows=len(holdout), official_val=None,
        ckpt="best.ckpt", notes=("Fixed (c) head; no official Validation. "
            "52D recomputed from full caller audio. LR interrupted fit restarts only that atomic head. "
            + ("Baseline reproduced." if reproduced else "Baseline reproduction FAILED."))))


def save_npy(path, value, np):
    with path.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)


def smoke():
    """외부 의존 없이 점검한다: 모델을 내려받지 않고, 학습하지 않고, 공식 데이터에도 접근하지 않는다."""
    assert window_ranges(0) == [(0, 0)]
    assert window_ranges(7999) == [(0, 7999)]
    assert len(window_ranges(23 * 16000 - 1)) == 1
    assert window_ranges(23 * 16000) == [(0, 320000), (320000, 368000)]
    assert len(window_ranges(40 * 16000)) == 2
    assert len(window_ranges(43 * 16000)) == 3
    assert window_ranges(100 * 16000)[-1] == (960000, 1280000)
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
    print("Smoke OK: window boundaries, atomic writes, archive staging/resume, path traversal guard; no training")


if __name__ == "__main__":
    main()
