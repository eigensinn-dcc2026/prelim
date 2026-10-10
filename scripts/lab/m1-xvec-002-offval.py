"""M1-XVEC-002-OFFVAL: evaluation only of the frozen combined LR head.

No fitting or model selection. Atomic feature chunks survive runtime loss.
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

EXP_ID = "M1-XVEC-002-OFFVAL"
MODEL_ID = "microsoft/wavlm-base-plus-sv"
# Public immutable HF revision: https://huggingface.co/microsoft/wavlm-base-plus-sv/tree/1bfd64eca136543feb28c5ffaf05381c6af33121
REVISION = "1bfd64eca136543feb28c5ffaf05381c6af33121"
SEED = 42
CONFIG = dict(exp_id=EXP_ID, model=MODEL_ID, revision=REVISION, seed=SEED,
              chunk_size=50, expected_calls=3640, sample_rate=16000,
              window_seconds=20, max_windows=4, tail_min_seconds=3, schema=1)



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



def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare_sklearn(ckpt, work, requested=None):
    """Probe in a child process; import sklearn only from the isolated target."""
    import subprocess
    import sys
    probe = '''import json, joblib, warnings, sys
from sklearn.exceptions import InconsistentVersionWarning
warnings.simplefilter("error", InconsistentVersionWarning)
try:
    joblib.load(sys.argv[1])
except InconsistentVersionWarning as e:
    print(e.original_sklearn_version)
else:
    import sklearn
    print(sklearn.__version__)
'''
    result = subprocess.run([sys.executable, '-c', probe, str(ckpt)],
                            capture_output=True, text=True)
    detected = result.stdout.strip().splitlines()[-1] if result.returncode == 0 else None
    if requested and detected and requested != detected:
        raise ValueError(f'sklearn version mismatch: checkpoint={detected}, requested={requested}')
    version = detected or requested
    import re
    if not version or not re.fullmatch(r'\d+\.\d+\.\d+', version):
        raise ValueError('Cannot determine training sklearn version; supply --sklearn-version from training provenance')
    target = work / ('sklearn-' + version)
    marker = target / '.complete'
    if not marker.exists():
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
                        '--target', str(target), 'scikit-learn==' + version], check=True)
        atomic(marker, lambda p: p.write_text(version))
    sys.path.insert(0, str(target))
    import sklearn
    if sklearn.__version__ != version or not Path(sklearn.__file__).is_relative_to(target):
        raise ValueError('Isolated sklearn installation was not loaded')
    return version


def prepare_rows(work):
    from dcc_data.colab_env import find_data_root
    from dcc_data.manifest import ManifestRow
    source = find_data_root() / 'Validation'
    paths = [source / 'VL_서울_구급.zip', source / 'VS_서울_구급.zip']
    archives = {}
    indexed = []
    for source_path, suffix in zip(paths, ['.json', '.wav']):
        root, members = unpack(source_path, work)
        archives[source_path.name] = sha256(work / source_path.name)
        found = {}
        for name in members:
            path = Path(name)
            if path.suffix.lower() != suffix:
                continue
            key = unicodedata.normalize('NFC', path.stem).casefold()
            if key in found:
                raise ValueError(f'Duplicate call stem: {key}')
            found[key] = root / path
        indexed.append(found)
    labels, audio = indexed
    if labels.keys() != audio.keys() or len(labels) != 3640:
        raise ValueError(f'Expected 3640 paired Validation calls: {len(labels)}, {len(audio)}')
    rows, calls = [], []
    for key in sorted(labels):
        # Only identifiers here. Gender is read after all predictions are fixed.
        doc = json.loads(labels[key].read_text(encoding='utf-8-sig'))
        calls.append(str(doc.get('call_id') or doc.get('_id') or key))
        rows.append(ManifestRow('Validation', labels[key].stem,
                                str(audio[key].relative_to(work)), str(labels[key].relative_to(work))))
    if len(set(calls)) != 3640:
        raise ValueError('Duplicate call_id')
    return rows, calls, archives


def window_bounds(length, sr):
    bounds = []
    for start in range(0, min(length, 80 * sr), 20 * sr):
        end = min(start + 20 * sr, length)
        # Always retain the first window for short calls, matching prefix fallback.
        if start == 0 or end - start >= 3 * sr:
            bounds.append((start, end))
    if not bounds:
        raise ValueError('Empty caller waveform')
    return bounds


def extract(out, work, rows, processor, model, torch, np):
    import librosa
    from dcc_data.mission1 import extract_caller_segments, concatenate_caller_waveforms
    from dcc_data.mission2_features import acoustic_feature_vector
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    results = []
    for start in range(0, len(rows), CONFIG['chunk_size']):
        batch = rows[start:start + CONFIG['chunk_size']]
        path = out / 'features' / f'chunk_{start:05d}.npz'
        if not path.exists():
            acoustic, vectors, durations, counts = [], [], [], []
            elapsed = 0.
            for row in batch:
                segments = extract_caller_segments(row, work)
                waveform = concatenate_caller_waveforms(segments)
                if waveform.ndim != 2 or waveform.shape[0] != 1:
                    raise ValueError(f'Expected mono audio: {row.stem}')
                sr = segments[0].sample_rate
                wave = waveform[0]
                acoustic.append(acoustic_feature_vector(wave, sr))
                durations.append(len(wave) / sr)
                bounds = window_bounds(len(wave), sr)
                counts.append(len(bounds))
                if device == 'cuda':
                    torch.cuda.synchronize()
                began = time.perf_counter()
                embeddings, weights = [], []
                for left, right in bounds:
                    samples = wave[left:right]
                    if sr != 16000:
                        samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
                    samples = np.asarray(samples, dtype=np.float32)
                    samples = np.pad(samples, (0, max(0, 8000 - len(samples))))
                    inputs = processor(samples, sampling_rate=16000, return_tensors='pt',
                                       return_attention_mask=True)
                    with torch.inference_mode():
                        embedding = model(**{k: v.to(device) for k, v in inputs.items()}).embeddings
                        embedding = torch.nn.functional.normalize(embedding, dim=-1)
                    embeddings.append(embedding[0].cpu().numpy())
                    weights.append(right - left)
                mean = np.average(embeddings, axis=0, weights=weights)
                norm = np.linalg.norm(mean)
                if not np.isfinite(norm) or norm <= 0:
                    raise ValueError('Invalid x-vector norm')
                vectors.append(mean / norm)
                if device == 'cuda':
                    torch.cuda.synchronize()
                elapsed += time.perf_counter() - began
            def writer(p):
                with p.open('wb') as stream:
                    np.savez(stream, acoustic=np.asarray(acoustic), xvector=np.asarray(vectors),
                             durations=np.asarray(durations), counts=np.asarray(counts),
                             stems=np.asarray([r.stem for r in batch]), elapsed=np.asarray(elapsed))
            atomic(path, writer)
            print(f'Completed Validation feature chunk {start}:{start + len(batch)}', flush=True)
        with np.load(path, allow_pickle=False) as chunk:
            item = {k: chunk[k].copy() for k in chunk.files}
        if (item['acoustic'].shape != (len(batch), 52)
                or item['xvector'].shape != (len(batch), 512)
                or item['stems'].tolist() != [r.stem for r in batch]
                or not np.isfinite(item['acoustic']).all()
                or not np.isfinite(item['xvector']).all()):
            raise ValueError(f'Invalid cached chunk: {path}')
        results.append(item)
    return {k: (sum(float(c[k]) for c in results) if k == 'elapsed'
                else np.concatenate([c[k] for c in results])) for k in results[0]}


def read_predictions(path):
    import csv
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 3640:
        raise ValueError(f'Expected 3640 CSV rows: {path}')
    result = {}
    for row in rows:
        key = row.get('call_id') or row.get('stem') or row.get('filename')
        value = row.get('prediction') or row.get('gender') or row.get('pred_gender')
        if not key or value not in {'F', 'M'} or key in result:
            raise ValueError(f'Invalid/duplicate prediction CSV row: {path}')
        result[key] = value
    return result


def smoke():
    assert window_bounds(1, 1) == [(0, 1)]
    assert window_bounds(22, 1) == [(0, 20)]
    assert window_bounds(23, 1) == [(0, 20), (20, 23)]
    assert window_bounds(100, 1) == [(0, 20), (20, 40), (40, 60), (60, 80)]
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        source = root / 'VL.zip'
        with zipfile.ZipFile(source, 'w') as archive:
            archive.writestr('VL/example.json', '{}')
        destination, members = unpack(source, root / 'work')
        assert (destination / members[0]).exists()
        assert unpack(source, root / 'work')[1] == members
        write_json(root / 'metrics.json', {'fitting_performed': False})
        assert json.loads((root / 'metrics.json').read_text())['fitting_performed'] is False
    print('Smoke OK: window boundaries, atomic writes, ZIP staging/resume; no training')


def main():
    import csv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=os.environ.get('LAB_OUT'))
    parser.add_argument('--baseline-csv', type=Path)
    parser.add_argument('--sklearn-version', help='Training version, only needed if checkpoint probe fails')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    if args.out is None:
        parser.error('--out or LAB_OUT is required')
    if os.environ.get('LAB_EXP_ID', EXP_ID) != EXP_ID:
        raise ValueError('LAB_EXP_ID mismatch')
    out = args.out
    work = Path(os.environ['LAB_WORK']) / EXP_ID.lower()
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    source_ckpt = Path(os.environ['DCC_RUN_ROOT']) / 'lab' / 'M1-XVEC-002' / 'best.ckpt'
    if not source_ckpt.is_file():
        raise FileNotFoundError(source_ckpt)
    # Version and weights fail fast, before any Validation archive copying.
    version = prepare_sklearn(source_ckpt, work, args.sklearn_version)
    import joblib
    import numpy as np
    import torch
    from sklearn.pipeline import Pipeline
    from dcc_data.mission1_train import compute_metrics, encode_gender
    from scipy.stats import binomtest
    pipeline = joblib.load(source_ckpt)
    if not isinstance(pipeline, Pipeline) or pipeline.n_features_in_ != 564:
        raise ValueError('Expected the frozen 564D combined sklearn Pipeline')
    if set(pipeline.classes_.tolist()) != {0, 1}:
        raise ValueError('Expected trained class encoding F=0, M=1')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    processor, model = prepare_model()
    identity = dict(CONFIG, ckpt_sha256=sha256(source_ckpt), sklearn_version=version)
    if (out / 'config.json').exists() and json.loads((out / 'config.json').read_text()) != identity:
        raise ValueError('Output configuration/checkpoint mismatch')
    write_json(out / 'config.json', identity)
    if (out / 'metrics.json').exists() and (out / 'best.ckpt').exists() and (out / 'predictions.csv').exists():
        read_predictions(out / 'predictions.csv')
        if sha256(out / 'best.ckpt') != identity['ckpt_sha256']:
            raise ValueError('Completed checkpoint mismatch')
        print('Evaluation already complete', flush=True)
        return
    rows, calls, archives = prepare_rows(work)
    inputs = dict(archives=archives, calls=calls, stems=[r.stem for r in rows])
    input_digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    if (out / 'inputs.json').exists() and json.loads((out / 'inputs.json').read_text()) != inputs:
        raise ValueError('Cached input identity mismatch')
    write_json(out / 'inputs.json', inputs)
    features = extract(out, work, rows, processor, model, torch, np)
    predicted = pipeline.predict(np.concatenate([features['acoustic'], features['xvector']], axis=1))
    genders = []
    from dcc_data.mission1 import _read_gender
    for row in rows:
        path = work / row.json_path
        genders.append(_read_gender(json.loads(path.read_text(encoding='utf-8-sig')), path))
    y = encode_gender(genders)
    labels = ['F' if value == 0 else 'M' for value in predicted]
    def write_csv(path):
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['call_id', 'stem', 'prediction', 'target'])
            writer.writerows(zip(calls, [r.stem for r in rows], labels, genders))
    atomic(out / 'predictions.csv', write_csv)
    read_predictions(out / 'predictions.csv')
    scores = compute_metrics(y, predicted)
    correct = predicted == y
    duration = features['durations']
    buckets = {}
    for name, mask in [('le20', duration <= 20), ('gt20_le40', (duration > 20) & (duration <= 40)),
                       ('gt40', duration > 40)]:
        buckets[name] = dict(rows=int(mask.sum()), accuracy=float(correct[mask].mean()) if mask.any() else None)
    baseline = args.baseline_csv
    if baseline is None:
        candidates = sorted(out.glob('*M1-XVEC-001-OFFVAL*.csv')) + sorted(out.glob('*m1-xvec-001-offval*.csv'))
        if len(candidates) > 1:
            raise ValueError('Multiple baseline CSVs; specify --baseline-csv')
        if not candidates:
            # A runner may place the prior output here under its generic CSV name.
            for candidate in sorted(out.glob('*.csv')):
                if candidate.name == 'predictions.csv':
                    continue
                try:
                    read_predictions(candidate)
                except (ValueError, KeyError):
                    continue
                candidates.append(candidate)
            if len(candidates) > 1:
                raise ValueError('Multiple compatible baseline CSVs; specify --baseline-csv')
        baseline = candidates[0] if candidates else None
    paired = None
    if baseline is not None:
        prior = read_predictions(baseline)
        keys = calls if set(prior) == set(calls) else [r.stem for r in rows]
        if set(prior) != set(keys):
            raise ValueError('Baseline CSV call set differs from Validation')
        old_correct = np.asarray([prior[key] == gender for key, gender in zip(keys, genders)])
        table = [[int(np.sum(old_correct & correct)), int(np.sum(old_correct & ~correct))],
                 [int(np.sum(~old_correct & correct)), int(np.sum(~old_correct & ~correct))]]
        discordant = table[0][1] + table[1][0]
        paired = dict(table=table, axes=['baseline correct/wrong', 'current correct/wrong'],
                      exact_two_sided_p=float(binomtest(table[0][1], discordant, .5).pvalue) if discordant else 1.,
                      csv_sha256=sha256(baseline))
    atomic(out / 'best.ckpt', lambda p: shutil.copyfile(source_ckpt, p))
    metrics = dict(scores, caller_duration_accuracy=buckets,
                   windows_per_call={str(k): int(np.sum(features['counts'] == k)) for k in range(1, 5)},
                   xvector_extraction_seconds=features['elapsed'], paired_mcnemar=paired,
                   fitting_performed=False, ckpt_sha256=identity['ckpt_sha256'],
                   model_revision=REVISION, input_sha256=input_digest, archive_sha256=archives,
                   sklearn_version=version, baseline_accuracy=0.9747252747252747,
                   accuracy_gain=scores['accuracy'] - 0.9747252747252747,
                   fixed_success_threshold=0.9777, success_criterion_met=scores['accuracy'] >= 0.9777)
    write_json(out / 'metrics.json', dict(exp_id=EXP_ID,
               primary_metric=dict(name='accuracy', value=scores['accuracy'], split='official_val'),
               metrics=metrics, seed=SEED, train_rows=0, eval_rows=3640, official_val=scores,
               fitting_performed=False, ckpt='best.ckpt',
               notes='Frozen M1-XVEC-002 (c) evaluation only; no fitting or tuning.'))
    print(f"Official Validation accuracy: {scores['accuracy']:.8f}", flush=True)


if __name__ == '__main__':
    main()
