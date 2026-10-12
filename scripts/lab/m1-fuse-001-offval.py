"""Evaluate frozen M1-FUSE-001 on official Validation; never fit or tune.

Reuses the XVEC OFFVAL implementation and its exact input/window conventions.
ECAPA uses raw encode_batch(normalize=False), per-window L2 normalization,
length-weighted pooling and final L2 normalization as recorded in FUSE.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time


def load_script(name):
    path = Path(__file__).with_name(name + '.py')
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


xv = load_script('m1-xvec-002-offval')
fuse = load_script('m1-fuse-001')
# All publications, including inherited feature writes, stage locally first.
atomic = fuse.atomic
write_json = fuse.write_json
xv.atomic = atomic
xv.write_json = write_json
EXP_ID = 'M1-FUSE-001-OFFVAL'
EC_ID = 'speechbrain/spkrec-ecapa-voxceleb'
SEED = 42


def validate_checkpoint(bundle):
    from sklearn.pipeline import Pipeline
    expected = dict(weights={'c': .5, 'h': .5}, threshold=.5, probability_of='M',
                    classes={'F': 0, 'M': 1}, windows=fuse.WINDOWS,
                    model_revisions={xv.MODEL_ID: xv.REVISION, EC_ID: fuse.EC_REVISION},
                    feature_layout={'c': ['acoustic_52d', 'xvector_512d'],
                                    'h': ['acoustic_52d', 'ecapa_192d']})
    if not isinstance(bundle, dict) or any(bundle.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen FUSE checkpoint metadata mismatch')
    heads = bundle.get('pipelines', {})
    if set(heads) != {'c', 'h'}:
        raise ValueError('Expected only frozen c/h pipelines')
    for key, dim in [('c', 564), ('h', 244)]:
        head = heads[key]
        if not isinstance(head, Pipeline) or head.n_features_in_ != dim or head.classes_.tolist() != [0, 1]:
            raise ValueError(f'Invalid frozen pipeline {key}')
    return heads


def prepare_ecapa(work, torch):
    from huggingface_hub import snapshot_download
    import torchaudio
    # SpeechBrain 1.0.3 still probes this removed torchaudio API.
    if not hasattr(torchaudio, 'list_audio_backends'):
        torchaudio.list_audio_backends = lambda: ['soundfile']
    from importlib.metadata import version
    if version('speechbrain') != '1.0.3':
        raise ValueError('Expected speechbrain==1.0.3')
    from speechbrain.inference.classifiers import EncoderClassifier
    root = Path(os.environ['DCC_RUN_ROOT']) / 'models' / 'spkrec-ecapa-voxceleb'
    marker = root / 'dcc_revision.json'
    identity = dict(repo_id=EC_ID, revision=fuse.EC_REVISION)
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError('ECAPA revision mismatch')
    else:
        snapshot_download(repo_id=EC_ID, revision=fuse.EC_REVISION, local_dir=str(root))
    # Rewrite the YAML's remote pretrained_path to the pinned local snapshot.
    # All fetches then refer to local files; HF offline also prevents fallback.
    local = work / 'ecapa_model'
    local.mkdir(parents=True, exist_ok=True)
    import re
    yaml = (root / 'hyperparams.yaml').read_text()
    yaml, count = re.subn(r'(?m)^pretrained_path:.*$',
                         'pretrained_path: ' + json.dumps(str(root)), yaml)
    if count != 1:
        raise ValueError('Unexpected ECAPA hyperparams schema')
    (local / 'hyperparams.yaml').write_text(yaml)
    for file in root.glob('*.ckpt'):
        target = local / file.name
        if not target.exists():
            shutil.copyfile(file, target)
    previous = os.environ.get('HF_HUB_OFFLINE')
    os.environ['HF_HUB_OFFLINE'] = '1'
    try:
        model = EncoderClassifier.from_hparams(source=str(local), savedir=str(local),
                    run_opts={'device': 'cuda' if torch.cuda.is_available() else 'cpu'})
    finally:
        if previous is None:
            os.environ.pop('HF_HUB_OFFLINE', None)
        else:
            os.environ['HF_HUB_OFFLINE'] = previous
    model.eval()
    write_json(marker, identity)
    return model


def cache_compatible(root, inputs):
    try:
        config = json.loads((root / 'config.json').read_text())
        prior = json.loads((root / 'inputs.json').read_text())
        keys = ['model', 'revision', 'expected_calls', 'sample_rate', 'window_seconds',
                'max_windows', 'tail_min_seconds', 'schema', 'chunk_size']
        return prior == inputs and all(config.get(k) == xv.CONFIG[k] for k in keys)
    except (OSError, ValueError):
        return False


def reuse_xvector(root, out, rows, np):
    """Validate every source chunk before publishing any, else re-extract all."""
    paths = []
    for start in range(0, len(rows), xv.CONFIG['chunk_size']):
        batch = rows[start:start + xv.CONFIG['chunk_size']]
        path = root / 'features' / f'chunk_{start:05d}.npz'
        try:
            with np.load(path, allow_pickle=False) as z:
                if (z['stems'].tolist() != [r.stem for r in batch]
                        or z['acoustic'].shape != (len(batch), 52)
                        or z['xvector'].shape != (len(batch), 512)
                        or any(z[k].shape != (len(batch),) for k in ['counts', 'durations'])
                        or any(not np.isfinite(z[k]).all() for k in ['acoustic', 'xvector', 'counts', 'durations', 'elapsed'])):
                    return False
        except (OSError, ValueError, KeyError):
            return False
        paths.append(path)
    for path in paths:
        atomic(out / 'features' / path.name, lambda p, src=path: shutil.copyfile(src, p))
    return True


def extract_ecapa(out, work, rows, model, torch, np):
    import librosa
    from dcc_data.mission1 import extract_caller_segments, concatenate_caller_waveforms
    results = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for start in range(0, len(rows), 50):
        batch = rows[start:start + 50]
        path = out / 'ecapa_features' / f'chunk_{start:05d}.npz'
        if not path.exists():
            vectors, durations, counts = [], [], []
            elapsed = 0.
            for row in batch:
                segments = extract_caller_segments(row, work)
                waveform = concatenate_caller_waveforms(segments)
                if waveform.ndim != 2 or waveform.shape[0] != 1:
                    raise ValueError('Expected mono caller waveform')
                wave, sr = waveform[0], segments[0].sample_rate
                bounds = xv.window_bounds(len(wave), sr)
                durations.append(len(wave) / sr)
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
                    with torch.inference_mode():
                        vector = model.encode_batch(torch.from_numpy(samples).unsqueeze(0).to(device), normalize=False)
                        vector = torch.nn.functional.normalize(vector.reshape(-1), dim=0)
                    embeddings.append(vector.cpu().numpy())
                    weights.append(right - left)
                mean = np.average(embeddings, axis=0, weights=weights)
                norm = np.linalg.norm(mean)
                if not np.isfinite(norm) or norm <= 0:
                    raise ValueError('Invalid ECAPA norm')
                vectors.append(mean / norm)
                if device == 'cuda':
                    torch.cuda.synchronize()
                elapsed += time.perf_counter() - began
            def writer(p):
                with p.open('wb') as stream:
                    np.savez(stream, ecapa=np.asarray(vectors), durations=durations, counts=counts,
                             stems=np.asarray([r.stem for r in batch]), elapsed=elapsed)
            atomic(path, writer)
            print(f'ECAPA chunk {start}:{start + len(batch)} complete', flush=True)
        with np.load(path, allow_pickle=False) as z:
            item = {k: z[k].copy() for k in z.files}
        if (item['ecapa'].shape != (len(batch), 192)
                or item['stems'].tolist() != [r.stem for r in batch]
                or any(item[k].shape != (len(batch),) for k in ['counts', 'durations'])
                or any(not np.isfinite(item[k]).all() for k in ['ecapa', 'durations', 'counts', 'elapsed'])):
            raise ValueError(f'Invalid ECAPA chunk: {path}')
        results.append(item)
    return {k: sum(float(c[k]) for c in results) if k == 'elapsed'
            else np.concatenate([c[k] for c in results]) for k in results[0]}


def baseline_csv(out, root, explicit):
    if explicit is not None:
        return explicit
    candidates = [p for p in sorted(out.glob('*.csv'))
                  if 'm1-xvec-002-offval' in p.name.lower()]
    candidates += [root / 'predictions.csv']
    for path in candidates:
        if path.is_file():
            try:
                xv.read_predictions(path)
                return path
            except ValueError:
                continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=os.environ.get('LAB_OUT'))
    parser.add_argument('--baseline-csv', type=Path)
    parser.add_argument('--sklearn-version')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    if args.out is None:
        parser.error('--out or LAB_OUT required')
    if os.environ.get('LAB_EXP_ID', EXP_ID) != EXP_ID:
        raise ValueError('LAB_EXP_ID mismatch')
    root, out = Path(os.environ['DCC_RUN_ROOT']), args.out
    work = Path(os.environ['LAB_WORK']) / EXP_ID.lower()
    source = root / 'lab/M1-FUSE-001/best.ckpt'
    if not source.is_file():
        raise FileNotFoundError(source)
    work.mkdir(parents=True, exist_ok=True)
    version = xv.prepare_sklearn(source, work, args.sklearn_version)
    import joblib
    heads = validate_checkpoint(joblib.load(source))
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    identity = dict(exp_id=EXP_ID, schema=1, seed=SEED, ckpt_sha256=xv.sha256(source),
                    sklearn_version=version, windows=fuse.WINDOWS,
                    revisions={xv.MODEL_ID: xv.REVISION, EC_ID: fuse.EC_REVISION})
    if (out / 'config.json').exists() and fuse.read_json(out / 'config.json') != identity:
        raise ValueError('Resume configuration mismatch')
    if all((out / n).exists() for n in ['config.json', 'metrics.json', 'best.ckpt', 'predictions.csv']):
        xv.read_predictions(out / 'predictions.csv')
        if xv.sha256(out / 'best.ckpt') != identity['ckpt_sha256']:
            raise ValueError('Completed checkpoint mismatch')
        print('Evaluation already complete')
        return
    # Both model loads fail before staging Validation ZIPs.
    processor, model = xv.prepare_model()
    ec_model = prepare_ecapa(work, torch)
    write_json(out / 'config.json', identity)
    rows, calls, archives = xv.prepare_rows(work)
    inputs = dict(archives=archives, calls=calls, stems=[r.stem for r in rows])
    if (out / 'inputs.json').exists() and fuse.read_json(out / 'inputs.json') != inputs:
        raise ValueError('Resume input mismatch')
    write_json(out / 'inputs.json', inputs)
    prior_root = root / 'lab/M1-XVEC-002-OFFVAL'
    provenance_path = out / 'xvector_source.json'
    if provenance_path.exists():
        provenance = fuse.read_json(provenance_path)
    else:
        reused = cache_compatible(prior_root, inputs) and reuse_xvector(prior_root, out, rows, np)
        provenance = dict(source='reused' if reused else 'extracted', cache=str(prior_root) if reused else None)
        write_json(provenance_path, provenance)
    features = xv.extract(out, work, rows, processor, model, torch, np)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ec = extract_ecapa(out, work, rows, ec_model, torch, np)
    if not np.allclose(features['durations'], ec['durations'], atol=1e-3, rtol=0) or not np.array_equal(features['counts'], ec['counts']):
        raise ValueError('XVEC/ECAPA window identity mismatch')
    c = heads['c'].predict_proba(np.concatenate([features['acoustic'], features['xvector']], axis=1))[:, 1]
    h = heads['h'].predict_proba(np.concatenate([features['acoustic'], ec['ecapa']], axis=1))[:, 1]
    p = (c + h) / 2
    fuse.check_probs(p)
    predicted = (p >= .5).astype(int)
    # Gender is accessed for scoring only, after frozen predictions.
    from dcc_data.mission1 import _read_gender
    from dcc_data.mission1_train import compute_metrics, encode_gender
    genders = [_read_gender(json.loads((work / r.json_path).read_text(encoding='utf-8-sig')), work / r.json_path) for r in rows]
    y = encode_gender(genders)
    import csv
    def writer(path):
        with path.open('w', encoding='utf-8', newline='') as stream:
            output = csv.writer(stream)
            output.writerow(['call_id', 'prediction', 'probability_M', 'probability_c_M', 'probability_h_M'])
            output.writerows(zip(calls, ['M' if v else 'F' for v in predicted], p, c, h))
    atomic(out / 'predictions.csv', writer)
    xv.read_predictions(out / 'predictions.csv')
    baseline = baseline_csv(out, prior_root, args.baseline_csv)
    paired, mismatch = None, None
    if baseline is not None:
        prior = xv.read_predictions(baseline)
        keys = calls if set(prior) == set(calls) else [r.stem for r in rows]
        if set(prior) != set(keys):
            raise ValueError('Baseline call identity mismatch')
        old = np.asarray([1 if prior[k] == 'M' else 0 for k in keys])
        paired = fuse.paired(y, old.astype(float), p)
        paired.update(csv_sha256=xv.sha256(baseline), csv=str(baseline))
        mismatch = int(np.sum(old != (c >= .5)))
    scores = compute_metrics(y, predicted)
    correct, duration = predicted == y, features['durations']
    buckets = {}
    for name, mask in [('le20', duration <= 20), ('gt20_le40', (duration > 20) & (duration <= 40)), ('gt40', duration > 40)]:
        buckets[name] = dict(rows=int(mask.sum()), accuracy=float(correct[mask].mean()) if mask.any() else None)
    metrics = dict(scores, caller_duration_accuracy=buckets,
        windows_per_call={str(k): int(np.sum(features['counts'] == k)) for k in range(1, 5)},
        extraction_seconds=dict(xvector=features['elapsed'], ecapa=ec['elapsed']),
        extraction_seconds_per_call=dict(xvector=features['elapsed']/3640, ecapa=ec['elapsed']/3640),
        xvector_cache=provenance, paired_mcnemar=paired,
        diagnostic_c=compute_metrics(y, (c >= .5).astype(int)),
        diagnostic_h=compute_metrics(y, (h >= .5).astype(int)),
        c_baseline_prediction_disagreements=mismatch,
        c_baseline_reproduced=None if mismatch is None else mismatch == 0,
        fitting_performed=False, ckpt_sha256=identity['ckpt_sha256'], model_revisions=identity['revisions'],
        input_sha256=hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        archive_sha256=archives, sklearn_version=version, baseline_accuracy=3572/3640,
        accuracy_gain=scores['accuracy'] - 3572/3640, fixed_success_threshold=.98434,
        fixed_success_correct_calls=3583, correct_calls=int(correct.sum()),
        success_criterion_met=int(correct.sum()) >= 3583)
    atomic(out / 'best.ckpt', lambda path: shutil.copyfile(source, path))
    write_json(out / 'metrics.json', dict(exp_id=EXP_ID,
        primary_metric=dict(name='accuracy', value=scores['accuracy'], split='official_val'),
        metrics=metrics, seed=SEED, train_rows=0, eval_rows=3640, official_val=scores,
        fitting_performed=False, ckpt='best.ckpt',
        notes='Frozen equal probability fusion; no fitting, tuning or model selection. Cached XVEC timing is original extraction time. Candidate adoption is left to the commander.'))
    print(f"Official Validation accuracy: {scores['accuracy']:.8f}", flush=True)


def smoke():
    import csv
    import zipfile
    assert xv.window_bounds(23, 1) == [(0, 20), (20, 23)]
    assert xv.window_bounds(22, 1) == [(0, 20)]
    assert xv.window_bounds(100, 1) == [(0, 20), (20, 40), (40, 60), (60, 80)]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        write_json(path / 'state.json', {'done': True})
        assert fuse.read_json(path / 'state.json') == {'done': True}
        assert not cache_compatible(path, {})
        inputs = {'calls': ['example'], 'stems': ['example'], 'archives': {}}
        write_json(path / 'config.json', xv.CONFIG)
        write_json(path / 'inputs.json', inputs)
        assert cache_compatible(path, inputs)
        assert not cache_compatible(path, dict(inputs, calls=['different']))
        write_json(path / 'config.json', dict(xv.CONFIG, revision='wrong'))
        assert not cache_compatible(path, inputs)
        archive = path / 'VL.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('VL/example.json', '{}')
        destination, members = xv.unpack(archive, path / 'work')
        assert (destination / members[0]).exists()
        assert xv.unpack(archive, path / 'work')[1] == members
        bad = path / 'bad.zip'
        with zipfile.ZipFile(bad, 'w') as z:
            z.writestr('../escape.json', '{}')
        try:
            xv.unpack(bad, path / 'work')
        except ValueError:
            pass
        else:
            raise AssertionError('Unsafe ZIP accepted')
        predictions = path / 'predictions.csv'
        with predictions.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['call_id', 'prediction', 'probability_M'])
            writer.writerows((str(i), 'F' if i % 2 else 'M', .25 if i % 2 else .75) for i in range(3640))
        assert len(xv.read_predictions(predictions)) == 3640
        assert len(xv.sha256(predictions)) == 64
    print('Smoke OK: cache identity/rejection, ZIP staging/resume/safety, atomic writes, 3640 F/M CSV rows; no training/downloads')


if __name__ == '__main__':
    main()
