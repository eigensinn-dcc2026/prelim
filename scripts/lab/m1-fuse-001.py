"""M1-FUSE-001: fixed score fusion of cached Training-only speaker features.

No audio, Validation, downloads, or neural training. A nonincremental sklearn
LR fit interrupted mid-stage restarts that stage; completed stages are reused.
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

EXP_ID = "M1-FUSE-001"
SEED = 42
CV_SEED = 2026
XV_REVISION = "1bfd64eca136543feb28c5ffaf05381c6af33121"
EC_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
WINDOWS = dict(window_seconds=20, max_windows=4, tail_min_seconds=3, pad_min_seconds=.5,
               pooling="length-weighted mean of L2-normalized windows, then L2 normalization")

def atomic(path, writer):
    """Serialize locally, then publish through a size-verified Drive temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    local_root = Path(os.environ.get('LAB_WORK', tempfile.gettempdir()))
    local_root.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=local_root, suffix='.tmp')
    os.close(fd)
    local = Path(name)
    target = path.with_name(path.name + '.tmp')
    try:
        writer(local)
        for attempt in range(6):
            try:
                shutil.copyfile(local, target)
                if target.stat().st_size != local.stat().st_size:
                    raise OSError('Incomplete artifact copy')
                os.replace(target, path)
                break
            except OSError:
                if attempt == 5:
                    raise
                time.sleep(2 ** (attempt + 1))
    finally:
        local.unlink(missing_ok=True)

def write_json(path, value):
    atomic(path, lambda p: p.write_text(json.dumps(value, ensure_ascii=False,
                       indent=2, allow_nan=False), encoding='utf-8'))

def save_npy(path, array):
    import numpy as np
    def writer(p):
        with p.open('wb') as stream:
            np.save(stream, array, allow_pickle=False)
    atomic(path, writer)

def average_windows(vectors, lengths):
    import numpy as np
    normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)
    result = (normalized * lengths[..., None]).sum(axis=1)
    # A zero-length call still has one padded first window.
    empty = lengths.sum(axis=1) == 0
    result[empty] = normalized[empty, 0]
    return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)

def load_baseline(root):
    """Load legacy x-vector chunks; split rows_sha256 verifies manifest order."""
    import numpy as np
    split = json.loads((root / 'split.json').read_text())
    chunks = []
    offset = 0
    for path in sorted((root / 'features').glob('chunk_*.npz')):
        with np.load(path, allow_pickle=False) as z:
            required = ['acoustic', 'window_xvectors', 'window_lengths', 'durations', 'window_counts']
            if not all(k in z for k in required):
                raise ValueError(f'Cache requires {required}: {path}')
            item = {k: z[k].copy() for k in required}
        n = len(item['acoustic'])
        if n == 0:
            raise ValueError('Empty baseline chunk')
        if path.name != f'chunk_{offset:05d}_{offset+n:05d}.npz':
            raise ValueError('Non-contiguous baseline chunks')
        if item['acoustic'].shape != (n, 52) or item['window_xvectors'].shape != (n, 4, 512) or item['window_lengths'].shape != (n, 4):
            raise ValueError(f'Invalid baseline shape: {path}')
        if item['durations'].shape != (n,) or item['window_counts'].shape != (n,):
            raise ValueError('Invalid baseline metadata shape')
        for key in required:
            if not np.isfinite(item[key]).all():
                raise ValueError(f'Nonfinite baseline {key}')
        lengths = item['window_lengths']
        if (lengths < 0).any() or not np.array_equal(np.maximum(1, (lengths > 0).sum(1)), item['window_counts']):
            raise ValueError('Invalid baseline window weights/counts')
        chunks.append(item)
        offset += n
    if offset != 29200:
        raise ValueError(f'Expected 29200 cached rows, got {offset}')
    data = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    train, val = np.asarray(split['train']), np.asarray(split['holdout'])
    if (train.shape != (23360,) or val.shape != (5840,)
            or not np.array_equal(np.sort(np.concatenate([train, val])), np.arange(29200))
            or 'rows_sha256' not in split):
        raise ValueError('Invalid cached row identity/split')
    return data, split


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest_array(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def load_inputs(root):
    """Cross-check legacy anonymous rows against extraction-time provenance."""
    import numpy as np
    from dcc_data.mission1_train import encode_gender
    from sklearn.model_selection import StratifiedKFold, train_test_split
    xv, ec = root / 'lab/M1-XVEC-002', root / 'lab/M1-ECAPA-002'
    manifest = read_json(ec / 'manifest.json')
    config = read_json(ec / 'config.json')
    calls, stems, genders = (manifest[k] for k in ('calls', 'stems', 'genders'))
    if (any(len(v) != 29200 for v in (calls, stems, genders))
            or len(set(calls)) != 29200 or len(set(stems)) != 29200
            or any(not isinstance(v, str) or not v or '\n' in v or '\r' in v for v in calls)):
        raise ValueError('Expected 29200 unique manifest calls/stems')
    if config['revision'] != EC_REVISION or read_json(xv / 'config.json')['revision'] != XV_REVISION:
        raise ValueError('Source model revision mismatch')
    for key in ('window_seconds', 'max_windows', 'tail_min_seconds'):
        if config[key] != WINDOWS[key] or read_json(xv / 'config.json')[{'tail_min_seconds': 'tail_seconds'}.get(key, key)] != WINDOWS[key]:
            raise ValueError('Window configuration mismatch')
    cached, split = load_baseline(xv)
    if (config['baseline_split'] != split
            or config['stems_sha256'] != hashlib.sha256(json.dumps(stems).encode()).hexdigest()
            or config['baseline_arrays_sha256'] != {k: digest_array(v) for k, v in cached.items()}):
        raise ValueError('Baseline/ECAPA manifest order or feature identity mismatch')
    chunks, offset = [], 0
    for path in sorted((ec / 'features').glob('chunk_*.npz')):
        with np.load(path, allow_pickle=False) as z:
            item = {k: z[k].copy() for k in ('window_ecapa', 'window_lengths', 'durations', 'window_counts', 'stems')}
        n = len(item['stems'])
        if (not n or path.name != f'chunk_{offset:05d}_{offset+n:05d}.npz'
                or item['window_ecapa'].shape != (n, 4, 192)
                or item['window_lengths'].shape != (n, 4)
                or item['durations'].shape != (n,) or item['window_counts'].shape != (n,)
                or item['stems'].tolist() != stems[offset:offset+n]
                or any(not np.isfinite(item[k]).all() for k in item if k != 'stems')
                or not np.allclose(item['window_lengths'], cached['window_lengths'][offset:offset+n], atol=1e-3, rtol=0)
                or not np.allclose(item['durations'], cached['durations'][offset:offset+n], atol=1e-3, rtol=0)
                or not np.array_equal(item['window_counts'], cached['window_counts'][offset:offset+n])):
            raise ValueError(f'Invalid ECAPA chunk: {path}')
        chunks.append(item['window_ecapa'])
        offset += n
    if offset != 29200:
        raise ValueError(f'Expected 29200 ECAPA rows, got {offset}')
    y = encode_gender(genders)
    train, holdout = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=42)
    if train.tolist() != split['train'] or holdout.tolist() != split['holdout']:
        raise ValueError('Manifest labels/order do not match cached Training split')
    if (cached['durations'] < 0).any():
        raise ValueError('Negative caller duration')
    folds = np.empty(len(y), dtype=int)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=CV_SEED).split(y, y))
    for fold, (_, va) in enumerate(splits):
        folds[va] = fold
    if (ec / 'oof_folds.npy').exists() and not np.array_equal(np.load(ec / 'oof_folds.npy', allow_pickle=False), folds):
        raise ValueError('Stored CV fold assignment mismatch')
    saved = None
    if (ec / 'oof_probs.npy').exists():
        columns = read_json(ec / 'oof_probs_columns.json')
        saved = np.load(ec / 'oof_probs.npy', allow_pickle=False)
        if columns != ['c', 'h', 'i', 'j'] or saved.shape != (29200, 4):
            raise ValueError('Stored probability schema mismatch')
        check_probs(saved)
    x = average_windows(cached['window_xvectors'], cached['window_lengths'])
    e = average_windows(np.concatenate(chunks), cached['window_lengths'])
    features = dict(c=np.concatenate([cached['acoustic'], x], axis=1),
                    h=np.concatenate([cached['acoustic'], e], axis=1), i=e)
    identity = dict(schema=1, exp_id=EXP_ID, seed=SEED, cv_seed=CV_SEED,
                    manifest=manifest, feature_hashes={k: digest_array(v) for k, v in features.items()},
                    duration_hash=digest_array(cached['durations']),
                    stored_probs_hash=digest_array(saved) if saved is not None else None,
                    windows=WINDOWS, weights=[.5, .5], threshold=.5,
                    revisions=[XV_REVISION, EC_REVISION])
    return features, y, calls, cached['durations'], folds, splits, saved, identity


def check_probs(probs):
    import numpy as np
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError('Invalid probabilities')


def fuse(c, h):
    import numpy as np
    def logit(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return np.log(p) - np.log1p(-p)
    return (c + h) / 2, 1 / (1 + np.exp(-(logit(c) + logit(h)) / 2))


def paired(y, a, b):
    import numpy as np
    from scipy.stats import binomtest
    ac, bc = (a >= .5) == y, (b >= .5) == y
    table = [[int(np.sum(ac & bc)), int(np.sum(ac & ~bc))],
             [int(np.sum(~ac & bc)), int(np.sum(~ac & ~bc))]]
    n = table[0][1] + table[1][0]
    return dict(table=table, axes=['first correct/wrong', 'second correct/wrong'],
                exact_two_sided_p=float(binomtest(table[0][1], n, .5).pvalue) if n else 1.)


def calibration(y, p):
    import numpy as np
    bins = np.minimum((p * 15).astype(int), 14)
    ece, details = 0., []
    for i in range(15):
        mask = bins == i
        count = int(mask.sum())
        mean_p, rate = (float(p[mask].mean()), float(y[mask].mean())) if count else (None, None)
        if count:
            ece += count / len(y) * abs(mean_p - rate)
        details.append(dict(bin=i, rows=count, mean_probability=mean_p, positive_rate=rate))
    return dict(ece_15=float(ece), brier=float(np.mean((p-y)**2)),
                ece_definition='equal-width P(M) bins; abs(mean P(M) - fraction M)', bins=details)


def stage(out, head, fold, tr, va, features, y):
    import numpy as np
    from dcc_data.mission1_train import _build_lr_pipeline
    path = out / 'stages' / f'fold_{fold}_{head}.json'
    if path.exists():
        result = read_json(path)
        if result['indices'] != va.tolist() or result['head'] != head:
            raise ValueError('Stage identity mismatch')
        probs = np.asarray(result['probs'])
    else:
        model = _build_lr_pipeline(SEED)
        model.fit(features[head][tr], y[tr])
        if model.classes_.tolist() != [0, 1]:
            raise ValueError('Expected F=0, M=1')
        probs = model.predict_proba(features[head][va])[:, 1]
        write_json(path, dict(head=head, indices=va.tolist(), probs=probs.tolist()))
    if probs.shape != (len(va),):
        raise ValueError('Stage probability length mismatch')
    check_probs(probs)
    return probs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=os.environ.get('LAB_OUT'))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    if args.out is None:
        parser.error('--out or LAB_OUT required')
    if os.environ.get('LAB_EXP_ID', EXP_ID) != EXP_ID:
        raise ValueError('LAB_EXP_ID mismatch')
    # Cached features are the inputs: validate immediately; no pretrained load needed.
    root = Path(os.environ['DCC_RUN_ROOT'])
    features, y, calls, durations, folds, splits, saved, identity = load_inputs(root)
    import numpy as np
    import joblib
    import sklearn
    from dcc_data.mission1_train import _build_lr_pipeline, compute_metrics
    random.seed(SEED)
    np.random.seed(SEED)
    identity['sklearn_version'] = sklearn.__version__
    out = args.out
    if (out / 'config.json').exists() and read_json(out / 'config.json') != identity:
        raise ValueError('Resume input/configuration mismatch')
    write_json(out / 'config.json', identity)
    if all((out / name).exists() for name in ('metrics.json', 'best.ckpt', 'oof_probs.npy', 'oof_rows.txt', 'oof_folds.npy', 'oof_probs_columns.json')):
        print('Experiment already complete', flush=True)
        return
    reproduced = {h: np.empty(len(y)) for h in ('c', 'h')}
    if saved is None:
        reproduced['i'] = np.empty(len(y))
    for fold, (tr, va) in enumerate(splits):
        for head in reproduced:
            reproduced[head][va] = stage(out, head, fold, tr, va, features, y)
        print(f'Reproduction fold {fold} complete', flush=True)
    reproduction = {}
    valid = True
    for head, expected, column in [('c', .97973, 0), ('h', .98253, 1)]:
        accuracy = float(np.mean((reproduced[head] >= .5) == y))
        mismatch = int(np.sum((reproduced[head] >= .5) != (saved[:, column] >= .5))) if saved is not None else None
        passed = abs(accuracy-expected) <= .0005 and (mismatch is None or mismatch/len(y) <= .001)
        reproduction[head] = dict(accuracy=accuracy, reference_accuracy=expected,
                                 stored_prediction_disagreements=mismatch,
                                 stored_prediction_disagreement_rate=mismatch/len(y) if mismatch is not None else None,
                                 passed=passed)
        valid = valid and passed
    c, h, i = (saved[:, 0], saved[:, 1], saved[:, 2]) if saved is not None else (reproduced['c'], reproduced['h'], reproduced['i'])
    k, logits = fuse(c, h)
    probs = dict(c=c, h=h, k=k, ch_logit_mean=logits, ci_probability_mean=(c+i)/2)
    scores = {key: compute_metrics(y, (p >= .5).astype(int)) for key, p in probs.items()}
    fold_scores = {key: [compute_metrics(y[folds == f], (p[folds == f] >= .5).astype(int)) for f in range(5)] for key, p in probs.items()}
    ck, hk = paired(y, c, k), paired(y, h, k)
    gain = scores['k']['accuracy'] - scores['c']['accuracy']
    wins = sum(b['accuracy'] > a['accuracy'] for a, b in zip(fold_scores['c'], fold_scores['k']))
    criteria = dict(gain_ge_003=gain >= .003, fold_wins_ge_4=wins >= 4, mcnemar_p_lt_05=ck['exact_two_sided_p'] < .05)
    buckets = {}
    for name, mask in [('le20', durations <= 20), ('gt20_le40', (durations > 20) & (durations <= 40)), ('gt40', durations > 40)]:
        buckets[name] = dict(rows=int(mask.sum()), accuracies={key: float(np.mean((p[mask] >= .5) == y[mask])) if mask.any() else None for key, p in probs.items()})
    models = {}
    for head in ('c', 'h'):
        path = out / 'stages' / f'full_{head}.ckpt'
        if path.exists():
            models[head] = joblib.load(path)
        else:
            models[head] = _build_lr_pipeline(SEED).fit(features[head], y)
            atomic(path, lambda p, head=head: joblib.dump(models[head], p))
        if models[head].classes_.tolist() != [0, 1] or models[head].n_features_in_ != features[head].shape[1]:
            raise ValueError('Full-head checkpoint schema mismatch')
    comparison = dict(available=False)
    prior = root / 'lab/M1-ECAPA-002/best.ckpt'
    if prior.exists():
        old = joblib.load(prior)['pipeline']
        comparison = dict(available=True)
        for component, attrs in [('clf', ['coef_', 'intercept_']), ('scaler', ['mean_', 'scale_'])]:
            for attr in attrs:
                a, b = getattr(models['h'].named_steps[component], attr), getattr(old.named_steps[component], attr)
                comparison[f'{component}_{attr}'] = dict(shape_match=a.shape == b.shape,
                    allclose=bool(a.shape == b.shape and np.allclose(a, b, rtol=1e-7, atol=1e-9)),
                    max_abs_difference=float(np.max(np.abs(a-b))) if a.shape == b.shape else None)
    atomic(out / 'best.ckpt', lambda p: joblib.dump(dict(pipelines=models, weights={'c': .5, 'h': .5},
           threshold=.5, probability_of='M', classes={'F': 0, 'M': 1}, windows=WINDOWS,
           model_revisions={'microsoft/wavlm-base-plus-sv': XV_REVISION,
                            'speechbrain/spkrec-ecapa-voxceleb': EC_REVISION},
           feature_layout={'c': ['acoustic_52d', 'xvector_512d'], 'h': ['acoustic_52d', 'ecapa_192d']}), p))
    save_npy(out / 'oof_probs.npy', np.column_stack(list(probs.values())))
    save_npy(out / 'oof_folds.npy', folds)
    write_json(out / 'oof_probs_columns.json', list(probs))
    atomic(out / 'oof_rows.txt', lambda p: p.write_text(''.join(call+'\n' for call in calls), encoding='utf-8'))
    metrics = dict(oof=scores, cv_folds=fold_scores, calibration={key: calibration(y, probs[key]) for key in ('c', 'h', 'k')},
                   paired_mcnemar={'c_k': ck, 'h_k': hk}, duration_buckets=buckets,
                   disagreement_calls=int(np.sum((c >= .5) != (h >= .5))),
                   fusion_correct_on_disagreement=int(np.sum(((c >= .5) != (h >= .5)) & ((k >= .5) == y))),
                   gain_k_minus_c=gain, gain_k_minus_h=scores['k']['accuracy']-scores['h']['accuracy'],
                   fold_wins=wins, criteria=criteria, reproduction=reproduction,
                   reproduction_passed=valid, full_h_checkpoint_comparison=comparison,
                   probability_source='stored' if saved is not None else 'reproduced',
                   decision='deferred_reproduction_failure' if not valid else 'criteria_met' if all(criteria.values()) else 'criteria_not_met',
                   inference_seconds_per_call=dict(xvector=.394, ecapa=.058, total_approx=.45),
                   seeds=dict(random=SEED, numpy=SEED, cv_random_state=CV_SEED))
    write_json(out / 'metrics.json', dict(exp_id=EXP_ID, primary_metric=dict(name='accuracy', value=scores['k']['accuracy'], split='internal_cv'),
         metrics=metrics, seed=SEED, train_rows=len(y), eval_rows=len(y), official_val=None, ckpt='best.ckpt',
         notes='Fixed equal probability fusion and >=0.5 threshold; diagnostic columns do not affect criteria. No audio/Validation/network. Interrupted nonincremental LR fits restart only that stage.'))
    print(f"OOF fusion accuracy={scores['k']['accuracy']:.8f}; {metrics['decision']}", flush=True)


def smoke():
    import numpy as np
    c = np.array([0., .5, 1., .9])
    h = np.array([1., .5, 0., .7])
    k, logits = fuse(c, h)
    assert np.allclose(k, [.5, .5, .5, .8]) and np.isfinite(logits).all()
    y = np.array([0, 1, 1, 1])
    assert paired(y, c, c)['exact_two_sided_p'] == 1.
    assert calibration(y, y.astype(float))['brier'] == 0.
    assert calibration(y, y.astype(float))['ece_15'] == 0.
    assert np.allclose(average_windows(np.array([[[3., 4.], [0., 0.]]]), np.zeros((1, 2))), [[.6, .8]])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'stage.json'
        write_json(path, {'done': True})
        assert read_json(path) == {'done': True}
        save_npy(Path(directory) / 'probs.npy', k)
        assert np.array_equal(np.load(Path(directory) / 'probs.npy'), k)
    print('Smoke OK: fixed fusion, exact paired test, calibration, pooling, atomic artifacts; no fitting')


if __name__ == '__main__':
    main()
