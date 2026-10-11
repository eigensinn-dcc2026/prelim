"""Fixed Training-only ECAPA replacement experiment; no head or window tuning.

LR has no incremental optimizer API. Interrupted fits restart only that stage;
completed feature chunks and fold/head stages are durable. No neural training.
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

EXP_ID = 'M1-ECAPA-002'
REVISION = '0f99f2d0ebe89ac095bcc5903c4dd8f72b367286'
MODEL_ID = 'speechbrain/spkrec-ecapa-voxceleb'
SEED = 42
HEADS = ['c', 'h', 'i', 'j']
CONFIG = dict(exp_id=EXP_ID, revision=REVISION, seed=42, cv_seed=2026,
              rows=29200, chunk_size=2000, window_seconds=20, max_windows=4,
              tail_min_seconds=3, pad_min_seconds=.5, schema=1)


def baseline_helpers():
    spec = importlib.util.spec_from_file_location('m1_training_helpers',
                                                 Path(__file__).with_name('m1-xvec-001.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def save_npz(path, **arrays):
    import numpy as np
    def writer(p):
        with p.open('wb') as stream:
            np.savez(stream, **arrays)
    atomic(path, writer)


def save_npy(path, array):
    import numpy as np
    def writer(p):
        with p.open('wb') as stream:
            np.save(stream, array, allow_pickle=False)
    atomic(path, writer)


def window_bounds(length, sr=16000):
    if length < 0:
        raise ValueError('Negative caller waveform length')
    if length == 0:
        return [(0, 0)]
    return [(start, min(start + 20 * sr, length))
            for start in range(0, min(length, 80 * sr), 20 * sr)
            if start == 0 or min(start + 20 * sr, length) - start >= 3 * sr]


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


def stage_model(root, local, required):
    """Copy snapshot files atomically; SpeechBrain must never write to Drive.

    Refresh the small set of model files on every process start to recover from
    interrupted copies and avoid trusting a stale local snapshot.
    """
    local.mkdir(parents=True, exist_ok=True)
    for name in required:
        atomic(local / name, lambda tmp, name=name: shutil.copyfile(root / name, tmp))
    return local


def prepare_model():
    import torch
    from huggingface_hub import snapshot_download
    import torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        torchaudio.list_audio_backends = lambda: []
    # The compatibility shim must run before any SpeechBrain import.
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
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
    local = stage_model(
        root, Path(os.environ["LAB_WORK"]) / EXP_ID.lower() / "models"
        / "spkrec-ecapa-voxceleb" / REVISION, required)
    # Upstream YAML has explicit HF paths: override them before collection.
    # All source and collection paths are local, including label_encoder.ckpt.
    # COPY also avoids symlinks on filesystems that do not support them.
    model = EncoderClassifier.from_hparams(
        source=str(local), savedir=str(local),
        overrides={"pretrained_path": str(local)},
        local_strategy=LocalStrategy.COPY,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"})
    model.eval()
    with torch.inference_mode():
        check = model.encode_batch(torch.zeros(1, 8000, device=model.device))
    if check.numel() != 192 or not torch.isfinite(check).all():
        raise ValueError("Expected finite 192D ECAPA output")
    write_json(marker, identity)
    return model



def extract(out, work, rows, cached, model):
    import librosa
    import numpy as np
    import torch
    from dcc_data.mission1 import extract_caller_segments, concatenate_caller_waveforms
    collected = []
    for start in range(0, len(rows), 2000):
        stop = min(start + 2000, len(rows))
        path = out / 'features' / f'chunk_{start:05d}_{stop:05d}.npz'
        if not path.exists():
            vectors = np.zeros((stop-start, 4, 192), dtype=np.float32)
            lengths = np.zeros((stop-start, 4), dtype=np.float64)
            durations, counts, seconds = [], [], []
            for k, row in enumerate(rows[start:stop]):
                segments = extract_caller_segments(row, work)
                wave = concatenate_caller_waveforms(segments)
                if wave.shape[0] != 1:
                    raise ValueError('Expected mono caller waveform')
                sr = segments[0].sample_rate
                duration = wave.shape[1] / sr
                bounds = window_bounds(wave.shape[1], sr)
                if abs(duration-cached['durations'][start+k]) > 1e-3 or len(bounds) != cached['window_counts'][start+k]:
                    raise ValueError(f'Caller duration/window mismatch: {row.stem}')
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                began = time.perf_counter()
                for w, (a, b) in enumerate(bounds):
                    audio = wave[0, a:b]
                    if sr != 16000 and len(audio):
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    audio = np.pad(np.asarray(audio, dtype=np.float32), (0, max(0, 8000-len(audio))))
                    with torch.inference_mode():
                        embedding = model.encode_batch(torch.from_numpy(audio).unsqueeze(0).to(model.device)).reshape(-1)
                        embedding = torch.nn.functional.normalize(embedding, dim=0)
                    if embedding.numel() != 192:
                        raise ValueError('Expected 192D ECAPA')
                    vectors[k, w] = embedding.cpu().numpy()
                    lengths[k, w] = (b-a)/sr
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                seconds.append(time.perf_counter()-began)
                durations.append(duration)
                counts.append(len(bounds))
            save_npz(path, window_ecapa=vectors, window_lengths=lengths,
                     durations=np.asarray(durations), window_counts=np.asarray(counts),
                     ecapa_seconds=np.asarray(seconds), stems=np.asarray([r.stem for r in rows[start:stop]]))
        with np.load(path, allow_pickle=False) as z:
            item = {key: z[key].copy() for key in z.files}
        if (item['window_ecapa'].shape != (stop-start, 4, 192)
                or item['window_lengths'].shape != (stop-start, 4)
                or item['stems'].tolist() != [r.stem for r in rows[start:stop]]
                or not np.allclose(item['durations'], cached['durations'][start:stop], atol=1e-3, rtol=0)
                or not np.array_equal(item['window_counts'], cached['window_counts'][start:stop])
                or any(not np.isfinite(item[k]).all() for k in item if k != 'stems')
                or item['ecapa_seconds'].shape != (stop-start,)
                or (item['ecapa_seconds'] < 0).any()
                or not np.allclose(item['window_lengths'], cached['window_lengths'][start:stop], atol=1e-3, rtol=0)
                or not np.array_equal(np.maximum(1, (item['window_lengths'] > 0).sum(1)), item['window_counts'])):
            raise ValueError(f'Invalid ECAPA chunk: {path}')
        collected.append(item)
        print(f'ECAPA chunk complete {start}:{stop}', flush=True)
    return {k: np.concatenate([c[k] for c in collected]) for k in collected[0]}


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
    import numpy as np
    import torch
    import joblib
    from scipy.stats import binomtest
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from dcc_data.mission1_train import _build_lr_pipeline, compute_metrics, encode_gender
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cached, split = load_baseline(Path(os.environ['DCC_RUN_ROOT']) / 'lab' / 'M1-XVEC-002')
    model = prepare_model()  # Fail fast before archives are copied.
    out, work = args.out, Path(os.environ['LAB_WORK']) / EXP_ID.lower()
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    rows, genders, calls = baseline_helpers().prepare_rows(work)
    row_identity = dict(calls=calls, genders=genders, paths=[(r.wav_path, r.json_path) for r in rows])
    digest = hashlib.sha256(json.dumps(row_identity, sort_keys=True).encode()).hexdigest()
    y = encode_gender(genders)
    train, val = train_test_split(np.arange(len(rows)), test_size=.2, stratify=y, random_state=42)
    if (len(rows) != 29200 or len({r.stem for r in rows}) != len(rows)
            or digest != split['rows_sha256']
            or train.tolist() != split['train'] or val.tolist() != split['holdout']):
        raise ValueError('Baseline manifest/split does not match Training')
    identity = dict(CONFIG, baseline_split=split,
                    stems_sha256=hashlib.sha256(json.dumps([r.stem for r in rows]).encode()).hexdigest(),
                    baseline_arrays_sha256={k: hashlib.sha256(v.tobytes()).hexdigest()
                                            for k, v in cached.items()})
    if (out / 'config.json').exists() and json.loads((out / 'config.json').read_text()) != identity:
        raise ValueError('Output configuration mismatch')
    write_json(out / 'config.json', identity)
    if (out / 'metrics.json').exists() and (out / 'best.ckpt').exists():
        print('Experiment already complete')
        return
    write_json(out / 'manifest.json', dict(stems=[r.stem for r in rows], calls=calls, genders=genders))
    ecapa = extract(out, work, rows, cached, model)
    del model
    x = average_windows(cached['window_xvectors'], cached['window_lengths'])
    e = average_windows(ecapa['window_ecapa'], ecapa['window_lengths'])
    features = dict(c=np.concatenate([cached['acoustic'], x], axis=1),
                    h=np.concatenate([cached['acoustic'], e], axis=1), i=e,
                    j=np.concatenate([cached['acoustic'], ecapa['window_ecapa'][:, 0]], axis=1))
    oof = np.zeros((len(y), 4))
    folds = np.zeros(len(y), dtype=np.int64)
    fold_scores = {h: [] for h in HEADS}

    def stage(name, h, tr, va):
        path = out / 'stages' / f'{name}_{h}.json'
        if path.exists():
            result = json.loads(path.read_text())
            if result['indices'] != va.tolist() or len(result['probs']) != len(va):
                raise ValueError('Stage identity mismatch')
            probs = np.asarray(result['probs'])
            if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
                raise ValueError('Invalid stage probabilities')
        else:
            pipeline = _build_lr_pipeline(SEED)
            pipeline.fit(features[h][tr], y[tr])
            probs = pipeline.predict_proba(features[h][va])[:, 1]
            result = dict(indices=va.tolist(), probs=probs.tolist(),
                          metrics=compute_metrics(y[va], (probs > .5).astype(int)))
            write_json(path, result)
        return probs, result['metrics']

    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=2026).split(e, y)):
        folds[va] = fold
        for col, h in enumerate(HEADS):
            oof[va, col], score = stage(f'fold_{fold}', h, tr, va)
            fold_scores[h].append(score)
        print(f'CV fold {fold} complete', flush=True)
    holdout = np.zeros((len(val), 4))
    holdout_scores = {}
    for col, h in enumerate(HEADS):
        holdout[:, col], holdout_scores[h] = stage('seed_42', h, train, val)
    save_npy(out / 'oof_probs.npy', oof)
    save_npy(out / 'oof_folds.npy', folds)
    save_npy(out / 'seed_42' / 'holdout_probs.npy', holdout)
    write_json(out / 'seed_42' / 'holdout_probs_columns.json', HEADS)
    write_json(out / 'oof_probs_columns.json', HEADS)
    atomic(out / 'seed_42' / 'val_files.txt', lambda p: p.write_text(''.join(rows[k].stem+'\n' for k in val)))
    scores = {h: compute_metrics(y, (oof[:, col] > .5).astype(int)) for col, h in enumerate(HEADS)}
    c, h = (oof[:, 0] > .5) == y, (oof[:, 1] > .5) == y
    table = [[int((c & h).sum()), int((c & ~h).sum())],
             [int((~c & h).sum()), int((~c & ~h).sum())]]
    discordant = table[0][1] + table[1][0]
    p = float(binomtest(table[0][1], discordant, .5).pvalue) if discordant else 1.
    gain = scores['h']['accuracy'] - scores['c']['accuracy']
    wins = sum(b['accuracy'] > a['accuracy'] for a, b in zip(fold_scores['c'], fold_scores['h']))
    reproduced = abs(holdout_scores['c']['accuracy']-.9772) <= .002
    buckets = {}
    d = cached['durations']
    for name, mask in [('le20', d <= 20), ('gt20_le40', (d > 20) & (d <= 40)), ('gt40', d > 40)]:
        buckets[name] = dict(rows=int(mask.sum()), c=float(c[mask].mean()) if mask.any() else None,
                            h=float(h[mask].mean()) if mask.any() else None)
    if not (out / 'best.ckpt').exists():
        pipeline = _build_lr_pipeline(SEED)
        pipeline.fit(features['h'], y)
        atomic(out / 'best.ckpt', lambda p: joblib.dump(dict(pipeline=pipeline, head='h',
                     config=CONFIG, model_id=MODEL_ID, revision=REVISION, classes={'F': 0, 'M': 1}), p))
    seconds = ecapa['ecapa_seconds']
    timing = dict(mean=float(seconds.mean()), p50=float(np.median(seconds)),
                  p95=float(np.quantile(seconds, .95)), xvector_reference_seconds=.394,
                  mean_ratio_to_xvector=float(seconds.mean()/.394),
                  scope='window resampling, padding, ECAPA encode and pooling transfer; excludes audio I/O')
    write_json(out / 'metrics.json', dict(exp_id=EXP_ID,
        primary_metric=dict(name='accuracy', value=scores['h']['accuracy'], split='internal_cv'),
        metrics=dict(cv_folds=fold_scores, oof=scores, holdout=holdout_scores,
                     paired_mcnemar=dict(table=table, axes=['c correct/wrong', 'h correct/wrong'], exact_two_sided_p=p),
                     error_overlap=dict(both_wrong=table[1][1], c_only_wrong=table[1][0], h_only_wrong=table[0][1]),
                     duration_buckets=buckets, window_count_distribution={str(k): int((ecapa['window_counts'] == k).sum()) for k in range(1, 5)},
                     ecapa_seconds_per_call=timing, oof_accuracy_gain=gain, fold_wins=wins,
                     holdout_baseline_reproduced=reproduced, criteria=dict(gain_ge_003=gain >= .003, wins_ge_4=wins >= 4, p_lt_05=p < .05),
                     decision='deferred_reproduction_failure' if not reproduced else 'criteria_met' if gain >= .003 and wins >= 4 and p < .05 else 'criteria_not_met',
                     seeds=dict(random=42, numpy=42, torch=42, cuda=42, cv_random_state=2026)),
        seed=SEED, train_rows=len(y), eval_rows=len(y), official_val=None, ckpt='best.ckpt',
        notes='Fixed primary head h; i/j diagnostic only. Training call-level CV; seed42 holdout continuity only. Interrupted LR fit restarts its stage.'))


def smoke():
    import numpy as np
    assert window_bounds(0) == [(0, 0)]
    assert window_bounds(7999) == [(0, 7999)]
    vectors = np.array([[[3., 4.], [0., 0.]]])
    assert np.allclose(average_windows(vectors, np.zeros((1, 2))), [[.6, .8]])
    assert np.allclose(average_windows(vectors, np.array([[.1, 0.]])), [[.6, .8]])
    assert window_bounds(16000) == [(0, 16000)]
    assert len(window_bounds(22*16000)) == 1
    assert len(window_bounds(23*16000)) == 2
    assert len(window_bounds(100*16000)) == 4
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'features').mkdir()
        write_json(root / 'split.json', {})
        # Legacy cache has no stems; reach the total-row check successfully.
        save_npz(root / 'features' / 'chunk_00000_00002.npz',
                 acoustic=np.zeros((2, 52)), window_xvectors=np.zeros((2, 4, 512)),
                 window_lengths=np.array([[0., 0., 0., 0.], [.1, 0., 0., 0.]]),
                 durations=np.array([0., .1]), window_counts=np.ones(2))
        try:
            load_baseline(root)
        except ValueError as error:
            assert str(error) == 'Expected 29200 cached rows, got 2', str(error)
        else:
            raise AssertionError('Incomplete cache accepted')
        target = root / 'atomic.json'
        write_json(target, CONFIG)
        assert json.loads(target.read_text()) == CONFIG
        write_json(target, {'resume': True})
        assert json.loads(target.read_text()) == {'resume': True}
    print('Smoke OK: legacy cache without stems, zero/short windows and pooling, local serialization and atomic publication; no training/download')


if __name__ == '__main__':
    main()
