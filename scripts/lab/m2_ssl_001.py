#!/usr/bin/env python3
"""Frozen local WavLM layers 4/8/12 vs context-free 53D L1, Seoul Training only.

Pre-stage microsoft/wavlm-base-plus with save_pretrained into --model-dir.
LR is logistic regression optimized by resumable SGD (30 passes, no tuning).
LGBM uses baseline defaults, 100 trees, checkpointed every 10 trees.
Only the initial tar packaging touches loose source files; extraction reads local disk.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tarfile
import time
from pathlib import Path

EXP_ID = "M2-SSL-001"
LAYERS = (4, 8, 12)
SEED = 42


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def atomic_dump(path, value):
    import joblib
    temp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temp)
    os.replace(temp, path)


def seed_all():
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    import numpy as np
    import torch
    return dict(random=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['random'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def checkpoint(out, value):
    import torch
    temp = out / 'ckpt_last.pt.tmp'
    torch.save({**value, 'rng': rng_state()}, temp)
    os.replace(temp, out / 'ckpt_last.pt')


def crop_bounds(length):
    start = max(0, (length - 96000) // 2)
    return start, min(length, start + 96000), max(0, 8000 - length)


def crop(segment):
    import numpy as np
    start, end, padding = crop_bounds(len(segment))
    return np.pad(segment[start:end], (0, padding)).astype('float32')


def stage(out, work, n_calls):
    from dcc_data.colab_env import find_data_root
    from dcc_data.mission2_features import is_seoul_path
    from dcc_data.manifest import ManifestRow
    archive = out / 'training_sample.tar'
    manifest = out / 'sample.json'
    if not manifest.exists():
        root = find_data_root()
        pairs = {}
        # Never walk Validation. Read source contents only when building the tar.
        for path in sorted((root / 'Training').rglob('*')):
            if path.suffix.lower() in ('.wav', '.json') and is_seoul_path(path.relative_to(root).as_posix()):
                role = path.suffix.lower()
                group = pairs.setdefault(path.stem, {})
                if role in group:
                    raise ValueError(f'duplicate source stem: {path.stem}')
                group[role] = path.relative_to(root).as_posix()
        stems = sorted(s for s, p in pairs.items() if len(p) == 2)
        if len(stems) < 2:
            raise ValueError('Need at least two paired Seoul Training calls (loose WAV/JSON layout)')
        stems = random.Random(SEED).sample(stems, min(n_calls, len(stems)))
        rows = [dict(split='Training', stem=s, wav_path=pairs[s]['.wav'], json_path=pairs[s]['.json']) for s in stems]
        temp = archive.with_suffix('.tar.tmp')
        with tarfile.open(temp, 'w') as tar:
            for row in rows:
                for key in ('wav_path', 'json_path'):
                    tar.add(root / row[key], arcname=row[key], recursive=False)
        os.replace(temp, archive)
        atomic_json(manifest, rows)
    rows = json.loads(manifest.read_text())
    local = work / 'training_sample.tar'
    marker = work / 'staged.json'
    signature = {'bytes': archive.stat().st_size, 'sample': rows}
    if not marker.exists() or json.loads(marker.read_text()) != signature:
        shutil.copyfile(archive, local)
        with tarfile.open(local) as tar:
            for member in tar.getmembers():
                target = (work / member.name).resolve()
                if not target.is_relative_to(work.resolve()) or not member.isfile():
                    raise ValueError('Unsafe tar member')
            tar.extractall(work, filter='data')
        atomic_json(marker, signature)
        local.unlink()
    return [ManifestRow(**row) for row in rows]


def extract(args, out, work, rows):
    import numpy as np
    import torch
    import librosa
    from dcc_data.mission2_features import (build_utterance_records, read_call_waveform,
                                          utterance_frame_range, utterance_feature_vector)
    from transformers import AutoFeatureExtractor, WavLMModel
    index_path = out / 'utterances.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = []
        for row in rows:
            for r in build_utterance_records(row, work):
                index.append(dict(call_id=r.call_id, stem=row.stem, utt_idx=r.utt_idx,
                                  start=r.start_at, end=r.end_at, label=r.speaker))
        atomic_json(index_path, index)
    if not index:
        raise ValueError('No utterances')
    paths = {str(layer): out / f'layer_{layer}.npy' for layer in LAYERS}
    paths['l1'] = out / 'l1_53.npy'
    arrays = {}
    progress_path = out / 'extraction.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else dict(done=[], seconds=0., rows=0, gpu=None)
    for key, path in paths.items():
        if not path.exists() and progress['done']:
            raise ValueError(f'Missing committed feature array: {path}')
        expected_shape = (len(index), 53 if key == 'l1' else 1536)
        expected_dtype = np.dtype('float32' if key == 'l1' else 'float16')
        arrays[key] = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+',
                                               dtype=expected_dtype, shape=expected_shape)
        if arrays[key].shape != expected_shape or arrays[key].dtype != expected_dtype:
            raise ValueError(f'Feature cache shape/dtype mismatch: {path}')
    pending = [row for row in rows if row.stem not in progress['done']]
    if not pending:
        return index, arrays, progress
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for full extraction; --smoke is CPU-only')
    processor = AutoFeatureExtractor.from_pretrained(args.model_dir, local_files_only=True)
    model = WavLMModel.from_pretrained(args.model_dir, local_files_only=True).cuda().eval()
    model.requires_grad_(False)
    progress['gpu'] = torch.cuda.get_device_name(0)
    positions = {}
    for i, record in enumerate(index):
        positions.setdefault(record['stem'], []).append(i)
    for row in pending:
        started = time.perf_counter()
        waveform, sr = read_call_waveform(work / row.wav_path)
        segments = []
        indices = positions[row.stem]
        for i in indices:
            record = index[i]
            start, end = utterance_frame_range(record['start'], record['end'], sr, len(waveform))
            segment = waveform[start:end]
            # Baseline retains the complete target utterance, matching existing L1.
            arrays['l1'][i] = utterance_feature_vector(segment, sr, (record['end'] - record['start']) / 1000)
            if sr != 16000:
                segment = librosa.resample(segment, orig_sr=sr, target_sr=16000)
            segments.append(crop(segment))
        for offset in range(0, len(indices), args.batch_size):
            batch = segments[offset:offset + args.batch_size]
            inputs = processor(batch, sampling_rate=16000, padding=True,
                               return_attention_mask=True, return_tensors='pt')
            inputs = {key: value.cuda() for key, value in inputs.items()}
            with torch.inference_mode():
                hidden = model(**inputs, output_hidden_states=True).hidden_states
                lengths = model._get_feat_extract_output_lengths(inputs['attention_mask'].sum(-1))
                for layer in LAYERS:
                    for j, length in enumerate(lengths.tolist()):
                        frames = hidden[layer][j, :length].float()
                        vector = torch.cat((frames.mean(0), frames.std(0, unbiased=False)))
                        arrays[str(layer)][indices[offset + j]] = vector.cpu().numpy().astype('float16')
        torch.cuda.synchronize()
        for array in arrays.values():
            array.flush()
        progress['done'].append(row.stem)
        progress['rows'] += len(indices)
        progress['seconds'] += time.perf_counter() - started
        atomic_json(progress_path, progress)
        print(f"extracted {len(progress['done'])}/{len(rows)} calls", flush=True)
    return index, arrays, progress


def heads(out, index, arrays, progress):
    import numpy as np
    import torch
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import accuracy_score
    import joblib
    y = np.array([r['label'] for r in index])
    groups = np.array([r['call_id'] for r in index])
    train, holdout = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(y, groups=groups))
    assert not set(groups[train]) & set(groups[holdout])
    atomic_json(out / 'split.json', dict(train_call_ids=sorted(set(groups[train])),
                                       holdout_call_ids=sorted(set(groups[holdout]))))
    scores = {}
    for feature, head in [('l1', 'LGBM')] + [(str(l), h) for l in LAYERS for h in ('LR', 'LGBM')]:
        name = f'{feature}_{head}'
        artifact = out / f'{name}.ckpt'
        score_path = out / f'{name}.json'
        if artifact.exists() and score_path.exists():
            scores[name] = json.loads(score_path.read_text())['accuracy']
            continue
        x_train = np.asarray(arrays[feature][train], dtype=np.float32)
        x_eval = np.asarray(arrays[feature][holdout], dtype=np.float32)
        last = out / 'ckpt_last.pt'
        state = torch.load(last, map_location='cpu', weights_only=False) if last.exists() else None
        if state and state['name'] == name:
            model, scaler, epoch = state['model'], state['scaler'], state['epoch']
            restore_rng(state['rng'])
        else:
            scaler = StandardScaler().fit(x_train) if head == 'LR' else None
            model = SGDClassifier(loss='log_loss', alpha=.0001, random_state=SEED) if head == 'LR' else None
            epoch = 0
        if scaler is not None:
            x_train, x_eval = scaler.transform(x_train), scaler.transform(x_eval)
        total = 30 if head == 'LR' else 10
        for step in range(epoch, total):
            if head == 'LR':
                order = np.random.permutation(len(train))
                for offset in range(0, len(order), 2048):
                    batch = order[offset:offset + 2048]
                    model.partial_fit(x_train[batch], y[train][batch], classes=np.array([0, 1]))
            else:
                previous = model.booster_ if model is not None else None
                model = LGBMClassifier(n_estimators=10, random_state=SEED, verbosity=-1,
                                       deterministic=True, force_col_wise=True, n_jobs=2)
                model.fit(x_train, y[train], init_model=previous)
            # SGD optimizer/learning-rate counter is contained in estimator t_/coef_.
            checkpoint(out, dict(name=name, model=model, scaler=scaler, epoch=step + 1,
                                 optimizer=model.__dict__ if head == 'LR' else model.booster_.model_to_string(),
                                 scheduler={'learning_rate': 'optimal'} if head == 'LR' else None))
        accuracy = float(accuracy_score(y[holdout], model.predict(x_eval)))
        fitted = make_pipeline(scaler, model) if scaler is not None else model
        atomic_dump(artifact, dict(model=fitted, feature=feature, head=head, seed=SEED,
                                  pretrained_model='microsoft/wavlm-base-plus', layer=int(feature) if feature != 'l1' else None,
                                  pooling='mean+std', sample_rate=16000, max_seconds=6, min_seconds=.5))
        atomic_json(score_path, dict(accuracy=accuracy))
        scores[name] = accuracy
    winner = max((k for k in scores if not k.startswith('l1_')), key=scores.get)
    atomic_dump(out / 'best.ckpt', joblib.load(out / f'{winner}.ckpt'))
    baseline = scores['l1_LGBM']
    atomic_json(out / 'metrics.json', dict(exp_id=EXP_ID,
        primary_metric=dict(name='accuracy', value=scores[winner], split='internal_holdout'),
        metrics=dict(holdout_accuracy=scores, l1_same_split_accuracy=baseline,
                     delta_vs_l1=scores[winner] - baseline, meets_success_criteria=scores[winner] - baseline >= .02,
                     extraction_utterances_per_second=progress['rows'] / max(progress['seconds'], 1e-9),
                     extraction_seconds=progress['seconds'], gpu=progress['gpu'], selected=winner,
                     train_calls=len(set(groups[train])), eval_calls=len(set(groups[holdout]))),
        seed=SEED, train_rows=len(train), eval_rows=len(holdout), official_val=None,
        ckpt='best.ckpt', notes='LR: SGD log_loss, 30 epochs; LGBM: baseline defaults, 100 trees. Throughput includes L1 extraction and cache writes. No official Validation read.'))


def smoke():
    # Dependency-free policy checks; no data, downloads or training.
    assert crop_bounds(1000) == (0, 1000, 7000)
    assert crop_bounds(100000) == (2000, 98000, 0)
    assert crop_bounds(8000) == (0, 8000, 0)
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'state.json'
        atomic_json(path, {'done': ['call']})
        assert json.loads(path.read_text()) == {'done': ['call']}
    print('smoke OK: crop, zero padding, atomic progress; no training')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=os.environ.get('LAB_OUT'))
    parser.add_argument('--work', type=Path, default=os.environ.get('LAB_WORK'))
    parser.add_argument('--model-dir', default=os.path.join(os.environ.get('LAB_WORK', '.'), 'models/wavlm-base-plus'))
    parser.add_argument('--n-calls', type=int, default=4000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    if not args.out or not args.work:
        parser.error('LAB_OUT and LAB_WORK (or --out/--work) are required')
    if args.n_calls < 2 or args.batch_size < 1:
        parser.error('n-calls >= 2 and batch-size >= 1 required')
    if os.environ.get('LAB_EXP_ID', EXP_ID) != EXP_ID:
        parser.error('LAB_EXP_ID must be M2-SSL-001')
    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    config = dict(exp_id=EXP_ID, seed=SEED, n_calls=args.n_calls,
                  model_dir=str(Path(args.model_dir).resolve()), layers=list(LAYERS), schema=1)
    config_path = args.out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('LAB_OUT config mismatch: use a fresh output directory')
    atomic_json(config_path, config)
    if (args.out / 'metrics.json').exists() and (args.out / 'best.ckpt').exists():
        print('Experiment already complete')
        return
    seed_all()
    rows = stage(args.out, args.work, args.n_calls)
    index, arrays, progress = extract(args, args.out, args.work, rows)
    heads(args.out, index, arrays, progress)


if __name__ == '__main__':
    main()
