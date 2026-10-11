#!/usr/bin/env python3
"""M2-MIX-001: fixed A/B Mixup comparison; Training Seoul only, no external models.

Fallback split uses sorted call IDs and random.Random(42).sample: H first,
S from the remainder, D from S. Prior SSL code is absent in this checkout;
without its split.json, equivalence is explicitly unverified (never claimed).
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
import zipfile

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from torch import nn

from dcc_data.colab_env import find_data_root
from dcc_data.manifest import ManifestRow
from dcc_data.mission2_features import (
    build_utterance_records, read_call_waveform, utterance_frame_range,
    slice_and_pad, N_FFT, HOP_LENGTH,
)
from dcc_data.mission2_spectrogram import mel_spectrogram, SPEC_FLOOR_DB
from dcc_data.mission2_cnn import (
    SpeakerCNNLarge, PackedSpectrogramDataset, collate_batch,
    move_optimizer_state_to_device,
)
from dcc_data.mission2_checkpoint import build_checkpoint_metadata

EXP_ID = 'M2-MIX-001'


def publish(local: Path, dest: Path):
    """Always stage locally, copy to Drive temporary name, verify, then replace."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(6):
        try:
            tmp = dest.with_name(dest.name + '.tmp')
            shutil.copyfile(local, tmp)
            if tmp.stat().st_size != local.stat().st_size:
                raise OSError('incomplete copy')
            os.replace(tmp, dest)
            return
        except OSError:
            if attempt == 5:
                raise
            time.sleep(2 ** (attempt + 1))


def write_json(value, dest: Path, work: Path):
    with tempfile.NamedTemporaryFile(mode='w', dir=work, suffix='.json', delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        local = Path(f.name)
    try:
        publish(local, dest)
    finally:
        local.unlink(missing_ok=True)


def save_torch(value, dest, work):
    local = work / 'checkpoint_local.pt'
    torch.save(value, local)
    publish(local, dest)
    local.unlink()


def digest(ids):
    return hashlib.sha256(('\n'.join(sorted(ids)) + '\n').encode()).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def zip_members(root, work, folder, suffix):
    source = root / 'Training' / (folder + '.zip')
    local = work / source.name
    if not local.exists() or local.stat().st_size != source.stat().st_size:
        staging = local.with_suffix('.tmp')
        shutil.copyfile(source, staging)
        os.replace(staging, local)
    archive = zipfile.ZipFile(local)
    members = {}
    for name in archive.namelist():
        p = Path(name)
        if p.suffix.lower() == suffix:
            if p.stem in members:
                raise ValueError(f'duplicate call: {p.stem}')
            members[p.stem] = name
    return archive, members


def label_call_members(archive, members):
    """Index by JSON _id (stem fallback), retaining stems for audio pairing."""
    calls = {}
    stems = {}
    for stem, member in sorted(members.items()):
        document = json.loads(archive.read(member))
        call_id = str(document.get('_id') or stem)
        if call_id in calls:
            raise ValueError(f'duplicate JSON call_id {call_id!r}: '
                             f'{stems[call_id]!r}, {stem!r}')
        calls[call_id] = member
        stems[call_id] = stem
    return calls, stems


def make_split(ids, out, run_root, work, stems=None):
    ids = set(ids)
    own = out / 'split.json'
    prior = run_root / 'lab/M2-SSL-002/split.json'
    path = own if own.exists() else prior
    if path.exists():
        raw = load_json(path)
        # Accept common baseline naming conventions, but validate all membership.
        aliases = {'H': ('H', 'h_calls', 'holdout_calls', 'holdout_call_ids'),
                   'S': ('S', 's_calls', 'sample_calls', 'train_calls', 'train_call_ids'),
                   'D': ('D', 'd_calls', 'dev_calls', 'dev_call_ids')}
        split = {}
        for key, names in aliases.items():
            value = next((raw[n] for n in names if n in raw), None)
            if isinstance(value, dict):
                value = value.get('call_ids')
            if not isinstance(value, list):
                raise ValueError(f'{path}: missing {key} call list; unsupported split schema')
            split[key] = value
        source = raw.get('source', str(path))
        verified = raw.get('baseline_equivalence_verified', path == prior)
    else:
        rng = random.Random(42)
        h = rng.sample(sorted(ids), 1500)
        s = rng.sample(sorted(set(ids) - set(h)), 6000)
        split = {'H': h, 'S': s, 'D': rng.sample(sorted(s), 300)}
        source = 'fallback_random.Random(42).sample_sorted_H_then_S_then_D'
        verified = False
    for k, count in [('H', 1500), ('S', 6000), ('D', 300)]:
        values = split[k]
        if any(not isinstance(value, str) for value in values):
            raise ValueError(f'{path}: invalid {k} split: call IDs must be strings; '
                             f'sample={values[:5]!r}')
        missing = set(values) - ids
        if len(values) != count or len(set(values)) != count or missing:
            stem_ids = set(stems.values()) if stems else set()
            raise ValueError(
                f'{path}: invalid {k} split: expected={count}, rows={len(values)}, '
                f'unique={len(set(values))}, available_json_call_ids={len(ids)}, '
                f'missing={len(missing)}, missing_sample={sorted(missing)[:5]!r}, '
                f'json_call_id_sample={sorted(ids)[:5]!r}, '
                f'stem_sample={sorted(stem_ids)[:5]!r}, '
                f'missing_matching_filename_stems={len(missing & stem_ids)}; '
                'call IDs use JSON _id with filename stem fallback; '
                'the saved split is not regenerated on validation failure')
    if set(split['H']) & set(split['S']) or not set(split['D']) <= set(split['S']):
        raise ValueError('call split contamination')
    hashes = {k: digest(split[k]) for k in ('H', 'S', 'D')}
    split.update(source=source, baseline_equivalence_verified=verified,
                 sha256=hashes, seed=42,
                 sha256_encoding='sorted UTF-8 call IDs joined by newline, trailing newline')
    write_json(split, own, work)
    return split


def prepare(out, work, run_root):
    cache = out / 'features'
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / 'done.json').exists():
        manifest = load_json(cache / 'done.json')
        for name in ('packed.npy', 'index.json'):
            dst = work / name
            if not dst.exists() or dst.stat().st_size != manifest['sizes'][name]:
                shutil.copyfile(cache / name, dst)
            if dst.stat().st_size != manifest['sizes'][name]:
                raise OSError('feature cache copy truncated')
        return pd.DataFrame(load_json(work / 'index.json')), load_json(out / 'split.json')
    root = find_data_root()
    labels, lm = zip_members(root, work, 'TL_서울_구급', '.json')
    audio, am = zip_members(root, work, 'TS_서울_구급', '.wav')
    try:
        if set(lm) != set(am) or len(lm) != 29200:
            raise ValueError('expected exactly 29,200 paired Training Seoul calls')
        lm, stems = label_call_members(labels, lm)
        split = make_split(lm, out, run_root, work, stems=stems)
        calls = sorted(set(split['S']) | set(split['H']))
        # Persistent per-call chunks live INSIDE one append-only large file.
        # progress.json commits byte offsets; interrupted trailing writes truncate.
        progress_path = cache / 'progress.json'
        progress = load_json(progress_path) if progress_path.exists() else {'calls': [], 'rows': [], 'frames': 0}
        raw = work / 'frames.f16'
        durable = cache / 'frames.f16'
        if progress['frames'] and (not raw.exists() or raw.stat().st_size < progress['frames'] * 40 * 2):
            shutil.copyfile(durable, raw)
        committed = progress['frames'] * 40 * 2
        with raw.open('ab') as stream:
            stream.truncate(committed)
        done = set(progress['calls'])
        unpack = work / 'unpacked'
        unpack.mkdir(exist_ok=True)
        last_save = time.monotonic()
        for call in calls:
            if call in done:
                continue
            stem = stems[call]
            jp, wp = unpack / f'{stem}.json', unpack / f'{stem}.wav'
            jp.write_bytes(labels.read(lm[call]))
            with audio.open(am[stem]) as src, wp.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            row = ManifestRow('Training', stem, wp.name, jp.name)
            records = build_utterance_records(row, unpack)
            waveform, sr = read_call_waveform(wp)
            if sr != 8000 or any(r.call_id != call for r in records):
                raise ValueError(f'unexpected sample rate/call_id: {call}')
            specs, rows = [], []
            offset = progress['frames']
            for r in records:
                start, end = utterance_frame_range(r.start_at, r.end_at, sr, len(waveform))
                spec = mel_spectrogram(slice_and_pad(waveform, start, end, N_FFT), sr).T.astype(np.float16)
                specs.append(spec)
                rows.append({'call_id': call, 'utt_idx': r.utt_idx, 'speaker': r.speaker,
                             'duration_sec': r.duration_sec, 'frame_offset': offset, 'n_frames': len(spec)})
                offset += len(spec)
            with raw.open('ab') as stream:
                for spec in specs:
                    stream.write(spec.tobytes())
            progress['frames'] = offset
            progress['rows'].extend(rows)
            progress['calls'].append(call)
            jp.unlink()
            wp.unlink()
            # Completed calls are committed as packed chunks every ~15 minutes.
            if time.monotonic() - last_save >= 900:
                publish(raw, durable)
                write_json(progress, progress_path, work)
                last_save = time.monotonic()
                print(f'features {len(progress["calls"])}/{len(calls)}', flush=True)
        publish(raw, durable)
        write_json(progress, progress_path, work)
        packed = np.lib.format.open_memmap(work / 'packed.npy', mode='w+', dtype=np.float16,
                                          shape=(progress['frames'], 40))
        source = np.memmap(raw, mode='r', dtype=np.float16, shape=packed.shape)
        for start in range(0, len(packed), 100000):
            packed[start:start + 100000] = source[start:start + 100000]
        packed.flush()
        del packed, source
        write_json(progress['rows'], work / 'index.json', work)
        for name in ('packed.npy', 'index.json'):
            publish(work / name, cache / name)
        write_json({'sizes': {n: (work / n).stat().st_size for n in ('packed.npy', 'index.json')}},
                   cache / 'done.json', work)
        return pd.DataFrame(progress['rows']), split
    finally:
        labels.close()
        audio.close()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def collate_eval(batch):
    padded = [(nn.functional.pad(x, (0, max(0, 300 - x.shape[1])), value=SPEC_FLOOR_DB), y)
              for x, y in batch]
    return collate_batch(padded)


def loader(df, work, *, evaluation=True, order=None):
    ds = PackedSpectrogramDataset(df, work / 'packed.npy')
    if order is not None:
        ds = torch.utils.data.Subset(ds, order)
    return torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0,
                                      pin_memory=True, collate_fn=collate_eval if evaluation else collate_batch)


@torch.no_grad()
def predict(model, batches, device):
    model.eval()
    result = []
    for x, _ in batches:
        with torch.autocast('cuda', enabled=device == 'cuda', dtype=torch.float16):
            logits = model(x.to(device, non_blocking=True))
        result.append(logits.float().softmax(1)[:, 1].cpu().numpy())
    return np.concatenate(result)


def train(arm, seed, train_df, dev_df, h_df, out, work, split):
    name = f'{arm}{seed}'
    directory = out / name
    directory.mkdir(exist_ok=True)
    done_path = out / f'{name}_done.json'
    probs_path = directory / 'holdout_probs.npy'
    if done_path.exists():
        return np.load(probs_path), load_json(done_path)
    seed_all(seed)
    device = 'cuda'
    model = SpeakerCNNLarge().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=2)
    scaler = torch.amp.GradScaler('cuda')
    dev_loader = loader(dev_df, work)
    state = {'epoch': 1, 'step': 0, 'order': None, 'loss_sum': 0., 'seen': 0,
             'best': -1., 'best_epoch': 0, 'history': [], 'collapsed': False}
    signature = {'exp_id': EXP_ID, 'arm': arm, 'seed': seed, 'split': split['sha256'],
                 'alpha': .2, 'batch_size': 64, 'padding_train': 'batch_max', 'padding_eval': 300}
    last = directory / 'ckpt_last.pt'
    if last.exists():
        ckpt = torch.load(last, map_location='cpu', weights_only=False)
        if ckpt['signature'] != signature:
            raise ValueError('resume configuration mismatch')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        move_optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        state = ckpt['state']
        restore_rng(ckpt['rng'])
    def checkpoint():
        save_torch({'signature': signature, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                    'state': state, 'rng': rng_state()}, last, work)
    majority = float(dev_df.speaker.value_counts(normalize=True).max())
    while state['epoch'] <= 30 and state['epoch'] - 1 - state['best_epoch'] < 5:
        if state['order'] is None:
            state['order'] = torch.randperm(len(train_df)).tolist()
        # Creating a loader consumes torch RNG. Isolate that draw so resume
        # cannot change dropout/Mixup RNG at the next step.
        saved_rng = torch.get_rng_state()
        batches = iter(loader(train_df, work, evaluation=False, order=state['order'][state['step'] * 64:]))
        torch.set_rng_state(saved_rng)
        model.train()
        last_save = time.monotonic()
        for x, y in batches:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast('cuda', dtype=torch.float16):
                if arm == 'B':
                    lam = float(np.random.beta(.2, .2))
                    perm = torch.randperm(len(y), device=device)
                    logits = model(lam * x + (1 - lam) * x[perm])
                    loss = lam * nn.functional.cross_entropy(logits, y) + (1 - lam) * nn.functional.cross_entropy(logits, y[perm])
                else:
                    loss = nn.functional.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            state['loss_sum'] += float(loss.detach()) * len(y)
            state['seen'] += len(y)
            state['step'] += 1
            if time.monotonic() - last_save >= 900:
                checkpoint()
                last_save = time.monotonic()
        accuracy = float(((predict(model, dev_loader, device) >= .5) == dev_df.speaker.to_numpy()).mean())
        scheduler.step(accuracy)
        epoch = state['epoch']
        state['collapsed'] |= accuracy <= majority
        state['history'].append({'epoch': epoch, 'train_loss': state['loss_sum'] / state['seen'],
                                 'D_accuracy': accuracy, 'lr': optimizer.param_groups[0]['lr']})
        if accuracy > state['best']:
            state['best'], state['best_epoch'] = accuracy, epoch
            metadata = build_checkpoint_metadata(model_name='large', model_class='SpeakerCNNLarge',
                index_provenance={'split': 'Training', 'label_source': 'provided_json', 'n_mels': 40,
                                  'hop_length': HOP_LENGTH, 'n_fft': N_FFT, 'sample_rate': 8000},
                split_seed=42, test_size=1500 / 29200, torch_seed=seed)
            save_torch({'best_model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'metadata': metadata, 'best_epoch': epoch}, directory / 'best.ckpt', work)
        state.update(epoch=epoch + 1, step=0, order=None, loss_sum=0., seen=0)
        checkpoint()
        print(f'{name} epoch={epoch} D={accuracy:.5f}', flush=True)
    best = torch.load(directory / 'best.ckpt', map_location='cpu', weights_only=True)
    model.load_state_dict(best['best_model_state_dict'])
    probs = predict(model, loader(h_df, work), device)
    np.save(work / 'probs.npy', probs)
    publish(work / 'probs.npy', probs_path)
    result = {'best_epoch': state['best_epoch'], 'stopped_epoch': state['epoch'] - 1,
              'D_best_accuracy': state['best'], 'D_majority_fraction': majority,
              'collapsed': state['collapsed'], 'history': state['history']}
    write_json(result, done_path, work)
    return probs, result


def scores(p, y, durations, groups):
    pred = (p >= .5).astype(int)
    correct = pred == y
    cm = np.zeros((2, 2), dtype=int)
    np.add.at(cm, (y, pred), 1)
    confidence = np.maximum(p, 1 - p)
    bins = np.minimum((confidence * 15).astype(int), 14)
    ece = sum(float(np.abs(correct[bins == i].mean() - confidence[bins == i].mean())) *
              float((bins == i).mean()) for i in range(15) if np.any(bins == i))
    duration_masks = {'<1': durations < 1, '1~3': (durations >= 1) & (durations < 3),
                      '3~6': (durations >= 3) & (durations <= 6), '>6': durations > 6}
    return {'accuracy': float(correct.mean()), 'confusion_matrix': cm.tolist(),
            'recall': {name: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None
                       for i, name in enumerate(('대원', '신고자'))},
            'group_accuracy': [float(correct[groups == i].mean()) for i in range(5)],
            'duration_accuracy': {k: {'rows': int(m.sum()), 'accuracy': float(correct[m].mean()) if m.any() else None}
                                  for k, m in duration_masks.items()},
            'ECE_15': ece, 'Brier': float(np.mean(2 * (p - y) ** 2))}


def summarize(probabilities, records, df, split):
    y = df.speaker.to_numpy(dtype=int)
    calls = sorted(split['H'])
    random.Random(42).shuffle(calls)
    group_map = {call: i for i, chunk in enumerate(np.array_split(calls, 5)) for call in chunk}
    groups = df.call_id.map(group_map).to_numpy()
    details = {k: dict(scores(p, y, df.duration_sec.to_numpy(), groups), **records[k])
               for k, p in probabilities.items()}
    means = {a: (probabilities[a + '1'] + probabilities[a + '2']) / 2 for a in 'AB'}
    ensembles = {a: scores(p, y, df.duration_sec.to_numpy(), groups) for a, p in means.items()}
    ca, cb = (means['A'] >= .5) == y, (means['B'] >= .5) == y
    table = [[int((ca & cb).sum()), int((ca & ~cb).sum())],
             [int((~ca & cb).sum()), int((~ca & ~cb).sum())]]
    b, c = table[0][1], table[1][0]
    pvalue = float(binomtest(b, b + c, .5).pvalue) if b + c else 1.
    deltas = [details[f'B{s}']['accuracy'] - details[f'A{s}']['accuracy'] for s in (1, 2)]
    wins = sum(b > a for a, b in zip(ensembles['A']['group_accuracy'], ensembles['B']['group_accuracy']))
    conditions = [float(np.mean(deltas)) >= .003, all(d > 0 for d in deltas), pvalue < .05, wins >= 4]
    return {'models': details, 'ensembles': ensembles, 'mean_seed_accuracy_delta_B_minus_A': float(np.mean(deltas)),
            'McNemar': {'table_A_correct_rows_B_correct_columns': table, 'exact_two_sided_p': pvalue},
            'group_B_wins': wins, 'criteria_pass': conditions,
            'success': all(conditions) and not any(r['collapsed'] for r in records.values()),
            'failed_training_collapse': any(r['collapsed'] for r in records.values()),
            'split_sha256': split['sha256'], 'baseline_equivalence_verified': split['baseline_equivalence_verified'],
            'label_source': 'provided_json', 'split_seed': 42, 'alpha': .2, 'Brier_definition': 'sum_squared_error_over_two_classes'}


def smoke(out, work):
    """Synthetic forward-only contract check; never trains or reads contest data."""
    torch.set_num_threads(1)
    seed_all(1)
    x, y = collate_eval([(torch.full((40, 12), -40.), 0), (torch.full((40, 18), -30.), 1)])
    model = SpeakerCNNLarge().eval()
    lam, perm = .2, torch.tensor([1, 0])
    with torch.no_grad():
        logits = model(lam * x + (1 - lam) * x[perm])
        loss = lam * nn.functional.cross_entropy(logits, y) + (1 - lam) * nn.functional.cross_entropy(logits, y[perm])
    assert x.shape == (2, 1, 40, 300) and torch.isfinite(loss)
    save_torch({'best_model_state_dict': model.state_dict(), 'smoke_only': True}, out / 'best.ckpt', work)
    write_json({'exp_id': EXP_ID, 'primary_metric': {'name': 'accuracy', 'value': None, 'split': 'internal_holdout'},
                'metrics': {'smoke_passed': True}, 'seed': [1, 2], 'train_rows': 0, 'eval_rows': 0,
                'official_val': None, 'ckpt': 'best.ckpt', 'notes': 'synthetic forward-only smoke; no experiment score'},
               out / 'metrics.json', work)
    print('smoke passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=os.environ.get('LAB_OUT'))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.output is None:
        parser.error('--output or LAB_OUT required')
    out = args.output.resolve()
    work = Path(os.environ.get('LAB_WORK', '/tmp/m2-mix-001')) / EXP_ID
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        smoke(out, work)
        return
    if os.environ.get('LAB_EXP_ID', EXP_ID) != EXP_ID:
        raise ValueError('LAB_EXP_ID mismatch')
    if not torch.cuda.is_available():
        raise RuntimeError('T4/CUDA required; no data preparation performed')
    if 'DCC_RUN_ROOT' not in os.environ or 'LAB_WORK' not in os.environ:
        raise RuntimeError('DCC_RUN_ROOT and LAB_WORK required')
    if (out / 'metrics.json').exists() and not load_json(out / 'metrics.json').get('metrics', {}).get('smoke_passed'):
        print('completed experiment; skipped')
        return
    df, split = prepare(out, work, Path(os.environ['DCC_RUN_ROOT']))
    h = df[df.call_id.isin(split['H'])].sort_values(['call_id', 'utt_idx']).reset_index(drop=True)
    dev = df[df.call_id.isin(split['D'])].reset_index(drop=True)
    training = df[df.call_id.isin(set(split['S']) - set(split['D']))].reset_index(drop=True)
    probabilities, records = {}, {}
    for seed in (1, 2):
        for arm in 'AB':
            probabilities[f'{arm}{seed}'], records[f'{arm}{seed}'] = train(arm, seed, training, dev, h, out, work, split)
        directory = out / f'seed_{seed}'
        directory.mkdir(exist_ok=True)
        np.save(work / 'holdout_probs.npy', np.column_stack([probabilities[f'A{seed}'], probabilities[f'B{seed}']]))
        publish(work / 'holdout_probs.npy', directory / 'holdout_probs.npy')
        write_json(['A_신고자_probability', 'B_신고자_probability'], directory / 'holdout_probs_columns.json', work)
        (work / 'val_files.txt').write_text(''.join(f'{r.call_id}\t{r.utt_idx}\n' for r in h.itertuples()), encoding='utf-8')
        publish(work / 'val_files.txt', directory / 'val_files.txt')
    shutil.copyfile(out / 'B1/best.ckpt', work / 'best_local.ckpt')
    publish(work / 'best_local.ckpt', out / 'best.ckpt')
    metrics = summarize(probabilities, records, h, split)
    metrics.update(train_calls=len(set(training.call_id)), train_utterances=len(training),
                   S_calls=6000, D_calls=300, D_utterances=len(dev), H_calls=1500, H_utterances=len(h))
    value = metrics['mean_seed_accuracy_delta_B_minus_A']
    write_json({'exp_id': EXP_ID, 'primary_metric': {'name': 'accuracy_delta_B_minus_A', 'value': value, 'split': 'internal_holdout'},
                'metrics': metrics, 'seed': [1, 2], 'train_rows': len(training), 'eval_rows': len(h),
                'official_val': None, 'ckpt': 'best.ckpt',
                'notes': 'D-only epoch selection. Primary comparison: mean B-A. ' +
                         ('Prior split reused.' if split['baseline_equivalence_verified'] else
                          'Fallback split: equivalence to M2-SSL-002 unverified; compare A/B within this run.')},
               out / 'metrics.json', work)


if __name__ == '__main__':
    main()
