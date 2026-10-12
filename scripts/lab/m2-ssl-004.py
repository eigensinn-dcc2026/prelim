#!/usr/bin/env python3
"""M2-SSL-004: Training-only WavLM; run from repository root with PYTHONPATH=src.

All call relationships are confined to sampling/splitting and post-hoc analysis.
The model sees only individually normalized target-utterance windows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import unicodedata
import zipfile

EXP_ID = 'M2-SSL-004'
REVISION = 'c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c'
SR, WIN = 16000, 48000


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def publish(local, target):
    """Stage locally, copy to Drive temporary name, verify size, atomically replace."""
    target = Path(target)
    for attempt in range(6):
        try:
            shutil.copyfile(local, str(target) + '.tmp')
            if os.path.getsize(str(target) + '.tmp') != os.path.getsize(local):
                raise OSError('incomplete artifact copy')
            os.replace(str(target) + '.tmp', target)
            return
        except OSError:
            if attempt == 5:
                raise
            time.sleep(2 ** (attempt + 1))


def write_json(path, value):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(temp, path)


def prepare_model(run_root):
    from huggingface_hub import snapshot_download
    from transformers import WavLMModel
    path = run_root / 'models/wavlm-large'
    marker = path / 'download_complete.json'
    if not marker.exists():
        snapshot_download('microsoft/wavlm-large', revision=REVISION, local_dir=str(path))
        write_json(marker, {'revision': REVISION})
    if json.loads(marker.read_text())['revision'] != REVISION:
        raise ValueError('local model revision mismatch')
    backbone = WavLMModel.from_pretrained(str(path), local_files_only=True)
    backbone.config.layerdrop = 0.0
    return backbone


def unpack(source, work):
    work.mkdir(parents=True, exist_ok=True)
    local = work / source.name
    destination = work / source.stem
    marker = destination / '.complete'
    if not marker.exists():
        if not local.exists() or local.stat().st_size != source.stat().st_size:
            shutil.copyfile(source, str(local) + '.tmp')
            os.replace(str(local) + '.tmp', local)
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(local) as archive:
            for item in archive.infolist():
                resolved = (destination / item.filename).resolve()
                if not resolved.is_relative_to(destination.resolve()):
                    raise ValueError('unsafe zip member')
            archive.extractall(destination)
        marker.touch()
    return destination


def stem_key(path):
    return unicodedata.normalize('NFC', Path(path).stem).casefold()


def prepare_rows(work):
    from dcc_data.colab_env import find_data_root
    source = find_data_root() / 'Training'
    labels = unpack(source / 'TL_서울_구급.zip', work)
    audio = unpack(source / 'TS_서울_구급.zip', work)
    wavs = {}
    for path in audio.rglob('*.wav'):  # Local extracted files only.
        key = stem_key(path)
        if key in wavs:
            raise ValueError('duplicate wav stem')
        wavs[key] = str(path)
    rows, seen = [], set()
    for path in sorted(labels.rglob('*.json')):
        document = json.loads(path.read_text(encoding='utf-8'))
        call = str(document.get('call_id') or document.get('_id') or stem_key(path))
        if call in seen:
            raise ValueError('duplicate call_id')
        seen.add(call)
        utterances = document['utterances']
        for idx, utt in enumerate(utterances):
            start, end = utt['startAt'], utt['endAt']
            if end <= start:
                continue
            if utt['speaker'] not in (0, 1) or start < 0:
                raise ValueError(f'invalid provided label: {path}:{idx}')
            overlap = any(j != idx and other['startAt'] < end and other['endAt'] > start
                          for j, other in enumerate(utterances))
            rows.append(dict(call_id=call, utt_idx=idx, startAt=start, endAt=end,
                             speaker=utt['speaker'], wav=wavs[stem_key(path)], overlap=overlap))
    if not rows:
        raise ValueError('no valid Training utterances')
    return rows


def make_split(rows, out, run):
    source = run / 'lab/M2-SSL-003/split.json'
    split = json.loads(source.read_text())  # Missing baseline split must fail.
    if digest({k: v for k, v in split.items() if k != 'sha256'}) != split['sha256']:
        raise ValueError('SSL-003 split hash mismatch')
    calls = sorted({r['call_id'] for r in rows})
    if sorted(split['train'] + split['D'] + split['H']) != calls:
        raise ValueError('Training calls differ from SSL-003 split')
    if len(split['H']) != 1500 or len(split['D']) != 1000:
        raise ValueError('unexpected baseline holdout sizes')
    target = out / 'split.json'
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise ValueError('persisted split differs from SSL-003')
    shutil.copyfile(source, str(target) + '.tmp')
    os.replace(str(target) + '.tmp', target)
    return split


def window_starts(length):
    if length <= WIN:
        return [0]
    starts = list(range(0, length - WIN + 1, SR * 3 // 2))
    if starts[-1] != length - WIN:
        starts.append(length - WIN)
    if len(starts) > 8:
        starts = [starts[round(i * (len(starts) - 1) / 7)] for i in range(8)]
    return starts


def load_segment(row, target_sr=SR):
    import torch
    from torchaudio.functional import resample
    from dcc_data.audio import read_wav_info, read_wav_frames
    info = read_wav_info(row['wav'])
    start = round(row['startAt'] * info.sample_rate / 1000)
    end = round(row['endAt'] * info.sample_rate / 1000)
    start = min(start, info.frame_count)
    end = min(end, info.frame_count)
    if end <= start:
        return torch.zeros(1)
    segment = read_wav_frames(row['wav'], start, end)
    signal = torch.from_numpy(segment.waveform).mean(0)
    sr = segment.sample_rate
    if sr != target_sr:
        signal = resample(signal, sr, target_sr)
    return signal


def audio_preflight():
    """Exercise the actual segment reader and resampler before archive preparation."""
    import struct
    import tempfile
    import wave
    import torch

    with tempfile.TemporaryDirectory(prefix='m2-ssl-003-audio-') as temp:
        path = Path(temp) / 'sample.wav'
        # Stereo 8 kHz PCM; only the middle 100 ms is the target utterance.
        with wave.open(str(path), 'wb') as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes(b''.join(struct.pack('<hh', 8192, 0) if 400 <= i < 1200
                                    else struct.pack('<hh', -16384, -16384)
                                    for i in range(1600)))
        row = dict(wav=str(path), startAt=50, endAt=150)
        original = load_segment(row, target_sr=8000)
        if original.shape != (800,) or not torch.all(original == .125).item():
            raise RuntimeError('audio preflight failed: segment boundaries or mono conversion')
        signal = load_segment(row)
        if signal.shape != (1600,) or not torch.isfinite(signal).all().item():
            raise RuntimeError('audio preflight failed: 8 kHz to 16 kHz resampling')
        clipped = load_segment(dict(row, startAt=150, endAt=250), target_sr=8000)
        if clipped.shape != (400,) or not torch.all(clipped == -.5).item():
            raise RuntimeError('audio preflight failed: end-of-file clipping')
        empty = load_segment(dict(row, startAt=250, endAt=300))
        if empty.shape != (1,) or empty.item() != 0:
            raise RuntimeError('audio preflight failed: empty segment')
    print('audio preflight OK: PCM segment, stereo mono, clipping, resampling', flush=True)


def normalize(x):
    std = x.std(unbiased=False)
    return (x - x.mean()) / std if std >= 1e-5 else x * 0


def fit_length(x, rng=None, start=None):
    if len(x) < WIN:
        x = x.repeat(math.ceil(WIN / max(len(x), 1)))[:WIN]
    elif len(x) > WIN:
        offset = start if start is not None else (rng.randrange(len(x)-WIN+1) if rng else (len(x)-WIN)//2)
        x = x[offset:offset+WIN]
    return normalize(x)


def build_model(backbone):
    import torch
    from torch import nn
    import torch.nn.functional as F

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.layer_weights = nn.Parameter(torch.zeros(backbone.config.num_hidden_layers + 1))
            self.embedding = nn.Sequential(nn.Linear(backbone.config.hidden_size * 2, 512), nn.ReLU(), nn.Dropout(.1), nn.Linear(512, 256))
            self.classifier = nn.Linear(256, 1)
            self.backbone.feature_extractor._freeze_parameters()

        def forward(self, waveform):
            states = self.backbone(waveform, output_hidden_states=True).hidden_states
            if len(states) != len(self.layer_weights):
                raise ValueError('unexpected hidden state count')
            pooled = sum(weight * state for weight, state in zip(self.layer_weights.softmax(0), states))
            emb = F.normalize(self.embedding(torch.cat([pooled.mean(1), pooled.std(1, unbiased=False)], 1)), dim=1)
            return self.classifier(emb).squeeze(1), emb
    return Classifier()


def supcon(embedding, labels):
    import torch
    n = len(labels)
    mask = ~torch.eye(n, dtype=torch.bool, device=labels.device)
    positive = (labels[:, None] == labels[None, :]) & mask
    logits = embedding.float() @ embedding.float().T / .07
    log_prob = logits - torch.logsumexp(logits.masked_fill(~mask, -torch.inf), 1, keepdim=True)
    counts = positive.sum(1)
    loss = -(log_prob.masked_fill(~positive, 0).sum(1) / counts.clamp_min(1))
    return loss[counts > 0].mean() if (counts > 0).any() else embedding.sum() * 0


def epoch_order(rows, epoch):
    groups = {}
    for idx, row in enumerate(rows):
        groups.setdefault(row['call_id'], []).append(idx)
    rng = random.Random(42 + epoch)
    calls = sorted(groups)
    rng.shuffle(calls)
    order = []
    for call in calls:
        pool = groups[call]
        order.extend(rng.sample(pool, 4) if len(pool) >= 4 else rng.choices(pool, k=4))
    return order


# Top-level class supports DataLoader spawn and deterministic worker-independent augmentation.
class Windows:
    def __init__(self, rows, order, epoch):
        self.rows, self.order, self.epoch = rows, order, epoch
        self.by_label = {label: [i for i, r in enumerate(rows) if r['speaker'] == label] for label in (0, 1)}

    def __len__(self):
        return len(self.order)

    def __getitem__(self, position):
        idx = self.order[position]
        row = self.rows[idx]
        rng = random.Random(42 + self.epoch * 100000000 + position)
        x = fit_length(load_segment(row), rng)
        if rng.random() < .15:
            pool = self.by_label[1-row['speaker']]
            if pool:
                # Rejection sampling uses no same-call waveform-derived feature.
                for _ in range(100):
                    other = self.rows[rng.choice(pool)]
                    if other['call_id'] != row['call_id']:
                        y = fit_length(load_segment(other), rng)
                        size = int(WIN * rng.uniform(.1, .25))
                        start = 0 if rng.random() < .5 else WIN-size
                        x = x.clone()
                        x[start:start+size] += rng.uniform(.15, .40) * y[start:start+size]
                        x = normalize(x)
                        break
        return x, row['speaker']


def rng_state():
    import torch
    import numpy as np
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    import torch
    import numpy as np
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def save_torch(value, name, work, out):
    import torch
    local = work / name
    torch.save(value, str(local) + '.tmp')
    os.replace(str(local) + '.tmp', local)
    publish(local, out / name)


def load_torch(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def evaluate(model, rows, device, batch_windows=32):
    import numpy as np
    import torch
    model.eval()
    probs = []
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode():
        for row in rows:
            x = load_segment(row)
            windows = torch.stack([fit_length(x, start=s) for s in window_starts(len(x))])
            logits = []
            for batch in windows.split(batch_windows):
                with torch.autocast(device_type='cuda', enabled=device.startswith('cuda')):
                    logits.append(model(batch.to(device))[0].float())
            probs.append(torch.cat(logits).mean().sigmoid().item())
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    return np.asarray(probs), time.monotonic()-started


def train(model, rows, d_rows, out, work, device, split_sha, epochs, micro_batch):
    import torch
    import numpy as np
    from transformers import get_cosine_schedule_with_warmup
    from torch.utils.data import DataLoader
    model.to(device)
    backbone = list(model.backbone.parameters())
    head = [p for name, p in model.named_parameters() if not name.startswith('backbone.')]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': 3e-5}, {'params': head, 'lr': 3e-4}], weight_decay=.01)
    steps = math.ceil(len({r['call_id'] for r in rows}) / 8)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(steps*epochs*.05), steps*epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=device.startswith('cuda'))
    epoch, step, best, history, loss_sum, loss_count, epoch_seconds = 0, 0, -1., [], 0., 0, 0.
    resume = None
    if (out / 'ckpt_last.pt').exists():
        shutil.copyfile(out / 'ckpt_last.pt', work / 'resume.pt')
        resume = load_torch(work / 'resume.pt')
        if resume.get('epochs') != epochs or resume.get('revision') != REVISION:
            raise ValueError('resume epoch schedule or revision mismatch')
        micro_batch = resume['micro_batch_size']
        if resume['split_sha256'] != split_sha:
            raise ValueError('checkpoint split mismatch')
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(resume['scheduler'])
        scaler.load_state_dict(resume['scaler'])
        epoch, step, best, history = resume['epoch'], resume['step'], resume['best'], resume['history']
        loss_sum, loss_count, epoch_seconds = resume['loss_sum'], resume['loss_count'], resume['epoch_seconds']
        restore_rng(resume['rng'])
        for key in ('model', 'optimizer', 'scheduler', 'scaler', 'rng'):
            del resume[key]
    for ep in range(epoch, epochs):
        for p in model.backbone.parameters():
            p.requires_grad_(True)
        for p in model.backbone.encoder.parameters():
            p.requires_grad_(ep > 0)
        model.backbone.feature_extractor._freeze_parameters()
        order = resume['sample_order'] if resume is not None and ep == epoch and resume['sample_order'] else epoch_order(rows, ep)
        dataset = Windows(rows, order, ep)
        # A dedicated generator prevents iterator construction consuming model RNG on resume.
        loader = DataLoader(dataset, batch_size=32, sampler=range(step*32, len(order)),
                            num_workers=os.cpu_count() or 1, pin_memory=True,
                            generator=torch.Generator().manual_seed(42+ep))
        model.train()
        started = last_save = time.monotonic()
        def checkpoint(next_step, next_epoch=ep):
            save_torch(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                            epoch=next_epoch, step=next_step, sampler_epoch=next_epoch,
                            sample_order=order if next_epoch == ep else [], rng=rng_state(),
                            best=best, history=history, split_sha256=split_sha,
                            epochs=epochs, revision=REVISION, micro_batch_size=micro_batch, effective_batch_size=32,
                            loss_sum=loss_sum, loss_count=loss_count,
                            epoch_seconds=epoch_seconds + time.monotonic()-started if next_epoch == ep else 0.),
                       'ckpt_last.pt', work, out)
        for batch_step, (waveform, labels) in enumerate(loader, start=step):
            waveform, labels = waveform.to(device), labels.to(device)
            batch_rng = rng_state()
            while True:
                optimizer.zero_grad(set_to_none=True)
                try:
                    with torch.autocast(device_type='cuda', enabled=device.startswith('cuda')):
                        if micro_batch == 32:
                            logits, embedding = model(waveform)
                        else:
                            # Activation recomputation keeps global-batch SupCon identical,
                            # unlike averaging independent micro-batch contrastive losses.
                            from torch.utils.checkpoint import checkpoint as recompute
                            pieces = [recompute(model, x, use_reentrant=False, preserve_rng_state=True)
                                      for x in waveform.split(micro_batch)]
                            logits = torch.cat([x[0] for x in pieces])
                            embedding = torch.cat([x[1] for x in pieces])
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels.float()) + .5 * supcon(embedding, labels)
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite training loss')
                    scaler.scale(loss).backward()
                    break
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    if micro_batch == 1:
                        raise
                    logits = embedding = loss = None
                    pieces = []
                    torch.cuda.empty_cache()
                    restore_rng(batch_rng)
                    micro_batch //= 2
                    print(f'OOM: micro batch {micro_batch}, effective batch 32 (global SupCon)', flush=True)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            loss_sum += loss.item() * len(labels)
            loss_count += len(labels)
            if time.monotonic()-last_save >= 900:
                checkpoint(batch_step+1)
                last_save = time.monotonic()
        # Persist completed training steps before evaluation, which may also be interrupted.
        checkpoint(steps)
        probabilities, _ = evaluate(model, d_rows, device)
        accuracy = float(np.mean((probabilities >= .5) == np.array([r['speaker'] for r in d_rows])))
        minutes = (epoch_seconds + time.monotonic()-started)/60
        targets = np.array([r['speaker'] for r in d_rows])
        clipped = np.clip(probabilities, 1e-7, 1-1e-7)
        d_loss = float(-(targets*np.log(clipped)+(1-targets)*np.log1p(-clipped)).mean())
        history.append(dict(epoch=ep+1, d_accuracy=accuracy, d_loss=d_loss,
                            train_loss=loss_sum/max(loss_count, 1), minutes=minutes,
                            micro_batch_size=micro_batch, effective_batch_size=32))
        print(json.dumps(history[-1]), flush=True)
        if ep == 0:
            print(f'{epochs} epoch estimate: {minutes*epochs/60:.2f} hours', flush=True)
        if accuracy > best:
            best = accuracy
            weights = {k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu() for k, v in model.state_dict().items()}
            save_torch(dict(model_state_dict=weights, best_epoch=ep+1, d_accuracy=best,
                            emb_dim=256, win_sec=3., sr=SR, revision=REVISION, layers=model.backbone.config.num_hidden_layers + 1,
                            hidden_size=model.backbone.config.hidden_size,
                            architecture='wavlm-large-weighted-mean-std', label_source='provided_json',
                            split_sha256=split_sha, inference=dict(stride_sec=1.5, max_windows=8, aggregation='mean_logit', threshold=.5)),
                       'best.ckpt', work, out)
        step, loss_sum, loss_count, epoch_seconds = 0, 0., 0, 0.
        checkpoint(0, ep+1)
        resume = None
    write_json(out / 'train_done.json', dict(history=history, best_d_accuracy=best, epochs=epochs, micro_batch_size=micro_batch, effective_batch_size=32))


def summaries(rows, probs):
    import numpy as np
    from sklearn.metrics import confusion_matrix
    y = np.array([r['speaker'] for r in rows])
    predicted = probs >= .5
    correct = predicted == y
    durations = np.array([(r['endAt']-r['startAt'])/1000 for r in rows])
    def subgroup(mask):
        return dict(rows=int(mask.sum()), accuracy=float(correct[mask].mean()) if mask.any() else None)
    overlap = np.array([r['overlap'] for r in rows], dtype=bool)
    confidence = np.maximum(probs, 1-probs)
    ece = 0.
    for idx in range(15):
        mask = (confidence >= idx/15) & (confidence < (idx+1)/15 if idx < 14 else confidence <= 1)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    cm = confusion_matrix(y, predicted, labels=[0, 1])
    return dict(accuracy=float(correct.mean()), recall={'대원': float(cm[0, 0]/max(cm[0].sum(), 1)),
                '신고자': float(cm[1, 1]/max(cm[1].sum(), 1))}, confusion_matrix=cm.tolist(),
                duration_accuracy={name: subgroup(mask) for name, mask in [
                    ('<0.5s', durations < .5), ('0.5~1s', (durations >= .5)&(durations < 1)),
                    ('1~3s', (durations >= 1)&(durations <= 3)), ('>3s', durations > 3)]},
                overlap_accuracy={'overlap': subgroup(overlap), 'no_overlap': subgroup(~overlap)},
                ece_15=float(ece), brier=float(((probs-y)**2).mean()))


def row_key(row):
    return f"{row['call_id']}\t{row['utt_idx']}\t{row['startAt']}\t{row['endAt']}"


def align_baseline_rows(rows, run):
    path = run / 'lab/M2-SSL-003/H_rows.txt'
    if not path.exists():
        return rows
    keys = path.read_text().splitlines()
    lookup = {row_key(r): r for r in rows}
    if len(keys) != len(rows) or len(set(keys)) != len(keys) or set(keys) != set(lookup):
        raise ValueError('SSL-003 H row identities differ')
    return [lookup[key] for key in keys]


def cnn_diagnostics(rows, large, run, split, work, out, device):
    import numpy as np
    from scipy.stats import binomtest
    root = run / 'lab/M2-SSL-003'
    result = dict(skipped_reason=None, cnn_split_suspect=False)
    methods = {'large': large}
    if not all((root / name).exists() for name in ('H_rows.txt', 'H_probs.npy')):
        result['skipped_reason'] = 'SSL-003 H_rows.txt or H_probs.npy missing'
    else:
        keys = (root / 'H_rows.txt').read_text().splitlines()
        if keys != [row_key(r) for r in rows]:
            raise ValueError('baseline row order mismatch')
        probs = np.load(root / 'H_probs.npy')
        if probs.ndim != 2 or probs.shape[0] != len(rows) or not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
            raise ValueError('invalid baseline probabilities')
        methods['base'] = probs[:, 0]
        methods['base+large'] = (probs[:, 0] + large) / 2
        if probs.shape[1] > 1:
            cnn = probs[:, 1:].mean(1)
            methods.update(CNN=cnn, **{'large+CNN': (large+cnn)/2,
                                      'base+large+CNN': (probs[:, 0]+large+cnn)/3})
        else:
            result['skipped_reason'] = 'SSL-003 CNN probability columns missing; base comparisons available'
    y = np.array([r['speaker'] for r in rows])
    result['accuracies'] = {k: float(((v >= .5) == y).mean()) for k, v in methods.items()}
    if 'base' in methods:
        a, b = (large >= .5) == y, (methods['base'] >= .5) == y
        table = [[int((a & b).sum()), int((a & ~b).sum())],
                 [int((~a & b).sum()), int((~a & ~b).sum())]]
        n = table[0][1] + table[1][0]
        result['mcnemar_large_vs_base'] = dict(table=table, axes='large/base correct/incorrect',
            p_value=float(binomtest(table[0][1], n, .5).pvalue) if n else 1., method='exact_two_sided')
    calls = sorted({r['call_id'] for r in rows})
    np.random.RandomState(42).shuffle(calls)
    result['call_groups'] = []
    for group in np.array_split(calls, 5):
        group_calls = set(group)
        mask = np.array([r['call_id'] in group_calls for r in rows])
        result['call_groups'].append(dict(calls=len(group), rows=int(mask.sum()),
            accuracy={k: float(((v[mask] >= .5) == y[mask]).mean()) for k, v in methods.items()}))
    return result


def persist_npy(value, name, work, out):
    import numpy as np
    np.save(work / name, value)
    publish(work / name, out / name)


def smoke():
    """No model download, training, or competition data access."""
    import importlib.util
    import tempfile
    if all(importlib.util.find_spec(name) is not None for name in ('torch', 'torchaudio', 'numpy')):
        audio_preflight()
    else:
        print('audio preflight SKIPPED: torch/torchaudio/numpy unavailable; required on normal startup')
    assert window_starts(1) == [0]
    assert window_starts(WIN) == [0]
    assert window_starts(WIN+1) == [0, 1]
    assert stem_key('서울.WAV') == stem_key(unicodedata.normalize('NFD', '서울') + '.wav')
    assert stem_key('Example.wav') == stem_key('EXAMPLE.json')
    starts = window_starts(SR*100)
    assert len(starts) == 8 and starts[0] == 0 and starts[-1] == SR*100-WIN
    rows = [dict(call_id='a'), dict(call_id='b'), dict(call_id='b')]
    order = epoch_order(rows, 0)
    assert order == epoch_order(rows, 0) and len(order) == 8
    assert all(sum(rows[idx]['call_id'] == c for idx in order) == 4 for c in ('a', 'b'))
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        local, target = root/'local', root/'out'
        local.write_bytes(b'artifact')
        publish(local, target)
        assert target.read_bytes() == b'artifact'
        write_json(root/'split.json', {'hash': digest(order)})
        assert json.loads((root/'split.json').read_text())['hash'] == digest(order)
        baseline = root/'lab/M2-SSL-003'
        baseline.mkdir(parents=True)
        output = root/'result'
        output.mkdir()
        fixture = dict(train=['train'], D=[f'd{i}' for i in range(1000)],
                       H=[f'h{i}' for i in range(1500)])
        fixture['sha256'] = digest(fixture)
        write_json(baseline/'split.json', fixture)
        fixture_rows = [dict(call_id=c) for k in ('train', 'D', 'H') for c in fixture[k]]
        assert make_split(fixture_rows, output, root) == fixture
        assert (output/'split.json').read_bytes() == (baseline/'split.json').read_bytes()
        fixture['sha256'] = 'invalid'
        write_json(baseline/'split.json', fixture)
        try:
            make_split(fixture_rows, output, root)
        except ValueError:
            pass
        else:
            raise AssertionError('corrupt baseline split accepted')
        row_fixture = [dict(call_id='h', utt_idx=i, startAt=i, endAt=i+1) for i in range(2)]
        (baseline/'H_rows.txt').write_text('\n'.join(row_key(r) for r in reversed(row_fixture))+'\n')
        assert align_baseline_rows(row_fixture, root) == list(reversed(row_fixture))
        archive = root/'TL_서울_구급.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('TL_서울_구급/example.json', '{}')
        folder = unpack(archive, root/'work')
        assert (folder/'TL_서울_구급/example.json').exists()
        assert unpack(archive, root/'work') == folder
    print('smoke OK: baseline split hash and row alignment, sliding windows, grouped sampler, atomic publication, local zip extraction')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=os.environ.get('LAB_OUT'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--micro-batch-size', type=int, choices=(1, 2, 4, 8, 16, 32), default=32)
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    if args.output is None:
        parser.error('--output or LAB_OUT required')
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    run = Path(os.environ['DCC_RUN_ROOT'])
    if not (run / 'lab/M2-SSL-003/split.json').exists():
        raise FileNotFoundError('required SSL-003 split.json missing')
    # Fail on model/dependency problems before touching data archives.
    backbone = prepare_model(run)
    import torch
    audio_preflight()
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    out, work = args.output, Path(os.environ['LAB_WORK']) / EXP_ID.lower()
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if (out/'metrics.json').exists() and (out/'h_eval_done.json').exists():
        completed = json.loads((out/'metrics.json').read_text())
        if completed['requested_epochs'] != args.epochs or completed['revision'] != REVISION:
            raise ValueError('completed experiment configuration mismatch')
        print('experiment already complete', flush=True)
        return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = prepare_rows(work)
    split = make_split(rows, out, run)
    sets = {name: set(split[name]) for name in ('train', 'D', 'H')}
    assert not (sets['train'] & sets['D'] or sets['train'] & sets['H'] or sets['H'] & sets['D'])
    train_rows = [r for r in rows if r['call_id'] in sets['train']]
    h_rows = align_baseline_rows([r for r in rows if r['call_id'] in sets['H']], run)
    d_all = [r for r in rows if r['call_id'] in sets['D']]
    groups = {}
    for row in d_all:
        groups.setdefault(row['call_id'], []).append(row)
    rng = random.Random(42)
    d_rows = [row for call in sorted(groups) for row in rng.sample(groups[call], min(10, len(groups[call])))]
    h_row_sha = digest([row_key(r) for r in h_rows])
    model = build_model(backbone)
    if not (out/'train_done.json').exists():
        train(model, train_rows, d_rows, out, work, device, split['sha256'], args.epochs, args.micro_batch_size)
    done = json.loads((out/'train_done.json').read_text())
    if done['epochs'] != args.epochs:
        raise ValueError('completed training used a different epoch schedule')
    best = load_torch(out/'best.ckpt')
    if best['revision'] != REVISION:
        raise ValueError('best checkpoint revision mismatch')
    if best['split_sha256'] != split['sha256']:
        raise ValueError('best checkpoint split mismatch')
    model.load_state_dict(best['model_state_dict'])
    model.to(device)
    if (out/'h_eval_done.json').exists():
        eval_done = json.loads((out/'h_eval_done.json').read_text())
        if eval_done.get('h_rows_sha256') != h_row_sha:
            raise ValueError('cached H row order mismatch')
        if eval_done['split_sha256'] != split['sha256']:
            raise ValueError('H evaluation split mismatch')
        probabilities = np.load(out/'wavlm_H_probs.npy')
        if probabilities.shape != (len(h_rows),):
            raise ValueError('H cached row count mismatch')
    else:
        probabilities, seconds = evaluate(model, h_rows, device)
        persist_npy(probabilities, 'wavlm_H_probs.npy', work, out)
        # CPU benchmark uses batch size 1 even for the sliding windows.
        torch.set_num_threads(2)
        sample = np.random.RandomState(42).choice(len(h_rows), min(300, len(h_rows)), replace=False)
        model.to('cpu')
        _, cpu_seconds = evaluate(model, [h_rows[i] for i in sample], 'cpu', batch_windows=1)
        model.to(device)
        eval_done = dict(split_sha256=split['sha256'], h_rows_sha256=h_row_sha, seconds=seconds, device=device,
                         gpu_utterances_per_second=len(h_rows)/seconds if device == 'cuda' else None,
                         cpu_rows=len(sample), cpu_seconds=cpu_seconds,
                         cpu_official_val_estimated_minutes=cpu_seconds/len(sample)*111947/60)
        write_json(out/'h_eval_done.json', eval_done)
    diagnostics = cnn_diagnostics(h_rows, probabilities, run, split, work, out, device)
    write_json(out/'cnn_diagnostics.json', diagnostics)
    persist_npy(probabilities[:, None], 'H_probs.npy', work, out)
    write_json(out/'H_probs_columns.json', ['wavlm_large'])
    local_rows = work/'H_rows.txt'
    local_rows.write_text(''.join(f"{r['call_id']}\t{r['utt_idx']}\t{r['startAt']}\t{r['endAt']}\n" for r in h_rows))
    publish(local_rows, out/'H_rows.txt')
    summary = summaries(h_rows, probabilities)
    majority = max(sum(r['speaker'] == 0 for r in d_rows), sum(r['speaker'] == 1 for r in d_rows))/len(d_rows)
    metrics = dict(exp_id=EXP_ID, primary_metric=dict(name='accuracy', value=summary['accuracy'], split='internal_holdout'),
        metrics=dict(summary, epoch_history=done['history'], best_epoch=best['best_epoch'],
                     best_d_accuracy=best['d_accuracy'], d_majority_accuracy=majority,
                     training_collapse=best['d_accuracy'] <= majority,
                     layer_weights=model.layer_weights.detach().float().softmax(0).cpu().tolist(),
                     inference_timing=eval_done, cnn=diagnostics,
                     counts={k: dict(calls=len(sets[k]), rows=len(v)) for k, v in [('train', train_rows), ('D', d_rows), ('H', h_rows)]},
                     d_all_rows=len(d_all), sampled_train_rows_per_epoch=len(sets['train'])*4),
        seed=42, train_rows=len(train_rows), eval_rows=len(h_rows), official_val=None, ckpt='best.ckpt',
        notes='Training 서울 only; provided speaker labels; H used only for final reporting; no official Validation read',
        label_source='provided_json', split_sha256=split['sha256'], cnn_split_reproduced=split['cnn_split_reproduced'],
        cnn_split_suspect=diagnostics['cnn_split_suspect'], revision=REVISION, actual_epochs=len(done['history']), requested_epochs=args.epochs,
        effective_batch_size=32, micro_batch_size=done['micro_batch_size'],
        gradient_accumulation_steps=32 // done['micro_batch_size'],
        split_file_sha256=hashlib.sha256((out/'split.json').read_bytes()).hexdigest(),
        baseline_split_sha256=split['sha256'],
        epoch_budget_note='SSL-003 epoch timing unavailable: recorded first-batch failure; specification default 10')
    write_json(out/'metrics.json', metrics)
    print(json.dumps(metrics['primary_metric']), flush=True)


if __name__ == '__main__':
    main()
