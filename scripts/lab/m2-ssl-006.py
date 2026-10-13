#!/usr/bin/env python3
"""M2-SSL-006/007/008: continue M2-SSL-003 WavLM-base-plus for 3 epochs; run from repository root with PYTHONPATH=src.

One script, three experiments chosen by --exp-id:
  M2-SSL-006 asp      context-aware attentive statistics pooling replaces mean||std pooling
                      (zero-initialized score, so step 0 equals the M2-SSL-003 forward pass).
  M2-SSL-007 consist  clean-teacher -> overlapped-student consistency; teacher = frozen M2-SSL-003.
  M2-SSL-008 control  the same 3-epoch continuation with the unchanged M2-SSL-003 recipe.

All call relationships are confined to sampling/splitting and post-hoc analysis.
The model sees only individually normalized target-utterance windows. Interfering
utterances for training-time mixing come from other calls with the opposite label.
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

EXP_ID = 'M2-SSL-006'  # set from --exp-id
VARIANT = 'asp'
VARIANTS = {'M2-SSL-006': 'asp', 'M2-SSL-007': 'consist', 'M2-SSL-008': 'control'}
PARENT = 'M2-SSL-003'
REVISION = '4c66d4806a428f2e922ccfa1a962776e232d487b'
SR, WIN = 16000, 48000
# Sampling epochs continue after the parent's 15 so new utterances are drawn per call.
EPOCHS, EPOCH_OFFSET = 3, 15
LR_BACKBONE, LR_HEAD, LR_POOL = 1e-5, 1e-4, 3e-4
CONSIST_MIX_P, CONSIST_KD, CONSIST_COS = .5, .5, .1


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
    path = run_root / 'models/wavlm-base-plus'
    marker = path / 'download_complete.json'
    if not marker.exists():
        snapshot_download('microsoft/wavlm-base-plus', revision=REVISION, local_dir=str(path))
        write_json(marker, {'revision': REVISION})
    if json.loads(marker.read_text())['revision'] != REVISION:
        raise ValueError('local model revision mismatch')
    backbone = WavLMModel.from_pretrained(str(path), local_files_only=True)
    backbone.config.layerdrop = 0.0
    if backbone.config.num_hidden_layers + 1 != 13:
        raise ValueError('expected 13 hidden states')
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


def build_asp(dim, context=5):
    """Attentive mean||std over frames; a zero-initialized output conv gives uniform weights (== mean||std)."""
    import torch
    from torch import nn

    class ASP(nn.Module):
        def __init__(self):
            super().__init__()
            self.score = nn.Sequential(nn.Conv1d(dim, 128, context, padding=context // 2), nn.Tanh(),
                                       nn.Conv1d(128, 1, 1))
            nn.init.zeros_(self.score[2].weight)
            nn.init.zeros_(self.score[2].bias)

        def forward(self, hidden):
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                hidden = hidden.float()
                weights = self.score(hidden.transpose(1, 2)).softmax(-1).transpose(1, 2)
                mean = (weights * hidden).sum(1)
                var = (weights * (hidden - mean[:, None]).pow(2)).sum(1)
                return torch.cat([mean, var.clamp_min(1e-10).sqrt()], 1)
    return ASP()


def build_model(backbone, pool=False):
    import torch
    from torch import nn
    import torch.nn.functional as F

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.layer_weights = nn.Parameter(torch.zeros(13))
            self.pool = build_asp(768) if pool else None
            self.embedding = nn.Sequential(nn.Linear(1536, 512), nn.ReLU(), nn.Dropout(.1), nn.Linear(512, 256))
            self.classifier = nn.Linear(256, 1)
            self.backbone.feature_extractor._freeze_parameters()

        def forward(self, waveform):
            states = self.backbone(waveform, output_hidden_states=True).hidden_states
            if len(states) != 13:
                raise ValueError('unexpected hidden state count')
            pooled = sum(weight * state for weight, state in zip(self.layer_weights.softmax(0), states))
            if self.pool is None:
                stats = torch.cat([pooled.mean(1), pooled.std(1, unbiased=False)], 1)
            else:
                stats = self.pool(pooled)
            emb = F.normalize(self.embedding(stats), dim=1)
            return self.classifier(emb).squeeze(1), emb
    return Classifier()


def load_parent(model, run, split_sha):
    """Load M2-SSL-003 best.ckpt; only the new pooling parameters may be missing."""
    parent = load_torch(run / f'lab/{PARENT}/best.ckpt')
    if parent['split_sha256'] != split_sha:
        raise ValueError('parent checkpoint split mismatch')
    state = {k: v.float() if v.is_floating_point() else v for k, v in parent['model_state_dict'].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    expected_missing = sorted(k for k in model.state_dict() if k.startswith('pool.'))
    if unexpected or sorted(missing) != expected_missing:
        raise ValueError(f'parent state mismatch: missing={missing} unexpected={unexpected}')
    return dict(best_epoch=parent['best_epoch'], d_accuracy=parent['d_accuracy'])


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


class ConsistWindows(Windows):
    """Clean target window plus an overlapped view of it (M2-SSL-007 only).

    The interfering utterance has the opposite provided label and comes from another
    call; nothing from the target's own call is used.
    """

    def __getitem__(self, position):
        idx = self.order[position]
        row = self.rows[idx]
        rng = random.Random(42 + self.epoch * 100000000 + position)
        clean = fit_length(load_segment(row), rng)
        mixed, is_mixed = clean, False
        if rng.random() < CONSIST_MIX_P:
            pool = self.by_label[1-row['speaker']]
            if pool:
                for _ in range(100):
                    other = self.rows[rng.choice(pool)]
                    if other['call_id'] != row['call_id']:
                        y = fit_length(load_segment(other), rng)
                        size = int(WIN * rng.uniform(.1, .5))
                        start = rng.choice([0, (WIN-size)//2, WIN-size])
                        mixed = clean.clone()
                        mixed[start:start+size] += rng.uniform(.2, .6) * y[start:start+size]
                        mixed, is_mixed = normalize(mixed), True
                        break
        return mixed, clean, is_mixed, row['speaker']


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
    torch.save(value, local)
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


def d_slices(rows, probs):
    """Accuracy on the D rows overall and by duration/overlap (overlap flag is the provided-label timing)."""
    import numpy as np
    y = np.array([r['speaker'] for r in rows])
    correct = (np.asarray(probs) >= .5) == y
    durations = np.array([(r['endAt']-r['startAt'])/1000 for r in rows])
    overlap = np.array([r['overlap'] for r in rows], dtype=bool)
    def acc(mask):
        return float(correct[mask].mean()) if mask.any() else None
    return dict(accuracy=float(correct.mean()), dur_0_5_1=acc((durations >= .5) & (durations < 1)),
                overlap=acc(overlap), no_overlap=acc(~overlap),
                rows=dict(all=len(rows), dur_0_5_1=int(((durations >= .5) & (durations < 1)).sum()),
                          overlap=int(overlap.sum())))


def stop_reason(variant, step0, first):
    """Pre-fixed kill rules, checked once after the first continued epoch."""
    if variant == 'asp' and first['accuracy'] < step0['accuracy'] - .002:
        return f"asp: D accuracy {first['accuracy']:.4f} < step0 {step0['accuracy']:.4f} - 0.002"
    if variant == 'consist':
        if first['overlap'] < step0['overlap'] + .005:
            return f"consist: D overlap {first['overlap']:.4f} < step0 {step0['overlap']:.4f} + 0.005"
        if first['no_overlap'] < step0['no_overlap'] - .003:
            return f"consist: D no_overlap {first['no_overlap']:.4f} < step0 {step0['no_overlap']:.4f} - 0.003"
    return None


def train(model, rows, d_rows, out, work, device, split_sha, step0, teacher=None):
    import torch
    import numpy as np
    import torch.nn.functional as F
    from transformers import get_cosine_schedule_with_warmup
    from torch.utils.data import DataLoader
    model.to(device)
    backbone = list(model.backbone.parameters())
    pool = [p for name, p in model.named_parameters() if name.startswith('pool.')]
    head = [p for name, p in model.named_parameters() if not name.startswith(('backbone.', 'pool.'))]
    groups = [{'params': backbone, 'lr': LR_BACKBONE}, {'params': head, 'lr': LR_HEAD}]
    if pool:
        groups.append({'params': pool, 'lr': LR_POOL})
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    steps = math.ceil(len({r['call_id'] for r in rows}) / 8)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(steps*EPOCHS*.05), steps*EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=device.startswith('cuda'))
    epoch, step, best, history, loss_sum, loss_count, epoch_seconds = 0, 0, -1., [], 0., 0, 0.
    kd_sum, kd_count = 0., 0
    resume = None
    if (out / 'ckpt_last.pt').exists():
        shutil.copyfile(out / 'ckpt_last.pt', work / 'resume.pt')
        resume = load_torch(work / 'resume.pt')
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
        kd_sum, kd_count = resume.get('kd_sum', 0.), resume.get('kd_count', 0)
        restore_rng(resume['rng'])
    stopped = next((h['stopped_early'] for h in history if h.get('stopped_early')), None)
    for ep in range(epoch, EPOCHS):
        if stopped:
            break
        sample_epoch = EPOCH_OFFSET + ep
        # The parent is already fine-tuned: no transformer freeze; the CNN front end stays frozen.
        for p in model.backbone.parameters():
            p.requires_grad_(True)
        model.backbone.feature_extractor._freeze_parameters()
        order = resume['sample_order'] if resume is not None and ep == epoch and resume['sample_order'] else epoch_order(rows, sample_epoch)
        dataset = (ConsistWindows if teacher is not None else Windows)(rows, order, sample_epoch)
        # A dedicated generator prevents iterator construction consuming model RNG on resume.
        loader = DataLoader(dataset, batch_size=32, sampler=range(step*32, len(order)),
                            num_workers=os.cpu_count() or 1, pin_memory=True,
                            generator=torch.Generator().manual_seed(42+sample_epoch))
        model.train()
        started = last_save = time.monotonic()
        def checkpoint(next_step, next_epoch=ep):
            save_torch(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                            epoch=next_epoch, step=next_step, sampler_epoch=next_epoch,
                            sample_order=order if next_epoch == ep else [], rng=rng_state(),
                            best=best, history=history, split_sha256=split_sha,
                            loss_sum=loss_sum, loss_count=loss_count, kd_sum=kd_sum, kd_count=kd_count,
                            epoch_seconds=epoch_seconds + time.monotonic()-started if next_epoch == ep else 0.),
                       'ckpt_last.pt', work, out)
        for batch_step, batch in enumerate(loader, start=step):
            if teacher is None:
                waveform, labels = batch
            else:
                waveform, clean, mixed, labels = batch
            waveform, labels = waveform.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=device.startswith('cuda')):
                logits, embedding = model(waveform)
                loss = F.binary_cross_entropy_with_logits(logits.float(), labels.float()) + .5 * supcon(embedding, labels)
                if teacher is not None and mixed.any():
                    mask = mixed.to(device)
                    with torch.no_grad():
                        teacher_logits, teacher_embedding = teacher(clean.to(device)[mask])
                    kd = F.binary_cross_entropy_with_logits(logits.float()[mask], teacher_logits.float().sigmoid())
                    cosine = (1 - (embedding.float()[mask] * teacher_embedding.float()).sum(1)).mean()
                    loss = loss + CONSIST_KD * kd + CONSIST_COS * cosine
                    kd_sum += kd.item() * int(mask.sum())
                    kd_count += int(mask.sum())
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            scaler.scale(loss).backward()
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
        slices = d_slices(d_rows, probabilities)
        accuracy = slices['accuracy']
        minutes = (epoch_seconds + time.monotonic()-started)/60
        entry = dict(epoch=ep+1, sample_epoch=sample_epoch, d_accuracy=accuracy, d_slices=slices,
                     train_loss=loss_sum/max(loss_count, 1), minutes=minutes)
        if teacher is not None:
            entry.update(kd_loss=kd_sum/max(kd_count, 1), mixed_fraction=kd_count/max(loss_count, 1))
        if ep == 0:
            stopped = stop_reason(VARIANT, step0, slices)
            entry['stopped_early'] = stopped
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if accuracy > best:
            best = accuracy
            weights = {k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu() for k, v in model.state_dict().items()}
            save_torch(dict(model_state_dict=weights, best_epoch=ep+1, d_accuracy=best,
                            emb_dim=256, win_sec=3., sr=SR, revision=REVISION, layers=13,
                            architecture='wavlm-base-plus-weighted-asp' if VARIANT == 'asp' else 'wavlm-base-plus-weighted-mean-std',
                            parent=PARENT, variant=VARIANT, label_source='provided_json',
                            split_sha256=split_sha, inference=dict(stride_sec=1.5, max_windows=8, aggregation='mean_logit', threshold=.5)),
                       'best.ckpt', work, out)
        step, loss_sum, loss_count, epoch_seconds, kd_sum, kd_count = 0, 0., 0, 0., 0., 0
        checkpoint(0, ep+1)
        resume = None
    write_json(out / 'train_done.json', dict(history=history, best_d_accuracy=best, stopped_early=stopped))


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


def asp_identity(model, rows, device):
    """Max |logit| difference between ASP at init and plain mean||std on the same windows (should be ~0)."""
    import torch
    model.eval()
    worst = 0.
    with torch.inference_mode():
        for row in rows:
            x = load_segment(row)
            batch = torch.stack([fit_length(x, start=s) for s in window_starts(len(x))]).to(device)
            with torch.autocast(device_type='cuda', enabled=device.startswith('cuda')):
                with_pool = model(batch)[0].float()
                pool, model.pool = model.pool, None
                try:
                    without = model(batch)[0].float()
                finally:
                    model.pool = pool
            worst = max(worst, float((with_pool - without).abs().max()))
    return worst


def ensemble_diagnostics(rows, candidate, run):
    """Fixed 1:1 probability means on H against base+large+CNN, using saved M2-SSL-003/004/005 outputs.

    Decision rule fixed before results: accuracy >= reference + 0.002, exact two-sided McNemar
    p < 0.05 with more candidate-only correct rows, and higher accuracy in >= 4 of 5 call groups
    (sorted call_id, contiguous fifths; same grouping as reports/mission2/ensemble_candidates).
    """
    import numpy as np
    from scipy.stats import binomtest
    keys = [f"{r['call_id']}\t{r['utt_idx']}\t{r['startAt']}\t{r['endAt']}" for r in rows]
    result = dict(skipped_reason=None)

    def saved(exp, name):
        directory = run / 'lab' / exp
        if not (directory / 'H_rows.txt').exists() or not (directory / name).exists():
            return None
        if (directory / 'H_rows.txt').read_text().splitlines() != keys:
            raise ValueError(f'{exp} H row order differs')
        return np.load(directory / name)
    ssl003 = saved('M2-SSL-003', 'H_probs.npy')
    ssl004 = saved('M2-SSL-004', 'wavlm_H_probs.npy')
    ssl005 = saved('M2-SSL-005', 'w2v2_H_probs.npy')
    if ssl003 is None or ssl004 is None or ssl003.ndim != 2 or ssl003.shape[1] != 4:
        result['skipped_reason'] = 'M2-SSL-003 (wavlm + 3 CNN columns) or M2-SSL-004 H probabilities missing'
        return result
    members = dict(base=ssl003[:, 0], cnn=ssl003[:, 1:].mean(1), large=ssl004.reshape(-1), cand=candidate)
    if ssl005 is not None:
        members['w2v2'] = ssl005.reshape(-1)
    y = np.array([r['speaker'] for r in rows])
    calls = sorted({r['call_id'] for r in rows})
    counts = [len(calls) // 5 + (i < len(calls) % 5) for i in range(5)]
    group_of, start = {}, 0
    for index, count in enumerate(counts):
        for call in calls[start:start+count]:
            group_of[call] = index
        start += count
    group = np.array([group_of[r['call_id']] for r in rows])

    def correct(names):
        return (np.mean([members[n] for n in names], 0) >= .5) == y

    def compare(names, reference):
        left, right = correct(names), correct(reference)
        only, ref_only = int((left & ~right).sum()), int((~left & right).sum())
        p = float(binomtest(only, only+ref_only, .5).pvalue) if only+ref_only else 1.
        ahead = int(sum(left[group == g].mean() > right[group == g].mean() for g in range(5)))
        delta = float(left.mean() - right.mean())
        return dict(accuracy=float(left.mean()), reference='+'.join(reference), delta=delta,
                    candidate_only=only, reference_only=ref_only, p_value=p, groups_ahead=ahead,
                    passes=bool(delta >= .002 and p < .05 and only > ref_only and ahead >= 4))
    reference = ['base', 'large', 'cnn']
    combos = {'cand+large+cnn': ['cand', 'large', 'cnn'], 'base+cand+large+cnn': ['base', 'cand', 'large', 'cnn'],
              'cand': ['cand'], 'base': ['base']}
    result['reference_accuracy'] = float(correct(reference).mean())
    result['comparisons'] = {name: compare(names, reference) for name, names in combos.items()}
    result['cand_vs_base'] = compare(['cand'], ['base'])
    if 'w2v2' in members:
        reference5 = reference + ['w2v2']
        result['reference_w2v2_accuracy'] = float(correct(reference5).mean())
        result['comparisons_w2v2'] = {name: compare(names + ['w2v2'], reference5)
                                     for name, names in combos.items() if 'cand' in names and len(names) > 1}
    else:
        result['w2v2_note'] = 'M2-SSL-005 H probabilities not present at evaluation time'
    result['any_pass'] = any(c['passes'] for k in ('comparisons', 'comparisons_w2v2')
                             for name, c in result.get(k, {}).items() if name not in ('cand', 'base'))
    return result


def persist_npy(value, name, work, out):
    import numpy as np
    np.save(work / name, value)
    publish(work / name, out / name)


def smoke():
    """No model download, training, or competition data access."""
    import importlib.util
    import tempfile
    have_torch = all(importlib.util.find_spec(name) is not None for name in ('torch', 'torchaudio', 'numpy'))
    if have_torch:
        audio_preflight()
        import torch
        torch.manual_seed(0)
        hidden = torch.randn(2, 150, 768) * 3
        pooled = build_asp(768)(hidden)
        plain = torch.cat([hidden.mean(1), hidden.std(1, unbiased=False)], 1)
        assert pooled.shape == (2, 1536) and torch.allclose(pooled, plain, atol=1e-4), 'ASP init must equal mean||std'
    else:
        print('audio preflight and ASP check SKIPPED: torch/torchaudio/numpy unavailable; required on normal startup')
    assert set(VARIANTS.values()) == {'asp', 'consist', 'control'}
    assert stop_reason('control', dict(accuracy=.9), dict(accuracy=.1)) is None
    assert stop_reason('asp', dict(accuracy=.9), dict(accuracy=.897)) is not None
    assert stop_reason('asp', dict(accuracy=.9), dict(accuracy=.899)) is None
    assert stop_reason('consist', dict(overlap=.8, no_overlap=.9), dict(overlap=.806, no_overlap=.9)) is None
    assert stop_reason('consist', dict(overlap=.8, no_overlap=.9), dict(overlap=.804, no_overlap=.9)) is not None
    assert stop_reason('consist', dict(overlap=.8, no_overlap=.9), dict(overlap=.81, no_overlap=.896)) is not None
    assert window_starts(1) == [0]
    assert window_starts(WIN) == [0]
    assert window_starts(WIN+1) == [0, 1]
    assert stem_key('서울.WAV') == stem_key(unicodedata.normalize('NFD', '서울') + '.wav')
    starts = window_starts(SR*100)
    assert len(starts) == 8 and starts[0] == 0 and starts[-1] == SR*100-WIN
    rows = [dict(call_id='a'), dict(call_id='b'), dict(call_id='b')]
    order = epoch_order(rows, EPOCH_OFFSET)
    assert order == epoch_order(rows, EPOCH_OFFSET) and len(order) == 8
    assert all(sum(rows[idx]['call_id'] == c for idx in order) == 4 for c in ('a', 'b'))
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        local, target = root/'local', root/'out'
        local.write_bytes(b'artifact')
        publish(local, target)
        assert target.read_bytes() == b'artifact'
    print('smoke OK: ASP init identity, kill rules, sliding windows, grouped sampler, atomic publication')


def main():
    global EXP_ID, VARIANT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp-id', choices=sorted(VARIANTS), default=os.environ.get('LAB_EXP_ID'))
    parser.add_argument('--output', type=Path, default=os.environ.get('LAB_OUT'))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    if args.output is None or args.exp_id not in VARIANTS:
        parser.error('--exp-id (M2-SSL-006/007/008) and --output or LAB_OUT required')
    EXP_ID, VARIANT = args.exp_id, VARIANTS[args.exp_id]
    run = Path(os.environ['DCC_RUN_ROOT'])
    if not (run / f'lab/{PARENT}/best.ckpt').exists():
        raise FileNotFoundError(f'parent checkpoint lab/{PARENT}/best.ckpt missing')
    # Fail on model/dependency problems before touching data archives.
    backbone = prepare_model(run)
    teacher_backbone = prepare_model(run) if VARIANT == 'consist' else None
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
    if (out/'metrics.json').exists():
        print('experiment already complete', flush=True)
        return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = prepare_rows(work)
    split = make_split(rows, out, run)
    sets = {name: set(split[name]) for name in ('train', 'D', 'H')}
    assert not (sets['train'] & sets['D'] or sets['train'] & sets['H'] or sets['H'] & sets['D'])
    train_rows = [r for r in rows if r['call_id'] in sets['train']]
    h_rows = [r for r in rows if r['call_id'] in sets['H']]
    d_all = [r for r in rows if r['call_id'] in sets['D']]
    groups = {}
    for row in d_all:
        groups.setdefault(row['call_id'], []).append(row)
    rng = random.Random(42)
    d_rows = [row for call in sorted(groups) for row in rng.sample(groups[call], min(10, len(groups[call])))]
    model = build_model(backbone, pool=VARIANT == 'asp')
    parent = load_parent(model, run, split['sha256'])
    teacher = None
    if VARIANT == 'consist':
        teacher = build_model(teacher_backbone)
        load_parent(teacher, run, split['sha256'])
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    if not (out/'step0_d.json').exists():
        model.to(device)
        identity = asp_identity(model, d_rows[:16], device) if VARIANT == 'asp' else None
        if identity is not None and identity > .05:
            raise RuntimeError(f'ASP init differs from mean||std forward: {identity}')
        probabilities, _ = evaluate(model, d_rows, device)
        write_json(out/'step0_d.json', dict(d_slices(d_rows, probabilities), parent=parent,
                                            asp_identity_max_abs_logit_diff=identity))
    step0 = json.loads((out/'step0_d.json').read_text())
    print(json.dumps(dict(step0=step0)), flush=True)
    if not (out/'train_done.json').exists():
        train(model, train_rows, d_rows, out, work, device, split['sha256'], step0, teacher)
    done = json.loads((out/'train_done.json').read_text())
    common = dict(exp_id=EXP_ID, seed=42, train_rows=len(train_rows), eval_rows=len(h_rows), official_val=None,
                  ckpt='best.ckpt', label_source='provided_json', split_sha256=split['sha256'], revision=REVISION,
                  variant=VARIANT, parent=PARENT)
    if done.get('stopped_early'):
        # Pre-fixed kill rule: no H evaluation, the candidate is discarded.
        write_json(out/'metrics.json', dict(common, primary_metric=dict(name='accuracy', value=None, split='internal_holdout'),
            metrics=dict(step0_d=step0, epoch_history=done['history'], stopped_early=done['stopped_early']),
            notes=f"stopped after first epoch by pre-fixed rule: {done['stopped_early']}; H not evaluated"))
        print('stopped early: ' + done['stopped_early'], flush=True)
        return
    best = load_torch(out/'best.ckpt')
    if best['split_sha256'] != split['sha256']:
        raise ValueError('best checkpoint split mismatch')
    model.load_state_dict(best['model_state_dict'])
    model.to(device)
    if (out/'h_eval_done.json').exists():
        eval_done = json.loads((out/'h_eval_done.json').read_text())
        if eval_done['split_sha256'] != split['sha256']:
            raise ValueError('H evaluation split mismatch')
        probabilities = np.load(out/'wavlm_H_probs.npy')
        if probabilities.shape != (len(h_rows),):
            raise ValueError('H cached row count mismatch')
    else:
        probabilities, seconds = evaluate(model, h_rows, device)
        persist_npy(probabilities, 'wavlm_H_probs.npy', work, out)
        eval_done = dict(split_sha256=split['sha256'], seconds=seconds, device=device,
                         gpu_utterances_per_second=len(h_rows)/seconds if device == 'cuda' else None)
        write_json(out/'h_eval_done.json', eval_done)
    persist_npy(probabilities[:, None], 'H_probs.npy', work, out)
    write_json(out/'H_probs_columns.json', [f'wavlm_{VARIANT}'])
    local_rows = work/'H_rows.txt'
    local_rows.write_text(''.join(f"{r['call_id']}\t{r['utt_idx']}\t{r['startAt']}\t{r['endAt']}\n" for r in h_rows))
    publish(local_rows, out/'H_rows.txt')
    summary = summaries(h_rows, probabilities)
    ensemble = ensemble_diagnostics(h_rows, probabilities, run)
    majority = max(sum(r['speaker'] == 0 for r in d_rows), sum(r['speaker'] == 1 for r in d_rows))/len(d_rows)
    metrics = dict(common, primary_metric=dict(name='accuracy', value=summary['accuracy'], split='internal_holdout'),
        metrics=dict(summary, ensemble=ensemble, step0_d=step0, epoch_history=done['history'], best_epoch=best['best_epoch'],
                     best_d_accuracy=best['d_accuracy'], d_majority_accuracy=majority,
                     training_collapse=best['d_accuracy'] <= majority,
                     layer_weights=model.layer_weights.detach().float().softmax(0).cpu().tolist(),
                     inference_timing=eval_done,
                     counts={k: dict(calls=len(sets[k]), rows=len(v)) for k, v in [('train', train_rows), ('D', d_rows), ('H', h_rows)]},
                     sampled_train_rows_per_epoch=len(sets['train'])*4,
                     hyperparameters=dict(epochs=EPOCHS, epoch_offset=EPOCH_OFFSET, lr_backbone=LR_BACKBONE,
                                          lr_head=LR_HEAD, lr_pool=LR_POOL if VARIANT == 'asp' else None,
                                          consist=dict(mix_p=CONSIST_MIX_P, kd=CONSIST_KD, cos=CONSIST_COS) if VARIANT == 'consist' else None)),
        notes='Training 서울 only; continued from M2-SSL-003 best.ckpt; provided speaker labels; '
              'H used only for final reporting; no official Validation read')
    write_json(out/'metrics.json', metrics)
    print(json.dumps(metrics['primary_metric']), flush=True)
    print(json.dumps(dict(ensemble_any_pass=ensemble.get('any_pass'))), flush=True)


if __name__ == '__main__':
    main()
