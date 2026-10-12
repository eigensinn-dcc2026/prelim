#!/usr/bin/env python3
"""M2-SSL-003: Training-only WavLM; run from repository root with PYTHONPATH=src.

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

EXP_ID = 'M2-SSL-003'
REVISION = '4c66d4806a428f2e922ccfa1a962776e232d487b'
SR, WIN, EPOCHS = 16000, 48000, 15
CNN_SHA = '19ea482a80a2ceb2ea5b87981be0a381a11e01fd576b3ba6860ee5d7144e3246'


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
    import numpy as np
    calls = sorted({r['call_id'] for r in rows})
    path = out / 'split.json'
    if path.exists():
        split = json.loads(path.read_text())
        payload = {k: v for k, v in split.items() if k != 'sha256'}
        if digest(payload) != split['sha256']:
            raise ValueError('split hash mismatch')
        if sorted(split['train'] + split['H'] + split['D']) != calls:
            raise ValueError('Training calls differ from persisted split')
        return split
    reproduced, reason = True, ''
    try:
        import pandas as pd
        from dcc_data.mission2_train import make_group_split
        df = pd.read_parquet(run / 'm2_l2_spec/m2_spectrogram_index_full_provided_labels.parquet')
        if set(df.call_id.astype(str)) != set(calls):
            raise ValueError('CNN index call set differs from Training')
        _, indices = make_group_split(df, seed=42, test_size=0.2)
        v = sorted(set(df.iloc[indices].call_id.astype(str)))
    except Exception as exc:
        reproduced, reason = False, f'{type(exc).__name__}: {exc}'
        v = calls.copy()
    np.random.RandomState(42).shuffle(v)
    if len(v) < 2500 or len(calls) <= 2500:
        raise ValueError('need 2500 held-out calls and nonempty Training')
    h, d = v[:1500], v[1500:2500]
    held = set(h + d)
    split = dict(H=h, D=d, train=[c for c in calls if c not in held],
                 V_size=len(v) if reproduced else math.ceil(len(calls) * .2),
                 cnn_split_reproduced=reproduced, fallback_reason=reason,
                 seed=42, call_hashes={k: digest(x) for k, x in [('H', h), ('D', d),
                            ('train', [c for c in calls if c not in held])]})
    split['sha256'] = digest(split)
    write_json(path, split)
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
    import torchaudio
    import torch
    info = torchaudio.info(row['wav'])
    start = round(row['startAt'] * info.sample_rate / 1000)
    end = round(row['endAt'] * info.sample_rate / 1000)
    start = min(start, info.num_frames)
    end = min(end, info.num_frames)
    if end <= start:
        return torch.zeros(1)
    signal, sr = torchaudio.load(row['wav'], frame_offset=start, num_frames=end-start)
    signal = signal.mean(0)
    if sr != target_sr:
        signal = torchaudio.functional.resample(signal, sr, target_sr)
    return signal


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
            self.layer_weights = nn.Parameter(torch.zeros(13))
            self.embedding = nn.Sequential(nn.Linear(1536, 512), nn.ReLU(), nn.Dropout(.1), nn.Linear(512, 256))
            self.classifier = nn.Linear(256, 1)
            self.backbone.feature_extractor._freeze_parameters()

        def forward(self, waveform):
            states = self.backbone(waveform, output_hidden_states=True).hidden_states
            if len(states) != 13:
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


def train(model, rows, d_rows, out, work, device, split_sha):
    import torch
    import numpy as np
    from transformers import get_cosine_schedule_with_warmup
    from torch.utils.data import DataLoader
    model.to(device)
    backbone = list(model.backbone.parameters())
    head = [p for name, p in model.named_parameters() if not name.startswith('backbone.')]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': 3e-5}, {'params': head, 'lr': 3e-4}], weight_decay=.01)
    steps = math.ceil(len({r['call_id'] for r in rows}) / 8)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(steps*EPOCHS*.05), steps*EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=device.startswith('cuda'))
    epoch, step, best, history, loss_sum, loss_count, epoch_seconds = 0, 0, -1., [], 0., 0, 0.
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
        restore_rng(resume['rng'])
    for ep in range(epoch, EPOCHS):
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
                            loss_sum=loss_sum, loss_count=loss_count,
                            epoch_seconds=epoch_seconds + time.monotonic()-started if next_epoch == ep else 0.),
                       'ckpt_last.pt', work, out)
        for batch_step, (waveform, labels) in enumerate(loader, start=step):
            waveform, labels = waveform.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=device.startswith('cuda')):
                logits, embedding = model(waveform)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels.float()) + .5 * supcon(embedding, labels)
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
        accuracy = float(np.mean((probabilities >= .5) == np.array([r['speaker'] for r in d_rows])))
        minutes = (epoch_seconds + time.monotonic()-started)/60
        history.append(dict(epoch=ep+1, d_accuracy=accuracy, train_loss=loss_sum/max(loss_count, 1), minutes=minutes))
        print(json.dumps(history[-1]), flush=True)
        if ep == 0:
            print(f'15 epoch estimate: {minutes*15/60:.2f} hours; epoch count remains 15', flush=True)
        if accuracy > best:
            best = accuracy
            weights = {k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu() for k, v in model.state_dict().items()}
            save_torch(dict(model_state_dict=weights, best_epoch=ep+1, d_accuracy=best,
                            emb_dim=256, win_sec=3., sr=SR, revision=REVISION, layers=13,
                            architecture='wavlm-base-plus-weighted-mean-std', label_source='provided_json',
                            split_sha256=split_sha, inference=dict(stride_sec=1.5, max_windows=8, aggregation='mean_logit', threshold=.5)),
                       'best.ckpt', work, out)
        step, loss_sum, loss_count, epoch_seconds = 0, 0., 0, 0.
        checkpoint(0, ep+1)
        resume = None
    write_json(out / 'train_done.json', dict(history=history, best_d_accuracy=best))


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


def cnn_diagnostics(rows, wavlm, run, split, work, out, device):
    import numpy as np
    import torch
    result = dict(skipped_reason=None, cnn_split_suspect=False, valid=False)
    checkpoint = run / 'lab/M2-SSL-003/cnn/m2_cnn_large_rerun_ensemble.pt'
    if not split['cnn_split_reproduced']:
        result['skipped_reason'] = 'CNN split reproduction failed: ' + split['fallback_reason']
        return None, [], result
    if not checkpoint.exists():
        result['skipped_reason'] = 'CNN checkpoint missing'
        return None, [], result
    local = work / 'cnn.pt'
    shutil.copyfile(checkpoint, local)
    sha = hashlib.sha256()
    with local.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            sha.update(block)
    if sha.hexdigest() != CNN_SHA:
        result['skipped_reason'] = 'CNN checkpoint SHA256 mismatch'
        return None, [], result
    from dcc_data.mission2_checkpoint import resolve_l2_checkpoint_metadata, require_l2_inference_config, l2_member_state_dicts
    from dcc_data.mission2_cnn import SpeakerCNNLarge
    from dcc_data.mission2_spectrogram import mel_spectrogram, SPEC_FLOOR_DB
    from dcc_data.audio import read_wav_info, read_wav_frames
    from dcc_data.mission2_features import utterance_frame_range
    from scipy.stats import binomtest
    ckpt = load_torch(local)
    metadata, _ = resolve_l2_checkpoint_metadata(ckpt, local)
    require_l2_inference_config(metadata)
    if int(metadata['sample_rate']) != 8000 or metadata.get('split_seed') != 42 or float(metadata.get('test_size', -1)) != .2:
        result['skipped_reason'] = 'CNN metadata split/sample-rate mismatch'
        return None, [], result
    models = []
    for state in l2_member_state_dicts(ckpt, metadata):
        member = SpeakerCNNLarge().to(device)
        member.load_state_dict(state)
        member.eval()
        models.append(member)
    if len(models) != 3:
        raise ValueError('expected exactly three CNN members')
    # Version the cache: previous diagnostics used torchaudio resampling.
    cache = out / 'cnn_H_probs_submission_v2.npy'
    if cache.exists():
        probabilities = np.load(cache)
        if probabilities.shape != (len(rows), len(models)):
            raise ValueError('CNN cached row count mismatch')
    else:
        probabilities = []
        with torch.inference_mode():
            for row in rows:
                # Submission's PCM decoder and frame bounds, target segment only.
                # mel_spectrogram applies its own minimum N_FFT zero padding.
                info = read_wav_info(row['wav'])
                if info.sample_rate != int(metadata['sample_rate']) or info.channels != 1:
                    raise ValueError('CNN submission requires native 8kHz mono WAV')
                start, end = utterance_frame_range(
                    row['startAt'], row['endAt'], info.sample_rate, info.frame_count)
                signal = read_wav_frames(row['wav'], start, end).waveform[0]
                spec = mel_spectrogram(signal, info.sample_rate)
                if spec.shape[1] < 300:
                    spec = np.pad(spec, ((0, 0), (0, 300-spec.shape[1])), constant_values=SPEC_FLOOR_DB)
                batch = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(device)
                probabilities.append([float(model(batch).softmax(1)[0, 1]) for model in models])
        probabilities = np.array(probabilities)
        np.save(work / 'cnn_H_probs.npy', probabilities)
        publish(work / 'cnn_H_probs.npy', cache)
    y = np.array([r['speaker'] for r in rows])
    seeds = metadata.get('ensemble_member_torch_seeds', list(range(len(models))))
    columns = [f'cnn_seed{seed}' for seed in seeds]
    result['probability_columns'] = columns
    methods = {name: probabilities[:, i] for i, name in enumerate(columns)}
    methods.update(wavlm=wavlm, cnn_mean=probabilities.mean(1), wavlm_cnn_mean=(wavlm+probabilities.mean(1))/2)
    result['accuracies'] = {name: float(((p >= .5) == y).mean()) for name, p in methods.items()}
    result['cnn_split_suspect'] = result['accuracies']['cnn_mean'] > .93
    result['valid'] = not result['cnn_split_suspect']
    if result['cnn_split_suspect']:
        result['invalid_reason'] = 'CNN H accuracy > 0.93; possible overlap with CNN training calls'
    result['mcnemar'] = {}
    names = list(methods)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            ca, cb = (methods[a] >= .5) == y, (methods[b] >= .5) == y
            table = [[int((ca & cb).sum()), int((ca & ~cb).sum())],
                     [int((~ca & cb).sum()), int((~ca & ~cb).sum())]]
            n = table[0][1] + table[1][0]
            result['mcnemar'][a+'__'+b] = dict(table=table, axes='correct/incorrect',
                p_value=float(binomtest(table[0][1], n, .5).pvalue) if n else 1., method='exact_two_sided')
    calls = sorted({r['call_id'] for r in rows})
    np.random.RandomState(42).shuffle(calls)
    result['call_groups'] = []
    for group in np.array_split(calls, 5):
        group_calls = set(group)
        mask = np.array([r['call_id'] in group_calls for r in rows])
        result['call_groups'].append(dict(calls=len(group), rows=int(mask.sum()),
            accuracy={name: float(((p[mask] >= .5) == y[mask]).mean()) for name, p in methods.items()}))
    return probabilities, columns, result


def persist_npy(value, name, work, out):
    import numpy as np
    np.save(work / name, value)
    publish(work / name, out / name)


def smoke():
    """No model download, training, data access, or third-party imports."""
    import tempfile
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
        archive = root/'TL_서울_구급.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('TL_서울_구급/example.json', '{}')
        folder = unpack(archive, root/'work')
        assert (folder/'TL_서울_구급/example.json').exists()
        assert unpack(archive, root/'work') == folder
    print('smoke OK: sliding windows, grouped sampler, atomic publication, local zip extraction')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=os.environ.get('LAB_OUT'))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    if args.output is None:
        parser.error('--output or LAB_OUT required')
    run = Path(os.environ['DCC_RUN_ROOT'])
    # Fail on model/dependency problems before touching data archives.
    backbone = prepare_model(run)
    import torch
    import torchaudio  # noqa: F401: validate audio dependency before expensive data preparation
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
    model = build_model(backbone)
    if not (out/'train_done.json').exists():
        train(model, train_rows, d_rows, out, work, device, split['sha256'])
    done = json.loads((out/'train_done.json').read_text())
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
        # CPU benchmark uses batch size 1 even for the sliding windows.
        torch.set_num_threads(2)
        sample = np.random.RandomState(42).choice(len(h_rows), min(300, len(h_rows)), replace=False)
        model.to('cpu')
        _, cpu_seconds = evaluate(model, [h_rows[i] for i in sample], 'cpu', batch_windows=1)
        model.to(device)
        eval_done = dict(split_sha256=split['sha256'], seconds=seconds, device=device,
                         gpu_utterances_per_second=len(h_rows)/seconds if device == 'cuda' else None,
                         cpu_rows=len(sample), cpu_seconds=cpu_seconds,
                         cpu_official_val_estimated_minutes=cpu_seconds/len(sample)*111947/60)
        write_json(out/'h_eval_done.json', eval_done)
    diagnostics_path = out/'cnn_diagnostics.json'
    if diagnostics_path.exists():
        diagnostics = json.loads(diagnostics_path.read_text())
        cnn_probs = np.load(out/'cnn_H_probs.npy') if diagnostics.get('accuracies') else None
        columns = diagnostics['probability_columns'] if cnn_probs is not None else []
    else:
        cnn_probs, columns, diagnostics = cnn_diagnostics(h_rows, probabilities, run, split, work, out, device)
        write_json(diagnostics_path, diagnostics)
    combined = probabilities[:, None] if cnn_probs is None else np.column_stack([probabilities, cnn_probs])
    persist_npy(combined, 'H_probs.npy', work, out)
    write_json(out/'H_probs_columns.json', ['wavlm'] + columns)
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
        cnn_split_suspect=diagnostics['cnn_split_suspect'], revision=REVISION)
    write_json(out/'metrics.json', metrics)
    print(json.dumps(metrics['primary_metric']), flush=True)


if __name__ == '__main__':
    main()
