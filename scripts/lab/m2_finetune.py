"""One-epoch continuation from M2-SSL-004 for Mission 2 residual experiments.

The caller owns the dataset class (it must live in the launched script so a
DataLoader worker can import it). This module only runs the optimization loop.
"""
from __future__ import annotations

import math
import shutil
import time


def load_ssl004():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).with_name('m2-ssl-004.py')
    spec = importlib.util.spec_from_file_location('m2_ssl_004_residual', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def refurb_targets(logits, labels, confidence=0.95, mix=0.1):
    """Mix the provided label toward the model's probability only when it is sure.

    Confidence is max(p, 1-p). The stored JSON label is not rewritten.
    """
    import torch
    prob = logits.float().sigmoid().detach()
    sure = torch.maximum(prob, 1 - prob) >= confidence
    soft = (1 - mix) * labels.float() + mix * prob
    return torch.where(sure, soft, labels.float())


def bce_supcon(ssl, logits, embedding, labels):
    import torch.nn.functional as F
    return F.binary_cross_entropy_with_logits(logits.float(), labels.float()) + .5 * ssl.supcon(embedding, labels)


def refurb_loss(ssl, logits, embedding, labels):
    import torch.nn.functional as F
    target = refurb_targets(logits, labels)
    return F.binary_cross_entropy_with_logits(logits.float(), target) + .5 * ssl.supcon(embedding, labels)


def short_epoch_order(rows, epoch, fraction=0.7, short_sec=1.2):
    """Same step count as M2-SSL-004 (4 draws per call), with `fraction` from short utterances.

    Length uses only that utterance's own startAt/endAt. The model never receives it.
    """
    import random
    calls = {r['call_id'] for r in rows}
    total = 4 * len(calls)
    short, long = [], []
    for idx, row in enumerate(rows):
        (short if (row['endAt'] - row['startAt']) / 1000 < short_sec else long).append(idx)
    if not short or not long:
        raise ValueError('short and long utterance pools must both be non-empty')
    rng = random.Random(42 + epoch)
    n_short = int(round(total * fraction))
    order = [rng.choice(short) for _ in range(n_short)]
    order += [rng.choice(long) for _ in range(total - n_short)]
    rng.shuffle(order)
    return order


def paired(labels, left, right):
    """Exact two-sided McNemar of left vs right at threshold 0.5. `left_only` is left-correct and right-wrong."""
    import numpy as np
    from scipy.stats import binomtest
    y = np.asarray(labels)
    a = np.asarray(left) >= .5
    b = np.asarray(right) >= .5
    correct_a, correct_b = a == y, b == y
    left_only = int((correct_a & ~correct_b).sum())
    right_only = int((~correct_a & correct_b).sum())
    n = left_only + right_only
    return dict(left_only_correct=left_only, right_only_correct=right_only,
                p_value=float(binomtest(left_only, n, .5).pvalue) if n else 1.,
                method='exact_two_sided', threshold=0.5)


def finetune(ssl, model, train_rows, d_rows, out, work, device, split_sha, dataset_cls, order_fn, loss_fn,
             exp_id, architecture, micro_batch=32, backbone_lr=1e-5, head_lr=1e-4):
    """Continue one epoch from the in-memory weights. Epoch 0 does not freeze the transformer."""
    import random
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    epochs = 1
    model.to(device)
    backbone = list(model.backbone.parameters())
    head = [p for name, p in model.named_parameters() if not name.startswith('backbone.')]
    optimizer = torch.optim.AdamW(
        [{'params': backbone, 'lr': backbone_lr}, {'params': head, 'lr': head_lr}], weight_decay=.01)
    n_calls = len({r['call_id'] for r in train_rows})
    steps = math.ceil(n_calls / 8)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(steps * .1), steps)
    scaler = torch.amp.GradScaler('cuda', enabled=device.startswith('cuda'))
    epoch, step, best, history, loss_sum, loss_count, epoch_seconds = 0, 0, -1., [], 0., 0, 0.
    resume = None
    if (out / 'ckpt_last.pt').exists():
        shutil.copyfile(out / 'ckpt_last.pt', work / 'resume.pt')
        resume = ssl.load_torch(work / 'resume.pt')
        if resume.get('epochs') != epochs or resume.get('exp_id') != exp_id:
            raise ValueError('resume experiment or epoch schedule mismatch')
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
        ssl.restore_rng(resume['rng'])
        for key in ('model', 'optimizer', 'scheduler', 'scaler', 'rng'):
            del resume[key]
    for ep in range(epoch, epochs):
        for p in model.backbone.parameters():
            p.requires_grad_(True)
        model.backbone.feature_extractor._freeze_parameters()
        order = resume['sample_order'] if resume is not None and ep == epoch and resume['sample_order'] else order_fn(train_rows, ep)
        if len(order) != 4 * n_calls:
            raise ValueError('epoch order length drifted from 4 draws per call')
        dataset = dataset_cls(train_rows, order, ep)
        loader = DataLoader(
            dataset, batch_size=32, sampler=range(step * 32, len(order)), num_workers=0,
            generator=torch.Generator().manual_seed(42 + ep))
        model.train()
        started = last_save = time.monotonic()

        def checkpoint(next_step, next_epoch=ep):
            ssl.save_torch(dict(
                model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                epoch=next_epoch, step=next_step, sampler_epoch=next_epoch,
                sample_order=order if next_epoch == ep else [], rng=ssl.rng_state(),
                best=best, history=history, split_sha256=split_sha, epochs=epochs,
                exp_id=exp_id, revision=ssl.REVISION, micro_batch_size=micro_batch,
                effective_batch_size=32, backbone_lr=backbone_lr, head_lr=head_lr,
                loss_sum=loss_sum, loss_count=loss_count,
                epoch_seconds=epoch_seconds + time.monotonic() - started if next_epoch == ep else 0.),
                'ckpt_last.pt', work, out)

        for batch_step, (waveform, labels) in enumerate(loader, start=step):
            waveform, labels = waveform.to(device), labels.to(device)
            batch_rng = ssl.rng_state()
            while True:
                optimizer.zero_grad(set_to_none=True)
                try:
                    with torch.autocast(device_type='cuda', enabled=device.startswith('cuda')):
                        if micro_batch == 32:
                            logits, embedding = model(waveform)
                        else:
                            from torch.utils.checkpoint import checkpoint as recompute
                            pieces = [recompute(model, x, use_reentrant=False, preserve_rng_state=True)
                                      for x in waveform.split(micro_batch)]
                            logits = torch.cat([x[0] for x in pieces])
                            embedding = torch.cat([x[1] for x in pieces])
                        loss = loss_fn(logits, embedding, labels)
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite training loss')
                    scaler.scale(loss).backward()
                    break
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    if micro_batch == 1:
                        raise
                    logits = embedding = loss = None
                    torch.cuda.empty_cache()
                    ssl.restore_rng(batch_rng)
                    micro_batch //= 2
                    print(f'OOM: micro batch {micro_batch}, effective batch 32', flush=True)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            loss_sum += float(loss.item()) * len(labels)
            loss_count += len(labels)
            if time.monotonic() - last_save >= 900:
                checkpoint(batch_step + 1)
                last_save = time.monotonic()
        checkpoint(steps)
        probabilities, _ = ssl.evaluate(model, d_rows, device)
        accuracy = float(np.mean((probabilities >= .5) == np.array([r['speaker'] for r in d_rows])))
        minutes = (epoch_seconds + time.monotonic() - started) / 60
        targets = np.array([r['speaker'] for r in d_rows])
        clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
        d_loss = float(-(targets * np.log(clipped) + (1 - targets) * np.log1p(-clipped)).mean())
        history.append(dict(epoch=ep + 1, d_accuracy=accuracy, d_loss=d_loss,
                            train_loss=loss_sum / max(loss_count, 1), minutes=minutes,
                            micro_batch_size=micro_batch, effective_batch_size=32,
                            backbone_lr=backbone_lr, head_lr=head_lr))
        print(json_line(history[-1]), flush=True)
        if accuracy > best:
            best = accuracy
            weights = {k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()
                       for k, v in model.state_dict().items()}
            ssl.save_torch(dict(
                model_state_dict=weights, best_epoch=ep + 1, d_accuracy=best, emb_dim=256,
                win_sec=3., sr=ssl.SR, revision=ssl.REVISION,
                layers=model.backbone.config.num_hidden_layers + 1,
                hidden_size=model.backbone.config.hidden_size, architecture=architecture,
                label_source='provided_json', split_sha256=split_sha, exp_id=exp_id,
                parent='M2-SSL-004', backbone_lr=backbone_lr, head_lr=head_lr,
                inference=dict(stride_sec=1.5, max_windows=8, aggregation='mean_logit', threshold=.5)),
                'best.ckpt', work, out)
        step, loss_sum, loss_count, epoch_seconds = 0, 0., 0, 0.
        checkpoint(0, ep + 1)
        resume = None
    ssl.write_json(out / 'train_done.json', dict(
        history=history, best_d_accuracy=best, epochs=epochs, micro_batch_size=micro_batch,
        effective_batch_size=32, backbone_lr=backbone_lr, head_lr=head_lr, exp_id=exp_id))
    return history


def json_line(value):
    import json
    return json.dumps(value)


def init_from_ssl004(ssl, model, run):
    path = run / 'lab/M2-SSL-004/best.ckpt'
    if not path.exists():
        raise FileNotFoundError('M2-SSL-004 best.ckpt missing')
    best = ssl.load_torch(path)
    if best.get('revision') != ssl.REVISION:
        raise ValueError('SSL-004 checkpoint revision mismatch')
    if best.get('label_source') != 'provided_json':
        raise ValueError('SSL-004 checkpoint label source mismatch')
    model.load_state_dict(best['model_state_dict'])
    return best


def copied_windows(ssl):
    """Dataset matching M2-SSL-004.Windows, defined here for the parent process (num_workers=0)."""

    class Windows:
        def __init__(self, rows, order, epoch):
            self.rows, self.order, self.epoch = rows, order, epoch
            self.by_label = {label: [i for i, r in enumerate(rows) if r['speaker'] == label] for label in (0, 1)}

        def __len__(self):
            return len(self.order)

        def __getitem__(self, position):
            import random
            idx = self.order[position]
            row = self.rows[idx]
            rng = random.Random(42 + self.epoch * 100000000 + position)
            x = ssl.fit_length(ssl.load_segment(row), rng)
            if rng.random() < .15:
                pool = self.by_label[1 - row['speaker']]
                if pool:
                    for _ in range(100):
                        other = self.rows[rng.choice(pool)]
                        if other['call_id'] != row['call_id']:
                            y = ssl.fit_length(ssl.load_segment(other), rng)
                            size = int(ssl.WIN * rng.uniform(.1, .25))
                            start = 0 if rng.random() < .5 else ssl.WIN - size
                            x = x.clone()
                            x[start:start + size] += rng.uniform(.15, .40) * y[start:start + size]
                            x = ssl.normalize(x)
                            break
            return x, row['speaker']

    return Windows


def prepare_runtime(ssl, out, work, run, load_backbone=True):
    import numpy as np
    import random
    import torch
    if not (run / 'lab/M2-SSL-003/split.json').exists():
        raise FileNotFoundError('required SSL-003 split.json missing')
    if load_backbone and not (run / 'lab/M2-SSL-004/best.ckpt').exists():
        raise FileNotFoundError('required SSL-004 best.ckpt missing')
    backbone = ssl.prepare_model(run) if load_backbone else None
    ssl.audio_preflight()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = ssl.prepare_rows(work)
    split = ssl.make_split(rows, out, run)
    sets = {name: set(split[name]) for name in ('train', 'D', 'H')}
    if sets['train'] & sets['D'] or sets['train'] & sets['H'] or sets['H'] & sets['D']:
        raise ValueError('call split overlap')
    train_rows = [r for r in rows if r['call_id'] in sets['train']]
    h_rows = ssl.align_baseline_rows([r for r in rows if r['call_id'] in sets['H']], run)
    d_all = [r for r in rows if r['call_id'] in sets['D']]
    groups = {}
    for row in d_all:
        groups.setdefault(row['call_id'], []).append(row)
    rng = random.Random(42)
    d_rows = [row for call in sorted(groups) for row in rng.sample(groups[call], min(10, len(groups[call])))]
    model = ssl.build_model(backbone) if load_backbone else None
    return dict(device=device, rows=rows, split=split, sets=sets, train_rows=train_rows,
                h_rows=h_rows, d_rows=d_rows, d_all=d_all, model=model)


def load_parent_probs(run, rows):
    import numpy as np
    path = run / 'lab/M2-SSL-004/wavlm_H_probs.npy'
    keys_path = run / 'lab/M2-SSL-004/H_rows.txt'
    if not path.exists() or not keys_path.exists():
        raise FileNotFoundError('SSL-004 H probabilities missing')
    keys = keys_path.read_text().splitlines()
    if keys != [ssl_row_key(r) for r in rows]:
        raise ValueError('SSL-004 H row order mismatch')
    probs = np.load(path)
    if probs.shape != (len(rows),) or not np.isfinite(probs).all():
        raise ValueError('SSL-004 H probabilities invalid')
    return probs


def ssl_row_key(row):
    return f"{row['call_id']}\t{row['utt_idx']}\t{row['startAt']}\t{row['endAt']}"


def duration_mask(rows, name):
    import numpy as np
    durations = np.array([(r['endAt'] - r['startAt']) / 1000 for r in rows])
    if name == '<0.5s':
        return durations < .5
    if name == '0.5~1s':
        return (durations >= .5) & (durations < 1)
    if name == '1~3s':
        return (durations >= 1) & (durations <= 3)
    if name == '>3s':
        return durations > 3
    raise KeyError(name)
