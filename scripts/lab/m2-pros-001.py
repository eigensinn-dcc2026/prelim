#!/usr/bin/env python3
"""M2-PROS-001: a small CNN on F0, log-energy, and aperiodicity of the target utterance.

The spectral envelope is not an input. At the end, if saved base/large/CNN
probabilities exist for the same H rows, compare the equal-weight 3-member mean
with a 4-member mean that adds this model. Official Validation is not read.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

EXP_ID = 'M2-PROS-001'
FRAME, HOP, N_FRAMES = 400, 160, 300
EPOCHS = 6


def prosody_channels(signal, sr=16000):
    import numpy as np
    x = np.ascontiguousarray(signal, dtype=np.float32).reshape(-1)
    if x.size == 0:
        x = np.zeros(1, dtype=np.float32)
    out = np.zeros((3, N_FRAMES), dtype=np.float32)
    if x.size < FRAME:
        x = np.pad(x, (0, FRAME - int(x.size)))
    n = 1 + (x.size - FRAME) // HOP
    start = 0 if n <= N_FRAMES else (n - N_FRAMES) // 2
    take = min(n, N_FRAMES)
    view = np.lib.stride_tricks.as_strided(
        x, shape=(n, FRAME), strides=(x.strides[0] * HOP, x.strides[0]))
    frames = np.array(view[start:start + take], dtype=np.float32, copy=True)
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME) / (FRAME - 1))).astype(np.float32)
    spec = np.fft.rfft(frames * window, n=512, axis=1)
    ac = np.fft.irfft(spec * np.conjugate(spec), n=512, axis=1).real
    lag_min = int(sr / 400)
    lag_max = min(int(sr / 70), FRAME - 1)
    band = ac[:, lag_min:lag_max + 1]
    peak = band.argmax(1)
    lag = peak + lag_min
    periodicity = np.clip(band.max(1) / np.maximum(ac[:, 0], 1e-8), 0, 1)
    f0 = sr / np.maximum(lag, 1)
    voiced = periodicity >= 0.3
    log_f0 = np.zeros(take, dtype=np.float32)
    log_f0[voiced] = np.log(f0[voiced]).astype(np.float32)
    log_e = np.log(np.sqrt(np.mean(frames * frames, axis=1)) + 1e-6).astype(np.float32)
    out[0, :take] = log_f0
    out[1, :take] = log_e
    out[2, :take] = (1 - periodicity).astype(np.float32)
    return out


def build_model():
    import torch
    from torch import nn

    class ProsodyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(3, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1))
            self.head = nn.Linear(64, 1)

        def forward(self, waveform):
            hidden = self.net(waveform).squeeze(-1)
            return self.head(hidden).squeeze(1)

    return ProsodyCNN()


def smoke():
    import numpy as np
    tone = np.sin(2 * math.pi * 180 * np.arange(16000) / 16000).astype(np.float32)
    features = prosody_channels(tone)
    assert features.shape == (3, N_FRAMES)
    assert np.isfinite(features).all()
    assert float(features[0].max()) > 0
    empty = prosody_channels(np.zeros(80, dtype=np.float32))
    assert empty.shape == (3, N_FRAMES) and np.isfinite(empty).all()
    print('smoke OK: prosody map is three channels from one waveform')


class ProsodySet:
    def __init__(self, ssl, rows, order):
        self.ssl, self.rows, self.order = ssl, rows, order
        self.cache = {}

    def __len__(self):
        return len(self.order)

    def __getitem__(self, position):
        import torch
        row = self.rows[self.order[position]]
        key = (row['call_id'], row['utt_idx'])
        if key not in self.cache:
            self.cache[key] = prosody_channels(self.ssl.load_segment(row).numpy())
        return torch.from_numpy(self.cache[key]), row['speaker']


def predict(model, ssl, rows, device):
    import numpy as np
    import torch
    model.eval()
    dataset = ProsodySet(ssl, rows, list(range(len(rows))))
    probs = []
    with torch.inference_mode():
        for idx in range(len(rows)):
            features, _ = dataset[idx]
            logit = model(features.unsqueeze(0).to(device)).float()
            probs.append(float(logit.sigmoid().item()))
    return np.asarray(probs)


def ensemble_columns(run, rows):
    import numpy as np
    root = run / 'lab/M2-SSL-003'
    large_path = run / 'lab/M2-SSL-004/wavlm_H_probs.npy'
    needed = ('H_rows.txt', 'H_probs.npy', 'H_probs_columns.json')
    if not all((root / name).exists() for name in needed) or not large_path.exists():
        return None, 'SSL-003 or SSL-004 H probabilities missing'
    keys = (root / 'H_rows.txt').read_text().splitlines()
    if keys != [f"{r['call_id']}\t{r['utt_idx']}\t{r['startAt']}\t{r['endAt']}" for r in rows]:
        raise ValueError('SSL-003 H row order mismatch')
    columns = json.loads((root / 'H_probs_columns.json').read_text())
    matrix = np.load(root / 'H_probs.npy')
    if matrix.shape != (len(rows), len(columns)):
        raise ValueError('SSL-003 H probability shape mismatch')
    large = np.load(large_path)
    if large.shape != (len(rows),):
        raise ValueError('SSL-004 H probability shape mismatch')
    named = {name: matrix[:, i] for i, name in enumerate(columns)}
    named['wavlm_large'] = large
    if 'wavlm_base' not in named and 'wavlm' in named:
        named['wavlm_base'] = named['wavlm']
    cnn_cols = [named[name] for name in columns if name.lower().startswith('cnn')]
    if 'wavlm_base' not in named or not cnn_cols:
        return None, 'base or CNN probability column missing'
    cnn = np.mean(cnn_cols, axis=0)
    return dict(base=named['wavlm_base'], large=large, cnn=cnn), None


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
    from m2_finetune import duration_mask, load_ssl004, paired, prepare_runtime
    import numpy as np
    import random
    import torch
    ssl = load_ssl004()
    run = Path(os.environ['DCC_RUN_ROOT'])
    out, work = args.output, Path(os.environ['LAB_WORK']) / EXP_ID.lower()
    if (out / 'metrics.json').exists() and (out / 'h_eval_done.json').exists():
        print('experiment already complete', flush=True)
        return
    bundle = prepare_runtime(ssl, out, work, run, load_backbone=False)
    # prepare_runtime builds a WavLM. This experiment does not train it.
    del bundle['model']
    device = bundle['device']
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epoch, best, history = 0, -1., []
    if (out / 'ckpt_last.pt').exists():
        import shutil
        shutil.copyfile(out / 'ckpt_last.pt', work / 'resume.pt')
        resume = ssl.load_torch(work / 'resume.pt')
        if resume.get('exp_id') != EXP_ID or resume.get('epochs') != EPOCHS:
            raise ValueError('resume experiment mismatch')
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        epoch, best, history = resume['epoch'], resume['best'], resume['history']
    train_rows = bundle['train_rows']
    n_calls = len({r['call_id'] for r in train_rows})
    for ep in range(epoch, EPOCHS):
        order = ssl.epoch_order(train_rows, ep)
        dataset = ProsodySet(ssl, train_rows, order)
        model.train()
        started = last_save = time.monotonic()
        loss_sum = loss_count = 0
        loader_idx = list(range(0, len(order), 64))

        def checkpoint(next_epoch, done=False):
            ssl.save_torch(dict(
                model=model.state_dict(), optimizer=optimizer.state_dict(),
                epoch=next_epoch, best=best, history=history, exp_id=EXP_ID, epochs=EPOCHS,
                split_sha256=bundle['split']['sha256']), 'ckpt_last.pt', work, out)

        for start in loader_idx:
            batch_i = order[start:start + 64]
            xs, ys = [], []
            for position in range(start, start + len(batch_i)):
                x, y = dataset[position]
                xs.append(x)
                ys.append(y)
            waveform = torch.stack(xs).to(device)
            labels = torch.tensor(ys, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(waveform).float(), labels)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)
            loss_count += len(labels)
            if time.monotonic() - last_save >= 900:
                checkpoint(ep)
                last_save = time.monotonic()
        d_probs = predict(model, ssl, bundle['d_rows'], device)
        d_y = np.array([r['speaker'] for r in bundle['d_rows']])
        accuracy = float(((d_probs >= .5) == d_y).mean())
        history.append(dict(epoch=ep + 1, d_accuracy=accuracy,
                            train_loss=loss_sum / max(loss_count, 1),
                            minutes=(time.monotonic() - started) / 60))
        print(json.dumps(history[-1]), flush=True)
        if accuracy > best:
            best = accuracy
            ssl.save_torch(dict(
                model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                best_epoch=ep + 1, d_accuracy=best, exp_id=EXP_ID, architecture='prosody-cnn',
                label_source='provided_json', split_sha256=bundle['split']['sha256'],
                frame=FRAME, hop=HOP, n_frames=N_FRAMES, sr=16000,
                inference=dict(threshold=0.5)), 'best.ckpt', work, out)
        checkpoint(ep + 1, done=True)
    best_ckpt = ssl.load_torch(out / 'best.ckpt')
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.to(device)
    h_rows = bundle['h_rows']
    started = time.monotonic()
    probabilities = predict(model, ssl, h_rows, device)
    ssl.persist_npy(probabilities, 'prosody_H_probs.npy', work, out)
    y = np.array([r['speaker'] for r in h_rows])
    summary = ssl.summaries(h_rows, probabilities)
    members, skip = ensemble_columns(run, h_rows)
    ensemble = dict(skipped_reason=skip)
    adopted = False
    if members is not None:
        three = (members['base'] + members['large'] + members['cnn']) / 3
        four = (members['base'] + members['large'] + members['cnn'] + probabilities) / 4
        three_summary = ssl.summaries(h_rows, three)
        four_summary = ssl.summaries(h_rows, four)
        comparison = paired(y, four, three)
        adopted = (four_summary['accuracy'] - three_summary['accuracy'] >= 0.002
                   and comparison['p_value'] < 0.05
                   and comparison['left_only_correct'] > comparison['right_only_correct'])
        slices = {}
        for name in ('<0.5s', '0.5~1s', '1~3s', '>3s'):
            mask = duration_mask(h_rows, name)
            slices[name] = dict(
                rows=int(mask.sum()),
                four=float(((four[mask] >= .5) == y[mask]).mean()) if mask.any() else None,
                three=float(((three[mask] >= .5) == y[mask]).mean()) if mask.any() else None,
                prosody=float(((probabilities[mask] >= .5) == y[mask]).mean()) if mask.any() else None)
        ensemble = dict(skipped_reason=None, three=three_summary, four=four_summary,
                        mcnemar_four_vs_three=comparison, slices=slices, adopted=bool(adopted))
    ssl.write_json(out / 'h_eval_done.json', dict(seconds=time.monotonic() - started, device=device))
    local_rows = work / 'H_rows.txt'
    local_rows.write_text(''.join(
        f"{r['call_id']}\t{r['utt_idx']}\t{r['startAt']}\t{r['endAt']}\n" for r in h_rows))
    ssl.publish(local_rows, out / 'H_rows.txt')
    metrics = dict(
        exp_id=EXP_ID,
        primary_metric=dict(
            name='accuracy',
            value=ensemble['four']['accuracy'] if members is not None else summary['accuracy'],
            split='internal_holdout'),
        metrics=dict(prosody=summary, ensemble=ensemble, epoch_history=history,
                     best_epoch=best_ckpt['best_epoch'], best_d_accuracy=best_ckpt['d_accuracy'],
                     adopted=bool(adopted), draws_per_call=4, calls=n_calls),
        seed=42, train_rows=len(train_rows), eval_rows=len(h_rows), official_val=None,
        ckpt='best.ckpt',
        notes='Prosody channels from the target utterance only. 4-member mean is equal weight. Threshold 0.5. No official Validation.',
        label_source='provided_json', split_sha256=bundle['split']['sha256'])
    ssl.write_json(out / 'metrics.json', metrics)
    print(json.dumps(metrics['primary_metric']), flush=True)


if __name__ == '__main__':
    main()
