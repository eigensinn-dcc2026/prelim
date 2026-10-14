#!/usr/bin/env python3
"""M2-EDGE-001: average four views of one utterance through the frozen M2-SSL-004 model.

Views, all taken from the target startAt~endAt span after the training resample:
full span, drop the first 120 ms, drop the last 120 ms, and the center 400 ms.
The reported probability is the mean of the four sigmoid probabilities.
No gradient update. Official Validation is not read.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

EXP_ID = 'M2-EDGE-001'
TRIM = 1920   # 120 ms at 16 kHz
CENTER = 6400  # 400 ms at 16 kHz


def views_of(signal):
    n = int(signal.shape[-1])
    full = signal
    if n > TRIM + 1:
        head, tail = signal[TRIM:], signal[:-TRIM]
        trimmed = True
    else:
        head = tail = signal
        trimmed = False
    if n > CENTER:
        start = (n - CENTER) // 2
        mid = signal[start:start + CENTER]
        centered = True
    else:
        mid = signal
        centered = False
    return (full, head, tail, mid), trimmed, centered


def view_probabilities(ssl, model, signal, device, batch_windows=8):
    import torch
    chunks, spans = [], []
    chosen, _, _ = views_of(signal)
    cursor = 0
    for view in chosen:
        windows = [ssl.fit_length(view, start=s) for s in ssl.window_starts(len(view))]
        spans.append((cursor, cursor + len(windows)))
        chunks.extend(windows)
        cursor += len(windows)
    stacked = torch.stack(chunks)
    logits = []
    with torch.inference_mode():
        for batch in stacked.split(batch_windows):
            with torch.autocast(device_type='cuda', enabled=str(device).startswith('cuda')):
                logits.append(model(batch.to(device))[0].float())
        merged = torch.cat(logits)
    return [float(merged[a:b].mean().sigmoid().item()) for a, b in spans]


def smoke():
    import torch
    full = torch.arange(16000)
    chosen, trimmed, centered = views_of(full)
    assert trimmed and centered
    assert len(chosen[0]) == 16000
    assert len(chosen[1]) == 16000 - TRIM and len(chosen[2]) == 16000 - TRIM
    assert len(chosen[3]) == CENTER
    assert int(chosen[1][0]) == TRIM and int(chosen[3][0]) == (16000 - CENTER) // 2
    tiny = torch.zeros(100)
    chosen, trimmed, centered = views_of(tiny)
    assert not trimmed and not centered
    assert all(len(v) == 100 for v in chosen)
    print('smoke OK: edge views stay inside one utterance')


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
    from m2_finetune import load_ssl004, paired, duration_mask, load_parent_probs, prepare_runtime
    import numpy as np
    ssl = load_ssl004()
    run = Path(os.environ['DCC_RUN_ROOT'])
    out, work = args.output, Path(os.environ['LAB_WORK']) / EXP_ID.lower()
    if (out / 'metrics.json').exists() and (out / 'h_eval_done.json').exists():
        print('experiment already complete', flush=True)
        return
    bundle = prepare_runtime(ssl, out, work, run)
    device, model, h_rows = bundle['device'], bundle['model'], bundle['h_rows']
    parent = ssl.load_torch(run / 'lab/M2-SSL-004/best.ckpt')
    model.load_state_dict(parent['model_state_dict'])
    model.to(device).eval()
    progress = out / 'edge_progress.npz'
    done = 0
    matrix = np.zeros((len(h_rows), 4), dtype=np.float64)
    if progress.exists():
        saved = np.load(progress)
        if saved['probs'].shape[1] != 4 or saved['probs'].shape[0] > len(h_rows):
            raise ValueError('edge progress shape mismatch')
        done = int(saved['probs'].shape[0])
        matrix[:done] = saved['probs']
        print(f'resume edge inference at row {done}', flush=True)
    started = time.monotonic()
    fallback_trim = fallback_center = 0
    for idx in range(done, len(h_rows)):
        signal = ssl.load_segment(h_rows[idx])
        _, trimmed, centered = views_of(signal)
        fallback_trim += int(not trimmed)
        fallback_center += int(not centered)
        matrix[idx] = view_probabilities(ssl, model, signal, device)
        if (idx + 1) % 256 == 0 or idx + 1 == len(h_rows):
            tmp = work / 'edge_progress.npz'
            np.savez(tmp, probs=matrix[:idx + 1])
            ssl.publish(tmp, progress)
            print(json.dumps(dict(rows=idx + 1, seconds=round(time.monotonic() - started, 1))), flush=True)
    names = ('full', 'drop_head', 'drop_tail', 'center')
    probs = {name: matrix[:, i] for i, name in enumerate(names)}
    probs['edge_mean'] = matrix.mean(1)
    parent_probs = load_parent_probs(run, h_rows)
    y = np.array([r['speaker'] for r in h_rows])
    summary = ssl.summaries(h_rows, probs['edge_mean'])
    full_summary = ssl.summaries(h_rows, probs['full'])
    parent_summary = ssl.summaries(h_rows, parent_probs)
    slice_rows = {}
    for name in ('<0.5s', '0.5~1s', '1~3s', '>3s'):
        mask = duration_mask(h_rows, name)
        slice_rows[name] = dict(
            rows=int(mask.sum()),
            edge_mean=float(((probs['edge_mean'][mask] >= .5) == y[mask]).mean()) if mask.any() else None,
            full=float(((probs['full'][mask] >= .5) == y[mask]).mean()) if mask.any() else None,
            parent=float(((parent_probs[mask] >= .5) == y[mask]).mean()) if mask.any() else None)
    max_abs = float(np.max(np.abs(probs['full'] - parent_probs)))
    comparison = paired(y, probs['edge_mean'], probs['full'])
    ssl.persist_npy(matrix, 'edge_view_probs.npy', work, out)
    ssl.write_json(out / 'H_probs_columns.json', list(names) + ['edge_mean'])
    local_rows = work / 'H_rows.txt'
    local_rows.write_text(''.join(
        f"{r['call_id']}\t{r['utt_idx']}\t{r['startAt']}\t{r['endAt']}\n" for r in h_rows))
    ssl.publish(local_rows, out / 'H_rows.txt')
    ssl.persist_npy(probs['edge_mean'], 'wavlm_H_probs.npy', work, out)
    eval_done = dict(rows=len(h_rows), seconds=time.monotonic() - started, device=device,
                     fallback_trim=fallback_trim, fallback_center=fallback_center,
                     max_abs_full_vs_ssl004=max_abs)
    ssl.write_json(out / 'h_eval_done.json', eval_done)
    adopted = (summary['accuracy'] - full_summary['accuracy'] >= 0.002
               and comparison['p_value'] < 0.05
               and comparison['left_only_correct'] > comparison['right_only_correct']
               and slice_rows['0.5~1s']['edge_mean'] >= slice_rows['0.5~1s']['full']
               and slice_rows['>3s']['edge_mean'] >= slice_rows['>3s']['full'] - 0.002)
    metrics = dict(
        exp_id=EXP_ID,
        primary_metric=dict(name='accuracy', value=summary['accuracy'], split='internal_holdout'),
        metrics=dict(summary, full_view=full_summary, parent_ssl004=parent_summary,
                     slices=slice_rows, mcnemar_edge_vs_full=comparison,
                     reproduction=eval_done, adopted=bool(adopted),
                     counts=dict(H_rows=len(h_rows))),
        seed=42, train_rows=0, eval_rows=len(h_rows), official_val=None,
        ckpt='lab/M2-SSL-004/best.ckpt',
        notes='Frozen SSL-004. Four views of the target utterance only. Probability mean, threshold 0.5. No official Validation.',
        label_source='provided_json', split_sha256=bundle['split']['sha256'])
    ssl.write_json(out / 'metrics.json', metrics)
    ssl.save_torch(dict(exp_id=EXP_ID, parent='M2-SSL-004/best.ckpt', no_new_weights=True,
                        adopted=bool(adopted)), 'best.ckpt', work, out)
    print(json.dumps(metrics['primary_metric']), flush=True)


if __name__ == '__main__':
    main()
