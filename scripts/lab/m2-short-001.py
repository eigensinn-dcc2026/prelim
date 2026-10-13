#!/usr/bin/env python3
"""M2-SHORT-001: one more epoch of M2-SSL-004, drawing 70% of steps from utterances under 1.2 s.

Duration is used only to build the index list. The network input stays the waveform window.
The transformer is trainable on this epoch (SSL-004 froze it only on its first epoch from scratch).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

EXP_ID = 'M2-SHORT-001'


def smoke():
    from m2_finetune import short_epoch_order
    rows = []
    for i in range(10):
        rows.append(dict(call_id=f'c{i}', startAt=0, endAt=500, speaker=i % 2))
        rows.append(dict(call_id=f'c{i}', startAt=0, endAt=3000, speaker=i % 2))
    order = short_epoch_order(rows, 0)
    assert len(order) == 40
    short = sum((rows[i]['endAt'] - rows[i]['startAt']) < 1200 for i in order)
    assert short == 28
    assert short_epoch_order(rows, 0) == order
    print('smoke OK: 70% of continuation steps are the utterance\'s own short span')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=os.environ.get('LAB_OUT'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--micro-batch-size', type=int, choices=(1, 2, 4, 8, 16, 32), default=32)
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    if args.output is None:
        parser.error('--output or LAB_OUT required')
    from m2_finetune import (
        bce_supcon, copied_windows, duration_mask, finetune, init_from_ssl004,
        load_parent_probs, load_ssl004, paired, prepare_runtime, short_epoch_order)
    import numpy as np
    ssl = load_ssl004()
    run = Path(os.environ['DCC_RUN_ROOT'])
    out, work = args.output, Path(os.environ['LAB_WORK']) / EXP_ID.lower()
    if (out / 'metrics.json').exists() and (out / 'h_eval_done.json').exists():
        print('experiment already complete', flush=True)
        return
    bundle = prepare_runtime(ssl, out, work, run)
    model, device = bundle['model'], bundle['device']
    if not (out / 'ckpt_last.pt').exists():
        init_from_ssl004(ssl, model, run)
    loss_fn = lambda logits, embedding, labels: bce_supcon(ssl, logits, embedding, labels)
    if not (out / 'train_done.json').exists():
        finetune(ssl, model, bundle['train_rows'], bundle['d_rows'], out, work, device,
                 bundle['split']['sha256'], copied_windows(ssl), short_epoch_order, loss_fn,
                 EXP_ID, 'wavlm-large-short-oversample', args.micro_batch_size)
    done = json.loads((out / 'train_done.json').read_text())
    best = ssl.load_torch(out / 'best.ckpt')
    model.load_state_dict(best['model_state_dict'])
    model.to(device)
    h_rows = bundle['h_rows']
    probabilities, seconds = ssl.evaluate(model, h_rows, device)
    ssl.persist_npy(probabilities, 'wavlm_H_probs.npy', work, out)
    parent = load_parent_probs(run, h_rows)
    y = np.array([r['speaker'] for r in h_rows])
    summary = ssl.summaries(h_rows, probabilities)
    parent_summary = ssl.summaries(h_rows, parent)
    slices = {}
    for name in ('<0.5s', '0.5~1s', '1~3s', '>3s'):
        mask = duration_mask(h_rows, name)
        slices[name] = dict(
            rows=int(mask.sum()),
            model=float(((probabilities[mask] >= .5) == y[mask]).mean()) if mask.any() else None,
            parent=float(((parent[mask] >= .5) == y[mask]).mean()) if mask.any() else None)
    comparison = paired(y, probabilities, parent)
    adopted = (summary['accuracy'] - parent_summary['accuracy'] >= 0.002
               and comparison['p_value'] < 0.05
               and comparison['left_only_correct'] > comparison['right_only_correct']
               and slices['0.5~1s']['model'] >= slices['0.5~1s']['parent'])
    ssl.write_json(out / 'h_eval_done.json', dict(seconds=seconds, device=device))
    local_rows = work / 'H_rows.txt'
    local_rows.write_text(''.join(
        f"{r['call_id']}\t{r['utt_idx']}\t{r['startAt']}\t{r['endAt']}\n" for r in h_rows))
    ssl.publish(local_rows, out / 'H_rows.txt')
    metrics = dict(
        exp_id=EXP_ID,
        primary_metric=dict(name='accuracy', value=summary['accuracy'], split='internal_holdout'),
        metrics=dict(summary, parent_ssl004=parent_summary, slices=slices,
                     mcnemar_vs_ssl004=comparison, adopted=bool(adopted),
                     epoch_history=done['history'], best_epoch=best['best_epoch'],
                     sampler=dict(short_fraction=0.7, short_sec=1.2, draws_per_call=4)),
        seed=42, train_rows=len(bundle['train_rows']), eval_rows=len(h_rows),
        official_val=None, ckpt='best.ckpt',
        notes='One epoch from SSL-004. 70% of steps are utterances shorter than 1.2s. Duration is not a model input. Provided labels. Threshold 0.5.',
        label_source='provided_json', split_sha256=bundle['split']['sha256'])
    ssl.write_json(out / 'metrics.json', metrics)
    print(json.dumps(metrics['primary_metric']), flush=True)


if __name__ == '__main__':
    main()
