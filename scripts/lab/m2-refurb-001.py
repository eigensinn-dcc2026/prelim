#!/usr/bin/env python3
"""M2-REFURB-001: one epoch from M2-SSL-004 with confident-only soft targets.

When max(p, 1-p) >= 0.95, the BCE target is 0.9 * provided label + 0.1 * p.
Otherwise the provided label is used as-is. SupCon still uses the provided label.
JSON labels are not edited, and text is not read.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

EXP_ID = 'M2-REFURB-001'


def smoke():
    import torch
    from m2_finetune import refurb_targets
    logits = torch.tensor([4., 0., -4.])
    labels = torch.tensor([0., 1., 1.])
    target = refurb_targets(logits, labels)
    prob = torch.sigmoid(logits)
    assert torch.allclose(target[0], 0.1 * prob[0], atol=1e-6)
    assert target[1].item() == 1.
    assert torch.allclose(target[2], 0.9 + 0.1 * prob[2], atol=1e-6)
    print('smoke OK: soft target only when the utterance probability is confident')


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
        copied_windows, duration_mask, finetune, init_from_ssl004, load_parent_probs,
        load_ssl004, paired, prepare_runtime, refurb_loss)
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

    def loss_fn(logits, embedding, labels):
        return refurb_loss(ssl, logits, embedding, labels)

    order_fn = ssl.epoch_order
    if not (out / 'train_done.json').exists():
        finetune(ssl, model, bundle['train_rows'], bundle['d_rows'], out, work, device,
                 bundle['split']['sha256'], copied_windows(ssl), order_fn, loss_fn,
                 EXP_ID, 'wavlm-large-confident-refurb', args.micro_batch_size)
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
    overlap_ok = summary['overlap_accuracy']['overlap']['accuracy'] >= parent_summary['overlap_accuracy']['overlap']['accuracy']
    adopted = (summary['accuracy'] - parent_summary['accuracy'] >= 0.002
               and comparison['p_value'] < 0.05
               and comparison['left_only_correct'] > comparison['right_only_correct']
               and overlap_ok)
    ssl.write_json(out / 'h_eval_done.json', dict(seconds=seconds, device=device, overlap_ok=bool(overlap_ok)))
    local_rows = work / 'H_rows.txt'
    local_rows.write_text(''.join(
        f"{r['call_id']}\t{r['utt_idx']}\t{r['startAt']}\t{r['endAt']}\n" for r in h_rows))
    ssl.publish(local_rows, out / 'H_rows.txt')
    metrics = dict(
        exp_id=EXP_ID,
        primary_metric=dict(name='accuracy', value=summary['accuracy'], split='internal_holdout'),
        metrics=dict(summary, parent_ssl004=parent_summary, slices=slices,
                     mcnemar_vs_ssl004=comparison, overlap_gate_pass=bool(overlap_ok),
                     adopted=bool(adopted), epoch_history=done['history'],
                     best_epoch=best['best_epoch'],
                     refurb=dict(confidence=0.95, mix=0.1, hard_label_weight=0.9)),
        seed=42, train_rows=len(bundle['train_rows']), eval_rows=len(h_rows),
        official_val=None, ckpt='best.ckpt',
        notes='One epoch from SSL-004. Soft targets only at confidence >= 0.95. Provided JSON labels unchanged. Threshold 0.5. Overlap accuracy must not fall.',
        label_source='provided_json', split_sha256=bundle['split']['sha256'])
    ssl.write_json(out / 'metrics.json', metrics)
    print(json.dumps(metrics['primary_metric']), flush=True)


if __name__ == '__main__':
    main()
