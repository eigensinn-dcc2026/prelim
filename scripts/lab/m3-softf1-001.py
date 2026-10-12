"""M3-SOFTF1-001: fixed seed42, epoch4 pilot; no new best weights or Validation."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time

_spec = importlib.util.spec_from_file_location('roberta003', Path(__file__).with_name('m3-roberta-003.py'))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
EXP_ID = 'M3-SOFTF1-001'
CONFIG = dict(exp_id=EXP_ID, seed=42, epoch=4, lr=1e-6, weight_decay=.01,
              max_length=256, threshold=.5, arms=['A16', 'B16', 'A64', 'B64'],
              source='M3-ROBERTA-003', gradient_checkpointing=True)


def publish(local, destination):
    """Stage on local disk, then retry verified Drive copy and atomic replacement."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    for attempt in range(6):
        try:
            shutil.copyfile(local, temporary)
            if temporary.stat().st_size != Path(local).stat().st_size:
                raise OSError('Copy size mismatch')
            os.replace(temporary, destination)
            return
        except OSError:
            if attempt == 5:
                raise
            time.sleep(2 ** (attempt + 1))


def save_tensor(obj, destination, work):
    import torch
    fd, name = tempfile.mkstemp(dir=work, suffix='.pt')
    os.close(fd)
    try:
        torch.save(obj, name)
        publish(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def grouped_loss(logits, targets, criterion, global_f1):
    import torch
    if global_f1:
        return base.loss_terms(torch.cat(logits), torch.cat(targets), criterion)[0]
    count = sum(len(y) for y in targets)
    return sum(base.loss_terms(x, y, criterion)[0] * (len(y) / count)
               for x, y in zip(logits, targets))


def score(probs, truth):
    import numpy as np
    from sklearn.metrics import f1_score
    pred = (probs >= .5).astype(int)
    eces = []
    for j in range(len(base.LABELS)):
        bins = np.minimum((probs[:, j] * 15).astype(int), 14)
        eces.append(sum(float(mask.mean()) * abs(float(probs[mask, j].mean() - truth[mask, j].mean()))
                        for k in range(15) if (mask := bins == k).any()))
    macro = float(f1_score(truth, pred, average='macro', zero_division=0))
    empty = int((pred.sum(1) == 0).sum())
    return dict(macro_f1=macro, micro_f1=float(f1_score(truth, pred, average='micro', zero_division=0)),
                per_symptom_f1=dict(zip(base.LABELS, f1_score(truth, pred, average=None, zero_division=0).tolist())),
                empty_predictions=empty, ece=float(np.mean(eces)), per_symptom_ece=dict(zip(base.LABELS, eces)),
                brier=float(np.mean((probs - truth) ** 2)),
                probability_minus_prevalence=dict(zip(base.LABELS, (probs.mean(0) - truth.mean(0)).tolist())),
                training_collapse=empty > len(truth) * .5 or macro < .3)


def make_model(path):
    import torch
    from transformers import AutoModel
    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = AutoModel.from_pretrained(path, local_files_only=True)
            self.bert.gradient_checkpointing_enable()
            self.dropout = torch.nn.Dropout(.1)
            self.classifier = torch.nn.Linear(self.bert.config.hidden_size, len(base.LABELS))
        def forward(self, ids, mask):
            return self.classifier(self.dropout(self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]))
    return Classifier()


def load_initial(model, checkpoint):
    import torch
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state.get('seed') != 42 or state.get('epoch') != 3 or state.get('config') != base.CONFIG:
        raise ValueError('Source must be ROBERTA-003 seed42 final epoch3 checkpoint')
    model.load_state_dict(state['model'])


def evaluate(model, tokens, targets, holdout, micro):
    import torch
    model.eval()
    probabilities = []
    with torch.no_grad():
        for offset in range(0, len(holdout), micro):
            batch = holdout[offset:offset + micro]
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(tokens['input_ids'][batch].cuda(), tokens['attention_mask'][batch].cuda())
            probabilities.append(logits.float().sigmoid().cpu())
    return torch.cat(probabilities).numpy()


def run_arm(name, micro, out, work, model_path, source, tokens, targets, train, holdout, pos_weight):
    import numpy as np
    import torch
    arm = out / ('arm_' + name)
    arm.mkdir(exist_ok=True)
    identity = dict(CONFIG, arm=name, micro_batch_size=micro)
    identity_file = arm / 'config.json'
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        old = json.loads(identity_file.read_text())
        if name.endswith('64') and old == dict(CONFIG, arm=name, micro_batch_size=8) and micro == 4:
            for filename in ('done.json', 'metrics.json', 'holdout_probs.npy', 'ckpt_last.pt'):
                (arm / filename).unlink(missing_ok=True)
        else:
            raise ValueError('Arm configuration mismatch')
    base.write_json(identity_file, identity)
    done = arm / 'done.json'
    if done.exists():
        if json.loads(done.read_text()) != identity or not (arm / 'holdout_probs.npy').exists():
            raise ValueError('Invalid completed arm artifacts')
        return json.loads((arm / 'metrics.json').read_text())
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    base.seed_all(42, torch, np)
    model = make_model(model_path)
    load_initial(model, source)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    scaler = torch.amp.GradScaler('cuda')
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device='cuda'))
    generator = torch.Generator().manual_seed(42)
    # Match the original generator's fourth permutation; all arms share this order.
    for _ in range(3):
        torch.randperm(len(train), generator=generator)
    order = torch.tensor(train)[torch.randperm(len(train), generator=generator)]
    offset, elapsed, steps, peak = 0, 0., 0, 0
    last = arm / 'ckpt_last.pt'
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=False)
        if state['identity'] != identity or not torch.equal(state['order'], order):
            raise ValueError('Resume configuration/order mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        offset, elapsed, steps, peak = state['offset'], state['elapsed'], state['steps'], state['peak']
        random.setstate(state['rng']['random'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        torch.cuda.set_rng_state_all(state['rng']['cuda'])
        del state
    # Initialization and checkpoint loading must not alter the common dropout RNG.
    else:
        base.seed_all(42, torch, np)
    def checkpoint():
        save_tensor(dict(identity=identity, epoch=4, offset=offset, order=order,
                         model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                         elapsed=elapsed, steps=steps, peak=peak,
                         rng=dict(random=random.getstate(), numpy=np.random.get_state(),
                                  torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())), last, work)
    effective = int(name[1:])
    model.train()
    torch.cuda.reset_peak_memory_stats()
    saved_at = time.monotonic()
    while offset < len(order):
        torch.cuda.synchronize()
        started = time.monotonic()
        group = order[offset:offset + effective]
        optimizer.zero_grad(set_to_none=True)
        logits_list, target_list = [], []
        for start in range(0, len(group), micro):
            batch = group[start:start + micro]
            target = targets[batch].cuda()
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(tokens['input_ids'][batch].cuda(), tokens['attention_mask'][batch].cuda())
            if name.startswith('A'):
                loss = base.loss_terms(logits, target, criterion)[0] * (len(batch) / len(group))
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss')
                scaler.scale(loss).backward()
            else:
                logits_list.append(logits)
                target_list.append(target)
        if name.startswith('B'):
            loss = grouped_loss(logits_list, target_list, criterion, True)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            scaler.scale(loss).backward()
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            scheduler.step()
        # Drop graph references before saving/evaluation.
        del logits_list, target_list, logits, loss
        offset += len(group)
        steps += 1
        torch.cuda.synchronize()
        elapsed += time.monotonic() - started
        peak = max(peak, torch.cuda.max_memory_allocated())
        if time.monotonic() - saved_at >= 900 or offset == len(order):
            checkpoint()
            saved_at = time.monotonic()
        if steps % 100 == 0:
            print(f'{name}: rows={offset}/{len(order)} steps={steps}', flush=True)
    probs = evaluate(model, tokens, targets, holdout, micro)
    result = dict(score(probs, targets[holdout].numpy()), arm=name, micro_batch_size=micro,
                  effective_batch_size=effective, optimizer_steps=steps, seconds_per_step=elapsed / steps,
                  training_seconds=elapsed, max_gpu_memory_bytes=peak, seed=42)
    base.atomic_write(arm / 'holdout_probs.npy', lambda p: write_numpy(p, probs))
    base.write_json(arm / 'metrics.json', result)
    base.write_json(done, identity)
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return result


def write_numpy(path, array):
    import numpy as np
    with open(path, 'wb') as handle:
        np.save(handle, array)


def smoke():
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != 'torch':
            raise
        assert base.CONFIG['soft_f1_weight'] == 1.0
        assert base.CONFIG['micro_batch_size'] == 8
        assert CONFIG['lr'] == 1e-6 and CONFIG['seed'] == 42
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            local = root / 'local'
            local.write_bytes(b'checkpoint staging smoke')
            publish(local, root / 'out' / 'checkpoint')
            assert (root / 'out' / 'checkpoint').read_bytes() == local.read_bytes()
        print('smoke PARTIAL: import/config/staged copy OK; torch unavailable, CPU loss/gradient checks NOT RUN')
        return
    torch.set_num_threads(1)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1., 2., 3.]))
    truth = torch.tensor([[1., 0., 1.], [0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
    values = torch.arange(15).reshape(5, 3).float() / 5 - 1
    for global_f1 in (False, True):
        x = values.clone().requires_grad_()
        actual = grouped_loss(list(x.split(2)), list(truth.split(2)), criterion, global_f1)
        grad = torch.autograd.grad(actual, x)[0]
        y = values.clone().requires_grad_()
        if global_f1:
            expected = base.loss_terms(y, truth, criterion)[0]
        else:
            expected = sum(base.loss_terms(y[i:i+2], truth[i:i+2], criterion)[0] * len(truth[i:i+2]) / 5
                           for i in range(0, 5, 2))
        expected_grad = torch.autograd.grad(expected, y)[0]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(grad, expected_grad)
        assert torch.isfinite(grad).all()
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        save_tensor({'offset': 5, 'order': torch.arange(5)}, root / 'out' / 'ckpt_last.pt', root)
        assert torch.load(root / 'out' / 'ckpt_last.pt', weights_only=True)['offset'] == 5
    print('smoke OK: global loss/gradient equivalence, baseline micro loss/gradients, partial group, staged checkpoint; no training')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=os.environ.get('LAB_OUT'))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    if not args.out:
        parser.error('--out or LAB_OUT required')
    if os.environ.get('LAB_EXP_ID', EXP_ID) != EXP_ID:
        raise ValueError('LAB_EXP_ID mismatch')
    model_path, tokenizer = base.prepare_model()  # Fail before preparing any data.
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This pilot requires a CUDA GPU')
    out = Path(args.out)
    work = Path(os.environ['LAB_WORK']) / EXP_ID.lower()
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    config_file = out / 'config.json'
    if config_file.exists() and json.loads(config_file.read_text()) != CONFIG:
        raise ValueError('Output configuration mismatch')
    base.write_json(config_file, CONFIG)
    source_root = Path(os.environ['DCC_RUN_ROOT']) / 'lab' / 'M3-ROBERTA-003'
    source = source_root / 'seed_42' / 'best.ckpt'
    for path in (source, source_root / 'rows.json', source_root / 'split.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    # Restore existing data/split; never silently generate a fresh split.
    for filename in ('rows.json', 'split.json'):
        source_file = source_root / filename
        destination = out / filename
        if destination.exists():
            if destination.read_bytes() != source_file.read_bytes():
                raise ValueError(f'Source changed: {filename}')
        else:
            base.atomic_write(destination, lambda p: shutil.copyfile(source_file, p))
    rows = base.prepare_rows(out, work)
    train, holdout = base.prepare_split(out, rows)
    if len(holdout) != 5792:
        raise ValueError('Expected exactly 5792 holdout rows')
    holdout = sorted(holdout, key=lambda i: rows[i]['file'])
    pos_weight = base.weights([rows[i]['labels'] for i in train])
    base.write_json(out / 'pos_weight.json', dict(zip(base.LABELS, pos_weight)))
    cache = out / 'tokens.pt'
    if cache.exists():
        tokens = torch.load(cache, map_location='cpu', weights_only=True)
    else:
        tokens = dict(tokenizer([r['text'] for r in rows], padding='max_length', truncation=True,
                                max_length=256, return_token_type_ids=False, return_tensors='pt'))
        save_tensor(tokens, cache, work)
    targets = torch.tensor([r['labels'] for r in rows], dtype=torch.float32)
    initial_file = out / 'initial_metrics.json'
    if initial_file.exists():
        initial = json.loads(initial_file.read_text())
    else:
        model = make_model(model_path)
        load_initial(model, source)
        model.cuda()
        initial = score(evaluate(model, tokens, targets, holdout, 8), targets[holdout].numpy())
        base.write_json(initial_file, initial)
        del model
        torch.cuda.empty_cache()
    results = {}
    for effective in (16, 64):
        micro_file = out / f'pair_{effective}.json'
        micro = json.loads(micro_file.read_text())['micro_batch_size'] if micro_file.exists() else 8
        base.write_json(micro_file, dict(micro_batch_size=micro))
        while True:
            try:
                for prefix in ('A', 'B'):
                    name = f'{prefix}{effective}'
                    results[name] = run_arm(name, micro, out, work, model_path, source, tokens,
                                            targets, train, holdout, pos_weight)
                break
            except torch.cuda.OutOfMemoryError:
                if effective != 64 or micro != 8:
                    raise
                # Restart BOTH members at the source checkpoint; never compare micro8 vs micro4.
                micro = 4
                base.write_json(micro_file, dict(micro_batch_size=micro, reason='OOM at micro8; both arms restarted'))
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                for prefix in ('A', 'B'):
                    arm = out / f'arm_{prefix}{effective}'
                    for filename in ('done.json', 'metrics.json', 'holdout_probs.npy', 'ckpt_last.pt'):
                        (arm / filename).unlink(missing_ok=True)
    comparisons = {}
    for effective in (16, 64):
        a, b = results[f'A{effective}'], results[f'B{effective}']
        delta = b['macro_f1'] - a['macro_f1']
        degradation = {key: (b[key] / a[key] - 1 if a[key] else (0. if b[key] == 0 else None))
                       for key in ('ece', 'brier')}
        comparisons[str(effective)] = dict(macro_f1_delta=delta, calibration_relative_degradation=degradation,
                                           calibration_within_10_percent=all(v is not None and v <= .1 for v in degradation.values()),
                                           meets_f1_gain=delta >= .005,
                                           within_seed_variation=abs(delta) <= .004)
    reproduced = abs(initial['macro_f1'] - .6441) <= .001
    collapse = any(r['training_collapse'] for r in results.values())
    verdict = 'reproduction_failed_defer' if not reproduced else ('training_collapse_failed' if collapse else
               ('eligible_for_three_seed_extension' if any(c['meets_f1_gain'] for c in comparisons.values()) else 'no_extension_signal'))
    base.write_json(out / 'metrics.json', dict(exp_id=EXP_ID,
        primary_metric=dict(name='macro_f1', value=results['A16']['macro_f1'], split='internal_holdout'),
        metrics=dict(arms=results, initial_checkpoint=initial, reproduction_ok=reproduced,
                     comparisons=comparisons, training_collapse=collapse, verdict=verdict),
        seed=42, train_rows=len(train), eval_rows=len(holdout), official_val=None, ckpt=None,
        notes='Pilot: no new best.ckpt; probabilities only plus resume checkpoints. Primary metric is fixed A16 control. '
              'Threshold >=0.5; no postprocessing; no official Validation. Compare only within effective batch pairs. '
              'Single seed differences <=0.004 are inconclusive. Calibration degradation is reported for extension review.'))


if __name__ == '__main__':
    main()
