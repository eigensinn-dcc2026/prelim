"""M3-STRUCT-001: fixed ROBERTA-003 split, pairwise joint head, preset A.
Structured head/loss adapted from origin/sooyah/lab-m3-kfold-teacher
scripts/lab/m3_structured/{structured_head,losses,train_structured}.py.
Only transcription text is input; final epoch, fixed marginal threshold >= .5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import time
import unicodedata
import zipfile

EXP_ID = "M3-STRUCT-001"
MODEL_ID = "klue/roberta-large"
# Immutable safetensors addition commit, rather than the moving main ref:
# https://huggingface.co/klue/roberta-large/commits/main
REVISION = "28d9112"
LABELS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
SEEDS = [42, 43, 44]
CONFIG = dict(exp_id=EXP_ID, model=MODEL_ID, revision=REVISION, max_length=256,
              epochs=3, lr=1e-5, dropout=0.1, batch_size=16, micro_batch_size=8,
              accumulation_steps=2, weight_decay=0.01, warmup_ratio=0.1, fp16=True, threshold=0.5,
              split_seed=42, holdout_ratio=0.2, labels=LABELS,
              weight_formula="clip(0.5 * sqrt(neg / max(pos, 1)), 1, 6)",
              baseline_source="M3-ROBERTA-003", baseline_macro_f1=0.6427,
              soft_f1_weight=1.0, soft_f1_epsilon=1e-7, threshold_operator=">=",
              soft_f1_batch_size=8)



CONFIG.update(w_lr=1e-2, lambda_set=0, lambda_bce=1, lambda_f1=1, pairwise=True, exclude_empty=False, grad_ckpt=True)
BASELINES = {42: .6441, 43: .6404, 44: .6437}

def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        writer(Path(name))
        with open(name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path, obj):
    atomic_write(path, lambda p: p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"))


def weights(labels):
    if not labels:
        raise ValueError("Empty training split")
    positive = [sum(row[j] for row in labels) for j in range(len(LABELS))]
    return [min(6.0, max(1.0, math.sqrt((len(labels) - p) / max(p, 1)) * .5)) for p in positive]


def transcript(doc):
    return " ".join(text for u in doc.get("utterances", []) if isinstance(u, dict)
                    for text in [re.sub(r"\s+", " ", u.get("text", "")).strip()
                                 if isinstance(u.get("text", ""), str) else ""] if text)


def prepare_model():
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    root = Path(os.environ["DCC_RUN_ROOT"]) / "models" / "roberta-large"
    marker = root / "dcc_revision.json"
    if marker.exists():
        if json.loads(marker.read_text()) != {"repo_id": MODEL_ID, "revision": REVISION}:
            raise ValueError(f"Model cache revision differs: {root}")
    else:
        snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=str(root),
                          allow_patterns=["config.json", "tokenizer*", "vocab.txt", "model.safetensors"])
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    # Fail before touching the data if the cached model cannot actually be loaded.
    probe = AutoModel.from_pretrained(root, local_files_only=True)
    del probe
    write_json(marker, {"repo_id": MODEL_ID, "revision": REVISION})
    return root, tokenizer


def seed_all(seed, torch, np):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def publish(local, destination):
    """Stage on local disk; retry five times after the initial FUSE failure."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    for attempt in range(6):
        try:
            shutil.copyfile(local, temporary)
            if temporary.stat().st_size != Path(local).stat().st_size:
                raise OSError('Checkpoint copy size mismatch')
            os.replace(temporary, destination)
            return
        except OSError:
            if attempt == 5:
                raise
            time.sleep(2 ** (attempt + 1))


def save_tensor(obj, destination, work):
    import torch
    local = work / (Path(destination).name + '.local')
    torch.save(obj, local)
    try:
        publish(local, destination)
    finally:
        local.unlink(missing_ok=True)


def structured_head(hidden):
    """36 trainable upper-triangle entries; full 512-state support, fp32."""
    import itertools
    import torch

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unary = torch.nn.Sequential(torch.nn.Dropout(.1), torch.nn.Linear(hidden, 9))
            self.W = torch.nn.Parameter(torch.zeros(36))
            self.register_buffer('Y', torch.tensor(list(itertools.product([0., 1.], repeat=9))))
            self.register_buffer('pairs', torch.triu_indices(9, 9, offset=1))
            self.register_buffer('pair_features', self.Y[:, self.pairs[0]] * self.Y[:, self.pairs[1]])

        def matrix(self):
            matrix = self.W.new_zeros(9, 9)
            matrix[self.pairs[0], self.pairs[1]] = self.W
            return matrix + matrix.T

        def forward(self, h):
            with torch.autocast(device_type=h.device.type, enabled=False):
                s = self.unary(h.float())
                logp = (s @ self.Y.T + self.pair_features @ self.W).log_softmax(-1)
                return logp.exp() @ self.Y
    return Head()


def loss_terms(probs, targets, pos_weight):
    # Preset A from structured losses.py: no set CE, no distillation or EMA.
    p, y = probs.float().clamp(1e-6, 1 - 1e-6), targets.float()
    bce = -(pos_weight * y * p.log() + (1 - y) * (1 - p).log()).mean()
    soft = 1 - (2 * (probs.float() * y).sum(0) /
                (probs.float().sum(0) + y.sum(0) + 1e-7)).mean()
    return bce + soft, bce, soft


def scores(probs, truth):
    import numpy as np
    from sklearn.metrics import f1_score
    if probs.shape != truth.shape or not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError('Invalid holdout probabilities')
    pred = (probs >= .5).astype(int)
    per = f1_score(truth, pred, average=None, zero_division=0)
    ece = []
    for j in range(9):
        bins = np.minimum((probs[:, j] * 15).astype(int), 14)
        ece.append(sum(float((bins == b).mean()) * abs(float(probs[bins == b, j].mean()) -
                   float(truth[bins == b, j].mean())) for b in range(15) if (bins == b).any()))
    brier = ((probs - truth) ** 2).mean(0)
    empty = int((pred.sum(1) == 0).sum())
    return dict(macro_f1=float(per.mean()), micro_f1=float(f1_score(truth, pred, average='micro', zero_division=0)),
                per_symptom_f1=dict(zip(LABELS, per.tolist())), empty_predictions=empty,
                empty_prediction_rate=empty / len(truth), ece=float(np.mean(ece)),
                per_symptom_ece=dict(zip(LABELS, ece)), brier=float(brier.mean()),
                per_symptom_brier=dict(zip(LABELS, brier.tolist())),
                probability_minus_prevalence=dict(zip(LABELS, (probs.mean(0) - truth.mean(0)).tolist())),
                training_collapse=empty > len(truth) / 2 or float(per.mean()) < .3)


def baseline_rows(out):
    source = Path(os.environ['DCC_RUN_ROOT']) / 'lab' / 'M3-ROBERTA-003'
    rows = json.loads((source / 'rows.json').read_text(encoding='utf-8'))
    split = json.loads((source / 'split.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if split.get('rows_sha256') != digest:
        raise ValueError('ROBERTA-003 rows/split digest mismatch')
    train, holdout = split['train'], split['holdout']
    if len(holdout) != 5792 or not train or sorted(train + holdout) != list(range(len(rows))):
        raise ValueError('Expected original ROBERTA-003 partition (5792 holdout)')
    if {rows[i]['call_id'] for i in train} & {rows[i]['call_id'] for i in holdout}:
        raise ValueError('call_id leakage')
    for r in rows:
        if not isinstance(r['text'], str) or len(r['labels']) != 9 or any(x not in (0, 1) for x in r['labels']):
            raise ValueError('Invalid baseline row')
    for name, obj in [('rows.json', rows), ('split.json', split)]:
        path = out / name
        if path.exists() and json.loads(path.read_text()) != obj:
            raise ValueError('Baseline artifacts changed since this run started')
        if not path.exists():
            write_json(path, obj)
    return rows, train, holdout, digest


def run_seed(seed, out, work, model_path, tokens, targets, train, holdout, pos_weight, digest, rows):
    import numpy as np
    import torch
    from transformers import AutoModel, get_linear_schedule_with_warmup
    folder = out / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / 'done.json'
    required = ['holdout_probs.npy', 'W.npy', 'val_files.txt', 'metrics.json']
    if done.exists():
        if json.loads(done.read_text()) != dict(config=CONFIG, rows_sha256=digest, seed=seed):
            raise ValueError('Completed seed configuration mismatch')
        if not all((folder / name).exists() for name in required) or (seed == 42 and not (out / 'best.ckpt').exists()):
            raise ValueError('Completed seed missing artifacts')
        return json.loads((folder / 'metrics.json').read_text())
    seed_all(seed, torch, np)
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA; use --smoke for CPU verification')
    device = torch.device('cuda')

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained(model_path, local_files_only=True)
            self.enc.gradient_checkpointing_enable()
            self.head = structured_head(self.enc.config.hidden_size)

        def forward(self, ids, mask):
            return self.head(self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])

    model = Classifier().to(device)
    optimizer = torch.optim.AdamW([
        dict(params=model.enc.parameters(), lr=1e-5),
        dict(params=model.head.unary.parameters(), lr=1e-5),
        dict(params=[model.head.W], lr=1e-2, weight_decay=0.)], weight_decay=.01)
    updates = math.ceil(len(train) / 16) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(updates * .1), updates)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    generator = torch.Generator().manual_seed(seed)
    pw = torch.tensor(pos_weight, device=device)
    last = folder / 'ckpt_last.pt'
    epoch, offset, order, totals, history, elapsed, epoch_elapsed = 1, 0, None, [0., 0., 0.], [], 0., 0.
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=False)
        if state['config'] != CONFIG or state['rows_sha256'] != digest or state['seed'] != seed:
            raise ValueError('Resume configuration mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        epoch, offset, order = state['epoch'], state['offset'], state['order']
        totals, history, elapsed, epoch_elapsed = state['totals'], state['history'], state['elapsed'], state['epoch_elapsed']
        rng = state['rng']
        random.setstate(rng['random']); np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state_all(rng['cuda'])
        generator.set_state(rng['generator'])
        del state
    started = time.monotonic()
    epoch_started = started
    last_saved = started
    ids, masks = tokens['input_ids'], tokens['attention_mask']

    def checkpoint():
        state = dict(config=CONFIG, rows_sha256=digest, seed=seed, epoch=epoch, offset=offset,
                     order=order, totals=totals, history=history,
                     elapsed=elapsed + time.monotonic() - started,
                     epoch_elapsed=epoch_elapsed + time.monotonic() - epoch_started,
                     model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                     rng=dict(random=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                              cuda=torch.cuda.get_rng_state_all(), generator=generator.get_state()))
        save_tensor(state, last, work)

    def predict():
        model.eval()
        batches = []
        with torch.no_grad():
            for start in range(0, len(holdout), 8):
                batch = holdout[start:start + 8]
                with torch.autocast('cuda', dtype=torch.float16):
                    p = model(ids[batch].to(device), masks[batch].to(device))
                batches.append(p.cpu().numpy())
        return np.concatenate(batches)

    while epoch <= 3:
        if order is None:
            order = torch.tensor(train)[torch.randperm(len(train), generator=generator)]
        model.train()
        while offset < len(order):
            group = order[offset:offset + 16]
            optimizer.zero_grad(set_to_none=True)
            for m in range(0, len(group), 8):
                batch = group[m:m + 8]
                with torch.autocast('cuda', dtype=torch.float16):
                    probs = model(ids[batch].to(device), masks[batch].to(device))
                loss, bce, soft = loss_terms(probs, targets[batch].to(device), pw)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                scaler.scale(loss * len(batch) / len(group)).backward()
                for j, term in enumerate([loss, bce, soft]):
                    totals[j] += term.item() * len(batch)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            offset += len(group)  # No pending accumulated gradients at checkpoint boundaries.
            optimizer.zero_grad(set_to_none=True)
            if time.monotonic() - last_saved >= 900:
                checkpoint()
                last_saved = time.monotonic()
            if offset % 3200 == 0:
                print(f'seed={seed} epoch={epoch} rows={offset} loss={loss.item():.4f}', flush=True)
        probs = predict()
        result = dict(scores(probs, targets[holdout].numpy()), seed=seed, epoch=epoch,
                      loss=totals[0] / len(train), bce=totals[1] / len(train),
                      soft_f1_loss=totals[2] / len(train),
                      epoch_seconds=epoch_elapsed + time.monotonic() - epoch_started)
        history.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        epoch += 1
        offset, order, totals, epoch_elapsed = 0, None, [0., 0., 0.], 0.
        epoch_started = time.monotonic()
        checkpoint()
        last_saved = time.monotonic()
        write_json(folder / 'history.json', history)
    # A crash after final epoch checkpoint simply re-evaluates final weights.
    probs = predict()
    matrix = model.head.matrix().detach().cpu().numpy()
    def save_array(path, array):
        def writer(temporary):
            with temporary.open('wb') as handle:
                np.save(handle, array)
        atomic_write(path, writer)
    save_array(folder / 'holdout_probs.npy', probs)
    save_array(folder / 'W.npy', matrix)
    atomic_write(folder / 'val_files.txt', lambda p: p.write_text(
        '\n'.join(str(rows[i]['call_id']) for i in holdout) + '\n', encoding='utf-8'))
    if seed == 42:
        final = {name: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
                 for name, value in model.state_dict().items()}
        save_tensor(final, out / 'best.ckpt', work)
        del final
    summary = dict(scores(probs, targets[holdout].numpy()), seed=seed, epoch=3, history=history,
                   training_collapse=any(r['training_collapse'] for r in history),
                   seconds=elapsed + time.monotonic() - started, W_labels=LABELS, W=matrix.tolist(),
                   W_abs_max=float(abs(matrix).max()), W_abs_mean=float(abs(matrix).mean()),
                   W_pair_abs_mean=float(abs(matrix[np.triu_indices(9, 1)]).mean()))
    write_json(folder / 'metrics.json', summary)
    write_json(done, dict(config=CONFIG, rows_sha256=digest, seed=seed))
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return summary
def smoke():
    # Always check the 512-state identity and local-to-output copy without downloads.
    import itertools
    sets = list(itertools.product([0, 1], repeat=9))
    unary = [.1 * i - .4 for i in range(9)]
    unnormalized = [math.exp(sum(a * b for a, b in zip(y, unary))) for y in sets]
    for j in range(9):
        marginal = sum(z * y[j] for z, y in zip(unnormalized, sets)) / sum(unnormalized)
        assert abs(marginal - 1 / (1 + math.exp(-unary[j]))) < 1e-12
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        local = root / 'local'; local.write_bytes(b'checkpoint')
        publish(local, root / 'out' / 'best.ckpt')
        assert (root / 'out' / 'best.ckpt').read_bytes() == b'checkpoint'
        assert not (root / 'out' / 'best.ckpt.tmp').exists()
    try:
        import torch
        import numpy as np
    except ModuleNotFoundError as exc:
        if exc.name not in ('torch', 'numpy'):
            raise
        print('smoke PARTIAL: 512-state sigmoid identity and checkpoint publication passed; '
              f'{exc.name} unavailable, tensor/gradient/diagnostic checks skipped; no training')
        return
    torch.set_num_threads(1)
    torch.manual_seed(42)
    head = structured_head(4)
    head.eval()
    h = torch.randn(5, 4)
    p = head(h)
    assert head.W.numel() == 36 and head.Y.shape == (512, 9)
    assert torch.allclose(p, head.unary(h).sigmoid(), atol=1e-6)
    y = torch.randint(0, 2, (5, 9)).float()
    pw = torch.arange(1, 10).float()
    loss, bce, soft = loss_terms(p, y, pw)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(head.unary(h), y, pos_weight=pw)
    assert torch.allclose(bce, expected, atol=1e-6)
    loss.backward()
    assert torch.isfinite(head.W.grad).all() and head.W.grad.abs().sum() > 0
    with torch.no_grad():
        head.W[0] = .3
    matrix = head.matrix()
    assert torch.equal(matrix, matrix.T) and matrix.diag().sum() == 0
    empty = scores(np.zeros((3, 9)), np.zeros((3, 9)))
    assert empty['empty_predictions'] == 3 and empty['training_collapse']
    half = scores(np.full((3, 9), .5), np.ones((3, 9)))
    assert half['macro_f1'] == 1 and half['empty_predictions'] == 0
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        work = root / 'work'; work.mkdir()
        save_tensor({'W': head.W.detach()}, root / 'out' / 'ckpt_last.pt', work)
        assert torch.equal(torch.load(root / 'out' / 'ckpt_last.pt', weights_only=True)['W'], head.W)
        assert not list(work.iterdir())
    assert weights([[0] * 9] * 1000) == [6.] * 9
    print('smoke OK: W=0 sigmoid/BCE equivalence, pairwise gradients, threshold, diagnostics, local checkpoint publication; no training')


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
    model_path, tokenizer = prepare_model()  # Fail fast before data preparation.
    import torch
    import numpy as np
    out = Path(args.out)
    work = Path(os.environ['LAB_WORK']) / EXP_ID.lower()
    out.mkdir(parents=True, exist_ok=True); work.mkdir(parents=True, exist_ok=True)
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != CONFIG:
        raise ValueError('Output configuration mismatch')
    write_json(config_path, CONFIG)
    rows, train, holdout, digest = baseline_rows(out)
    if (out / 'metrics.json').exists():
        if not (out / 'best.ckpt').exists() or not all((out / f'seed_{s}' / 'done.json').exists() for s in SEEDS):
            raise ValueError('Incomplete final artifacts')
        print('All seeds complete; reusing metrics.json', flush=True)
        return
    pos_weight = weights([rows[i]['labels'] for i in train])
    write_json(out / 'pos_weight.json', dict(labels=LABELS, weights=pos_weight, formula=CONFIG['weight_formula']))
    cache = out / 'tokens.pt'
    if cache.exists():
        tokens = torch.load(cache, map_location='cpu', weights_only=True)
    else:
        tokens = dict(tokenizer([r['text'] for r in rows], padding='max_length', truncation=True,
                                max_length=256, return_token_type_ids=False, return_tensors='pt'))
        save_tensor(tokens, cache, work)
    if tokens['input_ids'].shape != (len(rows), 256) or tokens['attention_mask'].shape != (len(rows), 256):
        raise ValueError('Token cache shape mismatch')
    targets = torch.tensor([r['labels'] for r in rows], dtype=torch.float32)
    results = [run_seed(s, out, work, model_path, tokens, targets, train, holdout, pos_weight, digest, rows) for s in SEEDS]
    mean_macro = float(np.mean([r['macro_f1'] for r in results]))
    wins = sum(r['macro_f1'] > BASELINES[r['seed']] for r in results)
    collapsed = [r['seed'] for r in results if r['training_collapse']]
    seed_probs = {s: np.load(out / f'seed_{s}' / 'holdout_probs.npy') for s in SEEDS}
    probs = np.mean(list(seed_probs.values()), axis=0)
    truth = targets[holdout].numpy()
    predicted = (probs >= .5).astype(float)
    pair_comparison = []
    for i in range(9):
        for j in range(i + 1, 9):
            pair_comparison.append(dict(labels=[LABELS[i], LABELS[j]],
                observed=float((truth[:, i] * truth[:, j]).mean()),
                probability_mean_prediction=float((predicted[:, i] * predicted[:, j]).mean()),
                seeds={str(s): float(((seed_probs[s][:, i] >= .5) &
                                     (seed_probs[s][:, j] >= .5)).mean()) for s in SEEDS}))
    metrics = dict(exp_id=EXP_ID, primary_metric=dict(name='macro_f1', value=mean_macro, split='internal_holdout'),
                   metrics=dict(seeds=results, seeds_above_same_seed_baseline=wins,
                                same_seed_baselines=BASELINES, collapse_seeds=collapsed,
                                success_criteria_met=mean_macro >= .6527 and wins >= 2 and not collapsed,
                                status='failed_training_collapse' if collapsed else 'completed',
                                mean_micro_f1=float(np.mean([r['micro_f1'] for r in results])),
                                probability_mean_macro_f1=scores(probs, truth)['macro_f1'],
                                pair_cooccurrence=pair_comparison,
                                seconds=sum(r['seconds'] for r in results)),
                   seed=42, seeds=SEEDS, train_rows=len(train), eval_rows=len(holdout), official_val=None,
                   ckpt='best.ckpt', notes='Final epoch 3, representative seed42 weights only. Fixed marginal >=0.5; '
                   'empty predictions unchanged. Preset A: lambda_set=0, BCE+soft macro-F1; W lr=.01, decay=0. '
                   'Original ROBERTA-003 rows/split reused; text only. Official Validation unused. '
                   'Probability averaging and pair diagnostics are not adoption criteria.')
    write_json(out / 'metrics.json', metrics)


if __name__ == '__main__':
    main()
