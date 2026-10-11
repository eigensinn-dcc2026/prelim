#!/usr/bin/env python3
"""M2-SSL-002: matched-call CNN/WavLM fine-tuning experiment, Training only.

Run from the repository root with PYTHONPATH=src. --smoke never downloads,
reads contest data, or trains. All selection uses D; H is evaluated only after
both branches finish. Feature chunks are durable, packed arrays are local.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

EXP_ID = 'M2-SSL-002'
MODEL_ID = 'microsoft/wavlm-base-plus'
REVISION = '4c66d4806a428f2e922ccfa1a962776e232d487b'
CONFIG = dict(exp_id=EXP_ID, revision=REVISION, schema=1, split_seed=42,
              a_seed=1, b_seed=42, holdout_calls=1500, sampled_calls=6000,
              dev_calls=300, label_source='provided_json', cnn_sample_rate=8000,
              wavlm_sample_rate=16000, crop_seconds=6, min_seconds=.5)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def publish(local, destination):
    """All potentially large writes start on local disk; retry FUSE publication."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + '.tmp')
    for attempt in range(6):
        try:
            shutil.copyfile(local, tmp)
            if tmp.stat().st_size != local.stat().st_size:
                raise OSError('Incomplete Drive copy')
            os.replace(tmp, destination)
            return
        except OSError:
            if attempt == 5:
                raise
            time.sleep(2 ** (attempt + 1))


def save_checkpoint(value, destination, work):
    import torch
    local = work / (destination.parent.name + '_' + destination.name)
    torch.save(value, local)
    publish(local, destination)
    local.unlink()


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def load_checkpoint(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def prepare_model():
    from huggingface_hub import snapshot_download
    from transformers import AutoFeatureExtractor, WavLMModel
    root = Path(os.environ['DCC_RUN_ROOT']) / 'models' / 'wavlm-base-plus'
    marker = root / 'snapshot.json'  # Compatible with M2-SSL-001.
    identity = dict(repo_id=MODEL_ID, revision=REVISION)
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError('WavLM cache revision mismatch')
    else:
        snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=str(root),
                          allow_patterns=['config.json', 'preprocessor_config.json',
                                          'pytorch_model.bin', 'model.safetensors'])
    processor = AutoFeatureExtractor.from_pretrained(root, local_files_only=True)
    encoder = WavLMModel.from_pretrained(root, local_files_only=True)
    if encoder.config.hidden_size != 768 or encoder.config.num_hidden_layers != 12:
        raise ValueError('Expected WavLM 12 layers / 768 dimensions')
    write_json(marker, identity)
    encoder.freeze_feature_encoder()
    encoder.gradient_checkpointing_enable()
    return processor, encoder


def unpack(source, work, selected_stems=None):
    local = work / source.name
    if not local.exists() or local.stat().st_size != source.stat().st_size:
        tmp = local.with_suffix('.zip.tmp')
        shutil.copyfile(source, tmp)
        os.replace(tmp, local)
    root = work / source.stem
    with zipfile.ZipFile(local) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()
                   and (selected_stems is None or Path(m.filename).stem in selected_stems)]
        for m in members:
            if not (root / m.filename).resolve().is_relative_to(root.resolve()):
                raise ValueError('Unsafe archive member')
        tag = ('.complete' if selected_stems is None else '.complete_' +
               hashlib.sha256('\n'.join(sorted(selected_stems)).encode()).hexdigest()[:16])
        if not (root / tag).exists():
            root.mkdir(parents=True, exist_ok=True)
            archive.extractall(root, members=members)
            (root / tag).touch()
    return root, [m.filename for m in members]


def archive_index(root, members, suffix):
    result = {}
    for member in members:
        p = Path(member)
        if p.suffix.lower() == suffix:
            if p.stem in result:
                raise ValueError(f'Duplicate file stem: {p.stem}')
            result[p.stem] = root / p
    return result


def make_split(calls):
    if len(calls) != 29200:
        raise ValueError(f'Expected 29,200 Seoul Training calls, found {len(calls)}')
    h = random.Random(42).sample(sorted(calls), 1500)
    s = random.Random(42).sample(sorted(set(calls) - set(h)), 6000)
    d = random.Random(42).sample(sorted(s), 300)
    shuffled_h = h.copy()
    random.Random(42).shuffle(shuffled_h)
    return dict(config=CONFIG, H=h, S=s, D=d,
                H_groups=[shuffled_h[i::5] for i in range(5)])


def prepare_data(out, work):
    import numpy as np
    import librosa
    from dcc_data.colab_env import find_data_root
    from dcc_data.mission2_features import read_call_waveform, utterance_frame_range
    from dcc_data.mission2_spectrogram import mel_spectrogram
    source = find_data_root() / 'Training'
    lr, lm = unpack(source / 'TL_서울_구급.zip', work)
    labels = archive_index(lr, lm, '.json')
    calls = {}
    for stem, path in sorted(labels.items()):
        doc = json.loads(path.read_text(encoding='utf-8'))
        cid = str(doc.get('_id') or stem)
        if cid in calls:
            raise ValueError(f'Duplicate call_id {cid}')
        calls[cid] = dict(stem=stem, path=str(path))
    split_path = out / 'split.json'
    if split_path.exists():
        split = json.loads(split_path.read_text())
        if split['config'] != CONFIG:
            raise ValueError('Split configuration mismatch')
    else:
        split = make_split(calls)
        write_json(split_path, split)
    h, s, d = (set(split[k]) for k in ('H', 'S', 'D'))
    if h & s or not d <= s or len(h) != 1500 or len(s) != 6000 or len(d) != 300:
        raise ValueError('Invalid persisted call split')
    selected = sorted(h | s)
    if not set(selected) <= calls.keys():
        raise ValueError('Saved calls missing from archives')
    ar, am = unpack(source / 'TS_서울_구급.zip', work,
                    {calls[cid]['stem'] for cid in selected})
    audio = archive_index(ar, am, '.wav')
    chunks = out / 'features'
    chunks.mkdir(exist_ok=True)
    all_rows = []
    wave_file, mel_file = work / 'waveforms.f16', work / 'mels.f16'
    wave_offset = mel_offset = 0
    with wave_file.open('wb') as wf, mel_file.open('wb') as mf:
        for chunk_no, start in enumerate(range(0, len(selected), 50)):
            ids = selected[start:start + 50]
            durable = chunks / f'{chunk_no:04d}.npz'
            local = work / 'chunk.npz'
            if durable.exists():
                shutil.copyfile(durable, local)
            else:
                rows, waves, mels = [], [], []
                for cid in ids:
                    entry = calls[cid]
                    doc = json.loads(Path(entry['path']).read_text(encoding='utf-8'))
                    waveform, sr = read_call_waveform(audio[entry['stem']])
                    for idx, utt in enumerate(doc['utterances']):
                        label = utt['speaker']
                        if type(label) is not int or label not in (0, 1):
                            raise ValueError(f'Invalid provided speaker {cid}/{idx}')
                        a, b = utterance_frame_range(utt['startAt'], utt['endAt'], sr, len(waveform))
                        segment = waveform[a:b]
                        if not len(segment):
                            raise ValueError(f'Empty utterance {cid}/{idx}')
                        wave = (librosa.resample(segment, orig_sr=sr, target_sr=16000)
                                if sr != 16000 else segment).astype('float16')
                        # Baseline uses the original 8 kHz defaults, from the SAME target waveform.
                        cnn_wave = librosa.resample(wave.astype('float32'), orig_sr=16000, target_sr=8000)
                        mel = mel_spectrogram(cnn_wave, 8000).T.astype('float16')
                        rows.append(dict(call_id=cid, utt_idx=idx, label=label,
                                         duration=(float(utt['endAt']) - float(utt['startAt'])) / 1000,
                                         samples=len(wave), frames=len(mel)))
                        waves.append(wave)
                        mels.append(mel)
                np.savez(local, calls=np.array(ids), rows=np.array(json.dumps(rows)),
                         wave=np.concatenate(waves), mel=np.concatenate(mels))
                publish(local, durable)
            with np.load(local, allow_pickle=False) as chunk:
                if chunk['calls'].tolist() != ids:
                    raise ValueError('Feature chunk split mismatch')
                rows = json.loads(str(chunk['rows']))
                wave, mel = chunk['wave'], chunk['mel']
                if sum(r['samples'] for r in rows) != len(wave) or sum(r['frames'] for r in rows) != len(mel):
                    raise ValueError('Corrupt packed features')
                wave.tofile(wf)
                mel.tofile(mf)
                for r in rows:
                    r.update(wave_offset=wave_offset, mel_offset=mel_offset)
                    wave_offset += r['samples']
                    mel_offset += r['frames']
                    all_rows.append(r)
            print(f'features: {min(start + 50, len(selected))}/{len(selected)} calls', flush=True)
    return split, all_rows, np.memmap(wave_file, dtype='float16', mode='r'), np.memmap(
        mel_file, dtype='float16', mode='r', shape=(mel_offset, 40))


def wavlm_classifier(encoder):
    import torch
    from torch import nn

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.layer_weights = nn.Parameter(torch.zeros(13))
            self.head = nn.Linear(768, 2)

        def forward(self, input_values, attention_mask):
            states = self.encoder(input_values=input_values, attention_mask=attention_mask,
                                  output_hidden_states=True).hidden_states
            mixed = sum(w * h for w, h in zip(self.layer_weights.softmax(0), states))
            mask = self.encoder._get_feature_vector_attention_mask(mixed.shape[1], attention_mask)
            pooled = (mixed * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp_min(1).unsqueeze(-1)
            return self.head(pooled)

    return Classifier()


def crop_bounds(length, training):
    excess = max(0, length - 96000)
    start = random.randint(0, excess) if training else excess // 2
    return start, min(length, start + 96000), max(0, 8000 - length)


def crop(wave, training):
    import numpy as np
    wave = np.asarray(wave, dtype='float32')
    start, end, padding = crop_bounds(len(wave), training)
    return np.pad(wave[start:end], (0, padding))


def batch_inputs(branch, indices, rows, waves, mels, processor, training, device):
    import numpy as np
    import torch
    from dcc_data.mission2_cnn import collate_batch
    if branch == 'a':
        batch = []
        for i in indices:
            r = rows[i]
            spec = np.array(mels[r['mel_offset']:r['mel_offset'] + r['frames']].T, dtype='float32')
            # Training retains original variable-length padding; inference/dev minimum = 300.
            if not training and spec.shape[1] < 300:
                spec = np.pad(spec, ((0, 0), (0, 300 - spec.shape[1])), constant_values=-80)
            batch.append((torch.from_numpy(spec), r['label']))
        specs, labels = collate_batch(batch)
        return [specs.to(device)], labels.to(device)
    samples = [crop(waves[rows[i]['wave_offset']:rows[i]['wave_offset'] + rows[i]['samples']], training)
               for i in indices]
    inputs = processor(samples, sampling_rate=16000, padding=True,
                       return_attention_mask=True, return_tensors='pt')
    return [inputs['input_values'].to(device), inputs['attention_mask'].to(device)], torch.tensor(
        [rows[i]['label'] for i in indices], device=device)


def batches_for_epoch(branch, train_ids, rows):
    ids = train_ids.copy()
    random.shuffle(ids)
    size = 64 if branch == 'a' else 32
    if branch == 'b':
        # Randomized pools, sorted by this utterance's own length only.
        pools = [sorted(ids[i:i + 3200], key=lambda j: min(rows[j]['samples'], 96000))
                 for i in range(0, len(ids), 3200)]
        batches = [pool[i:i + size] for pool in pools for i in range(0, len(pool), size)]
        random.shuffle(batches)
        return batches
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def predict(model, branch, ids, data, processor, device, batch_size=None):
    import numpy as np
    import torch
    rows, waves, mels = data
    model.eval()
    size = batch_size or (64 if branch == 'a' else 32)
    probs = []
    with torch.inference_mode():
        for start in range(0, len(ids), size):
            inputs, _ = batch_inputs(branch, ids[start:start + size], rows, waves, mels, processor, False, device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=device == 'cuda'):
                p = model(*inputs).float().softmax(1)[:, 1]
            probs.append(p.cpu().numpy())
    return np.concatenate(probs)


def fit(branch, model, split, data, processor, out, work):
    import numpy as np
    import torch
    from dcc_data.mission2_cnn import move_optimizer_state_to_device
    from transformers import get_linear_schedule_with_warmup
    rows = data[0]
    train_calls = set(split['S']) - set(split['D'])
    train_ids = [i for i, r in enumerate(rows) if r['call_id'] in train_calls]
    dev_calls = set(split['D'])
    dev_ids = [i for i, r in enumerate(rows) if r['call_id'] in dev_calls]
    y = np.array([rows[i]['label'] for i in dev_ids])
    majority = float(max((y == 0).mean(), (y == 1).mean()))
    directory = out / branch
    directory.mkdir(exist_ok=True)
    done_path = out / f'{branch}_done.json'
    best_path = directory / 'best.ckpt' if branch == 'a' else out / 'best.ckpt'
    if done_path.exists():
        model.load_state_dict(load_checkpoint(best_path)['model'])
        return json.loads(done_path.read_text())
    seed_all(1 if branch == 'a' else 42)
    model.cuda()
    if branch == 'a':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=2)
        epochs = 30
    else:
        optimizer = torch.optim.AdamW([
            dict(params=[p for p in model.encoder.parameters() if p.requires_grad], lr=3e-5),
            dict(params=[model.layer_weights, *model.head.parameters()], lr=1e-3)], weight_decay=.01)
        total_steps = math.ceil(len(train_ids) / 32) * 2
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * .1), total_steps)
        epochs = 2
    scaler = torch.amp.GradScaler('cuda')
    state = dict(epoch=0, step=0, order=None, loss_sum=0., seen=0, history=[],
                 best=-1., best_epoch=-1, bad_epochs=0, majority=majority)
    last = directory / 'ckpt_last.pt'
    if last.exists():
        saved = load_checkpoint(last)
        if saved['config'] != CONFIG or saved['branch'] != branch:
            raise ValueError('Checkpoint experiment mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        move_optimizer_state_to_device(optimizer, 'cuda')
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        state = saved['progress']
        restore_rng(saved['rng'])
    last_save = time.monotonic()

    def checkpoint():
        nonlocal last_save
        save_checkpoint(dict(config=CONFIG, branch=branch, model=model.state_dict(),
                             optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                             scaler=scaler.state_dict(), progress=state, rng=rng_state()), last, work)
        last_save = time.monotonic()

    while state['epoch'] < epochs and not (branch == 'a' and state['bad_epochs'] >= 5):
        if state['order'] is None:
            state['order'] = batches_for_epoch(branch, train_ids, rows)
            checkpoint()
        model.train()
        for step in range(state['step'], len(state['order'])):
            inputs, labels = batch_inputs(branch, state['order'][step], *data, processor, True, 'cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(*inputs), labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward()
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if branch == 'b' and scaler.get_scale() >= old_scale:
                scheduler.step()
            state['loss_sum'] += float(loss.detach()) * len(labels)
            state['seen'] += len(labels)
            state['step'] = step + 1
            if time.monotonic() - last_save >= 900:
                checkpoint()
        p = predict(model, branch, dev_ids, data, processor, 'cuda')
        acc = float(((p >= .5) == y).mean())
        state['history'].append(dict(epoch=state['epoch'] + 1, dev_accuracy=acc,
                                     train_loss=state['loss_sum'] / state['seen'],
                                     collapsed=acc <= majority))
        if branch == 'a':
            scheduler.step(acc)
        if acc > state['best']:
            state.update(best=acc, best_epoch=state['epoch'] + 1, bad_epochs=0)
            weights = {k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()
                       for k, v in model.state_dict().items()}
            save_checkpoint(dict(model=weights, config=CONFIG, branch=branch,
                                 epoch=state['best_epoch'], revision=REVISION,
                                 crop=dict(train='random', evaluation='center', max_samples=96000,
                                           min_samples=8000), label_source='provided_json'), best_path, work)
        else:
            state['bad_epochs'] += 1
        print(f'{branch}: {state["history"][-1]}', flush=True)
        state.update(epoch=state['epoch'] + 1, step=0, order=None, loss_sum=0., seen=0)
        checkpoint()
    result = dict(history=state['history'], best_dev_accuracy=state['best'],
                  best_epoch=state['best_epoch'], dev_majority=majority,
                  failed=any(epoch['collapsed'] for epoch in state['history']), train_rows=len(train_ids),
                  dev_rows=len(dev_ids), train_calls=len(train_calls))
    model.load_state_dict(load_checkpoint(best_path)['model'])
    write_json(done_path, result)
    return result


def comparison(rows, ids, a, b, split):
    import numpy as np
    from scipy.stats import binomtest
    y = np.array([rows[i]['label'] for i in ids])
    duration = np.array([rows[i]['duration'] for i in ids])
    def score(prob, mask=None):
        mask = np.ones(len(y), dtype=bool) if mask is None else mask
        if not mask.any():
            return None
        pred, truth = (prob[mask] >= .5).astype(int), y[mask]
        cm = [[int(((truth == c) & (pred == p)).sum()) for p in (0, 1)] for c in (0, 1)]
        return dict(accuracy=float((pred == truth).mean()), confusion_matrix=cm,
                    recall={name: cm[c][c] / sum(cm[c]) if sum(cm[c]) else None
                            for c, name in enumerate(('대원', '신고자'))}, rows=int(mask.sum()))
    ca, cb = (a >= .5) == y, (b >= .5) == y
    table = [[int((ca & cb).sum()), int((ca & ~cb).sum())],
             [int((~ca & cb).sum()), int((~ca & ~cb).sum())]]
    discordant = table[0][1] + table[1][0]
    pvalue = float(binomtest(table[0][1], discordant, .5).pvalue) if discordant else 1.
    groups = []
    for group in split['H_groups']:
        group_calls = set(group)
        mask = np.array([rows[i]['call_id'] in group_calls for i in ids])
        groups.append(dict(a=score(a, mask), b=score(b, mask)))
    lengths = {}
    for name, mask in [('<1s', duration < 1), ('1-3s', (duration >= 1) & (duration < 3)),
                       ('3-6s', (duration >= 3) & (duration <= 6)), ('>6s', duration > 6)]:
        lengths[name] = dict(a=score(a, mask), b=score(b, mask))
    delta = float(cb.mean() - ca.mean())
    wins = sum(g['b']['accuracy'] > g['a']['accuracy'] for g in groups)
    return dict(a=score(a), b=score(b), delta_accuracy=delta, groups=groups,
                group_wins=wins, duration_buckets=lengths,
                mcnemar=dict(table=table, axes=['A correct/incorrect', 'B correct/incorrect'],
                             p_value=pvalue, method='two-sided exact binomial'),
                soft_vote=score((a + b) / 2), statistical_criteria_met=delta >= .01 and wins >= 4 and pvalue < .05)


def benchmark(model, ids, data, processor):
    import torch
    chosen = random.Random(42).sample(ids, min(500, len(ids)))
    # Warm-up excluded; timing includes collate/normalization and transfer, excludes disk staging.
    predict(model, 'b', chosen[:32], data, processor, 'cuda')
    torch.cuda.synchronize()
    start = time.perf_counter()
    predict(model, 'b', chosen, data, processor, 'cuda')
    torch.cuda.synchronize()
    gpu_seconds = time.perf_counter() - start
    model.cpu()
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        predict(model, 'b', chosen[:1], data, processor, 'cpu', batch_size=1)
        start = time.perf_counter()
        predict(model, 'b', chosen, data, processor, 'cpu', batch_size=1)
        cpu_seconds = time.perf_counter() - start
    finally:
        torch.set_num_threads(threads)
    return dict(gpu=torch.cuda.get_device_name(0), rows=len(chosen),
                gpu_batch_size=32, gpu_seconds=gpu_seconds,
                gpu_utterances_per_second=len(chosen) / gpu_seconds,
                cpu_threads=2, cpu_interop_threads=1, cpu_dtype='float32',
                cpu_batch_size=1, cpu_seconds=cpu_seconds,
                cpu_seconds_per_utterance=cpu_seconds / len(chosen),
                estimated_official_val_cpu_minutes=cpu_seconds / len(chosen) * 111947 / 60,
                reference_l2_ensemble_minutes=85.3,
                sample_ids=[f'{data[0][i]["call_id"]}\t{data[0][i]["utt_idx"]}' for i in chosen])


def smoke():
    """Standard-library checks, usable without ML packages; never trains."""
    split = make_split([f'call_{i:05}' for i in range(29200)])
    assert not set(split['H']) & set(split['S'])
    assert set(split['D']) <= set(split['S'])
    assert len(set().union(*map(set, split['H_groups']))) == 1500
    assert all(len(group) == 300 for group in split['H_groups'])
    assert crop_bounds(100, False) == (0, 100, 7900)
    assert crop_bounds(100000, False) == (2000, 98000, 0)
    seed_all_smoke = random.getstate()
    random.seed(42)
    for _ in range(100):
        start, end, padding = crop_bounds(100000, True)
        assert 0 <= start <= 4000 and end - start == 96000 and padding == 0
    random.setstate(seed_all_smoke)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / 'local').write_bytes(b'checkpoint')
        publish(root / 'local', root / 'out' / 'saved')
        assert (root / 'out' / 'saved').read_bytes() == b'checkpoint'
        write_json(root / 'split.json', split)
        assert json.loads((root / 'split.json').read_text()) == split
        source = root / 'Training.zip'
        with zipfile.ZipFile(source, 'w') as archive:
            archive.writestr('TL_서울_구급/example.json', '{}')
        stage = root / 'stage'
        stage.mkdir()
        extracted, members = unpack(source, stage)
        assert archive_index(extracted, members, '.json')['example'].read_text() == '{}'
        unpack(source, stage)  # Completed staging is reusable.
    print('smoke OK: splits, crop bounds, atomic publication, local zip staging; no training')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=os.environ.get('LAB_OUT'))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    # Fail on local model/cache issues before data preparation.
    processor, encoder = prepare_model()
    import numpy as np
    import torch
    from dcc_data.mission2_cnn import SpeakerCNNLarge
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('T4/CUDA required; use --smoke for local validation')
    if not args.output or os.environ.get('LAB_EXP_ID', EXP_ID) != EXP_ID:
        raise ValueError('LAB_OUT and matching LAB_EXP_ID required')
    out, work = args.output, Path(os.environ['LAB_WORK']) / EXP_ID.lower()
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists() and json.loads((out / 'config.json').read_text()) != CONFIG:
        raise ValueError('Output experiment configuration mismatch')
    write_json(out / 'config.json', CONFIG)
    if (out / 'metrics.json').exists() and (out / 'best.ckpt').exists():
        print('Experiment complete; reusing metrics.json')
        return
    split, rows, waves, mels = prepare_data(out, work)
    data = (rows, waves, mels)
    seed_all(1)
    a = SpeakerCNNLarge()
    a_result = fit('a', a, split, data, processor, out, work)
    a.cpu()
    seed_all(42)
    b = wavlm_classifier(encoder)
    b_result = fit('b', b, split, data, processor, out, work)
    holdout_calls, sampled_calls = set(split['H']), set(split['S'])
    ids = [i for i, r in enumerate(rows) if r['call_id'] in holdout_calls]
    a.cuda()
    pa = predict(a, 'a', ids, data, processor, 'cuda')
    a.cpu()
    b.cuda()
    pb = predict(b, 'b', ids, data, processor, 'cuda')
    metrics = comparison(rows, ids, pa, pb, split)
    metrics.update(a_training=a_result, b_training=b_result,
                   layer_weights=b.layer_weights.detach().float().softmax(0).cpu().tolist(),
                   benchmark=benchmark(b, ids, data, processor), seeds=dict(split=42, a=1, b=42),
                   sampled_calls=6000, dev_calls=300, holdout_calls=1500,
                   sampled_rows=sum(r['call_id'] in sampled_calls for r in rows),
                   revision=REVISION, label_source='provided_json', official_val_evaluated=False)
    seed_dir = out / 'seed_42'
    seed_dir.mkdir(exist_ok=True)
    np.save(work / 'holdout_probs.npy', np.column_stack([pa, pb]))
    publish(work / 'holdout_probs.npy', seed_dir / 'holdout_probs.npy')
    write_json(seed_dir / 'holdout_probs_columns.json', ['A_신고자_probability', 'B_신고자_probability'])
    (work / 'val_files.txt').write_text(''.join(f'{rows[i]["call_id"]}\t{rows[i]["utt_idx"]}\n' for i in ids), encoding='utf-8')
    publish(work / 'val_files.txt', seed_dir / 'val_files.txt')
    failed = a_result['failed'] or b_result['failed']
    write_json(out / 'metrics.json', dict(exp_id=EXP_ID,
        primary_metric=dict(name='accuracy_delta_B_minus_A', value=metrics['delta_accuracy'], split='internal_holdout'),
        metrics=metrics, seed=42, train_rows=b_result['train_rows'], eval_rows=len(ids), official_val=None,
        ckpt='best.ckpt', status='failed' if failed else 'completed',
        success_criteria_met=not failed and metrics['statistical_criteria_met'],
        notes='label_source=provided_json; epoch selection on D only; collapse failure if any epoch D <= majority'))


if __name__ == '__main__':
    main()
