# DCC 2026 Mission 2 L2 (CNN 3-seed ensemble) CPU inference timing

Measured 2026-10-06, 09:35–09:46 KST, on the shared Linux box. **No training was done. No commercial API was used. No Drive file was modified** (only read/download).

## What was run
- **Code:** frozen zip `MyDrive/dcc2026-prelim-main-4bfd2c4.zip` (Drive id 1vGQCKX3s6Nr6mCfrwDTIViyoX_qpSwWF), unmodified.
  Its `scripts/mission2_inference.py` already loads L2 `.pt` checkpoints, including packed ensembles (`ensemble_model_state_dicts` → soft-vote of `SpeakerCNNLarge` members), so the GitHub main (d91f8de) fallback was **not** needed.
  CLI flags: `--audio_dir --label_dir --ckpt_path --output` (all required, nothing else).
- **Checkpoints: RANDOM WEIGHTS.** No L2 `.pt` exists on Drive. I searched for m2_cnn_large / raw_ts / ensemble / cnn / l2 / seed / checkpoint / m2_cache / m2_raw_label_models, and listed dcc_runs/ (only m2_l1_9k, m2_l2_spec = spectrogram caches, m3_*). The repo reports name `checkpoints/mission2/m2_cnn_large_raw_ensemble.pt` and `m2_cache/m2_raw_label_models/m2_cnn_large_raw_ts{1,2,3}.pt`, but `checkpoints/` is git-ignored and none of those files are on Drive.
  `make_random_ckpts.py` builds 3 random-init `SpeakerCNNLarge` models (torch seeds 1/2/3, 421,954 params each) with metadata from the repo's `build_checkpoint_metadata` (model=large, split=Training, label_source=verified_against_provided_json, n_mels=40, hop=160, n_fft=512, sr=8000). The repo's own `scripts/pack_mission2_l2_ensemble.py` then packed them:
  - `ckpt/m2_cnn_large_RANDOM_ts{1,2,3}.pt`: 1,703,651 B each
  - `ckpt/m2_cnn_large_RANDOM_ensemble.pt`: 5,109,025 B (the one passed to `--ckpt_path`)
  The weights do not affect CPU time: same architecture, same per-utterance input shapes. Every prediction comes out as 0 because the weights are random.
- **Data:** official Validation, `대학부 데이터/Validation/1.원천데이터/VS_서울_구급` (flat, unzipped WAVs) + `2.라벨링데이터/VL_서울_구급` (JSON). I took the first 40 calls in folder order, as matching wav+json pairs.
  - 40 calls / **1,230 utterances**, 30.75 utt/call. The full set is 111,947/3,640 = 30.75, so the sample is representative.
  - 8 kHz mono PCM16. Mean utterance 2.02 s (median 1.46 s, max 10.4 s). 41.5 min of utterance audio in total.
  - The 5-call subset (`data5/`, symlinks) has 158 utterances.
- **Env:** venv with Python 3.13.5, torch 2.14.1+cpu, librosa 1.0.0, numpy (OpenBLAS).

Command (all runs; only env/taskset prefix varies):
```
/usr/bin/time -v python code/scripts/mission2_inference.py \
  --audio_dir data/wav --label_dir data/json \
  --ckpt_path ckpt/m2_cnn_large_RANDOM_ensemble.pt --output out/<tag>.csv
```

## Hardware
8 vCPU (`nproc`=8), Intel Xeon Processor (family 6 model 207, Sapphire/Emerald Rapids class, AVX-512/AMX), KVM guest, 1 thread per core. L2 16 MiB, L3 320 MiB. RAM 15 GiB (about 6.7 GiB free during runs). The box is shared with other agents, but load was low (load average 0.4–3.8).

## Results (wall clock, `/usr/bin/time -v`)
Overhead (python + torch/librosa import + ckpt load + pairing) and per-utterance cost come from the 5-call vs 40-call difference.
Extrapolation: `total = overhead + s_per_utt × 111,947`.

| config | 5 calls (158 utt) | 40 calls (1,230 utt) | s/utt | utt/s | overhead | **111,947 utt** |
|---|---|---|---|---|---|---|
| A. 8 vCPU, no env (**script default**) | 6.08 / 5.83 s | 25.33 / 25.25 s | 18.0 ms | 55 | 3.1 s | **0.56 h (34 min)** |
| B. 2 cores pinned (`taskset -c 0,1`), no env (**Colab-2vCPU-like default**) | 5.85 s | 24.33 / 24.53 s | 17.3 ms | 58 | 3.1 s | **0.54 h (32 min)** |
| C. 2 cores pinned + `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2` | 6.70 s | 32.38 / 32.15 s (cores 6,7: 30.95) | 23.9 ms | 42 | 2.9 s | **0.74 h (45 min)** |
| D. 2 cores pinned + OMP=2 MKL=2 **OPENBLAS_NUM_THREADS=1** | 4.57 s | 16.07 s | 10.7 ms | 93 | 2.9 s | **0.33 h (20 min)** |
| E. 8 vCPU + OMP=2 MKL=2 (unpinned) | 4.85 s | 16.01 / 16.63 s | 10.7 ms | 93 | 3.2 s | 0.33 h |
| F. 8 vCPU + OMP=8 MKL=8 OPENBLAS=1 (best here) | 4.01 s | 10.86 s | 6.4 ms | 157 | 3.0 s | 0.20 h (12 min) |
| G. 8 vCPU + OMP=1 MKL=1 | – | 25.11 s | ~17.9 ms | 56 | ~3.1 s | 0.56 h |

Other measurements:
- **Peak RSS:** 486–507 MB in every config. The script holds only one call's waveform at a time. The output-row list grows by about 112k small dicts over a full run, which adds only tens of MB.
- **Per-stage breakdown** (`stage_profile.py`, same functions the script calls, torch at 1 thread):
  - import ~2.0 s, ckpt load 0.02 s
  - WAV read ~2 ms/call
  - log-mel ~1.7 ms/utt
  - **3× CNN forward ~15.9 ms/utt (~88% of the per-utterance time)**
  - After 300-frame padding, spectrograms average 303 frames (median 300, max 521).
- **Output CSV check:** 1,230 rows = 1,230 JSON utterances. Header `audio_file_name,startAt,endAt,speaker`. Row order and startAt/endAt match the JSONs exactly. 0 fallbacks. Byte-identical across all thread configs (deterministic).

## Quirks / flags
1. **The script reads raw WAV + JSON directly** (`read_call_waveform` → slice → `librosa` log-mel → CNN). It does **not** use the packed spectrogram caches (`m2_spectrograms_validation_packed.npy` etc.). Good for the contest contract.
2. **The thread settings are effectively fixed by an import-order side effect.** `dcc_data/mission2_features.py` does `os.environ.setdefault("OMP_NUM_THREADS"/"MKL_NUM_THREADS"/"OPENBLAS_NUM_THREADS", "1")` at import. Two consequences:
   - **torch** is imported later (lazily, in `_load_l2_predictor`), so with no env set it runs with **1 intra-op thread** even on 8 cores. That is why the default (A) is no faster than 1 thread (G).
   - **numpy/OpenBLAS** is imported *before* that module (`import numpy as np` at line 80 of the script), so the OPENBLAS=1 default comes too late. OpenBLAS spins up 1 thread per core, which busy-wait in librosa's small mel matmuls. That gives ~585% CPU with no speedup in A. Pinned to 2 cores together with 2 torch threads (C), they compete and the run gets **slower** (0.74 h vs 0.54 h).
   - `OMP_NUM_THREADS=2` alone does **not** give torch 2 threads, because MKL_NUM_THREADS is still defaulted to 1. You need OMP **and** MKL =2.
   - The fastest env-only setting is `OMP_NUM_THREADS=N MKL_NUM_THREADS=N OPENBLAS_NUM_THREADS=1`. On 2 cores that is D (0.33 h, ~1.6× faster than default).
3. Inference runs one utterance at a time at batch=1 (by design: own-utterance 300-frame padding, isolated-fragment rule). The CNN dominates the time. Batching same-length (300-frame) utterances would speed this up a lot but would change the script; I did not test it.
4. **Hardware caveat for Colab:** this box's Xeon (AVX-512/AMX, 2026-era) is likely faster per core than a typical Colab 2-vCPU instance (often an older Xeon @2.2 GHz). Expect Colab CPU time of roughly 1.5–2.5× the B/D numbers, i.e. about **0.5–0.8 h (D) or 0.8–1.4 h (B, default)**. That is an estimate, not a measurement.
5. **I/O caveat:** here the WAVs were on local disk (~2 ms/call to read). Reading 3,640 WAVs (~4.5 GB) through a Colab Drive FUSE mount can add a lot. For reference, the team's Colab `m2_l2_spec` STATUS.txt shows the Validation spectrogram build (Drive read + mel + save, 2 workers, nproc=2) took 3,070 s, and a Drive manifest rglob took ~6 min. Copying the data to local disk first is advisable.

## Files
- `logs/*.log`: per-run `/usr/bin/time -v` logs, plus `logs/all_runs_combined.log`
- `extrapolation.txt`: computed table
- `out/*.csv`: outputs
- `make_random_ckpts.py`, `stage_profile.py`, `ckpt/`, `data/` (40 pairs), `data5/` (5-call subset), `code/` (unzipped frozen code), `venv/`
