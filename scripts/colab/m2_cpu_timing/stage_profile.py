"""Per-stage timing using the SAME functions mission2_inference.py calls (import, ckpt load, WAV read, mel, 3x CNN)."""
import time, sys, json
t0 = time.perf_counter()
sys.path.insert(0, "/workspace/m2_cpu_timing/code/scripts"); sys.path.insert(0, "/workspace/m2_cpu_timing/code/src")
import mission2_inference as mi
import torch, torch.nn.functional as F
from dcc_data.mission2_spectrogram import mel_spectrogram
from dcc_data.mission2_cnn import SpeakerCNNLarge
from dcc_data.mission2_checkpoint import l2_member_state_dicts, resolve_l2_checkpoint_metadata
from pathlib import Path
t_import = time.perf_counter() - t0
t0 = time.perf_counter()
ck = torch.load("ckpt/m2_cnn_large_RANDOM_ensemble.pt", map_location="cpu")
md, _ = resolve_l2_checkpoint_metadata(ck, "x")
models = []
for sd in l2_member_state_dicts(ck, md):
    m = SpeakerCNNLarge(); m.load_state_dict(sd); m.eval(); models.append(m)
t_load = time.perf_counter() - t0
pairs = mi.pair_by_stem_strict(Path(sys.argv[1]) / "wav", Path(sys.argv[1]) / "json")
tr = tm = tc = 0.0; n = 0; frames = []
with torch.no_grad():
    for stem, wav, js in pairs:
        utts = json.load(open(js))["utterances"]
        a = time.perf_counter(); w, sr = mi.read_call_waveform(wav); tr += time.perf_counter() - a
        for u in utts:
            s, e = mi.utterance_frame_range(u["startAt"], u["endAt"], sr, w.shape[0])
            seg = mi.slice_and_pad(w, s, e, mi.N_FFT)
            a = time.perf_counter(); spec = mi.pad_spectrogram_to_min_frames(mel_spectrogram(seg, sr)); tm += time.perf_counter() - a
            frames.append(spec.shape[1])
            b = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
            a = time.perf_counter(); p = sum(F.softmax(mm(b), dim=1) for mm in models) / 3; tc += time.perf_counter() - a
            n += 1
import numpy as np
print(f"threads={torch.get_num_threads()} import={t_import:.2f}s ckpt_load={t_load:.3f}s calls={len(pairs)} utts={n}")
print(f"wav_read={tr:.2f}s ({tr/len(pairs)*1000:.1f} ms/call)  mel={tm:.2f}s ({tm/n*1000:.2f} ms/utt)  cnn_x3={tc:.2f}s ({tc/n*1000:.2f} ms/utt)")
print(f"spec frames after 300-pad: mean={np.mean(frames):.0f} median={np.median(frames):.0f} max={max(frames)}")
