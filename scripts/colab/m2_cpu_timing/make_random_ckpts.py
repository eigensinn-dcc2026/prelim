"""Build 3 RANDOM-INIT SpeakerCNNLarge member checkpoints (timing only, NOT trained) in train_mission2_cnn.py's format."""
import sys, torch
sys.path.insert(0, "/workspace/m2_cpu_timing/code/src")
from dcc_data.mission2_cnn import SpeakerCNNLarge
from dcc_data.mission2_checkpoint import build_checkpoint_metadata
from dcc_data.mission2_spectrogram import N_MELS, LABEL_SOURCE_VERIFIED
from dcc_data.mission2_features import HOP_LENGTH, N_FFT
from dcc_data.guards import TRAINING
prov = {"split": TRAINING, "label_source": LABEL_SOURCE_VERIFIED, "n_mels": N_MELS,
        "hop_length": HOP_LENGTH, "n_fft": N_FFT, "sample_rate": 8000}
for seed in (1, 2, 3):
    torch.manual_seed(seed)
    m = SpeakerCNNLarge()
    md = build_checkpoint_metadata(model_name="large", model_class="SpeakerCNNLarge", index_provenance=prov,
                                   split_seed=42, test_size=0.2, torch_seed=seed)
    torch.save({"best_model_state_dict": m.state_dict(), "metadata": md, "best_val_acc": None,
                "NOTE": "RANDOM INIT - timing only, not trained"}, f"ckpt/m2_cnn_large_RANDOM_ts{seed}.pt")
    print(seed, sum(p.numel() for p in m.parameters()))
