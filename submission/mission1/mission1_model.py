"""Fixed 52D acoustic + normalized 512D WavLM feature pipeline.

This mirrors the officially evaluated pipeline. Gender/text are never read.
The encoder snapshot is located beside the LR checkpoint, under xvector-model/.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

from dcc_data.manifest import ManifestRow
from dcc_data.mission1 import extract_caller_segments, concatenate_caller_waveforms
from dcc_data.mission2_features import acoustic_feature_vector

HEAD_SHA = "4fffb8aa97c2f3083b05820ca45b84ecddbc60c5162e02961546f37d133dd9d5"
ENCODER_SHA = "94c3defe08248d81c7b2bd0a058ea9985269cefed13076434669c47fade41182"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_index(directory, suffix):
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() == suffix:
            if path.stem in result:
                raise ValueError(f"duplicate {suffix} filename stem: {path.stem}")
            result[path.stem] = path.resolve()
    return result


def feature_vector(wav_path, json_path, processor, encoder, device):
    row = ManifestRow("Test", wav_path.stem, str(wav_path.resolve()), str(json_path.resolve()))
    segments = extract_caller_segments(row, Path.cwd())
    waveform = concatenate_caller_waveforms(segments)[0]
    sr = segments[0].sample_rate
    acoustic = acoustic_feature_vector(waveform, sr)
    prefix = waveform[:20 * sr]
    if sr != 16000:
        prefix = librosa.resample(prefix, orig_sr=sr, target_sr=16000)
    prefix = np.asarray(prefix[:320000], dtype=np.float32)
    prefix = np.pad(prefix, (0, max(0, 8000 - len(prefix))))
    inputs = processor(prefix, sampling_rate=16000, return_tensors="pt",
                       return_attention_mask=True)
    with torch.inference_mode():
        embedding = encoder(**{k: v.to(device) for k, v in inputs.items()}).embeddings
        embedding = torch.nn.functional.normalize(embedding, dim=-1)[0].cpu().numpy()
    vector = np.concatenate([acoustic, embedding])
    if vector.shape != (564,) or not np.isfinite(vector).all():
        raise ValueError(f"invalid features for {wav_path.name}")
    return vector


def predict(audio_dir, label_dir, ckpt_path, *, device="auto", threads=4):
    ckpt_path = ckpt_path.resolve()
    model_dir = ckpt_path.parent / "xvector-model"
    if sha256(ckpt_path) != HEAD_SHA or sha256(model_dir / "model.safetensors") != ENCODER_SHA:
        raise ValueError("checkpoint does not match the officially evaluated M1-XVEC-001")
    wavs = file_index(audio_dir, ".wav")
    jsons = file_index(label_dir, ".json")
    if not wavs or set(wavs) != set(jsons):
        raise ValueError(f"audio/label pairing mismatch: missing WAV={sorted(set(jsons)-set(wavs))}, missing JSON={sorted(set(wavs)-set(jsons))}")
    torch.set_num_threads(threads)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, local_files_only=True)
    encoder = WavLMForXVector.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    head = joblib.load(ckpt_path)
    if head.n_features_in_ != 564 or list(head.classes_) != [0, 1]:
        raise ValueError("expected fixed 564D LR head with classes [0, 1]")
    names = sorted(wavs)
    # One call per encoder pass preserves the validated normalization/padding.
    features = np.stack([feature_vector(wavs[name], jsons[name], processor, encoder, device)
                         for name in names])
    predicted = head.predict(features)
    return pd.DataFrame({"audio file name": [wavs[name].name for name in names],
                         "gender": ["F" if value == 0 else "M" for value in predicted]})
