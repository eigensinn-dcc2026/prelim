"""Stage 12 frozen raw-waveform speech representation experiment.

The module is import-safe: it never downloads a checkpoint, reads a dataset,
or starts an experiment. Colab owns execution; all data and model I/O happens
only through explicit checkpoint functions called by the Stage 12 notebook.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import uuid
from collections import OrderedDict
import wave
import zipfile

import numpy as np


SCHEMA = "mission2_raw_waveform_pretrained_speech_v1"
REFERENCE_RUN = "data/processed/mission2_complex_tabular_screen/20261006T031510Z_a9a01a2d"
STAGE11_RUN = "data/processed/mission2_time_frequency_screen/20261007T141754Z_7b1e9644"
CLEAN_59D_FINGERPRINT = "49d65a35d1e98c280f98e45cbed864ef8bac36268ccffbe81aa0985923c90de3"
EXPECTED_SPLIT = {
    "seed": 42,
    "test_size": 0.2,
    "total_rows": 889358,
    "train_rows": 711467,
    "validation_rows": 177891,
    "train_calls": 23360,
    "validation_calls": 5840,
    "call_overlap": 0,
}
REFERENCE_ACCURACY = {"tabm": 0.8839795155460366, "tabular_resnet": 0.8837546587517076}
REFERENCE_COMMON_ERRORS = 17236
DURATION_BINS = ("<0.5", "0.5-1", "1-2", "2-5", ">=5")

CONTRACT = {
    "schema": SCHEMA,
    "question": "Does a frozen public speech encoder add useful utterance-waveform representation beyond the frozen 59D references?",
    "source_split": "Training",
    "evaluation_set": "fixed_internal_validation",
    "primary_metric": "accuracy",
    "seed": 42,
    "threshold": 0.5,
    "official_validation_used": False,
    "validation_gradient_update": False,
    "validation_classifier_fit": False,
    "validation_scaler_fit": False,
    "model_inputs": ["current_utterance_waveform", "own_valid_sample_length"],
    "prohibited_model_inputs": [
        "59D", "duration", "text", "ASR output", "other annotations", "other utterances",
        "call statistics", "utterance order", "absolute startAt/endAt",
    ],
    "split": EXPECTED_SPLIT,
    "clean_59d_fingerprint": CLEAN_59D_FINGERPRINT,
    "frozen_references": REFERENCE_ACCURACY,
}

# One candidate and one probe are fixed before any internal-validation metrics
# are available. The revision is the immutable Hugging Face commit hash.
CONFIG = {
    "schema": "mission2_raw_waveform_pretrained_speech_config_v1",
    "seed": 42,
    "encoder": {
        "repository": "microsoft/wavlm-base-plus",
        "revision": "4c66d4806a428f2e922ccfa1a962776e232d487b",
        "license": "CC-BY-SA-3.0 (Microsoft UniSpeech/WavLM license; model-card license metadata absent)",
        "parameter_scale": "Base+; exact loaded parameter count recorded at CP2",
        "required_sample_rate_hz": 16000,
        "weights_dtype": "float32",
        "inference_autocast": "float16 on CUDA; disabled on CPU",
        "processor_normalization": False,
    },
    "input_audio": {
        "source_sample_rate_hz": 8000,
        "resampler": "torchaudio.functional.resample; sinc_interp_hann; lowpass_filter_width=6; rolloff=0.99",
        "channel_policy": "existing manifest crop must be mono PCM WAV",
        "crop": "existing Stage 11 manifest half-open crop_start_frame:crop_end_frame",
        "padding": "right zero-padding to batch maximum; exact post-resample-length batches; minimum 400 samples (25 ms) only when required by encoder receptive field",
        "attention_mask": "sample-level valid mask passed to WavLM; hidden valid mask from official _get_feature_vector_attention_mask helper",
        "long_utterance_policy": "no truncation or cropping",
    },
    "representation": {
        "hidden_state": "last_hidden_state",
        "pooling": "last_hidden_valid_frame_masked_mean",
        "embedding_dtype": "float32",
        "embedding_dimension": 768,
        "embedding_cache": "uncompressed NPZ shards with exact row/key/label metadata and SHA-256",
    },
    "classifier": {
        "name": "SGDClassifier",
        "loss": "log_loss",
        "penalty": "l2",
        "alpha": 0.0001,
        "learning_rate": "optimal",
        "fit_intercept": True,
        "average": True,
        "epochs": 5,
        "shuffle": "deterministic per-epoch row shuffle; seed=42",
        "scaler": "StandardScaler(with_mean=True, with_std=True); incremental fit on internal-train embeddings only",
        "threshold": 0.5,
        "validation_use": "transform and evaluation only",
    },
    "extraction": {
        "initial_encoder_batch_size": 8,
        "shard_rows": 256,
        "oom_policy": "halve encoder microbatch only; representation and weights remain fixed; fail if batch size 1 OOMs",
        "resume_policy": "reuse a shard only when run config hash, data identity, model revision, split, row positions, keys, calls and labels all match",
    },
    "compute_gate": {"limit_hours": 20.0, "benchmark_target_rows": 3000,
                     "near_gate_band_hours": [18.0, 22.0], "confirmation_target_rows": 5000},
}

RESEARCH = """# Stage 12 candidate research (frozen before validation metrics)\n\n## Compared candidates\n\n| Family | Exact checkpoint considered | Public documentation | Transfer evidence | Technical notes |\n|---|---|---|---|---|\n| Wav2Vec2 Base | `facebook/wav2vec2-base` | [model card](https://huggingface.co/facebook/wav2vec2-base), [paper](https://arxiv.org/abs/2006.11477) | Self-supervised speech representations; common ASR-oriented checkpoint | 16 kHz; Apache-2.0 model card; no specific Korean/paralinguistic advantage asserted |\n| HuBERT Base LS960 | `facebook/hubert-base-ls960` | [model card](https://huggingface.co/facebook/hubert-base-ls960), [paper](https://arxiv.org/abs/2106.07447) | Masked prediction over clustered speech targets | 16 kHz; Apache-2.0 model card; checkpoint is speech/language pretraining, not Mission 2 evidence |\n| **WavLM Base+ (selected)** | **`microsoft/wavlm-base-plus` @ `4c66d4806a428f2e922ccfa1a962776e232d487b`** | [model card](https://huggingface.co/microsoft/wavlm-base-plus), [WavLM paper](https://arxiv.org/abs/2110.13900), [Microsoft repository](https://github.com/microsoft/unilm/tree/master/wavlm) | Paper and official card emphasize speaker identity, paralinguistics and general speech representation transfer; selection does not predict Mission 2 accuracy | 16 kHz; Base+ scale; processor says no waveform normalization and right padding; official WavLM implementation provides the downsampled feature-mask helper. Model card documents English pretraining, a material Korean transfer risk. License is CC BY-SA 3.0 from the linked Microsoft license; model card does not expose a machine-readable license tag. |\n\n## Selection rationale\n\nSelect one Base-scale candidate with direct speaker/paralinguistic transfer motivation and an exact immutable checkpoint revision. WavLM Base+ is chosen for that evidence, not for a forecast of Korean Mission 2 accuracy. Its English pretraining is a known limitation. The 8 kHz phone waveform is deterministically resampled to the checkpoint's 16 kHz input. The last-layer masked mean is fixed by the experiment request; no ASR decoding, CTC, text, other utterance, 59D feature, or annotation enters the encoder or classifier.\n\nThe official model config reports 768 hidden units, 12 Transformer layers, and convolution strides `[5,2,2,2,2,2,2]`; CP2 records the loaded parameter count. WavLM's official helper maps the sample-level attention mask to hidden frames. Exact post-resampling lengths are grouped before batching so group-normalization behavior is independent of unrelated batch padding; valid hidden frames alone are pooled.\n\n## Sources\n\n- [WavLM paper](https://arxiv.org/abs/2110.13900)\n- [WavLM official model card](https://huggingface.co/microsoft/wavlm-base-plus)\n- [WavLM official config](https://huggingface.co/microsoft/wavlm-base-plus/blob/4c66d4806a428f2e922ccfa1a962776e232d487b/config.json)\n- [WavLM official processor config](https://huggingface.co/microsoft/wavlm-base-plus/blob/4c66d4806a428f2e922ccfa1a962776e232d487b/preprocessor_config.json)\n- [WavLM official implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wavlm/modeling_wavlm.py)\n- [Microsoft UniSpeech license](https://github.com/microsoft/UniSpeech/blob/main/LICENSE)\n- [Wav2Vec2 Base model card](https://huggingface.co/facebook/wav2vec2-base)\n- [HuBERT Base LS960 model card](https://huggingface.co/facebook/hubert-base-ls960)\n"""


def fingerprint(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_new(path: str | Path, value) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    temp = target.with_name(target.name + f".{uuid.uuid4().hex}.partial")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists():
            raise FileExistsError(target)
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def write_json_atomic(path: str | Path, value) -> None:
    """Atomically replace the mutable progress manifest, never a data shard."""
    target = Path(path)
    temp = target.with_name(target.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def environment_info() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "torch", "torchaudio", "transformers", "safetensors", "opensmile"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
              "device": "unverified", "gpu": None, "cuda": None,
              "implementation_sha256": file_sha256(Path(__file__))}
    try:
        import torch
        result.update(device="cuda" if torch.cuda.is_available() else "cpu",
                      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      cuda=torch.version.cuda, torch_import_error=None)
    except (ImportError, OSError, RuntimeError) as error:
        result["torch_import_error"] = f"{type(error).__name__}: {error}"
    return result


def require_gate_pass(gate: dict) -> None:
    if not isinstance(gate, dict) or gate.get("status") != "PASS":
        raise RuntimeError("Compute gate is not PASS; full extraction is forbidden")


def compute_gate(estimate: dict, *, limit_hours: float = 20.0) -> dict:
    hours = float(estimate["projected_full_hours"])
    if not math.isfinite(hours) or hours < 0 or limit_hours <= 0:
        raise ValueError("Invalid compute projection")
    status = "PASS" if hours <= limit_hours else "FAIL"
    return {"schema": "mission2_stage12_compute_gate_v1", "status": status,
            "decision": "FULL_EXPERIMENT_ALLOWED" if status == "PASS" else "RESOURCE/TIME-BUDGET INFEASIBLE",
            "performance_status": "NOT_EVALUATED", "limit_hours": float(limit_hours),
            "projected_full_hours": hours, "projection": estimate}


def project_runtime(*, benchmark_rows: int, benchmark_seconds: float,
                    train_rows: int, validation_rows: int,
                    classifier_seconds: float, artifact_seconds: float,
                    duration_bin_rates: dict, duration_bin_shares: dict,
                    startup_seconds: float = 0.0) -> dict:
    """Project both observed mixed-batch and duration-weighted throughput."""
    if min(benchmark_rows, train_rows, validation_rows) <= 0 or min(
            benchmark_seconds, classifier_seconds, artifact_seconds, startup_seconds) < 0:
        raise ValueError("Invalid benchmark inputs")
    rows_per_second = benchmark_rows / benchmark_seconds
    total_rows = train_rows + validation_rows
    mixed_extract_seconds = total_rows / rows_per_second
    if set(duration_bin_rates) != set(DURATION_BINS) or set(duration_bin_shares) != set(DURATION_BINS):
        raise ValueError("All five duration bins are required for weighted projection")
    shares = np.asarray([duration_bin_shares[name] for name in DURATION_BINS], dtype=np.float64)
    rates = np.asarray([duration_bin_rates[name] for name in DURATION_BINS], dtype=np.float64)
    if (not np.isfinite(shares).all() or not np.isfinite(rates).all() or (shares < 0).any()
            or (rates <= 0).any() or not np.isclose(shares.sum(), 1.0, atol=1e-9)):
        raise ValueError("Invalid duration-bin shares or rates")
    weighted_rows_per_second = 1.0 / float(np.sum(shares / rates))
    weighted_extract_seconds = total_rows / weighted_rows_per_second
    full_seconds_mixed = mixed_extract_seconds + classifier_seconds + artifact_seconds + startup_seconds
    full_seconds_weighted = weighted_extract_seconds + classifier_seconds + artifact_seconds + startup_seconds
    return {
        "benchmark_rows": int(benchmark_rows), "benchmark_seconds": float(benchmark_seconds),
        "train_rows": int(train_rows), "validation_rows": int(validation_rows),
        "total_rows": int(total_rows), "mixed_rows_per_second": rows_per_second,
        "duration_weighted_rows_per_second": weighted_rows_per_second,
        "duration_bin_rates_rows_per_second": {k: float(duration_bin_rates[k]) for k in DURATION_BINS},
        "duration_bin_shares": {k: float(duration_bin_shares[k]) for k in DURATION_BINS},
        "mixed_projection_seconds": full_seconds_mixed,
        "duration_weighted_projection_seconds": full_seconds_weighted,
        "projected_full_hours": max(full_seconds_mixed, full_seconds_weighted) / 3600,
        "classifier_fit_train_prediction_validation_prediction_and_artifact_seconds":
            float(classifier_seconds + artifact_seconds),
        "startup_model_load_and_cuda_warmup_seconds": float(startup_seconds),
        "projection_policy": "conservative maximum of mixed representative throughput and duration-share-weighted throughput, plus measured classifier/evaluation/reload/artifact budget",
    }


def pad_for_encoder(waveforms, *, sample_rate_hz: int):
    """Right-pad a batch and preserve real lengths; never crop or repeat audio."""
    if sample_rate_hz != CONFIG["encoder"]["required_sample_rate_hz"]:
        raise ValueError("Encoder-rate waveforms required")
    arrays = [np.asarray(w, dtype=np.float32).reshape(-1) for w in waveforms]
    if not arrays or any(w.size == 0 or not np.isfinite(w).all() for w in arrays):
        raise ValueError("Empty or nonfinite waveform; rows cannot be skipped")
    valid_lengths = np.asarray([w.size for w in arrays], dtype=np.int64)
    minimum = 400  # WavLM's configured conv receptive field at 16 kHz.
    padded_length = max(minimum, int(valid_lengths.max()))
    padded = np.zeros((len(arrays), padded_length), dtype=np.float32)
    for index, wave in enumerate(arrays):
        padded[index, :wave.size] = wave
    return padded, valid_lengths, int((valid_lengths < minimum).sum())


def masked_mean_pool(hidden_state, frame_mask):
    hidden = np.asarray(hidden_state)
    mask = np.asarray(frame_mask, dtype=bool)
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or not mask.any(axis=1).all():
        raise ValueError("Invalid hidden-state valid-frame mask")
    if not np.isfinite(hidden).all():
        raise ValueError("Nonfinite encoder hidden state")
    weights = mask[:, :, None].astype(hidden.dtype, copy=False)
    pooled = (hidden * weights).sum(axis=1) / weights.sum(axis=1)
    if not np.isfinite(pooled).all():
        raise ValueError("Nonfinite pooled representation")
    return pooled


def _model_hidden_mask(model, attention_mask, hidden_length):
    helper = getattr(model, "_get_feature_vector_attention_mask", None)
    if helper is None:
        raise RuntimeError("Pinned WavLM feature-vector attention-mask helper is unavailable")
    mask = helper(hidden_length, attention_mask)
    if mask.shape != (attention_mask.shape[0], hidden_length):
        raise ValueError("Official WavLM helper returned a wrong-shaped hidden mask")
    return mask.bool()


def encode_batch(model, waveforms, *, device: str, use_amp: bool):
    """Encode only waveforms; labels and row metadata stay outside this function."""
    import torch

    padded, valid_lengths, padded_rows = pad_for_encoder(waveforms, sample_rate_hz=16000)
    values = torch.from_numpy(padded).to(device=device, dtype=torch.float32)
    positions = torch.arange(values.shape[1], device=device)[None, :]
    attention_mask = positions < torch.as_tensor(valid_lengths, device=device)[:, None]
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Frozen encoder has a trainable parameter")
    amp_context = (torch.autocast("cuda", dtype=torch.float16)
                   if use_amp and str(device).startswith("cuda") else torch.inference_mode())
    with torch.inference_mode():
        with amp_context:
            output = model(input_values=values, attention_mask=attention_mask.long(), return_dict=True)
        hidden = output.last_hidden_state
        frame_mask = _model_hidden_mask(model, attention_mask.long(), hidden.shape[1])
        weights = frame_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        denominator = weights.sum(dim=1)
        if (denominator <= 0).any():
            raise ValueError("WavLM marked an utterance with no valid hidden frames")
        pooled = (hidden * weights).sum(dim=1) / denominator
        pooled = pooled.float()
        if not torch.isfinite(pooled).all():
            raise ValueError("Nonfinite pooled embedding")
        embedding = pooled.cpu().numpy().astype(np.float32, copy=False)
    return embedding, {"padded_rows": padded_rows, "batch_rows": len(waveforms),
                       "input_samples": int(values.shape[1]), "embedding_dim": int(embedding.shape[1]),
                       "valid_hidden_frames": frame_mask.sum(dim=1).cpu().tolist(),
                       "hidden_frame_count": int(hidden.shape[1])}


def resample_waveform(waveform: np.ndarray, *, original_rate_hz: int, target_rate_hz: int = 16000):
    import torch
    import torchaudio.functional as audio_functional

    wave = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if wave.size == 0 or not np.isfinite(wave).all():
        raise ValueError("Empty/nonfinite crop; row cannot be skipped")
    if original_rate_hz != 8000 or target_rate_hz != 16000:
        raise ValueError("This pinned experiment only supports 8 kHz to 16 kHz")
    tensor = torch.from_numpy(wave)
    result = audio_functional.resample(tensor, orig_freq=original_rate_hz, new_freq=target_rate_hz,
                                      resampling_method="sinc_interp_hann", lowpass_filter_width=6,
                                      rolloff=0.99)
    expected = int(math.ceil(wave.size * target_rate_hz / original_rate_hz))
    if result.numel() != expected or not torch.isfinite(result).all():
        raise ValueError("Resampling output length/content is invalid")
    return result.numpy().astype(np.float32, copy=False)


def load_pinned_encoder(device="cuda"):
    """Load only the immutable WavLM revision; no latest/main fallback."""
    import torch
    from transformers import AutoFeatureExtractor, WavLMModel

    model_id = CONFIG["encoder"]["repository"]
    revision = CONFIG["encoder"]["revision"]
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, revision=revision,
                                                              trust_remote_code=False)
    model = WavLMModel.from_pretrained(model_id, revision=revision, trust_remote_code=False,
                                       torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if model.config.hidden_size != CONFIG["representation"]["embedding_dimension"]:
        raise ValueError("Pinned WavLM hidden dimension differs from frozen config")
    if feature_extractor.sampling_rate != 16000 or feature_extractor.do_normalize:
        raise ValueError("Pinned processor sample-rate/normalization config differs")
    return model, feature_extractor, {
        "repository": model_id, "revision": revision, "parameter_count": int(parameter_count),
        "hidden_size": int(model.config.hidden_size), "num_hidden_layers": int(model.config.num_hidden_layers),
        "sampling_rate_hz": int(feature_extractor.sampling_rate),
        "processor_do_normalize": bool(feature_extractor.do_normalize),
        "model_training": bool(model.training),
        "all_parameters_frozen": all(not p.requires_grad for p in model.parameters()),
    }


def _source_array_identity(source, train_idx, validation_idx):
    from .mission2_time_frequency_screen import _source_identity
    return _source_identity(source, train_idx, validation_idx)


def _saved_transfer_proof_matches(saved_proof: dict, source_proof: dict) -> bool:
    """Match Stage11's verified proof, which adds status=PASS to the source JSON."""
    return saved_proof == {**source_proof, "status": "PASS"}


class _ArchiveTrainingAudioReader:
    """Read WAV members from existing organizer ZIPs without extracting them."""

    def __init__(self, members: dict, *, cache_wavs: int = 8):
        self.members = dict(members)
        self.cache_wavs = int(cache_wavs)
        self._archives = {}
        self._wave_cache = OrderedDict()

    def _read_member(self, relative_path: str) -> bytes:
        if relative_path not in self.members:
            raise FileNotFoundError(f"WAV is absent from the verified Training archives: {relative_path}")
        archive_path, member_name, expected_size, expected_crc = self.members[relative_path]
        if relative_path in self._wave_cache:
            content = self._wave_cache.pop(relative_path)
            self._wave_cache[relative_path] = content
            return content
        archive_path = str(archive_path)
        archive = self._archives.get(archive_path)
        if archive is None:
            archive = zipfile.ZipFile(archive_path, "r")
            self._archives[archive_path] = archive
        info = archive.getinfo(member_name)
        if (info.file_size, info.CRC) != (int(expected_size), int(expected_crc)):
            raise ValueError(f"Organizer ZIP member changed after CP1 inventory: {relative_path}")
        content = archive.read(member_name)  # Reads and CRC-checks this WAV only.
        if len(content) != int(expected_size):
            raise ValueError(f"Truncated organizer ZIP member: {relative_path}")
        self._wave_cache[relative_path] = content
        while len(self._wave_cache) > self.cache_wavs:
            self._wave_cache.popitem(last=False)
        return content

    def read_wav_frames(self, relative_path: str, start_frame: int, end_frame: int):
        """Decode the same half-open PCM crop as ``audio.read_wav_frames``."""
        from .audio import AudioSegment

        content = self._read_member(str(relative_path))
        with wave.open(io.BytesIO(content), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise ValueError(f"Only uncompressed PCM WAV is supported: {relative_path}")
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            if start_frame < 0 or end_frame <= start_frame or end_frame > frame_count:
                raise ValueError(f"Invalid WAV frame interval [{start_frame}, {end_frame}): {relative_path}")
            wav_file.setpos(start_frame)
            raw = wav_file.readframes(end_frame - start_frame)
        requested_frames = end_frame - start_frame
        expected_bytes = requested_frames * channels * sample_width
        if len(raw) != expected_bytes:
            raise ValueError(f"Truncated WAV crop: {relative_path}")
        if sample_width == 1:
            values = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sample_width == 2:
            values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 4:
            values = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")
        return AudioSegment(values.reshape(-1, channels).T.copy(), sample_rate)

    def close(self):
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()
        self._wave_cache.clear()


def load_existing_training_context(repo_root: str | Path, raw_root: str | Path,
                                  transfer_identity_path: str | Path,
                                  stage11_identity_path: str | Path,
                                  *, archive_root: str | Path | None = None):
    """Reuse the existing clean manifest/split and saved Stage11 byte proof.

    Deliberately does not call Stage11.prepare_context: that function stages
    shared ZIPs and rereads every raw file to recompute a costly content hash.
    Reuse the old local raw root when it still exists. If Colab has lost its
    ephemeral /content tree, index the same existing organizer ZIPs and stream
    WAV members on demand; never extract or reconstruct the corpus.
    """
    from .mission2_clean_stage4 import _source_rows
    from .mission2_model_experiments import standalone_training_cache_paths
    from .mission2_time_frequency_screen import WaveformContext, fixed_split, fingerprint as stage11_fingerprint

    root = Path(repo_root).resolve()
    raw = Path(raw_root).resolve()
    use_archives = not raw.is_dir()
    if raw.is_dir() and (not raw.is_relative_to(Path("/content")) or raw.is_relative_to(Path("/content/drive"))):
        raise ValueError("The reused extracted Training root must be local /content storage")
    if use_archives and archive_root is None:
        raise ValueError("The existing /content raw root is absent and no organizer archive root was supplied")
    saved = read_json(stage11_identity_path)
    transfer_path = Path(transfer_identity_path)
    proof = read_json(transfer_path)
    paths = standalone_training_cache_paths(root)
    manifest_path, metadata_path = Path(paths.manifest_parquet), Path(paths.manifest_metadata)
    if file_sha256(manifest_path) != saved.get("manifest_sha256"):
        raise ValueError("Training manifest differs from the saved Stage11 identity")
    if file_sha256(metadata_path) != saved.get("manifest_metadata_sha256"):
        raise ValueError("Training manifest metadata differs from the saved Stage11 identity")
    if file_sha256(transfer_path) != saved.get("transfer_identity_sha256"):
        raise ValueError("Existing official-source transfer proof differs from saved Stage11 identity")
    for key, expected in {
        "schema": "mission2_egemaps_transfer_identity_v1",
        "clean_reference_fingerprint": CLEAN_59D_FINGERPRINT,
        "raw_file_count": 58400,
        "source_split": "Training",
        "rows": EXPECTED_SPLIT["total_rows"],
    }.items():
        if proof.get(key) != expected:
            raise ValueError(f"Organizer Training transfer proof differs: {key}")
    if not _saved_transfer_proof_matches(saved.get("transfer_proof"), proof):
        raise ValueError("Stage11 data identity and existing transfer proof differ")
    saved_root = Path(saved.get("raw_root", "")).resolve()
    if saved_root != raw:
        raise ValueError("Current raw-root path differs from the previously byte-verified Colab path")
    payload = {key: value for key, value in saved.items() if key != "fingerprint"}
    if stage11_fingerprint(payload) != saved.get("fingerprint"):
        raise ValueError("Saved Stage11 data identity fingerprint is invalid")

    import pandas as pd
    manifest = pd.read_parquet(manifest_path)
    source = _source_rows(manifest, expected_rows=EXPECTED_SPLIT["total_rows"], expected_calls=29200)
    if not (source.crop_end_frame > source.crop_start_frame).all():
        raise ValueError("Training manifest contains an invalid/empty WAV crop")
    train_idx, validation_idx, split = fixed_split(source.call_id.to_numpy().astype(str))
    if split != EXPECTED_SPLIT:
        raise ValueError(f"Fixed call_id split contract differs: {split}")
    arrays = _source_array_identity(source, train_idx, validation_idx)
    if arrays != saved.get("arrays") or split != saved.get("split"):
        raise ValueError("Current row/key/call/label/split identity differs from Stage11")
    calls_train = set(source.iloc[train_idx].call_id.astype(str))
    calls_validation = set(source.iloc[validation_idx].call_id.astype(str))
    if calls_train & calls_validation:
        raise ValueError("call_id overlap detected")

    # Count the already extracted files without rereading their bytes. This is
    # a continuity check, not a new full raw-content verification.
    archive_reader = None
    if use_archives:
        from .mission2_egemaps_colab import inspect_shared_training_archives
        from .mission2_model_family_screen import shared_training_archive_paths

        source_root = Path(archive_root).resolve()
        archive_paths = shared_training_archive_paths(source_root)
        archive_plan = inspect_shared_training_archives(source, archive_paths)
        if archive_plan["files"] != proof["raw_file_count"]:
            raise ValueError("Organizer ZIP member inventory differs from the saved transfer proof")
        if archive_plan["wav_files"] != 29200 or archive_plan["json_files"] != 29200:
            raise ValueError("Organizer ZIP Training WAV/JSON counts differ from the saved proof")
        archive_reader = _ArchiveTrainingAudioReader(archive_plan["members"])
        raw_file_count = archive_plan["files"]
        continuity = (
            "same saved manifest/proof hashes and fixed split; existing organizer ZIPs cover the exact "
            "58,400 Training manifest paths; WAV members are streamed without extraction"
        )
        active_source = str(source_root)
    else:
        raw_file_count = sum(1 for path in raw.rglob("*") if path.is_file() and path.suffix.lower() in {".wav", ".json"})
        if raw_file_count != proof["raw_file_count"]:
            raise ValueError("Current raw folder file count differs from the saved transfer proof")
        continuity = "same saved raw-root path, same manifest/proof hashes, same 58,400-file inventory count, exact rows/keys/labels/crops and fixed split"
        active_source = str(raw)
    identity = saved
    train_idx.setflags(write=False)
    validation_idx.setflags(write=False)
    context = WaveformContext(source, train_idx, validation_idx, raw, identity, None,
                              {"status": "not_requested"})
    if archive_reader is not None:
        context.archive_reader = archive_reader
    return context, {
        "status": "PASS_REUSED_EXISTING_PROOF",
        "provenance": "existing Stage11 transfer proof from shared organizer Training ZIPs",
        "current_content_rehash": False,
        "continuity": continuity,
        "clean_59d_fingerprint": CLEAN_59D_FINGERPRINT,
        "data_identity_fingerprint": identity["fingerprint"],
        "split": split,
        "row_array_identity": "PASS",
        "call_overlap": 0,
        "raw_root": str(raw), "active_source_root": active_source,
        "source_mode": "stream_existing_organizer_zips" if use_archives else "existing_extracted_training_copy",
        "raw_file_count": raw_file_count,
        "transfer_proof": proof,
        "limitation": "The previous full byte-level transfer proof is reused; this checkpoint does not reread/hash all WAV/JSON bytes.",
    }


def create_run(output_parent: str | Path, context, environment: dict, *, run_id=None):
    config = json.loads(json.dumps(CONFIG))
    config_hash = fingerprint(config)
    environment_identity = {key: environment.get(key) for key in
                            ("python", "platform", "packages", "device", "gpu", "cuda", "implementation_sha256")}
    environment_hash = fingerprint(environment_identity)
    identifier = run_id or f"stage12_wavlm_{config_hash[:10]}_{context.identity['fingerprint'][:10]}_{environment_hash[:8]}"
    if Path(identifier).name != identifier or identifier in ("", ".", ".."):
        raise ValueError("Invalid Stage12 run ID")
    parent = Path(output_parent)
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / identifier
    if root.exists():
        saved_contract = read_json(root / "contract.json")
        saved_config = read_json(root / "config.json")
        saved_environment = read_json(root / "environment.json")
        saved_identity = read_json(root / "data_identity.json")
        saved_environment_identity = {key: saved_environment.get(key) for key in environment_identity}
        if (saved_contract.get("config_hash") != config_hash or saved_config != config
                or saved_contract.get("data_identity_fingerprint") != context.identity["fingerprint"]
                or saved_identity != context.identity
                or fingerprint(saved_environment_identity) != environment_hash):
            raise ValueError("Existing versioned Stage12 run has a different config/data/environment identity")
    else:
        root.mkdir(exist_ok=False)
        write_json_new(root / "contract.json", {**CONTRACT, "config_hash": config_hash,
                                                  "data_identity_fingerprint": context.identity["fingerprint"],
                                                  "environment_identity_hash": environment_hash})
        write_json_new(root / "config.json", config)
        write_json_new(root / "environment.json", environment)
        write_json_new(root / "data_identity.json", context.identity)
        (root / "candidate_research.md").write_text(RESEARCH, encoding="utf-8", newline="\n")
        for split in ("benchmark", "train", "validation"):
            (root / "embeddings" / split).mkdir(parents=True)
    return root, config, config_hash


def crop_and_resample(context, position: int):
    from .audio import read_wav_frames

    row = context.source.iloc[int(position)]
    relative_path = str(row.audio_path)
    start, end = int(row.crop_start_frame), int(row.crop_end_frame)
    if start < 0 or end <= start:
        raise ValueError(f"Invalid crop for {row.record_key}")
    archive_reader = getattr(context, "archive_reader", None)
    if archive_reader is not None:
        segment = archive_reader.read_wav_frames(relative_path, start, end)
    else:
        root = Path(context.raw_root).resolve()
        path = (root / relative_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"WAV path is missing or escapes raw root: {relative_path}")
        segment = read_wav_frames(path, start_frame=start, end_frame=end)
    wave = np.asarray(segment.waveform, dtype=np.float32)
    if segment.sample_rate != 8000 or wave.shape != (1, end - start) or not np.isfinite(wave).all():
        raise ValueError(f"Invalid native 8 kHz mono crop: {row.record_key}")
    return resample_waveform(wave[0], original_rate_hz=segment.sample_rate)


def _row_records(context, positions):
    frame = context.source.iloc[np.asarray(positions, dtype=np.int64)]
    return {"record_key": frame.record_key.astype(str).tolist(),
            "call_id": frame.call_id.astype(str).tolist(),
            "y_true": frame.speaker.to_numpy(dtype=np.int64).tolist()}


def save_shard(directory: str | Path, split: str, shard_index: int, rows: dict,
               embeddings: np.ndarray, *, config_hash: str, data_fingerprint: str,
               row_positions=None):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    keys, calls, labels = (np.asarray(rows[name]) for name in ("record_key", "call_id", "y_true"))
    matrix = np.asarray(embeddings, dtype=np.float32)
    if (matrix.ndim != 2 or len(matrix) != len(keys) or len(calls) != len(keys) or len(labels) != len(keys)
            or not len(keys) or len(set(keys.tolist())) != len(keys)
            or not np.isin(labels, [0, 1]).all() or not np.isfinite(matrix).all()):
        raise ValueError("Invalid shard row identity, labels, or embeddings")
    target = root / f"{split}_{int(shard_index):06d}.npz"
    sidecar = target.with_suffix(".json")
    if target.exists() or sidecar.exists():
        raise FileExistsError(f"Immutable shard already exists: {target}")
    temp = target.with_name(target.name + f".{uuid.uuid4().hex}.partial")
    try:
        with temp.open("xb") as stream:
            np.savez(stream, record_key=keys.astype(str), call_id=calls.astype(str),
                     y_true=labels.astype(np.int64), embedding=matrix,
                     row_position=np.asarray(row_positions if row_positions is not None else [], dtype=np.int64),
                     config_hash=np.asarray(config_hash),
                     data_identity_fingerprint=np.asarray(data_fingerprint),
                     model_repository=np.asarray(CONFIG["encoder"]["repository"]),
                     model_revision=np.asarray(CONFIG["encoder"]["revision"]))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
        metadata = {
            "schema": "mission2_stage12_embedding_shard_v1", "split": split,
            "shard_index": int(shard_index), "config_hash": config_hash,
            "data_identity_fingerprint": data_fingerprint,
            "model_repository": CONFIG["encoder"]["repository"],
            "model_revision": CONFIG["encoder"]["revision"],
            "rows": int(len(keys)), "embedding_dimension": int(matrix.shape[1]),
            "dtype": "float32", "sha256": file_sha256(target),
        }
        write_json_new(sidecar, metadata)
    finally:
        if temp.exists():
            temp.unlink()
    return target, metadata


def load_shard(path: str | Path, *, expected_meta: dict, expected_config_hash: str,
               expected_data_fingerprint: str, expected_rows=None):
    target = Path(path)
    metadata = read_json(target.with_suffix(".json"))
    for key, expected in {"config_hash": expected_config_hash,
                          "data_identity_fingerprint": expected_data_fingerprint,
                          "model_repository": CONFIG["encoder"]["repository"],
                          "model_revision": CONFIG["encoder"]["revision"]}.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Embedding shard identity mismatch: {key}")
    if expected_meta is not None and metadata != expected_meta:
        raise ValueError("Embedding shard metadata differs from manifest")
    if file_sha256(target) != metadata.get("sha256"):
        raise ValueError("Embedding shard checksum mismatch")
    with np.load(target, allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files}
    keys = result["record_key"].astype(str).tolist()
    if (len(keys) != metadata["rows"] or len(set(keys)) != len(keys)
            or result["embedding"].dtype != np.float32
            or result["embedding"].shape != (len(keys), metadata["embedding_dimension"])
            or not np.isfinite(result["embedding"]).all()
            or str(result["config_hash"].item()) != expected_config_hash
            or str(result["data_identity_fingerprint"].item()) != expected_data_fingerprint
            or str(result["model_repository"].item()) != CONFIG["encoder"]["repository"]
            or str(result["model_revision"].item()) != CONFIG["encoder"]["revision"]):
        raise ValueError("Embedding shard content is incomplete or invalid")
    if expected_rows is not None:
        for field in ("record_key", "call_id", "y_true"):
            if result[field].tolist() != expected_rows[field]:
                raise ValueError(f"Embedding shard current-source row mismatch: {field}")
    return result


def recover_shard_without_sidecar(path: str | Path, *, expected_rows: dict,
                                  expected_positions, expected_config_hash: str,
                                  expected_data_fingerprint: str):
    """Rebuild metadata only when a complete atomic NPZ was saved before disconnect."""
    target = Path(path)
    if not target.is_file() or target.with_suffix(".json").exists():
        raise ValueError("Only a complete shard without its sidecar can be recovered")
    with np.load(target, allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files}
    for field, expected in {
        "record_key": expected_rows["record_key"],
        "call_id": expected_rows["call_id"],
        "y_true": expected_rows["y_true"],
        "row_position": np.asarray(expected_positions, dtype=np.int64).tolist(),
        "config_hash": expected_config_hash,
        "data_identity_fingerprint": expected_data_fingerprint,
        "model_repository": CONFIG["encoder"]["repository"],
        "model_revision": CONFIG["encoder"]["revision"],
    }.items():
        value = result[field].item() if result[field].ndim == 0 else result[field].tolist()
        if value != expected:
            raise ValueError(f"Orphan shard identity differs; cannot recover: {field}")
    matrix = result["embedding"]
    if (matrix.dtype != np.float32 or matrix.ndim != 2
            or len(matrix) != len(expected_rows["record_key"])
            or not np.isfinite(matrix).all()):
        raise ValueError("Orphan embedding shard is incomplete/nonfinite")
    metadata = {
        "schema": "mission2_stage12_embedding_shard_v1", "split": target.stem.rsplit("_", 1)[0],
        "shard_index": int(target.stem.rsplit("_", 1)[1]), "config_hash": expected_config_hash,
        "data_identity_fingerprint": expected_data_fingerprint,
        "model_repository": CONFIG["encoder"]["repository"],
        "model_revision": CONFIG["encoder"]["revision"],
        "rows": int(len(matrix)), "embedding_dimension": int(matrix.shape[1]),
        "dtype": "float32", "sha256": file_sha256(target),
    }
    write_json_new(target.with_suffix(".json"), metadata)
    return metadata


def _save_manifest(manifest_path: Path, manifest):
    write_json_atomic(manifest_path, manifest)


def _encode_by_exact_length(model, waves, *, device, use_amp, batch_size):
    """Exact-length groups prevent unrelated batch padding from changing WavLM GroupNorm inputs."""
    result = [None] * len(waves)
    groups = {}
    for index, wave in enumerate(waves):
        groups.setdefault(int(len(wave)), []).append(index)
    observed_padded = 0
    actual_batch_sizes = []
    oom_adjustments = []
    valid_hidden_frames = []

    def encode_indices(selected):
        try:
            embeddings, info = encode_batch(model, [waves[i] for i in selected], device=device, use_amp=use_amp)
        except RuntimeError as error:
            import torch
            is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
            if not is_oom or len(selected) <= 1:
                raise
            reduced = max(1, len(selected) // 2)
            oom_adjustments.append({"from_batch_size": len(selected), "to_batch_size": reduced,
                                    "reason": f"{type(error).__name__}: {error}"})
            del error
            import gc
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            for offset in range(0, len(selected), reduced):
                encode_indices(selected[offset:offset + reduced])
            return
        nonlocal observed_padded
        observed_padded += info["padded_rows"]
        actual_batch_sizes.append(len(selected))
        valid_hidden_frames.extend(int(value) for value in info["valid_hidden_frames"])
        for local, original in enumerate(selected):
            result[original] = embeddings[local]

    for _, indices in sorted(groups.items()):
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            encode_indices(selected)
    if any(item is None for item in result):
        raise ValueError("Encoder skipped one or more rows")
    return np.stack(result).astype(np.float32, copy=False), {
        "padded_rows": observed_padded, "skipped_rows": 0,
        "batch_size_min": min(actual_batch_sizes), "batch_size_max": max(actual_batch_sizes),
        "oom_adjustments": oom_adjustments,
        "valid_hidden_frames": valid_hidden_frames,
    }


def _duration_seconds(context, positions):
    rows = context.source.iloc[np.asarray(positions, dtype=np.int64)]
    return (rows.crop_end_frame.to_numpy(dtype=np.int64) - rows.crop_start_frame.to_numpy(dtype=np.int64)) / 8000.0


def duration_bin(seconds):
    if seconds < .5: return "<0.5"
    if seconds < 1: return "0.5-1"
    if seconds < 2: return "1-2"
    if seconds < 5: return "2-5"
    return ">=5"


def representative_training_sample(context, *, target_rows=3000, seed=42):
    positions = context.train_idx
    durations = _duration_seconds(context, positions)
    bins = np.asarray([duration_bin(x) for x in durations], dtype=object)
    populations = {name: np.flatnonzero(bins == name) for name in DURATION_BINS}
    shares = {name: float(len(populations[name]) / len(positions)) for name in DURATION_BINS}
    quotas = {name: min(len(populations[name]), max(1, int(math.floor(target_rows * shares[name]))))
              if len(populations[name]) else 0 for name in DURATION_BINS}
    while sum(quotas.values()) > min(target_rows, len(positions)):
        candidates = [name for name in DURATION_BINS if quotas[name] > 1]
        if not candidates:
            raise ValueError("Requested benchmark is too small to represent every nonempty duration bin")
        name = min(candidates, key=lambda item: (target_rows * shares[item] - quotas[item], -DURATION_BINS.index(item)))
        quotas[name] -= 1
    remainder = min(target_rows, len(positions)) - sum(quotas.values())
    while remainder > 0:
        candidates = [name for name in DURATION_BINS if quotas[name] < len(populations[name])]
        if not candidates: break
        name = max(candidates, key=lambda item: (target_rows * shares[item] - quotas[item], -DURATION_BINS.index(item)))
        quotas[name] += 1
        remainder -= 1
    rng = np.random.default_rng(seed)
    result, counts = [], {}
    for name in DURATION_BINS:
        chosen = rng.choice(populations[name], size=quotas[name], replace=False) if quotas[name] else np.asarray([], dtype=int)
        selected = positions[np.sort(chosen)]
        result.extend(selected.tolist())
        counts[name] = {"population": int(len(populations[name])), "sampled": int(len(chosen)),
                        "share": shares[name], "sample_positions": selected.astype(int).tolist()}
    return np.asarray(result, dtype=np.int64), counts


def benchmark_encoder(context, run_root, model, *, device="cuda", use_amp=True,
                      target_rows=3000, batch_size=8, startup_seconds=0.0):
    from time import perf_counter

    positions, bin_info = representative_training_sample(context, target_rows=target_rows)
    overall_start = perf_counter()
    benchmark_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_") + uuid.uuid4().hex[:8]
    output_dir = Path(run_root) / "benchmarks" / benchmark_id
    all_rows, all_embeddings, bin_rates, bin_seconds, cache_seconds, cache_bytes = [], [], {}, {}, 0.0, 0
    total_audio_seconds = 0.0
    padded_rows = 0
    batch_sizes, oom_adjustments = [], []
    shard_index = 0
    for name in DURATION_BINS:
        picked = np.asarray(bin_info[name]["sample_positions"], dtype=np.int64)
        if not len(picked):
            # An absent bin has zero population weight and is not assigned an invented rate.
            bin_rates[name] = None
            bin_seconds[name] = 0.0
            continue
        started = perf_counter()
        waves = [crop_and_resample(context, int(position)) for position in picked]
        embeddings, batch_info = _encode_by_exact_length(model, waves, device=device, use_amp=use_amp,
                                                         batch_size=batch_size)
        padded_rows += batch_info["padded_rows"]
        batch_sizes.extend([batch_info["batch_size_min"], batch_info["batch_size_max"]])
        oom_adjustments.extend(batch_info["oom_adjustments"])
        total_audio_seconds += float(np.sum(_duration_seconds(context, picked)))
        rows = _row_records(context, picked)
        cache_started = perf_counter()
        _, metadata = save_shard(output_dir, "benchmark", shard_index, rows, embeddings,
                                 config_hash=fingerprint(CONFIG),
                                 data_fingerprint=context.identity["fingerprint"],
                                 row_positions=picked)
        cache_seconds += perf_counter() - cache_started
        cache_bytes += int((output_dir / f"benchmark_{shard_index:06d}.npz").stat().st_size)
        bin_seconds[name] = perf_counter() - started
        bin_rates[name] = len(picked) / bin_seconds[name]
        all_rows.extend(zip(rows["record_key"], rows["call_id"], rows["y_true"]))
        all_embeddings.append(embeddings)
        bin_info[name].update({"measured_seconds_including_shard_write": bin_seconds[name], "rows_per_second": bin_rates[name],
                               "audio_seconds": float(np.sum(_duration_seconds(context, picked))),
                               "embedding_shard_sha256": metadata["sha256"],
                               "embedding_shard_bytes": int((output_dir / f"benchmark_{shard_index:06d}.npz").stat().st_size)})
        shard_index += 1
    wall = perf_counter() - overall_start
    sampled = int(sum(value["sampled"] for value in bin_info.values()))
    if sampled == 0 or any(bin_rates[name] is None for name in DURATION_BINS if bin_info[name]["population"] > 0):
        raise ValueError("Benchmark lacks a duration-bin throughput observation")
    rates = {name: float(bin_rates[name]) if bin_rates[name] is not None else 1e-12 for name in DURATION_BINS}
    projected_classifier_seconds = benchmark_classifier_projection(
        np.vstack(all_embeddings), np.asarray([row[2] for row in all_rows], dtype=np.int64),
        full_rows=EXPECTED_SPLIT["train_rows"] + EXPECTED_SPLIT["validation_rows"],
        artifact_dir=output_dir / "classifier_timing")
    # Include measured full-size prediction-file extrapolation plus a fixed
    # minute for contract/metrics/manifest writes not represented by row files.
    artifact_budget = projected_classifier_seconds["projected_artifact_seconds"] + 60.0
    estimate = project_runtime(
        benchmark_rows=sampled, benchmark_seconds=wall,
        train_rows=EXPECTED_SPLIT["train_rows"], validation_rows=EXPECTED_SPLIT["validation_rows"],
        classifier_seconds=projected_classifier_seconds["projected_seconds"], artifact_seconds=artifact_budget,
        duration_bin_rates=rates,
        duration_bin_shares={name: bin_info[name]["share"] for name in DURATION_BINS},
        startup_seconds=startup_seconds,
    )
    result = {
        "schema": "mission2_stage12_throughput_benchmark_v1", "status": "PASS",
        "config_hash": fingerprint(CONFIG), "data_identity_fingerprint": context.identity["fingerprint"],
        "environment_device": device, "batch_size_requested": batch_size, "amp": bool(use_amp),
        "batch_size_min_observed": min(batch_sizes), "batch_size_max_observed": max(batch_sizes),
        "oom_adjustments": oom_adjustments,
        "benchmark_utterances": sampled, "total_audio_seconds": total_audio_seconds,
        "wall_seconds_including_wav_io_crop_resample_batching_forward_pool_cpu_transfer_and_shard_write": wall,
        "rows_per_second": sampled / wall, "audio_seconds_per_second": total_audio_seconds / wall,
        "real_time_factor": total_audio_seconds / wall,
        "duration_bins": bin_info, "padded_short_rows": int(padded_rows), "skipped_rows": 0,
        "peak_gpu_memory_bytes": None,
        "cache_write_seconds": cache_seconds,
        "cache_write_bytes": cache_bytes,
        "cache_write_mib_per_second": cache_bytes / (1024 ** 2) / cache_seconds if cache_seconds else None,
        "classifier_probe_timing": projected_classifier_seconds,
        "artifact_budget_seconds": artifact_budget,
        "startup_model_load_and_cuda_warmup_seconds": float(startup_seconds),
        "projection": estimate,
        "load_and_cuda_warmup_seconds": None,
        "timing_notes": "Checkpoint load and CUDA warm-up are recorded separately by CP2/CP3 and excluded from steady-state throughput.",
    }
    return result


def benchmark_classifier_projection(embeddings, labels, *, full_rows, artifact_dir):
    """Time the single fixed probe and sample artifact round trip on Training rows."""
    from time import perf_counter
    import joblib
    import pandas as pd
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    matrix = np.asarray(embeddings, dtype=np.float32)
    target = np.asarray(labels, dtype=np.int64)
    if len(matrix) != len(target) or len(matrix) < 2 or set(np.unique(target)) != {0, 1}:
        raise ValueError("Classifier benchmark requires both labels from internal-train rows")
    recipe = CONFIG["classifier"]
    start = perf_counter()
    scaler = StandardScaler(with_mean=True, with_std=True).fit(matrix)
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=recipe["alpha"],
                          learning_rate=recipe["learning_rate"], fit_intercept=recipe["fit_intercept"],
                          average=recipe["average"], random_state=42, max_iter=1, tol=None, shuffle=False)
    transformed = scaler.transform(matrix)
    for epoch in range(recipe["epochs"]):
        order = np.random.default_rng(42 + epoch).permutation(len(target))
        model.partial_fit(transformed[order], target[order], classes=np.array([0, 1]) if epoch == 0 else None)
    probabilities = model.predict_proba(transformed)[:, 1]
    elapsed = perf_counter() - start
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=False)
    classifier_path = output / "timing_probe.joblib"
    artifact_started = perf_counter()
    with classifier_path.open("xb") as stream:
        joblib.dump({"scaler": scaler, "classifier": model, "config": recipe}, stream)
    restored = joblib.load(classifier_path)
    if not np.array_equal(restored["classifier"].predict_proba(
            restored["scaler"].transform(matrix))[:, 1], probabilities):
        raise ValueError("Benchmark classifier artifact reload changed probabilities")
    model_round_trip_seconds = perf_counter() - artifact_started
    prediction_path = output / "timing_predictions.parquet"
    prediction_frame = pd.DataFrame({"y_true": target, "probability": probabilities,
                                     "prediction": (probabilities >= .5).astype(np.int64)})
    prediction_started = perf_counter()
    prediction_frame.to_parquet(prediction_path, index=False)
    restored_frame = pd.read_parquet(prediction_path)
    if not restored_frame.equals(prediction_frame):
        raise ValueError("Benchmark prediction artifact reload differs")
    prediction_round_trip_seconds = perf_counter() - prediction_started
    projected_artifact_seconds = (model_round_trip_seconds
                                  + prediction_round_trip_seconds * full_rows / len(matrix))
    return {"scope": "internal-training benchmark embeddings only", "rows": int(len(matrix)),
            "observed_fit_and_prediction_seconds": elapsed,
            "projected_seconds": elapsed * full_rows / len(matrix),
            "projection_method": "linear row-count extrapolation from this fixed probe; included conservatively in compute gate",
            "model_artifact_save_reload_seconds": model_round_trip_seconds,
            "prediction_artifact_save_reload_seconds": prediction_round_trip_seconds,
            "projected_artifact_seconds": projected_artifact_seconds,
            "artifact_projection_method": "measured probe save/reload plus row-scaled parquet prediction save/reload",
            "classifier_artifact_sha256": file_sha256(classifier_path),
            "prediction_artifact_sha256": file_sha256(prediction_path),
            "scaler_fit_rows": int(len(matrix)), "validation_used": False}


def run_preflight(context, model, *, device="cuda", use_amp=True):
    """Small Training-only contract check; no validation rows or classifier fit."""
    train = context.train_idx
    durations = _duration_seconds(context, train)
    targets = [np.argmin(durations), np.argmin(np.abs(durations - np.median(durations))), np.argmax(durations)]
    positions = train[np.unique(targets)]
    if len(positions) < 2:
        raise ValueError("Preflight needs at least two distinct Training rows")
    waveforms = [crop_and_resample(context, int(i)) for i in positions]
    resample_repeat = crop_and_resample(context, int(positions[0]))
    resample_repeatable = bool(np.array_equal(waveforms[0], resample_repeat))
    if not resample_repeatable:
        raise ValueError("Deterministic resampling repeated with a different result")
    identities = _row_records(context, positions)
    first, first_info = _encode_by_exact_length(model, [waveforms[0]], device=device, use_amp=use_amp, batch_size=1)
    # Duplicating one current utterance checks row independence and batch-size stability
    # without sending any metadata or another utterance into the model.
    duplicate, duplicate_info = _encode_by_exact_length(model, [waveforms[0], waveforms[0]], device=device,
                                                         use_amp=use_amp, batch_size=2)
    batch_delta = float(np.max(np.abs(first[0] - duplicate[0])))
    if batch_delta > 2e-3 or not np.allclose(duplicate[0], duplicate[1], rtol=2e-3, atol=2e-3):
        raise ValueError(f"Batch-size/duplicate waveform embedding drift exceeds tolerance: {batch_delta}")
    all_embeddings, all_info = _encode_by_exact_length(model, waveforms, device=device, use_amp=use_amp, batch_size=2)
    if all_embeddings.shape != (len(positions), CONFIG["representation"]["embedding_dimension"]):
        raise ValueError("Preflight embedding shape mismatch")
    if not np.isfinite(all_embeddings).all():
        raise ValueError("Preflight found NaN/Inf embeddings")
    if identities != _row_records(context, positions) or len(identities["record_key"]) != len(all_embeddings):
        raise ValueError("Preflight row/key/label alignment failed")
    return {"schema": "mission2_stage12_preflight_v1", "status": "PASS",
            "source_split": "internal_train_only", "rows": len(positions),
            "row_identity_alignment": "PASS", "original_sample_rate_hz": 8000,
            "resampled_sample_rate_hz": 16000,
            "resampled_lengths": [int(len(w)) for w in waveforms],
            "resampling_repeatable_bitwise": resample_repeatable,
            "embedding_shape": list(all_embeddings.shape), "embedding_dtype": str(all_embeddings.dtype),
            "valid_hidden_frames": all_info["valid_hidden_frames"],
            "nan_count": int(np.isnan(all_embeddings).sum()), "inf_count": int(np.isinf(all_embeddings).sum()),
            "batch_size_stability_max_abs_delta": batch_delta, "batch_size_stability_tolerance": 2e-3,
            "batch_size_max_observed": max(first_info["batch_size_max"], duplicate_info["batch_size_max"],
                                            all_info["batch_size_max"]),
            "oom_adjustments": first_info["oom_adjustments"] + duplicate_info["oom_adjustments"]
                               + all_info["oom_adjustments"],
            "validation_rows_accessed": 0, "validation_fit_or_gradient_use": False,
            "skipped_rows": 0, "minimum_receptive_field_padding_count": int(sum(len(w) < 400 for w in waveforms)),
            "model_eval": not model.training,
            "all_parameters_frozen": all(not p.requires_grad for p in model.parameters())}


def _manifest_for_split(split, config_hash, data_fingerprint, expected_rows):
    return {"schema": "mission2_stage12_embedding_manifest_v1", "split": split,
            "config_hash": config_hash, "data_identity_fingerprint": data_fingerprint,
            "model_repository": CONFIG["encoder"]["repository"],
            "model_revision": CONFIG["encoder"]["revision"],
            "expected_rows": int(expected_rows), "embedding_dimension": CONFIG["representation"]["embedding_dimension"],
            "shards": []}


def extract_split(context, run_root, split, model, *, config_hash, device="cuda", use_amp=True,
                  batch_size=8, shard_rows=256):
    """Extract fixed-order shards and atomically checkpoint a resume manifest."""
    from time import perf_counter

    if split not in {"train", "validation"}:
        raise ValueError("Only fixed internal train/validation are valid extraction splits")
    indices = context.train_idx if split == "train" else context.validation_idx
    rows_expected = len(indices)
    shard_dir = Path(run_root) / "embeddings" / split
    manifest_path = Path(run_root) / f"{split}_embedding_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        expected_manifest = _manifest_for_split(split, config_hash, context.identity["fingerprint"], rows_expected)
        for field in expected_manifest:
            if manifest.get(field) != expected_manifest[field]:
                raise ValueError(f"Resume manifest differs from current run: {field}")
    else:
        manifest = _manifest_for_split(split, config_hash, context.identity["fingerprint"], rows_expected)
        write_json_new(manifest_path, manifest)
    done = {int(item["shard_index"]): item for item in manifest["shards"]}
    written, reused, padded_rows = 0, 0, 0
    batch_sizes, oom_adjustments = [], []
    started = perf_counter()
    for shard_index, start in enumerate(range(0, rows_expected, shard_rows)):
        local_positions = indices[start:min(rows_expected, start + shard_rows)]
        expected_rows = _row_records(context, local_positions)
        path = shard_dir / f"{split}_{shard_index:06d}.npz"
        if shard_index in done:
            item = done[shard_index]
            result = load_shard(path, expected_meta=item["metadata"], expected_config_hash=config_hash,
                                expected_data_fingerprint=context.identity["fingerprint"],
                                expected_rows=expected_rows)
            if result["row_position"].tolist() != local_positions.tolist():
                raise ValueError("Resumed shard row positions differ")
            reused += len(local_positions)
            continue
        if path.exists():
            # Recover only a fully written shard after verifying its embedded
            # row/config/data/model identity and checksum.
            metadata = (read_json(path.with_suffix(".json")) if path.with_suffix(".json").exists()
                        else recover_shard_without_sidecar(
                            path, expected_rows=expected_rows, expected_positions=local_positions,
                            expected_config_hash=config_hash,
                            expected_data_fingerprint=context.identity["fingerprint"]))
            record = load_shard(path, expected_meta=metadata, expected_config_hash=config_hash,
                                expected_data_fingerprint=context.identity["fingerprint"],
                                expected_rows=expected_rows)
            if record["row_position"].tolist() != local_positions.tolist():
                raise ValueError("Unmanifested completed shard has different row positions")
            item = {"shard_index": shard_index, "row_start": start,
                    "row_end_exclusive": start + len(local_positions),
                    "path": path.relative_to(run_root).as_posix(), "metadata": metadata,
                    "recovered_after_manifest_interruption": True}
            manifest["shards"].append(item)
            _save_manifest(manifest_path, manifest)
            reused += len(local_positions)
            continue
        if path.exists() or path.with_suffix(".json").exists():
            raise ValueError("Incomplete/unverifiable unmanifested shard exists; refusing to overwrite it")
        waves = [crop_and_resample(context, int(position)) for position in local_positions]
        embeddings, batch_info = _encode_by_exact_length(model, waves, device=device, use_amp=use_amp,
                                                         batch_size=batch_size)
        padded_rows += batch_info["padded_rows"]
        batch_sizes.extend([batch_info["batch_size_min"], batch_info["batch_size_max"]])
        oom_adjustments.extend(batch_info["oom_adjustments"])
        saved_path, metadata = save_shard(shard_dir, split, shard_index, expected_rows, embeddings,
                                          config_hash=config_hash,
                                          data_fingerprint=context.identity["fingerprint"],
                                          row_positions=local_positions)
        item = {"shard_index": shard_index, "row_start": start,
                "row_end_exclusive": start + len(local_positions),
                "path": saved_path.relative_to(run_root).as_posix(), "metadata": metadata}
        manifest["shards"].append(item)
        _save_manifest(manifest_path, manifest)
        written += len(local_positions)
    total = written + reused
    if total != rows_expected:
        raise ValueError("Embedding extraction row accounting mismatch")
    expected_keys = context.source.iloc[indices].record_key.astype(str).tolist()
    seen_keys, seen_positions = [], []
    for item in manifest["shards"]:
        shard_path = Path(run_root) / item["path"]
        record = load_shard(shard_path, expected_meta=item["metadata"], expected_config_hash=config_hash,
                             expected_data_fingerprint=context.identity["fingerprint"])
        seen_keys.extend(record["record_key"].astype(str).tolist())
        seen_positions.extend(record["row_position"].astype(np.int64).tolist())
    if len(seen_keys) != rows_expected or len(set(seen_keys)) != rows_expected or seen_keys != expected_keys:
        raise ValueError("Embedding split has duplicate, missing, or misordered rows")
    if seen_positions != indices.tolist():
        raise ValueError("Embedding split row-position identity mismatch")
    return {"status": "PASS", "split": split, "expected_rows": rows_expected,
            "written_rows": written, "resumed_rows": reused, "skipped_rows": 0,
            "duplicate_rows": 0, "missing_rows": 0,
            "embedding_dimension": CONFIG["representation"]["embedding_dimension"],
            "config_hash": config_hash, "data_identity_fingerprint": context.identity["fingerprint"],
            "wall_seconds": perf_counter() - started,
            "minimum_receptive_field_padded_rows": padded_rows,
            "batch_size_min_observed": min(batch_sizes) if batch_sizes else None,
            "batch_size_max_observed": max(batch_sizes) if batch_sizes else None,
            "oom_adjustments": oom_adjustments,
            "manifest_path": str(manifest_path)}


def iter_embeddings(run_root, split, *, config_hash, data_fingerprint):
    manifest = read_json(Path(run_root) / f"{split}_embedding_manifest.json")
    for item in manifest["shards"]:
        result = load_shard(Path(run_root) / item["path"], expected_meta=item["metadata"],
                            expected_config_hash=config_hash,
                            expected_data_fingerprint=data_fingerprint)
        yield result


def fit_probe(run_root, context, *, config_hash):
    """Fit scaler and one logistic linear probe on train embeddings only."""
    from time import perf_counter
    import joblib
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    started = perf_counter()
    identity = context.identity["fingerprint"]
    scaler = StandardScaler(with_mean=True, with_std=True)
    train_manifest = read_json(Path(run_root) / "train_embedding_manifest.json")
    if train_manifest["expected_rows"] != EXPECTED_SPLIT["train_rows"]:
        raise ValueError("Training embedding row count differs from frozen contract")
    for shard in iter_embeddings(run_root, "train", config_hash=config_hash, data_fingerprint=identity):
        scaler.partial_fit(shard["embedding"])
    recipe = CONFIG["classifier"]
    classifier = SGDClassifier(loss="log_loss", penalty="l2", alpha=recipe["alpha"],
                               learning_rate=recipe["learning_rate"], fit_intercept=recipe["fit_intercept"],
                               average=recipe["average"], random_state=42, max_iter=1, tol=None, shuffle=False)
    for epoch in range(recipe["epochs"]):
        rng = np.random.default_rng(42 + epoch)
        for shard_index, shard in enumerate(iter_embeddings(run_root, "train", config_hash=config_hash,
                                                             data_fingerprint=identity)):
            order = rng.permutation(len(shard["y_true"]))
            x = scaler.transform(shard["embedding"][order])
            y = shard["y_true"][order].astype(np.int64)
            classifier.partial_fit(x, y, classes=np.array([0, 1]) if epoch == 0 and shard_index == 0 else None)
    artifact = Path(run_root) / "classifier.joblib"
    if artifact.exists():
        raise FileExistsError(f"Classifier artifact already exists: {artifact}")
    temp_artifact = artifact.with_name(artifact.name + f".{uuid.uuid4().hex}.partial")
    try:
        with temp_artifact.open("xb") as stream:
            joblib.dump({"scaler": scaler, "classifier": classifier,
                         "config": CONFIG["classifier"], "config_hash": config_hash,
                         "data_identity_fingerprint": identity,
                         "model_revision": CONFIG["encoder"]["revision"]}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if artifact.exists():
            raise FileExistsError(f"Classifier artifact already exists: {artifact}")
        os.replace(temp_artifact, artifact)
    finally:
        if temp_artifact.exists():
            temp_artifact.unlink()
    return scaler, classifier, {"status": "PASS", "scaler_fit_split": "internal_train_only",
                                "scaler_fit_rows": EXPECTED_SPLIT["train_rows"],
                                "classifier_fit_split": "internal_train_only",
                                "classifier_fit_rows": EXPECTED_SPLIT["train_rows"],
                                "validation_fit_rows": 0,
                                "classifier": CONFIG["classifier"],
                                "fit_seconds": perf_counter() - started,
                                "artifact": str(artifact), "artifact_sha256": file_sha256(artifact)}


def _predict_split(run_root, split, scaler, classifier, context, *, config_hash):
    pieces = []
    for shard in iter_embeddings(run_root, split, config_hash=config_hash,
                                 data_fingerprint=context.identity["fingerprint"]):
        probabilities = classifier.predict_proba(scaler.transform(shard["embedding"]))[:, 1]
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Classifier emitted invalid probabilities")
        pieces.append({"record_key": shard["record_key"].astype(str),
                       "call_id": shard["call_id"].astype(str), "y_true": shard["y_true"].astype(np.int64),
                       "probability": probabilities.astype(np.float64)})
    if not pieces:
        raise ValueError("No embeddings found for prediction")
    return {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}


def evaluate_predictions(expected_rows, probabilities):
    rows = {field: [str(item[field]) if field != "y_true" else int(item[field]) for item in expected_rows]
            for field in ("record_key", "call_id", "y_true")}
    keys, calls, truth = np.asarray(rows["record_key"]), np.asarray(rows["call_id"]), np.asarray(rows["y_true"], dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    if (len(keys) == 0 or len(set(keys.tolist())) != len(keys) or probability.shape != truth.shape
            or not np.isin(truth, [0, 1]).all() or not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()):
        raise ValueError("Invalid/duplicate prediction rows or probabilities")
    prediction = (probability >= .5).astype(np.int64)
    cm = [[int(((truth == actual) & (prediction == predicted)).sum()) for predicted in (0, 1)]
          for actual in (0, 1)]
    recall0 = cm[0][0] / (cm[0][0] + cm[0][1]) if sum(cm[0]) else None
    recall1 = cm[1][1] / (cm[1][0] + cm[1][1]) if sum(cm[1]) else None
    correct = prediction == truth
    confidence = np.maximum(probability, 1 - probability)
    hce = (~correct) & (confidence >= .90)
    metrics = {"rows": int(len(truth)), "accuracy": float(correct.mean()), "confusion_matrix": cm,
               "recall_0": recall0, "recall_1": recall1, "errors": int((~correct).sum()),
               "fp": int(((truth == 0) & (prediction == 1)).sum()),
               "fn": int(((truth == 1) & (prediction == 0)).sum()),
               "hce": int(hce.sum()),
               "hce_fp": int((hce & (truth == 0)).sum()),
               "hce_fn": int((hce & (truth == 1)).sum()),
               "hce_confidence_ge_0_90": int(hce.sum()),
               "mean_confidence": float(confidence.mean()), "threshold": 0.5}
    output = {**rows, "probability": probability.tolist(), "prediction": prediction.tolist(),
              "confidence": confidence.tolist(), "correct": correct.tolist()}
    return metrics, output


def duration_slice_report(expected_rows, prediction_rows, duration_seconds):
    """Validation-only diagnostics; duration is never passed into the model/probe."""
    expected = list(expected_rows)
    aligned = align_predictions(prediction_rows, expected)
    durations = np.asarray(duration_seconds, dtype=np.float64)
    if durations.shape != (len(expected),) or not np.isfinite(durations).all() or (durations <= 0).any():
        return {"status": "tracking_unavailable", "reason": "Invalid or missing aligned duration metadata",
                "overlap_tracking": "not_attempted"}
    probabilities = np.asarray([row["probability"] for row in aligned], dtype=np.float64)
    bins = [("duration_<0.5", durations < .5),
            ("duration_0.5-1", (durations >= .5) & (durations < 1)),
            ("duration_1-2", (durations >= 1) & (durations < 2)),
            ("duration_2-5", (durations >= 2) & (durations < 5)),
            ("duration_>=5", durations >= 5)]
    slices = {}
    for name, mask in bins:
        if not mask.any():
            slices[name] = {"sample_count": 0, "accuracy": None}
            continue
        metrics, _ = evaluate_predictions([expected[i] for i in np.flatnonzero(mask)], probabilities[mask])
        slices[name] = {"sample_count": int(mask.sum()), "accuracy": metrics["accuracy"],
                        "errors": metrics["errors"], "hce": metrics["hce"]}
    return {"status": "PASS", "source": "same-row manifest startAt/endAt metadata; diagnostic only",
            "overlap_tracking": "not_attempted", "slices": slices}


def align_predictions(prediction_rows, expected_rows):
    """Exact row-set alignment with call/label checks; never use a dropping join."""
    expected = list(expected_rows)
    if isinstance(prediction_rows, dict):
        fields = ("record_key", "call_id", "y_true")
        if any(field not in prediction_rows for field in fields):
            raise ValueError("Prediction identity columns are incomplete")
        all_fields = tuple(prediction_rows)
        predicted = [dict(zip(all_fields, values))
                     for values in zip(*(prediction_rows[field] for field in all_fields))]
    else:
        predicted = list(prediction_rows)
    expected_keys = [str(row["record_key"]) for row in expected]
    predicted_keys = [str(row["record_key"]) for row in predicted]
    if (not expected_keys or len(set(expected_keys)) != len(expected_keys)
            or len(set(predicted_keys)) != len(predicted_keys) or set(expected_keys) != set(predicted_keys)):
        raise ValueError("Prediction row set has duplicate, missing, or extra keys")
    by_key = {str(row["record_key"]): row for row in predicted}
    aligned = [by_key[key] for key in expected_keys]
    for reference, candidate in zip(expected, aligned):
        if str(reference["call_id"]) != str(candidate["call_id"]):
            raise ValueError("Prediction call_id differs from expected row identity")
        if int(reference["y_true"]) != int(candidate["y_true"]):
            raise ValueError("Prediction label differs from expected row identity")
    return aligned


def save_predictions(path, rows):
    import pandas as pd
    target = Path(path)
    frame = pd.DataFrame(rows)
    if target.exists():
        existing = pd.read_parquet(target)
        if not existing.equals(frame):
            raise ValueError(f"Existing prediction artifact differs; refusing to overwrite: {target}")
        return {"path": str(target), "sha256": file_sha256(target), "rows": int(len(frame)),
                "reused_exact_existing_artifact": True}
    temp = target.with_name(target.name + f".{uuid.uuid4().hex}.partial")
    try:
        frame.to_parquet(temp, index=False)
        if target.exists():
            raise FileExistsError(target)
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()
    return {"path": str(target), "sha256": file_sha256(target), "rows": int(len(frame))}


def evaluate_and_save(run_root, context, scaler, classifier, *, config_hash):
    from time import perf_counter
    result = {}
    predictions = {}
    durations = {}
    for split, indices in (("train", context.train_idx), ("validation", context.validation_idx)):
        started = perf_counter()
        predicted = _predict_split(run_root, split, scaler, classifier, context, config_hash=config_hash)
        expected = _row_records(context, indices)
        if predicted["record_key"].tolist() != expected["record_key"] or predicted["call_id"].tolist() != expected["call_id"] or predicted["y_true"].tolist() != expected["y_true"]:
            raise ValueError(f"{split} prediction row/key/call/label alignment failed")
        metrics, output = evaluate_predictions(
            [{"record_key": k, "call_id": c, "y_true": y} for k, c, y in zip(
                predicted["record_key"], predicted["call_id"], predicted["y_true"])], predicted["probability"])
        prediction_path = Path(run_root) / f"{split}_predictions.parquet"
        artifact = save_predictions(prediction_path, output)
        result[split] = metrics
        result[f"{split}_prediction_artifact"] = artifact
        predictions[split] = output
        durations[split] = perf_counter() - started
    result["train_minus_validation_accuracy_gap"] = result["train"]["accuracy"] - result["validation"]["accuracy"]
    validation_source = context.source.iloc[context.validation_idx]
    duration_seconds = ((validation_source.endAt_ms.to_numpy(dtype=np.float64)
                         - validation_source.startAt_ms.to_numpy(dtype=np.float64)) / 1000.0)
    expected_validation = _row_records(context, context.validation_idx)
    validation_tracking_rows = [{"record_key": key, "call_id": call, "y_true": label}
                                for key, call, label in zip(expected_validation["record_key"],
                                                           expected_validation["call_id"],
                                                           expected_validation["y_true"])]
    result["validation_duration_slices"] = duration_slice_report(
        validation_tracking_rows, predictions["validation"], duration_seconds)
    result["prediction_seconds"] = durations
    result["config_hash"] = config_hash
    result["data_identity_fingerprint"] = context.identity["fingerprint"]
    result["official_validation_used"] = False
    result["validation_gradient_update"] = False
    result["validation_scaler_fit_rows"] = 0
    write_json_new(Path(run_root) / "metrics.json", result)
    return result, predictions


def _validate_prediction_frame(frame, expected):
    frame = frame.copy()
    if "probability" not in frame and "probability_speaker1" in frame:
        frame = frame.rename(columns={"probability_speaker1": "probability"})
    required = {"record_key", "call_id", "y_true", "probability", "prediction"}
    if not required.issubset(frame.columns) or frame.record_key.isna().any():
        raise ValueError("Reference prediction schema/key identity is invalid")
    frame["record_key"] = frame.record_key.astype(str)
    if frame.record_key.duplicated().any() or set(frame.record_key) != set(expected["record_key"]):
        raise ValueError("Reference row set differs from Stage12 validation")
    aligned = frame.set_index("record_key").loc[expected["record_key"]].reset_index()
    for field in ("call_id", "y_true"):
        if aligned[field].astype(str if field == "call_id" else int).tolist() != expected[field]:
            raise ValueError(f"Reference row identity differs: {field}")
    p = aligned.probability.to_numpy(dtype=np.float64)
    pred = (p >= .5).astype(int)
    if not np.isfinite(p).all() or not np.array_equal(pred, aligned.prediction.to_numpy(dtype=int)):
        raise ValueError("Reference probability/threshold contract is invalid")
    return pred


def load_reference_predictions(repo_root, context, candidate_validation_predictions):
    import pandas as pd
    from .mission2_time_frequency_screen import REFERENCE_ACCURACY as EXPECTED_ACCURACY

    root = Path(repo_root) / REFERENCE_RUN
    contract = read_json(root / "contract.json")
    prep = contract["preprocessing"]
    if (prep.get("provenance_fingerprint") != CLEAN_59D_FINGERPRINT or prep.get("seed") != 42
            or prep.get("threshold") != 0.5 or prep.get("official_validation_used") is not False
            or any(prep.get("split", {}).get(key) != value for key, value in EXPECTED_SPLIT.items())):
        raise ValueError("Frozen 59D reference contract differs from Stage12")
    expected = _row_records(context, context.validation_idx)
    references = {}
    for name, accuracy in EXPECTED_ACCURACY.items():
        path = root / name / "predictions.parquet"
        frame = pd.read_parquet(path)
        prediction = _validate_prediction_frame(frame, expected)
        if not np.isclose(float((prediction == np.asarray(expected["y_true"])).mean()), accuracy, atol=1e-15):
            raise ValueError(f"Frozen reference accuracy differs: {name}")
        references[name] = {"frame": frame, "prediction": prediction, "sha256": file_sha256(path)}
    common_wrong = ((references["tabm"]["prediction"] != np.asarray(expected["y_true"]))
                    & (references["tabular_resnet"]["prediction"] != np.asarray(expected["y_true"])))
    if int(common_wrong.sum()) != REFERENCE_COMMON_ERRORS:
        raise ValueError("Frozen TabM/ResNet common-error count differs")
    candidate_frame = pd.DataFrame(candidate_validation_predictions)
    candidate_prediction = _validate_prediction_frame(candidate_frame, expected)
    paired = {"rows": len(expected["record_key"]), "references": {}}
    for name, reference in references.items():
        ref_ok = reference["prediction"] == np.asarray(expected["y_true"])
        cand_ok = candidate_prediction == np.asarray(expected["y_true"])
        paired["references"][name] = {
            "both_correct": int((ref_ok & cand_ok).sum()),
            "both_wrong": int((~ref_ok & ~cand_ok).sum()),
            "reference_only_correct": int((ref_ok & ~cand_ok).sum()),
            "stage12_only_correct": int((~ref_ok & cand_ok).sum()),
            "net_stage12_corrections": int((~ref_ok & cand_ok).sum() - (ref_ok & ~cand_ok).sum()),
            "reference_predictions_sha256": reference["sha256"],
        }
        categories = paired["references"][name]
        if sum(categories[key] for key in ("both_correct", "both_wrong", "reference_only_correct", "stage12_only_correct")) != len(expected["record_key"]):
            raise ValueError(f"Paired comparison categories do not sum to validation row count: {name}")
    if references["tabm"]["sha256"] == "" or references["tabular_resnet"]["sha256"] == "":
        raise ValueError("Reference prediction files are not verifiable")
    return paired


def reload_verify(run_root, context, *, config_hash, original_predictions):
    import joblib
    artifact = Path(run_root) / "classifier.joblib"
    loaded = joblib.load(artifact)
    if (loaded["config_hash"] != config_hash or loaded["config"] != CONFIG["classifier"]
            or loaded["data_identity_fingerprint"] != context.identity["fingerprint"]
            or loaded["model_revision"] != CONFIG["encoder"]["revision"]):
        raise ValueError("Reloaded classifier/config/data/model identity differs")
    result = {"schema": "mission2_stage12_reload_report_v1", "status": "PASS",
              "classifier_sha256": file_sha256(artifact), "config_hash": config_hash,
              "data_identity_fingerprint": context.identity["fingerprint"], "splits": {}}
    for split in ("train", "validation"):
        current = _predict_split(run_root, split, loaded["scaler"], loaded["classifier"], context,
                                 config_hash=config_hash)
        probability = np.asarray(original_predictions[split]["probability"], dtype=np.float64)
        expected_keys = np.asarray(original_predictions[split]["record_key"], dtype=str)
        if not np.array_equal(current["record_key"].astype(str), expected_keys):
            raise ValueError(f"Reload {split} row/key identity mismatch")
        if not np.array_equal(current["call_id"].astype(str),
                              np.asarray(original_predictions[split]["call_id"], dtype=str)):
            raise ValueError(f"Reload {split} call_id identity mismatch")
        if not np.array_equal(current["y_true"], np.asarray(original_predictions[split]["y_true"])):
            raise ValueError(f"Reload {split} label identity mismatch")
        if not np.allclose(current["probability"], probability, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Reload {split} probabilities changed")
        prediction = (current["probability"] >= .5).astype(int)
        if not np.array_equal(prediction, np.asarray(original_predictions[split]["prediction"])):
            raise ValueError(f"Reload {split} predictions changed")
        result["splits"][split] = {"rows": len(probability), "row_key_label": "PASS",
                                   "probability_max_abs_delta": float(np.max(np.abs(current["probability"] - probability))),
                                   "prediction_identical": True, "accuracy_identical": True}
    write_json_new(Path(run_root) / "reload_report.json", result)
    return result


def append_result_to_log(repo_root, run_root, status, result):
    """Append one verified Stage 12 outcome, refusing duplicates or replacement."""
    log_path = Path(repo_root) / "docs" / "mission2_experiment_log.md"
    marker = "### Stage 12 result — Raw-waveform Pretrained Speech"
    content = log_path.read_text(encoding="utf-8")
    if marker in content:
        if f"**Run artifacts:** `{Path(run_root).as_posix()}`" in content:
            return "already_present"
        raise FileExistsError("A different Stage 12 result already exists in the experiment log")
    section = ["", marker, "", f"**Status:** `{status}`", "",
               f"**Run artifacts:** `{Path(run_root).as_posix()}`", ""]
    if status == "RESOURCE/TIME-BUDGET INFEASIBLE":
        section += ["**Performance:** `NOT EVALUATED`", "",
                    f"- Candidate: `{CONFIG['encoder']['repository']}@{CONFIG['encoder']['revision']}`",
                    f"- Benchmark environment: `{result['environment']}`",
                    f"- Measured throughput: `{result['throughput']['rows_per_second']:.6f}` rows/s; "
                    f"`{result['throughput']['audio_seconds_per_second']:.6f}` audio-s/s",
                    f"- Projected full runtime: `{result['compute_gate']['projected_full_hours']:.3f}` hours",
                    "- Validation metrics and Stage 12 performance were not evaluated.", ""]
    else:
        metrics = result["metrics"]
        train, validation = metrics["train"], metrics["validation"]
        section += [f"- Encoder: `{CONFIG['encoder']['repository']}@{CONFIG['encoder']['revision']}`; "
                    f"last hidden-state valid-frame masked mean; 16 kHz; 8→16 kHz sinc resampling.",
                    f"- Classifier: `SGDClassifier(loss=log_loss, penalty=l2, alpha=0.0001, epochs=5, seed=42)`; "
                    "train-only incremental `StandardScaler`; threshold `0.5`.",
                    f"- Data/split: clean 59D fingerprint `{CLEAN_59D_FINGERPRINT}`; "
                    "711,467/177,891 rows; 23,360/5,840 calls; overlap `0`; seed `42`.",
                    f"- Internal-train Accuracy: `{train['accuracy']:.10f}`; errors `{train['errors']}`; "
                    f"CM `{train['confusion_matrix']}`.",
                    f"- Internal-validation Accuracy: `{validation['accuracy']:.10f}`; "
                    f"Train−Validation gap: `{metrics['train_minus_validation_accuracy_gap']:.10f}`.",
                    f"- Validation Recall speaker 0/1: `{validation['recall_0']:.10f}` / `{validation['recall_1']:.10f}`; "
                    f"CM `{validation['confusion_matrix']}`; FP/FN `{validation['fp']}`/`{validation['fn']}`; "
                    f"errors `{validation['errors']}`; HCE≥0.90 `{validation['hce']}`.",
                    f"- Duration slices: `{metrics['validation_duration_slices']['status']}`; "
                    "overlap slices not evaluated.",
                    f"- Paired comparison: `{result['paired_comparison']}`.",
                    f"- Compute gate projected hours `{result['compute_gate']['projected_full_hours']:.3f}`; "
                    f"measured extraction `{result['throughput']['rows_per_second']:.6f}` rows/s.",
                    f"- Reload verification: `{result['reload']['status']}`; "
                    f"classifier SHA-256 `{result['reload']['classifier_sha256']}`.", ""]
    with log_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(section))
    return "appended"
