"""Evaluate the fixed M1-XVEC-001 head on official Validation; never fit.

Feature chunks resume only when data, head and encoder identities match.
Gender is read for scoring after feature extraction, never used as input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--from-lab", action="store_true", help="Use existing Drive M1-XVEC-001 artifacts and unpack Validation locally")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.worker_index < args.workers:
        parser.error("worker-index must be between 0 and workers-1")
    if args.from_lab:
        import importlib.util
        from dcc_data.colab_env import find_data_root
        spec = importlib.util.spec_from_file_location("m1_xvec_training", ROOT / "scripts/lab/m1-xvec-001.py")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        run_root = Path(os.environ["DCC_RUN_ROOT"])
        args.head = run_root / "lab/M1-XVEC-001/best.ckpt"
        args.model_dir = run_root / "models/wavlm-base-plus-sv"
        work = Path(os.environ["LAB_WORK"]) / "Validation"
        work.mkdir(parents=True, exist_ok=True)
        source = find_data_root() / "Validation"
        helper.unpack(source / "VL_서울_구급.zip", work)
        helper.unpack(source / "VS_서울_구급.zip", work)
        args.data_root = work.parent
    if any(value is None for value in (args.data_root, args.model_dir, args.head)):
        parser.error("provide data-root, model-dir and head, or --from-lab")
    import joblib
    import librosa
    import numpy as np
    import pandas as pd
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    from dcc_data.manifest import ManifestRow
    from dcc_data.mission1 import extract_caller_segments, concatenate_caller_waveforms
    from dcc_data.mission1_train import encode_gender, compute_metrics
    from dcc_data.mission2_features import acoustic_feature_vector

    torch.set_num_threads(args.threads)
    root = args.data_root.resolve()
    validation = root / "Validation"
    def index(suffix):
        found = {}
        for path in sorted(validation.rglob("*" + suffix)):
            if path.stem in found:
                raise ValueError(f"duplicate {suffix} stem: {path.stem}")
            found[path.stem] = path
        return found
    wavs, jsons = index(".wav"), index(".json")
    if set(wavs) != set(jsons) or len(jsons) != 3640:
        raise ValueError(f"expected 3640 paired Validation calls, got {len(wavs)}/{len(jsons)}")
    names = sorted(jsons)
    rows = [ManifestRow("Validation", name, str(wavs[name]), str(jsons[name])) for name in names]
    args.out.mkdir(parents=True, exist_ok=True)
    identity = dict(exp_id="M1-XVEC-001", split="official_validation", rows=3640,
                    head_sha256=sha256(args.head), data_root=str(root),
                    model_revision=json.loads((args.model_dir / "dcc_revision.json").read_text()),
                    encoder_sha256=sha256(args.model_dir / "model.safetensors"),
                    input_sha256=hashlib.sha256("".join(
                        name + sha256(jsons[name]) + sha256(wavs[name]) for name in names
                    ).encode()).hexdigest(), max_seconds=20, min_seconds=.5)
    if identity["head_sha256"] != "4fffb8aa97c2f3083b05820ca45b84ecddbc60c5162e02961546f37d133dd9d5":
        raise ValueError("head does not match the fixed M1-XVEC-001 candidate")
    if identity["encoder_sha256"] != "94c3defe08248d81c7b2bd0a058ea9985269cefed13076434669c47fade41182":
        raise ValueError("encoder does not match the fixed WavLM snapshot")
    if identity["model_revision"] != {"repo_id": "microsoft/wavlm-base-plus-sv", "revision": "1bfd64eca136543feb28c5ffaf05381c6af33121"}:
        raise ValueError("WavLM revision mismatch")
    identity_path = args.out / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("cached feature identity mismatch")
    identity_temp = args.out / f"identity.{os.getpid()}.tmp"
    identity_temp.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
    identity_temp.replace(identity_path)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_dir, local_files_only=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    encoder = WavLMForXVector.from_pretrained(args.model_dir, local_files_only=True).to(device).eval()
    head = joblib.load(args.head)
    if head.n_features_in_ != 564:
        raise ValueError("expected the fixed 564D combined head")
    started = time.perf_counter()
    chunks = []
    for start in range(0, len(rows), 40):
        if (start // 40) % args.workers != args.worker_index:
            continue
        stop = min(start + 40, len(rows))
        path = args.out / f"features_{start:04d}_{stop:04d}.npy"
        if path.exists():
            features = np.load(path, allow_pickle=False)
        else:
            values = []
            for row in rows[start:stop]:
                segments = extract_caller_segments(row, root)
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
                    xvector = torch.nn.functional.normalize(encoder(**{k: v.to(device) for k, v in inputs.items()}).embeddings, dim=-1)[0].cpu().numpy()
                values.append(np.concatenate([acoustic, xvector]))
            features = np.asarray(values)
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, features)
            temporary.replace(path)
        if features.shape != (stop - start, 564) or not np.isfinite(features).all():
            raise ValueError(f"invalid feature chunk: {path}")
        chunks.append(features)
        print(f"Validation {stop}/3640, elapsed {time.perf_counter()-started:.1f}s", flush=True)
    all_paths = [args.out / f"features_{start:04d}_{min(start+40,len(rows)):04d}.npy"
                 for start in range(0, len(rows), 40)]
    if not all(path.exists() for path in all_paths):
        print(f"worker {args.worker_index} finished; other chunks still running", flush=True)
        return
    chunks = [np.load(path, allow_pickle=False) for path in all_paths]
    prediction = head.predict(np.concatenate(chunks))
    truth = encode_gender([json.loads(jsons[name].read_text(encoding="utf-8-sig"))["gender"] for name in names])
    frame = pd.DataFrame({"audio file name": [wavs[name].name for name in names],
                          "gender": ["F" if value == 0 else "M" for value in prediction]})
    csv_temp = args.out / f"mission1.{os.getpid()}.tmp"
    frame.to_csv(csv_temp, index=False, encoding="utf-8")
    csv_temp.replace(args.out / "mission1.csv")
    scores = compute_metrics(truth, prediction)
    result = dict(exp_id="M1-XVEC-001-OFFVAL", primary_metric=dict(name="accuracy", value=scores["accuracy"], split="official_validation"),
                  identity=identity, metrics=scores, seed=42, eval_rows=3640, official_val=True,
                  elapsed_seconds=time.perf_counter()-started, threads=args.threads,
                  fitting_performed=False, inference_device=device)
    metrics_temp = args.out / f"metrics.{os.getpid()}.tmp"
    metrics_temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_temp.replace(args.out / "metrics.json")
    print(json.dumps(result["metrics"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
