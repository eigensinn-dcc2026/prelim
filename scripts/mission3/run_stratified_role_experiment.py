"""Mission 3의 90/10 선택 실험과 Training 100% 최종 재학습을 실행한다.

실험 순서
1. Training을 multilabel stratified 90/10으로 고정 분할
2. 오심을 제외하고 전신쇠약·호흡곤란 hard-negative 채굴
3. 모든 원본 샘플을 매 epoch 1회 보장하고 hard-negative만 1.5배 노출
4. 기존 입력, 화자 역할, 담화·증상 동의어·증거 상태 통합 입력을 같은 split에서 비교
5. 고정 0.5의 내부 검증 Macro F1로 best epoch/모델 선택
6. 허용 시 best checkpoint에서만 증상별 임계값 선택
7. 선택된 epoch 수와 입력 방식으로 Training 100% 재학습

공식 Validation은 이 스크립트에서 읽지 않는다.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from config import CHUNK_STRATEGY, TARGET_SYMPTOMS, get_train_label_dir
from conversation_features import feature_versions
from data_utils import list_label_files, load_label, symptoms_to_multihot
from experiment_utils import (
    GuaranteedFullPassSampler,
    iterative_multilabel_split,
    sampler_config,
)
from hard_negative_utils import load_hard_negative_sampling
from inference import load_model
from loss_utils import compute_pos_weights
from metrics_utils import compute_multilabel_metrics
from mine_hard_negatives import mine_candidates
from model_mission3 import SymptomClassifier, pool_chunk_logits
from threshold_utils import tune_thresholds
from train import Mission3Dataset, collate_conversations, set_seed, train_epoch


DEFAULT_FOCUS_SYMPTOMS = ["전신쇠약", "호흡곤란"]
VARIANT_FEATURE_MODES = {
    "current_input": "none",
    "speaker_role_tokens": "speaker",
    "feature_annotation_v1": "all",
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f"{path.name}.pending")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(pending, path)


def atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f"{path.name}.pending")
    torch.save(payload, pending)
    os.replace(pending, path)


def label_matrix(label_files: list[Path]) -> np.ndarray:
    return np.asarray(
        [symptoms_to_multihot(load_label(path).get("symptom", [])) for path in label_files],
        dtype=np.int8,
    )


def split_training_files(
    label_files: list[Path], val_ratio: float, seed: int
) -> tuple[list[Path], list[Path], dict]:
    y = label_matrix(label_files)
    train_files, val_files = iterative_multilabel_split(
        label_files, y, val_ratio=val_ratio, seed=seed,
    )
    index = {path: i for i, path in enumerate(label_files)}
    train_y = y[[index[path] for path in train_files]]
    val_y = y[[index[path] for path in val_files]]
    details = {
        "method": "greedy_iterative_multilabel_stratification",
        "seed": seed,
        "val_ratio": val_ratio,
        "num_total": len(label_files),
        "num_train": len(train_files),
        "num_validation": len(val_files),
        "label_counts_total": dict(zip(TARGET_SYMPTOMS, y.sum(axis=0).astype(int).tolist())),
        "label_counts_train": dict(zip(TARGET_SYMPTOMS, train_y.sum(axis=0).astype(int).tolist())),
        "label_counts_validation": dict(zip(TARGET_SYMPTOMS, val_y.sum(axis=0).astype(int).tolist())),
        "train_files": [path.name for path in train_files],
        "validation_files": [path.name for path in val_files],
    }
    return train_files, val_files, details


def make_loader(
    dataset,
    batch_size: int,
    *,
    shuffle: bool = False,
    sampler=None,
    seed: int = 42,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    use_cuda = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        generator=generator,
        num_workers=0,
        pin_memory=use_cuda,
        collate_fn=collate_conversations,
    )


@torch.no_grad()
def collect_scores(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all, scores_all = [], []
    for batch in tqdm(loader, desc="internal validation", leave=False):
        labels = batch["labels"].numpy().astype(np.int8)
        chunk_logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        logits = pool_chunk_logits(
            chunk_logits,
            batch["conversation_ids"].to(device),
            len(labels),
        )
        labels_all.append(labels)
        scores_all.append(torch.sigmoid(logits).cpu().numpy())
    if not labels_all:
        raise ValueError("내부 검증 데이터가 없습니다.")
    return np.vstack(labels_all), np.vstack(scores_all)


def fixed_threshold_metrics(labels, scores) -> dict:
    return compute_multilabel_metrics(
        labels,
        scores > 0.5,
        TARGET_SYMPTOMS,
    )


def checkpoint_payload(
    model,
    *,
    args,
    epoch: int,
    best_f1: float,
    feature_mode: str,
    loss_config: dict,
    sampling: dict,
    split_details: dict | None,
    decision_thresholds=None,
    threshold_selection=None,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "mission": 3,
        "epoch": epoch,
        "best_f1": best_f1,
        "selection_metric": "internal_validation_macro_f1_at_fixed_0.5",
        "model_name": args.model_name,
        "max_length": args.max_length,
        "chunk_overlap": args.chunk_overlap,
        "chunk_pooling": "max_logits",
        "chunk_strategy": CHUNK_STRATEGY,
        "speaker_role_tokens": feature_mode == "speaker",
        "feature_mode": feature_mode,
        "feature_versions": feature_versions(),
        "speaker_role_markers": {"0": "[상담원]", "1": "[신고자]"},
        "loss_config": loss_config,
        "sampling_config": sampling,
        "split_config": split_details,
        "decision_thresholds": decision_thresholds or [0.5] * len(TARGET_SYMPTOMS),
        "threshold_selection": threshold_selection,
        "target_symptoms": list(TARGET_SYMPTOMS),
    }


def write_hard_negative_manifest(
    label_files: list[Path],
    checkpoint_path: Path,
    output: Path,
    args,
    split_name: str,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, checkpoint = load_model(checkpoint_path, device)
    max_length = int(checkpoint.get("max_length", args.max_length))
    chunk_overlap = int(checkpoint.get("chunk_overlap", args.chunk_overlap))
    role_tokens = bool(checkpoint.get("speaker_role_tokens", False))
    feature_mode = checkpoint.get("feature_mode")
    dataset = Mission3Dataset(
        label_files,
        tokenizer,
        max_length=max_length,
        chunk_overlap=chunk_overlap,
        speaker_role_tokens=role_tokens,
        feature_mode=feature_mode,
    )
    loader = make_loader(dataset, args.mine_batch_size, seed=args.seed)
    file_map = {path.name: path for path in label_files}
    rows, details = mine_candidates(
        model,
        loader,
        file_map,
        device,
        list(args.hard_negative_symptoms),
        args.hard_negative_threshold,
        require_no_clear_evidence=True,
    )
    rows.sort(key=lambda row: (row["symptom"], -row["probability"], row["file_name"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "file_name", "symptom", "truth", "prediction", "probability",
        "threshold", "error_type", "evidence_bucket",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(output, index=False, encoding="utf-8-sig")
    counts = Counter(row["symptom"] for row in rows)
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "manifest": str(output.resolve()),
        "split_name": split_name,
        "num_input_files": len(label_files),
        "focus_symptoms": list(args.hard_negative_symptoms),
        "nausea_excluded": "오심" not in args.hard_negative_symptoms,
        "hard_negative_pairs_by_symptom": {
            symptom: int(counts.get(symptom, 0)) for symptom in args.hard_negative_symptoms
        },
        **details,
    }
    atomic_json(output.with_suffix(".summary.json"), summary)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def train_variant(
    variant_name: str,
    feature_mode: str,
    train_files: list[Path],
    val_files: list[Path],
    manifest: Path,
    split_details: dict,
    args,
) -> dict:
    print(f"\n===== variant={variant_name} feature_mode={feature_mode} =====")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = Mission3Dataset(
        train_files, tokenizer, args.max_length, args.chunk_overlap,
        feature_mode=feature_mode,
    )
    val_dataset = Mission3Dataset(
        val_files, tokenizer, args.max_length, args.chunk_overlap,
        feature_mode=feature_mode,
    )
    sample_weights, manifest_config = load_hard_negative_sampling(
        manifest,
        train_files,
        TARGET_SYMPTOMS,
        args.hard_negative_symptoms,
        args.hard_negative_oversample,
    )
    sampler = GuaranteedFullPassSampler(sample_weights, seed=args.seed)
    sampling = {**manifest_config, **sampler_config(sample_weights)}
    train_loader = make_loader(
        train_dataset, args.batch_size, sampler=sampler, seed=args.seed,
    )
    val_loader = make_loader(val_dataset, args.batch_size, seed=args.seed)

    model = SymptomClassifier(args.model_name, len(TARGET_SYMPTOMS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    pos_weight, loss_config = compute_pos_weights(
        train_files, TARGET_SYMPTOMS, args.pos_weight_scale, args.pos_weight_max,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    variant_dir = args.output_dir / variant_name
    best_path = variant_dir / "best.pt"
    history = []
    best_f1 = -1.0
    best_epoch = 0
    for epoch in range(1, args.max_epochs + 1):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            args.gradient_accumulation_steps,
        )
        labels, scores = collect_scores(model, val_loader, device)
        metrics = fixed_threshold_metrics(labels, scores)
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "per_label": metrics["per_label"],
        }
        history.append(row)
        print(
            f"[{variant_name}] epoch={epoch}/{args.max_epochs} loss={loss:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} micro_f1={metrics['micro_f1']:.4f}"
        )
        # 동점이면 더 짧은 epoch를 유지한다.
        if metrics["macro_f1"] > best_f1:
            best_f1 = float(metrics["macro_f1"])
            best_epoch = epoch
            atomic_checkpoint(
                best_path,
                checkpoint_payload(
                    model,
                    args=args,
                    epoch=epoch,
                    best_f1=best_f1,
                    feature_mode=feature_mode,
                    loss_config=loss_config,
                    sampling=sampling,
                    split_details=split_details,
                ),
            )

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    labels, scores = collect_scores(model, val_loader, device)
    threshold_result = None
    selected_thresholds = [0.5] * len(TARGET_SYMPTOMS)
    tuned_metrics = fixed_threshold_metrics(labels, scores)
    if args.tune_thresholds:
        threshold_result = tune_thresholds(
            labels,
            scores,
            TARGET_SYMPTOMS,
            compute_multilabel_metrics,
            objective="macro_f1",
        )
        selected_thresholds = [float(value) for value in threshold_result["thresholds"]]
        tuned_metrics = compute_multilabel_metrics(
            labels,
            scores > np.asarray(selected_thresholds),
            TARGET_SYMPTOMS,
        )
        atomic_json(variant_dir / "thresholds.json", threshold_result)

    best_checkpoint["decision_thresholds"] = selected_thresholds
    best_checkpoint["threshold_selection"] = threshold_result
    atomic_checkpoint(best_path, best_checkpoint)
    result = {
        "variant": variant_name,
        "speaker_role_tokens": feature_mode == "speaker",
        "feature_mode": feature_mode,
        "checkpoint": str(best_path.resolve()),
        "best_epoch": best_epoch,
        "best_macro_f1_fixed_0_5": best_f1,
        "macro_f1_selected_thresholds": float(tuned_metrics["macro_f1"]),
        "decision_thresholds": dict(zip(TARGET_SYMPTOMS, selected_thresholds)),
        "threshold_method": threshold_result["method"] if threshold_result else "fixed_0.5",
        "history": history,
        "sampling_config": sampling,
    }
    atomic_json(variant_dir / "result.json", result)
    del model, optimizer, scaler, best_checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def train_final(
    selected: dict,
    all_files: list[Path],
    manifest: Path,
    args,
) -> Path:
    print(
        f"\n===== final 100%: variant={selected['variant']} "
        f"epochs={selected['best_epoch']} ====="
    )
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    feature_mode = selected.get(
        "feature_mode",
        "speaker" if selected.get("speaker_role_tokens", False) else "none",
    )
    dataset = Mission3Dataset(
        all_files,
        tokenizer,
        args.max_length,
        args.chunk_overlap,
        feature_mode=feature_mode,
    )
    sample_weights, manifest_config = load_hard_negative_sampling(
        manifest,
        all_files,
        TARGET_SYMPTOMS,
        args.hard_negative_symptoms,
        args.hard_negative_oversample,
    )
    sampler = GuaranteedFullPassSampler(sample_weights, seed=args.seed)
    sampling = {**manifest_config, **sampler_config(sample_weights)}
    loader = make_loader(dataset, args.batch_size, sampler=sampler, seed=args.seed)
    model = SymptomClassifier(args.model_name, len(TARGET_SYMPTOMS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    pos_weight, loss_config = compute_pos_weights(
        all_files, TARGET_SYMPTOMS, args.pos_weight_scale, args.pos_weight_max,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    for epoch in range(1, selected["best_epoch"] + 1):
        loss = train_epoch(
            model,
            loader,
            optimizer,
            criterion,
            device,
            scaler,
            args.gradient_accumulation_steps,
        )
        print(f"[final] epoch={epoch}/{selected['best_epoch']} loss={loss:.4f}")

    final_path = args.output_dir / "final" / "final.pt"
    atomic_checkpoint(
        final_path,
        checkpoint_payload(
            model,
            args=args,
            epoch=int(selected["best_epoch"]),
            best_f1=float(selected["best_macro_f1_fixed_0_5"]),
            feature_mode=feature_mode,
            loss_config=loss_config,
            sampling=sampling,
            split_details={
                "method": "final_training_100_percent",
                "num_train": len(all_files),
                "internal_validation": 0,
                "epoch_source": "selected_on_internal_multilabel_stratified_validation",
            },
            decision_thresholds=list(selected["decision_thresholds"].values()),
            threshold_selection={
                "source": "selected_variant_internal_validation",
                "method": selected["threshold_method"],
            },
        ),
    )
    return final_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--baseline_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="klue/roberta-large")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--internal_val_ratio", type=float, default=0.10)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--chunk_overlap", type=int, default=64)
    parser.add_argument("--pos_weight_scale", type=float, default=0.25)
    parser.add_argument("--pos_weight_max", type=float, default=3.0)
    parser.add_argument("--hard_negative_oversample", type=float, default=1.5)
    parser.add_argument(
        "--hard_negative_symptoms", nargs="+", default=DEFAULT_FOCUS_SYMPTOMS,
    )
    parser.add_argument("--hard_negative_threshold", type=float, default=0.5)
    parser.add_argument("--mine_batch_size", type=int, default=4)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANT_FEATURE_MODES),
        default=list(VARIANT_FEATURE_MODES),
        help="같은 내부 split에서 비교할 입력 방식",
    )
    parser.add_argument(
        "--tune_thresholds", action="store_true",
        help="best epoch 선택 후 내부 검증에서만 증상별 임계값을 선택",
    )
    parser.add_argument("--skip_final_retrain", action="store_true")
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="파이프라인 smoke test용. 실제 최종 실험에서는 지정하지 않음",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="완료된 split, 채굴, variant 결과와 final checkpoint를 재사용",
    )
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.internal_val_ratio < 1.0:
        parser.error("internal_val_ratio는 0과 1 사이여야 합니다.")
    for name in ("max_epochs", "batch_size", "gradient_accumulation_steps", "mine_batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"{name}는 1 이상이어야 합니다.")
    if not math.isfinite(args.hard_negative_oversample) or args.hard_negative_oversample < 1:
        parser.error("hard_negative_oversample은 유한한 1 이상이어야 합니다.")
    if "오심" in args.hard_negative_symptoms:
        parser.error("이 실험에서는 오심을 hard-negative 대상에서 제외해야 합니다.")
    unknown = sorted(set(args.hard_negative_symptoms) - set(TARGET_SYMPTOMS))
    if unknown:
        parser.error(f"알 수 없는 증상: {unknown}")
    return args


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    plan = {
        "split": f"multilabel stratified {1-args.internal_val_ratio:.0%}/{args.internal_val_ratio:.0%}",
        "hard_negative_symptoms": list(args.hard_negative_symptoms),
        "hard_negative_oversample": args.hard_negative_oversample,
        "sampler": "guaranteed full pass + hard-negative extras",
        "variants": list(args.variants),
        "epoch_selection": "internal validation Macro F1 at threshold 0.5",
        "threshold_selection": "per symptom on internal validation" if args.tune_thresholds else "fixed 0.5",
        "final": "selected configuration retrained on Training 100%",
        "official_validation_used": False,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    if not args.allow_cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 없습니다. 실제 large 모델 실험은 GPU에서 실행하세요.")
    if not args.baseline_checkpoint.is_file():
        raise FileNotFoundError(f"기준 체크포인트가 없습니다: {args.baseline_checkpoint}")
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not (args.overwrite or args.resume)
    ):
        raise FileExistsError(
            f"출력 폴더가 비어 있지 않습니다: {args.output_dir}. "
            "새 폴더를 쓰거나 --overwrite를 지정하세요."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_files = list_label_files(get_train_label_dir(args.data_root))
    if args.max_samples is not None:
        rng = random.Random(args.seed)
        all_files = rng.sample(all_files, min(args.max_samples, len(all_files)))
    split_path = args.output_dir / "split.json"
    if args.resume and split_path.is_file():
        split_details = json.loads(split_path.read_text(encoding="utf-8"))
        file_by_name = {path.name: path for path in all_files}
        missing_names = [
            name
            for name in (
                split_details.get("train_files", [])
                + split_details.get("validation_files", [])
            )
            if name not in file_by_name
        ]
        if missing_names:
            raise FileNotFoundError(
                f"저장된 split 파일이 현재 Training에 없습니다: {missing_names[:10]}"
            )
        train_files = [file_by_name[name] for name in split_details["train_files"]]
        val_files = [
            file_by_name[name] for name in split_details["validation_files"]
        ]
        print(f"저장된 split 재사용: {split_path}")
    else:
        train_files, val_files, split_details = split_training_files(
            all_files, args.internal_val_ratio, args.seed,
        )
        atomic_json(split_path, split_details)

    train_manifest = args.output_dir / "hard_negatives_train90.csv"
    train_mining_path = train_manifest.with_suffix(".summary.json")
    if args.resume and train_manifest.is_file() and train_mining_path.is_file():
        train_mining = json.loads(train_mining_path.read_text(encoding="utf-8"))
        print(f"Training 90% hard-negative manifest 재사용: {train_manifest}")
    else:
        train_mining = write_hard_negative_manifest(
            train_files,
            args.baseline_checkpoint,
            train_manifest,
            args,
            split_name="internal_train",
        )

    variant_specs = [
        (variant_name, VARIANT_FEATURE_MODES[variant_name])
        for variant_name in args.variants
    ]
    results = []
    for variant_name, feature_mode in variant_specs:
        result_path = args.output_dir / variant_name / "result.json"
        checkpoint_path = args.output_dir / variant_name / "best.pt"
        if args.resume and result_path.is_file() and checkpoint_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            print(f"완료된 variant 재사용: {variant_name}")
        else:
            result = train_variant(
                variant_name,
                feature_mode,
                train_files,
                val_files,
                train_manifest,
                split_details,
                args,
            )
        results.append(result)
    selected = max(results, key=lambda row: row["best_macro_f1_fixed_0_5"])
    final_path = None
    full_mining = None
    if not args.skip_final_retrain:
        full_manifest = args.output_dir / "hard_negatives_train100.csv"
        full_mining_path = full_manifest.with_suffix(".summary.json")
        if args.resume and full_manifest.is_file() and full_mining_path.is_file():
            full_mining = json.loads(
                full_mining_path.read_text(encoding="utf-8")
            )
            print(f"Training 100% hard-negative manifest 재사용: {full_manifest}")
        else:
            full_mining = write_hard_negative_manifest(
                all_files,
                args.baseline_checkpoint,
                full_manifest,
                args,
                split_name="final_train_100_percent",
            )
        candidate_final = args.output_dir / "final" / "final.pt"
        if args.resume and candidate_final.is_file():
            final_path = candidate_final
            print(f"완료된 final checkpoint 재사용: {final_path}")
        else:
            final_path = train_final(selected, all_files, full_manifest, args)

    summary = {
        "plan": plan,
        "split": split_details,
        "train90_hard_negative_mining": train_mining,
        "variant_results": results,
        "selected_variant": selected["variant"],
        "selected_epoch": selected["best_epoch"],
        "selected_macro_f1_fixed_0_5": selected["best_macro_f1_fixed_0_5"],
        "selected_thresholds": selected["decision_thresholds"],
        "final_training_hard_negative_mining": full_mining,
        "final_checkpoint": str(final_path.resolve()) if final_path else None,
        "official_validation_used": False,
    }
    atomic_json(args.output_dir / "experiment_summary.json", summary)
    print("\n실험 완료")
    print(json.dumps({
        "selected_variant": summary["selected_variant"],
        "selected_epoch": summary["selected_epoch"],
        "selected_macro_f1_fixed_0_5": summary["selected_macro_f1_fixed_0_5"],
        "final_checkpoint": summary["final_checkpoint"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
