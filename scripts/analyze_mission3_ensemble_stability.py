"""CPU-only analysis of fixed M3 probabilities; never fits or selects a model.

NPZ columns follow submission/mission3/mission3_model.py LABELS. The dumps do
not contain label metadata, so that order is an explicit input assumption.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

LABELS = ("고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란")
TAGS = ("r3", "k12", "p0")
WEIGHTS = (0.45, 0.15, 0.40)
MODELS = (*TAGS, "ensemble")


def load_split(folder: Path, split: str):
    reference = None
    probabilities = []
    sources = {}
    for tag in TAGS:
        path = folder / f"{tag}_{split}.npz"
        with np.load(path, allow_pickle=False) as data:
            names, probs, gold = data["names"], data["probs"], data["gold"]
        if names.ndim != 1 or not len(names) or len(set(names.tolist())) != len(names):
            raise ValueError(f"{path.name}: empty or duplicate/non-vector names")
        if probs.shape != (len(names), len(LABELS)) or gold.shape != probs.shape:
            raise ValueError(f"{path.name}: invalid shape")
        if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
            raise ValueError(f"{path.name}: invalid probabilities")
        if not np.isin(gold, [0, 1]).all():
            raise ValueError(f"{path.name}: nonbinary gold")
        if reference is None:
            reference = (names.copy(), gold.astype(bool))
        elif not np.array_equal(names, reference[0]) or not np.array_equal(gold, reference[1]):
            raise ValueError(f"{path.name}: names/order/gold mismatch")
        probabilities.append(probs.astype(np.float64))
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    ensemble = sum(w * p for w, p in zip(WEIGHTS, probabilities))
    return reference[0], reference[1], np.stack([*probabilities, ensemble]) >= 0.5, sources


def scores(counts):
    """Counts have (..., model, label, TP/FP/FN) axes."""
    tp, fp, fn = np.moveaxis(counts, -1, 0)
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)


def interval(values):
    return np.percentile(values, [2.5, 97.5], axis=0).T.tolist()


def analyse(gold, predictions, iterations=10000, seed=42):
    # One bootstrap unit is a call (all nine labels together). Exactly the same
    # sampled calls are used for every model, preserving pairing and label dependence.
    events = np.stack((predictions & gold, predictions & ~gold, ~predictions & gold), axis=-1)
    counts = events.sum(axis=1)
    per_label = scores(counts)
    rng = np.random.default_rng(seed)
    samples = np.empty((iterations, len(MODELS), len(LABELS)))
    # Bounded batches use BLAS over call multiplicities instead of materializing
    # huge resampled prediction tensors. CPU thread limits are set by the caller.
    flat = events.transpose(1, 0, 2, 3).reshape(len(gold), -1).astype(float)
    for start in range(0, iterations, 64):
        size = min(64, iterations - start)
        multiplicities = np.array([np.bincount(rng.integers(0, len(gold), len(gold)), minlength=len(gold))
                                   for _ in range(size)], dtype=float)
        samples[start:start + size] = scores((multiplicities @ flat).reshape(size, len(MODELS), len(LABELS), 3))
    result = {"rows": len(gold), "models": {}, "paired_comparisons": {}, "single_model_error_overlap": {}}
    for m, name in enumerate(MODELS):
        tp, fp, fn = counts[m].T
        precision = np.divide(tp, tp + fp, out=np.zeros(9), where=tp + fp != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros(9), where=tp + fn != 0)
        per_ci = interval(samples[:, m])
        result["models"][name] = {
            "macro_f1": float(per_label[m].mean()), "macro_f1_ci95": interval(samples[:, m].mean(axis=1)),
            "micro_f1": float(scores(counts[m].sum(axis=0))),
            "empty_predictions": int((~predictions[m].any(axis=1)).sum()),
            "per_symptom": {label: {"support": int(gold[:, j].sum()), "tp": int(tp[j]),
                "fp": int(fp[j]), "fn": int(fn[j]), "tn": int(len(gold) - tp[j] - fp[j] - fn[j]),
                "precision": float(precision[j]), "recall": float(recall[j]),
                "f1": float(per_label[m, j]), "f1_ci95": per_ci[j]}
                for j, label in enumerate(LABELS)}}
    for m, name in enumerate(TAGS):
        diff = samples[:, -1] - samples[:, m]
        old_error, new_error = predictions[m] != gold, predictions[-1] != gold
        per_ci = interval(diff)
        result["paired_comparisons"][name] = {
            "macro_f1_delta": float((per_label[-1] - per_label[m]).mean()),
            "macro_f1_delta_ci95": interval(diff.mean(axis=1)),
            "bootstrap_positive_fraction": float((diff.mean(axis=1) > 0).mean()),
            "per_symptom": {label: {"f1_delta": float(per_label[-1, j] - per_label[m, j]),
                "f1_delta_ci95": per_ci[j], "corrected_errors": int((old_error[:, j] & ~new_error[:, j]).sum()),
                "introduced_errors": int((~old_error[:, j] & new_error[:, j]).sum()),
                "shared_errors": int((old_error[:, j] & new_error[:, j]).sum())}
                for j, label in enumerate(LABELS)}}
    for a in range(len(TAGS)):
        for b in range(a + 1, len(TAGS)):
            ea, eb = predictions[a] != gold, predictions[b] != gold
            result["single_model_error_overlap"][f"{TAGS[a]}+{TAGS[b]}"] = {
                label: {"both_wrong": int((ea[:, j] & eb[:, j]).sum()),
                    "only_first_wrong": int((ea[:, j] & ~eb[:, j]).sum()),
                    "only_second_wrong": int((~ea[:, j] & eb[:, j]).sum()),
                    "both_false_negative": int((~predictions[a, :, j] & ~predictions[b, :, j] & gold[:, j]).sum()),
                    "both_false_positive": int((predictions[a, :, j] & predictions[b, :, j] & ~gold[:, j]).sum()),
                    "error_jaccard": float((ea[:, j] & eb[:, j]).sum() / max(1, (ea[:, j] | eb[:, j]).sum()))}
                for j, label in enumerate(LABELS)}
    result["all_single_models_wrong"] = {
        label: int(((predictions[:3, :, j] != gold[:, j]).all(axis=0)).sum()) for j, label in enumerate(LABELS)}
    # Co-label counts describe associations only, not semantic causes of errors.
    result["ensemble_error_gold_colabels"] = {
        label: {kind: {other: int((mask & gold[:, k]).sum()) for k, other in enumerate(LABELS) if k != j}
                for kind, mask in {"fp": predictions[-1, :, j] & ~gold[:, j],
                                   "fn": ~predictions[-1, :, j] & gold[:, j]}.items()}
        for j, label in enumerate(LABELS)}
    return result


def write_report(result, out):
    def fmt(value):
        return f"{value:.4f}"
    def ci(value):
        return f"[{value[0]:+.4f}, {value[1]:+.4f}]"
    lines = ["# M3 고정 앙상블 오류·안정성 분석", "",
        "RoBERTa-003(r3) 0.45 + KcBERT-012(k12) 0.15 + TF-IDF(p0) 0.40, 판정 `>= 0.5`.",
        "저장된 확률만 읽었으며 학습·가중치 탐색·공식 Validation 재추론은 수행하지 않았다.", "",
        f"통화를 단위로 paired bootstrap {result['bootstrap_iterations']:,}회, seed {result['seed']}, percentile 95% CI를 계산했다.",
        "동일한 재표집 행을 모든 모델에 적용하고 통화 내 9개 라벨을 함께 유지했다. CI는 고정 모델의 평가 표본 변동만 반영한다.",
        "여러 seed의 학습 변동, 가중치 선택 불확실성, 새 데이터로의 일반화는 이 CI에 포함되지 않는다.",
        "홀드아웃은 이미 가중치 선택에 쓰여 낙관적일 수 있다. 증상별 CI는 다중 비교를 보정하지 않은 탐색적 결과다.",
        "NPZ에 라벨 이름이 없으므로 기존 dump 코드의 LABELS 순서를 가정했다. 파일명·순서·정답 일치와 확률 범위를 검증했다.",
        "예측·정답만으로 분석했으므로 텍스트 근거 유무와 부정 표현의 오류 원인은 판정하지 않았다.", ""]
    val = result["splits"]["val"]
    delta = val["paired_comparisons"]["r3"]
    symptom = val["models"]["ensemble"]["per_symptom"]
    improved = sum(row["f1_delta"] > 0 for row in delta["per_symptom"].values())
    excluding_zero = sum(row["f1_delta_ci95"][0] > 0 for row in delta["per_symptom"].values())
    weak = delta["per_symptom"]["전신쇠약"]
    overlaps = val["single_model_error_overlap"]
    nausea_colabels = val["ensemble_error_gold_colabels"]["오심"]["fp"]
    lines += ["## 해석", "",
        f"공식 Validation에서 앙상블 Macro F1은 {val['models']['ensemble']['macro_f1']:.4f}, RoBERTa 대비 {delta['macro_f1_delta']:+.4f} (95% CI {ci(delta['macro_f1_delta_ci95'])})다.",
        f"기존 채택 기준 +0.01에 {'도달했다' if delta['macro_f1_delta'] >= .01 else '점 추정값이 미달한다'}. 제출 변경 여부는 별도 결정 사항이다.",
        f"공식 Validation에서 {improved}/9개 증상의 F1 점 추정값이 상승했고, {excluding_zero}/9개의 개선 CI 하한이 0보다 크다. 증상별 판단은 다중 비교를 보정하지 않은 탐색 결과다.",
        f"오심 F1 {symptom['오심']['f1']:.4f}, 두통 {symptom['두통']['f1']:.4f}, 전신쇠약 {symptom['전신쇠약']['f1']:.4f}. 오심 FP {symptom['오심']['fp']}·FN {symptom['오심']['fn']}, 두통 FP {symptom['두통']['fp']}·FN {symptom['두통']['fn']}건이다.",
        f"전신쇠약은 RoBERTa 오류 {weak['corrected_errors']}개를 고치고 {weak['introduced_errors']}개를 새로 만들었고 F1 차이는 {weak['f1_delta']:+.4f}다. F1 상승이 모든 오류 수의 감소를 뜻하지는 않는다.",
        f"빈 예측은 RoBERTa {val['models']['r3']['empty_predictions']}→앙상블 {val['models']['ensemble']['empty_predictions']}건이다. 정답이 빈 통화도 있을 수 있으므로 빈 예측 수 자체를 오답 수로 해석하지 않는다.",
        f"오심 오류의 Jaccard 겹침은 RoBERTa·KcBERT {overlaps['r3+k12']['오심']['error_jaccard']:.3f}, RoBERTa·TF-IDF {overlaps['r3+p0']['오심']['error_jaccard']:.3f}이다. 이것만으로 가중치별 기여를 분리할 수는 없다.",
        f"오심 FP 통화 {symptom['오심']['fp']}개 중 어지러움 정답은 {nausea_colabels['어지러움']}개, 구토는 {nausea_colabels['구토']}개다(중복 가능). 이는 정답 증상의 동반 집계이며 전사 해석이나 오류 원인의 증거는 아니다.", ""]
    for split, data in result["splits"].items():
        lines += [f"## {split} ({data['rows']:,}통화)", "",
            "| 모델 | Macro F1 | 95% CI | Micro F1 | 빈 예측 |", "|---|---:|---|---:|---:|"]
        for name, row in data["models"].items():
            lines.append(f"| {name} | {fmt(row['macro_f1'])} | {ci(row['macro_f1_ci95'])} | {fmt(row['micro_f1'])} | {row['empty_predictions']} |")
        lines += ["", "| 앙상블 비교 대상 | Macro F1 차이 | paired 95% CI |", "|---|---:|---|"]
        for name, row in data["paired_comparisons"].items():
            lines.append(f"| {name} | {row['macro_f1_delta']:+.4f} | {ci(row['macro_f1_delta_ci95'])} |")
        lines += ["", "| 증상 | 양성 | 앙상블 P / R | F1 | r3 대비 ΔF1 [95% CI] | FP / FN | r3 오류 수정 / 새 오류 | 3개 단일 모두 오답 |",
            "|---|---:|---|---:|---|---|---|---:|"]
        for label in LABELS:
            row = data["models"]["ensemble"]["per_symptom"][label]
            delta = data["paired_comparisons"]["r3"]["per_symptom"][label]
            lines.append(f"| {label} | {row['support']} | {row['precision']:.3f} / {row['recall']:.3f} | {row['f1']:.4f} | {delta['f1_delta']:+.4f} {ci(delta['f1_delta_ci95'])} | {row['fp']} / {row['fn']} | {delta['corrected_errors']} / {delta['introduced_errors']} | {data['all_single_models_wrong'][label]} |")
        lines.append("")
        lines += ["| 증상 | r3+k12 공통 FN / FP | r3+p0 공통 FN / FP | r3+k12 오류 Jaccard | r3+p0 오류 Jaccard |",
            "|---|---|---|---:|---:|"]
        for label in LABELS:
            first = data["single_model_error_overlap"]["r3+k12"][label]
            second = data["single_model_error_overlap"]["r3+p0"][label]
            lines.append(f"| {label} | {first['both_false_negative']} / {first['both_false_positive']} | {second['both_false_negative']} / {second['both_false_positive']} | {first['error_jaccard']:.3f} | {second['error_jaccard']:.3f} |")
        lines += ["", "오류 Jaccard = 공통 오답 / 두 모델 오답의 합집합. 위 건수의 단위는 통화·증상 쌍이다.", ""]
    lines += ["## 재현", "", "```powershell",
        "$env:OMP_NUM_THREADS='2'; $env:OPENBLAS_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'",
        "python scripts/analyze_mission3_ensemble_stability.py --probs-dir reports/mission3/ensemble/probs --output-dir reports/mission3/ensemble/stability --iterations 10000 --seed 42",
        "```", "", "JSON에는 모델 간 오류 겹침과 FP/FN 통화의 다른 정답 증상 집계도 포함한다. 전사 원문과 통화별 결과는 출력하지 않는다."]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.iterations < 100:
        parser.error("--iterations must be at least 100")
    loaded = {split: load_split(args.probs_dir, split) for split in ("holdout", "val")}
    if set(loaded["holdout"][0]) & set(loaded["val"][0]):
        raise ValueError("holdout and official Validation overlap")
    result = {"models": dict(zip(TAGS, WEIGHTS)), "labels": LABELS, "threshold": 0.5,
        "bootstrap_iterations": args.iterations, "seed": args.seed,
        "bootstrap_unit": "call", "ci_method": "percentile", "fitting_performed": False,
        "weight_search_performed": False, "source_sha256": {}, "splits": {}}
    for split, (_, gold, predictions, sources) in loaded.items():
        result["source_sha256"].update(sources)
        result["splits"][split] = analyse(gold, predictions, args.iterations, args.seed)
        print(f"{split}: {len(gold)} calls, analysis complete", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "per_symptom.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "model", "symptom", "support", "tp", "fp", "fn", "precision", "recall", "f1"])
        for split, data in result["splits"].items():
            for model, row in data["models"].items():
                for label, values in row["per_symptom"].items():
                    writer.writerow([split, model, label, *[values[k] for k in ("support", "tp", "fp", "fn", "precision", "recall", "f1")]])
    write_report(result, args.output_dir)


if __name__ == "__main__":
    main()
