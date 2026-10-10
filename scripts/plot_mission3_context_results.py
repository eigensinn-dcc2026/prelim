"""Plot measured fixed-epoch M3 scores and diagnostic subsets for slides."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["svg.hashsalt"] = "m3-context-measured-results"

LABELS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
ENGLISH = ["Fever", "Vomiting", "Headache", "Abdominal pain", "Dizziness", "Laceration",
           "Nausea", "Weakness", "Dyspnea"]
COLORS = {"control": "#6e7782", "memory": "#25829b", "evidence": "#d08729"}


def draw(frozen_path, fine_path, output):
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("smoke"):
        raise ValueError("Presentation results require a full-data experiment")
    fine = json.loads(fine_path.read_text(encoding="utf-8")) if fine_path else None
    if fine and (fine.get("smoke") or fine.get("source") != frozen["source"]):
        raise ValueError("Fine results must share the full-data frozen source")
    output.mkdir(parents=True, exist_ok=True)
    stages = [("Frozen encoder / epoch 10", frozen["pilot"], "")]
    if fine:
        stages.append(("Encoder fine-tuning / epoch 1", fine["pilot"], "_fine"))
    fig, axes = plt.subplots(1, len(stages), figsize=(6.3 * len(stages), 4.3), squeeze=False)
    for ax, (title, scores, suffix) in zip(axes[0], stages):
        keys = ["roberta003", "existing_ensemble"] + [m + suffix for m in COLORS]
        keys = [key for key in keys if key in scores]
        names = ["RoBERTa-003", "Existing ensemble"] + ["CLS control", "A: similar calls", "B: answer evidence"]
        values = [scores[key]["macro_f1"] for key in keys]
        bars = ax.barh(names[:len(keys)], values, color=["#b9c5cf", "#334f66", *COLORS.values()][:len(keys)])
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
        ax.invert_yaxis()
        ax.set(xlim=(0, .75), xlabel="Macro F1 at threshold 0.5", title=title)
        ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle("M3 internal holdout: 5,792 calls / seed 42", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "measured_comparison.png", dpi=200)
    svg_path = output / "measured_comparison.svg"
    fig.savefig(svg_path, metadata={"Date": None})
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    for method, color in COLORS.items():
        path = frozen_path.parent / f"frozen_{method}_history.json"
        history = json.loads(path.read_text(encoding="utf-8"))
        ax.plot([m["epoch"] for m in history], [m["macro_f1"] for m in history], marker="o", color=color, label=method)
    ax.axhline(frozen["pilot"]["existing_ensemble"]["macro_f1"], color="#334f66", linestyle="--", label="Existing ensemble")
    ax.axvline(10, color="#bbbbbb", linestyle=":", label="Fixed evaluated epoch")
    ax.set(xlabel="Frozen-head epoch", ylabel="Macro F1", title="Fixed-epoch evaluation; the existing ensemble remains stronger", xticks=range(1, 11))
    ax.legend(ncol=2, fontsize=9)
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout(); fig.savefig(output / "frozen_learning_curves.png", dpi=200); plt.close(fig)

    fine_diagnostics = fine and all("per_label" in fine["pilot"].get(m + "_fine", {}) for m in COLORS)
    scores = fine["pilot"] if fine_diagnostics else frozen["pilot"]
    suffix = "_fine" if fine_diagnostics else ""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    x = np.arange(9)
    for method, color in COLORS.items():
        key = method + suffix
        if key not in scores:
            continue
        base = scores["roberta003"]["per_label"]
        measured = scores[key]["per_label"]
        axes[0].plot(x, [measured[s]["f1"] - base[s]["f1"] for s in LABELS], "o-", color=color, label=method)
        axes[1].plot(x, [(np.nan if measured[s]["regex_no_match"]["recall"] is None else measured[s]["regex_no_match"]["recall"]) - (np.nan if base[s]["regex_no_match"]["recall"] is None else base[s]["regex_no_match"]["recall"])
                        for s in LABELS], "o-", color=color, label=method)
    for ax in axes:
        ax.axhline(0, color="#999999", linewidth=1)
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes[0].set(ylabel="Label F1 change", title=("Fine" if fine_diagnostics else "Frozen") + " changes from original RoBERTa-003")
    axes[0].legend(ncol=3)
    axes[1].set(ylabel="Recall change: regex-missed positives", xticks=x, xticklabels=ENGLISH)
    axes[1].tick_params(axis="x", labelrotation=30)
    fig.text(.02, .015, "Heuristic subsets use full transcripts; model inputs retain the first 256 tokens. Single seed, no adoption claim.", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, 1)); fig.savefig(output / "measured_diagnostics.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, default=Path("reports/mission3/context/frozen_metrics.json"))
    parser.add_argument("--fine", type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/mission3/context"))
    args = parser.parse_args()
    draw(args.frozen, args.fine, args.out)
