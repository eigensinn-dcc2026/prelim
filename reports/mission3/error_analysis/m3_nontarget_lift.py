"""Lift between non-target symptoms (rows) and the 9 Mission 3 targets (columns), Training labels only.

lift(n, t) = P(t | n) / P(t). > 1: t is recorded more often when n is recorded. Only the `symptom`
field is read (no transcript, no other fields). Rows: non-target symptoms with >= 100 Training calls,
ordered by clinical group. Writes a PNG (for slides) and a CSV (table view).

python m3_nontarget_lift.py <Training label dir> <out_dir>
"""
import csv
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

TARGETS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
GROUPS = [  # clinical grouping fixed in the plan (not chosen from the data)
    ("신경", ["의식장애", "실신", "경련/발작", "마비", "정신장애"]),
    ("흉부·심혈관", ["흉통", "가슴불편감", "심계항진"]),
    ("호흡기", ["기침"]),
    ("소화기", ["설사", "혈변", "토혈", "변비"]),
    ("외상·출혈", ["찰과상", "타박상", "그 밖의출혈"]),
    ("통증", ["기타통증", "그밖의통증기타", "요통"]),
    ("기타", ["기타", "배뇨장애"]),
]
MIN_CALLS = 100
# Diverging pair from the dataviz reference palette: blue <-> red, neutral gray midpoint.
BLUE, GRAY, RED = "#2a78d6", "#f0efec", "#e34948"
INK, MUTED = "#1f1f1e", "#6b6a66"


def main():
    label_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    calls = []
    for p in sorted(label_dir.rglob("*.json")):
        s = json.loads(p.read_text(encoding="utf-8-sig")).get("symptom", [])
        s = [s] if isinstance(s, str) else s
        calls.append({unicodedata.normalize("NFC", x.strip()) for x in s if isinstance(x, str)})
    n = len(calls)
    counts = Counter(x for c in calls for x in c)
    rows, row_groups = [], []
    for g, members in GROUPS:
        for m in members:
            if counts[m] >= MIN_CALLS:
                rows.append(m)
                row_groups.append(g)
    p_t = np.array([counts[t] / n for t in TARGETS])
    lift = np.zeros((len(rows), len(TARGETS)))
    joint = np.zeros_like(lift, dtype=int)
    for i, r in enumerate(rows):
        with_r = [c for c in calls if r in c]
        for j, t in enumerate(TARGETS):
            joint[i, j] = sum(t in c for c in with_r)
            lift[i, j] = (joint[i, j] / len(with_r)) / p_t[j]

    with (out / "m3_nontarget_lift.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "non_target", "calls"] + [f"lift_{t}" for t in TARGETS] + [f"joint_{t}" for t in TARGETS])
        for i, r in enumerate(rows):
            w.writerow([row_groups[i], r, counts[r]] + [f"{v:.3f}" for v in lift[i]] + list(joint[i]))

    plt.rcParams["font.family"] = ["Malgun Gothic"]
    plt.rcParams["axes.unicode_minus"] = False
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, GRAY, RED])
    log_lift = np.log2(np.clip(lift, 1 / 16, 16))
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    fig, ax = plt.subplots(figsize=(9.6, 0.42 * len(rows) + 1.9), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.imshow(np.clip(log_lift, -2, 2), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(TARGETS)), TARGETS, fontsize=10, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), [f"{r}  ({counts[r]:,})" for r in rows], fontsize=9.5, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # 2px-ish surface gaps between cells
    ax.set_xticks(np.arange(-.5, len(TARGETS)), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(len(rows)):
        for j in range(len(TARGETS)):
            v = lift[i, j]
            strong = abs(log_lift[i, j]) >= 1.2
            txt = f"{v:.1f}" if v < 9.95 else f"{v:.0f}"
            few = joint[i, j] < 20
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if strong else (MUTED if few else INK), alpha=.55 if (strong and few) else 1,
                    fontweight="bold" if v >= 2 and not few else "normal")
    # group separators and labels
    starts = [0] + [i for i in range(1, len(rows)) if row_groups[i] != row_groups[i - 1]]
    for k, s0 in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(rows)
        if s0:
            ax.axhline(s0 - .5, color=INK, linewidth=1.0, xmin=-.35, clip_on=False)
        ax.text(len(TARGETS) - .3, (s0 + e - 1) / 2, row_groups[s0], ha="left", va="center", fontsize=9, color=MUTED)
    ax.set_title("9개 외 증상이 기록된 통화에서 9개 평가 증상이 함께 기록되는 정도 (Lift)",
                 fontsize=12, color=INK, pad=34, loc="left")
    fig.text(0.01, 0.01, f"Lift = P(평가 증상 | 9개 외 증상) / P(평가 증상).  1보다 크면(빨강) 함께 더 자주, 작으면(파랑) 덜 기록됨.\n"
             f"Training {n:,}통화의 symptom 라벨만 사용.  흐린 글자: 함께 기록된 통화 20건 미만(해석 주의).",
             fontsize=7.5, color=MUTED)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, fraction=.025, pad=.11, ticks=[-2, -1, 0, 1, 2])
    cb.ax.set_yticklabels(["¼배", "½배", "1배", "2배", "4배 이상"], fontsize=8, color=INK)
    cb.outline.set_visible(False)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(out / "m3_nontarget_lift.png", facecolor=fig.get_facecolor())
    top = sorted(((lift[i, j], rows[i], TARGETS[j], joint[i, j]) for i in range(len(rows)) for j in range(len(TARGETS))
                  if joint[i, j] >= 20), reverse=True)[:12]
    print("rows", len(rows), "calls", n)
    print("top lifts (joint>=20):", [(r, t, round(v, 2), int(c)) for v, r, t, c in top])


if __name__ == "__main__":
    main()
