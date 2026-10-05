"""Figures and Markdown section for the deep-dive EDA (:mod:`dcc_data.eda_deep`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .eda_report import (
    TRAIN_COLOR,
    VALIDATION_COLOR,
    _configure_matplotlib,
    _finish_figure,
)
from .mission3 import SYMPTOM_CLASSES


ACOUSTIC_FEATURES = (
    "duration_sec",
    "rms_mean",
    "zcr_mean",
    "spectral_centroid_mean",
    "silence_ratio_abs_le_0_01",
)
FEATURE_LABELS = {
    "duration_sec": "연결 음성(초)",
    "rms_mean": "RMS",
    "zcr_mean": "ZCR",
    "spectral_centroid_mean": "Spectral centroid",
    "silence_ratio_abs_le_0_01": "무음 비율",
}
GENDER_LABELS = {
    "call_duration_s": "통화 길이",
    "caller_speaking_s": "신고자 발화 시간",
    "caller_utterance_count": "신고자 발화 수",
    "caller_speaking_share": "신고자 발화 비중",
}
POSITIVE_COLOR = "#009E73"
NEGATIVE_COLOR = "#CC79A7"

FIGURE_NAMES = {
    "ecdf": "deep_01_split_ecdf.png",
    "overlap": "deep_02_overlap.png",
    "gender": "deep_03_gender_effects.png",
    "lift": "deep_04_symptom_lift.png",
    "acoustic": "deep_05_acoustic_effects.png",
}


def render_deep_figures(
    summary: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Render the five evidence figures referenced by the deep-dive section."""

    _configure_matplotlib()
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    created: list[Path] = []

    def save(fig: Any, key: str) -> None:
        path = output_dir / FIGURE_NAMES[key]
        _finish_figure(fig, path)
        created.append(path)

    # 1. Train/Validation ECDFs.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, column, label in (
        (axes[0], "call_duration_s", "통화 길이 (초)"),
        (axes[1], "utterance_count", "통화당 발화 수"),
    ):
        for split, color in (("Training", TRAIN_COLOR), ("Validation", VALIDATION_COLOR)):
            values = np.sort(
                frame.loc[frame["split"] == split, column].dropna().to_numpy(float)
            )
            ax.step(
                values,
                np.arange(1, len(values) + 1) / len(values),
                where="post",
                color=color,
                linewidth=2,
                label=split,
            )
        ax.set_xlabel(label)
        ax.set_ylabel("누적 비율")
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Training과 Validation의 경험적 누적분포(ECDF)")
    save(fig, "ecdf")

    # 2. Overlap: frequent in calls, small in time.
    metrics = (
        ("겹침 있는 통화", lambda b: 100 * b["calls_with_overlap"] / b["n_calls"]),
        (
            "화자 간 겹침 통화",
            lambda b: 100 * b["calls_with_cross_speaker_overlap"] / b["n_calls"],
        ),
        ("0.1초 이하 겹침 쌍", lambda b: 100 * b["pair_counts"]["micro_le_100ms_rate"]),
        ("통화당 겹침시간 비율(중앙값)", lambda b: 100 * b["overlap_time_ratio_per_call"]["p50"]),
    )
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, split, color in (
        (-width / 2, "Training", TRAIN_COLOR),
        (width / 2, "Validation", VALIDATION_COLOR),
    ):
        block = summary["overlap"][split]
        values = [getter(block) for _, getter in metrics]
        bars = ax.bar(x + offset, values, width, label=split, color=color)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], fontsize=8, padding=2)
    ax.set_xticks(x, [name for name, _ in metrics], rotation=18, ha="right")
    ax.set_ylabel("비율 (%)")
    ax.set_title("겹침은 거의 모든 통화에 있지만, 차지하는 시간은 작다")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "overlap")

    # 3. Gender effect sizes (M1).
    values = [
        summary["gender"][key]["cliffs_delta_left_minus_right"] for key in GENDER_LABELS
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.barh(
        np.arange(len(values)),
        values,
        color=[POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in values],
    )
    ax.set_yticks(np.arange(len(values)), list(GENDER_LABELS.values()))
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Cliff's δ (여성 − 남성)")
    ax.set_title("성별에 따른 통화 길이·발화량 차이 (M1 shortcut 점검)")
    ax.bar_label(bars, labels=[f"{v:+.3f}" for v in values], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "gender")

    # 4. Symptom pair lift (M3).
    pairwise = summary["symptoms"]["Combined"]["pairwise"]
    size = len(SYMPTOM_CLASSES)
    matrix = np.ones((size, size), dtype=float)
    for i, left in enumerate(SYMPTOM_CLASSES):
        for j, right in enumerate(SYMPTOM_CLASSES):
            if i != j:
                key = f"{left}__{right}" if i < j else f"{right}__{left}"
                matrix[i, j] = pairwise[key]["lift"]
    fig, ax = plt.subplots(figsize=(8.2, 7))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=0, vmax=3)
    ax.set_xticks(np.arange(size), SYMPTOM_CLASSES, rotation=45, ha="right")
    ax.set_yticks(np.arange(size), SYMPTOM_CLASSES)
    ax.set_title("증상 쌍 Lift (1 = 독립일 때의 기대 수준)")
    for i in range(size):
        for j in range(size):
            if i != j:
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if matrix[i, j] > 2.2 or matrix[i, j] < 0.35 else "#222222",
                )
    fig.colorbar(image, ax=ax, shrink=0.8, label="Lift")
    save(fig, "lift")

    # 5. Acoustic effect sizes by split (M1/M2 audio features).
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ax, split in zip(axes, ("Training", "Validation")):
        values = [
            summary["acoustic_sample"]["by_split"][split][feature][
                "cliffs_delta_left_minus_right"
            ]
            for feature in ACOUSTIC_FEATURES
        ]
        ax.barh(
            np.arange(len(values)),
            values,
            color=[POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in values],
        )
        ax.set_yticks(
            np.arange(len(values)), [FEATURE_LABELS[key] for key in ACOUSTIC_FEATURES]
        )
        ax.axvline(0, color="#333333", linewidth=1)
        ax.set_title(split)
        ax.set_xlabel("Cliff's δ (신고자 − 119대원)")
        ax.set_xlim(-1, 1)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("화자별 음향 특징 효과크기 (통화 단위 짝 비교)")
    save(fig, "acoustic")
    return created


# ---------------------------------------------------------------------------
# Markdown section
# ---------------------------------------------------------------------------

_SPLITS = (("Training", "Training"), ("Validation", "Validation"))


def _effect_label(delta: float) -> str:
    """Romano et al. (2006) verbal thresholds for |Cliff's delta|."""

    magnitude = abs(delta)
    if magnitude < 0.147:
        return "무시할 수준"
    if magnitude < 0.33:
        return "작음"
    if magnitude < 0.474:
        return "중간"
    return "큼"


def _num(profile: Mapping[str, Any], key: str = "p50") -> float:
    value = profile.get(key)
    return 0.0 if value is None else float(value)


def _fig(key: str, caption: str) -> list[str]:
    return [f"![{caption}](figures/{FIGURE_NAMES[key]})", ""]


def build_deep_section(deep: Mapping[str, Any], number: int) -> list[str]:
    """Return Markdown lines for the deep-dive section numbered ``number``."""

    n = number
    lines: list[str] = [
        f"## {n}. 심화 분석 — 분포 이동·겹침·효과크기",
        "",
        "> 전수(32,840통화) JSON에서 집계 통계만 계산했고 전사 원문·주소·식별자는 저장하지 않았습니다 "
        "(`reports/eda/eda_deep_summary.json`). "
        "**Cliff's δ**(효과크기: 한 집단 값이 다른 집단보다 클 확률 − 작을 확률, −1~1)는 "
        "|δ| 0.147 미만 무시할 수준, 0.33 미만 작음, 0.474 미만 중간, 그 이상 큼으로 읽습니다. "
        "표본이 매우 커서 p-value는 의사결정 기준으로 쓰지 않고 효과크기와 신뢰구간으로 판단합니다.",
        "",
    ]

    # --- Split shift ---------------------------------------------------
    shift = deep["split_comparison"]
    labels = {
        "call_duration_s": ("통화 길이", "초"),
        "utterance_count": ("통화당 발화 수", "개"),
        "caller_speaking_share": ("신고자 발화 비중", ""),
    }
    lines += [
        f"### {n}.1 Training ↔ Validation 분포 이동",
        "",
        "| 변수 | Training 중앙값 | Validation 중앙값 | KS(누적분포 최대 차이) | Wasserstein(분포 이동 거리) | 표준화 평균차(Valid−Train) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, (label, unit) in labels.items():
        block = shift[key]
        lines.append(
            f"| {label} | {_num(block['training']):.3f}{unit} | "
            f"{_num(block['validation']):.3f}{unit} | {block['ks_statistic']:.4f} | "
            f"{block['wasserstein_distance']:.4f} | "
            f"{block['standardized_mean_difference_validation_minus_training']:+.4f} |"
        )
    lines.append("")
    lines += _fig("ecdf", "Training과 Validation의 ECDF")

    # --- Overlap ---------------------------------------------------------
    lines += [
        f"### {n}.2 발화 겹침 (M2 오류 슬라이스 설계)",
        "",
        "| Split | 겹침 있는 통화 | 화자 간 겹침 통화 | 통화당 겹침시간 비율 중앙값 | p90 | 0.1초 이하 겹침 쌍 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, label in _SPLITS:
        block = deep["overlap"][split]
        total = block["n_calls"] or 1
        ratio = block["overlap_time_ratio_per_call"]
        lines.append(
            f"| {label} | {100 * block['calls_with_overlap'] / total:.2f}% | "
            f"{100 * block['calls_with_cross_speaker_overlap'] / total:.2f}% | "
            f"{100 * _num(ratio):.2f}% | {100 * _num(ratio, 'p90'):.2f}% | "
            f"{100 * block['pair_counts']['micro_le_100ms_rate']:.1f}% |"
        )
    lines.append("")
    lines += _fig("overlap", "발화 겹침 지표")

    # --- Gender ----------------------------------------------------------
    lines += [
        f"### {n}.3 성별 효과 (M1 shortcut 점검)",
        "",
        "| 변수 | 여성 중앙값 | 남성 중앙값 | Cliff's δ(여−남) | 효과 크기 | 중앙값 차이 95% 구간 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for key, label in GENDER_LABELS.items():
        block = deep["gender"][key]
        delta = block["cliffs_delta_left_minus_right"]
        ci = block["median_difference_bootstrap_95ci"]
        lines.append(
            f"| {label} | {_num(block['left']):.3f} | {_num(block['right']):.3f} | "
            f"{delta:+.3f} | {_effect_label(delta)} | "
            f"{ci['estimate']:+.3f} [{ci['ci95_low']:+.3f}, {ci['ci95_high']:+.3f}] |"
        )
    lines.append("")
    lines += _fig("gender", "성별 효과크기")

    # --- Speaker utterances ----------------------------------------------
    lines += [
        f"### {n}.4 화자별 발화 길이 (M2)",
        "",
        "| Split | 화자 | 발화 수 | 중앙값 | p90 | 0.5초 미만 | 1초 미만 | 4초 이상 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, label in _SPLITS:
        block = deep["speaker_utterances"][split]
        for role, role_label in (("dispatcher", "119대원"), ("caller", "신고자")):
            profile, buckets = block[role], block["duration_bucket_pct"][role]
            lines.append(
                f"| {label} | {role_label} | {profile['n']:,} | {_num(profile):.3f}초 | "
                f"{_num(profile, 'p90'):.3f}초 | {buckets['lt_0_5s']:.1f}% | "
                f"{buckets['lt_1s']:.1f}% | {buckets['ge_4s']:.1f}% |"
            )
    lines.append("")
    for split, label in _SPLITS:
        block = deep["speaker_utterances"][split]
        within = block["within_call_median_effect"]
        delta = block["cliffs_delta_caller_minus_dispatcher"]
        lines.append(
            f"- {label}: 발화 단위 Cliff's δ(신고자−119대원) {delta:+.3f}({_effect_label(delta)}), "
            f"같은 통화 안에서 신고자의 중앙 발화가 더 긴 통화의 우세도 "
            f"{within['paired_probability_difference']:+.3f}"
        )
    lines.append("")

    # --- Symptoms --------------------------------------------------------
    combined = deep["symptoms"]["Combined"]
    cardinality = combined["cardinality_distribution"]
    multi = sum(v["pct"] for key, v in cardinality.items() if int(key) >= 2)
    pairs = sorted(combined["pairwise"].values(), key=lambda item: item["lift"])
    distribution = ", ".join(f"{key}개 {value['pct']:.1f}%" for key, value in cardinality.items())
    lines += [
        f"### {n}.5 증상 공출현 구조 (M3 Macro F1)",
        "",
        f"- 통화당 레이블 수 분포: {distribution} (2개 이상 {multi:.1f}%)",
        "",
        "| 가장 흔한 레이블 조합 | 통화 수 | 비율 |",
        "|---|---:|---:|",
    ]
    for item in combined["top_combinations"][:5]:
        name = " + ".join(item["labels"]) or "(없음)"
        lines.append(f"| {name} | {item['count']:,} | {item['pct']:.2f}% |")
    lines += ["", "| Lift 상위(함께 나오기 쉬움) | Lift | Jaccard |", "|---|---:|---:|"]
    for item in reversed(pairs[-3:]):
        lines.append(
            f"| {item['left']} · {item['right']} | {item['lift']:.3f} | {item['jaccard']:.3f} |"
        )
    lines += ["", "| Lift 하위(함께 나오기 어려움) | Lift | Jaccard |", "|---|---:|---:|"]
    for item in pairs[:3]:
        lines.append(
            f"| {item['left']} · {item['right']} | {item['lift']:.3f} | {item['jaccard']:.3f} |"
        )
    lines.append("")
    lines += _fig("lift", "증상 쌍 Lift")

    # --- Acoustic --------------------------------------------------------
    acoustic = deep["acoustic_sample"]
    lines += [
        f"### {n}.6 화자·성별 음향 효과 (M1/M2, 층화 표본)",
        "",
        "| 특징 | Training δ(신고자−119대원) | Validation δ | 효과 크기(Training) |",
        "|---|---:|---:|---|",
    ]
    for feature in ACOUSTIC_FEATURES:
        train_delta = acoustic["by_split"]["Training"][feature]["cliffs_delta_left_minus_right"]
        valid_delta = acoustic["by_split"]["Validation"][feature]["cliffs_delta_left_minus_right"]
        lines.append(
            f"| {FEATURE_LABELS[feature]} | {train_delta:+.3f} | {valid_delta:+.3f} | "
            f"{_effect_label(train_delta)} |"
        )
    lines += [
        "",
        "| 신고자 음향 특징 | Cliff's δ(여성−남성) | 효과 크기 |",
        "|---|---:|---|",
    ]
    for feature, block in acoustic["caller_gender"].items():
        delta = block["cliffs_delta_left_minus_right"]
        lines.append(f"| {FEATURE_LABELS[feature]} | {delta:+.3f} | {_effect_label(delta)} |")
    lines.append("")
    lines += _fig("acoustic", "화자별 음향 효과크기")

    # --- Implications ----------------------------------------------------
    lines += [f"### {n}.7 Mission별 시사점", ""]
    gender_deltas = {
        label: deep["gender"][key]["cliffs_delta_left_minus_right"]
        for key, label in GENDER_LABELS.items()
    }
    strongest = max(gender_deltas, key=lambda name: abs(gender_deltas[name]))
    caller_acoustic = {
        FEATURE_LABELS[feature]: block["cliffs_delta_left_minus_right"]
        for feature, block in acoustic["caller_gender"].items()
    }
    top_acoustic = max(caller_acoustic, key=lambda name: abs(caller_acoustic[name]))
    peak_gender = max(abs(value) for value in (*gender_deltas.values(), *caller_acoustic.values()))
    if peak_gender < 0.147:
        lines.append(
            f"- **M1**: 통화 길이·발화량·음량·ZCR·spectral centroid 어느 것도 성별을 의미 있게 구분하지 못합니다"
            f"(최대 |δ| {peak_gender:.3f}). 길이 기반 shortcut 위험은 낮지만, 성별 신호는 이 요약 특징이 아니라 "
            "MFCC 등 세부 스펙트럼 특징에 있을 것으로 추정되므로 오류 분석에서 확인합니다."
        )
    else:
        lines.append(
            f"- **M1**: 길이·발화량 변수 중 성별 효과가 가장 큰 것은 {strongest}"
            f"(δ {gender_deltas[strongest]:+.3f}, {_effect_label(gender_deltas[strongest])}), "
            f"음향 특징 중에는 {top_acoustic}(δ {caller_acoustic[top_acoustic]:+.3f}, "
            f"{_effect_label(caller_acoustic[top_acoustic])})입니다. 효과가 작지 않은 변수는 shortcut 여부를 점검합니다."
        )
    train_overlap = deep["overlap"]["Training"]
    ratio = train_overlap["overlap_time_ratio_per_call"]
    overlap_pct = 100 * train_overlap["calls_with_overlap"] / (train_overlap["n_calls"] or 1)
    speaker_delta = deep["speaker_utterances"]["Training"]["cliffs_delta_caller_minus_dispatcher"]
    speaker_acoustic = {
        FEATURE_LABELS[feature]: acoustic["by_split"]["Training"][feature]["cliffs_delta_left_minus_right"]
        for feature in ACOUSTIC_FEATURES
    }
    valid_speaker_acoustic = {
        FEATURE_LABELS[feature]: acoustic["by_split"]["Validation"][feature]["cliffs_delta_left_minus_right"]
        for feature in ACOUSTIC_FEATURES
    }
    top_speaker = max(speaker_acoustic, key=lambda name: abs(speaker_acoustic[name]))
    lines.append(
        f"- **M2**: 통화의 {overlap_pct:.1f}%에 겹침이 있어 '겹침 유무'는 슬라이스로 쓸 수 없고, "
        f"통화당 겹침시간 비율(중앙값 {100 * _num(ratio):.1f}%, p90 {100 * _num(ratio, 'p90'):.1f}%) 상위 구간을 "
        "슬라이스 후보로 제안합니다(baseline 오류와의 연관은 별도 검증 필요). "
        f"발화 길이의 화자 효과는 {_effect_label(speaker_delta)}(δ {speaker_delta:+.3f})이라 길이만으로 화자를 구분하긴 어렵고, "
        f"음향 요약 중 효과가 가장 큰 것은 {top_speaker}(Training δ {speaker_acoustic[top_speaker]:+.3f}, "
        f"Validation δ {valid_speaker_acoustic[top_speaker]:+.3f}, {_effect_label(speaker_acoustic[top_speaker])})입니다."
    )
    lines.append(
        f"- **M3**: 2개 이상 레이블 통화가 {multi:.1f}%이므로 단일 레이블 가정은 맞지 않습니다. "
        f"Lift가 가장 높은 쌍은 {pairs[-1]['left']}·{pairs[-1]['right']}({pairs[-1]['lift']:.2f}), "
        f"가장 낮은 쌍은 {pairs[0]['left']}·{pairs[0]['right']}({pairs[0]['lift']:.2f})입니다."
    )
    max_ks = max(block["ks_statistic"] for block in shift.values())
    verdict = "분포가 매우 유사" if max_ks < 0.05 else "차이를 점검할 필요"
    lines.append(
        f"- **공통**: Training↔Validation KS 통계량 최대값은 {max_ks:.4f}({verdict})이며, "
        "Validation 점수가 Training 분포 가정에서 크게 벗어나지 않는지 해석의 전제로 사용합니다."
    )
    lines.append("")
    return lines


def build_deep_recommendations(deep: Mapping[str, Any]) -> list[str]:
    """Return numbered recommendation lines grounded in the deep-dive numbers."""

    acoustic = deep["acoustic_sample"]
    gender_deltas = [
        deep["gender"][key]["cliffs_delta_left_minus_right"] for key in GENDER_LABELS
    ]
    gender_deltas += [
        block["cliffs_delta_left_minus_right"]
        for block in acoustic["caller_gender"].values()
    ]
    peak_gender = max(abs(value) for value in gender_deltas)
    if peak_gender < 0.147:
        m1 = (
            f"M1은 통화 길이·발화량·음량·ZCR·spectral centroid의 성별 효과가 모두 무시할 수준"
            f"(최대 |δ| {peak_gender:.3f})이라 길이 기반 shortcut(본질적인 신호 대신 우연한 쉬운 단서를 "
            "학습하는 현상) 위험은 낮다고 봅니다. 오류 분석은 통화 길이·신고자 발화 시간 구간별 정확도를 "
            "확인하는 수준으로 두고, 개선 실험은 MFCC 등 세부 스펙트럼 특징을 우선 시도합니다."
        )
    else:
        m1 = (
            f"M1은 성별 효과가 작지 않은 변수가 있어(최대 |δ| {peak_gender:.3f}) 성별별 발화량·음향 특징을 "
            "함께 통제해 길이 기반 shortcut(본질적인 신호 대신 우연한 쉬운 단서를 학습하는 현상) 여부를 점검합니다."
        )

    overlap = deep["overlap"]["Training"]
    ratio = overlap["overlap_time_ratio_per_call"]
    buckets = deep["speaker_utterances"]["Training"]["duration_bucket_pct"]
    short_half = max(b["lt_0_5s"] for b in buckets.values())
    short_one = max(b["lt_1s"] for b in buckets.values())
    feature_delta = {
        feature: acoustic["by_split"]["Training"][feature]["cliffs_delta_left_minus_right"]
        for feature in ACOUSTIC_FEATURES
    }
    top = max(feature_delta, key=lambda name: abs(feature_delta[name]))
    m2 = (
        f"M2는 겹침 유무(통화의 {100 * overlap['calls_with_overlap'] / (overlap['n_calls'] or 1):.0f}%에 해당)가 아니라 "
        f"통화당 겹침시간 비율 상위 구간(p90 {100 * _num(ratio, 'p90'):.1f}% 이상), "
        f"0.5초 미만 발화(최대 {short_half:.1f}%)와 1초 미만 발화(최대 {short_one:.1f}%), "
        "단일 화자 비중이 큰 통화를 고정 오류 슬라이스(특정 조건으로 나눈 오류 분석 구간)로 둡니다. "
        f"또한 {FEATURE_LABELS[top]}의 화자 효과가 {_effect_label(feature_delta[top])}"
        f"(δ {feature_delta[top]:+.3f})이므로 해당 특징을 뺀 ablation(특징 제거 비교)으로 모델이 "
        "녹음 채널 차이에 의존하는지 점검합니다."
    )

    symptoms = deep["symptoms"]["Combined"]
    multi = sum(
        value["pct"]
        for key, value in symptoms["cardinality_distribution"].items()
        if int(key) >= 2
    )
    pairs = sorted(symptoms["pairwise"].values(), key=lambda item: item["lift"])
    m3 = (
        f"M3는 2개 이상 레이블 통화가 {multi:.1f}%이므로 희소 레이블과 공출현 조합을 보존하는 "
        "multilabel stratification(여러 레이블 조합의 비율을 유지하는 분할 방식)을 유지합니다. "
        f"공출현이 두드러진 {pairs[-1]['left']}·{pairs[-1]['right']}(Lift {pairs[-1]['lift']:.2f}) 쌍은 "
        "오류 분석에서 함께 틀리는지 확인하고, 거의 함께 나오지 않는 "
        f"{pairs[0]['left']}·{pairs[0]['right']}(Lift {pairs[0]['lift']:.2f}) 쌍은 "
        "상호배타 후처리가 이득인지 Training 교차검증으로 검토합니다(Validation은 반복 조정에 쓰지 않음)."
    )

    max_ks = max(block["ks_statistic"] for block in deep["split_comparison"].values())
    quality = deep["existing_quality"]
    identifier_overlap = sum(quality["identifier_overlap_count"].values())
    transcript_overlap = quality["exact_transcript_hash_overlap_count"]
    split = (
        f"Training↔Validation 분포 차이는 KS 최대 {max_ks:.3f}로 작아 Validation 점수를 Training 분포의 "
        f"대리 지표로 해석할 수 있습니다. 식별자 중복은 {identifier_overlap}건이고 전사 해시 중복 "
        f"{transcript_overlap}건은 원문을 노출하지 않는 별도 수동 감사 대상으로 둡니다."
    )

    selected = acoustic["sample_design"]["selected_calls"]
    limits = (
        "본 보고서는 기술통계이며 상관관계를 원인으로 해석하지 않습니다. 음향 효과는 층화 표본"
        f"(Training {selected['Training']:,} · Validation {selected['Validation']:,}통화, "
        "통화당 최대 15초) 기준이라 전수 결과로 일반화하지 않습니다."
    )
    return [f"{index}. {text}" for index, text in enumerate((m1, m2, m3, split, limits), start=1)]
