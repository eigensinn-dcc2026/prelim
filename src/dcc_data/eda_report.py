"""Markdown and static-chart rendering for :mod:`dcc_data.eda`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .mission3 import SYMPTOM_CLASSES


TRAIN_COLOR = "#0072B2"
VALIDATION_COLOR = "#E69F00"
SPEAKER_COLORS = {"0": "#009E73", "1": "#CC79A7"}


def _configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": ["Malgun Gothic", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 160,
        }
    )


def _finish_figure(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def render_figures(summary: dict[str, Any], output_dir: Path) -> list[Path]:
    """Render the fixed set of report figures and return created paths."""

    _configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    # Gender composition.
    genders = sorted(
        set(summary["training"]["mission1"]["gender_distribution"])
        | set(summary["validation"]["mission1"]["gender_distribution"])
    )
    x = np.arange(len(genders))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    for offset, split, color, label in (
        (-width / 2, "training", TRAIN_COLOR, "Training"),
        (width / 2, "validation", VALIDATION_COLOR, "Validation"),
    ):
        values = [
            summary[split]["mission1"]["gender_distribution"]
            .get(gender, {})
            .get("pct", 0)
            for gender in genders
        ]
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_title("Train/Validation 신고자 성별 구성")
    ax.set_xlabel("성별 라벨")
    ax.set_ylabel("통화 비율 (%)")
    ax.set_xticks(x, genders)
    ax.set_ylim(0, max(60, ax.get_ylim()[1] * 1.12))
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    path = output_dir / "gender_distribution.png"
    _finish_figure(fig, path)
    created.append(path)

    # Call-duration histogram (fixed bins make the two splits comparable).
    train_hist = summary["training"]["common"]["call_duration_histogram"]
    val_hist = summary["validation"]["common"]["call_duration_histogram"]
    edges = train_hist["edges"]
    labels = [f"{int(left)}–{int(right)}" for left, right in zip(edges, edges[1:])]
    train_total = sum(train_hist["counts"]) or 1
    val_total = sum(val_hist["counts"]) or 1
    train_pct = [100 * value / train_total for value in train_hist["counts"]]
    val_pct = [100 * value / val_total for value in val_hist["counts"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, train_pct, marker="o", linewidth=2, color=TRAIN_COLOR, label="Training")
    ax.plot(
        x,
        val_pct,
        marker="o",
        linewidth=2,
        color=VALIDATION_COLOR,
        label="Validation",
    )
    ax.set_title("Train/Validation 통화 길이 분포")
    ax.set_xlabel("통화 길이 구간 (초)")
    ax.set_ylabel("통화 비율 (%)")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    path = output_dir / "call_duration_distribution.png"
    _finish_figure(fig, path)
    created.append(path)

    # M2 utterance duration buckets by speaker, Training only.
    buckets = summary["training"]["mission2"]["duration_bucket_count_by_speaker"]
    bucket_labels = list(next(iter(buckets.values())).keys()) if buckets else []
    x = np.arange(len(bucket_labels))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for offset, speaker, label in (
        (-width / 2, "0", "119대원 (0)"),
        (width / 2, "1", "신고자 (1)"),
    ):
        values = [buckets.get(speaker, {}).get(bucket, 0) for bucket in bucket_labels]
        total = sum(values) or 1
        percentages = [100 * value / total for value in values]
        ax.bar(
            x + offset,
            percentages,
            width,
            label=label,
            color=SPEAKER_COLORS[speaker],
        )
    ax.set_title("Training 발화 길이 구간별 화자 분포")
    ax.set_xlabel("발화 길이")
    ax.set_ylabel("화자별 발화 비율 (%)")
    ax.set_xticks(x, bucket_labels)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    path = output_dir / "speaker_duration_buckets.png"
    _finish_figure(fig, path)
    created.append(path)

    # Target symptom prevalence.
    y = np.arange(len(SYMPTOM_CLASSES))
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for offset, split, color, label in (
        (-width / 2, "training", TRAIN_COLOR, "Training"),
        (width / 2, "validation", VALIDATION_COLOR, "Validation"),
    ):
        values = [
            summary[split]["mission3"]["target_symptom_prevalence"][symptom][
                "pct_of_calls"
            ]
            for symptom in SYMPTOM_CLASSES
        ]
        ax.barh(y + offset, values, width, label=label, color=color)
    ax.set_title("목표 증상별 통화 단위 유병률")
    ax.set_xlabel("해당 증상을 포함한 통화 비율 (%)")
    ax.set_ylabel("증상")
    ax.set_yticks(y, SYMPTOM_CLASSES)
    ax.invert_yaxis()
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    path = output_dir / "symptom_prevalence.png"
    _finish_figure(fig, path)
    created.append(path)

    # Co-occurrence heatmap.
    cooccurrence = summary["training"]["mission3"]["cooccurrence"]
    matrix = np.asarray(
        [
            [cooccurrence[left][right] for right in SYMPTOM_CLASSES]
            for left in SYMPTOM_CLASSES
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues", aspect="equal")
    ax.set_title("Training 목표 증상 동시 출현 횟수")
    ax.set_xticks(np.arange(len(SYMPTOM_CLASSES)), SYMPTOM_CLASSES, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(SYMPTOM_CLASSES)), SYMPTOM_CLASSES)
    fig.colorbar(image, ax=ax, shrink=0.8, label="통화 수")
    path = output_dir / "symptom_cooccurrence_training.png"
    _finish_figure(fig, path)
    created.append(path)

    # Train/validation prevalence deltas.
    deltas = summary["cross_split"]["symptom_prevalence_delta_percentage_points"]
    values = [deltas[label] for label in SYMPTOM_CLASSES]
    colors = [VALIDATION_COLOR if value >= 0 else TRAIN_COLOR for value in values]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(np.arange(len(values)), values, color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("Validation 증상 유병률 변화 (Training 대비)")
    ax.set_xlabel("퍼센트포인트 차이 (Validation − Training)")
    ax.set_yticks(np.arange(len(values)), SYMPTOM_CLASSES)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    path = output_dir / "symptom_train_validation_shift.png"
    _finish_figure(fig, path)
    created.append(path)

    return created


def _value(profile: dict[str, Any], field: str, default: Any = 0) -> Any:
    value = profile.get(field, default)
    return default if value is None else value


def _quality_findings(summary: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    for split in ("training", "validation"):
        label = "Training" if split == "training" else "Validation"
        common = summary[split]["common"]
        failed = sum(common["json_read_failures"].values())
        wav_failed = sum(common["wav"]["read_errors"].values())
        if failed or wav_failed:
            findings.append(
                f"{label}: JSON 읽기 실패 {failed:,}건, WAV 헤더 실패 {wav_failed:,}건"
            )
        for name, values in common["identifiers"].items():
            if values["duplicate_groups"]:
                findings.append(
                    f"{label}: `{name}` 중복 그룹 {values['duplicate_groups']:,}개"
                )
        invalid_intervals = common["timestamp_issue_counts"].get(
            "invalid_utterance_interval", 0
        )
        outside = common["timestamp_issue_counts"].get(
            "utterance_outside_wav_50ms", 0
        )
        if invalid_intervals or outside:
            findings.append(
                f"{label}: 잘못된 발화 구간 {invalid_intervals:,}건, "
                f"WAV 범위 초과(50ms 허용) {outside:,}건"
            )
        overlap_calls = common["timestamp_issue_counts"].get(
            "calls_with_overlapping_utterances", 0
        )
        if overlap_calls:
            findings.append(
                f"{label}: 발화 구간이 하나 이상 겹치는 통화 {overlap_calls:,}건 "
                f"({100 * overlap_calls / (common['json_parsed'] or 1):.2f}%) — "
                "동시 발화와 라벨링 관행을 함께 검토해야 함"
            )
    overlap = summary["cross_split"]["identifier_overlap_count"]
    for name, count in overlap.items():
        if count:
            findings.append(f"Train/Validation `{name}` 중복 {count:,}개")
    transcript_overlap = summary["cross_split"]["exact_transcript_hash_overlap_count"]
    if transcript_overlap:
        findings.append(
            f"Train/Validation 정규화 전사(공백·대소문자 등을 통일한 텍스트) "
            f"해시(원문 대신 사용하는 지문값) 중복 {transcript_overlap:,}개"
        )
    return findings or ["현재 자동 점검 기준에서 치명적인 무결성 오류가 발견되지 않았습니다."]


def build_markdown_report(
    summary: dict[str, Any], deep: dict[str, Any] | None = None
) -> str:
    """Build a compact, source-backed Korean EDA report.

    ``deep`` is the optional :mod:`dcc_data.eda_deep` summary; when given, a
    deep-dive section is inserted before the recommendations.
    """

    train = summary["training"]
    val = summary["validation"]
    meta = summary["meta"]
    train_m1 = train["mission1"]
    val_m1 = val["mission1"]
    train_m2 = train["mission2"]
    train_m3 = train["mission3"]
    val_m3 = val["mission3"]
    oov = summary["cross_split"]["validation_token_oov"]

    lines = [
        "# DCC 2026 통합 EDA 보고서",
        "",
        f"> 생성 시각(UTC, 협정 세계시): `{meta['generated_at']}`  ",
        f"> 재현 seed(난수 생성 초기값): `{meta['seed']}` · JSON(구조화 라벨 파일)/WAV(음성 파일) 헤더(기본 메타정보): 전수 · 파형: 층화 표본(그룹별 비율을 유지한 표본 추출)",
        "",
        "## 1. 데이터 개요",
        "",
        "| Split(데이터 분할) | Manifest(파일 연결 목록) | JSON 성공 | WAV 헤더 오류 | 통화 길이 중앙값(가운데 값) | 발화 수 중앙값(가운데 값) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, block in (("Training", train), ("Validation", val)):
        common = block["common"]
        lines.append(
            f"| {label} | {common['manifest_rows']:,} | {common['json_parsed']:,} | "
            f"{sum(common['wav']['read_errors'].values()):,} | "
            f"{_value(common['call_duration_s'], 'p50'):.2f}초 | "
            f"{_value(common['utterances_per_call'], 'p50'):.1f}개 |"
        )

    lines.extend(
        [
            "",
            "![통화 길이 분포](figures/call_duration_distribution.png)",
            "",
            "## 2. 데이터 품질",
            "",
        ]
    )
    lines.extend(f"- {finding}" for finding in _quality_findings(summary))
    train_diff = train["common"]["wav"]["json_minus_wav_duration_ms"]
    val_diff = val["common"]["wav"]["json_minus_wav_duration_ms"]
    lines.extend(
        [
            f"- JSON 통화 길이 − WAV 길이 중앙값(가운데 값): Training "
            f"{_value(train_diff, 'p50'):.3f}ms(밀리초), Validation {_value(val_diff, 'p50'):.3f}ms(밀리초)",
            "- 주소와 원문 전사는 산출물에 저장하지 않았으며, 전사 중복은 SHA-256 해시(원문을 복원하지 않고 동일 여부만 비교하는 지문)로만 비교했습니다.",
            "",
            "## 3. Mission 1 — 신고자 성별",
            "",
            "| Split(데이터 분할) | F(여성) | M(남성) | 신고자 음성 중앙값(가운데 값) | 신고자 발화 수 중앙값(가운데 값) |",
            "|---|---:|---:|---:|---:|",
            f"| Training | {train_m1['gender_distribution'].get('F', {}).get('pct', 0):.2f}% | "
            f"{train_m1['gender_distribution'].get('M', {}).get('pct', 0):.2f}% | "
            f"{_value(train_m1['caller_duration_s'], 'p50'):.2f}초 | "
            f"{_value(train_m1['caller_utterances_per_call'], 'p50'):.1f}개 |",
            f"| Validation | {val_m1['gender_distribution'].get('F', {}).get('pct', 0):.2f}% | "
            f"{val_m1['gender_distribution'].get('M', {}).get('pct', 0):.2f}% | "
            f"{_value(val_m1['caller_duration_s'], 'p50'):.2f}초 | "
            f"{_value(val_m1['caller_utterances_per_call'], 'p50'):.1f}개 |",
            "",
            "![성별 구성](figures/gender_distribution.png)",
            "",
            "성별 비율뿐 아니라 성별별 신고자 발화량과 표본 음향 특징을 함께 확인해야 "
            "길이 차이를 성별 신호로 오인하는 위험을 줄일 수 있습니다.",
            "",
            "## 4. Mission 2 — 화자 구분",
            "",
            "| 화자 | 발화 수 | 발화 비율 | 길이 중앙값(가운데 값) | 길이 p90(90%가 이 값 이하인 지점) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for speaker, label in (("0", "119대원"), ("1", "신고자")):
        count_info = train_m2["utterance_count_by_speaker"].get(speaker, {})
        durations = train_m2["utterance_duration_s_by_speaker"].get(speaker, {})
        lines.append(
            f"| {label} ({speaker}) | {count_info.get('count', 0):,} | "
            f"{count_info.get('pct', 0):.2f}% | {_value(durations, 'p50'):.3f}초 | "
            f"{_value(durations, 'p90'):.3f}초 |"
        )
    lines.extend(
        [
            "",
            "![화자별 발화 길이](figures/speaker_duration_buckets.png)",
            "",
            "발화 길이는 화자와 연관될 수 있지만 대화 구조에 의한 편향일 수 있으므로, "
            "단독 결정 변수로 해석하지 않습니다. 겹침·짧은 맞장구 구간은 별도 오류 분석 대상으로 둡니다.",
            "",
            "## 5. Mission 3 — 증상 다중 레이블",
            "",
            "| 증상 | Training 건수 | Training 비율 | Validation 건수 | Validation 비율 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for symptom in SYMPTOM_CLASSES:
        train_value = train_m3["target_symptom_prevalence"][symptom]
        val_value = val_m3["target_symptom_prevalence"][symptom]
        lines.append(
            f"| {symptom} | {train_value['count']:,} | {train_value['pct_of_calls']:.2f}% | "
            f"{val_value['count']:,} | {val_value['pct_of_calls']:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"- Training label cardinality(통화 하나에 붙은 평균 증상 레이블 수) 평균: "
            f"{_value(train_m3['label_cardinality'], 'mean'):.3f}",
            f"- 목표 9개 증상이 하나도 없는 통화: Training "
            f"{train_m3['zero_target_symptom_calls']:,}건, Validation "
            f"{val_m3['zero_target_symptom_calls']:,}건",
            f"- Validation token OOV(학습 데이터에 없었던 단어): {oov['oov_occurrences']:,}/"
            f"{oov['token_occurrences']:,} ({100 * oov['oov_rate']:.2f}%)",
            "",
            "![증상 유병률](figures/symptom_prevalence.png)",
            "",
            "![증상 동시 출현](figures/symptom_cooccurrence_training.png)",
            "",
            "![증상 분포 변화](figures/symptom_train_validation_shift.png)",
            "",
            "Macro F1(각 증상별 F1 점수를 같은 비중으로 평균낸 지표) 관점에서는 희소 증상의 표본 수와 동시 출현 구조를 함께 고려해야 합니다. "
            "Validation 분포 차이는 설명용이며 threshold(분류 여부를 결정하는 기준값)나 모델을 반복 조정하는 근거로 사용하지 않습니다.",
            "",
            "## 6. 파형 표본 분석",
            "",
        ]
    )
    waveform = summary.get("waveform_sample")
    if not waveform or waveform.get("skipped"):
        lines.append("- 파형 분석을 건너뛰었습니다.")
    else:
        for split, label in (("training", "Training"), ("validation", "Validation")):
            block = waveform[split]
            lines.append(
                f"- {label}: {block['selected_calls']:,}통화 선택, "
                f"{block['successful_calls']:,}통화 성공, {block['feature_rows']:,}개 화자 벡터, "
                f"실패 {block['failures']:,}건"
            )
        lines.extend(
            [
                "",
                "| Split / 화자 | 연결 음성 중앙값(가운데 값) | RMS(평균 음량 크기) 중앙값 | ZCR(신호 부호 변화율) 중앙값 | spectral centroid(주파수 에너지 중심) 중앙값 | 무음 비율 중앙값 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for split, split_label in (("training", "Training"), ("validation", "Validation")):
            for speaker, speaker_label in (("0", "119대원"), ("1", "신고자")):
                features = waveform[split]["by_speaker"][speaker]["features"]
                lines.append(
                    f"| {split_label} / {speaker_label} | "
                    f"{_value(features['duration_sec'], 'p50'):.2f}초 | "
                    f"{_value(features['rms_mean'], 'p50'):.4f} | "
                    f"{_value(features['zcr_mean'], 'p50'):.4f} | "
                    f"{_value(features['spectral_centroid_mean'], 'p50'):.1f}Hz | "
                    f"{100 * _value(features['silence_ratio_abs_le_0_01'], 'p50'):.2f}% |"
                )
        lines.append("")
        lines.append(
            "- 각 화자의 발화를 통화 단위로 연결하고 최대 15초를 사용해 MFCC(음색을 요약하는 음향 특성)/RMS(평균 음량 크기)/ZCR(신호 부호 변화율)/spectral(주파수 관련) 53차원 스키마를 EDA(탐색적 데이터 분석)용 NumPy(수치 계산 라이브러리) 구현으로 계산했습니다. 주파수 단위 Hz는 헤르츠(초당 진동 횟수)입니다. "
            "모델 파이프라인의 librosa(음성 분석 라이브러리) 특징과 수치 동일성을 보장하지 않으며, 표본 결과는 인과적 증거(원인이라고 확정할 근거)가 아닙니다."
        )
    recommendations = [
        "1. M1은 성별별 발화량과 음향 특징을 함께 통제해 길이 기반 shortcut(본질적인 신호 대신 우연한 쉬운 단서를 학습하는 현상) 여부를 점검합니다.",
        "2. M2는 0.5–1초 발화, 겹침 구간, 단일 화자 비중이 큰 통화를 고정 오류 슬라이스(특정 조건으로 나눈 오류 분석 구간)로 둡니다.",
        "3. M3는 희소 레이블과 공출현 조합을 보존하는 multilabel stratification(여러 레이블 조합의 비율을 유지하는 분할 방식)을 유지합니다.",
        "4. 현재 Train/Validation 식별자 중복은 0건이지만 전사 해시 중복 1건은 원문을 노출하지 않는 별도 수동 감사 대상으로 둡니다.",
        "5. 본 보고서는 기술통계이며 상관관계를 원인으로 해석하지 않습니다.",
    ]
    recommendation_number = 7
    if deep is not None:
        from .eda_deep_report import build_deep_recommendations, build_deep_section

        lines.append("")
        lines.extend(build_deep_section(deep, 7))
        recommendation_number = 8
        recommendations = build_deep_recommendations(deep)
    lines.extend(
        [
            "",
            f"## {recommendation_number}. 권고사항",
            "",
            *recommendations,
            "",
            "## 재현 방법",
            "",
            "```bash",
            "python scripts/eda_dcc_dataset.py",
            *(["python scripts/eda_deep_dive.py  # 심화 분석 절(7장) 생성"] if deep is not None else []),
            "```",
            "",
            "개발용 smoke run은 `--json-max-train`, `--json-max-val`, "
            "`--waveform-sample-train`, `--waveform-sample-val`로 범위를 제한합니다.",
            "",
        ]
    )
    return "\n".join(lines)
