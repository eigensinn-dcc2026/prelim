"""
Mission 3 TF-IDF baseline vs character TF-IDF comparison.

한 번의 실행에서 같은 train/validation split을 사용해
Baseline(word TF-IDF)과 Exp 1(character TF-IDF)을 공정하게 비교합니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from mission3_baseline import (
    TARGET_SYMPTOMS,
    _make_split,
    load_training_dataframe,
)


def evaluate_variant(
    name: str,
    vectorizer: TfidfVectorizer,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    threshold: float,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """같은 split에서 TF-IDF 표현 방식 하나만 바꿔 성능을 평가합니다."""

    # TF-IDF는 train에만 fit하고 validation에는 transform만 합니다.
    X_train = vectorizer.fit_transform(train_df["text"])
    X_valid = vectorizer.transform(valid_df["text"])

    # Baseline과 Exp 1에서 완전히 같은 분류기를 사용합니다.
    classifier = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        ),
        n_jobs=1,
    )
    classifier.fit(X_train, y_train)

    probability = classifier.predict_proba(X_valid)
    prediction = (probability >= threshold).astype(np.int8)

    precision, recall, f1_each, support = precision_recall_fscore_support(
        y_valid,
        prediction,
        average=None,
        zero_division=0,
    )

    predicted_positive = prediction.sum(axis=0)
    fp = ((y_valid == 0) & (prediction == 1)).sum(axis=0)
    fn = ((y_valid == 1) & (prediction == 0)).sum(axis=0)

    per_label = pd.DataFrame(
        {
            "experiment": name,
            "symptom": TARGET_SYMPTOMS,
            "actual_positive": support.astype(int),
            "predicted_positive": predicted_positive.astype(int),
            "fp": fp.astype(int),
            "fn": fn.astype(int),
            "precision": precision.astype(float),
            "recall": recall.astype(float),
            "f1": f1_each.astype(float),
        }
    )

    metrics = {
        "experiment": name,
        "macro_f1": float(
            f1_score(y_valid, prediction, average="macro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(
                y_valid, prediction, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_valid, prediction, average="macro", zero_division=0
            )
        ),
        "micro_f1": float(
            f1_score(y_valid, prediction, average="micro", zero_division=0)
        ),
        "subset_accuracy": float(accuracy_score(y_valid, prediction)),
    }

    return metrics, per_label


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Mission 3 word TF-IDF baseline and character TF-IDF"
    )
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 원본 Training JSON을 한 번만 읽습니다.
    df = load_training_dataframe(args.label_dir)

    # 2. 두 실험이 정확히 같은 정답 표현과 split을 사용하도록 고정합니다.
    mlb = MultiLabelBinarizer(classes=TARGET_SYMPTOMS)
    mlb.fit([TARGET_SYMPTOMS])

    train_df, valid_df = _make_split(
        df=df,
        test_size=args.test_size,
        seed=args.seed,
        mlb=mlb,
    )

    y_train = mlb.transform(train_df["symptoms"])
    y_valid = mlb.transform(valid_df["symptoms"])

    # 3. Baseline: word TF-IDF
    baseline_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )

    # 4. Exp 1: character TF-IDF
    # 다른 조건은 Baseline과 모두 동일합니다.
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )

    baseline_metrics, baseline_per_label = evaluate_variant(
        name="Baseline_word_tfidf",
        vectorizer=baseline_vectorizer,
        train_df=train_df,
        valid_df=valid_df,
        y_train=y_train,
        y_valid=y_valid,
        threshold=args.threshold,
        seed=args.seed,
    )

    exp_metrics, exp_per_label = evaluate_variant(
        name="Exp1_char_tfidf",
        vectorizer=char_vectorizer,
        train_df=train_df,
        valid_df=valid_df,
        y_train=y_train,
        y_valid=y_valid,
        threshold=args.threshold,
        seed=args.seed,
    )

    # 5. 전체 지표 비교
    comparison = pd.DataFrame([baseline_metrics, exp_metrics])

    delta_macro_f1 = (
        exp_metrics["macro_f1"] - baseline_metrics["macro_f1"]
    )

    # 0.005 이하 차이는 단일 split에서 우연일 가능성을 고려해 보류합니다.
    if delta_macro_f1 > 0.005:
        decision = "ADOPT_PROVISIONAL"
        reason = (
            "Character TF-IDF의 Macro F1이 baseline보다 0.005 초과 상승했습니다. "
            "다른 seed에서도 같은 방향인지 확인 후 최종 ADOPT를 판단합니다."
        )
    elif delta_macro_f1 < -0.005:
        decision = "REJECT"
        reason = (
            "Character TF-IDF의 Macro F1이 baseline보다 0.005 초과 하락했습니다."
        )
    else:
        decision = "INCONCLUSIVE"
        reason = (
            "Macro F1 차이가 ±0.005 이내라 단일 validation 결과만으로 "
            "개선 여부를 단정하기 어렵습니다."
        )

    # 6. 증상별 F1 변화 계산
    baseline_label = baseline_per_label.set_index("symptom")
    exp_label = exp_per_label.set_index("symptom")

    per_label_comparison = pd.DataFrame(
        {
            "symptom": TARGET_SYMPTOMS,
            "baseline_f1": [
                float(baseline_label.loc[s, "f1"]) for s in TARGET_SYMPTOMS
            ],
            "exp1_f1": [
                float(exp_label.loc[s, "f1"]) for s in TARGET_SYMPTOMS
            ],
        }
    )
    per_label_comparison["f1_delta"] = (
        per_label_comparison["exp1_f1"]
        - per_label_comparison["baseline_f1"]
    )

    decision_payload = {
        "baseline_macro_f1": baseline_metrics["macro_f1"],
        "exp1_macro_f1": exp_metrics["macro_f1"],
        "macro_f1_delta": float(delta_macro_f1),
        "decision": decision,
        "reason": reason,
        "seed": args.seed,
        "threshold": args.threshold,
        "train_sample_count": int(len(train_df)),
        "validation_sample_count": int(len(valid_df)),
    }

    # 7. 결과 저장
    comparison.to_csv(
        output_dir / "m3_tfidf_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    per_label_comparison.to_csv(
        output_dir / "m3_tfidf_per_label_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (output_dir / "m3_tfidf_decision.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(decision_payload, file, ensure_ascii=False, indent=2)

    # 8. 사람이 바로 읽을 수 있게 출력
    print("\n===== Mission 3 실험 비교 =====")
    print(comparison.to_string(index=False))

    print("\n===== 증상별 F1 변화 =====")
    print(
        per_label_comparison.sort_values(
            "f1_delta", ascending=False
        ).to_string(index=False)
    )

    print("\n===== 판정 =====")
    print(f"Macro F1 변화: {delta_macro_f1:+.4f}")
    print(f"판정: {decision}")
    print(reason)


if __name__ == "__main__":
    main()
