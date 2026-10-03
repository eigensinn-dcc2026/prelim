"""
Mission 3 TF-IDF + Logistic Regression baseline.

역할
1) Training JSON에서 utterances[].text만 읽기
2) 목표 9개 증상 라벨 정리
3) JSON 샘플 단위 다중라벨 층화 train/validation 분할
4) TF-IDF + One-vs-Rest Logistic Regression 학습
5) Macro F1 / 증상별 F1 계산
6) 추론에 사용할 joblib 체크포인트 저장
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except ImportError:
    MultilabelStratifiedShuffleSplit = None


# 2026 DCC 공식 규정: Decision Threshold는 모든 클래스에서 0.5 고정
DECISION_THRESHOLD = 0.5

# Mission 3 평가 대상 9개 증상
TARGET_SYMPTOMS = [
    "고열",
    "구토",
    "두통",
    "복통",
    "어지러움",
    "열상",
    "오심",
    "전신쇠약",
    "호흡곤란",
]


def clean_text(value: Any) -> str:
    """문장의 줄바꿈/연속 공백을 한 칸으로 정리합니다."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()



def transcript_from_json(data: dict[str, Any]) -> str:
    """JSON의 기존 utterances 순서를 유지하며 text만 이어 붙입니다."""
    utterances = data.get("utterances", [])
    if not isinstance(utterances, list):
        return ""

    valid_utterances = [item for item in utterances if isinstance(item, dict)]
    texts = [clean_text(item.get("text", "")) for item in valid_utterances]
    return " ".join(text for text in texts if text)


def _normalize_symptoms(values: Any) -> list[str]:
    """symptom 필드를 문자열 목록으로 안전하게 바꿉니다."""
    if isinstance(values, list):
        return [value.strip() for value in values if isinstance(value, str) and value.strip()]

    # 혹시 단일 문자열로 들어온 비정상 자료가 있어도 한 개 라벨로 처리
    if isinstance(values, str) and values.strip():
        return [values.strip()]

    return []


def filter_target_symptoms(values: Any) -> list[str]:
    """전체 symptom 중 평가 대상 9개만 공식 순서대로 남깁니다."""
    value_set = set(_normalize_symptoms(values))
    return [label for label in TARGET_SYMPTOMS if label in value_set]


def read_one_json(json_path: str | Path) -> dict[str, Any]:
    """JSON 한 파일을 학습용 DataFrame의 한 행으로 바꿉니다."""
    json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("JSON 최상위 구조가 객체(dict)가 아닙니다.")

    all_symptoms = _normalize_symptoms(data.get("symptom", []))
    target_symptoms = filter_target_symptoms(all_symptoms)
    non_target_symptoms = [
        label for label in all_symptoms if label not in TARGET_SYMPTOMS
    ]

    return {
        "label_file_name": json_path.name,
        "text": transcript_from_json(data),
        "symptoms": target_symptoms,
        "non_target_symptoms": non_target_symptoms,
    }


def load_training_dataframe(label_dir: str | Path) -> pd.DataFrame:
    """
    Training 폴더 아래 모든 JSON을 읽어 DataFrame으로 만듭니다.

    읽기에 실패한 파일은 DataFrame.attrs["load_failures"]에 기록합니다.
    """
    label_dir = Path(label_dir)

    if not label_dir.exists():
        raise FileNotFoundError(f"라벨 폴더를 찾을 수 없습니다: {label_dir}")

    json_paths = sorted(label_dir.rglob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"JSON 파일을 찾지 못했습니다: {label_dir}")

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for json_path in json_paths:
        try:
            records.append(read_one_json(json_path))
        except Exception as error:
            failures.append(
                {
                    "file": json_path.name,
                    "error": str(error)[:200],
                }
            )

    if not records:
        raise RuntimeError(
            f"JSON {len(json_paths)}개를 찾았지만 정상적으로 읽은 파일이 없습니다."
        )

    df = pd.DataFrame(records)
    df.attrs["load_failures"] = failures
    df.attrs["json_file_count"] = len(json_paths)

    return df


def make_dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """노트북에서 사용할 데이터 기본 점검 결과를 만듭니다."""
    required_columns = {"text", "symptoms"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame에 필요한 열이 없습니다: {sorted(missing)}")

    target_label_counts = {
        label: int(df["symptoms"].apply(lambda labels: label in labels).sum())
        for label in TARGET_SYMPTOMS
    }

    non_target_counter: Counter[str] = Counter()
    if "non_target_symptoms" in df.columns:
        for labels in df["non_target_symptoms"]:
            if isinstance(labels, list):
                non_target_counter.update(labels)

    return {
        "sample_count": int(len(df)),
        "empty_text_count": int(df["text"].eq("").sum()),
        "zero_target_label_count": int(df["symptoms"].map(len).eq(0).sum()),
        "load_failure_count": int(len(df.attrs.get("load_failures", []))),
        "target_label_counts": target_label_counts,
        "non_target_label_counts": dict(non_target_counter.most_common()),
    }


def _make_split(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
    mlb: MultiLabelBinarizer,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """JSON 샘플 단위로 다중라벨 분포를 최대한 유지하며 분할합니다."""
    if MultilabelStratifiedShuffleSplit is None:
        raise ImportError(
            "iterative-stratification 패키지가 필요합니다. "
            "requirements.txt를 설치해 주세요."
        )
    if not (0 < test_size < 1):
        raise ValueError("test_size는 0과 1 사이여야 합니다.")
    if len(df) < 2:
        raise ValueError("train/validation 분리를 하려면 최소 2개 샘플이 필요합니다.")

    y = mlb.transform(df["symptoms"])
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    train_idx, valid_idx = next(splitter.split(np.arange(len(df)), y))
    return df.iloc[train_idx].copy(), df.iloc[valid_idx].copy()


def _labels_to_string(labels: list[str]) -> str:
    """CSV에서 ['두통', '복통'] 같은 모양으로 저장하기 위한 함수입니다."""
    return str(list(labels))


def train_baseline(
    label_dir: str | Path,
    output_dir: str | Path,
    threshold: float = 0.5,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Mission 3 TF-IDF 베이스라인을 학습하고 결과 파일을 저장합니다.

    반환값:
    {
        "metrics": dict,
        "per_label": DataFrame,
        "checkpoint_path": str,
        ...
    }
    """
    if not np.isclose(threshold, DECISION_THRESHOLD):
        raise ValueError(
            "2026 DCC 공식 규정상 Decision Threshold는 0.5로 고정해야 합니다. "
            f"입력값: {threshold}"
        )
    threshold = DECISION_THRESHOLD

    random.seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training JSON 읽기
    df = load_training_dataframe(label_dir)
    dataset_summary = make_dataset_summary(df)

    # 2. 정답을 9자리 0/1 배열로 바꿀 준비
    mlb = MultiLabelBinarizer(classes=TARGET_SYMPTOMS)
    mlb.fit([TARGET_SYMPTOMS])

    # 3. JSON 샘플 단위 train/validation 분리
    train_df, valid_df = _make_split(
        df=df, test_size=test_size, seed=seed, mlb=mlb
    )

    if train_df.empty or valid_df.empty:
        raise RuntimeError("train 또는 validation 데이터가 비어 있습니다.")

    y_train = mlb.transform(train_df["symptoms"])
    y_valid = mlb.transform(valid_df["symptoms"])

    # 4. 전사 텍스트를 TF-IDF 숫자로 변환
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(train_df["text"])
    X_valid = vectorizer.transform(valid_df["text"])

    # 5. 9개 증상을 각각 예/아니오로 판단하는 로지스틱 회귀 학습
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

    # 6. Validation 예측
    valid_probability = classifier.predict_proba(X_valid)
    valid_prediction = (valid_probability >= threshold).astype(np.int8)

    # 7. 평가
    precision, recall, f1_each, support = precision_recall_fscore_support(
        y_valid,
        valid_prediction,
        average=None,
        zero_division=0,
    )

    per_label = pd.DataFrame(
        {
            "symptom": TARGET_SYMPTOMS,
            "support": support.astype(int),
            "precision": precision.astype(float),
            "recall": recall.astype(float),
            "f1": f1_each.astype(float),
        }
    )

    macro_f1 = float(
        f1_score(y_valid, valid_prediction, average="macro", zero_division=0)
    )
    micro_f1 = float(
        f1_score(y_valid, valid_prediction, average="micro", zero_division=0)
    )

    # 8. 추론에 필요한 모델 묶음 저장
    checkpoint = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "target_symptoms": TARGET_SYMPTOMS,
        "threshold": float(threshold),
        "seed": int(seed),
        "model_name": "tfidf_word_1_2gram_ovr_logistic_regression",
    }

    checkpoint_path = output_dir / "mission3_tfidf_logreg.joblib"
    joblib.dump(checkpoint, checkpoint_path)

    # 9. 노트북/PR에서 사용할 지표 저장
    metrics = {
        "train_sample_count": int(len(train_df)),
        "validation_sample_count": int(len(valid_df)),
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "threshold": float(threshold),
        "test_size": float(test_size),
        "seed": int(seed),
        "total_sample_count": int(len(df)),
        "load_failure_count": int(dataset_summary["load_failure_count"]),
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    per_label_path = output_dir / "per_label_metrics.csv"
    per_label.to_csv(per_label_path, index=False, encoding="utf-8-sig")

    # 10. Validation 예측 결과 저장
    true_label_lists = mlb.inverse_transform(y_valid)
    predicted_label_lists = mlb.inverse_transform(valid_prediction)

    validation_predictions = pd.DataFrame(
        {
            "label file name": valid_df["label_file_name"].tolist(),
            "true symptom": [
                _labels_to_string(list(labels)) for labels in true_label_lists
            ],
            "predicted symptom": [
                _labels_to_string(list(labels)) for labels in predicted_label_lists
            ],
        }
    )

    validation_prediction_path = output_dir / "validation_predictions.csv"
    validation_predictions.to_csv(
        validation_prediction_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Train samples: {len(train_df):,}")
    print(f"Validation samples: {len(valid_df):,}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Checkpoint: {checkpoint_path}")

    return {
        "metrics": metrics,
        "per_label": per_label,
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "per_label_path": str(per_label_path),
        "validation_prediction_path": str(validation_prediction_path),
        "dataset_summary": dataset_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    """터미널에서 학습할 때 사용할 옵션을 정의합니다."""
    parser = argparse.ArgumentParser(
        description="Mission 3 TF-IDF + Logistic Regression baseline training"
    )
    parser.add_argument(
        "--label_dir",
        required=True,
        help="서울 Training 라벨 JSON 폴더",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="학습 결과를 저장할 폴더",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DECISION_THRESHOLD,
        choices=[DECISION_THRESHOLD],
        help="2026 DCC 공식 고정 Decision Threshold (0.5만 허용)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="내부 validation 비율 (기본값: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="재현성을 위한 random seed (기본값: 42)",
    )
    return parser


def main() -> None:
    """명령줄에서 학습 전체 과정을 실행합니다."""
    args = build_parser().parse_args()

    result = train_baseline(
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        test_size=args.test_size,
        seed=args.seed,
    )

    print("\n증상별 F1")
    print(
        result["per_label"][
            ["symptom", "support", "precision", "recall", "f1"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
