"""
Mission 3 inference script.

학습된 mission3_tfidf_logreg.joblib을 불러와
JSON의 utterances[].text만 이용해 증상을 예측하고 mission3.csv를 만듭니다.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np


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


def load_inputs(label_dir: str | Path) -> tuple[list[str], list[str]]:
    """
    추론할 JSON 파일 이름과 전사 텍스트를 같은 순서로 읽습니다.

    추론에서는 입력 JSON 수와 출력 CSV 행 수가 같아야 하므로
    JSON 하나라도 읽기 실패하면 조용히 건너뛰지 않고 오류를 냅니다.
    """
    label_dir = Path(label_dir)

    if not label_dir.exists():
        raise FileNotFoundError(f"라벨 폴더를 찾을 수 없습니다: {label_dir}")

    json_paths = sorted(label_dir.rglob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"JSON 파일을 찾지 못했습니다: {label_dir}")

    file_names: list[str] = []
    transcripts: list[str] = []

    for json_path in json_paths:
        try:
            with json_path.open("r", encoding="utf-8-sig") as file:
                data = json.load(file)
        except Exception as error:
            raise RuntimeError(
                f"JSON 읽기 실패: {json_path} / {error}"
            ) from error

        if not isinstance(data, dict):
            raise ValueError(
                f"JSON 최상위 구조가 객체(dict)가 아닙니다: {json_path}"
            )

        file_names.append(json_path.name)
        transcripts.append(transcript_from_json(data))

    return file_names, transcripts


def predict(
    checkpoint: dict[str, Any],
    transcripts: list[str],
) -> list[list[str]]:
    """각 대화에서 예측된 증상 이름 목록을 반환합니다."""
    required_keys = {"vectorizer", "classifier", "target_symptoms"}
    missing = required_keys - set(checkpoint.keys())
    if missing:
        raise KeyError(
            f"체크포인트에 필요한 항목이 없습니다: {sorted(missing)}"
        )

    vectorizer = checkpoint["vectorizer"]
    classifier = checkpoint["classifier"]
    target_symptoms = list(checkpoint["target_symptoms"])
    threshold = float(checkpoint.get("threshold", 0.5))

    if not transcripts:
        return []

    # 학습 때 만든 TF-IDF 단어 사전을 그대로 사용합니다.
    features = vectorizer.transform(transcripts)

    # 각 JSON마다 9개 증상의 확률을 구합니다.
    probabilities = np.asarray(classifier.predict_proba(features))

    if probabilities.ndim != 2:
        raise RuntimeError(
            f"예측 확률 배열 모양이 예상과 다릅니다: {probabilities.shape}"
        )

    if probabilities.shape[1] != len(target_symptoms):
        raise RuntimeError(
            "체크포인트의 증상 수와 모델 출력 차원이 일치하지 않습니다. "
            f"증상 {len(target_symptoms)}개 / 모델 출력 {probabilities.shape[1]}개"
        )

    # 기본값 0.5 이상인 증상만 1로 선택합니다.
    binary = (probabilities >= threshold).astype(np.int8)

    predictions: list[list[str]] = []
    for row in binary:
        labels = [
            target_symptoms[index]
            for index, value in enumerate(row)
            if value == 1
        ]
        predictions.append(labels)

    return predictions


def write_csv(
    file_names: list[str],
    predictions: list[list[str]],
    output_path: str | Path,
) -> None:
    """Mission 3 형식의 CSV 파일을 저장합니다."""
    if len(file_names) != len(predictions):
        raise ValueError(
            "파일명 개수와 예측 결과 개수가 다릅니다: "
            f"{len(file_names)} != {len(predictions)}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["label file name", "symptom"],
        )
        writer.writeheader()

        for file_name, labels in zip(file_names, predictions):
            writer.writerow(
                {
                    "label file name": file_name,
                    # 기존 베이스라인 설계와 맞춰 ['두통', '복통'] 형태로 저장
                    "symptom": str(labels),
                }
            )


def build_parser() -> argparse.ArgumentParser:
    """주최 측 실행 구조를 고려한 네 가지 인자를 정의합니다."""
    parser = argparse.ArgumentParser(description="Mission 3 inference")

    # Mission 3 TF-IDF 모델은 WAV를 읽지 않지만
    # 공통 실행 명령 형식을 맞추기 위해 인자로 받습니다.
    parser.add_argument(
        "--audio_dir",
        required=True,
        help="공통 실행 규칙 호환용 WAV 폴더 (Mission 3에서는 사용하지 않음)",
    )
    parser.add_argument(
        "--label_dir",
        required=True,
        help="추론 대상 JSON 폴더",
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="학습된 mission3_tfidf_logreg.joblib 경로",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="결과 mission3.csv 저장 경로",
    )

    return parser


def main() -> None:
    """JSON 읽기 → 모델 불러오기 → 예측 → CSV 저장을 한 번에 실행합니다."""
    args = build_parser().parse_args()

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")

    checkpoint = joblib.load(ckpt_path)

    file_names, transcripts = load_inputs(args.label_dir)
    predictions = predict(checkpoint, transcripts)
    write_csv(file_names, predictions, args.output)

    print(
        f"Mission 3 CSV 저장 완료: {args.output} "
        f"({len(file_names):,}개)"
    )


if __name__ == "__main__":
    main()
