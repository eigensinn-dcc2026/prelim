"""Mission 3 record-level text Dataset and fixed multi-label encoding."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from .guards import ensure_split
from .manifest import ManifestRow


SYMPTOM_CLASSES = (
    "고열",
    "구토",
    "두통",
    "복통",
    "어지러움",
    "열상",
    "오심",
    "전신쇠약",
    "호흡곤란",
)
SYMPTOM_TO_INDEX = {label: index for index, label in enumerate(SYMPTOM_CLASSES)}


def encode_symptoms(symptoms: Iterable[str]) -> np.ndarray:
    """Encode only the nine evaluated labels; ignore all other labels."""

    target = np.zeros(len(SYMPTOM_CLASSES), dtype=np.float32)
    for symptom in symptoms:
        index = SYMPTOM_TO_INDEX.get(symptom)
        if index is not None:
            target[index] = 1.0
    return target


class Mission3Dataset:
    """Return record transcript text and a nine-dimensional multi-hot target."""

    def __init__(
        self,
        rows: list[ManifestRow],
        data_root: str | Path,
        split: str,
        *,
        text_separator: str = "\n",
    ) -> None:
        self.rows = ensure_split(rows, split)
        self.data_root = Path(data_root)
        self.split = split
        self.text_separator = text_separator

    def __len__(self) -> int:
        return len(self.rows)

    def _read_allowed_fields(self, row: ManifestRow) -> tuple[list[str], list[str]]:
        json_path = self.data_root / row.json_path
        with json_path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)

        utterances = document.get("utterances")
        symptoms = document.get("symptom")
        if not isinstance(utterances, list):
            raise ValueError(f"utterances must be a list: {json_path}")
        if not isinstance(symptoms, list) or not all(
            isinstance(item, str) for item in symptoms
        ):
            raise ValueError(f"symptom must be a string list: {json_path}")

        texts: list[str] = []
        for utterance in utterances:
            if not isinstance(utterance, dict) or not isinstance(
                utterance.get("text"), str
            ):
                raise ValueError(f"utterances[].text must be a string: {json_path}")
            texts.append(utterance["text"])
        return texts, symptoms

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        texts, symptoms = self._read_allowed_fields(row)
        return {
            "features": {"text": self.text_separator.join(texts)},
            "target": encode_symptoms(symptoms),
            "split": row.split,
            "stem": row.stem,
        }

    def unknown_symptom_counts(self) -> Counter[str]:
        """Count non-evaluation labels for later EDA without treating them as targets."""

        counts: Counter[str] = Counter()
        for row in self.rows:
            _, symptoms = self._read_allowed_fields(row)
            counts.update(
                symptom for symptom in symptoms if symptom not in SYMPTOM_TO_INDEX
            )
        return counts
