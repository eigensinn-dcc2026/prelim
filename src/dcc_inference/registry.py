"""Mission dispatch and official CSV contract for the root ``inference.py``.

Contract for every mission module ``dcc_inference.mission{N}``::

    def predict(audio_dir: Path, label_dir: Path, ckpt_path: Path) -> pd.DataFrame

The returned frame must have exactly ``MISSION_COLUMNS[N]`` (same names, same
order) and one row per input item. Mission modules must not write files.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Callable

import pandas as pd

# Official PDF p.11 column names (brackets in the PDF are notation only).
MISSION_COLUMNS: dict[int, tuple[str, ...]] = {
    1: ("audio file name", "gender"),
    2: ("audio file name", "startAt", "endAt", "speaker"),
    3: ("label file name", "symptom"),
}

# Official PDF: "UTF-8 CSV". No BOM so a plain utf-8 reader sees the exact header.
CSV_ENCODING = "utf-8"

_MISSION_IN_NAME = re.compile(r"mission\s*_?([123])", re.IGNORECASE)

PredictFn = Callable[[Path, Path, Path], pd.DataFrame]


def resolve_mission(output: Path, explicit: int | None = None) -> int:
    """Return the mission number from ``--mission`` or the output file name."""
    from_name = None
    match = _MISSION_IN_NAME.search(Path(output).name)
    if match:
        from_name = int(match.group(1))
    if explicit is not None:
        if explicit not in MISSION_COLUMNS:
            raise ValueError(f"unknown mission: {explicit}")
        if from_name is not None and from_name != explicit:
            raise ValueError(
                f"--mission {explicit} conflicts with output name {Path(output).name}"
            )
        return explicit
    if from_name is None:
        raise ValueError(
            f"cannot tell the mission from --output {output}; "
            "name it mission{1,2,3}.csv or pass --mission"
        )
    return from_name


def load_predict(mission: int) -> PredictFn:
    """Import ``dcc_inference.mission{N}.predict``."""
    if mission not in MISSION_COLUMNS:
        raise ValueError(f"unknown mission: {mission}")
    module = importlib.import_module(f"dcc_inference.mission{mission}")
    predict = getattr(module, "predict", None)
    if not callable(predict):
        raise AttributeError(f"dcc_inference.mission{mission} has no predict()")
    return predict


def validate_frame(mission: int, frame: pd.DataFrame) -> None:
    """Fail loudly if a module breaks the official CSV contract."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"mission {mission} predict() must return a DataFrame")
    expected = MISSION_COLUMNS[mission]
    got = tuple(frame.columns)
    if got != expected:
        raise ValueError(f"mission {mission} columns {got} != official {expected}")
    if frame.empty:
        raise ValueError(f"mission {mission} predict() returned no rows")
    if frame.isna().any().any():
        raise ValueError(f"mission {mission} output has missing values")