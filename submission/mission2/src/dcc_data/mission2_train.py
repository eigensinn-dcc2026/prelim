"""Mission 2 기준선(L1) 모델: 음향 통계 위의 GBM.

분할은 항상 두 가지를 계산해 기록하며, 하나만 쓰지 않는다.

- call_id 단위 분할(``GroupShuffleSplit``): 실제로 의미가 있는 분할이다. 같은 통화의 발화가 양쪽에
  나타나지 않으므로 모델이 "이 통화의 목소리 = 이 화자"를 외울 수 없다.
- 단순 무작위 분할: 순전히 진단용으로 기록한다. 두 정확도의 차이는 단순한 분할을 썼다면 화자를 외우는
  누수가 얼마나 생기는지를 보여 주는 증거이며, 이 숫자는 모델을 고르는 용도가 아니라 팀 발표 자료에
  바로 쓰려는 것이다.

모델 저장은 일부러 ``save_model``/``load_model`` 뒤에 격리해 두어, 디스크에 쓰는 형식(지금은 pickle)을
나중에 학습·추론 코드를 건드리지 않고 바꿀 수 있게 했다. 대회 제출 형식은 ``.pt``/``.pth``/``.ckpt``를
요구하는데 pickle로 저장한 GBM은 이에 해당하지 않으며, 그 불일치는 여기서 풀 문제가 아니라 나중의
패키징 결정 사항이다.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .mission2_features import FEATURE_COLUMNS


def _build_model(seed: int) -> Any:
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=seed, verbosity=-1)
    except ImportError:
        return HistGradientBoostingClassifier(random_state=seed)


@dataclass
class BaselineResult:
    model: Any
    feature_columns: tuple[str, ...]
    group_split_accuracy: float
    random_split_accuracy: float
    majority_class_accuracy: float
    majority_class: int
    confusion_matrix: list[list[int]]
    feature_importance_top20: list[tuple[str, float]]
    n_train_group: int
    n_val_group: int
    train_call_ids: int
    val_call_ids: int


def make_group_split(
    df: pd.DataFrame, *, seed: int, test_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """call_id 단위 80/20 분할의 (train_idx, val_idx)를 돌려준다.

        ``df``의 행 순서, ``seed``, ``test_size``가 같으면 항상 같은 결과이다. 학습 밖의 호출자(예: 같은
        특징 parquet을 다시 읽는 오류 분석 스크립트)가 이 함수를 직접 불러 어떤 행이 제외되었는지를 정확히
        재현할 수 있다. GroupShuffleSplit 호출을 복사해 ``fit_baseline``과 어긋날 위험을 피할 수 있다.
    """

    if df["call_id"].nunique() < 2:
        raise ValueError("need at least 2 distinct call_id values for GroupShuffleSplit")
    groups = df["call_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    # GroupShuffleSplit은 len(X)와 groups만 쓰고 X/y의 값은 쓰지 않으므로,
    # 값싼 자리표시 배열을 넘겨도 실제 X, y를 넘기는 것과 똑같다.
    placeholder = np.zeros(len(df))
    train_idx, val_idx = next(splitter.split(placeholder, placeholder, groups))
    return train_idx, val_idx


def fit_baseline(df: pd.DataFrame, *, seed: int = 42, test_size: float = 0.2) -> BaselineResult:
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")

    X = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    y = df["speaker"].to_numpy(dtype=int)
    groups = df["call_id"].to_numpy()

    majority_class = int(round(y.mean()))
    majority_class_accuracy = float(max(y.mean(), 1 - y.mean()))

    train_idx, val_idx = make_group_split(df, seed=seed, test_size=test_size)

    train_ids = set(groups[train_idx])
    val_ids = set(groups[val_idx])
    overlap = train_ids & val_ids
    if overlap:
        # 이런 일은 절대 있어서는 안 된다. GroupShuffleSplit은 서로 겹치지 않는
        # 그룹을 보장한다. 테스트에서만이 아니라 여기서도 단언하는 이유는, 바로 이 지점의 누수가
        # 가장 오해를 부르는 실패 유형이기 때문이다: 검증 정확도는
        # 훌륭해 보이지만 쓸모가 없게 된다.
        raise AssertionError(f"GroupShuffleSplit leaked {len(overlap)} call_id(s) across train/val")

    group_model = _build_model(seed)
    group_model.fit(X[train_idx], y[train_idx])
    group_predictions = group_model.predict(X[val_idx])
    group_split_accuracy = float(accuracy_score(y[val_idx], group_predictions))
    cm = confusion_matrix(y[val_idx], group_predictions, labels=[0, 1]).tolist()

    # 진단용일 뿐이다: 누수 격차를 재기 위한 일반(그룹 없는) 분할.
    # 이 모델과 분할은 다른 어떤 용도로도 쓰지 않는다.
    rx_train, rx_val, ry_train, ry_val = train_test_split(X, y, test_size=test_size, random_state=seed)
    random_model = _build_model(seed)
    random_model.fit(rx_train, ry_train)
    random_split_accuracy = float(accuracy_score(ry_val, random_model.predict(rx_val)))

    importances = getattr(group_model, "feature_importances_", None)
    if importances is None:
        top20: list[tuple[str, float]] = []
    else:
        order = np.argsort(importances)[::-1][:20]
        top20 = [(FEATURE_COLUMNS[i], float(importances[i])) for i in order]

    return BaselineResult(
        model=group_model,
        feature_columns=FEATURE_COLUMNS,
        group_split_accuracy=group_split_accuracy,
        random_split_accuracy=random_split_accuracy,
        majority_class_accuracy=majority_class_accuracy,
        majority_class=majority_class,
        confusion_matrix=cm,
        feature_importance_top20=top20,
        n_train_group=len(train_idx),
        n_val_group=len(val_idx),
        train_call_ids=len(train_ids),
        val_call_ids=len(val_ids),
    )


def save_model(path: str | Path, model: Any, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump({"model": model, "meta": meta}, stream)


def load_model(path: str | Path) -> tuple[Any, dict]:
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    return payload["model"], payload["meta"]


def build_experiment_log(result: BaselineResult, *, seed: int, n_sampled_calls: int) -> dict:
    """M2-EXP-001 양식에 맞춘다. ``verdict``는 실제 실행 때 채우도록 비워 둔다."""

    return {
        "실험 ID": "M2-EXP-001",
        "가설": "발화 조각의 음향 통계 feature와 duration만으로 speaker를 유의미하게 분류할 수 있다",
        "baseline": 0.520,
        "변경 변수": "없음 (첫 baseline 수립)",
        "고정 변수": f"통화 {n_sampled_calls}건 샘플, seed={seed}, LightGBM(또는 HistGradientBoosting) 기본 파라미터",
        "평가 지표": "Accuracy",
        "validation": "call_id GroupShuffleSplit 8:2",
        "채택 기준": "GroupSplit accuracy >= 0.70",
        "기각 기준": "GroupSplit accuracy < 0.60 (-> L2 딥러닝 경로로 조기 전환)",
        "verdict": None,
        "measured": {
            "group_split_accuracy": result.group_split_accuracy,
            "random_split_accuracy": result.random_split_accuracy,
            "leakage_gap_random_minus_group": result.random_split_accuracy - result.group_split_accuracy,
            "majority_class_accuracy_this_sample": result.majority_class_accuracy,
            "n_train_utterances": result.n_train_group,
            "n_val_utterances": result.n_val_group,
            "n_train_calls": result.train_call_ids,
            "n_val_calls": result.val_call_ids,
            "confusion_matrix_val_group_split": result.confusion_matrix,
            "feature_importance_top20": result.feature_importance_top20,
        },
    }
