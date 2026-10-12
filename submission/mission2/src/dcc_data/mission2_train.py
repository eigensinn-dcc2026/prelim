"""Mission 2 baseline(L1) 모델: 음향 통계 위의 GBM.

분할은 한 가지가 아니라 항상 두 가지를 계산해서 기록한다.

- call_id 단위 분할(``GroupShuffleSplit``). 실제로 중요한 분할이다. 같은 통화의 발화가 양쪽에
  동시에 들어가지 않으므로, 모델이 "이 통화의 목소리 = 이 화자"를 외울 수 없다.
- 일반 무작위 분할. 순전히 진단용으로만 기록한다. 두 정확도의 차이는, 순진하게 나눴을 때
  화자 암기 누수가 얼마나 생길지를 보여 주는 증거다. 이 숫자는 모델을 고르려는 것이 아니라
  팀 발표 자료에 그대로 넣으려는 것이다.

모델 저장은 일부러 ``save_model``/``load_model`` 뒤에 분리해 두었다. 그래야 저장 형식(현재는
pickle)을 나중에 학습 코드나 추론 코드를 건드리지 않고 바꿀 수 있다. 대회의 제출 형식은
``.pt``/``.pth``/``.ckpt``를 요구하는데 pickle로 저장한 GBM은 그 형식이 아니다. 그 불일치는
나중에 정할 패키징 문제이고 여기서 풀 일이 아니다.
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

    ``df``의 행 순서, ``seed``, ``test_size``가 같으면 결과도 항상 같다. 그래서 학습 밖의 호출자
    (예: 같은 feature parquet을 다시 읽는 오류 분석 스크립트)도 이 함수를 직접 불러 어떤 행이
    검증용으로 빠졌는지 정확히 재현할 수 있다. GroupShuffleSplit 호출을 복제해 두었다가
    ``fit_baseline``과 어긋나는 일을 피한다.
    """

    if df["call_id"].nunique() < 2:
        raise ValueError("need at least 2 distinct call_id values for GroupShuffleSplit")
    groups = df["call_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    # GroupShuffleSplit은 len(X)와 groups만 쓰고 X, y의 값은 쓰지 않는다. 그래서 값싼
    # 자리표시 배열을 넘기는 것이 진짜 X, y를 넘기는 것과 정확히 같다.
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
        # 절대 일어나면 안 된다. GroupShuffleSplit은 그룹이 겹치지 않음을 보장한다. 테스트에서만이
        # 아니라 여기서도 검사하는 이유는, 바로 이 지점의 누수가 가장 오해하기 쉬운 실패이기
        # 때문이다. 검증 정확도는 아주 좋아 보이지만 아무 가치가 없다.
        raise AssertionError(f"GroupShuffleSplit leaked {len(overlap)} call_id(s) across train/val")

    group_model = _build_model(seed)
    group_model.fit(X[train_idx], y[train_idx])
    group_predictions = group_model.predict(X[val_idx])
    group_split_accuracy = float(accuracy_score(y[val_idx], group_predictions))
    cm = confusion_matrix(y[val_idx], group_predictions, labels=[0, 1]).tolist()

    # 진단용일 뿐이다. 누수 격차를 재려는 일반(그룹 아닌) 분할이다.
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

