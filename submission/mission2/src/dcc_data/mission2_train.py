"""미션 2 학습/검증 분할: 통화(call_id) 단위로 Training 을 80/20 으로 나눈다.

같은 통화의 발화는 반드시 한쪽에만 들어간다. 같은 통화가 양쪽에 섞이면 모델이 "이 통화의 목소리 = 이 화자"를 외워서
검증 정확도가 실제보다 높게 나오기 때문이다.

CNN 학습(scripts/train_mission2_cnn.py), WavLM 학습(scripts/train_mission2_wavlm.py), 확률 계산
(scripts/predict_mission2_wavlm.py)이 모두 이 함수를 같은 seed(42)·비율(0.2)로 불러서, 세 모델이 정확히 같은 5,840통화를 떼어 둔다.

(제출 폴더 정리 메모: 초기 L1 모델(LightGBM) 학습·저장 코드는 최종 모델에 쓰이지 않아 이 파일에서 뺐다.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def make_group_split(
    df: pd.DataFrame, *, seed: int, test_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """통화 단위 80/20 분할의 (학습 행 번호, 검증 행 번호)를 돌려준다.

    같은 ``df`` 행 순서, ``seed``, ``test_size`` 면 항상 같은 결과가 나온다. 그래서 학습이 아닌 스크립트(확률 계산 등)도
    이 함수를 직접 불러 어떤 행이 떼어졌는지 정확히 재현한다.
    """

    if df["call_id"].nunique() < 2:
        raise ValueError("need at least 2 distinct call_id values for GroupShuffleSplit")
    groups = df["call_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    # GroupShuffleSplit 은 X, y 의 값은 보지 않고 길이와 groups 만 쓴다. 그래서 0 으로 채운 배열을 넣어도 결과가 같다.
    placeholder = np.zeros(len(df))
    train_idx, val_idx = next(splitter.split(placeholder, placeholder, groups))
    return train_idx, val_idx
