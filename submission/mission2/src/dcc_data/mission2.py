"""미션 2 시간 단위 변환: 라벨 JSON 의 밀리초(startAt/endAt)를 음성 샘플 위치로 바꾼다.

라벨의 speaker 값은 0 = 대원(119 접수 요원), 1 = 신고자다.

(제출 폴더 정리 메모: 초기 실험용 데이터셋 클래스 ``Mission2Dataset`` 은 최종 모델에 쓰이지 않아 뺐다.)
"""

from __future__ import annotations

from numbers import Real


def milliseconds_to_frame(time_ms: Real, sample_rate: int) -> int:
    """밀리초를 샘플 위치(정수)로 바꾼다. 반올림은 파이썬 ``round``(정확히 .5 이면 짝수 쪽)를 쓴다.

    ``round`` 를 일부러 쓴다. 샘플 경계가 정확히 반인 경우에도 항상 같은 쪽으로 정해져 자르는 위치가 흔들리지 않는다.
    """

    if not isinstance(time_ms, Real) or isinstance(time_ms, bool):
        raise TypeError("time_ms must be numeric")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise TypeError("sample_rate must be an integer")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return int(round(time_ms / 1000.0 * sample_rate))
