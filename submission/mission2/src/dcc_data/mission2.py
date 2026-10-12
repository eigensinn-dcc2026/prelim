"""Mission 2 공통 유틸리티: 밀리초 단위 시간을 프레임 인덱스로 바꾼다."""

from __future__ import annotations

from numbers import Real


def milliseconds_to_frame(time_ms: Real, sample_rate: int) -> int:
    """밀리초를 프레임 인덱스로 바꾼다. 반올림은 파이썬의 ties-to-even 방식이다.

    ``round``를 일부러 쓴다. 정확히 반 프레임인 값은 가장 가까운 짝수로 올림/내림되어,
    모든 통화에서 자르는 경계가 항상 똑같이 정해진다.
    """

    if not isinstance(time_ms, Real) or isinstance(time_ms, bool):
        raise TypeError("time_ms must be numeric")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise TypeError("sample_rate must be an integer")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return int(round(time_ms / 1000.0 * sample_rate))
