#!/usr/bin/env python3
"""Mission 2 (화자 분류: 대원 0 / 신고자 1) 제출용 추론 진입점.

    python inference.py --audio_dir {wav folder} --label_dir {json folder} \
        --ckpt_path checkpoints/m2_cnn_large_rerun_ensemble.pt --output ./outputs/mission2.csv

구현은 scripts/mission2_inference.py 에 있다 (학습 노트북의 공식 Validation 평가 셀도 이 파일로 실행한다).
발화 하나하나를 그 발화의 startAt/endAt 구간만 잘라 따로 추론하고, 라벨 JSON 에서는 startAt/endAt 만 읽는다.

출력 CSV 첫 줄은 출제문제 11쪽의 컬럼 이름 그대로 `audio file name,startAt,endAt,speaker` 로 쓴다.
scripts/mission2_inference.py 는 학습·검증에 쓴 코드 그대로 두기 위해 고치지 않았고(첫 줄을 `audio_file_name` 으로 씀),
여기서 다 쓴 CSV 의 첫 줄 이름만 바꾼다. 예측값과 행 순서는 그대로다.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from mission2_inference import main, parse_args  # noqa: E402

SCRIPT_HEADER = b"audio_file_name,startAt,endAt,speaker"
OFFICIAL_HEADER = b"audio file name,startAt,endAt,speaker"


def use_official_header(csv_path: Path) -> None:
    data = csv_path.read_bytes()
    if not data.startswith(SCRIPT_HEADER):
        raise SystemExit(f"{csv_path}: unexpected first line {data[:60]!r}")
    tmp_path = csv_path.with_name(csv_path.name + ".tmp")
    tmp_path.write_bytes(OFFICIAL_HEADER + data[len(SCRIPT_HEADER):])
    os.replace(tmp_path, csv_path)


if __name__ == "__main__":
    output = parse_args().output
    status = main()
    if status == 0:
        use_official_header(output)
    raise SystemExit(status)
