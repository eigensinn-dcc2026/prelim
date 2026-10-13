#!/usr/bin/env python3
"""Mission 2 (화자 분류: 대원 0 / 신고자 1) 제출용 추론 진입점.

    python inference.py --audio_dir {wav folder} --label_dir {json folder} \
        --ckpt_path checkpoints/m2_final_ensemble.pt --output ./outputs/mission2.csv

구현은 scripts/mission2_final_inference.py 에 있다. CNN 3개 앙상블과 미세조정한 WavLM-large 2개의 P(신고자)를
0.4 : 0.3 : 0.3 으로 가중 평균한 값이 0.5 이상이면 신고자(1), 아니면 대원(0)이다.
발화마다 그 발화의 startAt/endAt 구간 소리만 쓴다. GPU 가 있으면 GPU 를 쓰고, 없으면 CPU 로 돈다(느림).
출력 CSV 첫 줄은 출제문제 11쪽의 컬럼 이름 그대로 `audio file name,startAt,endAt,speaker` 다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from mission2_final_inference import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
