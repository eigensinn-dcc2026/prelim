#!/usr/bin/env python3
"""최종 미션 2 모델을 체크포인트 파일 하나로 묶는다: CNN 3개 앙상블 + 미세조정한 WavLM-large 2개 + 섞는 비율.

채점 명령은 ``--ckpt_path`` 를 하나만 받으므로 추론에 필요한 것을 모두 이 파일에 담는다: CNN 앙상블 체크포인트(그대로, 학습 설정 기록 포함),
WavLM 체크포인트 2개(모델 구조 설정 + float16 가중치, 그래서 채점 때 아무것도 내려받지 않음), 섞는 비율.

    python scripts/pack_mission2_final_ensemble.py --cnn-ckpt cnn.pt --wavlm-ckpt w1.pt w2.pt --weights 0.4 0.3 0.3 --output final.pt

비율(CNN 0.4, WavLM 0.3 + 0.3)은 떼어 둔 *Training* 발화 20,000개로 골랐고 공식 Validation 으로 고르지 않았다.
판정은 ``섞은 확률 >= 0.5`` (기준 고정)이고 다른 기준은 없다. 원본 체크포인트 3개의 sha256 도 함께 기록한다.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

WAVLM_FORMAT = "dcc2026-mission2-wavlm-v1"
FINAL_FORMAT = "dcc2026-mission2-final-v1"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cnn-ckpt", type=Path, required=True)
    parser.add_argument("--wavlm-ckpt", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=None, help="cnn wavlm1 wavlm2 ... (sum 1); default 0.4 0.3 0.3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    weights = args.weights or [0.4] + [0.6 / len(args.wavlm_ckpt)] * len(args.wavlm_ckpt)
    if len(weights) != 1 + len(args.wavlm_ckpt) or abs(sum(weights) - 1.0) > 1e-9:
        sys.exit(f"need {1 + len(args.wavlm_ckpt)} weights that sum to 1, got {weights}")

    cnn = torch.load(args.cnn_ckpt, map_location="cpu", weights_only=False)
    members = []
    for path in args.wavlm_ckpt:
        member = torch.load(path, map_location="cpu", weights_only=False)
        if member.get("format") != WAVLM_FORMAT:
            sys.exit(f"{path} is not a {WAVLM_FORMAT} checkpoint")
        member["state_dict"] = {k: (v.half() if v.is_floating_point() else v) for k, v in member["state_dict"].items()}
        members.append(member)
    if len({(m["split_seed"], m["test_size"], m["pretrained"]) for m in members}) != 1:
        sys.exit("the WavLM members were not trained with the same split/pretrained model")

    final = {
        "format": FINAL_FORMAT,
        "blend": {
            "weights": {"cnn": weights[0], "wavlm": weights[1:]},
            "threshold": 0.5,
            "weights_chosen_on": "20,000 held-out Training utterances (not the official Validation)",
        },
        "cnn": cnn,
        "wavlm": members,
        "sources_sha256": {
            "cnn": sha256_of(args.cnn_ckpt),
            "wavlm": [sha256_of(path) for path in args.wavlm_ckpt],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp")
    torch.save(final, tmp)
    tmp.replace(args.output)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.0f} MB), sha256 {sha256_of(args.output)}")
    print("sources:", final["sources_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
