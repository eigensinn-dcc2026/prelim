#!/usr/bin/env python3
"""미세조정한 WavLM 체크포인트(best.pt)로 발화별 P(신고자)를 계산해 CSV 로 쓴다.

``--split internal``: 떼어 둔 Training 통화(CNN 과 같은 통화, 같은 ``--split-seed``)에서 고정 표본 20,000발화.
  best.pt 와 마지막 step 모델 중 하나를 고르고, CNN 과 섞는 비율을 정할 때 쓴다. 공식 Validation 은 건드리지 않는다.
``--split official``: 공식 Validation 폴더. 평가 전용이며 최종 숫자를 한 번 보고할 때만 쓴다.

출력 CSV 열: ``audio file name,startAt,endAt,speaker,call_id,utt_idx,p1`` (``speaker`` 는 제공 라벨로, 채점에만 쓴다).
기본은 float32 추론이라 숫자가 GPU 에 좌우되지 않는다. ``--amp`` 면 float16 으로 돈다(학습 노트북은 --amp 로 실행).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from dcc_data.mission2_train import make_group_split  # noqa: E402
from dcc_data.mission2_wavlm import WavLMRoleClassifier  # noqa: E402
from train_mission2_wavlm import CHECKPOINT_FORMAT, UtteranceWindows, build_items  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True, help="best.pt from train_mission2_wavlm.py")
    parser.add_argument("--split", choices=["internal", "official"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--utts", type=int, default=0, help="internal only: score this many held-out utterances (0 = all)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def whole_number_or_value(value: float):
    return int(value) if float(value).is_integer() else value


def main() -> int:
    args = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise SystemExit(f"{args.ckpt} is not a {CHECKPOINT_FORMAT} checkpoint")

    if args.split == "official":
        items = build_items(args.manifest, args.data_root, validation=True)
    else:
        training = build_items(args.manifest, args.data_root)
        _, held_out = make_group_split(training, seed=checkpoint["split_seed"], test_size=checkpoint["test_size"])
        items = training.iloc[held_out]
        if args.utts and args.utts < len(items):
            generator = np.random.default_rng(checkpoint["split_seed"] + 1)
            items = items.iloc[np.sort(generator.choice(len(items), args.utts, replace=False))]
    items = items.reset_index(drop=True)
    print(f"{args.split}: {len(items)} utterances / {items['call_id'].nunique()} calls, device={device}", flush=True)

    model = WavLMRoleClassifier.from_config(checkpoint["backbone_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.float().to(device).eval()
    loader = torch.utils.data.DataLoader(
        UtteranceWindows(items, train=False, seed=0), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device == "cuda",
    )
    probabilities, started = [], time.time()
    with torch.no_grad():
        for done, (waveforms, _) in enumerate(loader, start=1):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp and device == "cuda"):
                logits = model(waveforms.to(device, non_blocking=True)).float()
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            if done % 100 == 0:
                print(f"  {min(done * args.batch_size, len(items))}/{len(items)}", flush=True)
    elapsed = time.time() - started
    p1 = np.concatenate(probabilities)

    result = pd.DataFrame(
        {
            "audio file name": [Path(path).name for path in items["wav_path"]],
            "startAt": items["start_at"].map(whole_number_or_value),
            "endAt": items["end_at"].map(whole_number_or_value),
            "speaker": items["speaker"], "call_id": items["call_id"], "utt_idx": items["utt_idx"], "p1": p1,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, float_format="%.8f")
    predicted = (p1 >= 0.5).astype(int)
    truth = items["speaker"].to_numpy()
    print(
        f"{args.split} accuracy at 0.5: {(predicted == truth).mean():.4%} ({int((predicted == truth).sum())}/{len(truth)}), "
        f"recall_0 {(predicted[truth == 0] == 0).mean():.4f} recall_1 {(predicted[truth == 1] == 1).mean():.4f}; "
        f"{elapsed / 60:.1f} min, {elapsed / len(items) * 1000:.1f} ms per utterance, batch {args.batch_size}",
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
