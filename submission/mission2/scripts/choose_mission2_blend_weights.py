"""모델들의 발화별 P(신고자)를 섞는 비율을 비교한다 (최종 비율 CNN 0.4 / WavLM 0.3 / 0.3 을 고른 도구).

규칙: 비율은 떼어 둔 Training 발화 20,000개(internal)에서만 고르고(0.1 간격으로 모든 조합), 공식 Validation 은 고른 비율로 한 번만 잰다.
판정은 섞은 확률 >= 0.5 (기준 고정).

사용법:
  python scripts/choose_mission2_blend_weights.py internal --models w1=wavlm1_internal.csv w2=wavlm2_internal.csv cnn=cnn_internal.csv
  python scripts/choose_mission2_blend_weights.py official --models w1=... w2=... cnn=... --weights 0.3 0.3 0.4   (--models 순서와 같은 순서)
CSV 의 확률 열은 p1 (WavLM, scripts/predict_mission2_wavlm.py) 또는 p1_cnn (CNN, scripts/predict_mission2_cnn_probs.py).
정답은 speaker 열이 있는 첫 파일에서 가져온다. 파일들의 행이 서로 하나라도 안 맞으면 멈춘다.
"""
import argparse
import itertools
import sys

import numpy as np
import pandas as pd

KEYS = ["audio file name", "startAt", "endAt"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["internal", "official"])
    parser.add_argument("--models", nargs="+", required=True, help="이름=파일 형식")
    parser.add_argument("--weights", type=float, nargs="+", default=None)
    parser.add_argument("--step", type=float, default=0.1)
    args = parser.parse_args()

    names, merged, truth = [], None, None
    for item in args.models:
        name, path = item.split("=", 1)
        frame = pd.read_csv(path)
        column = "p1" if "p1" in frame.columns else "p1_cnn"
        part = frame[KEYS + (["speaker"] if "speaker" in frame.columns else [])].copy()
        part[name] = frame[column]
        if truth is not None and "speaker" in part:
            part = part.drop(columns=["speaker"])
        merged = part if merged is None else merged.merge(part, on=KEYS)
        truth = truth if truth is not None else ("speaker" in part)
        names.append(name)
    sizes = [len(pd.read_csv(i.split("=", 1)[1], usecols=[0])) for i in args.models]
    print("행 수:", dict(zip(names, sizes)), "합친:", len(merged))
    if not all(s == len(merged) for s in sizes):
        sys.exit("행이 서로 안 맞아요. 중단.")
    y = merged["speaker"].to_numpy().astype(int)
    p = merged[names].to_numpy()

    def acc(w):
        return (((p @ np.asarray(w)) >= 0.5).astype(int) == y).mean() * 100

    for i, n in enumerate(names):
        print(f"  {n} 단독: {acc(np.eye(len(names))[i]):.3f}%")
    print(f"  전부 같은 비율: {acc(np.ones(len(names)) / len(names)):.3f}%")
    if args.mode == "internal":
        units = round(1 / args.step)
        grid = [tuple(c / units for c in comb) for comb in itertools.product(range(units + 1), repeat=len(names)) if sum(comb) == units]
        scored = sorted(((acc(w), w) for w in grid), reverse=True)
        print("순서:", names)
        for s, w in scored[:10]:
            print(f"  {w}: {s:.3f}%")
        print(f"최고에서 0.05%p 이내인 비율 {sum(1 for s, _ in scored if s >= scored[0][0] - 0.05)}개 / 격자 {len(grid)}개")
    else:
        if not args.weights or len(args.weights) != len(names):
            sys.exit("official 에서는 모델 개수만큼 --weights 가 필요해요.")
        print(f"공식 Validation, 비율 {dict(zip(names, args.weights))}: {acc(args.weights):.4f}%")


if __name__ == "__main__":
    main()
