#!/usr/bin/env python3
"""Mission 1 공식 Validation accuracy/recall의 부트스트랩 신뢰구간 계산.

------------------------------------------------------------------
부트스트랩이 뭔가요? (여기서 꼭 이해하고 넘어갈 점)
------------------------------------------------------------------
`eval_mission1_validation.py`가 준 숫자(예: LR accuracy 0.9511)는
"3,640건짜리 표본 하나"에서 나온 점추정치(point estimate)입니다. 만약
공식 Validation을 다른 3,640건으로 다시 받았다면 이 숫자가 정확히
0.9511이 나올 리는 없겠죠 — 조금 오르거나 내렸을 겁니다. 문제는
"그 흔들림이 어느 정도인지"를 알아야 0.9511이라는 숫자를 얼마나
믿어도 되는지 판단할 수 있다는 점입니다.

이론적으로 정확도(accuracy) 하나만 본다면 이항분포(binomial)
공식으로 표준오차를 바로 계산할 수 있습니다. 하지만
- recall_F, recall_M처럼 클래스별 지표도 같이 보고 싶고
- 나중에 Macro F1(Mission 3) 같은 더 복잡한 지표에도 같은 방법을
  재사용하고 싶다면

매번 다른 공식을 유도하는 대신, **컴퓨터로 "가상의 재표본"을 수천 번
만들어서 그 분포를 직접 관찰**하는 게 훨씬 범용적입니다. 이것이
부트스트랩(bootstrap)입니다.

절차:
  1) 지금 가진 3,640건을 "모집단"이라고 잠시 가정합니다.
  2) 거기서 3,640건을 "복원추출"(뽑은 걸 다시 넣고 또 뽑기)로
     다시 뽑습니다. 어떤 건 2번 뽑히고 어떤 건 0번 뽑힐 수 있습니다.
  3) 이 가상 표본에서 accuracy를 다시 계산합니다.
  4) 1~3을 10,000번 반복하면 accuracy가 "이 정도로 흔들리는 분포"를
     얻습니다.
  5) 이 분포의 2.5%~97.5% 지점을 잘라내면, 그게 95% 신뢰구간입니다.
     빈도주의에서 95% 신뢰구간은 "지금 구한 이 구간에 진짜 모수가
     들어 있을 확률이 95%"라는 뜻이 아닙니다. 같은 표집·구간 구성
     절차를 반복하면, 그렇게 만든 구간 중 약 95%가 고정된 모수
     (여기서는 공식 Validation이 대표하는 분포의 진짜 accuracy)를
     포함한다는 뜻입니다.

------------------------------------------------------------------
원본 row 데이터 없이도 가능한 이유
------------------------------------------------------------------
개별 3,640건의 raw feature/정답을 다시 불러오지 않아도 됩니다.
`eval_mission1_validation.py`가 이미 남긴 confusion matrix 4칸
(실제 F·예측 F / 실제 F·예측 M / 실제 M·예측 F / 실제 M·예측 M)
개수만 있으면, "그 칸의 개수만큼 해당 결과를 가진 가상의 행"을
재구성해서 복원추출할 수 있습니다. 원래 3,640개 행을 무작위 순서로
그대로 복원추출하는 것과 통계적으로 완전히 동일합니다(각 행이
어떤 순서였는지는 독립동일분포 가정 하에 의미가 없기 때문입니다).

사용법 (팀 저장소 루트에서 실행):
    python scripts/bootstrap_ci_mission1.py
    python scripts/bootstrap_ci_mission1.py --log reports/mission1/m1_val_eval.json --n-boot 20000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("reports/mission1/m1_val_eval.json"),
        help="eval_mission1_validation.py가 생성한 로그 파일",
    )
    parser.add_argument("--n-boot", type=int, default=10_000, help="부트스트랩 재표본 반복 횟수")
    parser.add_argument("--seed", type=int, default=42, help="재현성을 위한 random seed")
    parser.add_argument("--alpha", type=float, default=0.05, help="신뢰수준 1-alpha (기본 0.05 -> 95% CI)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/mission1/m1_val_bootstrap_ci.json"),
        help="결과를 저장할 경로",
    )
    return parser.parse_args()


def reconstruct_rows(confusion_matrix: list[list[int]], labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """confusion matrix 4칸을 실제 (정답, 예측) 행 배열로 복원한다.

    예: labels=["F","M"], matrix=[[1873,86],[99,1582]] 이면
      - (F,F) 1873행, (F,M) 86행, (M,F) 99행, (M,M) 1582행
    을 가진 배열을 만든다. 이렇게 만든 배열을 복원추출하는 것은
    "원본 3,640개 행을 그대로 복원추출하는 것"과 동일한 분포를 준다.
    """

    y_true: list[str] = []
    y_pred: list[str] = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            count = int(confusion_matrix[i][j])
            y_true.extend([true_label] * count)
            y_pred.extend([pred_label] * count)
    return np.asarray(y_true), np.asarray(y_pred)


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric_fn,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """metric_fn(y_true_sample, y_pred_sample) -> float 를 n_boot번 재계산한다."""

    n = len(y_true)
    boot_values = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        # 복원추출: 0..n-1 인덱스를 중복 허용해서 n개 뽑는다.
        idx = rng.integers(0, n, size=n)
        boot_values[b] = metric_fn(y_true[idx], y_pred[idx])
    return boot_values


def accuracy_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def recall_fn(label: str):
    def _inner(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mask = y_true == label
        if mask.sum() == 0:
            return float("nan")
        return float(np.mean(y_pred[mask] == label))

    return _inner


def percentile_ci(values: np.ndarray, alpha: float) -> tuple[float, float]:
    lower = float(np.nanpercentile(values, 100 * (alpha / 2)))
    upper = float(np.nanpercentile(values, 100 * (1 - alpha / 2)))
    return lower, upper


def summarize_model(name: str, measured: dict, *, n_boot: int, alpha: float, rng: np.random.Generator) -> dict:
    labels = measured["confusion_matrix_labels"]
    y_true, y_pred = reconstruct_rows(measured["confusion_matrix"], labels)

    point_accuracy = measured["accuracy"]
    boot_acc = bootstrap_metric(y_true, y_pred, metric_fn=accuracy_fn, n_boot=n_boot, rng=rng)
    acc_lo, acc_hi = percentile_ci(boot_acc, alpha)

    result = {
        "accuracy_point_estimate": point_accuracy,
        "accuracy_bootstrap_mean": float(np.mean(boot_acc)),
        "accuracy_bootstrap_std": float(np.std(boot_acc)),
        f"accuracy_ci_{int((1 - alpha) * 100)}": [acc_lo, acc_hi],
    }

    for label, key in (("F", "recall_F"), ("M", "recall_M")):
        boot_rec = bootstrap_metric(y_true, y_pred, metric_fn=recall_fn(label), n_boot=n_boot, rng=rng)
        lo, hi = percentile_ci(boot_rec, alpha)
        result[f"{key}_point_estimate"] = measured[key]
        result[f"{key}_bootstrap_mean"] = float(np.nanmean(boot_rec))
        result[f"{key}_ci_{int((1 - alpha) * 100)}"] = [lo, hi]

    print(f"\n=== {name} ===")
    print(f"accuracy: {point_accuracy:.4f}  (95% CI: [{acc_lo:.4f}, {acc_hi:.4f}])")
    print(
        f"recall_F: {measured['recall_F']:.4f}  "
        f"(95% CI: [{result['recall_F_ci_95'][0]:.4f}, {result['recall_F_ci_95'][1]:.4f}])"
    )
    print(
        f"recall_M: {measured['recall_M']:.4f}  "
        f"(95% CI: [{result['recall_M_ci_95'][0]:.4f}, {result['recall_M_ci_95'][1]:.4f}])"
    )
    return result


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    payload = json.loads(args.log.read_text(encoding="utf-8"))
    measured = payload["measured"]

    output = {
        "source_log": str(args.log),
        "n_validation": payload.get("n_validation"),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "alpha": args.alpha,
        "gbm": summarize_model("GBM", measured["gbm"], n_boot=args.n_boot, alpha=args.alpha, rng=rng),
        "lr": summarize_model("LR", measured["lr"], n_boot=args.n_boot, alpha=args.alpha, rng=rng),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
