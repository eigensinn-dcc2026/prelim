"""제안 2(B2) 우선순위 판정용 5분 진단. 현재 홀드아웃 확률로
  (1) 9개 전부 <0.5 인 통화 비율  (2) 그 통화들의 argmax 정밀도
  (3) 공집합 제외 재정규화 p_i' = p_i / (1 - Π_j(1-p_j)) 적용 시 macro F1
      = B2 헤드(W=0, unary 불변 가정)의 효과 추정치
  *** 진단 전용. 이 재정규화를 inference.py 에 넣으면 threshold 우회 후처리 (금지). ***
"""
import argparse, numpy as np
from sklearn.metrics import f1_score
from losses import SYMPTOMS

ap = argparse.ArgumentParser(); ap.add_argument("--probs", required=True); ap.add_argument("--labels", required=True)
a = ap.parse_args()
P, Y = np.load(a.probs).astype(np.float64), np.load(a.labels).astype(int)
f1 = lambda p: f1_score(Y, (p >= 0.5).astype(int), average=None, zero_division=0)

zero = (P >= 0.5).sum(1) == 0
print(f"all-zero 예측 비율: {zero.mean():.2%} ({zero.sum()}/{len(P)})")
if zero.any():
    am = P[zero].argmax(1)
    print(f"  argmax 정밀도: {Y[zero][np.arange(zero.sum()), am].mean():.3f}  (해당 통화 라벨 양성 평균 {Y[zero].sum(1).mean():.2f}개)")
    print("  argmax 분포: " + ", ".join(f"{SYMPTOMS[i]}:{(am == i).sum()}" for i in range(9) if (am == i).any()))
base = f1(P)
R = P / np.clip(1 - np.prod(1 - P, axis=1, keepdims=True), 1e-6, None)
rn = f1(R)
print(f"\nmacro F1  기준 {base.mean():.4f} -> 재정규화 {rn.mean():.4f}  (Δ {rn.mean() - base.mean():+.4f}), "
      f"재정규화 후 all-zero {((R >= 0.5).sum(1) == 0).mean():.2%}")
for s, b, r in zip(SYMPTOMS, base, rn):
    print(f"  {s:5s} {b:.3f} -> {r:.3f} ({r - b:+.3f})")
print("\n판정: Δ ≥ +0.004 이면 B2 를 멘토 질문 1순위로. Δ < +0.002 이면 B2 는 건너뛰고 B1(W 만) 진행.")
