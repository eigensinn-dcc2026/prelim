import numpy as np
import torch

SYMPTOMS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]


def pos_weight_from_labels(Y, scale=0.5, lo=1.0, hi=6.0):
    """ROBERTA-003 방식: clip(scale * sqrt(neg/pos), lo, hi)"""
    pos = np.clip(Y.sum(0), 1, None)
    neg = len(Y) - pos
    return torch.tensor(np.clip(scale * np.sqrt(neg / pos), lo, hi), dtype=torch.float32)


class SoftMacroF1:
    """soft macro-F1 손실 = 1 - mean_k F1_k
    beta > 0     : p -> sigmoid(beta*(p-0.5)). 0.5 경계 근처 샘플에만 기울기 집중 (제안 4)
    momentum > 0 : 배치 간 TP/FP/FN 을 EMA 로 누적(detach) -> 소수 클래스 기울기 잡음 완화. 작은 배치에서 권장
    """
    def __init__(self, beta=0.0, momentum=0.0, eps=1e-7):
        self.beta, self.momentum, self.eps, self.hist = beta, momentum, eps, None

    def reset(self):
        self.hist = None

    def __call__(self, p, y):
        p, y = p.float(), y.float()
        if self.beta > 0:
            p = torch.sigmoid(self.beta * (p - 0.5))
        tp, fp, fn = (p * y).sum(0), (p * (1 - y)).sum(0), ((1 - p) * y).sum(0)
        if self.momentum > 0:
            cur = torch.stack([tp, fp, fn]).detach()
            self.hist = cur if self.hist is None else self.momentum * self.hist + (1 - self.momentum) * cur
            tp, fp, fn = tp + self.hist[0], fp + self.hist[1], fn + self.hist[2]
        f1 = 2 * tp / (2 * tp + fp + fn + self.eps)
        return 1.0 - f1.mean()


def weighted_bce_on_probs(p, t, pos_weight=None, eps=1e-6):
    """확률 입력 가중 BCE. t 는 0/1 또는 soft target."""
    p, t = p.float().clamp(eps, 1 - eps), t.float()
    w = pos_weight.to(p.device) if pos_weight is not None else 1.0
    return -(w * t * p.log() + (1 - t) * (1 - p).log()).mean()


def macro_f1_at_half(probs, Y):
    """threshold 0.5 고정. 반환: (macro, per_class, all-zero 예측 비율)"""
    from sklearn.metrics import f1_score
    pred = (probs >= 0.5).astype(int)
    per = f1_score(Y.astype(int), pred, average=None, zero_division=0)
    return float(per.mean()), per, float((pred.sum(1) == 0).mean())
