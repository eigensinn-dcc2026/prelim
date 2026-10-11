import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import SoftMacroF1, weighted_bce_on_probs


def all_label_sets(K, exclude_empty):
    Y = torch.tensor(list(itertools.product([0, 1], repeat=K)), dtype=torch.float32)  # [2^K, K]
    return Y[1:] if exclude_empty else Y  # 행 0 = 공집합


class StructuredSetHead(nn.Module):
    """9개 라벨의 결합분포를 라벨 집합(511 또는 512개) 위 softmax 로 직접 모델링.
        logit(y) = y·s(h) + Σ_{i<j} y_i y_j W_ij
    use_pairwise=False & exclude_empty=False  -> 독립 sigmoid 헤드와 수학적으로 동일 (대조군)
    exclude_empty=True (B2)                   -> 지지집합에서 공집합 제거, 주변확률 자동 재정규화 (멘토 확인 후)
    """
    def __init__(self, hidden, K=9, exclude_empty=False, use_pairwise=True, dropout=0.1):
        super().__init__()
        self.K, self.exclude_empty, self.use_pairwise = K, exclude_empty, use_pairwise
        self.unary = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, K))
        self.W_raw = nn.Parameter(torch.zeros(K, K))  # 0 초기화 = 독립 모델에서 출발
        self.register_buffer("Y", all_label_sets(K, exclude_empty))            # [S, K]
        self.register_buffer("bits", 2 ** torch.arange(K - 1, -1, -1).float())  # 라벨벡터 -> 행 인덱스

    def pair_term(self):
        W = torch.triu(self.W_raw, diagonal=1)
        W = W + W.T                                    # 대칭, 대각 0
        return 0.5 * ((self.Y @ W) * self.Y).sum(-1)   # [S] = Σ_{i<j} y_i y_j W_ij

    def forward(self, h):
        # 헤드 전체를 fp32 로: autocast 안에서는 .float() 뒤의 matmul 도 다시 fp16 으로 내려감
        with torch.autocast(device_type=h.device.type, enabled=False):
            s = self.unary(h.float())                  # [B, K]
            logits = s @ self.Y.T                      # [B, S]
            if self.use_pairwise:
                logits = logits + self.pair_term()[None, :]
            logP = F.log_softmax(logits, dim=-1)
            marg = logP.exp() @ self.Y                 # [B, K] = P(y_i = 1)  -> 0.5 판정
        return logP, marg

    def set_index(self, y_hard):
        idx = (y_hard.float() @ self.bits).long()
        if self.exclude_empty:
            idx = idx - 1
            idx = torch.where(idx < 0, torch.full_like(idx, -100), idx)  # 공집합 라벨은 무시
        return idx


class StructuredLoss:
    """loss = λ_set·CE_set/K + λ_bce·BCE_pw(marg) + λ_f1·softF1(marg)
      CE_set : 라벨 집합 softmax 의 CE. t_soft 가 있으면 독립 Bernoulli 곱을 지지집합 안에서 정규화한
               q(y) 를 타깃으로 하는 soft CE (증류용). hard 라벨이면 일반 CE 와 동일.
      BCE_pw : 주변확률에 pos_weight BCE (기존 레시피 유지)
      softF1 : 주변확률에 soft macro-F1 (경계 이동)
    프리셋 A(최소 변경): λ_set=0, λ_bce=1,   λ_f1=1   -> W 는 주변확률을 통해서만 학습
    프리셋 B(결합 학습): λ_set=1, λ_bce=0.5, λ_f1=1
    """
    def __init__(self, head, pos_weight, lambda_set=1.0, lambda_bce=0.5, lambda_f1=1.0,
                 f1_beta=0.0, f1_momentum=0.0):
        self.head, self.pos_weight = head, pos_weight
        self.ls, self.lb, self.lf = lambda_set, lambda_bce, lambda_f1
        self.soft_f1 = SoftMacroF1(beta=f1_beta, momentum=f1_momentum)

    def __call__(self, logP, marg, y_hard, t_soft=None):
        Y, y_hard = self.head.Y, y_hard.float()
        loss = logP.new_zeros(())
        if self.ls > 0:
            if t_soft is None:
                idx = self.head.set_index(y_hard)
                if (idx >= 0).any():
                    loss = loss + self.ls * F.nll_loss(logP, idx, ignore_index=-100) / self.head.K
            else:
                t = t_soft.float().clamp(1e-6, 1 - 1e-6)
                logq = t.log() @ Y.T + (1 - t).log() @ (1 - Y).T   # [B, S]
                logq = F.log_softmax(logq, dim=-1)                   # 지지집합 안에서 정규화
                loss = loss + self.ls * (-(logq.exp() * logP).sum(-1).mean()) / self.head.K
        if self.lb > 0:
            t = y_hard if t_soft is None else t_soft
            loss = loss + self.lb * weighted_bce_on_probs(marg, t, self.pos_weight)
        if self.lf > 0:
            loss = loss + self.lf * self.soft_f1(marg, y_hard)
        return loss
