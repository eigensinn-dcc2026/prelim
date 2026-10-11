"""구조화 헤드 수식 검증 (CPU, 수 초). python test_head.py"""
import torch, torch.nn.functional as F
from structured_head import StructuredSetHead, StructuredLoss

torch.manual_seed(0); B, H, K = 5, 7, 9
h = torch.randn(B, H); y = (torch.rand(B, K) < 0.3).float(); y[y.sum(1) == 0, 0] = 1

# 1) W=0, B1 -> sigmoid 와 동일
head = StructuredSetHead(H, K, exclude_empty=False, use_pairwise=False).eval()
logP, marg = head(h); s = head.unary(h)
assert torch.allclose(marg, torch.sigmoid(s), atol=1e-5), "B1/W=0 != sigmoid"
crit = StructuredLoss(head, None, lambda_set=1, lambda_bce=0, lambda_f1=0)
ce = crit(logP, marg, y); bce = F.binary_cross_entropy_with_logits(s, y)
assert torch.allclose(ce, bce, atol=1e-5), "set CE != BCE"

# 2) B2 -> p/(1-prod(1-p))
head2 = StructuredSetHead(H, K, exclude_empty=True, use_pairwise=False).eval(); head2.unary = head.unary
_, m2 = head2(h); p = torch.sigmoid(s)
assert torch.allclose(m2, p / (1 - (1 - p).prod(1, keepdim=True)), atol=1e-5), "B2 renorm 불일치"

# 3) pairwise 항 brute-force 일치
head3 = StructuredSetHead(H, K, exclude_empty=False, use_pairwise=True).eval()
with torch.no_grad(): head3.W_raw.normal_()
W = torch.triu(head3.W_raw, 1); W = W + W.T
bf = torch.stack([sum(W[i, j] * yy[i] * yy[j] for i in range(K) for j in range(i + 1, K)) for yy in head3.Y])
assert torch.allclose(head3.pair_term(), bf, atol=1e-4), "pairwise 불일치"

# 4) soft CE 가 hard 라벨에서 일반 CE 로 수렴
crit3 = StructuredLoss(head3, None, lambda_set=1, lambda_bce=0, lambda_f1=0)
logP3, m3 = head3(h)
assert torch.allclose(crit3(logP3, m3, y), crit3(logP3, m3, y, t_soft=y), atol=1e-3), "soft CE != hard CE"
print("all tests passed")
