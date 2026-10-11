"""제안 1: 8B 한국어 LLM 교사 (LoRA + 9-label 헤드, non-pad mean pooling) -> 5-fold OOF 확률
  A100 40GB : bf16, --batch 4 --accum 4 --max_len 1024, grad checkpointing   (메모리 부족: --load_in_4bit)
  세션 병렬  : --folds 0 (게이트) / --folds 1,2 / --folds 3,4  -> 끝나면 --merge
  게이트     : fold0 1 epoch 후 fold-val macro F1 < --gate 이면 중단
  KFold seed 를 train_structured.py 와 같게 두면 fold 가 동일해져 fold 단위 비교가 가능.
"""
import argparse, glob, json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from losses import SYMPTOMS, pos_weight_from_labels, SoftMacroF1, macro_f1_at_half


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv", required=True); p.add_argument("--text_col", default="text")
    p.add_argument("--model", default="kakaocorp/kanana-1.5-8b-base")
    p.add_argument("--revision", default=None, help="HF 커밋 고정 (재현성)")
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4); p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lora_r", type=int, default=32); p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_lr", type=float, default=1e-4); p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=2); p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--lambda_f1", type=float, default=1.0)
    p.add_argument("--f1_momentum", type=float, default=0.9, help="배치 4 라서 EMA 누적 기본 ON")
    p.add_argument("--n_folds", type=int, default=5); p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gate", type=float, default=0.635, help="0 이면 게이트 없음")
    p.add_argument("--load_in_4bit", action="store_true"); p.add_argument("--append_eos", action="store_true")
    p.add_argument("--save_adapters", action="store_true")
    p.add_argument("--predict_csv", default=None, help="옵션: 공식 Val 등 (fold 별 pred 저장, 나중에 평균)")
    p.add_argument("--merge", action="store_true", help="fold 별 oof 파일 합쳐 oof_probs.npy 생성 후 종료")
    p.add_argument("--out_dir", default="runs/teacher")
    return p.parse_args()


class IdsDS(Dataset):
    def __init__(self, ids, Y): self.ids, self.Y = ids, Y
    def __len__(self): return len(self.ids)
    def __getitem__(self, i): return self.ids[i], self.Y[i]


class BucketBatchSampler(Sampler):
    """길이순 정렬 후 배치 단위로 섞어 패딩 낭비 최소화 (1024 토큰에서 속도 1.5~2배)"""
    def __init__(self, lengths, batch, seed):
        self.lengths, self.batch, self.rng = np.asarray(lengths), batch, np.random.default_rng(seed)
    def __iter__(self):
        idx = np.argsort(self.lengths + self.rng.random(len(self.lengths)) * 8)
        batches = [idx[i:i + self.batch].tolist() for i in range(0, len(idx), self.batch)]
        self.rng.shuffle(batches)
        return iter(batches)
    def __len__(self): return (len(self.lengths) + self.batch - 1) // self.batch


def make_collate(pad_id, bos_id=None):
    def collate(batch):
        ids, ys = zip(*batch); L = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), L), pad_id, dtype=torch.long)
        attn = torch.zeros((len(ids), L), dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, :len(x)] = torch.tensor(x); attn[i, :len(x)] = 1
        # 풀링 마스크: BOS 는 어텐션 싱크(hidden norm 비정상)라 평균에서 제외. 어텐션에는 그대로 둠
        pool = attn.clone()
        if bos_id is not None:
            pool[:, 0] *= (input_ids[:, 0] != bos_id).long()
        return {"input_ids": input_ids, "attention_mask": attn, "pool_mask": pool, "y": torch.tensor(np.stack(ys))}
    return collate


class LLMClassifier(nn.Module):
    def __init__(self, backbone, hidden, K):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, K))

    def forward(self, input_ids, attention_mask, pool_mask=None):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B,T,H]
        m = (attention_mask if pool_mask is None else pool_mask).unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)   # non-pad mean pooling (last-token 풀링 금지)
        with torch.autocast("cuda", enabled=False):
            return self.head(pooled.float())                 # logits fp32


def build_model(args, device):
    kw = dict(torch_dtype=torch.bfloat16)
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kw.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                  bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True), device_map={"": 0})
    backbone = AutoModel.from_pretrained(args.model, revision=args.revision, **kw)   # CausalLM 아님: lm_head(~0.5B) 불필요
    if args.load_in_4bit:
        from peft import prepare_model_for_kbit_training
        backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=True)
    else:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        backbone.enable_input_require_grads()
    backbone = get_peft_model(backbone, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model = LLMClassifier(backbone, backbone.config.hidden_size, len(SYMPTOMS))
    if not args.load_in_4bit:
        model.backbone.to(device)
    model.head.to(device)
    model.backbone.print_trainable_parameters()
    return model


@torch.no_grad()
def predict(model, dl, device):
    model.eval(); out = []
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["pool_mask"].to(device))
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def run_fold(args, ids, lengths, Y, tr, va, k, device, pred_ids=None):
    out = args.out_dir
    model = build_model(args, device)
    pw = pos_weight_from_labels(Y[tr]).to(device)
    soft_f1 = SoftMacroF1(momentum=args.f1_momentum)
    col = make_collate(args.pad_id, args.bos_id)
    dl_tr = DataLoader(IdsDS([ids[i] for i in tr], Y[tr]), batch_sampler=BucketBatchSampler(lengths[tr], args.batch, args.seed + k), collate_fn=col)
    dl_va = DataLoader(IdsDS([ids[i] for i in va], Y[va]), batch_size=args.batch * 2, collate_fn=col)
    lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": args.lora_lr},
                             {"params": model.head.parameters(), "lr": args.head_lr}], weight_decay=0.0)
    total = (len(dl_tr) * args.epochs + args.accum - 1) // args.accum
    sch = get_cosine_schedule_with_warmup(opt, int(total * args.warmup), total)
    log, probs = [], None
    for ep in range(args.epochs):
        model.train(); soft_f1.reset(); t0 = time.time(); run = 0.0
        for i, b in enumerate(dl_tr):
            b = {kk: v.to(device) for kk, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(b["input_ids"], b["attention_mask"], b["pool_mask"])
            loss = (F.binary_cross_entropy_with_logits(logits, b["y"], pos_weight=pw)
                    + args.lambda_f1 * soft_f1(torch.sigmoid(logits), b["y"]))
            (loss / args.accum).backward(); run += loss.item()
            if (i + 1) % args.accum == 0 or (i + 1) == len(dl_tr):
                torch.nn.utils.clip_grad_norm_(lora_params + list(model.head.parameters()), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); sch.step()
            if i % 200 == 0:
                print(f"  fold{k} ep{ep+1} {i}/{len(dl_tr)} loss {run/(i+1):.4f} {(time.time()-t0)/60:.0f}min", flush=True)
        probs = predict(model, dl_va, device)
        f1, per, zero_rate = macro_f1_at_half(probs, Y[va])
        log.append({"epoch": ep + 1, "loss": run / len(dl_tr), "macro_f1": f1, "per_class": per.tolist(),
                    "pred_allzero_rate": zero_rate, "min": (time.time() - t0) / 60})
        print(f"[fold{k}] ep{ep+1} F1 {f1:.4f} allzero {zero_rate:.2%} ({(time.time()-t0)/60:.0f}min) | "
              + " ".join(f"{s}:{v:.3f}" for s, v in zip(SYMPTOMS, per)), flush=True)
        json.dump(log, open(os.path.join(out, f"fold{k}_log.json"), "w"), ensure_ascii=False, indent=1)
        if k == 0 and ep == 0 and args.gate > 0 and f1 < args.gate:
            print(f"GATE FAIL: fold0 ep1 F1 {f1:.4f} < {args.gate}. 교사 포기."); return None
    np.save(os.path.join(out, f"oof_fold{k}.npy"), probs)        # 마지막 epoch 확률 = OOF (선택 편향 없음)
    np.save(os.path.join(out, f"oof_idx_fold{k}.npy"), va)
    if args.save_adapters:
        model.backbone.save_pretrained(os.path.join(out, f"fold{k}_adapter"))
        torch.save(model.head.state_dict(), os.path.join(out, f"fold{k}_head.pt"))
    if pred_ids is not None:
        dl_p = DataLoader(IdsDS(pred_ids, np.zeros((len(pred_ids), len(SYMPTOMS)), np.float32)), batch_size=args.batch * 2, collate_fn=col)
        np.save(os.path.join(out, f"pred_fold{k}.npy"), predict(model, dl_p, device))
    del model, opt; torch.cuda.empty_cache()
    return f1


def merge(args):
    df = pd.read_csv(args.data_csv); Y = df[SYMPTOMS].to_numpy().astype(np.float32)
    oof = np.full_like(Y, np.nan)
    for f in sorted(glob.glob(os.path.join(args.out_dir, "oof_fold*.npy"))):
        k = os.path.basename(f)[len("oof_fold"):-4]
        oof[np.load(os.path.join(args.out_dir, f"oof_idx_fold{k}.npy"))] = np.load(f)
    assert not np.isnan(oof).any(), "빠진 fold 있음"
    np.save(os.path.join(args.out_dir, "oof_probs.npy"), oof)
    f1, per, zero = macro_f1_at_half(oof, Y)
    print(f"teacher OOF macro F1 = {f1:.4f}  allzero {zero:.2%} | " + " ".join(f"{s}:{v:.3f}" for s, v in zip(SYMPTOMS, per)))


def main():
    args = get_args(); os.makedirs(args.out_dir, exist_ok=True)
    if args.merge:
        return merge(args)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    df = pd.read_csv(args.data_csv)
    texts = df[args.text_col].astype(str).tolist(); Y = df[SYMPTOMS].to_numpy().astype(np.float32)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    args.pad_id = tok.pad_token_id
    args.bos_id = tok.bos_token_id
    cut = args.max_len - int(args.append_eos)

    def tokenize(tt):
        ids = tok(tt, truncation=True, max_length=cut, add_special_tokens=True)["input_ids"]
        return [x + [tok.eos_token_id] for x in ids] if args.append_eos else ids

    ids = tokenize(texts); lengths = np.array([len(x) for x in ids])
    print(f"token len p50 {np.percentile(lengths, 50):.0f} p90 {np.percentile(lengths, 90):.0f} "
          f"p99 {np.percentile(lengths, 99):.0f} | truncated {(lengths >= cut).mean():.2%}")
    pred_ids = tokenize(pd.read_csv(args.predict_csv)[args.text_col].astype(str).tolist()) if args.predict_csv else None
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    splits = list(MultilabelStratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed).split(np.zeros((len(Y), 1)), Y))
    scores, gate_fail = {}, False
    for k in [int(x) for x in args.folds.split(",")]:
        tr, va = splits[k]
        done = os.path.join(args.out_dir, f"oof_fold{k}.npy")
        if os.path.exists(done):  # 러너 재시도: 끝난 fold 는 건너뜀
            assert (np.load(os.path.join(args.out_dir, f"oof_idx_fold{k}.npy")) == va).all(), "fold 분할 불일치"
            scores[k] = macro_f1_at_half(np.load(done), Y[va])[0]
            print(f"[fold{k}] 완료분 재사용 F1 {scores[k]:.4f}", flush=True)
            continue
        f1 = run_fold(args, ids, lengths, Y, tr, va, k, "cuda", pred_ids)
        if f1 is None:
            gate_fail = True
            break
        scores[k] = f1
    # dcc-lab 러너가 읽는 metrics.json. 게이트 탈락도 정상 종료(판정 결과)로 기록
    logs = {k: json.load(open(os.path.join(args.out_dir, f"fold{k}_log.json"), encoding="utf-8"))
            for k in [int(x) for x in args.folds.split(",")] if os.path.exists(os.path.join(args.out_dir, f"fold{k}_log.json"))}
    first = scores.get(min(scores)) if scores else (logs[0][-1]["macro_f1"] if 0 in logs else 0.0)
    json.dump(dict(exp_id=os.environ.get("LAB_EXP_ID", ""),
                   primary_metric=dict(name="fold_macro_f1_mean" if scores else "fold0_ep1_macro_f1",
                                       value=float(np.mean(list(scores.values())) if scores else first)),
                   metrics=dict(gate=args.gate, gate_fail=gate_fail, fold_macro_f1={str(k): v for k, v in scores.items()},
                                epoch_logs={str(k): v for k, v in logs.items()}),
                   seed=args.seed, official_val=None, notes="threshold >=0.5 fixed; OOF = last epoch"),
              open(os.path.join(args.out_dir, "metrics.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
