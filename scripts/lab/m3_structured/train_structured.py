"""RoBERTa-large + StructuredSetHead 학습 (제안 2 / 제안 3·제안 1 학생 겸용)
  --mode holdout : 80:20 (--split_npz 로 기존 홀드아웃 인덱스 재사용 권장)
  --mode kfold   : 5-fold, OOF 확률 저장 (제안 3·4·5 재료)  -> --select last 권장
  --soft_target_npy teacher_oof.npy : 증류 (t = soft_mix*y + (1-soft_mix)*teacher)
  4060 8GB : --fp16 --grad_ckpt --adam8bit --batch 4 --accum 4     A100 : --bf16 --batch 16
대조군(기존 레시피 재현): --no_pairwise --lambda_set 0 --lambda_bce 1 --lambda_f1 1 --head_lr 1e-5
"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from losses import SYMPTOMS, pos_weight_from_labels, macro_f1_at_half
from structured_head import StructuredSetHead, StructuredLoss


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv", required=True); p.add_argument("--text_col", default="text")
    p.add_argument("--model", default="klue/roberta-large")
    p.add_argument("--revision", default=None, help="HF 커밋 고정 (재현성)")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch", type=int, default=16); p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5); p.add_argument("--head_lr", type=float, default=1e-4)
    p.add_argument("--w_lr", type=float, default=None, help="pairwise W 전용 lr (기본: head_lr). W 는 0 에서 출발하므로 1e-2 권장")
    p.add_argument("--epochs", type=int, default=3); p.add_argument("--warmup", type=float, default=0.1)  # ROBERTA-003
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["holdout", "kfold"], default="holdout")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--split_npz", default=None, help="npz(train_idx, val_idx)")
    p.add_argument("--select", choices=["best", "last"], default="last",
                   help="last = ROBERTA-003 기준(마지막 epoch). best 는 홀드아웃으로 epoch 를 고르므로 비교 불가")
    # structured head
    p.add_argument("--exclude_empty", action="store_true", help="B2 (멘토 확인 후)")
    p.add_argument("--no_pairwise", action="store_true", help="W 제거 (대조군)")
    p.add_argument("--lambda_set", type=float, default=1.0)
    p.add_argument("--lambda_bce", type=float, default=0.5)
    p.add_argument("--lambda_f1", type=float, default=1.0)
    p.add_argument("--f1_beta", type=float, default=0.0); p.add_argument("--f1_momentum", type=float, default=0.0)
    # distillation
    p.add_argument("--soft_target_npy", default=None); p.add_argument("--soft_mix", type=float, default=0.5)
    # memory
    p.add_argument("--fp16", action="store_true"); p.add_argument("--bf16", action="store_true")
    p.add_argument("--grad_ckpt", action="store_true"); p.add_argument("--adam8bit", action="store_true")
    p.add_argument("--out_dir", default="runs/structured")
    return p.parse_args()


class TextDS(Dataset):
    def __init__(self, texts, Y, T): self.texts, self.Y, self.T = texts, Y, T
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i], self.Y[i], self.T[i]


def make_collate(tok, max_len):
    def collate(batch):
        texts, ys, ts = zip(*batch)
        enc = tok(list(texts), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "y": torch.tensor(np.stack(ys)), "t": torch.tensor(np.stack(ts))}
    return collate


class EncoderStructured(nn.Module):
    def __init__(self, name, K, exclude_empty, use_pairwise, grad_ckpt, revision=None):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name, revision=revision)
        if grad_ckpt:
            self.enc.gradient_checkpointing_enable()
        self.head = StructuredSetHead(self.enc.config.hidden_size, K, exclude_empty, use_pairwise)

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]  # CLS
        return self.head(h)


def split_indices(Y, args):
    if args.split_npz:
        z = np.load(args.split_npz)
        return [(z["train_idx"], z["val_idx"])]
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
    X = np.zeros((len(Y), 1))
    if args.mode == "holdout":
        return list(MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed).split(X, Y))
    return list(MultilabelStratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed).split(X, Y))


@torch.no_grad()
def predict(model, dl, device, amp_dtype):
    model.eval(); out = []
    for b in dl:
        with torch.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_dtype is not None):
            _, marg = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        out.append(marg.float().cpu().numpy())
    return np.concatenate(out)


def run_split(args, texts, Y, T, tr, va, tag, device):
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = EncoderStructured(args.model, len(SYMPTOMS), args.exclude_empty, not args.no_pairwise, args.grad_ckpt, args.revision).to(device)
    pw = pos_weight_from_labels(Y[tr]).to(device)
    crit = StructuredLoss(model.head, pw, args.lambda_set, args.lambda_bce, args.lambda_f1, args.f1_beta, args.f1_momentum)
    col = make_collate(tok, args.max_len)
    dl_tr = DataLoader(TextDS(texts[tr], Y[tr], T[tr]), batch_size=args.batch, shuffle=True, collate_fn=col)
    dl_va = DataLoader(TextDS(texts[va], Y[va], T[va]), batch_size=args.batch * 2, shuffle=False, collate_fn=col)
    groups = [{"params": [p for n, p in model.named_parameters() if n.startswith("enc.")], "lr": args.lr},
              {"params": [p for n, p in model.named_parameters() if n.startswith("head.") and n != "head.W_raw"], "lr": args.head_lr}]
    if not args.no_pairwise:  # W 는 별도 lr, weight decay 없음 (head_lr 1e-5 면 3 epoch 동안 거의 0 에 머묾)
        groups.append({"params": [model.head.W_raw], "lr": args.w_lr or args.head_lr, "weight_decay": 0.0})
    if args.adam8bit:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(groups, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(groups, weight_decay=0.01)
    total = (len(dl_tr) * args.epochs + args.accum - 1) // args.accum
    sch = get_linear_schedule_with_warmup(opt, int(total * args.warmup), total)
    amp_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else None)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
    use_soft = args.soft_target_npy is not None
    best_f1, best_probs, log = -1.0, None, []
    for ep in range(args.epochs):
        model.train(); crit.soft_f1.reset(); t0 = time.time(); run = 0.0
        for i, b in enumerate(dl_tr):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_dtype is not None):
                logP, marg = model(b["input_ids"], b["attention_mask"])
            loss = crit(logP, marg, b["y"], b["t"] if use_soft else None)
            scaler.scale(loss / args.accum).backward(); run += loss.item()
            if (i + 1) % args.accum == 0 or (i + 1) == len(dl_tr):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step()
        probs = predict(model, dl_va, device, amp_dtype)
        f1, per, zero_rate = macro_f1_at_half(probs, Y[va])
        log.append({"epoch": ep + 1, "loss": run / len(dl_tr), "macro_f1": f1, "per_class": per.tolist(),
                    "pred_allzero_rate": zero_rate, "min": (time.time() - t0) / 60})
        print(f"[{tag}] ep{ep+1} loss {run/len(dl_tr):.4f} F1 {f1:.4f} allzero {zero_rate:.2%} "
              f"({(time.time()-t0)/60:.0f}min) | " + " ".join(f"{s}:{v:.3f}" for s, v in zip(SYMPTOMS, per)), flush=True)
        if (args.select == "last" and ep == args.epochs - 1) or (args.select == "best" and f1 > best_f1):
            best_f1, best_probs = f1, probs.copy()
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"{tag}.pt"))
    json.dump(log, open(os.path.join(args.out_dir, f"{tag}_log.json"), "w"), ensure_ascii=False, indent=1)
    del model, opt; torch.cuda.empty_cache()
    return best_f1, best_probs


def write_metrics(args, name, value, extra):
    """dcc-lab 러너가 읽는 metrics.json (threshold 0.5 고정, 공식 Val 미사용)"""
    m = dict(exp_id=os.environ.get("LAB_EXP_ID", ""), primary_metric=dict(name=name, value=float(value)),
             metrics=extra, seed=args.seed, official_val=None, notes="threshold >=0.5 fixed, no postprocessing")
    json.dump(m, open(os.path.join(args.out_dir, "metrics.json"), "w"), ensure_ascii=False, indent=1)


def main():
    args = get_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out_dir, "args.json"), "w"), indent=1)
    df = pd.read_csv(args.data_csv)
    texts = df[args.text_col].astype(str).to_numpy()
    Y = df[SYMPTOMS].to_numpy().astype(np.float32)
    T = Y
    if args.soft_target_npy:
        teacher = np.load(args.soft_target_npy).astype(np.float32); assert teacher.shape == Y.shape
        T = args.soft_mix * Y + (1 - args.soft_mix) * teacher
    device, splits = "cuda", split_indices(Y, args)
    if args.mode == "holdout":
        tr, va = splits[0]
        f1, probs = run_split(args, texts, Y, T, tr, va, "holdout", device)
        np.save(os.path.join(args.out_dir, "holdout_probs.npy"), probs)
        np.save(os.path.join(args.out_dir, "holdout_labels.npy"), Y[va])
        np.save(os.path.join(args.out_dir, "holdout_idx.npy"), va)
        print(f"holdout macro F1 = {f1:.4f}")
        write_metrics(args, "holdout_macro_f1", f1, dict(per_class=dict(zip(SYMPTOMS, macro_f1_at_half(probs, Y[va])[1].tolist()))))
    else:
        oof, scores = np.zeros_like(Y), []
        for k, (tr, va) in enumerate(splits):
            done = os.path.join(args.out_dir, f"fold{k}_probs.npy")
            if os.path.exists(done):  # 러너 재시도: 끝난 fold 는 건너뜀
                assert (np.load(os.path.join(args.out_dir, f"fold{k}_idx.npy")) == va).all(), "fold 분할 불일치"
                probs = np.load(done); f1 = macro_f1_at_half(probs, Y[va])[0]
                print(f"[fold{k}] 완료분 재사용 F1 {f1:.4f}", flush=True)
            else:
                f1, probs = run_split(args, texts, Y, T, tr, va, f"fold{k}", device)
                np.save(os.path.join(args.out_dir, f"fold{k}_idx.npy"), va); np.save(done, probs)
            oof[va] = probs; scores.append(f1)
        np.save(os.path.join(args.out_dir, "oof_probs.npy"), oof)
        f1, per, zero = macro_f1_at_half(oof, Y)
        print("fold F1:", [f"{s:.4f}" for s in scores], f"| OOF macro F1 = {f1:.4f}")
        write_metrics(args, "oof_macro_f1", f1, dict(fold_macro_f1=scores, per_class=dict(zip(SYMPTOMS, per.tolist())),
                                                    pred_allzero_rate=zero))


if __name__ == "__main__":
    main()
