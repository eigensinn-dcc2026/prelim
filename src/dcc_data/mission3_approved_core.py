"""Auditable primitives for the four approved M3 experiments (no network IO)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dcc_inference.mission3_bert import LABELS


def soft_f1(logits, targets):
    p, y = logits.float().sigmoid(), targets.float()
    return 1 - (2 * (p * y).sum(0) / (p.sum(0) + y.sum(0) + 1e-8)).mean()


def grouped_objective(logits, targets, pos_weight, mode, micro_batch=8,
                      teacher=None, alpha=.5):
    """Same weighted BCE; only soft-F1 aggregation changes. KD is Bernoulli BCE.

    Caller retains all microbatch graphs until this one effective-batch backward.
    An uneven final microbatch is weighted by its actual number of examples.
    """
    if mode not in ("micro", "global") or micro_batch < 1:
        raise ValueError("Invalid aggregation")
    value = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), pos_weight=pos_weight)
    if mode == "global":
        value = value + soft_f1(logits, targets)
    else:
        value = value + sum(soft_f1(logits[i:i+micro_batch], targets[i:i+micro_batch]) *
                            len(logits[i:i+micro_batch]) / len(logits)
                            for i in range(0, len(logits), micro_batch))
    if teacher is not None:
        if teacher.shape != targets.shape or not torch.isfinite(teacher).all() or not ((teacher >= 0) & (teacher <= 1)).all():
            raise ValueError("Invalid teacher probabilities")
        if not 0 <= alpha <= 1:
            raise ValueError("Invalid distillation mixture")
        value = (1-alpha)*value + alpha*F.binary_cross_entropy_with_logits(logits.float(), teacher.float())
    return value


def assert_disjoint(rows, left, right):
    """Neither file, call identity nor exact normalized text may cross a fold."""
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        raise ValueError("Repeated row in split")
    for key in ("name", "call", "text_hash"):
        overlap = {rows[i][key] for i in left} & {rows[i][key] for i in right}
        if overlap:
            raise ValueError(f"Split leakage by {key}: {len(overlap)} overlap(s)")


def verify_oof(rows, train, holdout, folds, probabilities):
    """Require complete honest inner-fold provenance, excluding OUTER holdout."""
    assert_disjoint(rows, train, holdout)
    train_set, seen = set(train), set()
    if probabilities.shape != (len(train), 9) or not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("OOF probabilities are incomplete or invalid")
    for fold in folds:
        fit, predict = fold["fit"], fold["predict"]
        if set(fit) | set(predict) != train_set or seen & set(predict):
            raise ValueError("OOF fold coverage mismatch")
        assert_disjoint(rows, fit, predict)
        assert_disjoint(rows, fit, holdout)
        if fold.get("initialization") != "public_pretrained":
            raise ValueError("OOF teacher must not initialize from the task-finetuned checkpoint")
        seen.update(predict)
    if seen != train_set:
        raise ValueError("OOF predictions do not cover every training example")


def read_annotations(path, rows, train):
    """Human-only supervision; offsets refer to normalized, joined text."""
    from dcc_inference.mission3_bert import transcript
    lookup = {rows[i]["name"]: i for i in train}
    result = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("name") not in lookup:
                raise ValueError("Annotation outside fitting split")
            if item.get("provenance") != "human" or not str(item.get("reviewer", "")).strip():
                raise ValueError("Human review provenance required; rules/AI are not human annotations")
            if item.get("label") not in LABELS or item.get("state") not in ("positive", "negative", "unknown"):
                raise ValueError("Invalid annotation label/state")
            index = lookup[item["name"]]
            label = LABELS.index(item["label"])
            if (index, label) in result:
                raise ValueError("Duplicate annotation")
            text = transcript(rows[index]["doc"])
            span = item.get("span")
            if span is None:
                if item["state"] != "unknown":
                    raise ValueError("Positive/negative annotation requires evidence span")
            elif (not isinstance(span, list) or len(span) != 2 or
                  not all(type(x) is int for x in span) or not 0 <= span[0] < span[1] <= len(text)):
                raise ValueError("Invalid evidence character offsets")
            result[index, label] = dict(state=("positive", "negative", "unknown").index(item["state"]), span=span)
    if not result:
        raise ValueError("No reviewed human annotations")
    return result


class EvidenceClassifier(torch.nn.Module):
    """Label attention with human span/state supervision and residual prediction.

    Attention is a model attribution, not a medically verified explanation.
    Unannotated labels receive NO auxiliary state/span targets.
    """
    def __init__(self, base):
        super().__init__()
        self.base = base
        hidden = base.classifier.in_features
        self.queries = torch.nn.Parameter(torch.randn(9, hidden) * .02)
        self.states = torch.nn.Linear(hidden, 3)
        self.residual = torch.nn.Parameter(torch.zeros(9, hidden+3))

    def forward(self, ids, mask, detail=False):
        h = self.base.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        scores = torch.einsum("bth,lh->blt", h, self.queries) / h.shape[-1]**.5
        scores = scores.masked_fill(~mask[:, None, :].bool(), torch.finfo(scores.dtype).min)
        attention = scores.softmax(-1)
        pooled = torch.einsum("blt,bth->blh", attention, h)
        states = self.states(pooled)
        joined = torch.cat((pooled, states.softmax(-1)), -1)
        logits = self.base.classifier(self.base.dropout(h[:, 0])) + (joined*self.residual[None]).sum(-1)
        return (logits, states, attention) if detail else logits


def annotation_objective(states, attention, offsets, indices, annotations):
    terms = []
    for b, index in enumerate(indices):
        for label in range(9):
            item = annotations.get((index, label))
            if item is None:
                continue
            span = item["span"]
            selected = None
            if span is not None:
                # Do not supervise a positive/negative state if its evidence was truncated.
                visible = [pair for pair in offsets[b] if pair[1] > pair[0]]
                if not visible or span[1] > max(pair[1] for pair in visible):
                    continue
                selected = torch.tensor([end > start and end > span[0] and start < span[1]
                                         for start, end in offsets[b]], device=attention.device)
                if not selected.any():
                    continue
            target = torch.tensor([item["state"]], device=states.device)
            value = F.cross_entropy(states[b, label].float()[None], target)
            if selected is not None:
                value = value - attention[b, label, selected].float().sum().clamp_min(1e-8).log()
            terms.append(value)
    return torch.stack(terms).mean() if terms else states.sum()*0 + attention.sum()*0


def route_mask(front_probs, lengths, margin):
    if not 0 <= margin <= .5:
        raise ValueError("Invalid review margin")
    return (np.asarray(lengths) > 254) & (np.abs(np.asarray(front_probs)-.5).min(axis=1) <= margin)


def crossfit_cascade(gold, front, long, lengths, n_splits=5, seed=42):
    """Select one global review margin on other folds; assess on held-out fold.

    Random routing uses the same number of long calls as the selected policy.
    CPU latency must be measured separately, never inferred as a measured speedup.
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import f1_score
    gold, front, long, lengths = map(np.asarray, (gold, front, long, lengths))
    if gold.shape != front.shape or front.shape != long.shape or len(lengths) != len(gold):
        raise ValueError("Cascade input alignment mismatch")
    if gold.ndim != 2 or gold.shape[1] != 9 or not np.isin(gold,[0,1]).all():
        raise ValueError("Invalid cascade targets")
    if any(not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all() for p in (front,long)):
        raise ValueError("Invalid cascade probabilities")
    output, random_output = front.copy(), front.copy()
    decisions, summaries = np.zeros(len(gold), bool), []
    rng = np.random.default_rng(seed)
    score = lambda y, p: float(f1_score(y, p >= .5, average="macro", zero_division=0))
    for fit, test in KFold(n_splits, shuffle=True, random_state=seed).split(gold):
        target = score(gold[fit], long[fit]) - .003
        candidates = []
        for margin in (0., .025, .05, .1, .2, .5):
            chosen = route_mask(front[fit], lengths[fit], margin)
            mixed = np.where(chosen[:, None], long[fit], front[fit])
            candidates.append((int(chosen.sum()), score(gold[fit], mixed), margin))
        eligible = [item for item in candidates if item[1] >= target]
        best = min(eligible, key=lambda item: (item[0], -item[1], item[2])) if eligible else max(candidates, key=lambda item: (item[1], -item[0]))
        chosen = route_mask(front[test], lengths[test], best[2])
        output[test] = np.where(chosen[:, None], long[test], front[test])
        decisions[test] = chosen
        possible = test[lengths[test] > 254]
        random_rows = rng.choice(possible, int(chosen.sum()), replace=False)
        random_output[random_rows] = long[random_rows]
        summaries.append(dict(margin=best[2], test_rows=len(test), reviewed=int(chosen.sum())))
    return dict(front_f1=score(gold, front), long_f1=score(gold, long),
                crossfit_f1=score(gold, output), random_f1=score(gold, random_output),
                reviewed_fraction=float(decisions.mean()), folds=summaries), output
