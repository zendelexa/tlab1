"""Stage 2: reward model -- predicts the user's response to a track given the listening history.

Heads (all from the same user state):
  ratio : played ratio of the track            (soft-label BCE)      -> short-term engagement
  like  : the track gets a like                (BCE)
  cont  : the session continues after it        (BCE; next listen within 30 min) -> long-term proxy
Labels that would need information after the training cutoff are masked (no leakage).
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from scipy.stats import spearmanr

from common import (CKPT, CTX, RES, SESSION_GAP, RewardModel, item_stats, load_data, loglen, ratio_bucket,
                    sample_windows, train_end_index)


def labels(d, idx, cutoff):
    """Targets for events idx (-1 = padding). cont is masked if it cannot be known before `cutoff`."""
    ok = idx >= 0
    j = np.maximum(idx, 0)
    ratio = np.where(ok, d.ratio[j], 0.0)
    like = np.where(ok, d.liked[j], 0.0)
    gap = d.gap_next[j]
    known = d.ts[j] + SESSION_GAP < cutoff                  # outcome observable before the cutoff
    cont = np.where((gap >= 0) & (gap <= SESSION_GAP), 1.0, 0.0)
    cont_mask = ok & known
    return ratio.astype(np.float32), like.astype(np.float32), cont.astype(np.float32), cont_mask


def batch_tensors(d, idx):
    """Model inputs for windows idx [B, L+1]: history = idx[:, :-1], candidates = idx[:, 1:]."""
    ok = idx >= 0
    j = np.maximum(idx, 0)
    tok = np.where(ok, d.tok[j], 0)
    rb = np.where(ok, ratio_bucket(d.ratio[j]), 0)
    lk = np.where(ok, d.liked[j], 0).astype(np.int64)
    org = np.where(ok, d.organic[j], 0)
    ll = np.where(ok, loglen(d.length[j]), 0.0).astype(np.float32)
    h = [torch.from_numpy(a[:, :-1]) for a in (tok, rb, lk, org)]
    c = (torch.from_numpy(tok[:, 1:]), torch.from_numpy(ll[:, 1:]), torch.from_numpy(org[:, 1:].astype(np.float32)))
    return h, c


def loss_fn(logits, ratio, like, cont, mask, cont_mask, w_like=1.0, w_cont=0.5):
    l_r = F.binary_cross_entropy_with_logits(logits[..., 0], ratio, reduction="none")
    l_l = F.binary_cross_entropy_with_logits(logits[..., 1], like, reduction="none")
    l_c = F.binary_cross_entropy_with_logits(logits[..., 2], cont, reduction="none")
    m = mask.float()
    cm = cont_mask.float()
    return ((l_r * m).sum() + w_like * (l_l * m).sum()) / m.sum() + w_cont * (l_c * cm).sum() / cm.sum().clamp(min=1)


@torch.no_grad()
def predict_events(rm, d, ev_idx, batch=512):
    """Predictions for specific events, each with its own full context of up to CTX previous events."""
    rm.eval()
    out = []
    for i in range(0, len(ev_idx), batch):
        e = ev_idx[i:i + batch]
        u = np.searchsorted(d.ustart, e, side="right") - 1
        beg = np.maximum(d.ustart[u], e - CTX)
        L = e - beg
        idx = np.full((len(e), CTX + 1), -1, dtype=np.int64)
        for b in range(len(e)):
            idx[b, :L[b]] = np.arange(beg[b], e[b])
            idx[b, L[b]] = e[b]
        (tok, rb, lk, org), (ct, cl, co) = batch_tensors(d, idx)
        h = rm.state(tok, rb, lk, org)
        pos = torch.from_numpy(L - 1)
        ar = torch.arange(len(e))
        hs = h[ar, pos]
        lg = rm.head(hs, ct[ar, pos], cl[ar, pos], co[ar, pos])
        out.append(torch.sigmoid(lg))
    return torch.cat(out).numpy()


def evaluate_rm(rm, d, ev_idx, cutoff_end, pop_ratio, user_ratio):
    first = d.ustart[np.searchsorted(d.ustart, ev_idx, side="right") - 1]
    ev_idx = ev_idx[ev_idx > first]                        # need at least one previous event as context
    p = predict_events(rm, d, ev_idx)
    r = d.ratio[ev_idx]
    lk = d.liked[ev_idx]
    gap = d.gap_next[ev_idx]
    known = d.ts[ev_idx] + SESSION_GAP < cutoff_end
    cont = ((gap >= 0) & (gap <= SESSION_GAP)).astype(float)
    res = dict(n=int(len(ev_idx)),
               ratio_mae=float(mean_absolute_error(r, p[:, 0])),
               ratio_rmse=float(np.sqrt(np.mean((r - p[:, 0]) ** 2))),
               ratio_spearman=float(spearmanr(r, p[:, 0]).statistic),
               complete_auc=float(roc_auc_score(r >= 0.9, p[:, 0])),
               skip_auc=float(roc_auc_score(r < 0.3, -p[:, 0])),
               like_auc=float(roc_auc_score(lk, p[:, 1])), like_ap=float(average_precision_score(lk, p[:, 1])),
               like_base_rate=float(lk.mean()),
               cont_auc=float(roc_auc_score(cont[known], p[known, 2])),
               cont_base_rate=float(cont[known].mean()))
    # baselines for the ratio head
    tok = d.tok[ev_idx]
    u = np.searchsorted(d.ustart, ev_idx, side="right") - 1
    for name, pred in [("global_mean", np.full(len(r), pop_ratio[0])), ("item_mean", pop_ratio[1][tok]),
                       ("user_mean", user_ratio[u]), ("user_item_avg", 0.5 * (pop_ratio[1][tok] + user_ratio[u]))]:
        res[f"baseline_{name}_mae"] = float(mean_absolute_error(r, pred))
        res[f"baseline_{name}_complete_auc"] = float(roc_auc_score(r >= 0.9, pred)) if name != "global_mean" else 0.5
    return res, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--user_half", type=int, default=0, help="train on users of this half (0/1); -1 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="rm_a")
    ap.add_argument("--max_eval", type=int, default=30000)
    ap.add_argument("--max_test", type=int, default=0, help="0 = all test-day events")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    d = load_data()
    meta = d.meta
    V = meta["n_tokens"]
    cut = train_end_index(d, meta["val_start"])
    # RM-A and RM-B see *disjoint* halves of the users (fixed permutation) -> independent judges
    perm = np.random.default_rng(12345).permutation(len(d.users))
    half = np.empty(len(d.users), dtype=np.int64)
    half[perm] = np.arange(len(d.users)) % 2
    keep = np.ones(len(d.users), bool) if args.user_half < 0 else half == args.user_half
    cut_sub = np.where(keep, cut, d.ustart)            # dropped users have empty training ranges

    rm = RewardModel(V, d=args.d, n_layer=args.layers)
    opt = torch.optim.AdamW(rm.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(s, args.steps) / args.steps))))
    # validation events (subsample of the val day, in-vocabulary candidates)
    vmask = (d.ts >= meta["val_start"]) & (d.ts < meta["test_start"]) & (d.tok >= 2)
    v_idx = np.flatnonzero(vmask)
    v_idx = np.sort(rng.choice(v_idx, min(args.max_eval, len(v_idx)), replace=False))
    pop, _, iratio = item_stats(d, meta["val_start"])
    glob = float(d.ratio[d.ts < meta["val_start"]].mean())
    ur = np.array([d.ratio[s:e].mean() if e > s else glob for s, e in zip(d.ustart, cut)], dtype=np.float32)

    t0, log, best = time.time(), [], 1e9
    for step in range(1, args.steps + 1):
        rm.train()
        idx = sample_windows(d, rng, args.batch, cut_sub)
        (tok, rb, lk, org), (ct, cl, co) = batch_tensors(d, idx)
        ratio, like, cont, cmask = labels(d, idx[:, 1:], meta["val_start"])
        mask = torch.from_numpy((idx[:, 1:] >= 0) & (np.where(idx[:, 1:] >= 0, d.tok[np.maximum(idx[:, 1:], 0)], 0) >= 2))
        h = rm.state(tok, rb, lk, org)
        logits = rm.head(h, ct, cl, co)
        loss = loss_fn(logits, torch.from_numpy(ratio), torch.from_numpy(like), torch.from_numpy(cont), mask,
                       torch.from_numpy(cmask) & mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if step % 1000 == 0 or step == args.steps:
            m, _ = evaluate_rm(rm, d, v_idx, meta["test_start"], (glob, iratio), ur)
            m.update(step=step)
            log.append(m)
            print(json.dumps({k: round(v, 4) for k, v in m.items()}), flush=True)
            score = m["ratio_mae"] - 0.1 * m["like_auc"] - 0.1 * m["cont_auc"]
            if score < best:
                best = score
                torch.save(rm.state_dict(), CKPT / f"{args.name}.pt")

    # held-out evaluation on the full test day (history may include val-day events)
    rm.load_state_dict(torch.load(CKPT / f"{args.name}.pt"))
    _, _, iratio_t = item_stats(d, meta["test_start"])
    cut_t = train_end_index(d, meta["test_start"])
    ur_t = np.array([d.ratio[s:e].mean() if e > s else glob for s, e in zip(d.ustart, cut_t)], dtype=np.float32)
    t_idx = np.flatnonzero((d.ts >= meta["test_start"]) & (d.tok >= 2))
    if args.max_test:
        t_idx = np.sort(np.random.default_rng(0).choice(t_idx, args.max_test, replace=False))
    test_m, _ = evaluate_rm(rm, d, t_idx, meta["t_max"] + 1, (glob, iratio_t), ur_t)
    print("TEST", json.dumps({k: round(v, 4) for k, v in test_m.items()}), flush=True)
    json.dump(dict(args=vars(args), log=log, test=test_m), open(RES / f"{args.name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
