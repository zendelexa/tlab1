"""Stage 1: GPT-2 next-item prediction on Yambda-50M (item ids as tokens)."""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import CKPT, CTX, RES, item_stats, load_data, make_gpt, sample_windows, train_end_index
from evaluation import EvalSet, policy_scores, popularity_topk, repeat_topk, topk_from_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--name", default="gpt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    CKPT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    d = load_data()
    meta = d.meta
    V = meta["n_tokens"]
    cut = train_end_index(d, meta["val_start"])
    pop, ilen, _ = item_stats(d, meta["val_start"])
    val = EvalSet(d, meta["val_start"], meta["test_start"])
    print(f"val users {len(val)} (next in-vocab {val.next_ok.sum()})", flush=True)

    model = make_gpt(V, args.d, args.layers, args.heads)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(s, args.steps) / args.steps))))

    log, best, t0, run_loss = [], -1.0, time.time(), []
    for step in range(1, args.steps + 1):
        model.train()
        idx = sample_windows(d, rng, args.batch, cut)
        tok = np.where(idx >= 0, d.tok[np.maximum(idx, 0)], 0)
        inp = torch.from_numpy(tok[:, :-1])
        tgt = torch.from_numpy(tok[:, 1:].copy())
        tgt[tgt < 2] = -100                     # do not predict PAD / OOV
        logits = model(input_ids=inp).logits
        loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run_loss.append(loss.item())
        if step % 50 == 0:
            print(f"step {step} loss {np.mean(run_loss[-50:]):.4f} {time.time() - t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            sc = policy_scores(model, val)
            m = val.metrics(topk_from_scores(sc), pop, ilen, V - 2)
            m.update(step=step, loss=float(np.mean(run_loss[-200:])), time=time.time() - t0)
            log.append(m)
            print(json.dumps({k: round(v, 4) for k, v in m.items()}), flush=True)
            if m["next_ndcg@10"] > best:
                best = m["next_ndcg@10"]
                torch.save(model.state_dict(), CKPT / f"{args.name}.pt")
    json.dump(dict(args=vars(args), log=log), open(RES / f"{args.name}_train_log.json", "w"), indent=1)

    # final evaluation of the best checkpoint on the test day + simple baselines
    model.load_state_dict(torch.load(CKPT / f"{args.name}.pt"))
    pop_t, ilen_t, _ = item_stats(d, meta["test_start"])
    test = EvalSet(d, meta["test_start"], 2 ** 62)
    res = {}
    sc = policy_scores(model, test)
    res["gpt"] = test.metrics(topk_from_scores(sc), pop_t, ilen_t, V - 2)
    ptop = popularity_topk(d, meta["test_start"], V)
    res["popular_last_day"] = test.metrics(np.tile(ptop, (len(test), 1)), pop_t, ilen_t, V - 2)
    res["repeat_heuristic"] = test.metrics(repeat_topk(d, test, ptop), pop_t, ilen_t, V - 2)
    for k, v in res.items():
        print(k, json.dumps({a: round(b, 4) for a, b in v.items()}))
    json.dump(res, open(RES / f"{args.name}_test.json", "w"), indent=1)


if __name__ == "__main__":
    main()
