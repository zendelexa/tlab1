"""Stage 1b: add a repeat-aware head on top of the pretrained GPT-2 (next-item prediction, last position)."""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import CKPT, RES, Policy, RepeatHead, item_stats, load_data, make_gpt, sample_windows, train_end_index
from evaluation import EvalSet, policy_scores, popularity_topk, repeat_topk, topk_from_scores


def last_position_batch(d, rng, batch, cut):
    idx = sample_windows(d, rng, batch, cut)
    n = (idx >= 0).sum(1)
    tok_all = np.where(idx >= 0, d.tok[np.maximum(idx, 0)], 0)
    tgt = tok_all[np.arange(len(idx)), n - 1]
    inp = tok_all[:, :-1].copy()
    for b in range(len(idx)):              # hide the target itself from the input
        inp[b, n[b] - 1:] = 0
    keep = (tgt >= 2) & (n >= 2)
    return torch.from_numpy(inp[keep]), torch.from_numpy(n[keep] - 2), torch.from_numpy(tgt[keep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt", default="gpt")
    ap.add_argument("--name", default="gptrep")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--joint_lr", type=float, default=0.0, help=">0: also fine-tune GPT-2 with this lr")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=100)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    d = load_data()
    meta = d.meta
    V = meta["n_tokens"]
    gpt = make_gpt(V)
    gpt.load_state_dict(torch.load(CKPT / f"{args.gpt}.pt"))
    pol = Policy(gpt, RepeatHead(gpt.config.n_embd))
    pol.eval()
    groups = [{"params": pol.repeat.parameters(), "lr": args.lr}]
    if args.joint_lr > 0:
        groups.append({"params": pol.gpt.parameters(), "lr": args.joint_lr})
    else:
        for p in pol.gpt.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam(groups)
    cut = train_end_index(d, meta["val_start"])
    pop, ilen, _ = item_stats(d, meta["val_start"])
    val = EvalSet(d, meta["val_start"], meta["test_start"])
    t0, losses, log, best = time.time(), [], [], -1.0
    for step in range(1, args.steps + 1):
        tok, pos, tgt = last_position_batch(d, rng, args.batch, cut)
        if args.joint_lr > 0:
            hs = pol.hidden(tok)
        else:
            with torch.no_grad():
                hs = pol.hidden(tok)
        loss = F.cross_entropy(pol.logits_at(hs, tok, pos), tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"step {step} loss {np.mean(losses[-50:]):.4f} w={pol.repeat.lin.bias.data.numpy().round(2)} "
                  f"{time.time() - t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            m = val.metrics(topk_from_scores(policy_scores(pol, val)), pop, ilen, V - 2)
            m.update(step=step, loss=float(np.mean(losses[-100:])))
            log.append(m)
            print(json.dumps({k: round(v, 4) for k, v in m.items()}), flush=True)
            if m["next_ndcg@10"] > best:                 # model selection on the validation day only
                best = m["next_ndcg@10"]
                torch.save(pol.gpt.state_dict(), CKPT / f"{args.name}.pt")
                torch.save(pol.repeat.state_dict(), CKPT / f"{args.name}_repeat.pt")
    pol.gpt.load_state_dict(torch.load(CKPT / f"{args.name}.pt"))
    pol.repeat.load_state_dict(torch.load(CKPT / f"{args.name}_repeat.pt"))

    pop_t, ilen_t, _ = item_stats(d, meta["test_start"])
    test = EvalSet(d, meta["test_start"], 2 ** 62)
    res = {"gpt_repeat": test.metrics(topk_from_scores(policy_scores(pol, test)), pop_t, ilen_t, V - 2)}
    plain = make_gpt(V)
    plain.load_state_dict(torch.load(CKPT / f"{args.gpt}.pt"))
    res["gpt_plain"] = test.metrics(topk_from_scores(policy_scores(plain, test)), pop_t, ilen_t, V - 2)
    ptop = popularity_topk(d, meta["test_start"], V)
    res["popular_last_day"] = test.metrics(np.tile(ptop, (len(test), 1)), pop_t, ilen_t, V - 2)
    res["repeat_heuristic"] = test.metrics(repeat_topk(d, test, ptop), pop_t, ilen_t, V - 2)
    for k, v in res.items():
        print("TEST", k, json.dumps({a: round(b, 4) for a, b in v.items()}))
    json.dump(dict(args=vars(args), log=log, test=res), open(RES / f"{args.name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
