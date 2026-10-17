"""What happens when a policy generates several tracks ahead (each appended to its own context)?

For test users with >= H events on the test day we compare, at every step h = 1..H:
  * teacher-forced  : context = history + the user's real first h-1 test-day events (real feedback)
  * free-running    : context = history + the policy's own h-1 generated tracks (feedback simulated by an RM)
and report: Recall@10 of the real h-th event, KL(free || teacher-forced) of the next-track distributions,
reward of the generated track under RM A and RM B (each RM simulates feedback in its own context),
self-repetition inside the generated sequence and share of tracks already in the visible history.

Controls for the "predicted reward grows along the rollout" effect:
  reward_real_tf_X     : RM-X reward of the user's REAL h-th track in the REAL context (real feedback)
  reward_gen_realctx_X : RM-X reward of the GENERATED h-th track scored in the REAL context
  actual_ratio_real    : the real played ratio of the user's h-th track (ground truth by position)
Every metric also gets a standard error (key + "_se") over users.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from common import CKPT, CTX, RES, item_stats, load_data, load_policy, policy_logits, ratio_bucket
from evaluation import EvalSet
from rewards import RMScorer, combine, length_bias, rm_context, token_length_bias
from rl import load_models, simulate_feedback

H = 5


def contexts(d, es, users_idx, h, real):
    """Event-index rows: history before cutoff (+ the first h real test-day events if real=True)."""
    rows = []
    for i in users_idx:
        b, a = es.ctx_range[i]
        extra = np.arange(a, a + h) if real else np.arange(a, a)
        ids = np.concatenate([np.arange(b, a), extra])[-(CTX - H):]
        rows.append(ids)
    return rows


@torch.no_grad()
def analyse(policy, d, es, users, scorers, ilen, lbias, kind="ratio", greedy=True, seed=0):
    torch.manual_seed(seed)
    real_next = np.array([[d.tok[es.ctx_range[i][1] + h] for h in range(H)] for i in users])
    # free-running state (policy tokens + per-RM feedback contexts)
    base_rows = contexts(d, es, users, 0, real=False)
    (tok, _, _, _), last = rm_context(d, base_rows)
    ctx_tok = tok.clone()
    n = last + 1
    rm_ctx = {k: [t.clone() for t in rm_context(d, base_rows)[0]] for k in scorers}
    gen = []
    keys = ["recall10_free", "recall10_tf", "kl_free_tf", "self_repeat", "in_history", "logpop", "length",
            "actual_ratio_real"]
    for k in scorers:
        keys += [f"reward_{k}", f"ratio_{k}", f"reward_real_tf_{k}", f"reward_gen_realctx_{k}"]
    out = {k: [] for k in keys}
    out.update({k + "_se": [] for k in keys})

    def put(key, vals):
        vals = np.asarray(vals, dtype=np.float64)
        out[key].append(float(vals.mean()))
        out[key + "_se"].append(float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
    real_ratio = np.array([[d.ratio[es.ctx_range[i][1] + h] for h in range(H)] for i in users])
    il_t, lb_t = torch.from_numpy(ilen), torch.from_numpy(lbias)
    pop = np.log1p(item_stats(d, d.meta["test_start"])[0])
    for h in range(H):
        lg_free = policy_logits(policy, ctx_tok, last_pos=n - 1)
        tf_rows = contexts(d, es, users, h, real=True)
        (tf_tok, tf_rb, tf_lk, tf_org), tf_last = rm_context(d, tf_rows)
        lg_tf = policy_logits(policy, tf_tok, last_pos=tf_last)
        lp_f, lp_t = F.log_softmax(lg_free, -1), F.log_softmax(lg_tf, -1)
        tgt = torch.from_numpy(real_next[:, h])
        ok = tgt >= 2
        top_f = torch.topk(lp_f, 10, -1).indices
        top_t = torch.topk(lp_t, 10, -1).indices
        okn = ok.numpy()
        put("recall10_free", (top_f == tgt[:, None]).any(1)[ok].float().numpy())
        put("recall10_tf", (top_t == tgt[:, None]).any(1)[ok].float().numpy())
        put("kl_free_tf", (lp_f.exp() * (lp_f - lp_t)).sum(-1).numpy())
        a = lp_f.argmax(-1) if greedy else torch.multinomial(lp_f.exp(), 1).squeeze(-1)
        put("self_repeat", [int(a[i]) in set(int(g[i]) for g in gen) for i in range(len(a))] if gen
            else np.zeros(len(a)))
        put("in_history", [int(a[i]) in es.hist[u] for i, u in enumerate(users)])
        put("logpop", pop[a.numpy()])
        put("length", ilen[a.numpy()])
        put("actual_ratio_real", real_ratio[:, h])
        tgt_ok = tgt.clamp(min=2)
        for k, sc in scorers.items():
            # controls: real context with real feedback
            hs_tf = sc.states(tf_tok, tf_rb, tf_lk, tf_org, tf_last)
            pr = sc.score(hs_tf, torch.stack([tgt_ok, a], 1))          # [N,2,3]: real track, generated track
            r_real = combine(kind, pr[:, 0, 0], pr[:, 0, 1], pr[:, 0, 2], il_t[tgt_ok], lb_t[tgt_ok])
            r_gen_rc = combine(kind, pr[:, 1, 0], pr[:, 1, 1], pr[:, 1, 2], il_t[a], lb_t[a])
            put(f"reward_real_tf_{k}", r_real.numpy()[okn])
            put(f"reward_gen_realctx_{k}", r_gen_rc.numpy())
            # free-running: generated track in the policy's own (simulated) context
            t_, rb, lk, org = rm_ctx[k]
            hs = sc.states(t_, rb, lk, org, n - 1)
            p = sc.score(hs, a[:, None])[:, 0]
            r = combine(kind, p[:, 0], p[:, 1], p[:, 2], il_t[a], lb_t[a])
            put(f"reward_{k}", r.numpy())
            put(f"ratio_{k}", p[:, 0].numpy())
            ar = torch.arange(len(a))
            t_[ar, n] = a
            rb[ar, n] = simulate_feedback(p[:, 0])
            lk[ar, n] = (torch.rand(len(a)) < p[:, 1]).long()
            org[ar, n] = 0
        gen.append(a)
        ctx_tok[torch.arange(len(a)), n] = a
        n = n + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["gptrep"])
    ap.add_argument("--base", default="gptrep")
    ap.add_argument("--kind", default="ratio")
    ap.add_argument("--n_users", type=int, default=2000)
    ap.add_argument("--out", default="multistep")
    args = ap.parse_args()
    d = load_data()
    meta = d.meta
    V = meta["n_tokens"]
    es = EvalSet(d, meta["test_start"], 2 ** 62)
    # users with at least H test-day events
    cand = [i for i in range(len(es)) if d.uend[es.users[i]] - es.ctx_range[i][1] >= H]
    users = np.sort(np.random.default_rng(0).choice(cand, min(args.n_users, len(cand)), replace=False))
    _, (rm_a, rm_b) = load_models(d, args.base)
    _, ilen, _ = item_stats(d, meta["test_start"])
    edges, means = length_bias(d, meta["val_start"])
    lbias = token_length_bias(edges, means, ilen)
    scorers = {"A": RMScorer(rm_a, ilen), "B": RMScorer(rm_b, ilen)}
    res = {}
    for name in args.policies:
        pol = load_policy(args.base, V)
        if name != args.base:
            pol.load_state_dict(torch.load(CKPT / f"{name}.pt"))
        pol.eval()
        for mode in ("greedy", "sample"):
            r = analyse(pol, d, es, users, scorers, ilen, lbias, args.kind, greedy=mode == "greedy")
            res[f"{name}|{mode}"] = r
            print(name, mode, json.dumps({k: [round(x, 4) for x in v] for k, v in r.items() if not k.endswith("_se")}),
                  flush=True)
    json.dump(res, open(RES / f"{args.out}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
