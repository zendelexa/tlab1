"""Stage 3: RL fine-tuning of the GPT-2 recommender against a learned reward model.

Objective per state s (history prefix from the train period):
      J(θ) = E_{a_1..a_H ~ π_θ} [ Σ_h γ^h r(s_h, a_h) ]  -  β Σ_h KL( π_θ(·|s_h) || π_ref(·|s_h) )
* r comes from reward model A (frozen); reward model B (other users/seed) is kept only for evaluation.
* Policy-gradient term: REINFORCE with a leave-one-out baseline over G samples of the same state (RLOO).
* KL term: exact, over the whole catalogue (the action space is small enough), at every generated position.
* H > 1: generated tracks are appended to the context; their "feedback" (needed by the reward model)
  is simulated with reward model A's own prediction -- the classic user-simulator loop.
For H = 1 the optimum of J is π*(a|s) ∝ π_ref(a|s) exp(r(s,a)/β), which analysis.py evaluates directly.
"""
import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (CKPT, CTX, RES, RewardModel, item_stats, load_data, load_policy, policy_logits,
                    ratio_bucket, train_end_index)
from evaluation import EvalSet, policy_scores, topk_from_scores
from rewards import RMScorer, combine, length_bias, rm_context, token_length_bias


def load_models(d, gpt_name="gpt", rm_names=("rm_a", "rm_b")):
    V = d.meta["n_tokens"]
    ref = load_policy(gpt_name, V)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    rms = []
    for n in rm_names:
        rm = RewardModel(V)
        rm.load_state_dict(torch.load(CKPT / f"{n}.pt"))
        rms.append(rm)
    return ref, rms


def sample_states(d, rng, n, cut_end, ctx):
    """Random history prefixes (lists of event indices, length <= ctx) strictly before the train cutoff."""
    lens = cut_end - d.ustart
    ok = np.where(lens > 1)[0]
    us = rng.choice(ok, size=n, p=lens[ok] / lens[ok].sum())
    rows = []
    for u in us:
        end = rng.integers(d.ustart[u] + 1, cut_end[u] + 1)
        rows.append(np.arange(max(d.ustart[u], end - ctx), end))
    return rows


class Env:
    """Holds the policy context (tokens) and the reward-model context (tokens + feedback) for B*G rollouts."""

    def __init__(self, d, rows, G, scorer, horizon):
        B = len(rows)
        L = CTX
        self.tok = torch.zeros(B * G, L, dtype=torch.long)
        self.rb = torch.zeros_like(self.tok)
        self.lk = torch.zeros_like(self.tok)
        self.org = torch.zeros_like(self.tok)
        self.n = torch.zeros(B * G, dtype=torch.long)
        (tok, rb, lk, org), last = rm_context(d, rows)
        for b in range(B):
            for g in range(G):
                i = b * G + g
                self.tok[i], self.rb[i], self.lk[i], self.org[i] = tok[b], rb[b], lk[b], org[b]
                self.n[i] = last[b] + 1
        self.scorer = scorer

    def append(self, a, p):
        """p [N,3] reward-model probabilities for the appended tracks (ratio, like, cont)."""
        ar = torch.arange(len(a))
        self.tok[ar, self.n] = a
        self.rb[ar, self.n] = simulate_feedback(p[:, 0])
        self.lk[ar, self.n] = (torch.rand(len(a)) < p[:, 1]).long()
        self.org[ar, self.n] = 0
        self.n += 1


def simulate_feedback(p_ratio):
    """Stochastic user simulator: real played ratios are bimodal, so instead of feeding the expected ratio
    (a value rarely seen in real data) the track is 'completed' with probability p, otherwise 'skipped'."""
    done = torch.rand(len(p_ratio)) < p_ratio
    return torch.from_numpy(ratio_bucket(np.where(done.numpy(), 1.0, 0.05)))


def rollout(policy, ref, scorer, d, rows, G, H, reward_kind, item_len_t, lbias_t, gamma=1.0, greedy=False):
    """Sample H tracks for each of len(rows)*G rollouts. Returns actions [N,H], rewards [N,H], probs [N,H,3]."""
    env = Env(d, rows, G, scorer, H)
    N = len(env.n)
    acts, rews, probs = [], [], []
    with torch.no_grad():
        for h in range(H):
            lg = policy_logits(policy, env.tok, last_pos=env.n - 1)
            a = lg.argmax(-1) if greedy else torch.multinomial(F.softmax(lg, -1), 1).squeeze(-1)
            hs = scorer.states(env.tok, env.rb, env.lk, env.org, env.n - 1)
            p = scorer.score(hs, a[:, None])[:, 0]                  # [N,3]
            r = combine(reward_kind, p[:, 0], p[:, 1], p[:, 2], item_len_t[a], lbias_t[a])
            acts.append(a)
            rews.append(r)
            probs.append(p)
            if h < H - 1:
                env.append(a, p)
    return env, torch.stack(acts, 1), torch.stack(rews, 1), torch.stack(probs, 1)


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    d = load_data()
    meta = d.meta
    V = meta["n_tokens"]
    ref, (rm_a, rm_b) = load_models(d, args.gpt, (args.rm_a, args.rm_b))
    policy = copy.deepcopy(ref)
    for p in policy.parameters():
        p.requires_grad_(True)
    pop, ilen, _ = item_stats(d, meta["val_start"])
    edges, means = length_bias(d, meta["val_start"])
    lbias = torch.from_numpy(token_length_bias(edges, means, ilen))
    ilen_t = torch.from_numpy(ilen)
    sa, sb = RMScorer(rm_a, ilen), RMScorer(rm_b, ilen)
    cut = train_end_index(d, meta["val_start"])
    val = EvalSet(d, meta["val_start"], meta["test_start"], max_users=None)
    rng_val = np.random.default_rng(123)
    val_sub = np.sort(rng_val.choice(len(val), min(1500, len(val)), replace=False))
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    ctx = CTX - args.horizon                      # leave room for generated tokens within trained positions
    disc = torch.tensor([args.gamma ** h for h in range(args.horizon)])
    log, t0 = [], time.time()
    H, G, B = args.horizon, args.group, args.batch
    policy.eval()                                     # no dropout: sampling and gradients use the same π_θ
    for step in range(1, args.steps + 1):
        rows = sample_states(d, rng, B, cut, ctx)
        if H == 1:
            ret, kl, ent_mean, loss = single_step_loss(policy, ref, sa, d, rows, G, args, ilen_t, lbias)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            log_step(step, args, ret, kl, ent_mean, t0, log, policy, ref, val, val_sub, sa, sb, d, pop, ilen, lbias)
            continue
        env, acts, rews, _ = rollout(policy, ref, sa, d, rows, G, H, args.reward, ilen_t, lbias)
        ret = (rews * disc).sum(1).view(B, G)                                            # [B,G]
        # one forward over [context + generated] gives log-probs and full distributions at every step
        seq = env.tok.clone()
        first = env.n - (H - 1)                       # index of the first generated token in seq
        hs = policy.hidden(seq)                       # [N,L,d]
        with torch.no_grad():
            hs_ref = ref.hidden(seq)
        N = len(seq)
        ar = torch.arange(N)
        logp, kls, ent_sum = [], [], 0.0
        for h in range(H):
            pos = first - 1 + h                        # position whose output predicts generated token h
            lp = F.log_softmax(policy.logits_at(hs, seq, pos), -1)
            with torch.no_grad():
                lpr = F.log_softmax(ref.logits_at(hs_ref, seq, pos), -1)
            logp.append(lp[ar, acts[:, h]])
            pr = lp.exp()
            kls.append((pr * (lp - lpr)).sum(-1))      # exact KL at the visited state s_h (pathwise term)
            ent_sum = ent_sum + -(pr * lp).sum(-1)
        kl_mat = torch.stack(kls, 1)                   # [N,H]
        # score-function term: action a_h is credited with discounted future rewards AND the KL penalties of
        # the states it leads to (s_{h+1..H}); the pathwise KL at s_h is differentiated directly below
        pen = rews - args.beta * torch.cat([kl_mat[:, 1:].detach(), torch.zeros(N, 1)], 1)
        rtg = torch.stack([(pen[:, h:] * disc[: H - h]).sum(1) * disc[h] for h in range(H)], 1).view(B, G, H)
        base = (rtg.sum(1, keepdim=True) - rtg) / (G - 1)                                 # leave-one-out
        adv = (rtg - base).view(B * G, H)
        pg = -sum((adv[:, h] * logp[h]).mean() for h in range(H))
        kl_sum = (kl_mat * disc[None]).sum(1)
        kl = kl_sum.mean() / H                        # per-step KL (for logging)
        loss = pg + args.beta * kl_sum.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        log_step(step, args, ret, kl, ent_sum.mean().item() / H, t0, log, policy, ref, val, val_sub, sa, sb, d, pop,
                 ilen, lbias)
    torch.save(policy.state_dict(), CKPT / f"{args.name}.pt")
    json.dump(dict(args=vars(args), log=log), open(RES / f"{args.name}_rl_log.json", "w"), indent=1)


def single_step_loss(policy, ref, sa, d, rows, G, args, ilen_t, lbias):
    """H=1 fast path: the G samples share one state, so the policy runs once per state."""
    (tok, rb, lk, org), last = rm_context(d, rows)          # policy and reward model read the same events
    lp = F.log_softmax(policy_logits(policy, tok, last_pos=last), -1)            # [B,V], with grad
    with torch.no_grad():
        lpr = F.log_softmax(policy_logits(ref, tok, last_pos=last), -1)
        acts = torch.multinomial(lp.exp(), G, replacement=True)                  # [B,G] on-policy samples
        p = sa.score(sa.states(tok, rb, lk, org, last), acts)
        r = combine(args.reward, p[..., 0], p[..., 1], p[..., 2], ilen_t[acts], lbias[acts])
    adv = r - (r.sum(1, keepdim=True) - r) / (G - 1)                              # leave-one-out baseline
    pg = -(adv * lp.gather(1, acts)).mean()
    pr = lp.exp()
    kl_b = (pr * (lp - lpr)).sum(-1)                                              # exact KL per state
    ent = -(pr * lp).sum(-1)
    return r, kl_b.mean(), ent.mean().item(), pg + args.beta * kl_b.mean()


def log_step(step, args, ret, kl, ent_mean, t0, log, policy, ref, val, val_sub, sa, sb, d, pop, ilen, lbias):
    if step % 25 == 0:
        print(f"step {step} R {ret.mean():.4f} KL {kl.item():.4f} H {ent_mean:.3f} {time.time() - t0:.0f}s",
              flush=True)
    if step % args.eval_every == 0 or step == args.steps:
        m = quick_eval(policy, ref, val, val_sub, sa, sb, d, pop, ilen, lbias, args.reward)
        m.update(step=step, train_return=float(ret.mean()), train_kl=float(kl), time=time.time() - t0)
        log.append(m)
        print(json.dumps({k: round(v, 4) for k, v in m.items()}), flush=True)


@torch.no_grad()
def quick_eval(policy, ref, es, sub, sa, sb, d, pop, ilen, lbias, kind, n_samples=16):
    """Relevance on a subset of val users + expected reward (sampled) under both reward models + KL."""
    m = {}
    tok = torch.from_numpy(es.ctx[sub])
    last = torch.from_numpy(es.lens[sub] - 1)
    lg = policy_logits(policy, tok, last_pos=last)
    lr_ = policy_logits(ref, tok, last_pos=last)
    lp, lpr = F.log_softmax(lg, -1), F.log_softmax(lr_, -1)
    m["kl_ref"] = float((lp.exp() * (lp - lpr)).sum(-1).mean())
    m["entropy"] = float(-(lp.exp() * lp).sum(-1).mean())
    top = torch.topk(lp, 100, -1).indices.numpy()
    ok = es.next_ok[sub]
    hits10 = np.array([es.next[i] in set(top[j, :10]) for j, i in enumerate(sub)])
    m["next_recall@10"] = float(hits10[ok].mean())
    m["day_recall@10"] = float(np.mean([len(set(top[j, :10]) & es.day[i]) / min(10, len(es.day[i]))
                                        for j, i in enumerate(sub)]))
    m["coverage@10"] = float(len(np.unique(top[:, :10])) / (len(pop) - 2))
    m["repeat_share@10"] = float(np.mean([np.mean([t in es.hist[i] for t in top[j, :10]]) for j, i in enumerate(sub)]))
    m["mean_len@10"] = float(ilen[top[:, :10]].mean())
    rows = [np.arange(*es.ctx_range[i]) for i in sub]
    acts = torch.multinomial(lp.exp(), n_samples, replacement=True)
    ilen_t = torch.from_numpy(ilen)
    for name, sc in (("A", sa), ("B", sb)):
        (t_, rb, lk, org), lst = rm_context(d, rows)
        hs = sc.states(t_, rb, lk, org, lst)
        p = sc.score(hs, acts)
        r = combine(kind, p[..., 0], p[..., 1], p[..., 2], ilen_t[acts], lbias[acts])
        m[f"reward_{name}"] = float(r.mean())
        m[f"ratio_{name}"] = float(p[..., 0].mean())
        p10 = sc.score(hs, torch.from_numpy(top[:, :10]))
        m[f"top10_reward_{name}"] = float(combine(kind, p10[..., 0], p10[..., 1], p10[..., 2],
                                                  ilen_t[top[:, :10]], lbias[top[:, :10]]).mean())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", default="ratio")
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--gpt", default="gpt")
    ap.add_argument("--rm_a", default="rm_a")
    ap.add_argument("--rm_b", default="rm_b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    if args.name is None:
        args.name = f"rl_{args.reward}_b{args.beta:g}_h{args.horizon}"
    train(args)


if __name__ == "__main__":
    main()
