"""Evaluation of policies on the held-out test day under one protocol.

* relevance           : Recall/NDCG@K for the next track and for the day's set of tracks
* reward              : exact expected reward E_{a~π}[r(s,a)] over the whole catalogue under reward model A
                        (the one optimised) and B (held-out judge), plus top-10 averages
* behaviour           : KL to the reference, entropy, coverage, concentration, popularity, track length,
                        repeat share, artist diversity
* real outcomes       : actual played ratio / like rate of recommended tracks the user really played that day
* closed-form tilting : π_β ∝ π_ref · exp(r_A / β)  (the optimum of the single-step KL-regularised objective)
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from common import CKPT, PROC, RES, ROOT, item_stats, load_data, make_gpt, policy_logits
from evaluation import EvalSet
from rewards import REWARDS, RMScorer, combine, length_bias, rm_context, token_length_bias
from rl import load_models

CACHE = ROOT / "data" / "cache"
CHUNK = 256


class TestBed:
    def __init__(self, gpt="gptrep", rms=("rm_a", "rm_b")):
        CACHE.mkdir(parents=True, exist_ok=True)
        self.d = d = load_data()
        meta = d.meta
        self.V = meta["n_tokens"]
        self.es = EvalSet(d, meta["test_start"], 2 ** 62)
        self.ref, rm_models = load_models(d, gpt, rms)
        # statistics known at deployment time (before the test day)
        self.pop, self.ilen, _ = item_stats(d, meta["test_start"])
        edges, means = length_bias(d, meta["val_start"])
        self.lbias = token_length_bias(edges, means, self.ilen)
        self.scorers = {n: RMScorer(m, self.ilen) for n, m in zip(("A", "B"), rm_models)}
        self.scores = {n: self._full_scores(n, rms[i]) for i, n in enumerate(("A", "B"))}
        self.ref_logp = self.logp_of(self.ref)
        self.artist = self._artists()
        self._actual = self._actual_outcomes()

    def _full_scores(self, n, rm_name):
        """Full-catalogue RM probabilities for every test user, cached as float32. The cache key includes the
        checkpoint's mtime and the cutoff; the file is written to a temp name and atomically renamed."""
        stamp = int(os.path.getmtime(CKPT / f"{rm_name}.pt"))
        path = CACHE / f"{rm_name}_{stamp}_{self.d.meta['test_start']}_scores.npy"
        if path.exists():
            return np.load(path, mmap_mode="r")
        es, d, sc = self.es, self.d, self.scorers[n]
        tmp = path.with_suffix(".tmp.npy")
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(es), self.V, 3))
        for i in range(0, len(es), CHUNK):
            rows = [np.arange(*es.ctx_range[j]) for j in range(i, min(i + CHUNK, len(es)))]
            (t, rb, lk, org), last = rm_context(d, rows)
            h = sc.states(t, rb, lk, org, last)
            out[i:i + len(rows)] = sc.score_all(h).numpy()
            print(f"scored {i + len(rows)}/{len(es)} users with RM {n}", flush=True)
        out.flush()
        del out
        os.replace(tmp, path)
        return np.load(path, mmap_mode="r")

    def _artists(self):
        a = pl.read_parquet(ROOT / "data" / "raw" / "artist_item_mapping.parquet").group_by("item_id").agg(
            pl.col("artist_id").first())
        vocab = np.load(PROC / "vocab.npy")
        m = dict(zip(a["item_id"].to_list(), a["artist_id"].to_list()))
        art = np.zeros(self.V, dtype=np.int64)
        art[2:] = [m.get(int(x), -1) for x in vocab]
        return art

    def _actual_outcomes(self):
        """(user, track) -> (played ratio, liked) of the first listen of that track during the test day."""
        d, es = self.d, self.es
        out = []
        for u in es.users:
            s, e = d.ustart[u], d.uend[u]
            a = s + np.searchsorted(d.ts[s:e], d.meta["test_start"], "left")
            seen = {}
            for j in range(a, e):
                t = int(d.tok[j])
                if t >= 2 and t not in seen:
                    seen[t] = (float(d.ratio[j]), float(d.liked[j]))
            out.append(seen)
        return out

    @torch.no_grad()
    def logp_of(self, model):
        es = self.es
        out = torch.empty(len(es), self.V)
        model.eval()
        for i in range(0, len(es), 512):
            tok = torch.from_numpy(es.ctx[i:i + 512])
            last = torch.from_numpy(es.lens[i:i + 512] - 1)
            out[i:i + 512] = F.log_softmax(policy_logits(model, tok, last_pos=last), -1)
        return out

    def reward_matrix(self, n, kind, sl):
        S = torch.from_numpy(np.asarray(self.scores[n][sl], dtype=np.float32))
        return combine(kind, S[..., 0], S[..., 1], S[..., 2], torch.from_numpy(self.ilen)[None],
                       torch.from_numpy(self.lbias)[None])

    def evaluate(self, logp_fn, kind="ratio", label=""):
        """logp_fn(slice) -> log-probabilities [n, V] for users in the slice."""
        es = self.es
        tops, agg = [], {}

        def add(k, v):
            agg[k] = agg.get(k, 0.0) + float(v)
        for i in range(0, len(es), CHUNK):
            sl = slice(i, min(i + CHUNK, len(es)))
            lp = logp_fn(sl)
            p = lp.exp()
            tops.append(torch.topk(lp, 100, -1).indices.numpy())
            add("kl_ref", (p * (lp - self.ref_logp[sl])).sum(-1).sum())
            add("entropy", -(p * lp).sum(-1).sum())
            add("exp_len", (p * torch.from_numpy(self.ilen)[None]).sum())
            add("exp_logpop", (p * torch.log1p(torch.from_numpy(self.pop.astype(np.float32)))[None]).sum())
            for n in ("A", "B"):
                S = torch.from_numpy(np.asarray(self.scores[n][sl], dtype=np.float32))
                add(f"exp_reward_{n}", (p * self.reward_matrix(n, kind, sl)).sum())
                for k2 in REWARDS:
                    add(f"exp_{k2}_reward_{n}", (p * self.reward_matrix(n, k2, sl)).sum())
                for j, h in enumerate(("ratio", "like", "cont")):
                    add(f"exp_{h}_{n}", (p * S[..., j]).sum())
                add(f"exp_time_{n}", (p * S[..., 0] * torch.from_numpy(self.ilen)[None]).sum())
        U = len(es)
        m = {k: v / U for k, v in agg.items()}
        top = np.concatenate(tops)
        m.update(es.metrics(top, self.pop, self.ilen, self.V - 2))
        # top-10 under each reward model
        t10 = torch.from_numpy(top[:, :10])
        for n in ("A", "B"):
            vals = []
            for i in range(0, U, CHUNK):
                sl = slice(i, min(i + CHUNK, U))
                R = self.reward_matrix(n, kind, sl)
                vals.append(torch.gather(R, 1, t10[sl]))
            m[f"top10_reward_{n}"] = float(torch.cat(vals).mean())
        m["artists_per10"] = float(np.mean([len(set(self.artist[r])) for r in top[:, :10]]))
        # real outcomes for recommended tracks that the user actually played on the test day
        rr, ll, nh = [], [], 0
        for row, act in zip(top[:, :10], self._actual):
            for t in row:
                if int(t) in act:
                    rr.append(act[int(t)][0])
                    ll.append(act[int(t)][1])
        m["hits10_n"] = len(rr)
        m["hits10_actual_ratio"] = float(np.mean(rr)) if rr else float("nan")
        m["hits10_actual_like"] = float(np.mean(ll)) if ll else float("nan")
        m["label"] = label
        m["reward_kind"] = kind
        return m

    def tilted(self, kind, beta, n="A"):
        def f(sl):
            return F.log_softmax(self.ref_logp[sl] + self.reward_matrix(n, kind, sl) / beta, -1)
        return f

    def model_logp(self, model):
        lp = self.logp_of(model)
        return lambda sl: lp[sl]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt", default="gptrep", help="reference policy (must be the one RL starts from)")
    ap.add_argument("--pairs", default="", help="only evaluate these kind:beta pairs and append to tilting.json")
    args = ap.parse_args()
    tb = TestBed(gpt=args.gpt)
    if args.pairs:
        res = json.load(open(RES / "tilting.json"))
        for pair in args.pairs.split(","):
            kind, beta = pair.split(":")
            m = tb.evaluate(tb.tilted(kind, float(beta)), kind, f"tilt_{kind}_b{float(beta):g}")
            m["beta"] = float(beta)
            res.append(m)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}), flush=True)
        json.dump(res, open(RES / "tilting.json", "w"), indent=1)
        return
    res = []
    base = tb.evaluate(lambda sl: tb.ref_logp[sl], "ratio", "reference")
    res.append(base)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in base.items()}), flush=True)
    for kind in ("ratio", "time", "like", "cont", "debiased"):
        for beta in (1.0, 0.3, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005):
            m = tb.evaluate(tb.tilted(kind, beta), kind, f"tilt_{kind}_b{beta:g}")
            m["beta"] = beta
            res.append(m)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
                              if k in ("label", "next_ndcg@10", "day_recall@10", "exp_reward_A", "exp_reward_B",
                                       "kl_ref", "coverage@10", "mean_len@10", "repeat_share@10")}), flush=True)
    json.dump(res, open(RES / "tilting.json", "w"), indent=1)


if __name__ == "__main__":
    main()
