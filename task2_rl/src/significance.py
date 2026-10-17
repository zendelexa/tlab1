"""Paired bootstrap 95% CIs for relevance differences (policy − reference) on the test day."""
import json

import numpy as np
import torch

from analysis import TestBed
from common import CKPT, RES, load_policy

RUNS = ["rl_ratio_b0.1_h1", "rl_ratio_b0.03_h1", "rl_ratio_b0.01_h1", "rl_like_b0.03_h1", "rl_cont_b0.03_h1",
        "rl_debiased_b0.03_h1", "rl_ratio_b0.03_h5", "rl_cont_b0.03_h5", "rl2_ratio_b0.03_h1", "rl2_ratio_b0.01_h1",
        "rl2_time_b0.5_h1", "rl2_like_b0.012_h1", "rl2_cont_b0.004_h1", "rl2_debiased_b0.03_h1"]


def per_user(tb, logp):
    es = tb.es
    top = torch.topk(logp, 10, -1).indices.numpy()
    disc = 1.0 / np.log2(np.arange(2, 12))
    ndcg = np.array([disc[list(row).index(t)] if t in row else 0.0 for row, t in zip(top, es.next)])
    day = np.array([len(set(row) & d) / min(10, len(d)) for row, d in zip(top, es.day)])
    return ndcg[es.next_ok], day


def ci(diff, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n_boot, len(diff)))
    b = diff[idx].mean(1)
    return float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--out", default="significance")
    args = ap.parse_args()
    tb = TestBed()
    ref_n, ref_d = per_user(tb, tb.ref_logp)
    out = {}
    for name in args.runs:
        pol = load_policy("gptrep", tb.V)
        pol.load_state_dict(torch.load(CKPT / f"{name}.pt"))
        n, d = per_user(tb, tb.logp_of(pol))
        out[name] = {"next_ndcg@10": ci(n - ref_n), "day_recall@10": ci(d - ref_d)}
        print(name, {k: [round(x, 4) for x in v] for k, v in out[name].items()}, flush=True)
    json.dump(out, open(RES / f"{args.out}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
