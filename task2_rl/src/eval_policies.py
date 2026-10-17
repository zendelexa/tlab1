"""Evaluate the reference policy and RL-fine-tuned checkpoints on the test day under one protocol."""
import argparse
import json

import torch

from analysis import TestBed
from common import CKPT, RES, load_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="gptrep")
    ap.add_argument("--runs", nargs="+", required=True, help="RL checkpoint names, e.g. rl_ratio_b0.05_h1")
    ap.add_argument("--out", default="policies_eval")
    args = ap.parse_args()
    tb = TestBed(gpt=args.base)
    res = [tb.evaluate(lambda sl: tb.ref_logp[sl], "ratio", "reference")]
    for name in args.runs:
        cfg = json.load(open(RES / f"{name}_rl_log.json"))["args"]
        assert cfg["gpt"] == args.base, f"{name} was fine-tuned from {cfg['gpt']}, not {args.base}"
        pol = load_policy(args.base, tb.V)
        pol.load_state_dict(torch.load(CKPT / f"{name}.pt"))
        m = tb.evaluate(tb.model_logp(pol), cfg["reward"], name)
        m.update(beta=cfg["beta"], horizon=cfg["horizon"])
        res.append(m)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}), flush=True)
    json.dump(res, open(RES / f"{args.out}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
