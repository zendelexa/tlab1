"""Run the automatic feature search and evaluate the resulting feature sets on val/test."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from autofe import random_spec, search
from data import load
from engine import INF, Spec
from exp_baselines import run, tune
from features import MANUAL, FeatureStore
from model import CATEGORICAL, evaluate, fit

RES = Path(__file__).resolve().parents[1] / "results"
GRID = [3600.0, 86400.0, 7 * 86400.0, INF]   # windows a human would typically pick


def val_curve(fs):
    ev = fs.ev
    tr = (ev["split"] == "train").to_numpy()
    va = (ev["split"] == "val").to_numpy()
    y = ev["is_click"].to_numpy()

    def f(specs):
        Xtr, Xva = fs.matrix(specs, tr), fs.matrix(specs, va)
        bst = fit(Xtr, y[tr], Xva, y[va], cat=CATEGORICAL)
        m = evaluate(y[va], bst.predict(Xva.to_pandas(), num_iteration=bst.best_iteration), ev.filter(va))
        return {k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in m.items()}
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main", choices=["main", "grid", "random", "agent", "hybrid"])
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--objective", default="logloss", choices=["logloss", "listwise"])
    ap.add_argument("--agent_state", default="agent_state.json", help="state file of an LLM-agent run")
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    ev = load()
    fs = FeatureStore(ev)
    # the search only ever sees events up to the end of the train period (val/test physically absent)
    fs_search = FeatureStore(ev.filter(pl.col("split").is_in(["warmup", "train"])))
    fs_search.engine.max_groups = 40
    tag = f"autofe_{args.variant}_s{args.seed}" + ("" if args.objective == "logloss" else f"_{args.objective}")
    if args.variant == "agent" and args.agent_state != "agent_state.json":
        tag = "autofe_agent_" + Path(args.agent_state).stem.replace("agent_state_", "") + f"_s{args.seed}"
    out = {"args": vars(args)}
    t0 = time.time()
    if args.variant == "agent":
        st = json.load(open(RES / args.agent_state))
        specs = [Spec(p["agg"], p["signal"], tuple(p["key"]), p["filt"], p["window"], p["alpha"])
                 for p in st["accepted_specs"]]
        out["accepted"] = [s.name for s in specs]
        out["n_evaluated"] = sum(len(h["results"]) for h in st["history"])
    elif args.variant == "hybrid":
        # LLM warm start + evolutionary refinement: the agent's accepted features are the initial model and the
        # first mutation parents; the search then continues exactly like the main one
        st = json.load(open(RES / "agent_state.json"))
        init = [Spec(p["agg"], p["signal"], tuple(p["key"]), p["filt"], p["window"], p["alpha"])
                for p in st["accepted_specs"]]
        for s in init:
            fs_search.values(s)
        specs, hist, log = search(fs_search, rounds=args.rounds, seed=args.seed, init=init, seed_parents=init,
                                  n_random=40, n_mut=110, log_path=RES / f"{tag}_search.json",
                                  eval_fn=val_curve(fs) if args.curve else None)
        out["accepted"] = [s.name for s in specs]
        out["search_log"] = log
        out["n_evaluated"] = len([h for h in hist.values() if h["source"] != "init"])
    elif args.variant == "random":
        rng = np.random.default_rng(args.seed)
        n = len(json.load(open(RES / "autofe_main_s0.json"))["accepted"]) if (RES / "autofe_main_s0.json").exists() \
            else len(MANUAL)
        specs = [random_spec(rng) for _ in range(n)]
        out["accepted"] = [s.name for s in specs]
    else:
        specs, hist, log = search(fs_search, rounds=args.rounds, seed=args.seed,
                                  grid=GRID if args.variant == "grid" else None, objective=args.objective,
                                  log_path=RES / f"{tag}_search.json", eval_fn=val_curve(fs) if args.curve else None)
        del fs_search
        out["accepted"] = [s.name for s in specs]
        out["search_log"] = log
        out["n_evaluated"] = len(hist)
    out["search_time"] = time.time() - t0
    out["specs"] = [dict(agg=s.agg, signal=s.signal, key=list(s.key), filt=s.filt, window=s.window, alpha=s.alpha)
                    for s in specs]
    out["result"] = run(fs, specs, tag)
    if args.tune:
        best, trials = tune(fs, specs)
        out["tuning"] = trials
        out["result_tuned"] = run(fs, specs, tag + "_tuned", params=best["params"])
        out["result_tuned"]["params"] = best["params"]
    if args.variant == "main" and args.seed == 0 and args.objective == "logloss":
        out["result_union"] = run(fs, list(MANUAL) + [s for s in specs if s.name not in {m.name for m in MANUAL}],
                                  tag + "_union")
    json.dump(out, open(RES / f"{tag}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
