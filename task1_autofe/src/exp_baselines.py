"""Baselines: static-only, static+IDs, hand-crafted features (default and tuned LightGBM)."""
import json
import time
from pathlib import Path

import numpy as np

from data import load
from features import MANUAL, FeatureStore
from model import CATEGORICAL, IDS, evaluate, fit

RES = Path(__file__).resolve().parents[1] / "results"
SEEDS = (0, 1, 2)


def run(fs, specs, name, params=None, extra=(), cat=CATEGORICAL, seeds=SEEDS):
    ev = fs.ev
    tr = (ev["split"] == "train").to_numpy()
    va = (ev["split"] == "val").to_numpy()
    te = (ev["split"] == "test").to_numpy()
    Xtr, Xva, Xte = (fs.matrix(specs, m, extra=extra) for m in (tr, va, te))
    y = ev["is_click"].to_numpy()
    res = {"name": name, "n_features": Xtr.width, "seeds": []}
    for seed in seeds:
        t0 = time.time()
        bst = fit(Xtr, y[tr], Xva, y[va], params=dict(params or {}, seed=seed), cat=list(cat) + list(extra))
        r = {"seed": seed, "best_iter": bst.best_iteration, "time": time.time() - t0,
             "val": evaluate(y[va], bst.predict(Xva.to_pandas(), num_iteration=bst.best_iteration), ev.filter(va)),
             "test": evaluate(y[te], bst.predict(Xte.to_pandas(), num_iteration=bst.best_iteration), ev.filter(te))}
        res["seeds"].append(r)
        print(name, seed, json.dumps({s: {k: round(v, 4) for k, v in r[s]["all"].items()} for s in ("val", "test")}),
              flush=True)
    if seeds:
        imp = dict(zip(Xtr.columns, bst.feature_importance("gain").tolist()))
        res["importance"] = dict(sorted(imp.items(), key=lambda x: -x[1])[:60])
    return res


def tune(fs, specs, n_trials=12, seed=0):
    """Random search of LightGBM hyper-parameters on the validation split (val logloss)."""
    ev = fs.ev
    tr = (ev["split"] == "train").to_numpy()
    va = (ev["split"] == "val").to_numpy()
    Xtr, Xva = fs.matrix(specs, tr), fs.matrix(specs, va)
    y = ev["is_click"].to_numpy()
    rng = np.random.default_rng(seed)
    space = dict(num_leaves=[15, 31, 63, 127, 255], learning_rate=[0.05, 0.1],
                 min_data_in_leaf=[20, 100, 400, 1000], feature_fraction=[0.4, 0.6, 0.8, 1.0],
                 lambda_l2=[0.0, 1.0, 10.0, 50.0], bagging_fraction=[0.7, 0.85, 1.0])
    trials = []
    for i in range(n_trials):
        p = {k: v[rng.integers(len(v))] for k, v in space.items()}
        p = {k: (float(v) if isinstance(v, (np.floating, float)) else int(v)) for k, v in p.items()}
        bst = fit(Xtr, y[tr], Xva, y[va], params=p, cat=CATEGORICAL)
        ll = bst.best_score["valid_0"]["binary_logloss"]
        trials.append(dict(params=p, val_logloss=ll, iters=bst.best_iteration))
        print("trial", i, p, round(ll, 5), bst.best_iteration, flush=True)
    return min(trials, key=lambda t: t["val_logloss"]), trials


def main():
    RES.mkdir(exist_ok=True)
    ev = load()
    fs = FeatureStore(ev)
    t0 = time.time()
    for s in MANUAL:
        fs.values(s)
    print(f"computed {len(MANUAL)} manual features in {time.time() - t0:.1f}s", flush=True)
    out = {}
    out["static"] = run(fs, [], "static")
    out["static_ids"] = run(fs, [], "static+ids", extra=IDS)
    out["manual"] = run(fs, MANUAL, "manual")
    best, trials = tune(fs, MANUAL)
    out["manual_tuning"] = trials
    out["manual_tuned"] = run(fs, MANUAL, "manual_tuned", params=best["params"])
    out["manual_tuned"]["params"] = best["params"]
    json.dump(out, open(RES / "baselines.json", "w"), indent=1)


if __name__ == "__main__":
    main()
