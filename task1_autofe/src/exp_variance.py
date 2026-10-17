"""Why do some feature sets help logloss but not within-user ranking?  For every feature: the share of its
variance that lies *within* (user, day, traffic) groups of the test period, i.e. the part that can reorder
videos for the same user. Rank-transformed values are used so that heavy tails do not dominate."""
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

from data import load
from engine import Spec
from features import MANUAL, FeatureStore

RES = Path(__file__).resolve().parents[1] / "results"


def specs_from(fname):
    return [Spec(p["agg"], p["signal"], tuple(p["key"]), p["filt"], float(p["window"]), float(p["alpha"]))
            for p in json.load(open(RES / fname))["specs"]]


def within_share(x, groups):
    x = rankdata(np.nan_to_num(x, nan=-1e12)).astype(np.float64)
    df = pl.DataFrame({"g": groups, "x": x})
    gm = df.group_by("g").agg(pl.col("x").mean().alias("m"), pl.len().alias("n"))
    total = x.var()
    if total == 0:
        return float("nan")
    between = np.average((gm["m"].to_numpy() - x.mean()) ** 2, weights=gm["n"].to_numpy())
    return float(1 - between / total)


def main():
    ev = load()
    fs = FeatureStore(ev)
    te = (ev["split"] == "test").to_numpy()
    t = ev.filter(te)
    groups = (t["user_id"].to_numpy().astype(np.int64) * 100000 + t["date"].to_numpy() % 100000) * 2 + \
        t["is_rand"].to_numpy()
    sets = {"manual": list(MANUAL)}
    for name, f in [("autofe_logloss", "autofe_main_s0.json"), ("autofe_listwise", "autofe_main_s0_listwise.json"),
                    ("llm_agent", "autofe_agent_s0.json"), ("llm_agent_run2", "autofe_agent_run2_s0.json"),
                    ("llm_agent+search", "autofe_hybrid_s0.json")]:
        if (RES / f).exists():
            sets[name] = specs_from(f)
    out = {}
    for name, specs in sets.items():
        shares = [within_share(fs.engine.compute(s, rows=te), groups) for s in specs]
        out[name] = {"median": float(np.nanmedian(shares)), "mean": float(np.nanmean(shares)),
                     "share_gt_0.3": float(np.mean(np.array(shares) > 0.3)),
                     "per_feature": dict(zip([s.name for s in specs], shares))}
        print(f"{name:18s} median within-group share {out[name]['median']:.3f}  mean {out[name]['mean']:.3f}  "
              f"features with >30% within: {out[name]['share_gt_0.3']:.2f}", flush=True)
    json.dump(out, open(RES / "variance_decomposition.json", "w"), indent=1)


if __name__ == "__main__":
    main()
