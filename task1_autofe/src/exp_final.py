"""Final comparison: every feature set under one protocol (3 seeds) + paired bootstrap CIs over
(user, day, traffic) groups for the difference to the hand-crafted set (seed-0 predictions)."""
import json
from pathlib import Path

import numpy as np

from data import load
from engine import Spec
from exp_baselines import run
from features import MANUAL, FeatureStore
from model import CATEGORICAL, IDS, fit

RES = Path(__file__).resolve().parents[1] / "results"


def specs_from(fname):
    j = json.load(open(RES / fname))
    return [Spec(p["agg"], p["signal"], tuple(p["key"]), p["filt"], float(p["window"]), float(p["alpha"]))
            for p in j["specs"]]


def group_stats(y, p, groups):
    """Per-group: logloss sum, count, AUC, NDCG@10 and validity (has both classes)."""
    order = np.lexsort((-p, groups))
    g, yy, pp = groups[order], y[order], p[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ends = np.r_[starts[1:], len(g)]
    ll = -(yy * np.log(np.clip(pp, 1e-7, 1)) + (1 - yy) * np.log(np.clip(1 - pp, 1e-7, 1)))
    disc = 1.0 / np.log2(np.arange(2, 12))
    out = np.zeros((len(starts), 5))
    from scipy.stats import rankdata
    for i, (s, e) in enumerate(zip(starts, ends)):
        ys = yy[s:e]
        npos = ys.sum()
        out[i, 0] = ll[s:e].sum()
        out[i, 1] = e - s
        if 0 < npos < e - s:
            r = len(ys) + 1 - rankdata(-pp[s:e])
            out[i, 2] = (r[ys == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(ys) - npos))
            k = min(10, len(ys))
            out[i, 3] = (ys[:k] * disc[:k]).sum() / disc[: min(int(npos), k)].sum()
            out[i, 4] = 1
    return out


def summarize(st, idx):
    s = st[idx]
    v = s[:, 4] == 1
    return np.array([s[:, 0].sum() / s[:, 1].sum(), np.average(s[v, 2], weights=s[v, 1]), s[v, 3].mean()])


def main():
    ev = load()
    fs = FeatureStore(ev)
    sets = {
        "static": ([], ()),
        "static+ids": ([], tuple(IDS)),
        "manual": (list(MANUAL), ()),
        "autofe_logloss": (specs_from("autofe_main_s0.json"), ()),
        "autofe_listwise": (specs_from("autofe_main_s0_listwise.json"), ()),
        "llm_agent": (specs_from("autofe_agent_s0.json"), ()),
    }
    if (RES / "autofe_hybrid_s0.json").exists():
        sets["llm_agent+search"] = (specs_from("autofe_hybrid_s0.json"), ())
    if (RES / "autofe_agent_run2_s0.json").exists():
        sets["llm_agent (2-й запуск)"] = (specs_from("autofe_agent_run2_s0.json"), ())
    sets["autofe_logloss_grid"] = (specs_from("autofe_grid_s0.json"), ())
    union = sets["autofe_logloss"][0] + [s for s in sets["autofe_listwise"][0] if s not in sets["autofe_logloss"][0]]
    sets["autofe_logloss+listwise"] = (union, ())
    sets["autofe_logloss+listwise+ids"] = (union, tuple(IDS))
    sets["manual+ids"] = (list(MANUAL), tuple(IDS))

    te = (ev["split"] == "test").to_numpy()
    tr = (ev["split"] == "train").to_numpy()
    va = (ev["split"] == "val").to_numpy()
    y = ev["is_click"].to_numpy()
    tdf = ev.filter(te)
    groups = (tdf["user_id"].to_numpy().astype(np.int64) * 100000 + tdf["date"].to_numpy() % 100000) * 2 + \
        tdf["is_rand"].to_numpy()
    is_rand = tdf["is_rand"].to_numpy()
    # 3-seed results already computed by earlier experiments (same code, same seeds -> identical numbers)
    existing = {"static": ("baselines.json", "static"), "static+ids": ("baselines.json", "static_ids"),
                "manual": ("baselines.json", "manual"), "autofe_logloss": ("autofe_main_s0.json", "result"),
                "autofe_listwise": ("autofe_main_s0_listwise.json", "result"),
                "llm_agent": ("autofe_agent_s0.json", "result"), "llm_agent+search": ("autofe_hybrid_s0.json", "result"),
                "llm_agent (2-й запуск)": ("autofe_agent_run2_s0.json", "result"),
                "autofe_logloss_grid": ("autofe_grid_s0.json", "result")}
    results, stats = {}, {}
    for name, (specs, extra) in sets.items():
        if name in existing:
            f, k = existing[name]
            results[name] = json.load(open(RES / f))[k]
        else:
            results[name] = run(fs, specs, name, extra=extra)
        Xtr, Xva, Xte = (fs.matrix(specs, m, extra=extra) for m in (tr, va, te))
        bst = fit(Xtr, y[tr], Xva, y[va], params=dict(seed=0), cat=list(CATEGORICAL) + list(extra))
        p = bst.predict(Xte.to_pandas(), num_iteration=bst.best_iteration)
        stats[name] = {sub: group_stats(y[te][m], p[m], groups[m]) for sub, m in
                       (("std", is_rand == 0), ("rand", is_rand == 1))}
    # paired bootstrap over groups: difference to the hand-crafted set
    rng = np.random.default_rng(0)
    boot = {}
    for sub in ("std", "rand"):
        G = len(stats["manual"][sub])
        draws = [rng.integers(0, G, G) for _ in range(1000)]
        base = np.array([summarize(stats["manual"][sub], d) for d in draws])
        for name in sets:
            if name == "manual":
                continue
            other = np.array([summarize(stats[name][sub], d) for d in draws])
            diff = other - base
            point = summarize(stats[name][sub], np.arange(G)) - summarize(stats["manual"][sub], np.arange(G))
            boot.setdefault(name, {})[sub] = {
                m: [float(point[i]), float(np.percentile(diff[:, i], 2.5)), float(np.percentile(diff[:, i], 97.5))]
                for i, m in enumerate(("logloss", "gauc", "ndcg10"))}
            print(name, sub, {m: [round(x, 4) for x in v] for m, v in boot[name][sub].items()}, flush=True)
    json.dump(dict(results=results, bootstrap_vs_manual=boot), open(RES / "final_comparison.json", "w"), indent=1)


if __name__ == "__main__":
    main()
