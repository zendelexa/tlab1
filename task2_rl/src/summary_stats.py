"""Derived statistics quoted in §6.4–6.6 -> results/summary_stats.json:
* mean ± sample sd (ddof=1) over RL seeds for the replicated configurations;
* RL vs exact optimum at equal KL: the π_β curve (per reward kind, deduplicated by label, reference at KL=0) is
  interpolated in log-KL space at every RL run's KL (also reported with linear-KL interpolation)."""
import json

import numpy as np

from common import RES


def main():
    T = json.load(open(RES / "tilting.json"))
    P = {m["label"]: m for m in json.load(open(RES / "policies_eval.json"))}
    ref = T[0]

    def d(m, k, j):
        return m[f"exp_{k}_reward_{j}"] - ref[f"exp_{k}_reward_{j}"]

    out = {"seeds": {}, "rl_vs_optimum": {}, "counts": {}}
    for base in ("rl_ratio_b0.1_h1", "rl_ratio_b0.03_h1", "rl_cont_b0.03_h5", "rl_ratio_b0.03_h5"):
        runs = [P[base]] + [P[f"{base}_s{s}"] for s in (1, 2)]
        k = runs[0]["reward_kind"]
        stats = {}
        for name, f in (("kl", lambda m: m["kl_ref"]), ("dA", lambda m: d(m, k, "A")), ("dB", lambda m: d(m, k, "B")),
                        ("next_ndcg@10", lambda m: m["next_ndcg@10"]), ("next_recall@10", lambda m: m["next_recall@10"]),
                        ("coverage@10", lambda m: m["coverage@10"]), ("repeat_share@10", lambda m: m["repeat_share@10"]),
                        ("mean_len@10", lambda m: m["mean_len@10"])):
            v = np.array([f(m) for m in runs])
            stats[name] = dict(values=v.tolist(), mean=float(v.mean()), sd=float(v.std(ddof=1)))
        out["seeds"][base] = stats

    def curve(kind):
        seen, rows = set(), []
        for r in T:
            if r.get("reward_kind") == kind and "beta" in r and r["label"] not in seen:
                seen.add(r["label"])
                rows.append(r)
        return [ref] + sorted(rows, key=lambda r: r["kl_ref"])

    for label, m in P.items():
        if label == "reference":
            continue
        k = m["reward_kind"]
        rows = curve(k)
        kl = np.array([max(r["kl_ref"], 1e-6) for r in rows])
        res = {"horizon": m.get("horizon"), "kl": m["kl_ref"]}
        for mode, xs, x in (("log", np.log(kl), np.log(max(m["kl_ref"], 1e-6))), ("lin", kl, m["kl_ref"])):
            ip = {name: float(np.interp(x, xs, np.array([f(r) for r in rows])))
                  for name, f in (("dA", lambda r: d(r, k, "A")), ("dB", lambda r: d(r, k, "B")),
                                  ("ndcg", lambda r: r["next_ndcg@10"]))}
            res[mode] = dict(opt=ip, rl_over_opt_A=d(m, k, "A") / ip["dA"] if ip["dA"] else None,
                             rl_minus_opt_B_rel=d(m, k, "B") / ip["dB"] - 1 if ip["dB"] else None,
                             rl_ndcg_higher=m["next_ndcg@10"] > ip["ndcg"])
        out["rl_vs_optimum"][label] = res
    for mode in ("log", "lin"):
        for h in (1, 5):
            rr = [v for v in out["rl_vs_optimum"].values() if v["horizon"] == h]
            out["counts"][f"{mode}_H{h}"] = dict(
                n=len(rr), rl_below_opt_A=sum(v[mode]["rl_over_opt_A"] < 1 for v in rr),
                rl_not_worse_B=sum(v[mode]["rl_minus_opt_B_rel"] >= 0 for v in rr),
                rl_ndcg_higher=sum(v[mode]["rl_ndcg_higher"] for v in rr),
                rl_over_opt_A_range=[min(v[mode]["rl_over_opt_A"] for v in rr), max(v[mode]["rl_over_opt_A"] for v in rr)])
    json.dump(out, open(RES / "summary_stats.json", "w"), indent=1)
    print(json.dumps(out["counts"], indent=1))
    for b, s in out["seeds"].items():
        print(b, {k: f"{v['mean']:.4f}±{v['sd']:.4f}" for k, v in s.items()})


if __name__ == "__main__":
    main()
