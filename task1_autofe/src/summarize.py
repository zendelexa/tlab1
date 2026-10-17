"""Tables and figures for the task-1 report."""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
FIG = RES / "figures"
MIN_GAIN = 5e-5


def load(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else None


def agg(run, split, subset, metric):
    v = [s[split][subset][metric] for s in run["seeds"]]
    return float(np.mean(v)), float(np.std(v))


COLS = [("val", "all", "logloss"), ("test", "all", "logloss"), ("test", "all", "auc"), ("test", "std", "gauc"),
        ("test", "std", "ndcg10"), ("test", "rand", "gauc"), ("test", "rand", "ndcg10")]
HEAD = ("| набор признаков | #признаков | val logloss | test logloss | test AUC | GAUC лента | NDCG@10 лента | "
        "GAUC random | NDCG@10 random |\n|---|---|---|---|---|---|---|---|---|")


def row(label, run):
    cells = [label, str(run["n_features"])]
    for split, subset, metric in COLS:
        m, s = agg(run, split, subset, metric)
        cells.append(f"{m:.4f}" + (f" ±{s:.4f}" if s >= 5e-5 else ""))
    return "| " + " | ".join(cells) + " |"


def ci_table(boot):
    lines = ["| набор | Δlogloss лента | ΔGAUC лента | ΔNDCG@10 лента | Δlogloss random | ΔGAUC random | "
             "ΔNDCG@10 random |", "|---|---|---|---|---|---|---|"]
    for name, b in boot.items():
        cells = [name]
        for sub in ("std", "rand"):
            for m in ("logloss", "gauc", "ndcg10"):
                p, lo, hi = b[sub][m]
                sig = "**" if (lo > 0 or hi < 0) else ""
                cells.append(f"{sig}{p:+.4f}{sig} [{lo:+.4f}, {hi:+.4f}]")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def search_curves(b):
    runs = [("autofe_main_s0_search.json", "поиск по logloss", "tab:blue"),
            ("autofe_main_s0_listwise_search.json", "поиск по listwise", "tab:green")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for fname, label, c in runs:
        s = load(fname)
        if not s or "val" not in s["log"][0]:
            continue
        x = [e["n_accepted"] for e in s["log"]]
        axes[0].plot(x, [e["val"]["all"]["logloss"] for e in s["log"]], "o-", color=c, label=label)
        axes[1].plot(x, [e["val"]["std"]["gauc"] for e in s["log"]], "o-", color=c, label=label)
        axes[2].plot(x, [e["val"]["rand"]["gauc"] for e in s["log"]], "o-", color=c, label=label)
    for name, key, c in [("static", "static", "gray"), ("static + ID", "static_ids", "tab:orange"),
                         ("manual", "manual", "tab:red")]:
        for ax, (sub, met) in zip(axes, (("all", "logloss"), ("std", "gauc"), ("rand", "gauc"))):
            ax.axhline(agg(b[key], "val", sub, met)[0], ls="--", color=c, label=name)
    for ax, t in zip(axes, ("val logloss (меньше — лучше)", "val GAUC, обычная лента", "val GAUC, random-показы")):
        ax.set_title(t)
        ax.set_xlabel("число принятых признаков")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "search_curves.png", dpi=140)


def window_sweep():
    w = load("window_sweep.json")
    if not w:
        return
    x = np.array(w["windows_h"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    for ax, base in zip(axes, ("static", "manual")):
        for k, g in w["curves"].items():
            if k.startswith(base + "|"):
                g = np.array(g)
                ax.plot(x, g * 1e3, "o-", ms=3, label=k.split("|", 1)[1])
        ax.axvline(168, color="k", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("окно / полупериод, часы (пунктир — 7 дней)")
        ax.set_ylabel("gain скрининга, ×10⁻³ logloss")
        ax.set_title(f"поверх модели «{base}»")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "window_sweep.png", dpi=140)


def proposal_efficiency():
    """Hit rate (gain >= MIN_GAIN) per round for random / mutation proposals and for the LLM agent runs."""
    s = load("autofe_main_s0_search.json")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if s:
        for src, c in (("random", "tab:gray"), ("mutation", "tab:blue")):
            pts = [(e["round"], e["src_stats"][src]["hit"] / e["src_stats"][src]["n"]) for e in s["log"]
                   if e["src_stats"][src]["n"] > 0]
            ax.plot(*zip(*pts), "o-", color=c, label=f"автопоиск: {'случайные программы' if src == 'random' else 'мутации'}")
    for fname, label, c in (("agent_state.json", "LLM-агент, запуск 1", "tab:purple"),
                            ("agent_state_run2.json", "LLM-агент, запуск 2", "tab:pink")):
        a = load(fname)
        if a:
            ax.plot([h["round"] for h in a["history"]],
                    [np.mean([x["gain"] >= MIN_GAIN for x in h["results"]]) for h in a["history"]], "s-", color=c,
                    label=label)
    ax.set_xlabel("раунд")
    ax.set_ylabel(f"доля кандидатов с gain ≥ {MIN_GAIN:g}")
    ax.set_title("Эффективность генераторов признаков (скрининг одинаковый)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "proposal_efficiency.png", dpi=140)


def seeds_table():
    """Search-seed robustness: each search seed -> its own feature set, each evaluated with 3 LightGBM seeds."""
    lines = ["| вариант поиска | сид поиска | #признаков | кандидатов оценено | val logloss | test logloss | test AUC | "
             "GAUC лента | GAUC random |", "|---|---|---|---|---|---|---|---|---|"]
    agg_rows = {}
    for variant, label in (("main", "непрерывные окна + затухание"), ("grid", "окна только {1ч,1д,7д,∞}")):
        for s in (0, 1, 2):
            r = load(f"autofe_{variant}_s{s}.json")
            if not r:
                continue
            res = r["result"]
            vals = [agg(res, "val", "all", "logloss")[0], agg(res, "test", "all", "logloss")[0],
                    agg(res, "test", "all", "auc")[0], agg(res, "test", "std", "gauc")[0],
                    agg(res, "test", "rand", "gauc")[0]]
            agg_rows.setdefault(label, []).append(vals)
            lines.append(f"| {label} | {s} | {len(r['accepted'])} | {r.get('n_evaluated', '—')} | "
                         + " | ".join(f"{v:.4f}" for v in vals) + " |")
    for label, rows in agg_rows.items():
        a = np.array(rows)
        lines.append(f"| **{label}: среднее ± sd по сидам поиска** | — | — | — | "
                     + " | ".join(f"{m:.4f} ± {s:.4f}" for m, s in zip(a.mean(0), a.std(0, ddof=1) if len(a) > 1
                                                                        else np.zeros(a.shape[1]))) + " |")
    return "\n".join(lines)


def main():
    FIG.mkdir(exist_ok=True)
    b = load("baselines.json")
    fc = load("final_comparison.json")
    out = []
    if fc:
        out.append("### Основная таблица (3 сида, test)\n\n" + HEAD)
        for name, run in fc["results"].items():
            out.append(row(name, run))
        extra = [("manual, tuned LightGBM", b.get("manual_tuned") if b else None)]
        m = load("autofe_main_s0.json")
        if m:
            extra += [("autofe_logloss, tuned LightGBM", m.get("result_tuned")),
                      ("manual ∪ autofe_logloss", m.get("result_union"))]
        for fname, label in (("autofe_random_s0.json", "random DSL (без поиска)"),
                             ("autofe_grid_s0.json", "autofe_logloss, окна только {1ч,1д,7д,∞}")):
            r = load(fname)
            if r:
                extra.append((label, r["result"]))
        for label, run in extra:
            if run:
                out.append(row(label, run))
        out.append("\n### Разница с manual: точечная оценка и 95% ДИ (парный бутстрап по группам user×день)\n\n"
                   + ci_table(fc["bootstrap_vs_manual"]))
    out.append("\n### Разброс автоматического поиска по сидам\n\n" + seeds_table())
    (RES / "summary_tables.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    if b:
        search_curves(b)
    window_sweep()
    proposal_efficiency()


if __name__ == "__main__":
    main()
