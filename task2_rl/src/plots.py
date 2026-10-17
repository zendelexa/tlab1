"""Figures and tables for the task-2 report."""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
FIG = RES / "figures"
COLORS = {"ratio": "tab:blue", "time": "tab:orange", "like": "tab:green", "cont": "tab:red", "debiased": "tab:purple"}


def load(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else None


def _curve(tilt, kind):
    rows, seen = [], set()
    for r in tilt:
        if r.get("reward_kind") == kind and "beta" in r and r["label"] not in seen:
            seen.add(r["label"])
            rows.append(r)
    return sorted(rows, key=lambda r: r["kl_ref"])


def tradeoff(tilt, pol):
    """Per reward: Δreward under RM-A / RM-B and NDCG@10 as functions of KL(π||π_ref);
    lines = closed-form optimum π_β, markers = RL policies (★ H=1, ◆ H=5; filled = RM-A, hollow = RM-B)."""
    ref = tilt[0]
    kinds = list(COLORS)
    fig, axes = plt.subplots(2, len(kinds), figsize=(4.2 * len(kinds), 7.5))
    for j, kind in enumerate(kinds):
        c = COLORS[kind]
        rows = _curve(tilt, kind)
        kl = [max(r["kl_ref"], 1e-4) for r in rows]
        for judge, ls in (("A", "-"), ("B", "--")):
            axes[0, j].plot(kl, [r[f"exp_{kind}_reward_{judge}"] - ref[f"exp_{kind}_reward_{judge}"] for r in rows],
                            ls, marker="o", ms=3, color=c, label=f"π_β, RM-{judge}")
        axes[1, j].plot(kl, [r["next_ndcg@10"] for r in rows], "-o", ms=3, color=c, label="π_β")
        for r in (pol or [])[1:]:
            if r["reward_kind"] != kind:
                continue
            mk = "*" if r.get("horizon", 1) == 1 else "D"
            x = max(r["kl_ref"], 1e-4)
            axes[0, j].scatter(x, r[f"exp_{kind}_reward_A"] - ref[f"exp_{kind}_reward_A"], marker=mk, s=130,
                               color=c, edgecolor="k", zorder=5)
            axes[0, j].scatter(x, r[f"exp_{kind}_reward_B"] - ref[f"exp_{kind}_reward_B"], marker=mk, s=130,
                               facecolor="white", edgecolor=c, linewidth=1.5, zorder=5)
            axes[1, j].scatter(x, r["next_ndcg@10"], marker=mk, s=130, color=c, edgecolor="k", zorder=5)
        axes[0, j].set_title(f"награда «{kind}»")
        for i in range(2):
            axes[i, j].set_xscale("log")
            axes[i, j].set_xlabel("KL(π || π_ref), нат")
            axes[i, j].grid(alpha=0.3)
        axes[1, j].axhline(ref["next_ndcg@10"], color="k", ls=":", lw=1)
    axes[0, 0].set_ylabel("прирост ожидаемой награды")
    axes[1, 0].set_ylabel("next-item NDCG@10")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Линии — оптимум π_β ∝ π_ref·exp(r/β) (сплошная — RM-A, по которой оптимизируем; пунктир — "
                 "независимая RM-B). ★ — RL (H=1), ◆ — RL (H=5); закрашенные — RM-A, полые — RM-B.", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "tradeoff.png", dpi=130)


def diagnostics(tilt, pol=None):
    ref = tilt[0]
    keys = [("mean_len@10", "средняя длина трека в топ-10, с"), ("repeat_share@10", "доля уже слушанных в топ-10"),
            ("coverage@10", "покрытие каталога (топ-10)"), ("mean_logpop@10", "log-популярность топ-10"),
            ("artists_per10", "разных артистов в топ-10"), ("hits10_actual_ratio",
                                                            "реальная доля прослушивания угаданных треков")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, (k, title) in zip(axes.ravel(), keys):
        for kind, c in COLORS.items():
            rows = _curve(tilt, kind)
            ax.plot([max(r["kl_ref"], 1e-4) for r in rows], [r[k] for r in rows], "o-", color=c, ms=3, label=kind)
            for r in (pol or [])[1:]:
                if r["reward_kind"] == kind:
                    ax.scatter(max(r["kl_ref"], 1e-4), r[k], marker="*" if r.get("horizon", 1) == 1 else "D",
                               s=110, color=c, edgecolor="k", zorder=5)
        ax.axhline(ref[k], color="k", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("KL(π || π_ref), нат")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "diagnostics.png", dpi=130)


def rl_curves(names):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    cmap = plt.get_cmap("tab20")
    for i, n in enumerate(names):
        for ax in axes:
            ax.set_prop_cycle(None)
        log = load(f"{n}_rl_log.json")
        if not log:
            continue
        L = log["log"]
        s = [e["step"] for e in L]
        c = cmap(i % 20)
        axes[0].plot(s, [e["reward_A"] for e in L], "o-", ms=3, color=c, label=n)
        axes[1].plot(s, [e["reward_B"] for e in L], "o-", ms=3, color=c, label=n)
        axes[2].plot(s, [e["kl_ref"] for e in L], "o-", ms=3, color=c, label=n)
        axes[3].plot(s, [e["next_recall@10"] for e in L], "o-", ms=3, color=c, label=n)
    for ax, t in zip(axes, ("награда ratio по RM-A (val)", "награда ratio по RM-B (val)", "KL к π_ref (val)",
                            "next Recall@10 (val)")):
        ax.set_title(t)
        ax.set_xlabel("шаг RL")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "rl_curves.png", dpi=140)


MS_SHOW = {"gptrep": ("π_ref", "k"), "rl2_ratio_b0.01_h1": ("RL ratio β=0.01", "tab:blue"),
           "rl2_ratio_b0.03_h1": ("RL ratio β=0.03", "tab:cyan"), "rl_ratio_b0.03_h5": ("RL ratio H=5", "tab:purple"),
           "rl_cont_b0.03_h5": ("RL cont H=5", "tab:red")}


def multistep(ms):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    x = np.arange(1, 6)
    for key, r in ms.items():
        name, mode = key.split("|")
        if name not in MS_SHOW:
            continue
        label, c = MS_SHOW[name]
        ls = "-" if mode == "greedy" else "--"
        axes[0].plot(x, r["recall10_free"], "o" + ls, ms=3, color=c, label=f"{label} ({mode})")
        if mode == "sample":
            axes[0].plot(x, r["recall10_tf"], ":", color=c, lw=1)
        axes[1].plot(x, r["kl_free_tf"], "o" + ls, ms=3, color=c)
        axes[3].plot(x, r["self_repeat"], "o" + ls, ms=3, color=c)
    # echo-chamber control: same generated tracks scored in the simulated vs the real context
    for name, c in (("gptrep", "k"), ("rl2_ratio_b0.01_h1", "tab:blue")):
        r = ms.get(f"{name}|sample")
        if r is None:
            continue
        lab = MS_SHOW[name][0]
        axes[2].errorbar(x, r["reward_B"], yerr=1.96 * np.array(r["reward_B_se"]), fmt="o-", ms=3, color=c,
                         capsize=2, label=f"{lab}: свой трек, симулированный контекст")
        axes[2].errorbar(x, r["reward_gen_realctx_B"], yerr=1.96 * np.array(r["reward_gen_realctx_B_se"]), fmt="s--",
                         ms=3, color=c, capsize=2, label=f"{lab}: тот же трек, реальный контекст")
    r = ms["gptrep|sample"]
    axes[2].errorbar(x, r["reward_real_tf_B"], yerr=1.96 * np.array(r["reward_real_tf_B_se"]), fmt="^:", ms=3,
                     color="tab:green", capsize=2, label="реальный трек, реальный контекст")
    titles = ("Recall@10 реального h-го трека\n(линии — свой контекст, точки ··· — реальный контекст)",
              "KL(π(·|свой контекст) || π(·|реальный))",
              "предсказанная награда шага h (RM-B, ratio) и контроли\n(95% ДИ)",
              "доля треков, уже сгенерированных ранее")
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("шаг генерации h")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    axes[2].legend(fontsize=6)
    fig.suptitle("Генерация на 5 шагов вперёд: сплошные — greedy, пунктир — сэмплирование", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "multistep.png", dpi=130)


def table(rows, keys):
    head = "| политика | " + " | ".join(keys) + " |"
    sep = "|---" * (len(keys) + 1) + "|"
    out = [head, sep]
    seen = set()
    for r in rows:
        if r["label"] in seen:          # tilting.json contains a couple of re-evaluated duplicates
            continue
        seen.add(r["label"])
        out.append(f"| {r['label']} | " + " | ".join(f"{r[k]:.4f}" if isinstance(r.get(k), float) else str(r.get(k))
                                                    for k in keys) + " |")
    return "\n".join(out)


def main():
    FIG.mkdir(exist_ok=True)
    tilt, pol, ms = load("tilting.json"), load("policies_eval.json"), load("multistep.json")
    if tilt:
        tradeoff(tilt, pol)
        diagnostics(tilt, pol)
    import glob
    names = sorted(Path(p).name.replace("_rl_log.json", "") for p in glob.glob(str(RES / "*_rl_log.json")))
    names = [n for n in names if "_ratio_" in n]          # same reward scale -> comparable curves
    if names:
        rl_curves(names)
    if ms:
        multistep(ms)
    keys = ["next_recall@10", "next_ndcg@10", "day_recall@10", "exp_reward_A", "exp_reward_B", "kl_ref",
            "coverage@10", "repeat_share@10", "mean_len@10", "mean_logpop@10", "artists_per10", "hits10_actual_ratio"]
    md = []
    if pol:
        md.append(table(pol, keys))
    if tilt:
        md.append(table(tilt, keys))
    (RES / "tables.md").write_text("\n\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
