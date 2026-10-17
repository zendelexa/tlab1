"""Generate the report appendices (feature lists per method, SQL examples) from the result files."""
import json
from collections import Counter
from pathlib import Path

from engine import INF, Spec, fmt_window
from features import MANUAL

RES = Path(__file__).resolve().parents[1] / "results"

SOURCES = [("Ручной набор (MANUAL)", None),
           ("AutoFE, цель logloss, сид 0", "autofe_main_s0.json"),
           ("AutoFE, цель listwise", "autofe_main_s0_listwise.json"),
           ("LLM-агент, запуск 1", "autofe_agent_s0.json"),
           ("LLM-агент, запуск 2", "autofe_agent_run2_s0.json"),
           ("LLM-агент + поиск (гибрид)", "autofe_hybrid_s0.json"),
           ("AutoFE, окна из сетки, сид 0", "autofe_grid_s0.json")]


def specs_of(fname):
    if fname is None:
        return list(MANUAL)
    p = RES / fname
    if not p.exists():
        return None
    return [Spec(d["agg"], d["signal"], tuple(d["key"]), d["filt"], float(d["window"]), float(d["alpha"]))
            for d in json.load(open(p))["specs"]]


def describe(specs):
    """Short structural profile of a feature set."""
    agg = Counter(s.agg for s in specs)
    filt = Counter(s.filt for s in specs)
    item_side = sum(any(k in ("video_id", "author_id", "music_id", "tag", "video_type", "upload_type",
                              "music_type", "dur_bucket") for k in s.key) for s in specs)
    user_only = sum(all(k in ("user_id", "user_active_degree", "is_video_author", "follow_user_num_range",
                              "fans_user_num_range", "register_days_range", "is_lowactive_period", "tab",
                              "is_rand", "hour") for k in s.key) for s in specs)
    windows = [s.window for s in specs if s.agg != "recency"]
    round_w = sum(w in (3600.0, 86400.0, 7 * 86400.0, INF) for w in windows)
    return (f"{len(specs)} признаков; агрегаты {dict(agg)}; фильтры {dict(filt)}; "
            f"с item-side колонкой в ключе: {item_side}; только user/context ключ: {user_only}; "
            f"«круглых» окон (1ч/1д/7д/∞): {round_w} из {len(windows)}")


def main():
    out = ["## Приложение A. Списки признаков по методам\n"]
    for title, fname in SOURCES:
        specs = specs_of(fname)
        if specs is None:
            continue
        out.append(f"### {title}\n\n*{describe(specs)}*\n")
        out.append("| # | признак | агрегат | сигнал | ключ | фильтр | окно / полупериод | α |\n|---|---|---|---|---|---|---|---|")
        for i, s in enumerate(specs, 1):
            w = "—" if s.agg == "recency" else fmt_window(s.window)
            a = f"{s.alpha:g}" if s.agg in ("rate", "drate") else "—"
            out.append(f"| {i} | `{s.name.replace(chr(124), chr(92) + chr(124))}` | {s.agg} | {s.signal} | {'×'.join(s.key)} | {s.filt} | {w} | {a} |")
        out.append("")
    out.append("## Приложение B. Как признаки выглядят в SQL (DuckDB, `Spec.to_sql()`)\n")
    ex = [MANUAL[33]]
    for fname in ("autofe_main_s0.json", "autofe_main_s0_listwise.json", "autofe_agent_s0.json"):
        sp = specs_of(fname)
        if sp:
            ex.append(sp[0])
    out.append("```sql")
    for s in ex:
        out.append(f"-- {s.name}\n{s.to_sql()};\n")
    out.append("```")
    (RES / "appendix.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out)[:3000])


if __name__ == "__main__":
    main()
