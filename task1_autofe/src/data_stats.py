"""Dataset statistics quoted in §1 of the report -> results/data_stats.json:
split sizes / CTR / random share, simultaneous impressions and the 'label lag' check of §1.3."""
import json
from pathlib import Path

import polars as pl

from data import OUT

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    ev = pl.read_parquet(OUT / "events.parquet", columns=["user_id", "time_ms", "play_time_ms", "is_click",
                                                          "is_rand", "split"])
    out = {"n_events": ev.height}
    out["splits"] = {r["split"]: dict(n=r["len"], ctr=r["is_click"], random_share=r["is_rand"]) for r in
                     ev.group_by("split").agg(pl.len(), pl.col("is_click").mean(), pl.col("is_rand").mean())
                     .iter_rows(named=True)}
    out["ctr_by_traffic"] = {("random" if r["is_rand"] else "feed"): r["is_click"] for r in
                             ev.group_by("is_rand").agg(pl.col("is_click").mean()).iter_rows(named=True)}
    # adjacent impressions of the same user (table is sorted by time; stable sort by user keeps that order)
    p = (ev.sort(["user_id", "time_ms"], maintain_order=True)
         .with_columns(prev_t=pl.col("time_ms").shift(1).over("user_id"),
                       prev_play=pl.col("play_time_ms").shift(1).over("user_id"))
         .filter(pl.col("prev_t").is_not_null()))
    dt = p["time_ms"] - p["prev_t"]
    out["same_ms_pair_share"] = float((dt == 0).mean())
    later = p.filter(pl.col("time_ms") > pl.col("prev_t"))
    # next impression started before the previous one was watched to the end -> its label was not yet known
    out["label_lag_share_among_later_pairs"] = float(
        (later["time_ms"] < later["prev_t"] + later["prev_play"].clip(0, None)).mean())
    json.dump(out, open(RES / "data_stats.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
