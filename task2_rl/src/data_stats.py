"""Dataset statistics quoted in §1 of the report -> results/data_stats.json."""
import json

import numpy as np
import polars as pl

from common import PROC, RES, SESSION_GAP, load_data
from evaluation import EvalSet


def main():
    meta = json.loads((PROC / "meta.json").read_text())
    ev = pl.read_parquet(PROC / "events.parquet", columns=["uid", "ts", "item", "tok", "organic", "ratio", "length",
                                                           "liked", "gap_next"])
    tr = ev.filter(pl.col("ts") < meta["val_start"])
    # session continuation (RM `cont` label): next listen within SESSION_GAP; only events whose outcome is
    # observable before the validation cutoff (same masking as in train_rm.labels)
    gap = pl.col("gap_next").fill_null(-1)
    tr = tr.with_columns(((gap >= 0) & (gap <= SESSION_GAP)).cast(pl.Float64).alias("cont"),
                         (pl.col("ts") + SESSION_GAP < meta["val_start"]).alias("cont_known"))
    te = ev.filter(pl.col("ts") >= meta["test_start"])
    out = {}
    for name, part in (("train", tr), ("all", ev)):
        ul = part.group_by("uid").len()["len"]
        out[f"events_per_user_{name}"] = dict(median=float(ul.median()), p10=float(ul.quantile(0.1)),
                                              p90=float(ul.quantile(0.9)))
    out["organic_share_train"] = float(tr["organic"].mean())
    out["like_rate_train"] = float(tr["liked"].mean())
    out["ratio_eq_100_share_train"] = float((tr["ratio"] >= 0.999).mean())
    out["ratio_le_7pct_share_train"] = float((tr["ratio"] <= 0.07).mean())
    out["ratio_recommended_vs_organic"] = [float(tr.filter(pl.col("organic") == 0)["ratio"].mean()),
                                           float(tr.filter(pl.col("organic") == 1)["ratio"].mean())]
    # repeat listening: share of listens whose (user, track) occurred earlier for the same user
    rep = ev.select(pl.struct(["uid", "item"]).is_first_distinct().alias("first"))["first"]
    out["relisten_share_all"] = float(1 - rep.mean())
    out["test"] = dict(events=te.height, users=int(te["uid"].n_unique()), in_vocab_share=float((te["tok"] >= 2).mean()))
    rows = []
    for lo, hi, label in [(0, 120, "<2 мин"), (120, 180, "2–3 мин"), (180, 240, "3–4 мин"), (240, 300, "4–5 мин"),
                          (300, 600, "5–10 мин"), (600, 1e9, ">10 мин")]:
        for scope, part in (("all_train", tr), ("in_vocab_train", tr.filter(pl.col("tok") >= 2))):
            s = part.filter((pl.col("length") >= lo) & (pl.col("length") < hi))
            rows.append(dict(bucket=label, scope=scope, share=s.height / part.height, n=s.height,
                             mean_ratio=float(s["ratio"].mean()), complete=float((s["ratio"] >= 0.9).mean()),
                             listened_s=float((s["ratio"] * s["length"]).mean()), like=float(s["liked"].mean()),
                             cont=float(s.filter(pl.col("cont_known"))["cont"].mean())))
    out["length_table"] = rows
    # evaluation population of §1/§6.1 (same EvalSet as analysis.TestBed)
    del ev, tr, te
    es = EvalSet(load_data(), meta["test_start"], 2 ** 62)
    out["test_eval"] = dict(users=int(len(es.users)), next_users=int((es.next >= 2).sum()))
    json.dump(out, open(RES / "data_stats.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "length_table"}, indent=1, ensure_ascii=False))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
