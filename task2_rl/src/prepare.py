"""Prepare Yambda-50M listens for sequential modelling.

Output (data/processed/):
  events.parquet  -- one row per listen, sorted by (uid, ts):
                     uid, ts, item, tok, organic, ratio, length, liked, gap_next
  meta.json       -- split boundaries, vocab size, special tokens
  vocab.npy       -- raw item_id for every token >= N_SPECIAL

Split (global temporal, no leakage):
  train : ts <  val_start
  val   : val_start  <= ts < test_start   (1 day)
  test  : test_start <= ts                (last day)
Vocabulary and all statistics are computed on the train part only.
"""
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
DAY = 86400
VOCAB_SIZE = 30000
PAD, UNK = 0, 1
N_SPECIAL = 2
LIKE_WINDOW = 600  # a like within +-10 min of a listen of the same track is attributed to it


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lis = pl.read_parquet(RAW / "listens.parquet").with_row_index("rid")  # file is sorted by (uid, timestamp)
    t_max = int(lis["timestamp"].max())
    test_start = t_max - DAY + 1
    val_start = test_start - DAY

    # vocabulary: most listened tracks in the train period
    cnt = (lis.filter(pl.col("timestamp") < val_start).group_by("item_id").len()
              .sort(["len", "item_id"], descending=[True, False]).head(VOCAB_SIZE))
    vocab = cnt["item_id"].to_numpy()
    tok_map = pl.DataFrame({"item_id": vocab, "tok": np.arange(N_SPECIAL, N_SPECIAL + len(vocab), dtype=np.int32)})

    # likes -> nearest listen of the same (uid, item) within LIKE_WINDOW
    likes = (pl.read_parquet(RAW / "likes.parquet").select("uid", "item_id", pl.col("timestamp").alias("like_ts"))
               .sort("like_ts"))
    keys = lis.select("rid", "uid", "item_id", "timestamp").sort("timestamp")
    m = likes.join_asof(keys, left_on="like_ts", right_on="timestamp", by=["uid", "item_id"],
                        strategy="nearest", check_sortedness=False)
    m = m.filter(pl.col("rid").is_not_null() &
                 ((pl.col("like_ts").cast(pl.Int64) - pl.col("timestamp").cast(pl.Int64)).abs() <= LIKE_WINDOW))
    # never attribute a like across a split boundary (the label must be observable on the listen's side)
    for cut in (val_start, test_start):
        m = m.filter((pl.col("like_ts") < cut) == (pl.col("timestamp") < cut))
    liked_rids = m["rid"].unique()

    ev = (lis.join(tok_map, on="item_id", how="left", maintain_order="left")
             .sort("rid")
             .with_columns(tok=pl.col("tok").fill_null(UNK).cast(pl.Int32),
                           liked=pl.col("rid").is_in(liked_rids.implode()).cast(pl.Int8),
                           ratio=(pl.col("played_ratio_pct").cast(pl.Float32).clip(0, 100) / 100.0),
                           gap_next=(pl.col("timestamp").shift(-1).over("uid").cast(pl.Int64)
                                     - pl.col("timestamp").cast(pl.Int64)))
             .select(pl.col("uid"), pl.col("timestamp").alias("ts"), pl.col("item_id").alias("item"), "tok",
                     pl.col("is_organic").alias("organic"), "ratio",
                     pl.col("track_length_seconds").alias("length"), "liked", "gap_next"))
    ev.write_parquet(OUT / "events.parquet")
    np.save(OUT / "vocab.npy", vocab)

    tr = ev.filter(pl.col("ts") < val_start)
    meta = dict(t_max=t_max, val_start=val_start, test_start=test_start, vocab_size=int(len(vocab)),
                n_tokens=int(len(vocab) + N_SPECIAL), pad=PAD, unk=UNK, n_special=N_SPECIAL,
                n_events=ev.height, n_train=tr.height,
                n_val=ev.filter((pl.col("ts") >= val_start) & (pl.col("ts") < test_start)).height,
                n_test=ev.filter(pl.col("ts") >= test_start).height,
                train_unk_share=float((tr["tok"] == UNK).mean()),
                test_unk_share=float((ev.filter(pl.col("ts") >= test_start)["tok"] == UNK).mean()),
                like_share=float(ev["liked"].mean()), n_users=int(ev["uid"].n_unique()))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
