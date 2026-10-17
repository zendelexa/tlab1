"""Build a single chronologically ordered event table from KuaiRand-Pure.

Every row is an impression (user, video, time, labels). Rows from 2022-04-15 on are modelling rows;
earlier rows are history only. All history features are computed strictly from events with an
earlier timestamp (see engine.py), so splits only need to separate modelling rows by date.
"""
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "KuaiRand-Pure" / "data"
OUT = ROOT / "data" / "processed"

SPLITS = {  # inclusive local dates
    "warmup": (20220408, 20220414),
    "train": (20220415, 20220430),
    "val": (20220501, 20220504),
    "test": (20220505, 20220508),
}
# inner split of the train period used only by the feature search
SEARCH = {"fit": (20220415, 20220426), "screen_a": (20220427, 20220428), "screen_b": (20220429, 20220430)}

LABELS = ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate", "long_view",
          "is_profile_enter"]
USER_CAT = ["user_active_degree", "is_lowactive_period", "is_live_streamer", "is_video_author",
            "follow_user_num_range", "fans_user_num_range", "friend_user_num_range", "register_days_range"]
USER_NUM = ["follow_user_num", "fans_user_num", "friend_user_num", "register_days"] + [f"onehot_feat{i}" for i in
                                                                                      range(18)]
VIDEO_CAT = ["video_type", "upload_type", "music_type", "tag", "dur_bucket"]
VIDEO_NUM = ["video_duration", "server_width", "server_height", "n_tags"]


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    logs = [pl.read_csv(RAW / f) for f in ["log_standard_4_08_to_4_21_pure.csv",
                                           "log_standard_4_22_to_5_08_pure.csv",
                                           "log_random_4_22_to_5_08_pure.csv"]]
    ev = pl.concat(logs).sort(["time_ms", "user_id", "video_id"])

    vid = pl.read_csv(RAW / "video_features_basic_pure.csv", infer_schema_length=10000)
    vid = vid.with_columns(
        tag=pl.col("tag").fill_null("-1").str.split(",").list.first().cast(pl.Int64),
        n_tags=pl.col("tag").fill_null("").str.split(",").list.len(),
        upload_ts=pl.col("upload_dt").str.to_date().cast(pl.Datetime("ms")).cast(pl.Int64),
        dur_bucket=(pl.col("video_duration").fill_null(0) / 1000).log1p().mul(2).floor().cast(pl.Int64),
        video_type=pl.col("video_type").cast(pl.Categorical).to_physical().cast(pl.Int64),
        upload_type=pl.col("upload_type").cast(pl.Categorical).to_physical().cast(pl.Int64),
        music_type=pl.col("music_type").fill_null(-1).cast(pl.Int64),
    ).select(["video_id", "author_id", "music_id", "upload_ts"] + VIDEO_CAT + VIDEO_NUM)

    usr = pl.read_csv(RAW / "user_features_pure.csv", infer_schema_length=10000)
    usr = usr.with_columns([pl.col(c).cast(pl.Utf8).cast(pl.Categorical).to_physical().cast(pl.Int64)
                            for c in USER_CAT])

    ev = (ev.join(vid, on="video_id", how="left").join(usr.select(["user_id"] + USER_CAT + USER_NUM),
                                                       on="user_id", how="left")
            .with_columns(
                t=(pl.col("time_ms") - pl.col("time_ms").min()),
                hour=(pl.col("hourmin") // 100).cast(pl.Int64),
                weekday=pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d").dt.weekday().cast(pl.Int64),
                video_age_h=((pl.col("time_ms") - pl.col("upload_ts")) / 3.6e6).cast(pl.Float32),
                play_ratio=(pl.col("play_time_ms") / pl.col("duration_ms").clip(1, None)).clip(0, 1).cast(
                    pl.Float32),
                play_s=(pl.col("play_time_ms") / 1000).clip(0, 600).cast(pl.Float32),
                imp=pl.lit(1, dtype=pl.Int8),
            )
            .with_row_index("eid"))
    split = pl.lit("none")
    for name, (a, b) in SPLITS.items():
        split = pl.when((pl.col("date") >= a) & (pl.col("date") <= b)).then(pl.lit(name)).otherwise(split)
    ev = ev.with_columns(split=split)
    search = pl.lit("none")
    for name, (a, b) in SEARCH.items():
        search = pl.when((pl.col("date") >= a) & (pl.col("date") <= b)).then(pl.lit(name)).otherwise(search)
    ev = ev.with_columns(search=search)
    ev.write_parquet(OUT / "events.parquet")
    print(ev.group_by("split").agg(pl.len(), pl.col("is_click").mean(), pl.col("is_rand").mean()).sort("split"))
    print(ev.group_by("search").agg(pl.len()).sort("search"))
    return ev


def load():
    return pl.read_parquet(OUT / "events.parquet")


if __name__ == "__main__":
    build()
