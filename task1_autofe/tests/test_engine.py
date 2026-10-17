"""Cross-check: numpy/numba engine == DuckDB SQL semantics, and no feature uses the current/future events."""
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import load  # noqa: E402
from engine import INF, Engine, Spec  # noqa: E402

SPECS = [
    Spec("sum", "is_click", ("user_id",), "all", 7 * 86400),
    Spec("sum", "imp", ("user_id", "author_id"), "std", 3600),
    Spec("rate", "is_click", ("video_id",), "all", INF),
    Spec("rate", "long_view", ("user_id", "tag"), "rand", 2 * 86400, 5.0),
    Spec("rate", "play_ratio", ("author_id",), "clicked", 86400),
    Spec("recency", "is_click", ("user_id",), "all"),
    Spec("recency", "imp", ("user_id", "video_id"), "all"),
    Spec("dsum", "is_like", ("user_id",), "all", 6 * 3600),
    Spec("drate", "is_click", ("user_id", "music_type"), "std", 86400),
]


def main():
    ev = load()
    users = np.random.default_rng(0).choice(ev["user_id"].unique().to_numpy(), 300, replace=False)
    sub = ev.filter(pl.col("user_id").is_in(users)).drop("eid").with_row_index("eid")
    eng = Engine(sub)
    con = duckdb.connect()
    con.register("ev", sub.select(["eid", "t", "is_rand", "user_id", "video_id", "author_id", "tag",
                                   "music_type", "imp", "is_click", "long_view", "is_like", "play_ratio"]).to_arrow())
    worst = 0.0
    for s in SPECS:
        a = eng.compute(s).astype(np.float64)
        q = con.execute(s.to_sql() + " ORDER BY eid").pl()
        b = q["v"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        err = np.where(both_nan, 0.0, np.abs(a - b) / (1 + np.abs(b)))
        worst = max(worst, float(np.nanmax(err)))
        assert not np.isnan(err).any(), s.name
        assert err.max() < 1e-4, (s.name, err.max())
        rows = np.random.default_rng(1).random(len(a)) < 0.2
        part = eng.compute(s, rows=rows).astype(np.float64)
        assert np.allclose(part, a[rows], equal_nan=True, rtol=1e-6), f"row-subset mismatch {s.name}"
        print(f"OK  {s.name:55s} max rel err {err.max():.2e}  nan share {np.isnan(a).mean():.3f}")

    # leakage probe: perturbing the labels of events at t >= t0 must not change any feature value at t <= t0
    t0 = int(np.median(sub["t"].to_numpy()))
    flip = sub.with_columns(is_click=pl.when(pl.col("t") >= t0).then(1 - pl.col("is_click")).otherwise(
        pl.col("is_click")))
    eng2 = Engine(flip)
    past = (sub["t"] <= t0).to_numpy()
    for s in SPECS:
        a, b = eng.compute(s), eng2.compute(s)
        same = np.allclose(a[past], b[past], equal_nan=True)
        assert same, f"leak in {s.name}"
    print("leakage probe passed: features at t <= t0 are unaffected by labels at t >= t0")
    print("worst relative error", worst)


if __name__ == "__main__":
    main()
