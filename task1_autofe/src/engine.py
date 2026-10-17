"""Feature DSL + point-in-time computation engine.

A feature is a typed program
        AGG( SIGNAL  WHERE FILTER  GROUP BY KEY  OVER (WINDOW | DECAY τ) )  evaluated strictly before t
e.g. rate(is_click | user_id+author_id | std | 7d)  = smoothed CTR of this user on this author's videos
in standard-feed impressions over the last 7 days, computed from events with timestamp < t.

Leakage safety is a property of the compiler, not of whoever writes the spec: every program only sees
events with t_event < t_row.  The same spec compiles to (a) a fast numpy/numba implementation used in
experiments and (b) a DuckDB SQL query used to cross-check (see tests/test_engine.py).
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import numba
import numpy as np
import polars as pl

SIGNALS = ["imp", "is_click", "long_view", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
           "is_profile_enter", "play_ratio", "play_s"]
FILTERS = {"all": None, "std": ("is_rand", 0), "rand": ("is_rand", 1), "clicked": ("is_click", 1)}
AGGS = ["sum", "rate", "recency", "dsum", "drate"]
INF = float("inf")
DAY = 86400.0


def fmt_window(w: float) -> str:
    if w == INF:
        return "inf"
    for unit, s in (("d", DAY), ("h", 3600.0), ("m", 60.0)):
        if w >= s:
            return f"{w / s:.4g}{unit}"
    return f"{w:.4g}s"


@dataclass(frozen=True)
class Spec:
    agg: str                 # sum | rate | recency | dsum | drate
    signal: str              # column summed / averaged
    key: tuple               # GROUP BY columns (must exist in the event table)
    filt: str = "all"        # WHERE clause on past events
    window: float = INF      # seconds (sum/rate) or half-life in seconds (dsum/drate)
    alpha: float = 10.0      # prior strength for rates

    @property
    def name(self) -> str:
        w = "" if self.agg == "recency" else ("|hl=" if self.agg.startswith("d") else "|w=") + fmt_window(self.window)
        a = f"|a={self.alpha:g}" if self.agg in ("rate", "drate") else ""
        return f"{self.agg}({self.signal}|{'+'.join(self.key)}|{self.filt}{w}{a})"

    def valid(self) -> bool:
        if self.agg in ("rate", "drate") and self.signal == "imp":
            return False
        if self.agg == "recency" and self.window != INF:
            return False
        if self.agg in ("dsum", "drate") and self.window == INF:
            return False
        if self.filt == "clicked" and self.signal == "is_click":
            return False
        return True

    # ------------------------------------------------------------------ SQL (reference semantics)
    def _exprs(self, p: str = ""):
        """(filter condition, signal expression, count expression) with columns qualified by prefix p."""
        f = FILTERS[self.filt]
        cond = f"{p}{f[0]} = {f[1]}" if f else None
        x = f"CASE WHEN {cond} THEN {p}{self.signal} ELSE 0 END" if cond else f"{p}{self.signal}"
        n = f"CASE WHEN {cond} THEN 1 ELSE 0 END" if cond else "1"
        return cond, x, n

    def to_sql(self, table: str = "ev") -> str:
        """Reference semantics in DuckDB SQL. Rate priors are point-in-time: the global mean of the signal
        over filtered events strictly before t, (sum + 0.5) / (count + 1)."""
        keys = ", ".join(self.key)
        cond, x, n = self._exprs()
        g = "(ORDER BY t RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)"
        pri = (f"WITH p AS (SELECT *, (COALESCE(SUM({x}) OVER {g}, 0) + 0.5) / "
               f"(COALESCE(SUM({n}) OVER {g}, 0) + 1) AS prior FROM {table}) ")
        if self.agg in ("sum", "rate"):
            lo = "UNBOUNDED PRECEDING" if self.window == INF else f"{int(round(self.window * 1000))} PRECEDING"
            w = f"(PARTITION BY {keys} ORDER BY t RANGE BETWEEN {lo} AND 1 PRECEDING)"
            sx = f"COALESCE(SUM({x}) OVER {w}, 0)"
            if self.agg == "sum":
                return f"SELECT eid, {sx} AS v FROM {table}"
            sn = f"COALESCE(SUM({n}) OVER {w}, 0)"
            return pri + f"SELECT eid, ({sx} + {self.alpha!r} * prior) / ({sn} + {self.alpha!r}) AS v FROM p"
        if self.agg == "recency":
            pos = f"({cond}) AND {self.signal} > 0" if cond else f"{self.signal} > 0"
            w = f"(PARTITION BY {keys} ORDER BY t RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)"
            return f"SELECT eid, (t - MAX(CASE WHEN {pos} THEN t END) OVER {w}) / 1000.0 AS v FROM {table}"
        # exponential decay: explicit (truncated at 40 half-lives) self-join
        tau = self.window / math.log(2)
        on = " AND ".join(f"q.{k} = e.{k}" for k in self.key)
        _, xe, ne = self._exprs("e.")
        wgt = f"exp(-(q.t - e.t) / 1000.0 / {tau!r})"
        span = int(self.window * 40 * 1000)
        sx = f"COALESCE(SUM(({xe}) * {wgt}), 0)"
        body = (f"FROM p q LEFT JOIN {table} e ON {on} AND e.t < q.t AND e.t >= q.t - {span} "
                f"GROUP BY q.eid, q.prior")
        if self.agg == "dsum":
            return pri + f"SELECT q.eid AS eid, {sx} AS v {body}"
        sn = f"COALESCE(SUM(CASE WHEN e.eid IS NULL THEN 0 ELSE ({ne}) END * {wgt}), 0)"
        return pri + f"SELECT q.eid AS eid, ({sx} + {self.alpha!r} * q.prior) / ({sn} + {self.alpha!r}) AS v {body}"


@numba.njit(cache=True)
def _decay_scan(gid, ts, xs, ns, tau):
    """Decayed sums of x and n over strictly earlier events of the same group (inputs sorted by gid, t)."""
    n = len(gid)
    vx = np.empty(n)
    vn = np.empty(n)
    sx = 0.0
    sn = 0.0
    last_t = 0.0
    i = 0
    while i < n:
        g = gid[i]
        if i == 0 or g != gid[i - 1]:
            sx = 0.0
            sn = 0.0
            last_t = ts[i]
        j = i
        while j < n and gid[j] == g and ts[j] == ts[i]:   # tie block: same group, same timestamp
            j += 1
        dec = math.exp(-(ts[i] - last_t) / tau)
        sx *= dec
        sn *= dec
        last_t = ts[i]
        for k in range(i, j):
            vx[k] = sx
            vn[k] = sn
        for k in range(i, j):
            sx += xs[k]
            sn += ns[k]
        i = j
    return vx, vn


class Engine:
    """Computes Spec values for events of a time-sorted table (aligned with the table's row order)."""

    def __init__(self, ev: pl.DataFrame, max_groups: int = 24, prior_mode: str = "pit", prior_rows=None):
        """prior_mode: "pit" -- point-in-time global mean strictly before t (default, used everywhere);
        "const" -- one constant per (signal, filter) estimated on `prior_rows` (robustness check only)."""
        self.ev = ev
        self.prior_mode = prior_mode
        self.prior_rows = prior_rows
        self.t = ev["t"].to_numpy().astype(np.int64)            # ms since dataset start
        assert np.all(np.diff(self.t) >= 0), "event table must be sorted by time"
        self.M = int(self.t.max()) + 1
        self._before = np.searchsorted(self.t, self.t, "left")  # number of events strictly earlier than each row
        self._groups = OrderedDict()                             # LRU: key -> sorted index structures
        self.max_groups = max_groups
        self._cols = {}
        self._codes = {}
        self._masks = {}

    def col(self, name):
        if name not in self._cols:
            self._cols[name] = self.ev[name].fill_null(-1).to_numpy()
        return self._cols[name]

    def codes(self, name):
        """Dense integer codes of a key column and its cardinality."""
        if name not in self._codes:
            u, inv = np.unique(self.col(name), return_inverse=True)
            self._codes[name] = (inv.astype(np.int64), len(u))
        return self._codes[name]

    def filt_mask(self, filt):
        if filt not in self._masks:
            if filt == "all":
                m = np.ones(len(self.t), dtype=np.float64)
            elif filt == "std":
                m = (self.col("is_rand") == 0).astype(np.float64)
            elif filt == "rand":
                m = (self.col("is_rand") == 1).astype(np.float64)
            elif filt == "clicked":
                m = (self.col("is_click") == 1).astype(np.float64)
            else:
                raise ValueError(filt)
            self._masks[filt] = m
        return self._masks[filt]

    def prior(self, spec: Spec) -> np.ndarray:
        """Point-in-time prior for rates (table order): (sum of filtered signal strictly before t + 0.5) /
        (number of filtered events strictly before t + 1)."""
        m = self.filt_mask(spec.filt)
        x = self.col(spec.signal).astype(np.float64) * m
        if self.prior_mode == "const":
            r = self.prior_rows
            c = (x[r].sum() + 0.5) / (m[r].sum() + 1.0)
            return np.full(len(x), c)
        cx = np.concatenate([[0.0], np.cumsum(x)])
        cn = np.concatenate([[0.0], np.cumsum(m)])
        return (cx[self._before] + 0.5) / (cn[self._before] + 1.0)

    def group(self, key):
        """Sort events by (key, t). Returns order, inverse order, comp=(gid*M+t) sorted, group start, tie start."""
        key = tuple(key)
        if key in self._groups:
            self._groups.move_to_end(key)
            return self._groups[key]
        if len(self._groups) >= self.max_groups:
            self._groups.popitem(last=False)
        gid = np.zeros(len(self.t), dtype=np.int64)
        card = 1
        for k in key:
            c, n = self.codes(k)
            gid = gid * n + c
            card *= n
        if card * self.M >= 2 ** 62:                     # composite id too large -> dense re-coding
            _, gid = np.unique(gid, return_inverse=True)
            gid = gid.astype(np.int64)
        order = np.argsort(gid, kind="stable")           # table is time-sorted -> (gid, t) order
        comp = gid[order] * self.M + self.t[order]
        n = len(comp)
        idx = np.arange(n)
        new_g = np.empty(n, dtype=bool)
        new_g[0] = True
        new_g[1:] = (comp[1:] // self.M) != (comp[:-1] // self.M)
        gstart = np.maximum.accumulate(np.where(new_g, idx, 0)).astype(np.int32)
        new_t = np.empty(n, dtype=bool)
        new_t[0] = True
        new_t[1:] = comp[1:] != comp[:-1]
        hi = np.maximum.accumulate(np.where(new_t, idx, 0)).astype(np.int32)   # events strictly earlier
        inv = np.empty(n, dtype=np.int32)
        inv[order] = idx
        g = (order.astype(np.int32), inv, comp, gstart, hi)
        self._groups[key] = g
        return g

    def compute(self, spec: Spec, rows=None) -> np.ndarray:
        """Feature values for all events (rows=None) or for a boolean row mask / index array."""
        order, inv, comp, gstart, hi = self.group(spec.key)
        m = self.filt_mask(spec.filt)[order]
        xs = self.col(spec.signal).astype(np.float64)[order] * m
        if rows is None:
            q = np.arange(len(order))                     # every sorted position (all events)
        else:                                             # sorted positions of the requested rows
            q = inv[np.flatnonzero(rows)] if rows.dtype == bool else inv[rows]
        hq = hi[q]
        if spec.agg in ("sum", "rate"):
            if spec.window == INF:
                lq = gstart[q]
            else:
                cq = comp[q]
                gq = cq - (cq % self.M)                    # gid*M of the query
                tq = cq % self.M
                lq = np.searchsorted(comp, gq + np.maximum(tq - int(round(spec.window * 1000)), 0), "left")
            cx = np.concatenate([[0.0], np.cumsum(xs)])
            sx = cx[hq] - cx[lq]
            if spec.agg == "sum":
                v = sx
            else:
                cn = np.concatenate([[0.0], np.cumsum(m)])
                v = (sx + spec.alpha * self.prior(spec)[order[q]]) / (cn[hq] - cn[lq] + spec.alpha)
        elif spec.agg == "recency":
            lastpos = np.maximum.accumulate(np.where(xs > 0, np.arange(len(xs)), -1))
            r = hq.astype(np.int64) - 1
            lp = np.where(r >= 0, lastpos[np.maximum(r, 0)], -1)
            ok = (r >= gstart[q]) & (lp >= gstart[q])
            ts = comp % self.M
            v = np.where(ok, (ts[q] - ts[np.maximum(lp, 0)]) / 1000.0, np.nan)
        else:  # dsum / drate
            tau = spec.window / math.log(2)
            ts = (comp % self.M) / 1000.0
            gs = comp // self.M
            vx, vn = _decay_scan(gs, ts, xs, m, tau)
            vx, vn = vx[q], vn[q]
            v = vx if spec.agg == "dsum" else (vx + spec.alpha * self.prior(spec)[order[q]]) / (vn + spec.alpha)
        if rows is None:
            out = np.empty(len(order), dtype=np.float32)
            out[order] = v
            return out
        return v.astype(np.float32)
