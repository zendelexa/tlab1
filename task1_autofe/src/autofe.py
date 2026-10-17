"""Automatic feature search over the relational-aggregate DSL.

Loop (all decisions use only the train period; val/test are never touched):
  1. base model  : LightGBM on `fit` days with current features  -> honest raw scores on screen days
  2. proposals   : random programs from the grammar  +  mutations of the best programs so far
                   (+ optionally programs proposed by an external agent, e.g. an LLM)
  3. screening   : marginal value of a candidate *given the current model*:
                   a tiny booster on [candidate] with init_score = base scores, fit on screen_a,
                   gain = logloss reduction on screen_b
  4. selection   : accept the best candidates (gain > threshold, |spearman| < 0.95 to accepted ones)
  5. repeat
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from engine import INF, SIGNALS, Spec
from features import FeatureStore
from model import BASE_PARAMS, CATEGORICAL, STATIC

USER_KEYS = ["user_id", "user_active_degree", "is_video_author", "follow_user_num_range", "fans_user_num_range",
             "register_days_range", "is_lowactive_period"]
ITEM_KEYS = ["video_id", "author_id", "music_id", "tag", "video_type", "upload_type", "music_type", "dur_bucket"]
CTX_KEYS = ["tab", "is_rand", "hour"]
SIG_W = {"imp": 3, "is_click": 4, "long_view": 3, "play_ratio": 3, "play_s": 2, "is_like": 2, "is_follow": 0.5,
         "is_comment": 0.5, "is_forward": 0.5, "is_hate": 0.7, "is_profile_enter": 0.7}
AGG_W = {"sum": 0.25, "rate": 0.3, "recency": 0.1, "dsum": 0.15, "drate": 0.2}
FILT_W = {"all": 0.55, "std": 0.15, "rand": 0.15, "clicked": 0.15}
WMIN, WMAX = 600.0, 30 * 86400.0


def _pick(rng, w: dict):
    k = list(w)
    p = np.array([w[x] for x in k], dtype=float)
    return k[rng.choice(len(k), p=p / p.sum())]


def random_key(rng):
    r = rng.random()
    if r < 0.3:
        return (rng.choice(USER_KEYS + ITEM_KEYS),)
    if r < 0.75:
        return (rng.choice(USER_KEYS), rng.choice(ITEM_KEYS))
    if r < 0.9:
        return (rng.choice(USER_KEYS + ITEM_KEYS), rng.choice(CTX_KEYS))
    return (rng.choice(USER_KEYS), rng.choice(ITEM_KEYS), rng.choice(CTX_KEYS))


def random_window(rng, agg, grid=None):
    if grid is not None:
        return float(rng.choice(grid))
    if agg in ("sum", "rate") and rng.random() < 0.15:
        return INF
    hi = WMAX if agg in ("sum", "rate") else 10 * 86400.0
    return float(math.exp(rng.uniform(math.log(WMIN), math.log(hi))))


def random_spec(rng, grid=None) -> Spec:
    while True:
        agg = _pick(rng, AGG_W)
        if grid is not None and agg in ("dsum", "drate"):
            continue
        s = Spec(agg=agg, signal=_pick(rng, SIG_W), key=tuple(str(k) for k in random_key(rng)),
                 filt=_pick(rng, FILT_W), window=INF if agg == "recency" else random_window(rng, agg, grid),
                 alpha=float(rng.choice([1.0, 10.0, 100.0], p=[0.2, 0.6, 0.2])))
        if s.valid():
            return s


def mutate(rng, s: Spec, grid=None) -> Spec:
    for _ in range(20):
        op = rng.integers(7)
        d = asdict(s)
        if op == 0 and s.agg != "recency":                                 # window
            if grid is not None:
                d["window"] = float(rng.choice(grid))
            elif s.window == INF:
                d["window"] = float(rng.choice([7, 14, 30]) * 86400.0)
            else:
                d["window"] = float(np.clip(s.window * math.exp(rng.normal(0, 0.7)), WMIN, WMAX))
        elif op == 1:
            d["signal"] = _pick(rng, SIG_W)
        elif op == 2:
            d["filt"] = _pick(rng, FILT_W)
        elif op == 3:                                                       # aggregation
            d["agg"] = _pick(rng, AGG_W)
            if d["agg"] == "recency":
                d["window"] = INF
            elif d["agg"] in ("dsum", "drate"):
                if grid is not None:
                    continue
                d["window"] = float(np.clip(s.window / 3 if s.window != INF else 3 * 86400.0, WMIN, 10 * 86400.0))
            elif s.window == INF and s.agg != "recency":
                d["window"] = INF
            elif s.agg == "recency":
                d["window"] = random_window(rng, d["agg"], grid)
        elif op == 4:                                                       # replace a key component
            key = list(s.key)
            i = rng.integers(len(key))
            pool = USER_KEYS if key[i] in USER_KEYS else ITEM_KEYS if key[i] in ITEM_KEYS else CTX_KEYS
            key[i] = str(rng.choice(pool))
            d["key"] = tuple(dict.fromkeys(key))
        elif op == 5:                                                       # add / drop a key component
            key = list(s.key)
            if len(key) > 1 and rng.random() < 0.5:
                key.pop(rng.integers(len(key)))
            elif len(key) < 3:
                kinds = [p for p in (USER_KEYS, ITEM_KEYS, CTX_KEYS) if not any(k in p for k in key)]
                if not kinds:
                    continue
                key.append(str(rng.choice(kinds[rng.integers(len(kinds))])))
            d["key"] = tuple(key)
        elif s.agg in ("rate", "drate"):                                   # alpha only matters for rates
            d["alpha"] = float(np.clip(s.alpha * rng.choice([0.1, 10.0]), 1.0, 100.0))
        else:
            continue
        n = Spec(**d)
        if n.valid() and n.name != s.name:
            return n
    return random_spec(rng, grid)


class Screener:
    """Measures the marginal value of a candidate feature on top of the current model."""

    def __init__(self, fs: FeatureStore, target="is_click", objective="logloss"):
        ev = fs.ev
        self.fs = fs
        self.objective = objective
        self.fit_rows = (ev["search"] == "fit").to_numpy()
        self.a_rows = (ev["search"] == "screen_a").to_numpy()
        self.b_rows = (ev["search"] == "screen_b").to_numpy()
        self.ab_rows = self.a_rows | self.b_rows
        self.a_in_ab = self.a_rows[self.ab_rows]       # positions of screen_a inside the screen_a|b slice
        self.y = ev[target].to_numpy().astype(np.float64)
        self.params = dict(BASE_PARAMS, learning_rate=0.1, num_threads=4)
        self.tiny = dict(objective="binary", learning_rate=0.1, num_leaves=7, min_data_in_leaf=200,
                         verbose=-1, num_threads=4, seed=0)
        if objective == "listwise":
            self._init_listwise(ev)

    def refit(self, specs):
        X = {r: self.fs.matrix(specs, m).to_pandas() for r, m in
             (("fit", self.fit_rows), ("a", self.a_rows), ("b", self.b_rows))}
        cat = [c for c in CATEGORICAL if c in X["fit"].columns]
        dtr = lgb.Dataset(X["fit"], self.y[self.fit_rows], categorical_feature=cat)
        dva = lgb.Dataset(X["a"], self.y[self.a_rows], reference=dtr, categorical_feature=cat)
        bst = lgb.train(self.params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        raw_a = bst.predict(X["a"], raw_score=True)
        raw_b = bst.predict(X["b"], raw_score=True)
        # Platt recalibration on screen_a (separately for feed / random traffic): removes the gain any feature
        # would get just by correcting day-to-day CTR drift, so screening measures information, not calibration
        ra, rb_ = X["a"]["is_rand"].to_numpy(), X["b"]["is_rand"].to_numpy()
        za = np.column_stack([raw_a, ra, raw_a * ra])
        zb = np.column_stack([raw_b, rb_, raw_b * rb_])
        cal = LogisticRegression(C=1e4, max_iter=1000).fit(za, self.y[self.a_rows])
        self.base_a = cal.decision_function(za)
        self.base_b = cal.decision_function(zb)
        self.base_ll = self._ll(self.y[self.b_rows], self.base_b)
        return self.base_ll

    def _ll(self, y, raw):
        if self.objective == "listwise":
            return self._listwise(raw)
        p = 1 / (1 + np.exp(-raw))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    def _listwise(self, raw):
        """Mean over (user, day, traffic) groups of  -mean_{clicked i} log softmax_group(raw)_i.
        Invariant to any per-user shift of the scores -> measures within-user ranking only."""
        r = raw[self.lw_order]
        mx = np.maximum.reduceat(r, self.lw_starts)
        lse = np.log(np.add.reduceat(np.exp(r - np.repeat(mx, self.lw_sizes)), self.lw_starts)) + mx
        pos = np.add.reduceat(r * self.lw_y, self.lw_starts) / self.lw_npos
        return float((lse - pos)[self.lw_valid].mean())

    def _init_listwise(self, ev):
        b = ev.filter(pl.Series(self.b_rows))
        g = (b["user_id"].to_numpy().astype(np.int64) * 100000 + b["date"].to_numpy() % 100000) * 2 + \
            b["is_rand"].to_numpy()
        self.lw_order = np.argsort(g, kind="stable")
        gs = g[self.lw_order]
        self.lw_starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
        self.lw_sizes = np.diff(np.r_[self.lw_starts, len(gs)])
        self.lw_y = self.y[self.b_rows][self.lw_order]
        self.lw_npos = np.maximum(np.add.reduceat(self.lw_y, self.lw_starts), 1)
        npos = np.add.reduceat(self.lw_y, self.lw_starts)
        self.lw_valid = (npos > 0) & (npos < self.lw_sizes)

    def gain(self, xa, xb, rounds=40, update=False):
        """Logloss reduction on screen_b from a tiny booster on the candidate fit (on screen_a) on top of the
        current scores. With update=True the correction is folded into the base scores (forward stagewise),
        so later candidates are judged conditionally on this one."""
        if np.nanstd(xa) == 0 or np.isnan(xa).mean() > 0.995:
            return 0.0
        d = lgb.Dataset(xa.reshape(-1, 1), self.y[self.a_rows], init_score=self.base_a)
        bst = lgb.train(self.tiny, d, rounds)
        raw = self.base_b + bst.predict(xb.reshape(-1, 1), raw_score=True)
        g = self.base_ll - self._ll(self.y[self.b_rows], raw)
        if update:
            self.base_a = self.base_a + bst.predict(xa.reshape(-1, 1), raw_score=True)
            self.base_b = raw
            self.base_ll = self._ll(self.y[self.b_rows], raw)
        return g


def search(fs: FeatureStore, rounds=10, n_random=100, n_mut=60, per_round=8, min_gain=5e-5, seed=0,
           grid=None, extra_proposals=None, log_path=None, init=(), eval_fn=None, pool_size=16,
           objective="logloss", seed_parents=()):
    """Returns accepted specs and a log. `extra_proposals(round, history) -> list[Spec]` lets an external
    agent (e.g. an LLM) add candidates; they are screened exactly like the others."""
    rng = np.random.default_rng(seed)
    scr = Screener(fs, objective=objective)
    accepted = list(init)
    history = {}            # name -> dict(spec, gain, round, source)
    for s in seed_parents:  # e.g. features proposed by an LLM: used as mutation parents from round 0
        history[s.name] = dict(spec=s, gain=1.0, round=-1, source="init")
    log = []
    t_start = time.time()
    ranks_b = {}
    for s in accepted:
        ranks_b[s.name] = rankdata(np.nan_to_num(fs.engine.compute(s, rows=scr.b_rows), nan=-1e9))
    for r in range(rounds):
        t0 = time.time()
        base_ll = scr.refit(accepted)
        props = [(random_spec(rng, grid), "random") for _ in range(n_random)]
        top = sorted(history.values(), key=lambda h: -h["gain"])[:20]
        for _ in range(n_mut if top else 0):
            parent = top[min(int(rng.exponential(5)), len(top) - 1)]["spec"]
            props.append((mutate(rng, parent, grid), "mutation"))
        if extra_proposals is not None:
            props += [(s, "agent") for s in extra_proposals(r, history, accepted)]
        seen_round = set()
        scored = []
        for s, src in props:
            if s.name in seen_round or any(s.name == x.name for x in accepted):
                continue
            seen_round.add(s.name)
            v = fs.engine.compute(s, rows=scr.ab_rows)
            xa, xb = v[scr.a_in_ab], v[~scr.a_in_ab]
            g = scr.gain(xa, xb)
            history[s.name] = dict(spec=s, gain=g, round=r, source=src)
            scored.append((g, s, src, xa, xb))
        scored.sort(key=lambda x: -x[0])
        added = []
        # best-first forward-stagewise acceptance: among the best candidates by marginal gain, repeatedly take
        # the one with the largest gain *conditional* on everything accepted so far in this round
        pool = [c for c in scored[:pool_size] if c[0] >= min_gain]
        cond = {id(c): c[0] for c in pool}
        while pool and len(added) < per_round:
            if added:
                for c in pool:
                    cond[id(c)] = scr.gain(c[3], c[4])
            pool.sort(key=lambda c: -cond[id(c)])
            g, s, src, xa, xb = best = pool.pop(0)
            g_cond = cond[id(best)]
            if g_cond < min_gain:
                break
            rb = rankdata(np.nan_to_num(xb, nan=-1e9))
            if any(abs(np.corrcoef(rb, ranks_b[a.name])[0, 1]) > 0.95 for a in accepted if a.name in ranks_b):
                continue
            scr.gain(xa, xb, update=True)
            ranks_b[s.name] = rb
            accepted.append(s)
            fs.values(s)            # keep full column for the final model
            added.append(dict(name=s.name, gain=g, cond_gain=g_cond, source=src))
        entry = dict(round=r, base_screen_ll=base_ll, n_candidates=len(scored), added=added,
                     n_accepted=len(accepted), time=time.time() - t0, total_time=time.time() - t_start,
                     src_stats={src: dict(n=sum(1 for x in scored if x[2] == src),
                                          hit=sum(1 for x in scored if x[2] == src and x[0] >= min_gain),
                                          best=max([x[0] for x in scored if x[2] == src], default=0.0))
                                for src in ("random", "mutation", "agent")})
        if eval_fn is not None:
            entry["val"] = eval_fn(accepted)
        log.append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k != "added"}), flush=True)
        for a in added:
            print(f"   + {a['gain']:.5f} (cond {a['cond_gain']:.5f}) {a['source']:8s} {a['name']}", flush=True)
        if log_path:
            json.dump(dict(log=log, accepted=[s.name for s in accepted],
                           accepted_specs=[dict(agg=s.agg, signal=s.signal, key=list(s.key), filt=s.filt,
                                                window=s.window, alpha=s.alpha) for s in accepted],
                           history=[dict(name=k, gain=v["gain"], round=v["round"], source=v["source"])
                                    for k, v in history.items()]), open(log_path, "w"), indent=1)
        for k in [k for k, h in history.items() if h["source"] == "init"]:
            del history[k]          # seed parents only bootstrap round 0; afterwards real gains decide
        if not added:
            break
    return accepted, history, log
