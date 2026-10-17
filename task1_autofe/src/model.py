"""LightGBM training and ranking/classification metrics."""
import numpy as np
import lightgbm as lgb
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import log_loss, roc_auc_score

from data import USER_CAT, USER_NUM, VIDEO_CAT, VIDEO_NUM

CONTEXT = ["tab", "is_rand", "hour", "weekday", "video_age_h"]
STATIC = USER_CAT + USER_NUM + VIDEO_CAT + VIDEO_NUM + CONTEXT
CATEGORICAL = USER_CAT + ["video_type", "upload_type", "music_type", "tag", "tab"]
IDS = ["user_id", "video_id", "author_id", "music_id"]

BASE_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   max_cat_to_onehot=8, cat_smooth=20, verbose=-1, num_threads=4, seed=0)


def fit(X_tr, y_tr, X_va, y_va, params=None, cat=(), rounds=3000, early=50, init_tr=None, init_va=None):
    p = dict(BASE_PARAMS, **(params or {}))
    cat = [c for c in cat if c in X_tr.columns]
    dtr = lgb.Dataset(X_tr.to_pandas(), y_tr, categorical_feature=cat or "auto", init_score=init_tr,
                      free_raw_data=True)
    dva = lgb.Dataset(X_va.to_pandas(), y_va, reference=dtr, categorical_feature=cat or "auto", init_score=init_va)
    return lgb.train(p, dtr, rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(early, verbose=False)] if early else [])


def group_metrics(y, p, groups, k=10):
    """GAUC (impression-weighted AUC within groups) and NDCG@k over groups with >=1 positive and >=1 negative."""
    order = np.lexsort((-p, groups))
    g, yy, pp = groups[order], y[order], p[order]
    bounds = np.flatnonzero(np.diff(g)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(g)]])
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    aucs, ws, ndcgs = [], [], []
    for s, e in zip(starts, ends):
        ys = yy[s:e]
        npos = ys.sum()
        if npos == 0 or npos == len(ys):
            continue
        # Mann-Whitney AUC with average ranks for tied scores
        ranks = rankdata(-pp[s:e])
        ranks = len(ys) + 1 - ranks
        auc = (ranks[ys == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(ys) - npos))
        aucs.append(auc)
        ws.append(len(ys))
        kk = min(k, len(ys))
        dcg = (ys[:kk] * disc[:kk]).sum()
        idcg = disc[: min(int(npos), kk)].sum()
        ndcgs.append(dcg / idcg)
    return float(np.average(aucs, weights=ws)), float(np.mean(ndcgs)), len(aucs)


def evaluate(y, p, df: pl.DataFrame):
    """Metrics on all rows and separately on standard-feed and random-exposure impressions."""
    out = {}
    grp = (df["user_id"].to_numpy().astype(np.int64) * 100000 + (df["date"].to_numpy() % 100000)) * 2 + \
        df["is_rand"].to_numpy()
    for name, m in [("all", np.ones(len(y), bool)), ("std", df["is_rand"].to_numpy() == 0),
                    ("rand", df["is_rand"].to_numpy() == 1)]:
        if m.sum() == 0:
            continue
        gauc, ndcg, ng = group_metrics(y[m], p[m], grp[m])
        out[name] = dict(logloss=float(log_loss(y[m], np.clip(p[m], 1e-7, 1 - 1e-7))),
                         auc=float(roc_auc_score(y[m], p[m])), gauc=gauc, ndcg10=ndcg, n=int(m.sum()), groups=ng)
    return out
