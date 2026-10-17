"""Evaluation sets (global temporal split) and baseline recommenders."""
import numpy as np
import torch

from common import CTX, Data, policy_logits, rank_metrics, beyond_accuracy


class EvalSet:
    """Users active in [cutoff, end). Context = last CTX events before cutoff (any events are visible:
    the model is evaluated as if deployed at `cutoff`). Targets:
      next : first event of the period (users whose first event is OOV are dropped for this target)
      day  : set of distinct in-vocabulary tracks listened in the period."""

    def __init__(self, d: Data, cutoff: int, end: int, ctx: int = CTX, max_users=None, seed=0):
        users, ctxs, lens, nxt, day, hist = [], [], [], [], [], []
        hist_full = []
        for u, (s, e) in enumerate(zip(d.ustart, d.uend)):
            ts = d.ts[s:e]
            a = s + np.searchsorted(ts, cutoff, "left")
            b = s + np.searchsorted(ts, end, "left")
            if b <= a or a - s < 1:
                continue
            period = d.tok[a:b]
            dset = set(int(t) for t in period if t >= 2)
            if not dset:
                continue
            c = d.tok[max(s, a - ctx):a]
            users.append(u)
            ctxs.append(np.pad(c, (0, ctx - len(c))))
            lens.append(len(c))
            nxt.append(int(period[0]))
            day.append(dset)
            hist.append(set(int(t) for t in c if t >= 2))
            hist_full.append((max(s, a - ctx), a))
        self.users = np.array(users)
        self.ctx = np.stack(ctxs)
        self.lens = np.array(lens)
        self.next = np.array(nxt)
        self.day = day
        self.hist = hist
        self.ctx_range = hist_full
        self.cutoff = cutoff
        self.next_ok = self.next >= 2

    def __len__(self):
        return len(self.users)

    def metrics(self, topk, item_pop, item_len, n_items):
        m = {}
        ok = self.next_ok
        for k, v in rank_metrics(topk[ok], [{int(t)} for t in self.next[ok]]).items():
            m["next_" + k] = v
        for k, v in rank_metrics(topk, self.day).items():
            m["day_" + k] = v
        # "new" next items (not in the visible context) -- recall on exploration
        new = ok & np.array([int(t) not in h for t, h in zip(self.next, self.hist)])
        m["next_new_share"] = float(new.sum() / ok.sum())
        if new.any():
            m["next_new_recall@100"] = rank_metrics(topk[new], [{int(t)} for t in self.next[new]], ks=(100,))[
                "recall@100"]
        m.update(beyond_accuracy(topk, self.hist, item_pop, item_len, n_items))
        return m


@torch.no_grad()
def policy_scores(model, es: EvalSet, batch=512):
    """Log-probabilities of the next item for every eval user [U, V]."""
    model.eval()
    out = []
    for i in range(0, len(es), batch):
        tok = torch.from_numpy(es.ctx[i:i + batch])
        last = torch.from_numpy(es.lens[i:i + batch] - 1)
        lg = policy_logits(model, tok, last_pos=last)
        out.append(torch.log_softmax(lg, -1))
    return torch.cat(out)


def topk_from_scores(scores, k=100):
    return torch.topk(scores, k, dim=-1).indices.numpy()


def popularity_topk(d: Data, cutoff, n_tokens, k=100, window=86400):
    m = (d.ts >= cutoff - window) & (d.ts < cutoff) & (d.tok >= 2)
    cnt = np.bincount(d.tok[m], minlength=n_tokens)
    return np.argsort(-cnt)[:k]


def repeat_topk(d: Data, es: EvalSet, pop_top, k=100):
    """History heuristic: user's most frequent tracks in the visible context (recency tie-break), then popular."""
    out = np.zeros((len(es), k), dtype=np.int64)
    for i, c in enumerate(es.ctx):
        c = c[: es.lens[i]]
        score = {}
        for pos, t in enumerate(c):
            if t >= 2:
                score[int(t)] = score.get(int(t), 0) + 1 + pos * 1e-4
        ranked = [t for t, _ in sorted(score.items(), key=lambda x: -x[1])][:k]
        seen = set(ranked)
        for t in pop_top:
            if len(ranked) >= k:
                break
            if int(t) not in seen:
                ranked.append(int(t))
        out[i] = ranked[:k]
    return out
