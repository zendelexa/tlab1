"""Reward definitions on top of reward-model heads, and full-catalog reward-model scoring."""
import numpy as np
import torch

from common import CTX, RewardModel, loglen, ratio_bucket

REWARDS = ["ratio", "time", "like", "cont", "debiased"]


def combine(kind, p_ratio, p_like, p_cont, length, len_bias=None):
    """Scalar reward from RM probabilities. Tensors broadcast; `length` in seconds."""
    if kind == "ratio":            # expected share of the track listened (short-term engagement)
        return p_ratio
    if kind == "time":             # expected listened seconds, in units of ~3.3 minutes
        return p_ratio * length / 200.0
    if kind == "like":             # paper [2] form: like + 0.1 * ratio
        return p_like + 0.1 * p_ratio
    if kind == "cont":             # probability the listening session continues (long-term proxy)
        return p_cont
    if kind == "debiased":         # ratio above what is expected for a track of this length
        return p_ratio - len_bias
    if kind == "mix":              # engagement + retention
        return 0.5 * p_ratio + 0.5 * p_cont
    raise ValueError(kind)


def length_bias(d, cutoff, n_bins=20):
    """Mean played ratio as a function of log track length (train period) -> per-token expected ratio."""
    m = d.ts < cutoff
    ll = np.log(np.maximum(d.length[m], 1))
    edges = np.quantile(ll, np.linspace(0, 1, n_bins + 1)[1:-1])
    b = np.searchsorted(edges, ll)
    means = np.bincount(b, weights=d.ratio[m], minlength=n_bins) / np.maximum(np.bincount(b, minlength=n_bins), 1)
    return edges, means


def token_length_bias(edges, means, item_len):
    return means[np.searchsorted(edges, np.log(np.maximum(item_len, 1)))].astype(np.float32)


class RMScorer:
    """Wraps a trained RewardModel for (a) scoring sampled actions and (b) scoring the full catalog."""

    def __init__(self, rm: RewardModel, item_len: np.ndarray):
        self.rm = rm.eval()
        for p in self.rm.parameters():
            p.requires_grad_(False)
        self.item_len = torch.from_numpy(item_len.astype(np.float32))
        self.loglen = torch.from_numpy(loglen(item_len))

    @torch.no_grad()
    def states(self, tok, rb, lk, org, last):
        h = self.rm.state(tok, rb, lk, org)
        return h[torch.arange(len(tok)), last]

    @torch.no_grad()
    def score(self, h, actions):
        """h [B,d], actions [B,G] -> probabilities [B,G,3] (ratio, like, cont)."""
        B, G = actions.shape
        hh = h[:, None, :].expand(B, G, h.shape[-1])
        lg = self.rm.head(hh, actions, self.loglen[actions], torch.zeros(B, G))
        return torch.sigmoid(lg)

    @torch.no_grad()
    def score_all(self, h, chunk=2048):
        """Full-catalog scores [B, V, 3] computed with the first MLP layer factorised over items."""
        rm = self.rm
        V = rm.cand_item.num_embeddings
        allt = torch.arange(V)
        c = rm.cand_item(allt) + rm.cand_feat(torch.stack([self.loglen, torch.zeros(V)], -1))   # [V,d]
        W1 = rm.mlp[0].weight          # [H, 3d]
        b1 = rm.mlp[0].bias
        d = h.shape[-1]
        Wh, Wc, Whc = W1[:, :d], W1[:, d:2 * d], W1[:, 2 * d:]
        cc = c @ Wc.T + b1             # [V,H]
        out = torch.empty(len(h), V, 3)
        for i in range(0, len(h), 8):
            hb = h[i:i + 8]                                             # [b,d]
            z = (hb @ Wh.T)[:, None, :] + cc[None] + (hb[:, None, :] * c[None]) @ Whc.T
            z = rm.mlp[1](z)
            z = rm.mlp[4](rm.mlp[3](rm.mlp[2](z)))
            out[i:i + 8] = torch.sigmoid(z)
        return out


def rm_context(d, idx_rows):
    """RM inputs for contexts given as lists of event indices (right-padded to CTX)."""
    B = len(idx_rows)
    tok = np.zeros((B, CTX), dtype=np.int64)
    rb = np.zeros((B, CTX), dtype=np.int64)
    lk = np.zeros((B, CTX), dtype=np.int64)
    org = np.zeros((B, CTX), dtype=np.int64)
    last = np.zeros(B, dtype=np.int64)
    for b, ids in enumerate(idx_rows):
        n = len(ids)
        tok[b, :n] = d.tok[ids]
        rb[b, :n] = ratio_bucket(d.ratio[ids])
        lk[b, :n] = d.liked[ids].astype(np.int64)
        org[b, :n] = d.organic[ids]
        last[b] = n - 1
    return [torch.from_numpy(a) for a in (tok, rb, lk, org)], torch.from_numpy(last)
