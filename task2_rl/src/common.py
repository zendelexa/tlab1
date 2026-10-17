"""Shared data access, models and evaluation for the Yambda RL experiments."""
import json
import os
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CKPT = ROOT / "data" / "ckpt"
RES = ROOT / "results"
CTX = 128            # context length (events)
N_RATIO_BUCKETS = 6  # feedback buckets for the reward model input
SESSION_GAP = 1800   # 30 min without listening = session end

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))


def ratio_bucket(r):
    """played ratio in [0,1] -> bucket id 1..N_RATIO_BUCKETS-1 (0 = padding)."""
    edges = np.array([0.1, 0.3, 0.6, 0.9])
    return 1 + np.searchsorted(edges, r, side="right")


# ----------------------------------------------------------------------------- data
@dataclass
class Data:
    meta: dict
    uid: np.ndarray
    ts: np.ndarray
    tok: np.ndarray
    organic: np.ndarray
    ratio: np.ndarray
    length: np.ndarray
    liked: np.ndarray
    gap_next: np.ndarray
    ustart: np.ndarray   # offsets of each user's block
    uend: np.ndarray
    users: np.ndarray


def load_data() -> Data:
    meta = json.loads((PROC / "meta.json").read_text())
    ev = pl.read_parquet(PROC / "events.parquet")
    uid = ev["uid"].to_numpy()
    users, ustart = np.unique(uid, return_index=True)
    uend = np.append(ustart[1:], len(uid))
    gap = ev["gap_next"].fill_null(-1).to_numpy()
    return Data(meta, uid, ev["ts"].to_numpy().astype(np.int64), ev["tok"].to_numpy().astype(np.int64),
                ev["organic"].to_numpy().astype(np.int64), ev["ratio"].to_numpy().astype(np.float32),
                ev["length"].to_numpy().astype(np.float32), ev["liked"].to_numpy().astype(np.float32),
                gap, ustart, uend, users)


def train_end_index(d: Data, cutoff: int) -> np.ndarray:
    """For each user: index (exclusive) of the first event with ts >= cutoff."""
    out = np.empty(len(d.users), dtype=np.int64)
    for k, (s, e) in enumerate(zip(d.ustart, d.uend)):
        out[k] = s + np.searchsorted(d.ts[s:e], cutoff, side="left")
    return out


def item_stats(d: Data, cutoff: int):
    """Per-token statistics from events before `cutoff`: popularity, median track length, mean played ratio."""
    n = d.meta["n_tokens"]
    m = (d.ts < cutoff) & (d.tok >= 2)
    tok = d.tok[m]
    pop = np.bincount(tok, minlength=n).astype(np.float64)
    ratio = np.bincount(tok, weights=d.ratio[m], minlength=n) / np.maximum(pop, 1)
    order = np.lexsort((d.length[m], tok))
    t_s, l_s = tok[order], d.length[m][order]
    starts = np.searchsorted(t_s, np.arange(n), "left")
    ends = np.searchsorted(t_s, np.arange(n), "right")
    length = np.where(ends > starts, l_s[np.clip((starts + ends) // 2, 0, len(l_s) - 1)], 200.0)
    return pop, length.astype(np.float32), ratio.astype(np.float32)


def sample_windows(d: Data, rng, batch, cut_end, ctx=CTX, min_hist=1):
    """Sample training windows of `ctx+1` consecutive events strictly before the train cutoff.
    Returns indices array [batch, ctx+1] into the event arrays (-1 = padding at the right)."""
    lens = cut_end - d.ustart
    ok = np.where(lens > min_hist)[0]
    w = lens[ok].astype(np.float64)
    us = rng.choice(ok, size=batch, p=w / w.sum())
    idx = np.full((batch, ctx + 1), -1, dtype=np.int64)
    for b, u in enumerate(us):
        s, e = d.ustart[u], cut_end[u]
        end = rng.integers(s + min_hist + 1, e + 1)       # exclusive end
        beg = max(s, end - (ctx + 1))
        idx[b, : end - beg] = np.arange(beg, end)
    return idx


# ----------------------------------------------------------------------------- policy (GPT-2)
def make_gpt(n_tokens, n_embd=64, n_layer=2, n_head=2, ctx=CTX, dropout=0.1):
    cfg = GPT2Config(vocab_size=n_tokens, n_positions=ctx + 8, n_embd=n_embd, n_layer=n_layer, n_head=n_head,
                     resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout, bos_token_id=0, eos_token_id=0)
    return GPT2LMHeadModel(cfg)


class RepeatHead(nn.Module):
    """Repeat-aware output bias (in the spirit of RepeatNet's repeat mode). For every track already present in
    the context it adds  w(h) . [present, last-item, sum_occ exp(-age/4), sum_occ exp(-age/32),
    sum_occ exp(-age/128)]  to its logit, where age = number of events since the occurrence and w(h) is a
    linear function of the GPT state. Zero-initialised: starts exactly at the plain GPT-2."""
    TAUS = (4.0, 32.0, 128.0)

    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 2 + len(self.TAUS))
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, h, tok, pos, V):
        B, T = tok.shape
        ar = torch.arange(T)[None]
        valid = ((ar <= pos[:, None]) & (tok >= 2)).float()
        age = (pos[:, None] - ar).clamp(min=0).float()
        w = self.lin(h)                                                     # [B, 5]
        occ = w[:, 1:2] * (age == 0).float()
        for k, tau in enumerate(self.TAUS):
            occ = occ + w[:, 2 + k:3 + k] * torch.exp(-age / tau)
        bias = torch.zeros(B, V).scatter_add(1, tok, occ * valid)
        pres = torch.zeros(B, V).scatter_reduce(1, tok, valid, reduce="amax")
        return bias + w[:, 0:1] * pres


class Policy(nn.Module):
    """GPT-2 next-item model (+ optional repeat head). One API for evaluation, RL and analysis."""

    def __init__(self, gpt: GPT2LMHeadModel, repeat: RepeatHead = None):
        super().__init__()
        self.gpt = gpt
        self.repeat = repeat

    def hidden(self, tok):
        return self.gpt.transformer(input_ids=tok).last_hidden_state

    def logits_at(self, hs, tok, pos):
        """hs [B,T,d] hidden states of tok; returns masked logits [B,V] for predicting the item after `pos`."""
        h = hs[torch.arange(len(tok)), pos]
        lg = self.gpt.lm_head(h)
        if self.repeat is not None:
            lg = lg + self.repeat(h, tok, pos, lg.shape[-1])
        mask = torch.zeros(lg.shape[-1], dtype=torch.bool)
        mask[:2] = True
        return lg.masked_fill(mask, -1e9)


def as_policy(model):
    return model if isinstance(model, Policy) else Policy(model)


def load_policy(name, n_tokens):
    gpt = make_gpt(n_tokens)
    gpt.load_state_dict(torch.load(CKPT / f"{name}.pt"))
    rep_path = CKPT / f"{name}_repeat.pt"
    rep = None
    if rep_path.exists():
        rep = RepeatHead(gpt.config.n_embd)
        rep.load_state_dict(torch.load(rep_path))
    return Policy(gpt, rep)


def policy_logits(model, tokens, attn=None, last_pos=None):
    """tokens [B,T] (right padded with 0). Returns logits at `last_pos` [B,V] with PAD/UNK masked."""
    p = as_policy(model)
    return p.logits_at(p.hidden(tokens), tokens, last_pos)


# ----------------------------------------------------------------------------- reward model
class RewardModel(nn.Module):
    """Causal transformer over (item, feedback) events -> user state h_t.
    Heads predict the outcome of the *next* event given its item:
      ratio (played ratio, BCE with soft target), like (BCE), cont (session continues after it, BCE)."""

    def __init__(self, n_tokens, d=64, n_layer=2, n_head=2, ctx=CTX, dropout=0.1, hidden=128):
        super().__init__()
        cfg = GPT2Config(vocab_size=n_tokens, n_positions=ctx + 8, n_embd=d, n_layer=n_layer, n_head=n_head,
                         resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout, bos_token_id=0, eos_token_id=0)
        self.backbone = GPT2Model(cfg)
        self.item = nn.Embedding(n_tokens, d)
        self.fb_ratio = nn.Embedding(N_RATIO_BUCKETS, d)
        self.fb_like = nn.Embedding(2, d)
        self.fb_org = nn.Embedding(2, d)
        self.cand_item = nn.Embedding(n_tokens, d)  # candidate-side item embedding
        self.cand_feat = nn.Linear(2, d)             # [log length, organic flag of the candidate event]
        self.mlp = nn.Sequential(nn.Linear(3 * d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 3))
        self.d = d

    def state(self, tok, rb, lk, org, attn=None):
        x = self.item(tok) + self.fb_ratio(rb) + self.fb_like(lk) + self.fb_org(org)
        return self.backbone(inputs_embeds=x, attention_mask=attn).last_hidden_state

    def head(self, h, cand_tok, cand_loglen, cand_org):
        """h [...,d], cand_tok [...]. Returns logits [...,3] (ratio, like, cont)."""
        c = self.cand_item(cand_tok) + self.cand_feat(torch.stack([cand_loglen, cand_org], -1))
        return self.mlp(torch.cat([h, c, h * c], -1))


def loglen(length):
    return (np.log(np.maximum(length, 1.0)) - 5.3).astype(np.float32)  # centred at ~200s


# ----------------------------------------------------------------------------- metrics
def rank_metrics(topk: np.ndarray, targets: list, ks=(10, 100)):
    """topk [U,Kmax] item tokens; targets: list of sets (relevant tokens). Recall/NDCG averaged over users."""
    res = {}
    for k in ks:
        rec, ndcg = [], []
        disc = 1.0 / np.log2(np.arange(2, k + 2))
        for row, tgt in zip(topk[:, :k], targets):
            hits = np.array([t in tgt for t in row], dtype=np.float64)
            rec.append(hits.sum() / min(len(tgt), k))
            idcg = disc[: min(len(tgt), k)].sum()
            ndcg.append((hits * disc).sum() / idcg)
        res[f"recall@{k}"] = float(np.mean(rec))
        res[f"ndcg@{k}"] = float(np.mean(ndcg))
    return res


def beyond_accuracy(topk: np.ndarray, hist_sets: list, item_pop: np.ndarray, item_len: np.ndarray, n_items: int,
                    k=10):
    """Coverage, concentration, popularity, length and repeat statistics of top-k lists."""
    top = topk[:, :k]
    flat = top.ravel()
    uniq, cnt = np.unique(flat, return_counts=True)
    p = cnt / cnt.sum()
    srt = np.sort(cnt)
    n = len(srt)
    gini_rec = float((2 * np.arange(1, n + 1) - n - 1).dot(srt) / (n * srt.sum())) if n > 1 else 0.0
    top1_share = float(np.bincount(top[:, 0]).max() / len(top))
    rep = np.mean([np.mean([t in h for t in row]) for row, h in zip(top, hist_sets)])
    return {f"coverage@{k}": float(len(uniq) / n_items),
            f"entropy@{k}": float(-(p * np.log(p)).sum()),
            f"gini@{k}": gini_rec,
            "top1_max_share": top1_share,
            f"mean_logpop@{k}": float(np.log1p(item_pop[flat]).mean()),
            f"mean_len@{k}": float(item_len[flat].mean()),
            f"repeat_share@{k}": float(rep)}
