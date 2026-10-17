# Notes on the two referenced papers (for the report)

## [1] Petrov & Macdonald. Aligning GPTRec with Beyond-Accuracy Goals with RL. GenRec@WWW 2024 (arXiv 2403.04875)
- GPT-2 (3 blocks, d=256), item IDs as tokens, Next-K autoregressive list generation (K=10).
- Reward = metric on generated list (NOT a learned RM): NDCG@10 + λ·(ILD diversity | −popularity count), per-position decomposition.
- Stage 1: teacher-student pretraining (Markov-chain / BERT4Rec teacher top-K lists) — fixes Next-K degradation
  (SASRec loses 33.7% NDCG@10 when run autoregressively in Next-K mode).
- Stage 2: PPO actor-critic + GAE, clip only, **no KL to reference**, no entropy bonus; 64k steps, best ckpt on val.
- Leave-one-out split (not temporal). ML-1M, Steam-2M.
- Results: RL for NDCG alone ≈ +2.6% (n.s.); strong secondary weights cost 24–38% NDCG; RL beats greedy re-ranking 3/4.
- No reward hacking / long-term / multi-step drift analysis beyond Next-K vs Top-K.

## [2] Matveev et al. (AI VK / HSE). Session-Level Optimization for Large-Scale Retrieval using REINFORCE with Multi-Step Off-Policy Correction. KDD 2026 workshop (arXiv 2607.02818)
- Yambda-5B, temporal split (last week test). Two-tower: causal transformer user tower (4L, d=256), MLP item tower.
- Reward = logged signal r_t = 1{like} + 0.1·played_ratio; discounted session return, γ=0.9.
- Off-policy REINFORCE from logs (no environment): multi-step importance weights Π π_θ/π_b over K=10 steps, cap c=0.5;
  no baseline, **no KL**; lr 1e-4.
- Feedback model (learned) used only for OPE (direct method) and test-time scaling (best-of-N rollouts, T=8).
  Feedback model: ratio MAE 0.177, like PR-AUC 0.243.
- Results: REINFORCE improves OPE return (DR .6046 → .6306) with small Recall@100 drop (.3153 → .3086).
  Larger weight caps → more variance, worse recall.
- Authors note circularity: same feedback model selects and evaluates rollouts.

## How our work differs
- RL is done **on-policy against a learned reward model** (closer to RLHF), with explicit KL to the reference,
  exact (full-catalog) KL since the action space is small enough to enumerate.
- Explicit study of reward hacking (short tracks, repeats, popularity collapse), hold-out RM to measure over-optimization,
  multi-step drift, and long-term reward variants.
