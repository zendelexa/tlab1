#!/usr/bin/env bash
# Final task-2 evaluation: every RL run under one protocol, multi-step analysis, figures.
set -u
PY="../../.venv/Scripts/python.exe -X utf8"
cd "$(dirname "$0")/src"
until grep -q "QUEUE2 DONE" ../results/queue2.log 2>/dev/null; do sleep 20; done
RUNS="rl_ratio_b0.1_h1 rl_ratio_b0.03_h1 rl_ratio_b0.01_h1 rl_time_b0.03_h1 rl_like_b0.03_h1 rl_cont_b0.03_h1
rl_debiased_b0.03_h1 rl_ratio_b0.03_h5 rl_cont_b0.03_h5 rl2_ratio_b0.03_h1 rl2_ratio_b0.01_h1 rl2_time_b0.5_h1
rl2_like_b0.012_h1 rl2_cont_b0.004_h1 rl2_debiased_b0.03_h1"
echo "=== $(date) eval"
$PY eval_policies.py --base gptrep --runs $RUNS > ../results/eval_policies.out 2>&1
echo "=== $(date) multistep"
$PY multistep.py --base gptrep --policies gptrep rl2_ratio_b0.03_h1 rl2_ratio_b0.01_h1 rl2_like_b0.012_h1 \
    rl2_cont_b0.004_h1 rl_ratio_b0.03_h5 rl_cont_b0.03_h5 > ../results/multistep.out 2>&1
echo "=== $(date) plots"
$PY plots.py > ../results/plots.out 2>&1
echo "FINAL DONE $(date)"
