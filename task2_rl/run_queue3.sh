#!/usr/bin/env bash
# Robustness queue: seed replicates of key RL configs, a long run (under-training vs capacity),
# full re-evaluation of every policy, multi-step analysis with controls, significance, plots.
set -u
PY="../../.venv/Scripts/python.exe -X utf8"
cd "$(dirname "$0")/src"
export TORCH_THREADS=${TORCH_THREADS:-4}
log() { echo "=== $(date) $*"; }
rl() {  # name, then rl.py args
  local name="$1"; shift
  log "START $name"
  $PY rl.py --gpt gptrep --name "$name" "$@" > ../results/$name.out 2>&1
  log "END $name rc=$?"
}
for s in 1 2; do
  rl rl_ratio_b0.1_h1_s$s  --reward ratio --beta 0.1  --horizon 1 --steps 800 --seed $s
  rl rl_ratio_b0.03_h1_s$s --reward ratio --beta 0.03 --horizon 1 --steps 800 --seed $s
  rl rl_cont_b0.03_h5_s$s  --reward cont  --beta 0.03 --horizon 5 --steps 300 --batch 16 --group 8 --seed $s
  rl rl_ratio_b0.03_h5_s$s --reward ratio --beta 0.03 --horizon 5 --steps 300 --batch 16 --group 8 --seed $s
done
rl rl3_ratio_b0.03_h1_long --reward ratio --beta 0.03 --horizon 1 --steps 6000 --lr 1e-3 --eval_every 1000

ALL="rl_ratio_b0.1_h1 rl_ratio_b0.03_h1 rl_ratio_b0.01_h1 rl_time_b0.03_h1 rl_like_b0.03_h1 rl_cont_b0.03_h1
rl_debiased_b0.03_h1 rl_ratio_b0.03_h5 rl_cont_b0.03_h5 rl2_ratio_b0.03_h1 rl2_ratio_b0.01_h1 rl2_time_b0.5_h1
rl2_like_b0.012_h1 rl2_cont_b0.004_h1 rl2_debiased_b0.03_h1
rl_ratio_b0.1_h1_s1 rl_ratio_b0.03_h1_s1 rl_cont_b0.03_h5_s1 rl_ratio_b0.03_h5_s1
rl_ratio_b0.1_h1_s2 rl_ratio_b0.03_h1_s2 rl_cont_b0.03_h5_s2 rl_ratio_b0.03_h5_s2 rl3_ratio_b0.03_h1_long"
NEW="rl_ratio_b0.1_h1_s1 rl_ratio_b0.03_h1_s1 rl_cont_b0.03_h5_s1 rl_ratio_b0.03_h5_s1
rl_ratio_b0.1_h1_s2 rl_ratio_b0.03_h1_s2 rl_cont_b0.03_h5_s2 rl_ratio_b0.03_h5_s2 rl3_ratio_b0.03_h1_long"
log "eval"; $PY eval_policies.py --base gptrep --runs $ALL > ../results/eval_policies.out 2>&1
log "significance"; $PY significance.py --runs $NEW --out significance_seeds > ../results/significance_seeds.out 2>&1
log "multistep"; $PY multistep.py --base gptrep --policies gptrep rl2_ratio_b0.03_h1 rl2_ratio_b0.01_h1 \
    rl2_like_b0.012_h1 rl2_cont_b0.004_h1 rl_ratio_b0.03_h5 rl_cont_b0.03_h5 rl_ratio_b0.03_h5_s1 rl_cont_b0.03_h5_s1 \
    > ../results/multistep.out 2>&1
log "plots"; $PY plots.py > ../results/plots.out 2>&1
log "QUEUE3 DONE"
