#!/usr/bin/env bash
# Sequential queue of task-2 experiments after GPT / repeat head / reward models are trained.
set -u
PY="../../.venv/Scripts/python.exe -X utf8"
cd "$(dirname "$0")/src"
run() { echo "=== $(date) START $1"; shift; "$@"; echo "=== $(date) END rc=$?"; }
BASE=gptrep
STEPS=${STEPS:-800}

if [ "${SKIP_ANALYSIS:-0}" != "1" ]; then
  run "closed-form tilting" $PY analysis.py --gpt $BASE > ../results/analysis.out 2>&1
fi

RUNS=()
rl() {  # reward beta horizon [extra args]
  local name="rl_$1_b$2_h$3"
  RUNS+=("$name")
  run "RL $name" $PY rl.py --gpt $BASE --reward "$1" --beta "$2" --horizon "$3" --steps $STEPS --name "$name" "${@:4}" \
      > ../results/$name.out 2>&1
}
# beta sweep for the main reward
for b in ${BETAS_RATIO:-0.1 0.03 0.01}; do rl ratio $b 1; done
# other reward definitions at a fixed beta
for k in time like cont debiased; do rl $k ${BETA_OTHER:-0.03} 1; done
# multi-step (H=5) rollouts with the user simulator
STEPS=300 rl ratio ${BETA_OTHER:-0.03} 5 --batch 16 --group 8
STEPS=300 rl cont ${BETA_OTHER:-0.03} 5 --batch 16 --group 8

run "eval policies" $PY eval_policies.py --base $BASE --runs "${RUNS[@]}" > ../results/eval_policies.out 2>&1
run "multistep" $PY multistep.py --base $BASE --policies $BASE "${RUNS[@]}" > ../results/multistep.out 2>&1
run "plots" $PY plots.py > ../results/plots.out 2>&1
echo "QUEUE DONE"
