#!/usr/bin/env bash
# Second RL queue: longer runs closer to the KL-regularised optimum, and KL-matched beta for every reward
# (beta chosen from the closed-form curves so that KL(pi*||pi_ref) is ~0.5-1 nats for each reward).
set -u
PY="../../.venv/Scripts/python.exe -X utf8"
cd "$(dirname "$0")/src"
export TORCH_THREADS=${TORCH_THREADS:-2}
rl2() {  # reward beta
  local name="rl2_$1_b$2_h1"
  echo "=== $(date) START $name"
  $PY rl.py --gpt gptrep --reward "$1" --beta "$2" --horizon 1 --steps 2000 --lr 1e-3 --eval_every 500 \
      --name "$name" > ../results/$name.out 2>&1
  echo "=== $(date) END $name rc=$?"
}
rl2 ratio 0.03
rl2 ratio 0.01
rl2 time 0.5
rl2 like 0.012
rl2 cont 0.004
rl2 debiased 0.03
echo "QUEUE2 DONE"
