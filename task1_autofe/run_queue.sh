#!/usr/bin/env bash
# Sequential queue of all task-1 experiments (run from task1_autofe/). Each step logs to results/*.out.
set -u
PY="../../.venv/Scripts/python.exe -X utf8"
cd "$(dirname "$0")/src"

run() { echo "=== $(date) START $1"; shift; "$@"; echo "=== $(date) END rc=$?"; }

# wait for the baselines job if it is still running
until grep -q "manual_tuned 2" ../results/baselines.out 2>/dev/null; do sleep 30; done

run "autofe main (logloss)"   $PY exp_autofe.py --variant main --rounds 10 --curve --tune > ../results/autofe_main.out 2>&1
run "autofe main (listwise)"  $PY exp_autofe.py --variant main --rounds 8 --curve --objective listwise > ../results/autofe_listwise.out 2>&1
# replay the LLM agent's proposals with the final engine (same proposals, same screener as the searches)
rm -f ../results/agent_state.json
for r in 0 1 2 3; do
  run "agent replay r$r" $PY agent_round.py --state ../results/agent_state.json --proposals ../agent/proposals_r$r.json > ../agent/screen_r$r.txt 2>&1
done
run "agent eval"              $PY exp_autofe.py --variant agent > ../results/autofe_agent.out 2>&1
run "random programs"         $PY exp_autofe.py --variant random > ../results/autofe_random.out 2>&1
run "grid windows"            $PY exp_autofe.py --variant grid --rounds 6 > ../results/autofe_grid.out 2>&1
run "window sweep"            $PY exp_window.py > ../results/window_sweep.out 2>&1
run "summary"                 $PY summarize.py > ../results/summary.out 2>&1
echo "QUEUE DONE"
