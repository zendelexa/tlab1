#!/usr/bin/env bash
# Second task-1 queue: window sweep on the final engine, LLM+search hybrid, search-seed robustness,
# evaluation of the second LLM-agent run, final comparison with bootstrap CIs, summary tables/figures.
set -u
PY="../../.venv/Scripts/python.exe -X utf8"
cd "$(dirname "$0")/src"
log() { echo "=== $(date) $*"; }

log "window sweep";  $PY exp_window.py > ../results/window_sweep.out 2>&1
log "hybrid";        $PY exp_autofe.py --variant hybrid --rounds 5 --curve > ../results/autofe_hybrid.out 2>&1
for s in 1 2; do
  log "main seed $s"; $PY exp_autofe.py --variant main --rounds 10 --seed $s > ../results/autofe_main_s$s.out 2>&1
  log "grid seed $s"; $PY exp_autofe.py --variant grid --rounds 6 --seed $s > ../results/autofe_grid_s$s.out 2>&1
done
# the second LLM-agent run is screened interactively; wait until its 4 rounds are done
until [ -f ../results/agent_state_run2.json ] && \
      [ "$($PY -c "import json;print(json.load(open('../results/agent_state_run2.json'))['round'])")" = "4" ]; do
  sleep 60
done
log "agent run2";    $PY exp_autofe.py --variant agent --agent_state agent_state_run2.json > ../results/autofe_agent_run2.out 2>&1
log "final";         $PY exp_final.py > ../results/final_comparison.out 2>&1
log "summary";       $PY summarize.py > ../results/summary.out 2>&1
log "QUEUE2 DONE"
