"""Feedback text shown to the LLM feature proposer after a screening round (exactly what the agent sees).

usage: python agent_feedback.py ../results/agent_state_run2.json > ../agent/run2/feedback_r0.txt
"""
import json
import sys


def main():
    s = json.load(open(sys.argv[1]))
    h = s["history"][-1]
    print("Accepted so far (now part of the model):")
    for a in s["accepted_specs"]:
        print("  ", a["name"])
    print(f"Round {h['round']} marginal gains of your proposals (logloss reduction on held-out days vs. the model "
          f"at the start of the round; negative = hurts):")
    for r in h["results"]:
        print(f"  {r['gain']:+.5f} {'ACCEPTED' if r['accepted'] else ''} {r['name']}")
    for e in h.get("invalid", []):
        print("  INVALID", e["error"], json.dumps(e["proposal"]))


if __name__ == "__main__":
    main()
