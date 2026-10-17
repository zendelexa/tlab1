"""One round of the 'LLM agent as feature engineer' loop.

The agent writes feature *specifications* in the DSL (JSON), never SQL or code, so it cannot introduce
leakage. Each round:  read proposals -> screen them exactly like the automatic search -> accept the best ->
write a feedback report that is shown to the agent in the next round.

usage: python agent_round.py --state results/agent_state.json --proposals proposals_r0.json
"""
import argparse
import json
from pathlib import Path

import polars as pl

from autofe import search
from data import load
from engine import INF, SIGNALS, Spec
from features import FeatureStore

ALLOWED_KEYS = {"user_id", "user_active_degree", "is_video_author", "follow_user_num_range", "fans_user_num_range",
                "register_days_range", "is_lowactive_period", "video_id", "author_id", "music_id", "tag",
                "video_type", "upload_type", "music_type", "dur_bucket", "tab", "is_rand", "hour"}


def parse(p: dict) -> Spec:
    w = p.get("window_hours", "inf")
    window = INF if w in ("inf", None) or (isinstance(w, (int, float)) and w <= 0) else float(w) * 3600.0
    s = Spec(agg=p["agg"], signal=p["signal"], key=tuple(p["key"]), filt=p.get("filt", "all"),
             window=INF if p["agg"] == "recency" else window, alpha=float(p.get("alpha", 10.0)))
    if s.signal not in SIGNALS or not set(s.key) <= ALLOWED_KEYS or s.agg not in ("sum", "rate", "recency", "dsum",
                                                                                  "drate"):
        raise ValueError("unknown field")
    if not s.valid():
        raise ValueError("invalid combination")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--per_round", type=int, default=8)
    args = ap.parse_args()
    state = json.loads(Path(args.state).read_text()) if Path(args.state).exists() else \
        dict(round=0, accepted_specs=[], history=[])
    accepted = [Spec(p["agg"], p["signal"], tuple(p["key"]), p["filt"], p["window"], p["alpha"])
                for p in state["accepted_specs"]]
    props, errors = [], []
    for p in json.loads(Path(args.proposals).read_text()):
        try:
            props.append(parse(p))
        except Exception as e:  # noqa: BLE001 -- report invalid proposals back to the agent
            errors.append(dict(proposal=p, error=str(e)))

    ev = load()
    fs = FeatureStore(ev.filter(pl.col("split").is_in(["warmup", "train"])))
    fs.engine.max_groups = 40
    for s in accepted:
        fs.values(s)
    new_acc, hist, log = search(fs, rounds=1, n_random=0, n_mut=0, per_round=args.per_round, init=accepted,
                                extra_proposals=lambda r, h, a: props, seed=state["round"])
    added = {a["name"] for a in log[0]["added"]}
    rows = sorted(([dict(name=k, gain=round(v["gain"], 6), accepted=k in added) for k, v in hist.items()]),
                  key=lambda x: -x["gain"])
    state["history"].append(dict(round=state["round"], base_screen_ll=log[0]["base_screen_ll"], results=rows,
                                 invalid=errors))
    state["accepted_specs"] = [dict(name=s.name, agg=s.agg, signal=s.signal, key=list(s.key), filt=s.filt,
                                    window=s.window, alpha=s.alpha) for s in new_acc]
    state["round"] += 1
    Path(args.state).write_text(json.dumps(state, indent=1))
    print(f"ROUND {state['round'] - 1}: screened {len(rows)} valid proposals, {len(errors)} invalid, "
          f"accepted {len(added)}; total accepted {len(new_acc)}")
    for r in rows:
        print(f"  gain={r['gain']:+.6f} {'ACCEPTED' if r['accepted'] else '        '} {r['name']}")
    for e in errors:
        print("  INVALID", e)


if __name__ == "__main__":
    main()
