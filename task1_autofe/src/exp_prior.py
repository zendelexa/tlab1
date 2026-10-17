"""Robustness check: does the choice of the smoothing prior change the comparison of feature sets?
Default engine: point-in-time prior (global mean strictly before t). Alternative: one constant per
(signal, filter) estimated on all rows before the validation period (warm-up + train)."""
import json
from pathlib import Path

from data import load
from engine import Engine, Spec
from exp_baselines import run
from features import MANUAL, FeatureStore

RES = Path(__file__).resolve().parents[1] / "results"


def specs_from(fname):
    return [Spec(p["agg"], p["signal"], tuple(p["key"]), p["filt"], float(p["window"]), float(p["alpha"]))
            for p in json.load(open(RES / fname))["specs"]]


def main():
    ev = load()
    sets = {"manual": list(MANUAL), "autofe_logloss": specs_from("autofe_main_s0.json"),
            "autofe_listwise": specs_from("autofe_main_s0_listwise.json"),
            "llm_agent": specs_from("autofe_agent_s0.json")}
    fs = FeatureStore(ev)
    fs.engine = Engine(ev, prior_mode="const", prior_rows=ev["split"].is_in(["warmup", "train"]).to_numpy())
    out = {name: run(fs, specs, f"{name}_constprior") for name, specs in sets.items()}
    json.dump(out, open(RES / "prior_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
