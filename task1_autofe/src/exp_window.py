"""How much does the window choice matter?  Screening gain of a feature template as a function of its
window / half-life, measured on top of (a) the static model and (b) the hand-crafted model without the
template's family.  Train period only."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from autofe import Screener
from data import load
from engine import Spec
from features import MANUAL, FeatureStore

RES = Path(__file__).resolve().parents[1] / "results"
H = 3600.0
TEMPLATES = [
    Spec("rate", "is_click", ("user_id",), "all", 7 * 24 * H),
    Spec("rate", "is_click", ("video_id",), "all", 7 * 24 * H),
    Spec("rate", "is_click", ("user_id", "author_id"), "all", 7 * 24 * H),
    Spec("sum", "imp", ("video_id",), "all", 7 * 24 * H),
    Spec("rate", "long_view", ("user_id", "tag"), "all", 7 * 24 * H),
    Spec("drate", "is_click", ("user_id",), "all", 24 * H),
]
WINDOWS = np.exp(np.linspace(np.log(0.25 * H), np.log(30 * 24 * H), 17))


def main():
    ev = load()
    fs = FeatureStore(ev.filter(pl.col("split").is_in(["warmup", "train"])))
    scr = Screener(fs)
    out = {"windows_h": (WINDOWS / H).tolist(), "curves": {}}
    for base_name, base in (("static", []), ("manual", MANUAL)):
        for tpl in TEMPLATES:
            family = [s for s in base if not (s.key == tpl.key and s.signal == tpl.signal)]
            scr.refit(family)
            gains = []
            for w in WINDOWS:
                s = replace(tpl, window=float(w))
                v = fs.engine.compute(s, rows=scr.ab_rows)
                gains.append(scr.gain(v[scr.a_in_ab], v[~scr.a_in_ab]))
            key = f"{base_name}|{tpl.agg}({tpl.signal}|{'+'.join(tpl.key)})"
            out["curves"][key] = gains
            best = int(np.argmax(gains))
            print(f"{key:55s} best window {WINDOWS[best] / H:8.1f}h gain {gains[best]:.5f}  "
                  f"7d gain {np.interp(np.log(168 * H), np.log(WINDOWS), gains):.5f}", flush=True)
    json.dump(out, open(RES / "window_sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
