"""Print the DuckDB SQL that a few DSL specs compile to (appendix of the report)."""
import json
import sys
from pathlib import Path

from engine import INF, Spec

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    examples = [Spec("rate", "is_click", ("user_id", "author_id"), "std", 7 * 86400.0, 10.0),
                Spec("recency", "imp", ("user_id", "tab"), "all"),
                Spec("dsum", "is_click", ("video_id",), "all", 1.469 * 86400.0)]
    for fname in sys.argv[1:]:
        j = json.load(open(RES / fname))
        examples += [Spec(p["agg"], p["signal"], tuple(p["key"]), p["filt"], float(p["window"]), float(p["alpha"]))
                     for p in j["specs"][:2]]
    for s in examples:
        print(f"-- {s.name}\n{s.to_sql()};\n")


if __name__ == "__main__":
    main()
