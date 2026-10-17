"""Hand-crafted baseline feature set (expressed in the same DSL) and feature-matrix assembly."""
import numpy as np
import polars as pl

from engine import INF, Engine, Spec
from model import STATIC

D, H = 86400.0, 3600.0

MANUAL = [
    # user activity / preferences
    Spec("sum", "imp", ("user_id",), "all", H), Spec("sum", "imp", ("user_id",), "all", D),
    Spec("sum", "imp", ("user_id",), "all", 7 * D), Spec("sum", "imp", ("user_id",), "all", INF),
    Spec("sum", "is_click", ("user_id",), "all", D), Spec("sum", "is_click", ("user_id",), "all", 7 * D),
    Spec("rate", "is_click", ("user_id",), "all", 7 * D), Spec("rate", "is_click", ("user_id",), "all", INF),
    Spec("rate", "long_view", ("user_id",), "all", 7 * D), Spec("rate", "is_like", ("user_id",), "all", 7 * D),
    Spec("rate", "play_ratio", ("user_id",), "all", 7 * D),
    Spec("recency", "imp", ("user_id",)), Spec("recency", "is_click", ("user_id",)),
    Spec("rate", "is_click", ("user_id", "is_rand"), "all", 7 * D),
    # item popularity / quality
    Spec("sum", "imp", ("video_id",), "all", D), Spec("sum", "imp", ("video_id",), "all", 7 * D),
    Spec("sum", "imp", ("video_id",), "all", INF),
    Spec("rate", "is_click", ("video_id",), "all", D), Spec("rate", "is_click", ("video_id",), "all", 7 * D),
    Spec("rate", "is_click", ("video_id",), "all", INF),
    Spec("rate", "long_view", ("video_id",), "all", 7 * D), Spec("rate", "is_like", ("video_id",), "all", 7 * D),
    Spec("rate", "play_ratio", ("video_id",), "all", 7 * D),
    # author / music / tag
    Spec("sum", "imp", ("author_id",), "all", 7 * D), Spec("rate", "is_click", ("author_id",), "all", 7 * D),
    Spec("rate", "long_view", ("author_id",), "all", 7 * D), Spec("rate", "is_click", ("music_id",), "all", 7 * D),
    Spec("rate", "is_click", ("tag",), "all", 7 * D),
    # user x item
    Spec("sum", "imp", ("user_id", "video_id"), "all", INF), Spec("sum", "is_click", ("user_id", "video_id"), "all", INF),
    Spec("recency", "imp", ("user_id", "video_id")),
    # user x author
    Spec("sum", "imp", ("user_id", "author_id"), "all", INF),
    Spec("sum", "is_click", ("user_id", "author_id"), "all", INF),
    Spec("rate", "is_click", ("user_id", "author_id"), "all", INF),
    # user x tag
    Spec("sum", "imp", ("user_id", "tag"), "all", 7 * D), Spec("sum", "is_click", ("user_id", "tag"), "all", 7 * D),
    Spec("rate", "is_click", ("user_id", "tag"), "all", 7 * D),
    Spec("rate", "is_click", ("user_id", "tag"), "all", INF),
    # user x music, item x user segment
    Spec("sum", "is_click", ("user_id", "music_id"), "all", INF),
    Spec("rate", "is_click", ("video_id", "user_active_degree"), "all", 7 * D),
]


class FeatureStore:
    """Computes features once for all events and slices the rows needed by an experiment."""

    def __init__(self, ev: pl.DataFrame):
        self.ev = ev
        self.engine = Engine(ev)
        self.cache = {}

    def values(self, spec: Spec) -> np.ndarray:
        if spec not in self.cache:                     # keyed by the exact spec, not its rounded name
            self.cache[spec] = self.engine.compute(spec)
        return self.cache[spec]

    def matrix(self, specs, rows: np.ndarray, static=STATIC, extra=()) -> pl.DataFrame:
        cols = {c: self.ev[c].to_numpy()[rows] for c in list(static) + list(extra)}
        for s in specs:
            cols[s.name] = self.values(s)[rows]
        return pl.DataFrame(cols)

