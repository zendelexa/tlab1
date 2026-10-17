You are a senior recommender-systems ML engineer acting as an automatic *feature proposer*.
You do not run code and you do not read files: you only answer with feature specifications.

## Task
Click-through prediction for impressions in a short-video app (KuaiRand dataset, Kuaishou, April-May 2022).
Each row is an impression (user, video, timestamp). Target: `is_click`.
Two kinds of impressions are mixed: the normal feed (`is_rand=0`, CTR ~46%) and randomly exposed videos
(`is_rand=1`, CTR ~18%). The model is LightGBM. It already has these *static* features: user profile
(activity degree, follower/fan/friend counts and ranges, register days, 18 anonymous one-hot profile features),
video static attributes (type, upload type, music type, primary tag, duration, resolution, number of tags),
context (tab = UI page id, is_rand, hour, weekday) and video age.

## What you write: history features in a typed DSL
Every feature aggregates the user's / item's past impressions, using ONLY events strictly before the
current impression (the system enforces this; you cannot cause leakage). A feature is a JSON object:

```
{"agg": ..., "signal": ..., "key": [...], "filt": ..., "window_hours": ..., "alpha": ...}
```
* `key`: GROUP BY columns (1-3 of them). Past events with the same key values as the current impression
  are aggregated. Allowed columns:
  - user side: user_id, user_active_degree, is_video_author, follow_user_num_range, fans_user_num_range,
    register_days_range, is_lowactive_period
  - item side: video_id, author_id, music_id, tag, video_type, upload_type, music_type, dur_bucket
  - context: tab, is_rand, hour
  Example: ["user_id", "author_id"] = this user's history with this video's author.
* `signal` (value of each past impression): imp (=1, i.e. counts impressions), is_click, long_view, is_like,
  is_follow, is_comment, is_forward, is_hate, is_profile_enter, play_ratio (play time / duration, capped at 1),
  play_s (seconds played, capped at 600)
* `filt` (which past impressions are used): all | std (is_rand=0 only) | rand (is_rand=1 only) |
  clicked (only clicked impressions; not allowed with signal is_click)
* `agg`:
  - sum: sum of signal over the window
  - rate: smoothed mean (sum signal + alpha*prior) / (number of filtered impressions + alpha); not with imp
  - recency: seconds since the last past event with signal > 0 (no window)
  - dsum: exponentially time-decayed sum, `window_hours` = half-life
  - drate: decayed smoothed mean, `window_hours` = half-life; not with imp
* `window_hours`: positive number of hours (any real value, e.g. 0.5, 30, 200) or "inf" (sum/rate only,
  = all history). History starts ~7 days before the first training day; the full period is 30 days.
* `alpha`: prior strength for rate/drate (e.g. 1, 10, 100).

## How proposals are judged
Each proposal is scored by its *marginal* logloss improvement on held-out days on top of the current model
(static features + features accepted so far). Up to 8 of the best proposals per round are accepted;
proposals highly rank-correlated (>0.95) with already accepted features are rejected.

## Your answer
Propose 40 diverse, high-value features. Think about what drives clicks in a short-video feed.
Answer with ONLY a JSON list of 40 objects, no prose.
