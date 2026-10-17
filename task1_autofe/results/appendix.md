## Приложение A. Списки признаков по методам

### Ручной набор (MANUAL)

*40 признаков; агрегаты {'sum': 17, 'rate': 20, 'recency': 3}; фильтры {'all': 40}; с item-side колонкой в ключе: 26; только user/context ключ: 14; «круглых» окон (1ч/1д/7д/∞): 37 из 37*

| # | признак | агрегат | сигнал | ключ | фильтр | окно / полупериод | α |
|---|---|---|---|---|---|---|---|
| 1 | `sum(imp\|user_id\|all\|w=1h)` | sum | imp | user_id | all | 1h | — |
| 2 | `sum(imp\|user_id\|all\|w=1d)` | sum | imp | user_id | all | 1d | — |
| 3 | `sum(imp\|user_id\|all\|w=7d)` | sum | imp | user_id | all | 7d | — |
| 4 | `sum(imp\|user_id\|all\|w=inf)` | sum | imp | user_id | all | inf | — |
| 5 | `sum(is_click\|user_id\|all\|w=1d)` | sum | is_click | user_id | all | 1d | — |
| 6 | `sum(is_click\|user_id\|all\|w=7d)` | sum | is_click | user_id | all | 7d | — |
| 7 | `rate(is_click\|user_id\|all\|w=7d\|a=10)` | rate | is_click | user_id | all | 7d | 10 |
| 8 | `rate(is_click\|user_id\|all\|w=inf\|a=10)` | rate | is_click | user_id | all | inf | 10 |
| 9 | `rate(long_view\|user_id\|all\|w=7d\|a=10)` | rate | long_view | user_id | all | 7d | 10 |
| 10 | `rate(is_like\|user_id\|all\|w=7d\|a=10)` | rate | is_like | user_id | all | 7d | 10 |
| 11 | `rate(play_ratio\|user_id\|all\|w=7d\|a=10)` | rate | play_ratio | user_id | all | 7d | 10 |
| 12 | `recency(imp\|user_id\|all)` | recency | imp | user_id | all | — | — |
| 13 | `recency(is_click\|user_id\|all)` | recency | is_click | user_id | all | — | — |
| 14 | `rate(is_click\|user_id+is_rand\|all\|w=7d\|a=10)` | rate | is_click | user_id×is_rand | all | 7d | 10 |
| 15 | `sum(imp\|video_id\|all\|w=1d)` | sum | imp | video_id | all | 1d | — |
| 16 | `sum(imp\|video_id\|all\|w=7d)` | sum | imp | video_id | all | 7d | — |
| 17 | `sum(imp\|video_id\|all\|w=inf)` | sum | imp | video_id | all | inf | — |
| 18 | `rate(is_click\|video_id\|all\|w=1d\|a=10)` | rate | is_click | video_id | all | 1d | 10 |
| 19 | `rate(is_click\|video_id\|all\|w=7d\|a=10)` | rate | is_click | video_id | all | 7d | 10 |
| 20 | `rate(is_click\|video_id\|all\|w=inf\|a=10)` | rate | is_click | video_id | all | inf | 10 |
| 21 | `rate(long_view\|video_id\|all\|w=7d\|a=10)` | rate | long_view | video_id | all | 7d | 10 |
| 22 | `rate(is_like\|video_id\|all\|w=7d\|a=10)` | rate | is_like | video_id | all | 7d | 10 |
| 23 | `rate(play_ratio\|video_id\|all\|w=7d\|a=10)` | rate | play_ratio | video_id | all | 7d | 10 |
| 24 | `sum(imp\|author_id\|all\|w=7d)` | sum | imp | author_id | all | 7d | — |
| 25 | `rate(is_click\|author_id\|all\|w=7d\|a=10)` | rate | is_click | author_id | all | 7d | 10 |
| 26 | `rate(long_view\|author_id\|all\|w=7d\|a=10)` | rate | long_view | author_id | all | 7d | 10 |
| 27 | `rate(is_click\|music_id\|all\|w=7d\|a=10)` | rate | is_click | music_id | all | 7d | 10 |
| 28 | `rate(is_click\|tag\|all\|w=7d\|a=10)` | rate | is_click | tag | all | 7d | 10 |
| 29 | `sum(imp\|user_id+video_id\|all\|w=inf)` | sum | imp | user_id×video_id | all | inf | — |
| 30 | `sum(is_click\|user_id+video_id\|all\|w=inf)` | sum | is_click | user_id×video_id | all | inf | — |
| 31 | `recency(imp\|user_id+video_id\|all)` | recency | imp | user_id×video_id | all | — | — |
| 32 | `sum(imp\|user_id+author_id\|all\|w=inf)` | sum | imp | user_id×author_id | all | inf | — |
| 33 | `sum(is_click\|user_id+author_id\|all\|w=inf)` | sum | is_click | user_id×author_id | all | inf | — |
| 34 | `rate(is_click\|user_id+author_id\|all\|w=inf\|a=10)` | rate | is_click | user_id×author_id | all | inf | 10 |
| 35 | `sum(imp\|user_id+tag\|all\|w=7d)` | sum | imp | user_id×tag | all | 7d | — |
| 36 | `sum(is_click\|user_id+tag\|all\|w=7d)` | sum | is_click | user_id×tag | all | 7d | — |
| 37 | `rate(is_click\|user_id+tag\|all\|w=7d\|a=10)` | rate | is_click | user_id×tag | all | 7d | 10 |
| 38 | `rate(is_click\|user_id+tag\|all\|w=inf\|a=10)` | rate | is_click | user_id×tag | all | inf | 10 |
| 39 | `sum(is_click\|user_id+music_id\|all\|w=inf)` | sum | is_click | user_id×music_id | all | inf | — |
| 40 | `rate(is_click\|video_id+user_active_degree\|all\|w=7d\|a=10)` | rate | is_click | video_id×user_active_degree | all | 7d | 10 |

### AutoFE, цель logloss, сид 0

*38 признаков; агрегаты {'drate': 10, 'dsum': 5, 'recency': 4, 'sum': 7, 'rate': 12}; фильтры {'rand': 14, 'all': 19, 'std': 3, 'clicked': 2}; с item-side колонкой в ключе: 35; только user/context ключ: 3; «круглых» окон (1ч/1д/7д/∞): 12 из 34*

| # | признак | агрегат | сигнал | ключ | фильтр | окно / полупериод | α |
|---|---|---|---|---|---|---|---|
| 1 | `drate(is_click\|user_id+music_type+is_rand\|rand\|hl=13.36h\|a=10)` | drate | is_click | user_id×music_type×is_rand | rand | 13.36h | 10 |
| 2 | `dsum(is_click\|video_id\|all\|hl=1.469d)` | dsum | is_click | video_id | all | 1.469d | — |
| 3 | `recency(imp\|user_id+music_type\|all)` | recency | imp | user_id×music_type | all | — | — |
| 4 | `sum(play_ratio\|user_id+upload_type\|rand\|w=inf)` | sum | play_ratio | user_id×upload_type | rand | inf | — |
| 5 | `drate(play_ratio\|fans_user_num_range+music_id+tab\|std\|hl=3.318d\|a=1)` | drate | play_ratio | fans_user_num_range×music_id×tab | std | 3.318d | 1 |
| 6 | `recency(play_s\|user_id+video_type\|rand)` | recency | play_s | user_id×video_type | rand | — | — |
| 7 | `drate(is_click\|user_id+music_type+is_rand\|all\|hl=13.36h\|a=10)` | drate | is_click | user_id×music_type×is_rand | all | 13.36h | 10 |
| 8 | `dsum(imp\|user_id+tag\|clicked\|hl=18.33h)` | dsum | imp | user_id×tag | clicked | 18.33h | — |
| 9 | `rate(play_s\|user_id+video_type\|rand\|w=36.46m\|a=1)` | rate | play_s | user_id×video_type | rand | 36.46m | 1 |
| 10 | `sum(imp\|user_id+upload_type\|rand\|w=inf)` | sum | imp | user_id×upload_type | rand | inf | — |
| 11 | `recency(long_view\|user_id+music_type\|all)` | recency | long_view | user_id×music_type | all | — | — |
| 12 | `rate(long_view\|music_type\|all\|w=inf\|a=10)` | rate | long_view | music_type | all | inf | 10 |
| 13 | `drate(play_ratio\|fans_user_num_range+music_id\|all\|hl=9.842h\|a=10)` | drate | play_ratio | fans_user_num_range×music_id | all | 9.842h | 10 |
| 14 | `rate(play_s\|user_id+video_type\|rand\|w=inf\|a=1)` | rate | play_s | user_id×video_type | rand | inf | 1 |
| 15 | `drate(is_click\|is_video_author+music_id\|all\|hl=3.427d\|a=10)` | drate | is_click | is_video_author×music_id | all | 3.427d | 10 |
| 16 | `dsum(imp\|user_id+music_type\|all\|hl=3d)` | dsum | imp | user_id×music_type | all | 3d | — |
| 17 | `drate(is_click\|user_id+music_type\|rand\|hl=13.36h\|a=10)` | drate | is_click | user_id×music_type | rand | 13.36h | 10 |
| 18 | `sum(play_ratio\|user_id+upload_type\|clicked\|w=inf)` | sum | play_ratio | user_id×upload_type | clicked | inf | — |
| 19 | `rate(play_s\|user_id+video_type\|all\|w=inf\|a=1)` | rate | play_s | user_id×video_type | all | inf | 1 |
| 20 | `drate(is_click\|is_video_author+music_id\|rand\|hl=3.427d\|a=10)` | drate | is_click | is_video_author×music_id | rand | 3.427d | 10 |
| 21 | `dsum(long_view\|user_id+music_type\|rand\|hl=1.893d)` | dsum | long_view | user_id×music_type | rand | 1.893d | — |
| 22 | `dsum(play_s\|video_type+hour\|all\|hl=8.933h)` | dsum | play_s | video_type×hour | all | 8.933h | — |
| 23 | `drate(play_ratio\|author_id\|all\|hl=2.584d\|a=1)` | drate | play_ratio | author_id | all | 2.584d | 1 |
| 24 | `rate(long_view\|user_id+video_type\|rand\|w=inf\|a=1)` | rate | long_view | user_id×video_type | rand | inf | 1 |
| 25 | `sum(is_click\|user_id+author_id\|all\|w=12.75d)` | sum | is_click | user_id×author_id | all | 12.75d | — |
| 26 | `rate(play_s\|user_id+music_id\|all\|w=inf\|a=1)` | rate | play_s | user_id×music_id | all | inf | 1 |
| 27 | `rate(is_forward\|author_id+is_rand\|all\|w=2.911d\|a=10)` | rate | is_forward | author_id×is_rand | all | 2.911d | 10 |
| 28 | `rate(is_click\|is_lowactive_period+author_id\|std\|w=7.636d\|a=10)` | rate | is_click | is_lowactive_period×author_id | std | 7.636d | 10 |
| 29 | `recency(imp\|user_id+music_type+hour\|all)` | recency | imp | user_id×music_type×hour | all | — | — |
| 30 | `drate(play_ratio\|user_id+music_type\|std\|hl=4.143d\|a=10)` | drate | play_ratio | user_id×music_type | std | 4.143d | 10 |
| 31 | `sum(long_view\|video_id+is_rand\|all\|w=6.35d)` | sum | long_view | video_id×is_rand | all | 6.35d | — |
| 32 | `drate(is_click\|user_id+is_rand\|rand\|hl=13.36h\|a=10)` | drate | is_click | user_id×is_rand | rand | 13.36h | 10 |
| 33 | `rate(play_s\|user_id+video_type\|rand\|w=inf\|a=10)` | rate | play_s | user_id×video_type | rand | inf | 10 |
| 34 | `rate(play_s\|user_id+video_type+is_rand\|rand\|w=inf\|a=1)` | rate | play_s | user_id×video_type×is_rand | rand | inf | 1 |
| 35 | `rate(is_click\|fans_user_num_range+author_id\|all\|w=inf\|a=10)` | rate | is_click | fans_user_num_range×author_id | all | inf | 10 |
| 36 | `sum(play_s\|follow_user_num_range+tab\|all\|w=5.436d)` | sum | play_s | follow_user_num_range×tab | all | 5.436d | — |
| 37 | `rate(is_click\|user_id\|all\|w=26.83d\|a=10)` | rate | is_click | user_id | all | 26.83d | 10 |
| 38 | `sum(long_view\|user_id+upload_type\|rand\|w=inf)` | sum | long_view | user_id×upload_type | rand | inf | — |

### AutoFE, цель listwise

*38 признаков; агрегаты {'dsum': 9, 'rate': 5, 'sum': 14, 'recency': 2, 'drate': 8}; фильтры {'all': 22, 'std': 8, 'clicked': 2, 'rand': 6}; с item-side колонкой в ключе: 34; только user/context ключ: 4; «круглых» окон (1ч/1д/7д/∞): 1 из 36*

| # | признак | агрегат | сигнал | ключ | фильтр | окно / полупериод | α |
|---|---|---|---|---|---|---|---|
| 1 | `dsum(is_click\|video_id\|all\|hl=1.469d)` | dsum | is_click | video_id | all | 1.469d | — |
| 2 | `dsum(play_s\|fans_user_num_range+tag\|std\|hl=1.623d)` | dsum | play_s | fans_user_num_range×tag | std | 1.623d | — |
| 3 | `rate(long_view\|is_video_author+video_type\|all\|w=2.131d\|a=100)` | rate | long_view | is_video_author×video_type | all | 2.131d | 100 |
| 4 | `sum(is_follow\|register_days_range+dur_bucket+is_rand\|clicked\|w=6.782d)` | sum | is_follow | register_days_range×dur_bucket×is_rand | clicked | 6.782d | — |
| 5 | `recency(is_click\|follow_user_num_range+music_id\|all)` | recency | is_click | follow_user_num_range×music_id | all | — | — |
| 6 | `sum(imp\|register_days_range+video_id\|std\|w=2.813d)` | sum | imp | register_days_range×video_id | std | 2.813d | — |
| 7 | `drate(play_ratio\|fans_user_num_range+music_id+tab\|rand\|hl=3.318d\|a=1)` | drate | play_ratio | fans_user_num_range×music_id×tab | rand | 3.318d | 1 |
| 8 | `sum(imp\|is_video_author+music_type\|clicked\|w=4.031d)` | sum | imp | is_video_author×music_type | clicked | 4.031d | — |
| 9 | `rate(play_s\|is_lowactive_period+music_type\|std\|w=inf\|a=10)` | rate | play_s | is_lowactive_period×music_type | std | inf | 10 |
| 10 | `dsum(is_click\|video_id\|rand\|hl=1.469d)` | dsum | is_click | video_id | rand | 1.469d | — |
| 11 | `drate(long_view\|fans_user_num_range+tag\|rand\|hl=1.041d\|a=10)` | drate | long_view | fans_user_num_range×tag | rand | 1.041d | 10 |
| 12 | `sum(is_profile_enter\|is_video_author+author_id\|all\|w=6.327h)` | sum | is_profile_enter | is_video_author×author_id | all | 6.327h | — |
| 13 | `sum(long_view\|dur_bucket\|all\|w=9.01d)` | sum | long_view | dur_bucket | all | 9.01d | — |
| 14 | `drate(play_ratio\|fans_user_num_range+music_id+tab\|std\|hl=2.477d\|a=1)` | drate | play_ratio | fans_user_num_range×music_id×tab | std | 2.477d | 1 |
| 15 | `recency(is_forward\|video_id\|all)` | recency | is_forward | video_id | all | — | — |
| 16 | `dsum(imp\|video_id\|all\|hl=23.63h)` | dsum | imp | video_id | all | 23.63h | — |
| 17 | `rate(is_click\|is_video_author+tag\|all\|w=1.441d\|a=1)` | rate | is_click | is_video_author×tag | all | 1.441d | 1 |
| 18 | `drate(play_ratio\|follow_user_num_range\|all\|hl=2.453d\|a=100)` | drate | play_ratio | follow_user_num_range | all | 2.453d | 100 |
| 19 | `sum(imp\|register_days_range+video_id\|rand\|w=2.813d)` | sum | imp | register_days_range×video_id | rand | 2.813d | — |
| 20 | `drate(long_view\|tag+is_rand\|all\|hl=4.035d\|a=10)` | drate | long_view | tag×is_rand | all | 4.035d | 10 |
| 21 | `drate(play_ratio\|is_lowactive_period+upload_type\|std\|hl=18.69m\|a=10)` | drate | play_ratio | is_lowactive_period×upload_type | std | 18.69m | 10 |
| 22 | `sum(is_profile_enter\|is_video_author\|all\|w=4.608d)` | sum | is_profile_enter | is_video_author | all | 4.608d | — |
| 23 | `drate(long_view\|video_type\|all\|hl=9.758h\|a=10)` | drate | long_view | video_type | all | 9.758h | 10 |
| 24 | `dsum(is_like\|register_days_range+video_id\|all\|hl=1.123h)` | dsum | is_like | register_days_range×video_id | all | 1.123h | — |
| 25 | `sum(imp\|follow_user_num_range+author_id\|all\|w=48.59m)` | sum | imp | follow_user_num_range×author_id | all | 48.59m | — |
| 26 | `dsum(is_hate\|dur_bucket\|rand\|hl=3.937h)` | dsum | is_hate | dur_bucket | rand | 3.937h | — |
| 27 | `sum(is_click\|video_id\|all\|w=18.44h)` | sum | is_click | video_id | all | 18.44h | — |
| 28 | `sum(imp\|fans_user_num_range+hour\|all\|w=9.697d)` | sum | imp | fans_user_num_range×hour | all | 9.697d | — |
| 29 | `sum(imp\|register_days_range+video_id+is_rand\|std\|w=2.813d)` | sum | imp | register_days_range×video_id×is_rand | std | 2.813d | — |
| 30 | `rate(play_s\|upload_type+hour\|std\|w=1.397d\|a=10)` | rate | play_s | upload_type×hour | std | 1.397d | 10 |
| 31 | `sum(is_click\|music_id\|all\|w=18.44h)` | sum | is_click | music_id | all | 18.44h | — |
| 32 | `sum(is_forward\|is_lowactive_period+music_type\|all\|w=14.62d)` | sum | is_forward | is_lowactive_period×music_type | all | 14.62d | — |
| 33 | `sum(is_click\|upload_type\|all\|w=18.44h)` | sum | is_click | upload_type | all | 18.44h | — |
| 34 | `dsum(is_click\|follow_user_num_range+music_id\|all\|hl=10.86h)` | dsum | is_click | follow_user_num_range×music_id | all | 10.86h | — |
| 35 | `drate(is_follow\|fans_user_num_range+music_id+tab\|std\|hl=3.318d\|a=1)` | drate | is_follow | fans_user_num_range×music_id×tab | std | 3.318d | 1 |
| 36 | `rate(play_s\|user_active_degree+upload_type\|rand\|w=15.89h\|a=100)` | rate | play_s | user_active_degree×upload_type | rand | 15.89h | 100 |
| 37 | `dsum(is_profile_enter\|music_id\|all\|hl=6.068d)` | dsum | is_profile_enter | music_id | all | 6.068d | — |
| 38 | `dsum(is_profile_enter\|register_days_range\|all\|hl=5.117d)` | dsum | is_profile_enter | register_days_range | all | 5.117d | — |

### LLM-агент, запуск 1

*16 признаков; агрегаты {'rate': 10, 'drate': 2, 'recency': 3, 'sum': 1}; фильтры {'rand': 3, 'std': 2, 'all': 9, 'clicked': 2}; с item-side колонкой в ключе: 7; только user/context ключ: 9; «круглых» окон (1ч/1д/7д/∞): 10 из 13*

| # | признак | агрегат | сигнал | ключ | фильтр | окно / полупериод | α |
|---|---|---|---|---|---|---|---|
| 1 | `rate(is_click\|user_id\|rand\|w=inf\|a=10)` | rate | is_click | user_id | rand | inf | 10 |
| 2 | `rate(is_click\|video_id\|rand\|w=inf\|a=20)` | rate | is_click | video_id | rand | inf | 20 |
| 3 | `rate(is_click\|user_id\|std\|w=inf\|a=10)` | rate | is_click | user_id | std | inf | 10 |
| 4 | `drate(is_click\|user_id\|all\|hl=3h\|a=5)` | drate | is_click | user_id | all | 3h | 5 |
| 5 | `rate(is_click\|music_id\|all\|w=inf\|a=30)` | rate | is_click | music_id | all | inf | 30 |
| 6 | `rate(is_click\|user_id+tab\|all\|w=inf\|a=10)` | rate | is_click | user_id×tab | all | inf | 10 |
| 7 | `recency(imp\|user_id+tab\|all)` | recency | imp | user_id×tab | all | — | — |
| 8 | `rate(is_click\|author_id+tab\|all\|w=inf\|a=30)` | rate | is_click | author_id×tab | all | inf | 30 |
| 9 | `drate(is_click\|user_id\|rand\|hl=2d\|a=5)` | drate | is_click | user_id | rand | 2d | 5 |
| 10 | `recency(is_click\|user_id+tag\|all)` | recency | is_click | user_id×tag | all | — | — |
| 11 | `rate(is_hate\|user_id\|std\|w=inf\|a=50)` | rate | is_hate | user_id | std | inf | 50 |
| 12 | `sum(imp\|user_id+tab\|all\|w=15m)` | sum | imp | user_id×tab | all | 15m | — |
| 13 | `recency(long_view\|user_id\|all)` | recency | long_view | user_id | all | — | — |
| 14 | `rate(is_click\|user_id+tag\|all\|w=inf\|a=50)` | rate | is_click | user_id×tag | all | inf | 50 |
| 15 | `rate(is_follow\|author_id\|clicked\|w=inf\|a=100)` | rate | is_follow | author_id | clicked | inf | 100 |
| 16 | `rate(is_like\|video_id\|clicked\|w=inf\|a=50)` | rate | is_like | video_id | clicked | inf | 50 |

### LLM-агент, запуск 2

*19 признаков; агрегаты {'rate': 10, 'drate': 4, 'recency': 3, 'sum': 2}; фильтры {'all': 13, 'rand': 5, 'std': 1}; с item-side колонкой в ключе: 9; только user/context ключ: 10; «круглых» окон (1ч/1д/7д/∞): 12 из 16*

| # | признак | агрегат | сигнал | ключ | фильтр | окно / полупериод | α |
|---|---|---|---|---|---|---|---|
| 1 | `rate(is_click\|user_id+video_type\|all\|w=inf\|a=5)` | rate | is_click | user_id×video_type | all | inf | 5 |
| 2 | `rate(is_click\|author_id\|rand\|w=inf\|a=30)` | rate | is_click | author_id | rand | inf | 30 |
| 3 | `drate(is_click\|user_id+tab\|all\|hl=3h\|a=3)` | drate | is_click | user_id×tab | all | 3h | 3 |
| 4 | `rate(is_click\|video_id\|all\|w=inf\|a=20)` | rate | is_click | video_id | all | inf | 20 |
| 5 | `rate(is_click\|user_id+tab\|all\|w=inf\|a=10)` | rate | is_click | user_id×tab | all | inf | 10 |
| 6 | `rate(is_click\|user_id\|rand\|w=inf\|a=3)` | rate | is_click | user_id | rand | inf | 3 |
| 7 | `rate(is_click\|author_id+tab\|all\|w=inf\|a=30)` | rate | is_click | author_id×tab | all | inf | 30 |
| 8 | `recency(imp\|user_id+tab\|all)` | recency | imp | user_id×tab | all | — | — |
| 9 | `drate(is_click\|user_id+tag\|all\|hl=3d\|a=5)` | drate | is_click | user_id×tag | all | 3d | 5 |
| 10 | `sum(imp\|user_id\|all\|w=inf)` | sum | imp | user_id | all | inf | — |
| 11 | `recency(long_view\|user_id\|all)` | recency | long_view | user_id | all | — | — |
| 12 | `rate(is_click\|music_id\|rand\|w=inf\|a=30)` | rate | is_click | music_id | rand | inf | 30 |
| 13 | `sum(imp\|user_id\|rand\|w=inf)` | sum | imp | user_id | rand | inf | — |
| 14 | `drate(is_click\|user_id+is_rand\|all\|hl=6h\|a=3)` | drate | is_click | user_id×is_rand | all | 6h | 3 |
| 15 | `recency(is_click\|user_id+tag\|all)` | recency | is_click | user_id×tag | all | — | — |
| 16 | `drate(is_click\|user_id\|std\|hl=12h\|a=5)` | drate | is_click | user_id | std | 12h | 5 |
| 17 | `rate(is_click\|video_id+is_rand\|all\|w=inf\|a=50)` | rate | is_click | video_id×is_rand | all | inf | 50 |
| 18 | `rate(play_s\|user_id\|rand\|w=inf\|a=3)` | rate | play_s | user_id | rand | inf | 3 |
| 19 | `rate(is_click\|author_id+is_rand\|all\|w=inf\|a=30)` | rate | is_click | author_id×is_rand | all | inf | 30 |

### LLM-агент + поиск (гибрид)

*30 признаков; агрегаты {'rate': 20, 'drate': 3, 'recency': 3, 'sum': 3, 'dsum': 1}; фильтры {'rand': 8, 'std': 3, 'all': 17, 'clicked': 2}; с item-side колонкой в ключе: 19; только user/context ключ: 11; «круглых» окон (1ч/1д/7д/∞): 19 из 27*

| # | признак | агрегат | сигнал | ключ | фильтр | окно / полупериод | α |
|---|---|---|---|---|---|---|---|
| 1 | `rate(is_click\|user_id\|rand\|w=inf\|a=10)` | rate | is_click | user_id | rand | inf | 10 |
| 2 | `rate(is_click\|video_id\|rand\|w=inf\|a=20)` | rate | is_click | video_id | rand | inf | 20 |
| 3 | `rate(is_click\|user_id\|std\|w=inf\|a=10)` | rate | is_click | user_id | std | inf | 10 |
| 4 | `drate(is_click\|user_id\|all\|hl=3h\|a=5)` | drate | is_click | user_id | all | 3h | 5 |
| 5 | `rate(is_click\|music_id\|all\|w=inf\|a=30)` | rate | is_click | music_id | all | inf | 30 |
| 6 | `rate(is_click\|user_id+tab\|all\|w=inf\|a=10)` | rate | is_click | user_id×tab | all | inf | 10 |
| 7 | `recency(imp\|user_id+tab\|all)` | recency | imp | user_id×tab | all | — | — |
| 8 | `rate(is_click\|author_id+tab\|all\|w=inf\|a=30)` | rate | is_click | author_id×tab | all | inf | 30 |
| 9 | `drate(is_click\|user_id\|rand\|hl=2d\|a=5)` | drate | is_click | user_id | rand | 2d | 5 |
| 10 | `recency(is_click\|user_id+tag\|all)` | recency | is_click | user_id×tag | all | — | — |
| 11 | `rate(is_hate\|user_id\|std\|w=inf\|a=50)` | rate | is_hate | user_id | std | inf | 50 |
| 12 | `sum(imp\|user_id+tab\|all\|w=15m)` | sum | imp | user_id×tab | all | 15m | — |
| 13 | `recency(long_view\|user_id\|all)` | recency | long_view | user_id | all | — | — |
| 14 | `rate(is_click\|user_id+tag\|all\|w=inf\|a=50)` | rate | is_click | user_id×tag | all | inf | 50 |
| 15 | `rate(is_follow\|author_id\|clicked\|w=inf\|a=100)` | rate | is_follow | author_id | clicked | inf | 100 |
| 16 | `rate(is_like\|video_id\|clicked\|w=inf\|a=50)` | rate | is_like | video_id | clicked | inf | 50 |
| 17 | `rate(is_click\|user_id+tab+tag\|all\|w=inf\|a=10)` | rate | is_click | user_id×tab×tag | all | inf | 10 |
| 18 | `rate(is_click\|music_id+is_rand\|all\|w=inf\|a=30)` | rate | is_click | music_id×is_rand | all | inf | 30 |
| 19 | `rate(is_click\|user_id\|rand\|w=inf\|a=1)` | rate | is_click | user_id | rand | inf | 1 |
| 20 | `rate(is_click\|music_id+tab\|all\|w=inf\|a=30)` | rate | is_click | music_id×tab | all | inf | 30 |
| 21 | `sum(is_click\|user_id\|rand\|w=inf)` | sum | is_click | user_id | rand | inf | — |
| 22 | `drate(long_view\|user_id+tag\|all\|hl=3d\|a=1)` | drate | long_view | user_id×tag | all | 3d | 1 |
| 23 | `rate(long_view\|is_lowactive_period+video_id+is_rand\|all\|w=inf\|a=10)` | rate | long_view | is_lowactive_period×video_id×is_rand | all | inf | 10 |
| 24 | `dsum(long_view\|tag\|all\|hl=9.664d)` | dsum | long_view | tag | all | 9.664d | — |
| 25 | `rate(is_click\|music_id+is_rand\|rand\|w=30d\|a=30)` | rate | is_click | music_id×is_rand | rand | 30d | 30 |
| 26 | `rate(is_like\|user_id+tag\|all\|w=inf\|a=1)` | rate | is_like | user_id×tag | all | inf | 1 |
| 27 | `rate(play_ratio\|user_id+tab+tag\|all\|w=inf\|a=10)` | rate | play_ratio | user_id×tab×tag | all | inf | 10 |
| 28 | `sum(is_click\|music_id+is_rand\|rand\|w=30d)` | sum | is_click | music_id×is_rand | rand | 30d | — |
| 29 | `rate(is_click\|music_id+tab+is_lowactive_period\|std\|w=inf\|a=3)` | rate | is_click | music_id×tab×is_lowactive_period | std | inf | 3 |
| 30 | `rate(is_click\|user_id+music_type\|rand\|w=30d\|a=10)` | rate | is_click | user_id×music_type | rand | 30d | 10 |

### AutoFE, окна из сетки, сид 0

*27 признаков; агрегаты {'rate': 18, 'recency': 3, 'sum': 6}; фильтры {'all': 12, 'rand': 10, 'std': 3, 'clicked': 2}; с item-side колонкой в ключе: 22; только user/context ключ: 5; «круглых» окон (1ч/1д/7д/∞): 24 из 24*

| # | признак | агрегат | сигнал | ключ | фильтр | окно / полупериод | α |
|---|---|---|---|---|---|---|---|
| 1 | `rate(is_click\|is_video_author+author_id+tab\|all\|w=7d\|a=1)` | rate | is_click | is_video_author×author_id×tab | all | 7d | 1 |
| 2 | `rate(is_click\|user_id+tag\|rand\|w=7d\|a=1)` | rate | is_click | user_id×tag | rand | 7d | 1 |
| 3 | `recency(play_s\|user_id+video_type\|rand)` | recency | play_s | user_id×video_type | rand | — | — |
| 4 | `sum(play_ratio\|user_id+music_type\|all\|w=7d)` | sum | play_ratio | user_id×music_type | all | 7d | — |
| 5 | `sum(play_ratio\|user_id+hour\|all\|w=1d)` | sum | play_ratio | user_id×hour | all | 1d | — |
| 6 | `sum(play_s\|user_id+dur_bucket\|rand\|w=inf)` | sum | play_s | user_id×dur_bucket | rand | inf | — |
| 7 | `rate(play_ratio\|register_days_range+music_id\|rand\|w=7d\|a=10)` | rate | play_ratio | register_days_range×music_id | rand | 7d | 10 |
| 8 | `rate(is_click\|video_id\|std\|w=7d\|a=1)` | rate | is_click | video_id | std | 7d | 1 |
| 9 | `rate(is_click\|user_id\|rand\|w=7d\|a=1)` | rate | is_click | user_id | rand | 7d | 1 |
| 10 | `rate(play_s\|user_id+music_type+is_rand\|all\|w=inf\|a=10)` | rate | play_s | user_id×music_type×is_rand | all | inf | 10 |
| 11 | `rate(is_click\|music_id\|all\|w=7d\|a=1)` | rate | is_click | music_id | all | 7d | 1 |
| 12 | `rate(play_s\|user_id+video_type\|rand\|w=1d\|a=1)` | rate | play_s | user_id×video_type | rand | 1d | 1 |
| 13 | `sum(is_click\|fans_user_num_range+music_id\|rand\|w=inf)` | sum | is_click | fans_user_num_range×music_id | rand | inf | — |
| 14 | `rate(play_ratio\|user_id+hour\|all\|w=1d\|a=10)` | rate | play_ratio | user_id×hour | all | 1d | 10 |
| 15 | `recency(play_s\|user_id+video_type+hour\|all)` | recency | play_s | user_id×video_type×hour | all | — | — |
| 16 | `rate(is_click\|is_video_author+author_id+tab\|all\|w=inf\|a=1)` | rate | is_click | is_video_author×author_id×tab | all | inf | 1 |
| 17 | `rate(long_view\|user_id+music_type\|all\|w=inf\|a=10)` | rate | long_view | user_id×music_type | all | inf | 10 |
| 18 | `rate(is_click\|is_lowactive_period+music_id+is_rand\|all\|w=inf\|a=10)` | rate | is_click | is_lowactive_period×music_id×is_rand | all | inf | 10 |
| 19 | `rate(is_comment\|follow_user_num_range+author_id\|clicked\|w=1h\|a=10)` | rate | is_comment | follow_user_num_range×author_id | clicked | 1h | 10 |
| 20 | `rate(is_click\|user_id\|std\|w=7d\|a=1)` | rate | is_click | user_id | std | 7d | 1 |
| 21 | `rate(is_profile_enter\|user_id+tag\|std\|w=inf\|a=1)` | rate | is_profile_enter | user_id×tag | std | inf | 1 |
| 22 | `recency(play_ratio\|follow_user_num_range+tag+hour\|all)` | recency | play_ratio | follow_user_num_range×tag×hour | all | — | — |
| 23 | `sum(is_click\|music_id\|rand\|w=7d)` | sum | is_click | music_id | rand | 7d | — |
| 24 | `rate(play_s\|user_id+video_type\|rand\|w=inf\|a=1)` | rate | play_s | user_id×video_type | rand | inf | 1 |
| 25 | `rate(is_click\|is_video_author+video_id\|all\|w=inf\|a=10)` | rate | is_click | is_video_author×video_id | all | inf | 10 |
| 26 | `sum(is_click\|user_id\|rand\|w=7d)` | sum | is_click | user_id | rand | 7d | — |
| 27 | `rate(is_comment\|video_id+tab\|clicked\|w=inf\|a=100)` | rate | is_comment | video_id×tab | clicked | inf | 100 |

## Приложение B. Как признаки выглядят в SQL (DuckDB, `Spec.to_sql()`)

```sql
-- rate(is_click|user_id+author_id|all|w=inf|a=10)
WITH p AS (SELECT *, (COALESCE(SUM(is_click) OVER (ORDER BY t RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) + 0.5) / (COALESCE(SUM(1) OVER (ORDER BY t RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) + 1) AS prior FROM ev) SELECT eid, (COALESCE(SUM(is_click) OVER (PARTITION BY user_id, author_id ORDER BY t RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) + 10.0 * prior) / (COALESCE(SUM(1) OVER (PARTITION BY user_id, author_id ORDER BY t RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) + 10.0) AS v FROM p;

-- drate(is_click|user_id+music_type+is_rand|rand|hl=13.36h|a=10)
WITH p AS (SELECT *, (COALESCE(SUM(CASE WHEN is_rand = 1 THEN is_click ELSE 0 END) OVER (ORDER BY t RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) + 0.5) / (COALESCE(SUM(CASE WHEN is_rand = 1 THEN 1 ELSE 0 END) OVER (ORDER BY t RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) + 1) AS prior FROM ev) SELECT q.eid AS eid, (COALESCE(SUM((CASE WHEN e.is_rand = 1 THEN e.is_click ELSE 0 END) * exp(-(q.t - e.t) / 1000.0 / 69397.07407030776)), 0) + 10.0 * q.prior) / (COALESCE(SUM(CASE WHEN e.eid IS NULL THEN 0 ELSE (CASE WHEN e.is_rand = 1 THEN 1 ELSE 0 END) END * exp(-(q.t - e.t) / 1000.0 / 69397.07407030776)), 0) + 10.0) AS v FROM p q LEFT JOIN ev e ON q.user_id = e.user_id AND q.music_type = e.music_type AND q.is_rand = e.is_rand AND e.t < q.t AND e.t >= q.t - 1924095449 GROUP BY q.eid, q.prior;

-- dsum(is_click|video_id|all|hl=1.469d)
WITH p AS (SELECT *, (COALESCE(SUM(is_click) OVER (ORDER BY t RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) + 0.5) / (COALESCE(SUM(1) OVER (ORDER BY t RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) + 1) AS prior FROM ev) SELECT q.eid AS eid, COALESCE(SUM((e.is_click) * exp(-(q.t - e.t) / 1000.0 / 183070.7914324938)), 0) AS v FROM p q LEFT JOIN ev e ON q.video_id = e.video_id AND e.t < q.t AND e.t >= q.t - 5075800116 GROUP BY q.eid, q.prior;

-- rate(is_click|user_id|rand|w=inf|a=10)
WITH p AS (SELECT *, (COALESCE(SUM(CASE WHEN is_rand = 1 THEN is_click ELSE 0 END) OVER (ORDER BY t RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) + 0.5) / (COALESCE(SUM(CASE WHEN is_rand = 1 THEN 1 ELSE 0 END) OVER (ORDER BY t RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) + 1) AS prior FROM ev) SELECT eid, (COALESCE(SUM(CASE WHEN is_rand = 1 THEN is_click ELSE 0 END) OVER (PARTITION BY user_id ORDER BY t RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) + 10.0 * prior) / (COALESCE(SUM(CASE WHEN is_rand = 1 THEN 1 ELSE 0 END) OVER (PARTITION BY user_id ORDER BY t RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) + 10.0) AS v FROM p;

```