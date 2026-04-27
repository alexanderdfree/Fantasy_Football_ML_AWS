# data/

Project data files. NFL stats are sourced at runtime from [nflverse](https://github.com/nflverse) via [src/data/loader.py](../src/data/loader.py) and cached locally under `data/raw/` and `data/splits/` (both gitignored — see [.gitignore](../.gitignore)). First-time data pull: see [SETUP.md](../SETUP.md).

## Data access entry points

- [src/data/loader.py](../src/data/loader.py) — nflverse loader + caching (player stats, rosters, schedules, snap counts)
- [src/data/split.py](../src/data/split.py) — temporal split (train: 2012–2023, val: 2024, test: 2025)
- [src/data/preprocessing.py](../src/data/preprocessing.py) — null handling, schema normalization
- [src/shared/weather_features.py](../src/shared/weather_features.py) — Vegas odds + weather joins (game-week granularity, joined to player frames during feature engineering)

## Data sources

- **nflverse** — player stats, rosters, schedules, snap counts. 2012–2025 seasons. MIT-licensed.
- **Vegas implied team totals + weather snapshots** — see [docs/design_weather_and_odds.md](../docs/design_weather_and_odds.md) for sourcing and join semantics.
