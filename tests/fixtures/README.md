# Test Fixtures

This directory holds captured fixture files used by the loader/schema contract
tests. Fixtures are committed to Git so tests are deterministic and require no
network access.

## `weekly_2023_w1.parquet`

**What it represents.** A small slice of the `src.data.loader.load_raw_data`
output for the 2023 regular season, Week 1. The frame matches the schema of the
full loader output (all 57 columns, including the joined `position`,
`snap_pct`, `practice_status`, `game_status`, `depth_chart_rank` from rosters/
snap counts/injuries/depth charts).

- **Shape**: 32 rows x 57 columns
- **Positions**: WR=10, QB=8, RB=8, TE=6 (deterministic sample, `seed=42`)
- **Size**: ~38 KB (well under the 100 KB cap; Git LFS not required)
- **Captured on**: 2026-04-16 from the local cache parquet at
  `data/raw/weekly_2012_2025.parquet` (which itself was sourced from
  `nfl_data_py.import_weekly_data([2023])` + nflverse roster/snap/injury/depth
  joins via `src.data.loader.load_raw_data`).

## How to regenerate

From the project root, with the cache parquets present in `data/raw/`:

```python
import pandas as pd
from src.data.loader import load_raw_data

df = load_raw_data([2023])

w1 = df[(df["season"] == 2023) & (df["week"] == 1)].copy()
w1.attrs = {}  # clear non-serialisable schedules attr set by the loader

positions_to_sample = {"QB": 8, "RB": 8, "WR": 10, "TE": 6}
parts = []
for pos, n in positions_to_sample.items():
    sub = w1[w1["position"] == pos].sort_values("fantasy_points", ascending=False).head(n * 2)
    sub = sub.sample(n=min(n, len(sub)), random_state=42)
    sub.attrs = {}
    parts.append(sub)

fixture = pd.concat(parts, ignore_index=True)
fixture = fixture.sort_values(["position", "player_id"]).reset_index(drop=True)
fixture.to_parquet("tests/fixtures/weekly_2023_w1.parquet", compression="snappy")
```

## Size cap policy

Fixtures must be **under 100 KB** each. If a fixture grows beyond that, move
it to Git LFS rather than committing directly to the repo.
