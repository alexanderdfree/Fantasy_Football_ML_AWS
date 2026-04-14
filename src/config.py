# === Data ===
SEASONS = list(range(2018, 2026))  # 2018-2025
POSITIONS = ["QB", "RB", "WR", "TE"]
MIN_GAMES_PER_SEASON = 6
CACHE_DIR = "data/raw"
SPLITS_DIR = "data/splits"

# === Scoring ===
# Base scoring (shared across all formats)
_BASE_SCORING = {
    "passing_yards": 0.04, "passing_tds": 4, "interceptions": -2,
    "rushing_yards": 0.1, "rushing_tds": 6,
    "receiving_yards": 0.1, "receiving_tds": 6,
    "fumbles_lost": -2,
}

# Reception weights per format
PPR_FORMATS = {
    "standard": 0.0,
    "half_ppr": 0.5,
    "ppr": 1.0,
}

# Full scoring dicts per format
SCORING_STANDARD = {**_BASE_SCORING, "receptions": 0.0}
SCORING_HALF_PPR = {**_BASE_SCORING, "receptions": 0.5}
SCORING_PPR = {**_BASE_SCORING, "receptions": 1.0}

# Default (full PPR) — backwards compatible
SCORING = SCORING_PPR

# === Split ===
TRAIN_SEASONS = list(range(2018, 2024))
VAL_SEASONS = [2024]
TEST_SEASONS = [2025]

# === Features: Rolling ===
ROLLING_WINDOWS = [3, 5, 8]
ROLL_STATS = [
    "fantasy_points", "fantasy_points_floor", "targets", "receptions",
    "carries", "rushing_yards", "receiving_yards", "passing_yards",
    "attempts", "snap_pct",
]
ROLL_AGGS = ["mean", "std", "max"]

# === Features: EWMA ===
EWMA_STATS = ["fantasy_points", "targets", "carries", "receiving_yards",
              "rushing_yards", "passing_yards", "snap_pct"]
EWMA_SPANS = [3, 5]

# === Features: Trend/Momentum ===
TREND_STATS = ["fantasy_points", "targets", "carries", "snap_pct"]

# === Features: Share ===
SHARE_WINDOWS = [3, 5]

# === Features: Opponent/Matchup ===
OPP_ROLLING_WINDOW = 5

# === Ridge ===
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

# === Neural Net ===
NN_HIDDEN_LAYERS = [128, 64, 32]
NN_DROPOUT = 0.3
NN_LR = 1e-3
NN_WEIGHT_DECAY = 1e-4
NN_EPOCHS = 200
NN_BATCH_SIZE = 256
NN_PATIENCE = 15

# === LR Scheduler ===
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5

# === Backtest ===
TOP_K_RANKING = 12

# === Paths ===
FIGURES_DIR = "outputs/figures"
MODELS_DIR = "outputs/models"
