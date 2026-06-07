# === Data ===
SEASONS = list(range(2012, 2026))  # load 2012 for prior-season/rolling CONTEXT only
POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
MIN_GAMES_PER_SEASON = 6
CACHE_DIR = "data/raw"
SPLITS_DIR = "data/splits"

# === Scoring ===
# Base scoring (shared across all formats)
_BASE_SCORING = {
    "passing_yards": 0.04,
    "passing_tds": 4,
    "interceptions": -2,
    "rushing_yards": 0.1,
    "rushing_tds": 6,
    "receiving_yards": 0.1,
    "receiving_tds": 6,
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
TRAIN_SEASONS = list(range(2013, 2024))
VAL_SEASONS = [2024]
TEST_SEASONS = [2025]

# === Cross-Validation (expanding window) ===
CV_VAL_SEASONS = [2021, 2022, 2023, 2024]

# === Rolling-origin (walk-forward) multi-season TEST evaluation ===
# Each test season T is scored by a model trained on [min_train .. T-2], val =
# T-1, test = T. The final origin (test=2025) is byte-identical to the
# production single split (TRAIN_SEASONS / VAL_SEASONS / TEST_SEASONS above), so
# a rolling-origin run reproduces the headline number and adds A/B origins for a
# mean±std. Distinct from the rejected K-fold-over-seasons (D1): every origin
# trains strictly on the past, preserving the deployment-mirror.
ROLLING_ORIGIN_TEST_SEASONS = [2023, 2024, 2025]

# === Features: Rolling ===
ROLLING_WINDOWS = [3, 5, 8]
ROLL_STATS = [
    "fantasy_points",
    "targets",
    "receptions",
    "carries",
    "rushing_yards",
    "receiving_yards",
    "passing_yards",
    "attempts",
    "snap_pct",
]
ROLL_AGGS = ["mean", "std", "max"]

# === Features: EWMA ===
EWMA_STATS = [
    "fantasy_points",
    "targets",
    "carries",
    "receiving_yards",
    "rushing_yards",
    "passing_yards",
    "snap_pct",
]
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
