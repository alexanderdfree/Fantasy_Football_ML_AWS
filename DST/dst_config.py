# === DST Target Decomposition ===
DST_TARGETS = ["defensive_scoring", "td_points", "pts_allowed_bonus"]

# === DST-Specific Features ===
# Rolling defense stats (17 features)
DST_SPECIFIC_FEATURES = [
    # Core production rolling windows
    "sacks_L3",
    "sacks_L5",              # Longer sack stability anchor
    "turnovers_L3",
    "ints_L3",               # INTs separated — secondary quality signal
    "fumble_rec_L3",         # Fumble recoveries — more stochastic component
    "pts_allowed_L3",
    "pts_allowed_L5",
    "dst_pts_L3",
    "dst_pts_L5",
    "dst_pts_L8",            # Longer stability anchor
    # EWMA features (faster adaptation to regime changes)
    "pts_allowed_ewma",      # Exponential-weighted points allowed
    "dst_pts_ewma",          # Exponential-weighted D/ST scoring
    # Momentum / trend indicators
    "sack_trend",
    "turnover_trend",        # Turnover production trajectory
    "pts_allowed_trend",     # Defensive improvement/decline
    # Consistency metrics
    "pts_allowed_std_L3",
    "dst_scoring_std_L3",    # Base scoring consistency
]

# Contextual / matchup features (13 features)
DST_CONTEXTUAL_FEATURES = [
    "is_home",
    "week",
    "spread_line",
    "total_line",
    "opp_scoring_L3",       # Short-window opponent quality (responsive)
    "opp_scoring_L5",
    "opp_turnovers_L5",     # Opponent turnover proneness
    "opp_sacks_allowed_L5", # Opponent OL sack vulnerability
    "rest_days",             # Days since last game (short rest = worse D)
    "div_game",              # Divisional games — different scoring patterns
    "is_dome",               # Dome games — weather-controlled
    "prior_season_dst_pts_avg",
    "prior_season_pts_allowed_avg",
]

DST_ALL_FEATURES = DST_SPECIFIC_FEATURES + DST_CONTEXTUAL_FEATURES

# No general features — D/ST bypasses the player-level feature pipeline
DST_DROP_FEATURES = set()

# === Ridge ===
import numpy as np
DST_RIDGE_ALPHA_GRIDS = {
    "defensive_scoring": [round(x, 4) for x in np.logspace(-1, 3.5, 20)],
    "td_points":         [round(x, 4) for x in np.logspace(0, 4, 15)],   # Higher floor — sparse target needs strong reg
    "pts_allowed_bonus": [round(x, 4) for x in np.logspace(-1, 3.5, 20)],
}

# === Neural Net (30 features, 2012-2025 dataset) ===
DST_NN_BACKBONE_LAYERS = [128, 64]   # Wider backbone for 30 features
DST_NN_HEAD_HIDDEN = 32
DST_NN_HEAD_HIDDEN_OVERRIDES = {
    "td_points": 16,             # ST-only target is simpler (mostly 0), smaller head
    "pts_allowed_bonus": 48,     # Wide range [-4, +10] with discrete tiers needs capacity
}
DST_NN_DROPOUT = 0.30             # Slightly higher — slows convergence, better generalization
# pts_allowed_bonus ranges from -4 to +10 — must NOT be clamped to >= 0
DST_NN_NON_NEGATIVE_TARGETS = {"defensive_scoring", "td_points"}
DST_NN_LR = 3e-4                  # Lower — more exploration before convergence
DST_NN_WEIGHT_DECAY = 3e-4
DST_NN_EPOCHS = 300
DST_NN_BATCH_SIZE = 128
DST_NN_PATIENCE = 35

# === Loss Weights ===
# With corrected targets (no def_tds/safeties noise):
# - defensive_scoring: most predictable, largest contribution to total
# - pts_allowed_bonus: highest variance, biggest swing factor
# - td_points: now ST-only, very sparse, don't over-weight
DST_LOSS_WEIGHTS = {
    "defensive_scoring": 1.0,
    "td_points": 1.5,           # Lowered — ST TDs are sparse, less signal to learn
    "pts_allowed_bonus": 2.0,   # Raised — this is where matchup features shine
}
DST_LOSS_W_TOTAL = 0.5          # Raised — total coherence matters more for fantasy

# === Huber Deltas (per-target) ===
DST_HUBER_DELTAS = {
    "defensive_scoring": 2.5,   # Slightly wider — reduce over-penalizing big sack games
    "td_points": 2.0,           # Tighter — ST TDs are mostly 0, focus on that
    "pts_allowed_bonus": 3.0,
}

# === LR Scheduler ===
DST_SCHEDULER_TYPE = "cosine_warm_restarts"
DST_COSINE_T0 = 30              # Longer first cycle for wider backbone
DST_COSINE_T_MULT = 2
DST_COSINE_ETA_MIN = 1e-5
