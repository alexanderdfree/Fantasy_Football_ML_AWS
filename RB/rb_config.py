# === RB Target Decomposition ===
RB_TARGETS = ["rushing_floor", "receiving_floor", "td_points"]

# === RB-Specific Features ===
RB_SPECIFIC_FEATURES = [
    "yards_per_carry_L3",
    "reception_rate_L3",
    "weighted_opportunities_L3",
    "team_rb_carry_share_L3",
    "team_rb_target_share_L3",
    "rushing_epa_per_attempt_L3",
    "first_down_rate_L3",
    "yac_per_reception_L3",
]

# Features to drop from the general pipeline for RB model
RB_DROP_FEATURES = set()
for _stat in ["passing_yards", "attempts"]:
    for _window in [3, 5, 8]:
        for _agg in ["mean", "std", "max"]:
            RB_DROP_FEATURES.add(f"rolling_{_agg}_{_stat}_L{_window}")
for _span in [3, 5]:
    RB_DROP_FEATURES.add(f"ewma_passing_yards_L{_span}")
for _stat in ["passing_yards", "attempts"]:
    for _agg in ["mean", "std", "max"]:
        RB_DROP_FEATURES.add(f"prior_season_{_agg}_{_stat}")
RB_DROP_FEATURES |= {"pos_QB", "pos_RB", "pos_WR", "pos_TE"}

# Drop current-week features that cause data leakage (not available at prediction time)
RB_DROP_FEATURES |= {"snap_pct", "air_yards_share"}

# === Ridge ===
RB_RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

# === Neural Net ===
RB_NN_BACKBONE_LAYERS = [128, 64]
RB_NN_HEAD_HIDDEN = 32
RB_NN_DROPOUT = 0.3
RB_NN_LR = 1e-3
RB_NN_WEIGHT_DECAY = 1e-4
RB_NN_EPOCHS = 200
RB_NN_BATCH_SIZE = 256
RB_NN_PATIENCE = 15

# === Loss Weights ===
# TD weight is higher because td_points is discrete/zero-inflated (0, 6, 12, ...)
# and needs more optimization pressure. Rushing/receiving floors are continuous
# with higher variance and naturally dominate the loss signal at equal weights.
RB_LOSS_W_RUSHING = 1.0
RB_LOSS_W_RECEIVING = 1.0
RB_LOSS_W_TD = 2.0
RB_LOSS_W_TOTAL = 0.5

# === LR Scheduler ===
RB_SCHEDULER_PATIENCE = 5
RB_SCHEDULER_FACTOR = 0.5
