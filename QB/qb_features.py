import numpy as np
import pandas as pd
from src.features.engineer import get_feature_columns
from QB.qb_config import QB_DROP_FEATURES, QB_SPECIFIC_FEATURES


def get_qb_feature_columns() -> list[str]:
    """Return the complete ordered list of feature columns for the QB model."""
    general_cols = get_feature_columns()
    qb_cols = [c for c in general_cols if c not in QB_DROP_FEATURES]
    qb_cols.extend(QB_SPECIFIC_FEATURES)
    return qb_cols


def add_qb_specific_features(train_df, val_df, test_df):
    """Add 8 QB-specific engineered features to each split."""
    for df in [train_df, val_df, test_df]:
        _compute_qb_features(df)
    return train_df, val_df, test_df


def _compute_qb_features(df: pd.DataFrame) -> None:
    """Compute all 8 QB-specific features in-place."""
    df.sort_values(["player_id", "season", "week"], inplace=True)

    # Helper: shifted rolling L3
    def _roll_sum(col):
        return df.groupby(["player_id", "season"])[col].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).sum()
        )

    completions_roll = _roll_sum("completions")
    attempts_roll = _roll_sum("attempts")
    pass_yds_roll = _roll_sum("passing_yards")
    pass_tds_roll = _roll_sum("passing_tds")
    ints_roll = _roll_sum("interceptions")
    sacks_roll = _roll_sum("sacks")
    rush_yds_roll = _roll_sum("rushing_yards")
    pass_epa_roll = _roll_sum("passing_epa")
    air_yds_roll = _roll_sum("passing_air_yards")
    carries_roll = _roll_sum("carries")
    pass_first_downs_roll = _roll_sum("passing_first_downs")
    rush_first_downs_roll = _roll_sum("rushing_first_downs")
    rush_epa_roll = _roll_sum("rushing_epa")
    pass_yac_roll = _roll_sum("passing_yards_after_catch")
    sack_yds_roll = _roll_sum("sack_yards")

    dropbacks = attempts_roll + sacks_roll

    # 1. completion_pct_L3
    df["completion_pct_L3"] = (completions_roll / attempts_roll).fillna(0)
    df.loc[attempts_roll == 0, "completion_pct_L3"] = 0

    # 2. yards_per_attempt_L3
    df["yards_per_attempt_L3"] = (pass_yds_roll / attempts_roll).fillna(0)
    df.loc[attempts_roll == 0, "yards_per_attempt_L3"] = 0

    # 3. td_rate_L3
    df["td_rate_L3"] = (pass_tds_roll / attempts_roll).fillna(0)
    df.loc[attempts_roll == 0, "td_rate_L3"] = 0

    # 4. int_rate_L3
    df["int_rate_L3"] = (ints_roll / attempts_roll).fillna(0)
    df.loc[attempts_roll == 0, "int_rate_L3"] = 0

    # 5. sack_rate_L3
    df["sack_rate_L3"] = (sacks_roll / dropbacks).fillna(0)
    df.loc[dropbacks == 0, "sack_rate_L3"] = 0

    # 6. qb_rushing_share_L3 (dual-threat indicator)
    total_yds = pass_yds_roll + rush_yds_roll
    df["qb_rushing_share_L3"] = (rush_yds_roll / total_yds).fillna(0)
    df.loc[total_yds == 0, "qb_rushing_share_L3"] = 0

    # 7. passing_epa_per_dropback_L3
    df["passing_epa_per_dropback_L3"] = (pass_epa_roll / dropbacks).fillna(0)
    df.loc[dropbacks == 0, "passing_epa_per_dropback_L3"] = 0

    # 8. deep_ball_rate_L3 (air yards per attempt)
    df["deep_ball_rate_L3"] = (air_yds_roll / attempts_roll).fillna(0)
    df.loc[attempts_roll == 0, "deep_ball_rate_L3"] = 0

    # 9. pass_first_down_rate_L3 (first downs per attempt — drive-sustaining ability)
    df["pass_first_down_rate_L3"] = (pass_first_downs_roll / attempts_roll).fillna(0)
    df.loc[attempts_roll == 0, "pass_first_down_rate_L3"] = 0

    # 10. rushing_epa_per_carry_L3 (rushing quality beyond raw yards)
    df["rushing_epa_per_carry_L3"] = (rush_epa_roll / carries_roll).fillna(0)
    df.loc[carries_roll == 0, "rushing_epa_per_carry_L3"] = 0

    # 11. rush_first_down_rate_L3 (rushing first downs per carry)
    df["rush_first_down_rate_L3"] = (rush_first_downs_roll / carries_roll).fillna(0)
    df.loc[carries_roll == 0, "rush_first_down_rate_L3"] = 0

    # 12. yac_rate_L3 (YAC / passing yards — scheme & receiver quality)
    df["yac_rate_L3"] = (pass_yac_roll / pass_yds_roll).fillna(0)
    df.loc[pass_yds_roll == 0, "yac_rate_L3"] = 0

    # 13. sack_damage_per_dropback_L3 (sack yards lost per dropback — OL quality)
    df["sack_damage_per_dropback_L3"] = (sack_yds_roll / dropbacks).fillna(0)
    df.loc[dropbacks == 0, "sack_damage_per_dropback_L3"] = 0


def fill_qb_nans(train_df, val_df, test_df, qb_feature_cols):
    """Fill NaNs in QB-specific feature columns using training set statistics."""
    for split_df in [train_df, val_df, test_df]:
        split_df[qb_feature_cols] = split_df[qb_feature_cols].replace([np.inf, -np.inf], np.nan)
    train_means = train_df[qb_feature_cols].mean()
    for split_df in [train_df, val_df, test_df]:
        for col in qb_feature_cols:
            split_df[col] = split_df[col].fillna(train_means[col])
    return train_df, val_df, test_df
