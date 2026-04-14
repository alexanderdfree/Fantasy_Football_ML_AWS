import numpy as np
import pandas as pd


class SeasonAverageBaseline:
    """Predict each player's expanding season-to-date average fantasy points."""

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = df.sort_values(["player_id", "season", "week"])
        preds = df.groupby(["player_id", "season"])["fantasy_points"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        # First appearance: use 0
        preds = preds.fillna(0)
        return preds.values


class LastWeekBaseline:
    """Predict each player scored the same as last week."""

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = df.sort_values(["player_id", "season", "week"])
        preds = df.groupby(["player_id", "season"])["fantasy_points"].shift(1)
        # First appearance: use season expanding mean or 0
        season_avg = df.groupby(["player_id", "season"])["fantasy_points"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        preds = preds.fillna(season_avg).fillna(0)
        return preds.values
