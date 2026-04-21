import numpy as np
import pandas as pd


def _build_workframe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": df["player_id"].to_numpy(),
            "season": df["season"].to_numpy(),
            "week": df["week"].to_numpy(),
            "fantasy_points": df["fantasy_points"].to_numpy(),
            "_pos": np.arange(len(df)),
        }
    ).sort_values(["player_id", "season", "week"], kind="stable")


def _scatter_back(preds_sorted: pd.Series, positions: np.ndarray) -> np.ndarray:
    out = np.empty(positions.shape[0], dtype=np.float64)
    out[positions] = preds_sorted.to_numpy()
    return out


class SeasonAverageBaseline:
    """Predict each player's expanding season-to-date average fantasy points.

    Sort is handled internally; predictions are returned in the caller's row order.
    """

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        work = _build_workframe(df)
        preds = work.groupby(["player_id", "season"])["fantasy_points"].transform(
            lambda x: x.shift(1).expanding().mean()
        ).fillna(0)
        return _scatter_back(preds, work["_pos"].to_numpy())


class LastWeekBaseline:
    """Predict each player scored the same as last week.

    Sort is handled internally; predictions are returned in the caller's row order.
    """

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        work = _build_workframe(df)
        grouped = work.groupby(["player_id", "season"])["fantasy_points"]
        shifted = grouped.shift(1)
        season_avg = grouped.transform(lambda x: x.shift(1).expanding().mean())
        preds = shifted.fillna(season_avg).fillna(0)
        return _scatter_back(preds, work["_pos"].to_numpy())
