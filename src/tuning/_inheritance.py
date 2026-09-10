"""Pre-kickoff frame injection shared by inheritance experiments."""

import numpy as np


def inject_inheritance(train, val, test, *, positions, role_columns):
    """Add ``is_top_available`` + ``inherited_opportunity`` per player-week, within position.

    * role(player, W) = mean of the position's opportunity proxy (``role_columns``: RB snap-share,
      WR per-game targets) over that player's weeks < W (in-season) — prior-to-W, no leak.
    * ``is_top_available`` = this player has the top prior-role among *present* same-position
      teammates that week.
    * ``inherited_opportunity`` = Σ prior-role of same-team, same-position OUT/Doubtful
      players ranked above, but only for the top-available one (the next-man-up who absorbs
      the role).

    Computed WITHIN each position in ``positions`` (the splits are all-position; a
    cross-position group makes a back never "top" — a QB at snap~1.0 outranks every RB).
    The injector has no view of the cell's position, so it computes the column for every
    position in ``positions`` and each cell reads only its own rows. Same column serves
    both branches: its week-W value is the static feature; its per-game sequence is the
    history token.
    """
    from src.data import nfl_source

    seasons = sorted({int(s) for df in (train, val, test) for s in df["season"].unique()})
    inj = nfl_source.injuries(seasons)
    out = inj[(inj["report_status"].isin(["Out", "Doubtful"])) & (inj["position"].isin(positions))]
    outmap: dict = {}  # (position, season, team, week) -> {out player ids}
    for pos, s, t, w, g in zip(
        out["position"],
        out["season"].astype(int),
        out["team"],
        out["week"].astype(int),
        out["gsis_id"].astype(str),
        strict=True,  # all five are columns of `out` → equal-length by construction
    ):
        outmap.setdefault((pos, s, t, w), set()).add(g)

    def _add(df):
        df["player_id"] = df["player_id"].astype(str)
        # Per-position prior-to-W expanding-mean role table, each from its own opportunity
        # proxy (role_columns). Keyed by position so an OUT player's role reads the right column.
        pref: dict = {}  # position -> {(player, season): (weeks_sorted, cumulative-mean)}
        for pos in positions:
            col = role_columns[pos]
            table: dict = {}
            sub_pos = df[df["position"] == pos].sort_values("week")
            for (p, s), sub in sub_pos.groupby(["player_id", "season"]):
                wks = sub["week"].to_numpy()
                vals = np.nan_to_num(sub[col].to_numpy(float), nan=0.0)
                table[(p, s)] = (wks, np.cumsum(vals) / np.arange(1, len(vals) + 1))
            pref[pos] = table

        def role_before(pos, p, s, w):
            e = pref[pos].get((p, s))
            if e is None:
                return 0.0
            wks, cm = e
            i = int(np.searchsorted(wks, w, side="left")) - 1  # largest week < w
            return float(cm[i]) if i >= 0 else 0.0

        is_top = np.zeros(len(df))
        inh = np.zeros(len(df))
        for pos in positions:
            grp = df[df["position"] == pos]
            for (s, tm, w), idx in grp.groupby(["season", "recent_team", "week"]).groups.items():
                si, wi = int(s), int(w)
                pids = df.loc[idx, "player_id"].to_numpy()
                roles = np.array([role_before(pos, p, si, wi) for p in pids])
                out_set = outmap.get((pos, si, tm, wi), set())
                out_roles = np.array([role_before(pos, g, si, wi) for g in out_set])
                for j, rp in enumerate(roles):
                    top = 1.0 if (roles > rp).sum() == 0 else 0.0
                    oa = float(out_roles[out_roles > rp].sum()) if out_roles.size else 0.0
                    pi = df.index.get_loc(idx[j])
                    is_top[pi] = top
                    inh[pi] = top * oa
        df["is_top_available"] = is_top
        df["inherited_opportunity"] = inh
        return df

    return _add(train), _add(val), _add(test)
