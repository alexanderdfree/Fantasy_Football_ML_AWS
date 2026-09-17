"""Read-only checkpoint policy comparison on a single training trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite


@dataclass
class SelectionTrace:
    """Keep the actual legacy stop separate from a longer diagnostic run."""

    patience: int
    rows: list[dict] = field(default_factory=list)
    best: dict[str, dict] = field(default_factory=dict)
    stops: dict[str, int] = field(default_factory=dict)
    stale: dict[str, int] = field(default_factory=lambda: {"legacy": 0, "rmse": 0})

    def observe(self, epoch, weighted_mae, mae, rmse):
        if epoch != len(self.rows) + 1 or self.patience < 1:
            raise ValueError("Checkpoint epochs must be contiguous; patience must be positive")
        values = (weighted_mae, mae, rmse)
        if not all(isfinite(float(v)) and v >= 0 for v in values):
            raise ValueError("Nonfinite or negative checkpoint metrics")
        row = dict(epoch=epoch, weighted_mae=float(weighted_mae), mae=float(mae), rmse=float(rmse))
        self.rows.append(row)
        improved = []
        for policy, metric in (("legacy", "weighted_mae"), ("rmse", "rmse")):
            if policy in self.stops:
                continue
            if policy not in self.best or row[metric] < self.best[policy][metric]:
                self.best[policy] = row
                self.stale[policy] = 0
                improved.append(policy)
            else:
                self.stale[policy] += 1
                if self.stale[policy] >= self.patience:
                    self.stops[policy] = epoch
        return improved

    def finish(self):
        if not self.rows:
            raise ValueError("No checkpoint observations")
        anchor = self.best["legacy"]
        # The guarded policy is the minimum over the declared diagnostic
        # budget. Its feasible set must not inherit another policy's stop.
        # Only the legacy anchor is frozen at its original stopping point.
        eligible = [r for r in self.rows if r["mae"] <= anchor["mae"]]
        guarded = (
            min(eligible, key=lambda r: (r["rmse"], r["mae"], r["epoch"])) if eligible else None
        )
        qualifies = guarded is not None and guarded["rmse"] < anchor["rmse"]
        return {
            "legacy": anchor,
            "rmse": self.best["rmse"],
            "guarded": guarded if qualifies else None,
            "guarded_qualifies": qualifies,
            "guarded_search_epochs": len(self.rows),
            "stop_epochs": {p: self.stops.get(p, len(self.rows)) for p in ("legacy", "rmse")},
            "stop_reasons": {
                p: "patience" if p in self.stops else "budget" for p in ("legacy", "rmse")
            },
            "trajectory": self.rows,
        }
