# Design Document 2: LSTM + Multi-Head Architecture for Sequential Understanding

> **Status: SUPERSEDED by Attention-based model.** Instead of an LSTM, a `MultiHeadNetWithHistory`
> architecture was implemented in `shared/neural_net.py` using learned-query `AttentionPool` over
> padded game history sequences (up to 17 games). This achieves the same goal of learning from
> raw game-by-game sequences while being simpler to train. The attention model is trained for
> QB, RB, WR, and TE (configurable via `{POS}_TRAIN_ATTENTION_NN` in position configs).
> A `GatedTDHead` (sigmoid gate × Softplus value) handles zero-inflated TD prediction.
>
> Target names below (`rushing_floor`, `receiving_floor`, `td_points`) reflect the
> pre-migration fantasy-point-component decomposition. The current system predicts raw
> NFL stats (yards, TD counts, receptions) — see [ARCHITECTURE.md](ARCHITECTURE.md) Decision D2.

## Motivation

Your current models compress a player's recent history into summary statistics (rolling mean/std/max over L3/L5/L8 windows). This discards temporal ordering — a player who scored [5, 10, 15, 20] has the same L4 mean (12.5) as one who scored [20, 15, 10, 5], but the first is trending up and the second is declining. While your `trend_*` features (L3 - L8) partially capture this, they're a single scalar summary of trajectory.

An LSTM/GRU processes the raw game-by-game sequence and learns which temporal patterns matter. Kristinsson (2022) showed that GRU layers on last-5-games data significantly outperformed feedforward networks on match outcome prediction. This approach is complementary to your existing models — it's a third model for the ensemble, not a replacement.

## Architecture Overview

```
                 Per-Game Feature Vectors (last K games)
                 ┌─────┬─────┬─────┬─────┬─────┐
                 │ G-5 │ G-4 │ G-3 │ G-2 │ G-1 │    shape: (batch, K, F_seq)
                 └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘
                    │     │     │     │     │
                    ▼     ▼     ▼     ▼     ▼
                 ┌───────────────────────────────┐
                 │       Bidirectional LSTM       │    hidden_size=64, num_layers=2
                 │         (or GRU)               │    dropout=0.3 between layers
                 └──────────────┬────────────────┘
                                │
                        last hidden state           shape: (batch, 2*hidden_size)
                                │
                    ┌───────────┴───────────┐
                    │                       │
              Sequence Repr            Static Features     shape: (batch, F_static)
              (128-dim)                (vegas, venue, etc.)
                    │                       │
                    └───────────┬───────────┘
                                │
                          Concatenate                     shape: (batch, 128 + F_static)
                                │
                    ┌───────────┴───────────┐
                    │     Shared Backbone    │             Linear(concat_dim, 64) → BN → ReLU → Dropout
                    └───────────┬───────────┘
                    ┌───────┬───┴───┬───────┐
                    │       │       │       │
                 Head_1  Head_2  Head_3  (optional: boom/bust heads)
                    │       │       │
                Softplus Softplus Softplus
                    │       │       │
               rushing  receiving  td_points
                floor    floor
```

## Sequence Features (per game)

Each timestep in the LSTM input is a vector of per-game stats for one historical game. These are raw, unrolled stats — the LSTM learns its own temporal patterns.

### Per-Game Feature Vector (F_seq = ~20 features per timestep)

```python
SEQ_FEATURES = [
    # Volume
    "fantasy_points",
    "fantasy_points_floor",
    "targets",
    "receptions",
    "carries",
    "snap_pct",

    # Yardage
    "rushing_yards",
    "receiving_yards",
    "passing_yards",

    # Efficiency
    "rushing_epa",
    "receiving_epa",
    "passing_epa",

    # TDs & turnovers
    "rushing_tds",
    "receiving_tds",
    "passing_tds",
    "interceptions",
    "fumbles_lost",       # sum of sack/rush/recv fumbles_lost

    # Context (per game)
    "is_home",
    "opp_fantasy_pts_allowed_to_pos",   # matchup quality that game
]
```

Position-specific models can extend this list (e.g., QB adds `completions`, `attempts`; RB adds `receiving_yards_after_catch`).

### Static Features (F_static)

These are the non-sequential features that don't vary game-to-game in a meaningful sequence:
- All 12 new vegas/venue features from Design Doc 1
- Prior-season summary features (24)
- Position-specific features that are already L3 aggregations (8)
- `week` (game week)
- `is_returning_from_absence`

These get concatenated with the LSTM output before the shared backbone.

## Sequence Length (K)

**K = 5 games** (matching Kristinsson's finding and your L5 window).

Rationale:
- 3 is too few to capture multi-game trends
- 8 adds complexity and many players don't have 8 prior games in a season (early weeks)
- 5 balances information with data availability

For players with fewer than 5 prior games in the current context (e.g., week 2), pad with zeros from the left and use `pack_padded_sequence` to avoid the LSTM processing padding tokens.

## New File: `shared/seq_neural_net.py`

```python
class SequenceMultiHeadNet(nn.Module):
    def __init__(
        self,
        seq_input_dim: int,         # F_seq (per-timestep features)
        static_input_dim: int,      # F_static (non-sequential features)
        target_names: list[str],
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        bidirectional: bool = True,
        backbone_layers: list[int] = [64],
        head_hidden: int = 32,
        dropout: float = 0.3,
        head_hidden_overrides: dict = None,
    ):
        super().__init__()

        # Sequence encoder
        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # Shared backbone (takes concatenated LSTM output + static features)
        concat_dim = lstm_out_dim + static_input_dim
        backbone = []
        prev_dim = concat_dim
        for layer_dim in backbone_layers:
            backbone.extend([
                nn.Linear(prev_dim, layer_dim),
                nn.BatchNorm1d(layer_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = layer_dim
        self.backbone = nn.Sequential(*backbone)

        # Per-target heads (same as MultiHeadNet)
        self.heads = nn.ModuleDict()
        for name in target_names:
            h = (head_hidden_overrides or {}).get(name, head_hidden)
            self.heads[name] = nn.Sequential(
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Linear(h, 1),
            )
        self.target_names = target_names

    def forward(self, x_seq, x_static, lengths=None):
        """
        x_seq:    (batch, K, F_seq) — padded game sequences
        x_static: (batch, F_static) — non-sequential features
        lengths:  (batch,) — actual sequence lengths for packing
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x_seq, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, _) = self.lstm(packed)
        else:
            lstm_out, (h_n, _) = self.lstm(x_seq)

        # Use last hidden state from both directions
        if self.lstm.bidirectional:
            seq_repr = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, 2*hidden)
        else:
            seq_repr = h_n[-1]  # (batch, hidden)

        # Concatenate with static features
        combined = torch.cat([seq_repr, x_static], dim=1)
        shared = self.backbone(combined)

        # Per-target predictions
        preds = {}
        total = torch.zeros(shared.size(0), 1, device=shared.device)
        for name in self.target_names:
            p = F.softplus(self.heads[name](shared))
            preds[name] = p.squeeze(1)
            total += p
        preds["total"] = total.squeeze(1)
        return preds
```

## New File: `shared/seq_training.py`

Extends the existing training infrastructure to handle dual inputs (sequence + static).

### SequenceDataset

```python
class SequenceDataset(Dataset):
    """
    Stores pre-built (x_seq, x_static, y_dict, length) tuples.
    """
    def __init__(self, x_seq, x_static, targets_dict, lengths):
        self.x_seq = torch.tensor(x_seq, dtype=torch.float32)       # (N, K, F_seq)
        self.x_static = torch.tensor(x_static, dtype=torch.float32) # (N, F_static)
        self.targets = {k: torch.tensor(v, dtype=torch.float32) for k, v in targets_dict.items()}
        self.lengths = torch.tensor(lengths, dtype=torch.long)       # (N,)

    def __len__(self):
        return len(self.x_seq)

    def __getitem__(self, idx):
        y = {k: v[idx] for k, v in self.targets.items()}
        return self.x_seq[idx], self.x_static[idx], y, self.lengths[idx]
```

### Modified Trainer

The existing `MultiHeadTrainer` calls `model(X_batch)`. The sequence trainer calls `model(x_seq_batch, x_static_batch, lengths_batch)`. The loss computation is identical (reuse `MultiTargetLoss` as-is). The changes are:
- Dataloader unpacks 4-tuples instead of 2-tuples
- Forward pass takes three arguments instead of one
- Everything else (early stopping, LR scheduling, gradient clipping) stays the same

## Data Preparation: Building Sequences

This is the most significant new code. You need to convert the flat player-week DataFrame into (sequence, static, target) tuples.

### New Function: `build_sequences()` in `shared/seq_data.py`

```python
def build_sequences(
    df: pd.DataFrame,
    seq_feature_cols: list[str],
    static_feature_cols: list[str],
    target_cols: list[str],
    seq_len: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    For each row in df (a player-week), look back at the player's
    previous `seq_len` games and build the sequence.

    Returns:
        x_seq:    (N, seq_len, len(seq_feature_cols))
        x_static: (N, len(static_feature_cols))
        targets:  {target_name: (N,)}
        lengths:  (N,) actual sequence lengths (for packing)
    """
    df = df.sort_values(["player_id", "season", "week"])

    x_seq_list, x_static_list, lengths_list = [], [], []
    target_lists = {col: [] for col in target_cols}

    for player_id, player_df in df.groupby("player_id"):
        for i in range(len(player_df)):
            row = player_df.iloc[i]

            # Look back at previous games (not including current)
            start = max(0, i - seq_len)
            history = player_df.iloc[start:i]

            # Build sequence (left-pad with zeros if < seq_len games)
            actual_len = len(history)
            seq = np.zeros((seq_len, len(seq_feature_cols)))
            if actual_len > 0:
                seq[-actual_len:] = history[seq_feature_cols].values

            x_seq_list.append(seq)
            x_static_list.append(row[static_feature_cols].values)
            lengths_list.append(min(actual_len, seq_len))

            for col in target_cols:
                target_lists[col].append(row[col])

    return (
        np.array(x_seq_list),
        np.array(x_static_list, dtype=np.float32),
        {k: np.array(v, dtype=np.float32) for k, v in target_lists.items()},
        np.array(lengths_list),
    )
```

**Critical leakage prevention**: The loop at `player_df.iloc[start:i]` only looks at games *before* index `i`. The current game's stats are never in the sequence. This mirrors your existing `.shift(1)` logic.

**Cross-season handling**: If a player's first game of 2024 is at index `i`, and they played 16 games in 2023, the lookback naturally includes late-2023 games. This is intentional — recent form carries across seasons. If you want to reset at season boundaries, add a check and zero-pad.

## Integration with `shared/pipeline.py`

The sequence model slots into the pipeline alongside Ridge and the existing NN. The pipeline changes are:

1. After step 6 (fill NaNs), build sequences:
   ```python
   x_seq_train, x_static_train, y_train, len_train = build_sequences(
       train_df, seq_feature_cols, static_feature_cols, target_cols
   )
   ```
2. Scale `x_seq` and `x_static` independently with separate `StandardScaler` instances (fit on train only)
3. Create `SequenceDataset` and dataloaders
4. Train `SequenceMultiHeadNet` with the modified trainer
5. Ensemble becomes: `0.33 × Ridge + 0.33 × MLP + 0.33 × LSTM` (or tune weights on validation)

### Pipeline Config Additions

```python
# New keys in position config dict:
{
    # ... existing keys ...

    # Sequence model
    "seq_features": list[str],           # per-timestep features
    "seq_len": int,                      # lookback window (default 5)
    "lstm_hidden": int,                  # LSTM hidden size (default 64)
    "lstm_layers": int,                  # LSTM depth (default 2)
    "lstm_dropout": float,               # inter-layer dropout (default 0.3)
    "lstm_bidirectional": bool,          # (default True)
    "seq_backbone_layers": list[int],    # post-concat backbone (default [64])
    "seq_head_hidden": int,              # per-target head hidden (default 32)
    "seq_lr": float,                     # (default 1e-3)
    "seq_epochs": int,                   # (default 200)
    "seq_batch_size": int,               # (default 128)
    "seq_patience": int,                 # (default 20)
}
```

## Position-Specific Sequence Features

Each position uses different per-timestep features:

**QB** (F_seq ≈ 15):
```python
["fantasy_points", "passing_yards", "passing_tds", "interceptions", "attempts",
 "completions", "rushing_yards", "rushing_tds", "sacks", "passing_epa",
 "snap_pct", "is_home", "opp_fantasy_pts_allowed_to_pos",
 "carries", "fumbles_lost"]
```

**RB** (F_seq ≈ 15):
```python
["fantasy_points", "rushing_yards", "rushing_tds", "carries", "targets",
 "receptions", "receiving_yards", "receiving_tds", "rushing_epa",
 "receiving_epa", "snap_pct", "is_home", "opp_fantasy_pts_allowed_to_pos",
 "receiving_yards_after_catch", "fumbles_lost"]
```

**WR** (F_seq ≈ 14):
```python
["fantasy_points", "targets", "receptions", "receiving_yards", "receiving_tds",
 "receiving_air_yards", "receiving_yards_after_catch", "receiving_epa",
 "snap_pct", "is_home", "opp_fantasy_pts_allowed_to_pos",
 "rushing_yards", "rushing_tds", "fumbles_lost"]
```

**TE** (F_seq ≈ 13):
```python
["fantasy_points", "targets", "receptions", "receiving_yards", "receiving_tds",
 "receiving_air_yards", "receiving_yards_after_catch", "receiving_epa",
 "snap_pct", "is_home", "opp_fantasy_pts_allowed_to_pos",
 "rushing_yards", "fumbles_lost"]
```

## Recommended Starting Hyperparameters

| Parameter | RB | WR | QB | TE |
|-----------|----|----|----|----|
| `lstm_hidden` | 64 | 64 | 48 | 48 |
| `lstm_layers` | 2 | 2 | 2 | 2 |
| `lstm_dropout` | 0.3 | 0.25 | 0.4 | 0.35 |
| `bidirectional` | True | True | True | True |
| `seq_backbone_layers` | [64] | [64] | [48] | [48] |
| `seq_head_hidden` | 32 | 32 | 24 | 24 |
| `seq_lr` | 5e-4 | 8e-4 | 3e-4 | 5e-4 |
| `seq_epochs` | 200 | 150 | 250 | 200 |
| `seq_batch_size` | 128 | 256 | 64 | 128 |
| `seq_patience` | 20 | 15 | 25 | 20 |

Rationale: QB and TE have fewer samples, so use smaller hidden sizes and more regularization. RB/WR have larger datasets and can support bigger models.

## Files Summary

| File | Action | Description |
|------|--------|-------------|
| `shared/seq_neural_net.py` | **New** | `SequenceMultiHeadNet` class |
| `shared/seq_training.py` | **New** | `SequenceDataset` + modified trainer |
| `shared/seq_data.py` | **New** | `build_sequences()` function |
| `shared/pipeline.py` | **Modify** | Add LSTM training step, 3-way ensemble |
| `{POS}/{pos}_config.py` | **Modify** | Add seq-model hyperparameters |
| `shared/evaluation.py` | **Modify** | Report LSTM metrics alongside Ridge/NN |
| `benchmark.py` | **Modify** | Include LSTM in benchmark |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Small dataset for LSTM (especially QB/TE) | Use bidirectional LSTM for parameter efficiency; aggressive dropout; early stopping |
| Variable-length sequences at season starts | `pack_padded_sequence` handles this cleanly |
| Longer training time | LSTM is small (2 layers × 64 hidden); ~2-3x slower than existing NN, still under 2 minutes per position |
| Overfitting on sequence patterns | Separate sequence scaler fit on train only; dropout on LSTM layers; early stopping tracks same val loss |
| Sequence features overlap with rolling features in existing NN | This is intentional — the LSTM learns different patterns from the same underlying data. The ensemble benefits from model diversity. |

## Verification Plan

1. Unit test `build_sequences()`: confirm output shapes, verify no future data in sequences (check that sequence at index `i` only contains data from indices `< i`)
2. Train on RB first (fastest position, most data): compare LSTM-alone MAE vs existing NN MAE
3. Compare 2-way ensemble (Ridge + existing NN) vs 3-way ensemble (Ridge + NN + LSTM) on validation MAE
4. Check per-target contributions: LSTM should especially help `td_points` (the most volatile target, where trajectory matters most)
5. Inspect early-season weeks (1-3): LSTM should degrade gracefully with zero-padded short sequences, not blow up
6. Run full benchmark across all positions and update `benchmark_results.json`
