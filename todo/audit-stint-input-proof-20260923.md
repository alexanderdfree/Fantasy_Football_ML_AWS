# Stint input proof extension

This experiment-only branch extends `87213b853c4d724f6574dccddbb62dfcb6601cd1`
without rewriting that source or its image. Production model, data, training and
preprocessing implementations remain unchanged. The existing two-function
stint candidate remains byte-identical.

WR has 110 ordinary features and 47 attention static features; TE has 108 and
45 respectively. Both use 27 per-game history signals. The changed
`opportunity_index_L3` and `redzone_target_share_L3` columns appear only in the
ordinary feature lists. Their underlying per-game history values are unchanged.
Ridge, LightGBM and plain NN are therefore the proposed affected families.
Attention remains a control only if its actual inputs, fitted scaler,
checkpoint and forecasts remain identical in the paired run.

Each WR/TE cell now records `input_proof` with schema `audit-input-proof/v1`:

- Ordered feature/target lists; train/validation/test row identities, every
  prepared frame column, every raw X column and every target array.
- Fitted preprocessing fields and per-column imputation values.
- Per-column neural scaler statistics and transformed matrices.
- Actual trainer inputs/targets from stored resident tensors, including
  attention history/masks, plus the actual test prediction arguments.

The observer reads stored tensors without iterating loaders, sampling, fitting,
or changing return values. Context-local capture prevents saved-inference replay
from overwriting the original training/test-input evidence. Scope checks must
allow drift only in the two intended raw columns and their corresponding
imputation/scaling transforms. All other prepared columns, targets, row keys,
and every attention input/scaler must remain identical. Raw frame column order
may differ because the candidate assigns the two columns in a different order;
ordered model feature lists and actual input arrays must still match.

Validation: 37 no-fit tests pass, including current WR/TE pipeline boundaries
with fitting/scaling stubbed, output and Python/NumPy/Torch RNG equality,
resident-tensor/DataLoader capture and saved-replay isolation. Ruff and diff
checks pass. A real FP32 CUDA Batch smoke remains required before expansion.
The earlier observer source bridge applies to the frozen 87213b85 source;
this extension requires its own explicit image/source pin for stint runs.
