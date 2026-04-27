# models/

Trained model artifacts and model loading pointers.

## Where artifacts live

Trained model artifacts (Ridge / MultiHeadNet / Attention NN / LightGBM, one per position) are produced by training and stored in two places:

- **Runtime / local:** `src/{POSITION}/outputs/models/` (gitignored). Populated by running `python -m src.{POSITION}.run_{position}_pipeline` or the multi-position runner via `python -m src.batch.train`. Example: `src/QB/outputs/models/multihead_attn.pt`.
- **Production / S3:** `s3://<bucket>/models/{POSITION}/model.tar.gz`. The Flask serving container pulls from S3 at startup via [src/shared/model_sync.py](../src/shared/model_sync.py).

## Model class implementations

- [src/shared/models.py](../src/shared/models.py) — `RidgeMultiTarget`, `LightGBMMultiTarget`, `ElasticNetMultiTarget`, `TwoStageRidge`
- [src/shared/neural_net.py](../src/shared/neural_net.py) — `MultiHeadNet`, `AttentionPool`, `GatedTDHead`
- [src/models/](../src/models/) — `SeasonAverageBaseline`, `LastWeekBaseline`, `RidgeModel`, `ElasticNetModel`

## Model configurations

Each position has its own configuration module with hyperparameters (NN dims, Ridge alpha grids, LightGBM params, Huber deltas, loss weights):
- [src/QB/qb_config.py](../src/QB/qb_config.py)
- [src/RB/rb_config.py](../src/RB/rb_config.py)
- [src/WR/wr_config.py](../src/WR/wr_config.py)
- [src/TE/te_config.py](../src/TE/te_config.py)
- [src/K/k_config.py](../src/K/k_config.py)
- [src/DST/dst_config.py](../src/DST/dst_config.py)

## Loading entry points

- [src/shared/registry.py](../src/shared/registry.py) — `runner_module` and `model_dir` per position; the source of truth for "where do this position's artifacts live"
- [src/shared/model_sync.py](../src/shared/model_sync.py) — S3 ↔ local artifact sync used by both training (push) and serving (pull) paths
