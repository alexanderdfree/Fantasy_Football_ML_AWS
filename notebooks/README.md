# notebooks/

This project does not use Jupyter notebooks. Exploratory data analysis and model diagnostics are kept as runnable Python scripts under [src/analysis/](../src/analysis/) so they stay version-controlled alongside the model code, are CI-lintable, and are reproducible without notebook-state side effects.

## Where the analysis code lives

Cross-cutting analysis scripts ([src/analysis/](../src/analysis/)):
- [analysis_dst_rare_dispersion.py](../src/analysis/analysis_dst_rare_dispersion.py) — DST rare-event dispersion analysis (sacks/safeties/blocked kicks distributional shape)
- [analysis_rb_feature_signal.py](../src/analysis/analysis_rb_feature_signal.py) — RB feature-signal audit (which engineered features actually predict each target)
- [analysis_shap_lgbm.py](../src/analysis/analysis_shap_lgbm.py) — SHAP value attribution for LightGBM models
- [analysis_weather_vegas_correlation.py](../src/analysis/analysis_weather_vegas_correlation.py) — weather/Vegas-feature correlation with fantasy outcomes

Per-position diagnostic CLIs:
- [src/QB/diagnose_qb_outliers.py](../src/QB/diagnose_qb_outliers.py) — week-level outlier diagnostics for QB predictions
- [src/RB/analyze_rb_errors.py](../src/RB/analyze_rb_errors.py) — RB error decomposition by target

Each script is invocable via `python -m src.analysis.<filename>` (or `python src/analysis/<filename>.py`); outputs land in [analysis_output/](../analysis_output/) (gitignored).

## Why no notebooks

Notebooks were considered and rejected for this project. Rationale documented in [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md). Summary: pipelined experiments + benchmarks ([src/benchmarking/benchmark.py](../src/benchmarking/benchmark.py)) cover the "what's the model doing?" question better than ad-hoc notebook cells, and the per-position rerunnable scripts cover the "why is this prediction off?" question.
