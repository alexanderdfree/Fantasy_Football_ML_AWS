# Training-data audit fix validation — 2026-09-10

**The bundled corrections regress offline MAE for QB, WR, and D/ST on the common test cohort. They are not metric-neutral.** RB improves slightly; TE and K are mixed. These runs validate the requested data/contract corrections, not a claim about live performance. They do not identify the causal contribution of individual fixes: multiple changes are bundled, and the baseline availability feature used retrospective game participants. Retuning is outside this PR. The overall D/ST points-allowed scoring contract remains unchanged.

## Regime and comparison basis

- Baseline code: `6406cf213cc0a5fbcd1e56e449e3d7f82804dc01`, clean detached checkout, frozen audit raw/split inputs copied before execution. Fixed code: `5044ea6687a561871799052c97c67c85b4f8ea5e`.
- All six native production pipelines and their full `POSITION_CONFIG` ran with seeds **42, 123, 7**. K and D/ST retained their native input loading. Both arms used `/Users/alex/miniforge3/bin/python`, **CPU FP32, eager execution**, `FF_DEVICE=cpu`, `FF_AMP_DTYPE=fp32`, `FF_CUDA_GRAPH=0`, `FF_COMPILE=0`; BLAS, OpenMP, MKL, NumExpr, and Accelerate thread environment limits were all 1. The existing A/B harness ran two cells concurrently with its core pool and isolated output directories; the successful one-cell smoke was reused. Feature caching was disabled in both arms.
- **18/18 cells passed per arm.** Predictions for every raw target/model were saved, finite, and uniquely keyed. Test-row identities and actual raw targets matched across the three seeds within each arm. Source and input fingerprints remained unchanged.
- Both forecasts and fixed actuals were scored with production `predictions_to_fantasy_points` / `score_actual_components`. All comparisons use identical regular-season `(player_id, season, week)` keys and the same projected scoring components. Baseline forecasts were rescored against **fixed actuals**; changed labels alone cannot appear as a model improvement. No full-fantasy fallback, unavailable actual components, unavailable models, or raw-vs-stored forecast mismatches occurred.
- The tables below recompute matched-component actuals; canonical history files also retain the native pipeline headline metrics, so those headline numbers need not equal this matched-component report.
- Values are means ± sample standard deviation across three seeds. The paired delta is **fixed minus baseline**; positive MAE/RMSE deltas are worse. The standard deviation is across seeds, not a confidence interval.

## Population

**15 additional offensive test player-weeks are reported separately and excluded from paired deltas. No baseline test rows were removed.**

| Position | Baseline / common rows | Fixed rows | Added rows | Removed rows |
|---|---:|---:|---:|---:|
| QB | 686 | 687 | 1 | 0 |
| RB | 1757 | 1759 | 2 | 0 |
| WR | 2761 | 2768 | 7 | 0 |
| TE | 1655 | 1660 | 5 | 0 |
| K | 543 | 543 | 0 | 0 |
| DST | 544 | 544 | 0 | 0 |

## All models on the common cohort

| Position | Model | Baseline MAE on fixed actuals | Fixed MAE | Paired Δ MAE | Paired Δ RMSE |
|---|---|---:|---:|---:|---:|
| QB | Ridge | 5.7445 ± 0.0000 | 6.0299 ± 0.0000 | +0.2854 ± 0.0000 | +0.1940 ± 0.0000 |
| QB | NN | 5.6816 ± 0.0577 | 5.8071 ± 0.0369 | +0.1255 ± 0.0601 | +0.0924 ± 0.0541 |
| QB | Attention NN | 5.6534 ± 0.0232 | 5.8301 ± 0.0199 | +0.1768 ± 0.0094 | +0.1378 ± 0.0345 |
| QB | LightGBM | 5.6698 ± 0.0179 | 5.7286 ± 0.0162 | +0.0589 ± 0.0322 | +0.0614 ± 0.0265 |
| RB | Ridge | 3.9832 ± 0.0000 | 3.9797 ± 0.0000 | -0.0035 ± 0.0000 | -0.0097 ± 0.0000 |
| RB | NN | 3.9224 ± 0.0461 | 3.8742 ± 0.0142 | -0.0483 ± 0.0599 | -0.0749 ± 0.0597 |
| RB | Attention NN | 3.8197 ± 0.0386 | 3.8175 ± 0.0660 | -0.0022 ± 0.0408 | +0.0366 ± 0.0591 |
| RB | LightGBM | 3.9771 ± 0.0056 | 3.9702 ± 0.0142 | -0.0069 ± 0.0120 | -0.0052 ± 0.0198 |
| WR | Ridge | 3.8657 ± 0.0000 | 4.1196 ± 0.0000 | +0.2539 ± 0.0000 | +0.0550 ± 0.0000 |
| WR | NN | 3.8250 ± 0.0249 | 3.9443 ± 0.0251 | +0.1193 ± 0.0173 | -0.0008 ± 0.0151 |
| WR | Attention NN | 3.7781 ± 0.0403 | 3.8957 ± 0.0552 | +0.1177 ± 0.0785 | -0.0372 ± 0.0185 |
| WR | LightGBM | 3.9102 ± 0.0076 | 4.0233 ± 0.0133 | +0.1131 ± 0.0146 | +0.0270 ± 0.0073 |
| TE | Ridge | 2.8606 ± 0.0000 | 2.8603 ± 0.0000 | -0.0003 ± 0.0000 | +0.0012 ± 0.0000 |
| TE | NN | 2.8442 ± 0.0154 | 2.8587 ± 0.0250 | +0.0144 ± 0.0241 | +0.0215 ± 0.0508 |
| TE | Attention NN | 2.9113 ± 0.1194 | 2.8457 ± 0.0110 | -0.0656 ± 0.1192 | -0.2599 ± 0.4230 |
| TE | LightGBM | 2.9615 ± 0.0179 | 2.9611 ± 0.0019 | -0.0004 ± 0.0160 | +0.0008 ± 0.0112 |
| K | Ridge | 4.0648 ± 0.0000 | 4.0464 ± 0.0000 | -0.0184 ± 0.0000 | -0.0188 ± 0.0000 |
| K | NN | 4.1152 ± 0.0349 | 4.1263 ± 0.0395 | +0.0111 ± 0.0530 | +0.0275 ± 0.0777 |
| K | Attention NN | 4.1791 ± 0.0376 | 4.1490 ± 0.0221 | -0.0301 ± 0.0198 | -0.0512 ± 0.0468 |
| K | LightGBM | 4.0625 ± 0.0039 | 4.0701 ± 0.0113 | +0.0076 ± 0.0140 | +0.0137 ± 0.0257 |
| DST | Ridge | 5.4216 ± 0.0000 | 5.4632 ± 0.0000 | +0.0416 ± 0.0000 | +0.0051 ± 0.0000 |
| DST | NN | 5.4072 ± 0.0506 | 5.4528 ± 0.0463 | +0.0456 ± 0.0126 | +0.0215 ± 0.0294 |
| DST | Attention NN | 5.3425 ± 0.0225 | 5.3845 ± 0.0184 | +0.0419 ± 0.0302 | +0.0202 ± 0.0320 |
| DST | LightGBM | 5.3749 ± 0.0191 | 5.4250 ± 0.0133 | +0.0502 ± 0.0062 | +0.0207 ± 0.0058 |

## Affected rows and archived pregame reference

The directly affected test subsets are 39 QB rows whose availability flag changed, 2,276 WR rows whose depth rank changed, and 25 D/ST rows whose raw targets changed (24 changed fantasy totals). These subsets describe where data changed; they do not isolate the cause of a model-metric change.

| Subset | n | Model | Baseline MAE | Fixed MAE | Paired Δ MAE |
|---|---:|---|---:|---:|---:|
| QB availability changed | 39 | Ridge | 5.2798 ± 0.0000 | 6.1731 ± 0.0000 | +0.8934 ± 0.0000 |
| QB availability changed | 39 | NN | 5.3208 ± 0.0670 | 5.7481 ± 0.1410 | +0.4273 ± 0.0743 |
| QB availability changed | 39 | Attention NN | 5.7493 ± 0.0590 | 5.7948 ± 0.1573 | +0.0455 ± 0.1111 |
| QB availability changed | 39 | LightGBM | 5.5123 ± 0.0683 | 5.4576 ± 0.1055 | -0.0548 ± 0.1137 |
| WR depth rank changed | 2276 | Ridge | 3.3496 ± 0.0000 | 3.6571 ± 0.0000 | +0.3075 ± 0.0000 |
| WR depth rank changed | 2276 | NN | 3.3321 ± 0.0294 | 3.4721 ± 0.0203 | +0.1400 ± 0.0170 |
| WR depth rank changed | 2276 | Attention NN | 3.2517 ± 0.0490 | 3.4001 ± 0.0557 | +0.1484 ± 0.0825 |
| WR depth rank changed | 2276 | LightGBM | 3.4340 ± 0.0082 | 3.5713 ± 0.0134 | +0.1373 ± 0.0205 |
| D/ST raw targets changed | 25 | Ridge | 9.7759 ± 0.0000 | 9.4888 ± 0.0000 | -0.2871 ± 0.0000 |
| D/ST raw targets changed | 25 | NN | 9.1710 ± 0.1010 | 8.8201 ± 0.2254 | -0.3510 ± 0.1373 |
| D/ST raw targets changed | 25 | Attention NN | 9.5358 ± 0.1650 | 9.4223 ± 0.1317 | -0.1135 ± 0.1902 |
| D/ST raw targets changed | 25 | LightGBM | 9.2743 ± 0.0348 | 9.0121 ± 0.0134 | -0.2623 ± 0.0350 |

D/ST MAE improves on the corrected-target subset while overall D/ST MAE worsens slightly. WR attention RMSE improves despite worse MAE. Neither result establishes a live-performance effect.

The archived **pregame** top-24 reference is available for every position. Both arms use the same selected player-weeks and fixed actual components; no model selects its own comparison pool. Cohort identity hashes match across all three seeds.

| Position | Common reference rows | Ridge Δ MAE | NN Δ MAE | Attention Δ MAE | LightGBM Δ MAE |
|---|---:|---:|---:|---:|---:|
| QB | 432 | +0.0782 ± 0.0000 | +0.0400 ± 0.0285 | +0.0393 ± 0.0187 | +0.0069 ± 0.0124 |
| RB | 432 | -0.0133 ± 0.0000 | -0.0878 ± 0.0884 | -0.0115 ± 0.0805 | +0.0105 ± 0.0274 |
| WR | 432 | +0.0131 ± 0.0000 | +0.0263 ± 0.0488 | +0.0138 ± 0.0259 | +0.0371 ± 0.0157 |
| TE | 432 | +0.0042 ± 0.0000 | +0.0824 ± 0.0789 | -0.3164 ± 0.5157 | +0.0011 ± 0.0310 |
| K | 431 | -0.0180 ± 0.0000 | +0.0048 ± 0.0467 | -0.0358 ± 0.0285 | +0.0031 ± 0.0143 |
| DST | 432 | +0.0350 ± 0.0000 | +0.0437 ± 0.0195 | +0.0509 ± 0.0192 | +0.0574 ± 0.0057 |

## Coverage and provenance

- The sealed fixed input contains **125 producer files and 28 input files (57,373,581 bytes)**. Every positive offensive snap ID resolved. Offline preparation found no missing/non-finite selected features across all six positions.
- The compact D/ST source matches an independently fetched full-PBP reconstruction for **7,326 regular-season team-weeks, 2012–2025**, with zero mismatches: 913 defensive TDs, 334 special-teams TDs, and 183 blocked punts, including the 2012 context season. On the audited 6,814 modeled games, the event correction restores 292 omitted TDs and 162 punt blocks, removes one spurious TD, and changes 441 totals; points-allowed/yardage contracts and scoring weights were preserved.
- Baseline input SHA256: `52cfd8d6d67de51c9b68bd070dcd85b3b694dcaf2c3f17a4113dd1c9c4acca95`. Fixed input SHA256: `2a49e6c9acbbc1cba5bc945c40fc9a207c72760be9d6cd054baa99044e9a3a44`. Both fingerprint the sorted relative-path/content-hash manifest; the fixed fingerprint was checked before and after the complete run.

Fixed code fingerprints captured before training and matched after training:

| Position | Fingerprint |
|---|---|
| QB | `69a5381c6a5096ce7367edfd7c6a5bd1e5cad7e95ef53ca4cba7bc6891e9af3a` |
| RB | `ca68e1fd47354acd8578ffa6847b8f9383ed617d9c0a197716ea2e39b2d4cf64` |
| WR | `2d40dbae10f3505648451d49024e500933b9495385c2525de7426e936a27792a` |
| TE | `83f270ee170d18069e5a4a25de168cb85bcf60ca326fffc07b2a1dd7f6c2fbba` |
| K | `4cdca0c0b88fd8a96d2c17ce812273d836f1955703bde7a5bfe64172711d4a9b` |
| DST | `b2be0cfb07d32c942159d2a2b55167396cfc86931900a0982606469ac62b724a` |

Canonical local history entries were written with the existing `summarize_pipeline_result` / `append_to_history` helpers and pre-captured, rechecked code fingerprints. No S3 sync, push, or cloud training was performed:

- [Seed 42](../benchmark_history/2026-09-10T13-43-22_5044ea66_training_data_fix_seed42.json)
- [Seed 123](../benchmark_history/2026-09-10T13-43-22_5044ea66_training_data_fix_seed123.json)
- [Seed 7](../benchmark_history/2026-09-10T13-43-22_5044ea66_training_data_fix_seed7.json)

Local row-level forecasts, complete paired metrics/cohort hashes, verifier outputs, and reproduction scripts remain under `/tmp/training-data-fix-validation-1e2d/`: `baseline_outputs/`, `fixed_outputs/`, `paired-comparison.json`, and `specs/`. Datasets and prediction frames are not committed.

## Post-review boundary corrections

The native PR review reproduced three additional boundary failures: live ESPN receiver ranks did not match the rebuilt archive, a transient optional-source failure could be sealed without a replayable cache, and serving prediction caches omitted the new K/DST dependencies. The fixes cover actual ESPN athlete-slot payloads, cache-only replay before building/sealing, and full-content fingerprints for the new scoring/depth inputs and release metadata.

The review corrections preserve the inputs behind the three-seed comparison above. With historical source fetching forbidden, all six production preparation paths produced identical hashes for their train/validation/test feature arrays, targets, complete prepared frames, feature lists, and configurations before and after these corrections. The comparison used `4c672de4`, whose numerical source files are identical to the trained `5044ea66`, and the same frozen input bytes. This establishes input equivalence; the original benchmark fingerprints are retained as historical evidence rather than relabeled as new runs.

The real complete-cache replay and seal retained all 25 raw files unchanged and the same 261,714-row loader frame, while both original missing-opportunity reproductions now reject publication. Empty 2012 snap coverage remains explicit. The full local unit suite passed **3,992 tests, with 2 skipped** after these fixes.
