# Preprocessing Pipeline - Development Guide

## Architecture

This is a modular, **config-driven** pipeline for cell painting image preprocessing. The core philosophy:

- **Same analysis code** for all projects
- **Different config files** for different projects
- **No code modifications** needed to adapt to new datasets

## Repository Layout

- `cp_preprocess/` — the package; all modules below live here and import each
  other with relative imports (`from . import normalization`).
- `run_pipeline.py`, `run_full_pipeline.py`, `run_analysis_plots.py` (repo root) —
  thin wrappers around `cp_preprocess.<name>.main` so a fresh clone runs without
  installing; `pip install -e .` also provides `cp-run-*` console scripts.
- `configs/config_base.yaml` — the documented template; `configs/examples/` —
  real project configs; `configs/local/` — git-ignored user configs.
- The DB URI is never committed: `database.uri` is empty and resolved from the
  `IMAGEDB_URI` env var (`config.resolve_database_uri`).

## Module Organization

### `config.py` — Configuration Schema
Defines the complete configuration structure using Python dataclasses:
- `DatabaseConfig` — DB connection (if used)
- `MetadataConfig` — Column mappings, metadata file paths, **ID column references**
- `NormalizationConfig` — MAD, blocklist, clipping
- `BatchCorrectionConfig` — ComBat settings
- `ConsensusConfig` — Level 5 aggregation method
- `MetricsConfig` — What to compute (mAP, GRIT, etc.)
- `OutputConfig` — Output directories
- `PipelineConfig` — Top-level aggregation of all sub-configs

Key functions:
- `load_config_from_yaml()` — Load YAML, merge with defaults
- `_update_dataclass()` — Recursive dict → dataclass conversion

**ID Column Support:**
`MetadataConfig` now includes optional fields to automatically extract and preserve ID columns from your metadata:
- `qc_id_col` — QC/pipeline ID column (e.g., `Metadata_pipeline_id_qc`)
- `feat_id_col` — Feature analysis ID column (e.g., `Metadata_pipeline_id_feat_texture`)
- `plate_acq_id_col` — Plate acquisition ID column (e.g., `Metadata_plate_acquisition_id`)

These are automatically detected and preserved through all pipeline levels—no manual overrides needed.

### `pipeline.py` — Main Orchestrator
`CellPaintingPipeline` class manages the complete workflow:
- Loads Level 2.5 input
- Calls each level's function in sequence
- Handles I/O and error reporting
- Computes metrics at appropriate stage (Level 4b)
- Saves outputs with metadata

**Key method:** `run(input_level2_path, cell_line)` → returns paths to all outputs

### `normalization.py` — Levels 3, 4a, 4b
Three functions:
1. `aggregate_sites_to_wells()` — Site → well aggregation (median/mean)
2. `normalize_per_plate()` — MAD normalization + blocklist + invariant filtering
3. `feature_select()` — pycytominer feature_select operations

Handles:
- Cell count filtering (per-site and per-well)
- Metadata preservation
- pycytominer integration
- Post-normalization DMSO-centering validation (`_validate_dmso_centering()`)

See **`docs/NORMALIZATION.md`** for the exact, explicit spec (normalization
method, negative-control definition, strata/plate-wise behavior, features
excluded from normalization, preprocessing before normalization) and
**`docs/PIPELINE_LEVELS.md`** for a level-by-level workflow diagram (LINCS-style).

### `batch_correction.py` — Level 4c
`apply_batch_correction()` — dispatcher selecting the method via
`config.batch_correction.method`:
- `"spherize"` (default) — `apply_spherize()`: pycytominer ZCA whitening fit on
  the DMSO (negcon) reference — closest to the standard profiling-recipe/LINCS
  normalization philosophy (no cross-plate statistical model)
- `"combat"` — `apply_combat()`: scanpy ComBat per-plate correction
- `"harmony"` — `apply_harmony()`: harmonypy integration, plate = batch variable
  (runs in feature space; orientation-robust `Z_corr` handling; see `docs/HARMONY_NOTES.md`)

All methods share the same signature `(df, metadata, config) → (df_corrected,
metadata)`, operate in feature space, and **fall back to uncorrected data** (with
a warning) on failure so the pipeline never crashes at Level 4c.

Well **row/column removal is opt-in and off by default** (never run
automatically, regardless of method) — two post-correction steps the
orchestrator applies to L4c after *any* method, only when explicitly
configured:
- `regress_out_covariates()` — least-squares removal of row/column, per
  `config.batch_correction.batch_effect_fix` (`barcode_row` / `barcode_column` /
  `barcode_row_column`; `barcode` = no-op, the default)
- `regress_out_column_effect()` — control-based within-plate column de-trend, per
  `config.batch_correction.regress_out_column` (`False` by default)

### `consensus.py` — Level 5
Three consensus methods available via config:
- `_consensus_modz()` — Robust median (modified Z-score median)
- `_consensus_median()` — Simple median
- `_consensus_mean()` — Simple mean

Groups by compound+concentration (detected automatically from metadata).

### `metrics.py` — Metrics Computation
Computes on Level 4b (pre-batch) data via:
- **mAP** (copairs) — mean Average Precision for phenotypic ranking
- **GRIT** (cytominer_eval) — Gene Rank Index Threshold
- **Reproducibility** (cytominer_eval) — Replicate correlation

Each has:
- Primary computation function (calls external library)
- Fallback if library unavailable or fails (returns NaN)

Key: Returns dict of {metric_name: [values_per_sample]}

### `viability.py` — Cell Count-Based Toxicity (SITE level)
`compute_viability()` — computed at the imaging SITE level (before site→well
aggregation) and rolled up to wells:
- Finds the measured per-site cell-count column automatically
- Per plate: site count ÷ **cleaned** DMSO-site median × 100 (low-count DMSO
  outliers dropped from the reference via a low-side robust Z-score)
- Rolls up to wells by median-of-ratios; carries `n_sites` for low-confidence flags
- `_validate_viability()` self-checks the result (DMSO wells ≈100%, coverage,
  no negative %) and folds the diagnostics into the run summary
- Flags wells below threshold — never drops them (a low count IS the signal)

### `profile_metrics.py` — Standalone profile metrics (independent)
Not part of the pipeline package (no pipeline imports; numpy/pandas/scipy/
matplotlib only). Importable + CLI. Profile matching (Biosimilarity/Pearson) on
per-compound consensus, activity detection (Euclidean vs DMSO centroid) on wells,
and a z-score parallel-coordinates profile plot. Safe to edit without touching
pipeline behaviour.

### `io_utils.py` — Parquet I/O
`save_level_parquet()` — Saves a level as parquet:
- Reorders columns: metadata → metrics → features
- Ensures readability
- Returns output path

### `run_pipeline.py` / `run_full_pipeline.py` — CLI Entry Points
Command-line interface:
- Load YAML config
- Auto-discover Level 2.5 input files
- Run pipeline for all or specific cell lines
- Report summary

## Design Decisions

### Why Config Files?
**Config-driven over code-driven:**
- Metadata columns vary per project (Metadata_Barcode vs Plate_ID)
- Thresholds differ (min_cells_per_site, viability_threshold)
- Blocklists are dataset-specific
- No Python editing needed for new projects

### Why Levels 3→5?
**Follows Ewald/Nyffeler methodology:**
- Level 3: Site aggregation (preserves raw scale)
- Level 4a: Normalization (standardize intensity)
- Level 4b: Feature selection (remove noise)
- Level 4c: Batch correction (remove technical variation)
- Level 5: Consensus (one profile per compound/dose)

### Why Compute Metrics on Level 4b?
**Metrics reflect the raw replicate structure:**
- mAP and GRIT are computed on **Level 4b** (feature-selected, PRE batch
  correction), not on Level 4c — so they measure the actual per-well replicate
  signal rather than the residuals of a batch-correction model.
- They are attached as `Metadata_` annotations to **both** Level 5 variants
  (`level_5_batch_corrected` and `level_5_no_batch_correction`) by compound×dose.
- Viability is computed even earlier, at SITE level, before site→well aggregation.

### Why two Level 5 files?
Batch correction changes the consensus, so both are saved side by side to make
its effect auditable:
- `level_5_batch_corrected.parquet` — consensus of Level 4c
- `level_5_no_batch_correction.parquet` — consensus of Level 4b
When `batch_correction.run: false` the two are identical by construction.

### Why Parquet Output?
- Efficient compression
- Column-oriented (fast feature access)
- Metadata preserves column types
- Native support for large datasets

### Fallback Strategies
Each metric has a fallback (returns NaN) if:
- External library unavailable
- Dataset missing required columns
- Computation fails (e.g., ComBat with <2 batches)

**Rationale:** Pipeline continues; user sees NaN and investigates, rather than crashing.

## Adding Features

### New Metric
1. Add to `MetricsConfig` in `config.py`
2. Implement in `metrics.py` with fallback
3. Call from `pipeline.py` → add to metadata

Example:
```python
# In metrics.py
def _compute_my_metric(features_df, metadata_df, config):
    try:
        # compute
        return values
    except:
        return np.full(len(features_df), np.nan)

# In pipeline.py's run()
metrics_data["MyMetric"] = metrics.compute_my_metric(...)
```

### New Normalization Method
1. Implement in `normalization.py`
2. Call from `pipeline.py` → add to level 4a or 4b

### New Config Section
1. Add dataclass in `config.py`
2. Add field to `PipelineConfig`
3. Update `configs/config_base.yaml`
4. Use in relevant module

## Testing

For local testing without large datasets:
```python
import pandas as pd
from cp_preprocess.config import load_config_from_yaml
from cp_preprocess.pipeline import CellPaintingPipeline

config = load_config_from_yaml("configs/config_base.yaml")
pipeline = CellPaintingPipeline(config)

# Create mock Level 2.5
df_mock = pd.DataFrame({
    "Metadata_Barcode": ["PB001"] * 10,
    "Metadata_Well": ["A01"] * 10,
    "Metadata_Site": range(10),
    "Metadata_pert_type": ["negcon"] * 10,
    "Metadata_Compound_Name": ["DMSO"] * 10,
    "Metadata_conc_uM": [0] * 10,
    "feature_1": range(10),
    "feature_2": range(10, 20),
})
df_mock.to_csv("mock_level2_5.csv", index=False)

# Run
output_paths = pipeline.run("mock_level2_5.csv", "test_cellline")
```

Before merging a change, run an old-vs-new equivalence check on a few real
plates (`run_pipeline.py --level2-5 <subset> --to-step metrics` with both
versions, then `pd.testing.assert_frame_equal` per output parquet). Defaults
must not change existing results.


## Common Customizations

### Use Mean Instead of Median
```yaml
aggregation:
  operation: "mean"
```

### Skip Batch Correction
```yaml
batch_correction:
  run: false
```

### Blocklist
`normalization.blocklist_patterns` is not read by the code. Blocklisting is
done only by pycytominer's canonical blocklist via the `"blocklist"` entry in
`feature_selection.operations` (Level 4b) — keep it in the list.

### Remove Row/Column Positional Effect
```yaml
batch_correction:
  batch_effect_fix: "barcode_row"  # barcode | barcode_row | barcode_column | barcode_row_column
```

(Note: `last_chars` not yet implemented — add in pipeline if needed.)

### Read QC/Feature/Plate IDs from Metadata
Instead of hardcoding overrides, reference columns in your metadata file:
```yaml
metadata:
  file_path: "./metadata.csv"
  barcode_col: "Metadata_Barcode"
  well_col: "Metadata_Well"
  # ... other columns ...
  
  # Read these IDs directly from the metadata CSV
  qc_id_col: "Metadata_pipeline_id_qc"
  feat_id_col: "Metadata_pipeline_id_feat_texture"
  plate_acq_id_col: "Metadata_plate_acquisition_id"
```

The pipeline automatically extracts and preserves these IDs through all levels, eliminating the need for manual `plate_acq_id_overrides` entries.

## Known Limitations

1. **Metrics Fallbacks:** If copairs/cytominer_eval unavailable, uses approximate fallback (may not match published methodology exactly)

2. **ComBat Requirements:** Needs ≥3 DMSO wells per plate; falls back to uncorrected if insufficient

3. **Metadata Column Detection:** Assumes standard naming (Metadata_*, *Compound*, *conc*). Non-standard names must be specified in config

4. **Single Aggregation Method:** Uses one method (median/mean) for all aggregations. Could extend to per-step configs if needed

## Future Enhancements

- [x] UMAP/PCA visualization outputs (`plotting_analysis.py`, incl. L4b GRIT>1 &
      phenotypically-active maps in `05_metrics`)
- [x] QC plot generation (`run_analysis_plots.py` staged plots)
- [ ] Parallel processing by plate
- [ ] PostgreSQL data loading integration (database section in config)
- [ ] Pluggable metrics (register new metrics without editing core)
- [ ] Checkpointing (resume from Level N)
- [ ] Logging to file + structured logging

## References

- Ewald et al. (2025): *Cell painting as a window into disease mechanism*
- Nyffeler et al. (2023): *Functional compound profiling with high-resolution…*
- Nyffeler et al. (2021): *A 384-well image-based platform for phenotypic drug screening*
